"""
convert_signal_csv.py
---------------------
แปลง wide-format CSV ของ SignalSample ให้เป็น long-format
ที่ extract_features.py (mode=pred) รับได้

รูปแบบ input (wide):
    Signal | Signal1: | Signal.1 | Signal2: | Signal.2 | ...
    1      | 0.4305   | 1        | 0.6727   | 1        | ...
    2      | 0.4331   | 2        | 0.6736   | 2        | ...
    ...

รูปแบบ output (long) ที่ code ต้องการ:
    wave_id | sample | time_ms | value
    1       | 1      | 0.01    | 0.4305
    1       | 2      | 0.02    | 0.4331
    ...
    2       | 1      | 0.01    | 0.6727
    ...

Usage:
    python scripts/data/convert_signal_csv.py --in SignalSample.csv --out signal_long.csv
    python scripts/data/convert_signal_csv.py --in SignalSample.xlsx --out signal_long.csv
    python scripts/data/convert_signal_csv.py --in SignalSample.csv --out signal_long.csv --dt-ms 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LONG_COLUMNS = ["wave_id", "sample", "time_ms", "value"]


def read_signal_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # Long format check
            if {"wave_id", "sample", "value"}.issubset(df.columns):
                print(f"[Info] Found matching sheet: '{sheet_name}' (Long format)")
                return df
            # Wide format check
            if "Signal" in df.columns and any(str(c).endswith(":") for c in df.columns):
                print(f"[Info] Found matching sheet: '{sheet_name}' (Wide format)")
                return df
        print(f"[Warning] No sheet matched the expected schema. Defaulting to first sheet: '{xls.sheet_names[0]}'")
        return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    raise ValueError(f"Unsupported input file type: {suffix}. Use .csv or .xlsx.")


def normalize_long_table(df: pd.DataFrame, dt_ms: float = 0.01) -> pd.DataFrame | None:
    columns = set(df.columns)
    required_without_time = {"wave_id", "sample", "value"}
    if not required_without_time.issubset(columns):
        return None

    long_df = df.copy()
    if "time_ms" not in long_df.columns:
        long_df["time_ms"] = pd.to_numeric(long_df["sample"], errors="raise") * dt_ms

    long_df = long_df[LONG_COLUMNS].copy()
    long_df["wave_id"] = long_df["wave_id"]
    long_df["sample"] = pd.to_numeric(long_df["sample"], errors="raise").astype(int)
    long_df["time_ms"] = pd.to_numeric(long_df["time_ms"], errors="raise")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="raise")
    return long_df.sort_values(["wave_id", "sample"]).reset_index(drop=True)


def convert_wide_to_long(
    df: pd.DataFrame,
    sample_col: str = "Signal",
    dt_ms: float = 0.01,
) -> pd.DataFrame:
    """
    แปลง wide-format DataFrame เป็น long-format

    Parameters
    ----------
    df        : wide-format DataFrame จาก read_csv
    sample_col: ชื่อ column ที่เป็น sample index (default: 'Signal')
    dt_ms     : ระยะห่างระหว่าง sample ในหน่วย ms (default: 0.01)

    Returns
    -------
    long_df : DataFrame ที่มี columns [wave_id, sample, time_ms, value]
    """

    # ดึง sample index จาก column กลาง (1, 2, 3, ..., 1000)
    if sample_col not in df.columns:
        raise KeyError(
            f"Column '{sample_col}' not found. Available: {df.columns.tolist()}"
        )
    samples = df[sample_col].astype(int).to_numpy()

    # หา value columns: pattern คือชื่อที่ลงท้ายด้วย ':'
    # เช่น 'Signal1:', 'Signal2:', ..., 'Signal11:'
    value_cols = [c for c in df.columns if c.endswith(":")]

    if not value_cols:
        raise ValueError(
            "No value columns found. Expected columns ending with ':' such as 'Signal1:'.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Found {len(value_cols)} waves: {value_cols}")

    rows_list = []
    for vc in value_cols:
        wave_id = vc.rstrip(":")
        values = df[vc].to_numpy(float)
        chunk = pd.DataFrame(
            {
                "wave_id": wave_id,
                "sample":  samples,
                "time_ms": np.round(samples * dt_ms, 6),
                "value":   values,
            }
        )
        rows_list.append(chunk)

    long_df = pd.concat(rows_list, ignore_index=True)
    long_df = long_df.sort_values(["wave_id", "sample"]).reset_index(drop=True)

    return long_df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert SignalSample wide-format CSV/XLSX to long-format CSV for extract_features.py"
    )
    ap.add_argument("--in",  dest="in_path",  required=True, help="input CSV/XLSX path")
    ap.add_argument("--out", dest="out_path", required=True, help="output long-format CSV path")
    ap.add_argument(
        "--dt-ms",
        type=float,
        default=0.01,
        help="sample interval in ms (default: 0.01)",
    )
    ap.add_argument(
        "--sample-col",
        default="Signal",
        help="sample index column name (default: 'Signal')",
    )
    args = ap.parse_args()

    in_path  = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {in_path}")
    df = read_signal_table(in_path)
    print(f"Input shape: {df.shape}  columns: {df.columns.tolist()}")

    long_df = normalize_long_table(df, dt_ms=args.dt_ms)
    if long_df is None:
        long_df = convert_wide_to_long(df, sample_col=args.sample_col, dt_ms=args.dt_ms)

    long_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Output shape: {long_df.shape}")
    print(f"Waves: {sorted(long_df['wave_id'].unique())}")
    print(f"Samples per wave: {long_df.groupby('wave_id')['sample'].count().to_dict()}")
    print(f"\nPreview (first 5 rows):")
    print(long_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
