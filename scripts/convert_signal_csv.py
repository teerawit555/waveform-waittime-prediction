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
    python convert_signal_csv.py --in SignalSample.csv --out signal_long.csv
    python convert_signal_csv.py --in SignalSample.csv --out signal_long.csv --dt-ms 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
            "ไม่พบ value columns (คาดว่าชื่อต้องลงท้ายด้วย ':' เช่น 'Signal1:')\n"
            f"Columns ที่มี: {df.columns.tolist()}"
        )

    print(f"พบ {len(value_cols)} waves: {value_cols}")

    # แปลง wide → long ด้วย pd.melt (เร็วกว่า loop)
    # สร้าง mapping: wave_id → numeric id
    id_map: dict[str, int] = {}
    for vc in value_cols:
        # 'Signal1:' → ดึงตัวเลขออกมา
        name = vc.rstrip(":")          # 'Signal1'
        numeric_part = "".join(filter(str.isdigit, name))
        id_map[vc] = int(numeric_part) if numeric_part else (list(id_map.values())[-1] + 1 if id_map else 1)

    rows_list = []
    for vc in value_cols:
        wave_id = id_map[vc]
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
        description="แปลง SignalSample wide-format CSV → long-format สำหรับ extract_features.py"
    )
    ap.add_argument("--in",  dest="in_path",  required=True, help="path ของ input CSV (wide format)")
    ap.add_argument("--out", dest="out_path", required=True, help="path ของ output CSV (long format)")
    ap.add_argument(
        "--dt-ms",
        type=float,
        default=0.01,
        help="ระยะห่างระหว่าง sample ในหน่วย ms (default: 0.01 → 1000 samples = 10ms)",
    )
    ap.add_argument(
        "--sample-col",
        default="Signal",
        help="ชื่อ column ที่เป็น sample index (default: 'Signal')",
    )
    args = ap.parse_args()

    in_path  = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {in_path}")
    df = pd.read_csv(in_path)
    print(f"Input shape: {df.shape}  columns: {df.columns.tolist()}")

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