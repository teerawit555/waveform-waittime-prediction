from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def signal_wide_to_long(df: pd.DataFrame, sample_col: str = "Signal", dt_ms: float = 0.01) -> pd.DataFrame | None:
    if sample_col not in df.columns:
        return None

    value_cols = [col for col in df.columns if str(col).endswith(":")]
    if not value_cols:
        return None

    samples = pd.to_numeric(df[sample_col], errors="raise").astype(int).to_numpy()
    rows = []
    for col in value_cols:
        wave_id = str(col).rstrip(":")
        rows.append(pd.DataFrame({
            "wave_id": wave_id,
            "sample": samples,
            "time_ms": np.round(samples * dt_ms, 6),
            "value": pd.to_numeric(df[col], errors="raise").to_numpy(float),
        }))

    return pd.concat(rows, ignore_index=True).sort_values(["wave_id", "sample"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser("plot_pred_on_waveforms")
    ap.add_argument("--raw",     required=True, help="raw waveform csv")
    ap.add_argument("--pred",    required=True, help="prediction csv with wave_id + pred_wait_time_ms")
    ap.add_argument("--outdir",  required=True, help="output folder for plots")
    ap.add_argument("--topk",    type=int, default=30, help="number of waveforms to plot")
    ap.add_argument("--mode",    choices=["first", "low", "high", "random"], default="first")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--wave-id", default=None, dest="wave_id", help="plot only this single wave_id")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw  = pd.read_csv(args.raw)
    pred = pd.read_csv(args.pred)

    required_raw = ["wave_id", "sample", "time_ms", "value"]
    missing_raw  = [c for c in required_raw if c not in raw.columns]
    if missing_raw:
        long_raw = signal_wide_to_long(raw)
        if long_raw is None:
            raise KeyError(f"raw file missing columns: {missing_raw}")
        raw = long_raw

    required_pred = ["wave_id", "pred_wait_time_ms"]
    missing_pred  = [c for c in required_pred if c not in pred.columns]
    if missing_pred:
        raise KeyError(f"pred file missing columns: {missing_pred}")

    raw  = raw.sort_values(["wave_id", "sample"]).copy()
    true_col = next((col for col in ["wait_time_ms", "true", "true_wait_time", "label"] if col in pred.columns), None)
    pred_cols = ["wave_id", "pred_wait_time_ms"] + ([true_col] if true_col else [])
    pred = pred[pred_cols].drop_duplicates("wave_id").copy()

    # ── เลือก wave ที่จะ plot ──────────────────────────────────
    if args.wave_id:
        # on-demand: plot แค่ตัวเดียวที่ขอมา
        chosen = [args.wave_id]
    else:
        # batch: เลือกตาม mode/topk เหมือนเดิม
        merged_ids = pred["wave_id"].tolist()

        if args.mode == "first":
            chosen = merged_ids[: args.topk]
        elif args.mode == "low":
            chosen = (
                pred.sort_values("pred_wait_time_ms", ascending=True)
                .head(args.topk)["wave_id"]
                .tolist()
            )
        elif args.mode == "high":
            chosen = (
                pred.sort_values("pred_wait_time_ms", ascending=False)
                .head(args.topk)["wave_id"]
                .tolist()
            )
        else:  # random
            chosen = (
                pred.sample(n=min(args.topk, len(pred)), random_state=args.seed)
                ["wave_id"]
                .tolist()
            )

    pred_map = dict(zip(pred["wave_id"].astype(str), pred["pred_wait_time_ms"]))
    true_map = {}
    if true_col:
        true_map = dict(zip(pred["wave_id"].astype(str), pred[true_col]))
    elif "wait_time_ms" in raw.columns:
        true_map = raw.groupby(raw["wave_id"].astype(str))["wait_time_ms"].first().to_dict()

    count = 0
    for wave_id in chosen:
        wave_id_str = str(wave_id)
        g = raw[raw["wave_id"].astype(str) == wave_id_str].copy()
        if len(g) == 0:
            print(f"  [skip] {wave_id_str} — not found in raw csv")
            continue

        if wave_id_str not in pred_map:
            print(f"  [skip] {wave_id_str} — not found in pred csv")
            continue

        t       = g["time_ms"].to_numpy(dtype=float)
        x       = g["value"].to_numpy(dtype=float)
        pred_ms = float(pred_map[wave_id_str])
        true_value = true_map.get(wave_id_str)
        true_ms = None if true_value is None or pd.isna(true_value) else float(true_value)
        abs_error = abs(pred_ms - true_ms) if true_ms is not None else None

        plt.figure(figsize=(9, 4.8))
        plt.plot(t, x, linewidth=1.5, color="#00528A", label="waveform")
        if true_ms is not None:
            plt.axvline(true_ms, color="#00AEEF", linestyle="-", linewidth=2.2, label=f"label = {true_ms:.4f} ms")
        plt.axvline(pred_ms, color="#EF4444", linestyle="--", linewidth=2.2, label=f"prediction = {pred_ms:.4f} ms")
        plt.xlabel("time_ms")
        plt.ylabel("value")
        suffix = f" | abs error={abs_error:.4f} ms" if abs_error is not None else ""
        plt.title(f"wave_id={wave_id_str}{suffix}")
        plt.legend()
        plt.tight_layout()

        # ใช้ wave_id_str เป็นชื่อไฟล์ตรงๆ → routes.py จะหาเจอด้วย f"{wave_id}.png"
        save_path = outdir / f"{wave_id_str}.png"
        plt.savefig(save_path, dpi=180)
        plt.close()
        count += 1

    print(f"Saved {count} waveform plots to: {outdir}")


if __name__ == "__main__":
    main()
