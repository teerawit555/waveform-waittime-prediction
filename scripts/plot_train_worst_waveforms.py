from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser("plot_train_worst_waveforms")
    ap.add_argument("--raw", required=True, help="raw waveform csv, e.g. data/raw/train.csv")
    ap.add_argument("--worst", required=True, help="worst_cases.csv from analysis")
    ap.add_argument("--outdir", required=True, help="output folder for plots")
    ap.add_argument("--topk", type=int, default=30, help="number of worst cases to plot")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    worst = pd.read_csv(args.worst)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    required_raw = ["wave_id", "sample", "time_ms", "value"]
    missing_raw = [c for c in required_raw if c not in raw.columns]
    if missing_raw:
        raise KeyError(f"raw file missing columns: {missing_raw}")

    required_worst = ["wave_id", "wait_time_ms", "pred_wait_time_ms", "abs_error"]
    missing_worst = [c for c in required_worst if c not in worst.columns]
    if missing_worst:
        raise KeyError(f"worst file missing columns: {missing_worst}")

    raw = raw.sort_values(["wave_id", "sample"]).copy()
    worst = worst.sort_values("abs_error", ascending=False).head(args.topk).copy()

    count = 0
    for _, row in worst.iterrows():
        wave_id = row["wave_id"]
        true_ms = float(row["wait_time_ms"])
        pred_ms = float(row["pred_wait_time_ms"])
        abs_error = float(row["abs_error"])

        g = raw[raw["wave_id"] == wave_id].copy()
        if len(g) == 0:
            continue

        t = g["time_ms"].to_numpy(dtype=float)
        x = g["value"].to_numpy(dtype=float)

        plt.figure(figsize=(9, 4.8))
        plt.plot(t, x, linewidth=1.5, label="waveform")
        plt.axvline(true_ms, linestyle="--", color="green" , linewidth=4, label=f"true = {true_ms:.4f} ms")
        plt.axvline(pred_ms, linestyle="--", color="red" ,linewidth=2, label=f"pred = {pred_ms:.4f} ms")

        plt.xlabel("time_ms")
        plt.ylabel("value")
        plt.title(f"wave_id={int(wave_id)} | abs_error={abs_error:.4f} ms")
        plt.legend()
        plt.tight_layout()

        save_path = outdir / f"wave_{int(wave_id)}.png"
        plt.savefig(save_path, dpi=180)
        plt.close()
        count += 1

    print(f"Saved {count} waveform plots to: {outdir}")


if __name__ == "__main__":
    main()