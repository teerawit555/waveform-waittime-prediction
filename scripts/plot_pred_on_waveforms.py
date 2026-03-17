from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser("plot_pred_on_waveforms")
    ap.add_argument("--raw", required=True, help="raw waveform csv")
    ap.add_argument("--pred", required=True, help="prediction csv with wave_id + pred_wait_time_ms")
    ap.add_argument("--outdir", required=True, help="output folder for plots")
    ap.add_argument("--topk", type=int, default=30, help="number of waveforms to plot")
    ap.add_argument("--mode", choices=["first", "low", "high", "random"], default="first")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw)
    pred = pd.read_csv(args.pred)

    required_raw = ["wave_id", "sample", "time_ms", "value"]
    missing_raw = [c for c in required_raw if c not in raw.columns]
    if missing_raw:
        raise KeyError(f"raw file missing columns: {missing_raw}")

    required_pred = ["wave_id", "pred_wait_time_ms"]
    missing_pred = [c for c in required_pred if c not in pred.columns]
    if missing_pred:
        raise KeyError(f"pred file missing columns: {missing_pred}")

    raw = raw.sort_values(["wave_id", "sample"]).copy()
    pred = pred[["wave_id", "pred_wait_time_ms"]].drop_duplicates("wave_id").copy()

    merged_ids = pred["wave_id"].tolist()

    if args.mode == "first":
        chosen = merged_ids[: args.topk]
    elif args.mode == "low":
        chosen = pred.sort_values("pred_wait_time_ms", ascending=True).head(args.topk)["wave_id"].tolist()
    elif args.mode == "high":
        chosen = pred.sort_values("pred_wait_time_ms", ascending=False).head(args.topk)["wave_id"].tolist()
    else:
        rng = np.random.default_rng(args.seed)
        chosen = pred.sample(n=min(args.topk, len(pred)), random_state=args.seed)["wave_id"].tolist()

    pred_map = dict(zip(pred["wave_id"], pred["pred_wait_time_ms"]))

    count = 0
    for wave_id in chosen:
        g = raw[raw["wave_id"] == wave_id].copy()
        if len(g) == 0:
            continue

        t = g["time_ms"].to_numpy(dtype=float)
        x = g["value"].to_numpy(dtype=float)
        pred_ms = float(pred_map[wave_id])

        plt.figure(figsize=(9, 4.8))
        plt.plot(t, x, linewidth=1.5, label="waveform")
        plt.axvline(pred_ms, linestyle="--", linewidth=2, label=f"pred = {pred_ms:.4f} ms")
        plt.xlabel("time_ms")
        plt.ylabel("value")
        plt.title(f"wave_id={wave_id}")
        plt.legend()
        plt.tight_layout()

        save_path = outdir / f"wave_{int(wave_id)}.png"
        plt.savefig(save_path, dpi=180)
        plt.close()
        count += 1

    print(f"✅ Saved {count} waveform plots to: {outdir}")


if __name__ == "__main__":
    main()