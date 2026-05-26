from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser("analyze_regression_preds")
    ap.add_argument("--in", dest="input_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fast-ms", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    if "wait_time_ms" not in df.columns:
        df["wait_time_ms"] = df["true_wait_time"]

    if "pred_wait_time_ms" not in df.columns:
        df["pred_wait_time_ms"] = df["pred_wait_time"]

    y_true = df["wait_time_ms"].to_numpy(dtype=float)
    y_pred = df["pred_wait_time_ms"].to_numpy(dtype=float)

    df["error"] = y_pred - y_true
    df["abs_error"] = np.abs(df["error"])

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    fast_mask = y_true <= args.fast_ms + 1e-12
    slow_mask = ~fast_mask

    mae_fast = float(np.mean(np.abs(y_pred[fast_mask] - y_true[fast_mask]))) if np.any(fast_mask) else float("nan")
    mae_slow = float(np.mean(np.abs(y_pred[slow_mask] - y_true[slow_mask]))) if np.any(slow_mask) else float("nan")

    pred_fast = y_pred <= args.fast_ms + 1e-12
    true_fast = fast_mask

    tp = int(np.sum(pred_fast & true_fast))
    fp = int(np.sum(pred_fast & ~true_fast))
    fn = int(np.sum(~pred_fast & true_fast))

    fast_precision = tp / max(tp + fp, 1)
    fast_recall = tp / max(tp + fn, 1)

    summary_path = outdir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"MAE(all): {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE(fast<= {args.fast_ms} ms): {mae_fast:.6f}\n")
        f.write(f"MAE(slow> {args.fast_ms} ms): {mae_slow:.6f}\n")
        f.write(f"Fast precision@{args.fast_ms}: {fast_precision:.6f}\n")
        f.write(f"Fast recall@{args.fast_ms}: {fast_recall:.6f}\n")
        f.write(f"Pred fast count: {int(pred_fast.sum())}\n")
        f.write(f"True fast count: {int(true_fast.sum())}\n")
        f.write(f"False fast count: {fp}\n")

    # 1) true vs pred scatter
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lim_max = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([0, lim_max], [0, lim_max], linestyle="--")
    plt.xlabel("True wait_time_ms")
    plt.ylabel("Pred wait_time_ms")
    plt.title("True vs Pred")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_true_vs_pred.png", dpi=180)
    plt.close()

    # 2) residual plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, df["error"], s=10, alpha=0.6)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("True wait_time_ms")
    plt.ylabel("Pred - True")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(outdir / "residual_plot.png", dpi=180)
    plt.close()

    # 3) abs error histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["abs_error"], bins=50)
    plt.xlabel("Absolute Error (ms)")
    plt.ylabel("Count")
    plt.title("Absolute Error Histogram")
    plt.tight_layout()
    plt.savefig(outdir / "abs_error_hist.png", dpi=180)
    plt.close()

    # 4) true/pred histogram overlay
    plt.figure(figsize=(8, 5))
    plt.hist(y_true, bins=50, alpha=0.6, label="True")
    plt.hist(y_pred, bins=50, alpha=0.6, label="Pred")
    plt.xlabel("wait_time_ms")
    plt.ylabel("Count")
    plt.title("Distribution: True vs Pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "dist_true_vs_pred.png", dpi=180)
    plt.close()

    # 5) save worst cases
    sort_cols = ["abs_error"]
    if "wave_id" in df.columns:
        cols = ["wave_id", "wait_time_ms", "pred_wait_time_ms", "error", "abs_error"]
    else:
        cols = ["wait_time_ms", "pred_wait_time_ms", "error", "abs_error"]

    worst = df.sort_values(sort_cols, ascending=False).head(args.topk)[cols]
    worst.to_csv(outdir / "worst_cases.csv", index=False)

    print(f"Saved analysis to: {outdir}")
    print(f"MAE(all)={mae:.6f} | RMSE={rmse:.6f}")
    print(f"MAE(fast)={mae_fast:.6f} | MAE(slow)={mae_slow:.6f}")
    print(f"Fast precision={fast_precision:.6f} | Fast recall={fast_recall:.6f} | False fast={fp}")


if __name__ == "__main__":
    main()