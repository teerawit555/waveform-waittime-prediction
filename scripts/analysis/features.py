from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LATE_KEYWORDS = [
    "last_big_slope",
    "num_big_slopes_after_half",
    "num_mean_shifts",
    "largest_late_mean_shift",
    "largest_late_shift_time_ms",
    "plateau",
    "stable_run",
    "tail_range_last_20pct",
    "tail_std_last_20pct",
    "std_last_20pct",
    "mean_abs_slope_last_20pct",
    "recovery",
    "final_band",
]


def feature_group(name: str) -> str:
    if name.startswith("tcn_embed_"):
        return "tcn_embedding"
    if any(k in name for k in LATE_KEYWORDS):
        return "late_settle"
    return "handcrafted_other"


def main() -> None:
    ap = argparse.ArgumentParser("analyze_feature_importance")
    ap.add_argument("--in", dest="input_csv", required=True, help="feature importance csv")
    ap.add_argument("--outdir", required=True, help="output folder")
    ap.add_argument("--topn", type=int, default=30, help="top N features to plot")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    # รองรับกรณีชื่อ feature อยู่ใน index ที่ถูก save มาเป็นคอลัมน์แรก
    if "feature" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "feature"})

    required = ["feature", "importance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df["group"] = df["feature"].astype(str).map(feature_group)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    # save cleaned full table
    df.to_csv(outdir / "feature_importance_full_sorted.csv", index=False)

    # top-N
    topn = df.head(args.topn).copy()
    topn.to_csv(outdir / f"top_{args.topn}_feature_importance.csv", index=False)

    # ---------- Plot 1: top-N barh ----------
    H = max(6, args.topn * 0.32)
    plt.figure(figsize=(10, H))
    bars = plt.barh(topn["feature"][::-1], topn["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top-{args.topn} Feature Importance")

    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        plt.text(width, y, f" {width:.4f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    plt.savefig(outdir / f"top_{args.topn}_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ---------- Plot 2: group contribution (sum of importance) ----------
    grp_sum = (
        df.groupby("group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    grp_sum.to_csv(outdir / "feature_group_importance_sum.csv", index=False)

    plt.figure(figsize=(10, H))
    bars = plt.bar(grp_sum["group"], grp_sum["importance"])
    plt.ylabel("Summed Importance")
    plt.title("Feature Group Importance (Sum)")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(outdir / "feature_group_importance_sum.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ---------- Plot 3: group count in top-N ----------
    grp_topn = (
        topn.groupby("group", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    grp_topn.to_csv(outdir / f"feature_group_top_{args.topn}_count.csv", index=False)

    plt.figure(figsize=(10, H))
    bars = plt.bar(grp_topn["group"], grp_topn["count"])
    plt.ylabel("Count")
    plt.title(f"Feature Group Count in Top-{args.topn}")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9
        )
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(outdir / f"feature_group_top_{args.topn}_count.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ---------- Summary text ----------
    top10 = df.head(10).copy()
    top20 = df.head(20).copy()
    top30 = df.head(30).copy()

    top10_late = top10[top10["group"] == "late_settle"]
    top10_tcn = top10[top10["group"] == "tcn_embedding"]

    top20_late = top20[top20["group"] == "late_settle"]
    top20_tcn = top20[top20["group"] == "tcn_embedding"]

    with open(outdir / "feature_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== Feature Importance Summary ===\n\n")
        f.write(f"Input file: {args.input_csv}\n")
        f.write(f"Total features analyzed: {len(df)}\n")
        f.write(f"Top-N analyzed: {args.topn}\n\n")

        f.write("Top-10 features:\n")
        for i, row in top10.iterrows():
            f.write(
                f"{i+1:02d}. {row['feature']} | importance={row['importance']:.6f} | group={row['group']}\n"
            )

        f.write("\n--- Group summary ---\n")
        for _, row in grp_sum.iterrows():
            f.write(f"{row['group']}: summed_importance={row['importance']:.6f}\n")

        f.write("\n--- Count in Top-30 ---\n")
        grp_top30 = (
            top30.groupby("group", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
        )
        for _, row in grp_top30.iterrows():
            f.write(f"{row['group']}: count={int(row['count'])}\n")

        f.write("\n--- Interpretation ---\n")
        if len(top10_late) > 0:
            f.write(
                f"Late-settle features appear in Top-10 ({len(top10_late)} features), "
                "suggesting the model is using delayed-transition / settling behavior.\n"
            )
        else:
            f.write(
                "Late-settle features do not appear in Top-10, so they may not be contributing strongly yet.\n"
            )

        if len(top10_tcn) > 0:
            f.write(
                f"TCN embeddings appear in Top-10 ({len(top10_tcn)} features), "
                "suggesting learned waveform representations are contributing strongly.\n"
            )
        else:
            f.write(
                "TCN embeddings do not appear in Top-10, so handcrafted features may be dominating.\n"
            )

        if len(top20_late) > 0 and len(top20_tcn) > 0:
            f.write(
                "Both late-settle handcrafted features and TCN embeddings are active in Top-20. "
                "This usually means the hybrid approach is working as intended.\n"
            )
        elif len(top20_late) > 0:
            f.write(
                "Late-settle handcrafted features dominate more clearly than TCN embeddings in Top-20.\n"
            )
        elif len(top20_tcn) > 0:
            f.write(
                "TCN embeddings dominate more clearly than late-settle handcrafted features in Top-20.\n"
            )
        else:
            f.write(
                "Neither late-settle features nor TCN embeddings appear strongly in Top-20. "
                "Review feature engineering and model setup.\n"
            )

    print(f"Saved feature analysis to: {outdir}")
    print(f"Top-{args.topn} plot: {outdir / f'top_{args.topn}_feature_importance.png'}")
    print(f"Summary: {outdir / 'feature_summary.txt'}")


if __name__ == "__main__":
    main()