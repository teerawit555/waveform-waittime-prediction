from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "Split waveform CSV by wave_id so the same waveform never appears in multiple sets."
    )
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--outdir", default="data/splits")
    ap.add_argument("--id-col", default="wave_id")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--stratify-cols", default="type")
    ap.add_argument("--label-col", default="wait_time_ms")
    ap.add_argument("--target-bins", type=int, default=6)
    return ap.parse_args()


def validate_fracs(train_frac: float, valid_frac: float, test_frac: float) -> None:
    total = train_frac + valid_frac + test_frac
    if min(train_frac, valid_frac, test_frac) < 0:
        raise ValueError("Split fractions must be non-negative")
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")


def main() -> None:
    args = parse_args()
    validate_fracs(args.train_frac, args.valid_frac, args.test_frac)

    in_path = Path(args.in_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.id_col not in df.columns:
        raise KeyError(f"Missing id column: {args.id_col}")

    wave_meta = df.groupby(args.id_col, sort=False).first().reset_index()
    strat_cols = [col.strip() for col in args.stratify_cols.split(",") if col.strip()]
    keys = []
    for col in strat_cols:
        if col in wave_meta.columns:
            keys.append(wave_meta[col].astype(str))
    if args.label_col in wave_meta.columns and int(args.target_bins) > 1:
        labels = wave_meta[args.label_col].astype(float)
        try:
            bins = pd.qcut(labels, q=int(args.target_bins), duplicates="drop").astype(str)
        except ValueError:
            bins = pd.Series(["all"] * len(wave_meta), index=wave_meta.index)
        keys.append(bins)

    if keys:
        strat_key = keys[0]
        for key in keys[1:]:
            strat_key = strat_key + "|" + key
        wave_meta["_strat_key"] = strat_key
    else:
        wave_meta["_strat_key"] = "all"

    rng = np.random.default_rng(args.seed)
    ids_by_split = {"train": set(), "valid": set(), "test": set()}
    residual = {"train": 0.0, "valid": 0.0, "test": 0.0}
    fractions = {
        "train": float(args.train_frac),
        "valid": float(args.valid_frac),
        "test": float(args.test_frac),
    }

    groups = [group for _, group in wave_meta.groupby("_strat_key", sort=False)]
    group_order = rng.permutation(len(groups))

    for group_idx in group_order:
        group = groups[int(group_idx)]
        wave_ids = group[args.id_col].to_numpy()
        rng.shuffle(wave_ids)
        n_total = len(wave_ids)

        split_counts = {}
        assigned = 0
        for split_name, frac in fractions.items():
            desired = residual[split_name] + frac * n_total
            count = int(np.floor(desired))
            split_counts[split_name] = count
            residual[split_name] = desired - count
            assigned += count

        for _ in range(n_total - assigned):
            split_name = max(residual, key=residual.get)
            split_counts[split_name] += 1
            residual[split_name] -= 1.0

        start = 0
        for split_name in ("train", "valid", "test"):
            count = split_counts[split_name]
            ids_by_split[split_name].update(wave_ids[start:start + count])
            start += count

    train_ids = ids_by_split["train"]
    valid_ids = ids_by_split["valid"]
    test_ids = ids_by_split["test"]

    parts = {
        "train": df[df[args.id_col].isin(train_ids)],
        "valid": df[df[args.id_col].isin(valid_ids)],
        "test": df[df[args.id_col].isin(test_ids)],
    }

    name_prefix = f"{args.prefix}_" if args.prefix else ""
    for split_name, split_df in parts.items():
        out_path = outdir / f"{name_prefix}{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        n_waves = split_df[args.id_col].nunique()
        print(f"{split_name}: waves={n_waves} rows={len(split_df)} -> {out_path}")

    manifest = pd.DataFrame(
        [
            {"split": "train", args.id_col: wid} for wid in sorted(train_ids)
        ]
        + [{"split": "valid", args.id_col: wid} for wid in sorted(valid_ids)]
        + [{"split": "test", args.id_col: wid} for wid in sorted(test_ids)]
    )
    manifest.to_csv(outdir / f"{name_prefix}split_manifest.csv", index=False)
    print(f"manifest -> {outdir / f'{name_prefix}split_manifest.csv'}")


if __name__ == "__main__":
    main()
