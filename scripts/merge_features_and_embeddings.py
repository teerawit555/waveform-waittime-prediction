from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser("merge_features_and_embeddings")
    ap.add_argument("--features", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    emb = pd.read_csv(args.embeddings)

    if "wave_id" not in feat.columns or "wave_id" not in emb.columns:
        raise KeyError("Both files must contain wave_id")

    out = feat.merge(emb, on="wave_id", how="inner", validate="one_to_one")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out} | rows={len(out)} cols={len(out.columns)}")


if __name__ == "__main__":
    main()