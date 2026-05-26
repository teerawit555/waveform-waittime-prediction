from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


COLS_TO_DROP = ["force_mA", "range_V", "temp_C", "type"]
LABEL_LEAK_COLS = ["wait_time_ms", "wait_time_log", "is_fast", "is_zero"]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        print(f"[WARN] {len(missing)} missing features filled with 0: {missing[:10]}")
        for c in missing:
            out[c] = 0.0
    return out[required_cols]


def main() -> None:
    ap = argparse.ArgumentParser("predict_ag_1stage")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--in", dest="input_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[1/5] loading model...")
    predictor = TabularPredictor.load(args.model_path)

    print("[2/5] loading metadata...")
    meta = load_json(os.path.join(args.model_path, "meta.json"))
    feature_cols = load_json(os.path.join(args.model_path, "feature_cols.json"))

    print("[3/5] reading input...")
    df = pd.read_csv(args.input_csv)
    wave_id = df["wave_id"] if "wave_id" in df.columns else pd.Series(np.arange(len(df)))

    df_feat = df.drop(columns=COLS_TO_DROP + LABEL_LEAK_COLS, errors="ignore").copy()
    X = align_columns(df_feat, feature_cols)

    print(f"[4/5] predicting rows={len(X)}...")
    pred_fit = predictor.predict(X)
    pred_fit = np.asarray(pred_fit, dtype=float)
    pred_ms = np.expm1(pred_fit) if bool(meta.get("log_target", False)) else pred_fit
    pred_ms = np.clip(pred_ms, 0.0, None)

    out = pd.DataFrame({
        "wave_id": wave_id,
        "pred_wait_time_ms": pred_ms,
        "pred_is_fast_at_0p1": (pred_ms <= 0.1 + 1e-12).astype(int),
    })

    print("[5/5] saving output...")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out} | rows={len(out)}")


if __name__ == "__main__":
    main()