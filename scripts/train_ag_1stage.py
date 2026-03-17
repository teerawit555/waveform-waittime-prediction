from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor


# timestamp used for creating unique model folders
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# default location where AutoGluon models will be saved
DEFAULT_SAVE_PATH = f"AutogluonModels/ag-1stage-{ts}"

# columns that will be dropped from training
# these may leak information or are not useful for prediction
COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]

# columns that should always be removed if present
DROP_ALWAYS = ["type"]


def _ensure_dir(path: str) -> None:
    """
    Ensure output directory exists.
    """
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    """
    Save Python object as JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def log(msg: str, path: str) -> None:
    """
    Print message to console and append to log file.
    """
    print(msg)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def group_split(df: pd.DataFrame, group_col: str, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/test based on group IDs.

    This ensures that samples belonging to the same waveform
    (same wave_id) do not appear in both train and test sets.

    Parameters
    ----------
    group_col : column containing group identifiers (wave_id)
    test_frac : fraction of groups used for test
    seed      : random seed

    Returns
    -------
    train_df, test_df
    """

    ids = df[group_col].drop_duplicates().to_numpy()

    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_test = int(round(len(ids) * test_frac))

    test_ids = set(ids[:n_test])

    tr = df[~df[group_col].isin(test_ids)].reset_index(drop=True)
    te = df[df[group_col].isin(test_ids)].reset_index(drop=True)

    return tr, te


def main() -> None:

    # ===============================
    # Parse command-line arguments
    # ===============================

    ap = argparse.ArgumentParser("train_ag_1stage")

    ap.add_argument("--data", required=True)         # input CSV file
    ap.add_argument("--label", default="wait_time_ms")  # regression target

    ap.add_argument("--model-dir", default=None)     # directory to save models
    ap.add_argument("--time-limit", type=int, default=300)  # training time limit (seconds)

    ap.add_argument("--presets", default="medium_quality")  # AutoGluon preset
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--test-frac", type=float, default=0.2)  # test split ratio
    ap.add_argument("--log-target", action="store_true")     # optional log transform

    args = ap.parse_args()

    # ===============================
    # Load dataset
    # ===============================

    df = pd.read_csv(args.data)

    # ensure label exists
    if args.label not in df.columns:
        raise KeyError(f"Missing label: {args.label}")

    # ensure wave_id exists (used for group split)
    if "wave_id" not in df.columns:
        df["wave_id"] = np.arange(len(df), dtype=int)

    # remove rows with missing label
    df = df.dropna(subset=[args.label]).reset_index(drop=True)

    # drop unwanted columns
    df = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns], errors="ignore")

    # ===============================
    # Train/Test split (group-aware)
    # ===============================

    train_df, test_df = group_split(
        df,
        group_col="wave_id",
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    # ===============================
    # Setup output directory
    # ===============================

    save_path = args.model_dir or DEFAULT_SAVE_PATH

    _ensure_dir(save_path)

    log_path = os.path.join(save_path, f"train_log_{ts}.txt")

    # ===============================
    # GPU detection
    # ===============================

    gpu_count = 1 if torch.cuda.is_available() else 0

    log(f"data={args.data}", log_path)
    log(f"rows(train)={len(train_df)} rows(test)={len(test_df)}", log_path)
    log(f"gpu={'yes' if gpu_count > 0 else 'no'}", log_path)

    # ===============================
    # Drop unwanted columns
    # ===============================

    cols_to_drop_found = [c for c in COLS_TO_DROP if c in train_df.columns]

    train_fit = train_df.drop(columns=cols_to_drop_found, errors="ignore").copy()
    test_fit = test_df.drop(columns=cols_to_drop_found, errors="ignore").copy()

    # ===============================
    # Target transformation (optional)
    # ===============================

    label_fit = args.label

    if args.log_target:

        # log transform stabilizes regression if label distribution is skewed
        train_fit["wait_time_log"] = np.log1p(np.clip(train_fit[args.label].astype(float), 0.0, None))
        test_fit["wait_time_log"] = np.log1p(np.clip(test_fit[args.label].astype(float), 0.0, None))

        label_fit = "wait_time_log"

    # ===============================
    # Determine feature columns
    # ===============================

    feature_cols = [c for c in train_fit.columns if c not in [args.label, "wait_time_log"]]

    save_json(os.path.join(save_path, "feature_cols.json"), feature_cols)

    # save metadata
    save_json(os.path.join(save_path, "meta.json"), {
        "label": args.label,
        "label_fit": label_fit,
        "log_target": bool(args.log_target),
        "seed": int(args.seed),
    })

    # ===============================
    # Train AutoGluon model
    # ===============================

    predictor = TabularPredictor(
        label=label_fit,
        path=save_path,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        verbosity=2,
    ).fit(

        train_data=train_fit[[*feature_cols, label_fit]],

        presets=args.presets,
        time_limit=args.time_limit,

        num_gpus=gpu_count,

        dynamic_stacking=False,
    )

    # ===============================
    # Feature importance analysis
    # ===============================

    print("\n=== Feature Importance ===")

    try:

        sub_n = min(3000, len(test_fit))

        # sample subset for importance computation
        df_imp = test_fit[[*feature_cols, label_fit]].sample(
            n=sub_n,
            random_state=args.seed
        ) if len(test_fit) > sub_n else test_fit[[*feature_cols, label_fit]].copy()

        fi = predictor.feature_importance(
            data=df_imp,
            subsample_size=sub_n,
            num_shuffle_sets=5,
            include_confidence_band=True,
        )

        fi_path = os.path.join(save_path, f"feature_importance_{ts}.csv")

        fi.to_csv(fi_path, index=True)

        print(f"saved: {fi_path}")

        print("\nTop-30 important features:")
        print(fi.sort_values("importance", ascending=False).head(30).to_string())

    except Exception as e:

        print(f"[WARN] feature importance failed: {e}")

    # ===============================
    # Test prediction
    # ===============================

    X_test = test_fit[feature_cols].copy()

    pred_fit = predictor.predict(X_test)

    pred_fit = np.asarray(pred_fit, dtype=float)

    # inverse transform if log target was used
    pred_ms = np.expm1(pred_fit) if args.log_target else pred_fit

    pred_ms = np.clip(pred_ms, 0.0, None)

    # compute MAE
    y_true = test_fit[args.label].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(pred_ms - y_true)))

    log(f"TEST MAE(ms)={mae:.6f}", log_path)

    # ===============================
    # Save predictions
    # ===============================

    pred_out = test_fit[["wave_id", args.label]].copy()

    pred_out["pred_wait_time_ms"] = pred_ms

    pred_out["abs_error"] = np.abs(
        pred_out["pred_wait_time_ms"] - pred_out[args.label]
    )

    pred_out.to_csv(
        os.path.join(save_path, f"test_predictions_{ts}.csv"),
        index=False
    )

    log(f"✅ Saved model at {save_path}", log_path)


if __name__ == "__main__":
    main()