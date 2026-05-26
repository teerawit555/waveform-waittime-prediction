from __future__ import annotations

import argparse
import inspect
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor


TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_SAVE_PATH = f"AutogluonModels/ag-1stage-{TS}"

# Keep identifiers, labels, and label-derived/debug columns out of model features.
NON_FEATURE_COLS = {
    "wave_id",
    "dbg_label_reason",
    "force_mA",
    "range_V",
    "temp_C",
    "type",
    "true_wait_time_ms",
    "true_is_zero",
    "wait_time_log",
    "sample_weight",
    "is_fast",
    "is_zero",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("train_ag_1stage")
    ap.add_argument("--data", default=None, help="Legacy mode: one CSV, split inside this script")
    ap.add_argument("--train", default=None, help="Preferred: explicit train hybrid CSV")
    ap.add_argument("--valid", default=None, help="Optional explicit validation hybrid CSV")
    ap.add_argument("--test", default=None, help="Preferred: explicit final test hybrid CSV")
    ap.add_argument("--label", default="wait_time_ms")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--presets", default="medium_quality")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("--weight-fast-ms", type=float, default=0.1)
    ap.add_argument("--fast-weight", type=float, default=3.0)
    return ap.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def log(msg: str, path: str) -> None:
    print(msg)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def group_split(df: pd.DataFrame, group_col: str, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = df[group_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_frac))
    test_ids = set(ids[:n_test])
    train = df[~df[group_col].isin(test_ids)].reset_index(drop=True)
    test = df[df[group_col].isin(test_ids)].reset_index(drop=True)
    return train, test


def load_training_frame(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label not in df.columns:
        raise KeyError(f"Missing label: {label}")
    if "wave_id" not in df.columns:
        df["wave_id"] = np.arange(len(df), dtype=int)
    return df.dropna(subset=[label]).reset_index(drop=True)


def add_fit_label(df: pd.DataFrame, label: str, log_target: bool) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if not log_target:
        return out, label
    out["wait_time_log"] = np.log1p(np.clip(out[label].astype(float), 0.0, None))
    return out, "wait_time_log"


def add_sample_weight(df: pd.DataFrame, label: str, fast_ms: float, fast_weight: float) -> pd.DataFrame:
    out = df.copy()
    weights = np.ones(len(out), dtype=float)
    if fast_ms > 0 and fast_weight > 1:
        weights[out[label].astype(float).to_numpy() <= fast_ms] = float(fast_weight)
    out["sample_weight"] = weights
    return out


def select_feature_cols(df: pd.DataFrame, label: str, label_fit: str) -> list[str]:
    excluded = set(NON_FEATURE_COLS)
    excluded.add(label)
    excluded.add(label_fit)
    return [c for c in df.columns if c not in excluded]


def frame_for_autogluon(df: pd.DataFrame, feature_cols: list[str], label_fit: str) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:20]}")
    return df[[*feature_cols, label_fit]].copy()


def predict_and_save(
    predictor: TabularPredictor,
    df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    split_name: str,
    save_path: str,
    model_suffix: str,
    log_target: bool,
) -> dict[str, float]:
    pred_fit = predictor.predict(df[feature_cols].copy())
    pred_fit = np.asarray(pred_fit, dtype=float)
    pred_ms = np.expm1(pred_fit) if log_target else pred_fit
    pred_ms = np.clip(pred_ms, 0.0, None)

    y_true = df[label].to_numpy(dtype=float)
    abs_error = np.abs(pred_ms - y_true)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean((pred_ms - y_true) ** 2)))

    if "wave_id" in df.columns:
        pred_out = df[["wave_id", label]].copy()
    else:
        pred_out = df[[label]].copy()
    pred_out["pred_wait_time_ms"] = pred_ms
    pred_out["abs_error"] = abs_error
    pred_out.to_csv(
        os.path.join(save_path, f"{split_name}_predictions_{model_suffix}.csv"),
        index=False,
    )

    return {"mae": mae, "rmse": rmse, "rows": float(len(df))}


def main() -> None:
    args = parse_args()

    if args.train:
        train_df = load_training_frame(args.train, args.label)
        valid_df = load_training_frame(args.valid, args.label) if args.valid else None
        test_df = load_training_frame(args.test, args.label) if args.test else None
    elif args.data:
        df = load_training_frame(args.data, args.label)
        train_df, test_df = group_split(df, "wave_id", float(args.test_frac), int(args.seed))
        valid_df = None
    else:
        raise ValueError("Provide either --data or --train. Prefer --train/--valid/--test.")

    save_path = args.model_dir or DEFAULT_SAVE_PATH
    ensure_dir(save_path)
    model_suffix = args.model_name or TS
    log_path = os.path.join(save_path, f"train_log_{TS}.txt")

    gpu_count = 1 if torch.cuda.is_available() else 0
    log(f"train={args.train or args.data}", log_path)
    if args.valid:
        log(f"valid={args.valid}", log_path)
    if args.test:
        log(f"test={args.test}", log_path)
    log(f"rows(train)={len(train_df)}", log_path)
    if valid_df is not None:
        log(f"rows(valid)={len(valid_df)}", log_path)
    if test_df is not None:
        log(f"rows(test)={len(test_df)}", log_path)
    log(f"gpu={'yes' if gpu_count > 0 else 'no'}", log_path)

    train_fit, label_fit = add_fit_label(train_df, args.label, bool(args.log_target))
    valid_fit = add_fit_label(valid_df, args.label, bool(args.log_target))[0] if valid_df is not None else None
    test_fit = add_fit_label(test_df, args.label, bool(args.log_target))[0] if test_df is not None else None

    feature_cols = select_feature_cols(train_fit, args.label, label_fit)
    save_json(os.path.join(save_path, "feature_cols.json"), feature_cols)
    save_json(
        os.path.join(save_path, "meta.json"),
        {
            "label": args.label,
            "label_fit": label_fit,
            "log_target": bool(args.log_target),
            "seed": int(args.seed),
            "train_path": args.train or args.data,
            "valid_path": args.valid,
            "test_path": args.test,
            "feature_count": len(feature_cols),
            "excluded_feature_cols": sorted(NON_FEATURE_COLS),
            "sample_weight": {
                "weight_fast_ms": float(args.weight_fast_ms),
                "fast_weight": float(args.fast_weight),
            },
        },
    )

    weighted_train = add_sample_weight(train_fit, args.label, args.weight_fast_ms, args.fast_weight)
    fit_kwargs: dict[str, Any] = {
        "train_data": frame_for_autogluon(weighted_train, [*feature_cols, "sample_weight"], label_fit),
        "presets": args.presets,
        "time_limit": args.time_limit,
        "num_gpus": gpu_count,
        "dynamic_stacking": False,
    }
    if "sample_weight" in inspect.signature(TabularPredictor.fit).parameters:
        fit_kwargs["sample_weight"] = "sample_weight"
    else:
        log("[WARN] AutoGluon fit() does not expose sample_weight; continuing without weighted rows.", log_path)
        fit_kwargs["train_data"] = frame_for_autogluon(train_fit, feature_cols, label_fit)
    if valid_fit is not None:
        fit_kwargs["tuning_data"] = frame_for_autogluon(valid_fit, feature_cols, label_fit)

    predictor = TabularPredictor(
        label=label_fit,
        path=save_path,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        verbosity=2,
    ).fit(**fit_kwargs)

    print("\n=== Feature Importance ===")
    try:
        imp_fit = valid_fit if valid_fit is not None else test_fit
        if imp_fit is None:
            imp_fit = train_fit
        sub_n = min(3000, len(imp_fit))
        df_imp = frame_for_autogluon(imp_fit, feature_cols, label_fit)
        if len(df_imp) > sub_n:
            df_imp = df_imp.sample(n=sub_n, random_state=args.seed)

        fi = predictor.feature_importance(
            data=df_imp,
            subsample_size=sub_n,
            num_shuffle_sets=5,
            include_confidence_band=True,
        )
        fi_path = os.path.join(save_path, f"feature_importance_{model_suffix}.csv")
        fi.to_csv(fi_path, index=True)
        print(f"saved: {fi_path}")
        print("\nTop-30 important features:")
        print(fi.sort_values("importance", ascending=False).head(30).to_string())
    except Exception as exc:
        print(f"[WARN] feature importance failed: {exc}")

    metrics: dict[str, dict[str, float]] = {}
    if valid_fit is not None:
        metrics["valid"] = predict_and_save(
            predictor, valid_fit, feature_cols, args.label, "valid", save_path, model_suffix, bool(args.log_target)
        )
        log(f"VALID MAE(ms)={metrics['valid']['mae']:.6f} RMSE={metrics['valid']['rmse']:.6f}", log_path)
    if test_fit is not None:
        metrics["test"] = predict_and_save(
            predictor, test_fit, feature_cols, args.label, "test", save_path, model_suffix, bool(args.log_target)
        )
        log(f"TEST MAE(ms)={metrics['test']['mae']:.6f} RMSE={metrics['test']['rmse']:.6f}", log_path)
    save_json(os.path.join(save_path, "metrics.json"), metrics)
    log(f"Saved model at {save_path}", log_path)


if __name__ == "__main__":
    main()
