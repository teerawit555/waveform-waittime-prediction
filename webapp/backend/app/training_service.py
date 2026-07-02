# ==============================================================================
# This file manages asynchronous training and prediction pipelines.
# ==============================================================================
from __future__ import annotations

import subprocess
import uuid
import sys
import shutil
import pandas as pd
import json
import os

from pathlib import Path
from .config import (
    ANALYSIS_DIR,
    AUTOGLUON_DIR,
    DATA_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_AG_PRESETS,
    DEFAULT_AG_TIME_LIMIT,
    DEFAULT_TCN_AUGMENT,
    DEFAULT_TCN_EPOCHS,
    DEFAULT_TCN_EARLY_STOPPING_PATIENCE,
    DEFAULT_TCN_FAST_WEIGHT,
    DEFAULT_TCN_NOISE_STD,
    DEFAULT_TCN_SCALE_JITTER,
    DEFAULT_TCN_TIME_SHIFT,
    MODELS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPT_ANALYSIS_DIR,
    SCRIPT_AUTOGLUON_DIR,
    SCRIPT_DATA_DIR,
    SCRIPT_FEATURES_DIR,
    SCRIPT_TCN_DIR,
    TCN_DIR,
    UPLOAD_DIR,
)
from .job_store import job_store
from .job_queue import job_queue
from .mlflow_service import (
    begin_training_run,
    log_artifacts as mlflow_log_artifacts,
    log_metrics as mlflow_log_metrics,
    log_model_dirs_enabled,
    log_params as mlflow_log_params,
    register_model_version as mlflow_register_model_version,
    set_tags as mlflow_set_tags,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

SCRIPT_PATHS = {
    "analyze_regression_preds": SCRIPT_ANALYSIS_DIR / "analyze_regression_preds.py",
    "convert_signal_csv": SCRIPT_DATA_DIR / "convert_signal_csv.py",
    "features": SCRIPT_ANALYSIS_DIR / "features.py",
    "plot_pred_on_waveforms": SCRIPT_ANALYSIS_DIR / "plot_pred_on_waveforms.py",
    "split_csv_train_test": SCRIPT_DATA_DIR / "split_csv_train_test.py",
    "make_wave_tensor": SCRIPT_DATA_DIR / "make_wave_tensor.py",
    "extract_features": SCRIPT_FEATURES_DIR / "extract_features.py",
    "merge_features_and_embeddings": SCRIPT_FEATURES_DIR / "merge_features_and_embeddings.py",
    "predict_ag_1stage": SCRIPT_AUTOGLUON_DIR / "predict_ag_1stage.py",
    "train_ag_1stage": SCRIPT_AUTOGLUON_DIR / "train_ag_1stage.py",
    "export_tcn_encoder": SCRIPT_TCN_DIR / "export_tcn_encoder.py",
    "train_tcn_encoder": SCRIPT_TCN_DIR / "train_tcn_encoder.py",
    "predict_single_wave": SCRIPT_TCN_DIR.parent / "pipelines" / "predict_single_wave.py",
    "predict_batch_waves": SCRIPT_TCN_DIR.parent / "pipelines" / "predict_batch_waves.py",
}

def run_cmd(cmd: list[str]):
    """รัน subprocess command แล้วคืนค่า stdout; raise CalledProcessError ถ้าล้มเหลว"""
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=True,
    )
    return result.stdout


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def cleanup_failed_model_dirs(paths: list[Path]) -> list[str]:
    removed: list[str] = []
    allowed_roots = (AUTOGLUON_DIR, TCN_DIR)
    for path in paths:
        if not path:
            continue
        resolved = path.resolve()
        if not any(_is_relative_to(resolved, root) for root in allowed_roots):
            continue
        if resolved.exists():
            shutil.rmtree(resolved, ignore_errors=True)
            removed.append(str(resolved))
    return removed


def list_pngs(base: Path, job_id: str, category: str):
    """
    สแกนหาไฟล์ .png ทั้งหมดใน directory `base` แบบ recursive
    แล้วคืนค่าเป็น list ของ URL path สำหรับ serve ผ่าน API
    เช่น /api/files/<category>/<job_id>/<relative_path>
    """
    if not base.exists():
        return []
    paths = []
    for p in sorted(base.rglob("*.png")):
        rel = p.relative_to(base)
        paths.append(f"/api/files/{category}/{job_id}/{rel.as_posix()}")
    return paths


def build_analysis_manifest(pred_csv: Path, analysis_dir: Path, job_id: str, category: str):
    """
    สร้าง manifest สำหรับหน้า Analysis โดย join ข้อมูลจาก 2 แหล่ง:
      - pred_csv  : ผลการ predict (wave_id, pred, true)
      - analysis_dir : ไฟล์ภาพ .png ที่ตั้งชื่อตาม wave_id

    คืนค่าเป็น list of dict ที่มี:
        wave_id, image (URL), pred (float|None), true (float|None)
    """
    import pandas as pd

    items = []
    if not pred_csv.exists() or not analysis_dir.exists():
        return items

    pred_df = pd.read_csv(pred_csv)
    image_files = sorted(analysis_dir.glob("*.png"))

    for img in image_files:
        stem = img.stem  # ใช้ชื่อไฟล์ (ไม่มีนามสกุล) เป็น wave_id
        item = {
            "wave_id": stem,
            "image": f"/api/files/{category}/{job_id}/{img.name}",
            "pred": None,
            "true": None,
        }

        # จับคู่กับแถวใน prediction CSV ถ้ามี column wave_id
        if "wave_id" in pred_df.columns:
            matched = pred_df[pred_df["wave_id"].astype(str) == stem]
            if not matched.empty:
                row = matched.iloc[0]
                # หา prediction value จาก column ที่เป็นไปได้หลายชื่อ
                for col in ["pred", "prediction", "pred_wait_time_ms", "pred_wait_time"]:
                    if col in pred_df.columns:
                        item["pred"] = float(row[col])
                        break
                # หา ground-truth value
                for col in ["wait_time_ms", "true", "true_wait_time", "label"]:
                    if col in pred_df.columns:
                        item["true"] = float(row[col])
                        break
                val = row["wave_id"]
                if hasattr(val, "item"):
                    val = val.item()
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                item["wave_id"] = val
                if item["pred"] is not None and item["true"] is not None:
                    item["error"] = float(item["pred"]) - float(item["true"])
                    item["abs_error"] = abs(item["error"])

        items.append(item)

    return items


def sanitize_model_name(name: str) -> str:
    """
    ทำความสะอาดชื่อ model ให้ปลอดภัยสำหรับใช้เป็น directory name
    - เก็บเฉพาะ alphanumeric, '-', '_'
    - ตัด underscore ที่ขึ้นต้น/ลงท้ายออก
    - ถ้าผลลัพธ์ว่างเปล่า ให้ใช้ random hex แทน
    """
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    safe = safe.strip("_")
    return safe or f"model_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Model listing helpers
# ---------------------------------------------------------------------------

def list_available_models():
    """
    สแกน AUTOGLUON_DIR เพื่อ list โมเดล AutoGluon ทั้งหมดที่มีอยู่
    แต่ละโมเดลจะอ่าน model_meta.json เพื่อหา TCN path ที่เชื่อมกัน
    คืนค่า list of dict: name, tcn_path, ag_path, ready (bool)
    """
    items = []
    if not AUTOGLUON_DIR.exists():
        return items
    for ag_dir in sorted(AUTOGLUON_DIR.iterdir()):
        if not ag_dir.is_dir():
            continue
        meta_file = ag_dir / "model_meta.json"
        if meta_file.exists():
            meta    = json.loads(meta_file.read_text())
            tcn_dir = Path(meta["tcn_path"])
        else:
            tcn_dir = TCN_DIR / ag_dir.name  # fallback กรณีไม่มี metadata
        items.append({
            "name":     ag_dir.name,
            "tcn_path": str(tcn_dir),
            "ag_path":  str(ag_dir),
            "ready":    tcn_dir.exists(),  # ready = TCN model ยังอยู่ครบ
        })
    return items


def list_available_tcn_models():
    """
    สแกน TCN_DIR เพื่อ list TCN encoder model ทั้งหมดที่ train ไว้
    คืนค่า list of dict: name, path, ready (always True ถ้า dir มีอยู่)
    """
    items = []
    if not TCN_DIR.exists():
        return items

    for tcn_dir in sorted(TCN_DIR.iterdir()):
        if not tcn_dir.is_dir():
            continue
        items.append({
            "name": tcn_dir.name,
            "path": str(tcn_dir),
            "ready": True,
        })
    return items


# ---------------------------------------------------------------------------
# Post-training analysis helpers
# ---------------------------------------------------------------------------

def parse_feature_summary(fi_analysis_dir: Path, topn: int = 30) -> dict:
    """
    อ่าน CSV ผลวิเคราะห์ feature importance แล้วสรุปเป็น dict พร้อม report

    อ่านจาก 3 ไฟล์:
      - feature_importance_full_sorted.csv  → จำนวน feature ทั้งหมด
      - feature_group_importance_sum.csv    → importance รวมแยกตาม group
      - feature_group_top_<topn>_count.csv  → จำนวน feature ใน top-N แยกตาม group

    Feature groups: tcn_embedding, late_settle, handcrafted_other
    """
    summary = {
        "total_features": 0,
        "topn": topn,
        "group_sum": {
            "tcn_embedding": 0.0,
            "late_settle": 0.0,
            "handcrafted_other": 0.0,
        },
        "top30_count": {
            "tcn_embedding": 0,
            "late_settle": 0,
            "handcrafted_other": 0,
        },
    }

    if not fi_analysis_dir.exists():
        return summary

    full_csv     = fi_analysis_dir / "feature_importance_full_sorted.csv"
    grp_sum_csv  = fi_analysis_dir / "feature_group_importance_sum.csv"
    grp_topn_csv = fi_analysis_dir / f"feature_group_top_{topn}_count.csv"

    if full_csv.exists():
        df_full = pd.read_csv(full_csv)
        summary["total_features"] = int(len(df_full))

    if grp_sum_csv.exists():
        df_grp_sum = pd.read_csv(grp_sum_csv)
        for _, row in df_grp_sum.iterrows():
            group = str(row["group"])
            importance = float(row["importance"])
            if group in summary["group_sum"]:
                summary["group_sum"][group] = importance

    if grp_topn_csv.exists():
        df_grp_topn = pd.read_csv(grp_topn_csv)
        for _, row in df_grp_topn.iterrows():
            group = str(row["group"])
            count = int(row["count"])
            if group in summary["top30_count"]:
                summary["top30_count"][group] = count

    return summary


def analyze_overfitting(history_path: Path) -> dict:
    """
    วิเคราะห์ overfitting จาก train_history.json ของ TCN

    เกณฑ์การตัดสิน:
      - Strong Overfitting : val_rise > 0.02 และ gap_final > 0.02
      - Mild Overfitting   : val_rise > 0.005 หรือ gap_final > 0.01
      - Good Fit           : อื่นๆ

    โดยที่:
      val_rise  = final_val_loss - best_val_loss  (val loss เพิ่มขึ้นหลัง best epoch)
      gap_final = final_val_loss - final_train_loss
    """
    if not history_path.exists():
        return {"status": "unknown", "label": "Unknown", "message": "No history found"}
    
    history = json.loads(history_path.read_text())
    
    train_losses = [h["train_loss"] for h in history]
    val_losses   = [h["valid_loss"] for h in history]
    
    best_idx        = val_losses.index(min(val_losses))
    best_epoch      = best_idx + 1
    train_loss_best = train_losses[best_idx]
    val_loss_best   = val_losses[best_idx]
    final_train     = train_losses[-1]
    final_val       = val_losses[-1]
    gap_best        = val_loss_best - train_loss_best   # gap ที่ best epoch
    gap_final       = final_val - final_train           # gap ที่ epoch สุดท้าย
    val_rise        = final_val - val_loss_best         # val loss เพิ่มขึ้นเท่าไหร่หลัง best epoch

    if val_rise > 0.02 and gap_final > 0.02:
        status = "strong"
        label  = "Strong Overfitting"
        msg    = f"Val loss rose {val_rise:.4f} after best epoch. Large gap between train/val."
    elif val_rise > 0.005 or gap_final > 0.01:
        status = "mild"
        label  = "Mild Overfitting"
        msg    = f"Slight val loss increase after best epoch ({val_rise:.4f}). Monitor carefully."
    else:
        status = "good"
        label  = "Good Fit"
        msg    = f"Train and val loss converge well. Best epoch: {best_epoch}."

    return {
        "status":              status,
        "label":               label,
        "best_epoch":          best_epoch,
        "train_loss_best":     round(train_loss_best, 6),
        "val_loss_best":       round(val_loss_best, 6),
        "final_train_loss":    round(final_train, 6),
        "final_val_loss":      round(final_val, 6),
        "gap_best":            round(gap_best, 6),
        "gap_final":           round(gap_final, 6),
        "val_rise_after_best": round(val_rise, 6),
        "message":             msg,
    }


def load_tcn_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    try:
        raw_history = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(raw_history, list):
        return [
            {
                "epoch": int(item.get("epoch", index + 1)),
                "train_loss": float(item.get("train_loss", 0)),
                "val_loss": float(item.get("val_loss", item.get("valid_loss", 0))),
            }
            for index, item in enumerate(raw_history)
            if isinstance(item, dict)
        ]

    if isinstance(raw_history, dict):
        train_loss = raw_history.get("train_loss") or []
        val_loss = raw_history.get("val_loss") or raw_history.get("valid_loss") or []
        if isinstance(train_loss, list):
            return [
                {
                    "epoch": index + 1,
                    "train_loss": float(value or 0),
                    "val_loss": float((val_loss[index] if index < len(val_loss) else 0) or 0),
                }
                for index, value in enumerate(train_loss)
            ]

    return []


def build_histogram(values: list[float], bins: int = 36, min_value: float | None = None, max_value: float | None = None) -> list[dict]:
    clean = [float(value) for value in values if pd.notna(value)]
    if not clean:
        return []

    low = float(min_value if min_value is not None else min(clean))
    high = float(max_value if max_value is not None else max(clean))
    if high <= low:
        high = low + 1.0

    width = (high - low) / bins
    counts = [0 for _ in range(bins)]
    for value in clean:
        index = int((value - low) / width)
        index = max(0, min(bins - 1, index))
        counts[index] += 1

    return [
        {
            "bin": round(low + width * (index + 0.5), 6),
            "range_start": round(low + width * index, 6),
            "range_end": round(low + width * (index + 1), 6),
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


def build_distribution_overlay(y_true: list[float], y_pred: list[float], bins: int = 36) -> list[dict]:
    combined = [float(value) for value in [*y_true, *y_pred] if pd.notna(value)]
    if not combined:
        return []

    low = min(combined)
    high = max(combined)
    true_hist = build_histogram(y_true, bins=bins, min_value=low, max_value=high)
    pred_hist = build_histogram(y_pred, bins=bins, min_value=low, max_value=high)

    return [
        {
            "bin": item["bin"],
            "true": item["count"],
            "pred": pred_hist[index]["count"] if index < len(pred_hist) else 0,
        }
        for index, item in enumerate(true_hist)
    ]


def load_ag_evaluation_data(ag_model_dir: Path, max_points: int = 700) -> dict:
    candidates = sorted(
        [*ag_model_dir.glob("test_predictions_*.csv"), *ag_model_dir.glob("valid_predictions_*.csv")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {}

    pred_csv = candidates[0]
    df = pd.read_csv(pred_csv)
    true_col = "wait_time_ms" if "wait_time_ms" in df.columns else "true_wait_time"
    pred_col = "pred_wait_time_ms" if "pred_wait_time_ms" in df.columns else "pred_wait_time"
    if true_col not in df.columns or pred_col not in df.columns:
        return {}

    df = df.copy()
    df["wait_time_ms"] = pd.to_numeric(df[true_col], errors="coerce")
    df["pred_wait_time_ms"] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df.dropna(subset=["wait_time_ms", "pred_wait_time_ms"])
    if df.empty:
        return {}

    df["error"] = df["pred_wait_time_ms"] - df["wait_time_ms"]
    df["abs_error"] = df["error"].abs()

    if len(df) > max_points:
        sample_df = df.iloc[:: max(1, len(df) // max_points)].head(max_points)
    else:
        sample_df = df

    point_cols = ["wait_time_ms", "pred_wait_time_ms", "error", "abs_error"]
    if "wave_id" in sample_df.columns:
        point_cols.insert(0, "wave_id")

    points = []
    for row in sample_df[point_cols].to_dict(orient="records"):
        points.append({
            "wave_id": row.get("wave_id"),
            "true": round(float(row["wait_time_ms"]), 6),
            "pred": round(float(row["pred_wait_time_ms"]), 6),
            "error": round(float(row["error"]), 6),
            "abs_error": round(float(row["abs_error"]), 6),
        })

    worst_cols = point_cols
    worst_cases = []
    for row in df.sort_values("abs_error", ascending=False).head(8)[worst_cols].to_dict(orient="records"):
        worst_cases.append({
            "wave_id": row.get("wave_id"),
            "true": round(float(row["wait_time_ms"]), 6),
            "pred": round(float(row["pred_wait_time_ms"]), 6),
            "error": round(float(row["error"]), 6),
            "abs_error": round(float(row["abs_error"]), 6),
        })

    true_values = df["wait_time_ms"].tolist()
    pred_values = df["pred_wait_time_ms"].tolist()
    abs_errors = df["abs_error"].tolist()

    return {
        "source_csv": pred_csv.name,
        "total_points": int(len(df)),
        "sample_points": points,
        "abs_error_hist": build_histogram(abs_errors),
        "distribution": build_distribution_overlay(true_values, pred_values),
        "worst_cases": worst_cases,
    }


# ---------------------------------------------------------------------------
# Training Service
# ---------------------------------------------------------------------------

class TrainingService:
    """
    จัดการ pipeline การ train โมเดลแบบ async (background thread)

    Pipeline:
        1. extract_features      → handcrafted features จาก raw waveform
        2. make_wave_tensor      → แปลง waveform เป็น tensor สำหรับ TCN
        3. train_tcn_encoder     → train / reuse TCN encoder
        4. export_tcn_encoder    → export embedding จาก TCN
        5. merge_features        → รวม handcrafted + TCN embedding เป็น hybrid feature
        6. train_ag_1stage       → train AutoGluon tabular model
        7. analyze_regression    → วิเคราะห์ผล prediction
        8. analyze_feature_importance → วิเคราะห์ feature importance
    """

    @staticmethod
    def start_training(payload: dict) -> str:
        """
        เริ่ม training job ใหม่ใน background thread
        คืนค่า job_id สำหรับ polling status
        """
        job_id = str(uuid.uuid4())
        job_store.create(job_id, "train")
        job_queue.submit(job_id, "train", TrainingService._run_training, payload)
        return job_id

    @staticmethod
    def _run_training(job_id: str, payload: dict):
        """
        ฟังก์ชันหลักที่รันใน background thread
        อัปเดต progress ผ่าน job_store ในแต่ละขั้นตอน
        ถ้าล้มเหลวจะ catch error แล้ว set status = "failed"
        """
        mlflow_session = None
        cleanup_model_dirs: list[Path] = []
        try:
            job_store.update(job_id, status="running", progress=5, message="Starting clean split pipeline")

            split_dir_raw = (payload.get("split_dir") or "").strip()
            use_existing_split = bool(split_dir_raw)
            dataset_path = Path(payload["dataset_path"]) if not use_existing_split else None
            epochs        = int(payload.get("epochs", DEFAULT_TCN_EPOCHS))
            batch_size    = int(payload.get("batch_size", 64))
            lr            = float(payload.get("lr", 0.001))
            embedding_dim = int(payload.get("embedding_dim", 64))
            fast_ms       = float(payload.get("fast_ms", 0.1))
            target_col    = payload.get("target_col", "wait_time_ms")
            time_limit    = int(payload.get("time_limit", DEFAULT_AG_TIME_LIMIT))
            ag_presets    = str(payload.get("ag_presets", DEFAULT_AG_PRESETS))
            tcn_augment   = bool(payload.get("tcn_augment", DEFAULT_TCN_AUGMENT))
            tcn_noise_std = float(payload.get("tcn_noise_std", DEFAULT_TCN_NOISE_STD))
            tcn_scale_jitter = float(payload.get("tcn_scale_jitter", DEFAULT_TCN_SCALE_JITTER))
            tcn_time_shift = int(payload.get("tcn_time_shift", DEFAULT_TCN_TIME_SHIFT))
            fast_weight   = float(payload.get("fast_weight", DEFAULT_TCN_FAST_WEIGHT))
            early_stopping_patience = int(payload.get("early_stopping_patience", DEFAULT_TCN_EARLY_STOPPING_PATIENCE))

            requested_ag_name = payload.get("model_name", "")
            ag_model_name     = sanitize_model_name(requested_ag_name or f"model_{job_id[:8]}")

            train_new_tcn         = bool(payload.get("train_new_tcn", True))
            existing_tcn_name_raw = (payload.get("existing_tcn_name") or "").strip()

            if train_new_tcn:
                tcn_model_name = ag_model_name
            else:
                if not existing_tcn_name_raw:
                    raise Exception("existing_tcn_name is required when train_new_tcn=False")
                tcn_model_name = sanitize_model_name(existing_tcn_name_raw)

            model_dir    = TCN_DIR / tcn_model_name
            ag_model_dir = AUTOGLUON_DIR / ag_model_name

            if ag_model_dir.exists():
                raise Exception(f"AutoGluon model name already exists: {ag_model_name}")
            if train_new_tcn and model_dir.exists():
                raise Exception(f"TCN model name already exists: {tcn_model_name}")
            if not train_new_tcn and not model_dir.exists():
                raise Exception(f"TCN model not found: {tcn_model_name}")
            cleanup_model_dirs = [ag_model_dir]
            if train_new_tcn:
                cleanup_model_dirs.append(model_dir)

            mlflow_session = begin_training_run(
                run_name=ag_model_name,
                tags={
                    "job_id": job_id,
                    "pipeline": "tcn-autogluon",
                    "ag_model_name": ag_model_name,
                    "tcn_model_name": tcn_model_name,
                    "train_new_tcn": train_new_tcn,
                },
            )
            mlflow_info = dict(mlflow_session.info)
            mlflow_log_params(
                {
                    "job_id": job_id,
                    "dataset_path": str(dataset_path) if dataset_path is not None else "",
                    "split_dir_requested": split_dir_raw,
                    "use_existing_split": use_existing_split,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "embedding_dim": embedding_dim,
                    "fast_ms": fast_ms,
                    "target_col": target_col,
                    "time_limit": time_limit,
                    "ag_presets": ag_presets,
                    "tcn_augment": tcn_augment,
                    "tcn_noise_std": tcn_noise_std,
                    "tcn_scale_jitter": tcn_scale_jitter,
                    "tcn_time_shift": tcn_time_shift,
                    "fast_weight": fast_weight,
                    "early_stopping_patience": early_stopping_patience,
                    "ag_model_name": ag_model_name,
                    "tcn_model_name": tcn_model_name,
                    "train_new_tcn": train_new_tcn,
                }
            )

            split_dir = Path(split_dir_raw) if use_existing_split else DATA_DIR / "splits" / job_id
            if use_existing_split and not split_dir.is_absolute():
                split_dir = DATA_DIR.parent / split_dir
            processed_dir = DATA_DIR / "processed" / job_id
            if not use_existing_split:
                split_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)

            split_csv = {}
            for name in ("train", "valid", "test"):
                candidates = [
                    split_dir / f"{name}.csv",
                    split_dir / f"{name}_hybrid.csv",
                    split_dir / f"{name}_features.csv",
                ]
                split_csv[name] = next((c for c in candidates if c.exists()), candidates[0])
            feature_csv = {name: processed_dir / f"{name}_features.csv" for name in split_csv}
            tensor_npz  = {name: processed_dir / f"{name}_wave_tensor.npz" for name in split_csv}
            embed_csv   = {name: processed_dir / f"{name}_tcn_embed.csv" for name in split_csv}
            hybrid_csv  = {name: processed_dir / f"{name}_hybrid.csv" for name in split_csv}

            pipeline_manifest = {
                "job_id": job_id,
                "dataset_path": str(dataset_path) if dataset_path is not None else None,
                "split_dir": str(split_dir),
                "processed_dir": str(processed_dir),
                "split_csv": {name: str(path) for name, path in split_csv.items()},
                "feature_csv": {name: str(path) for name, path in feature_csv.items()},
                "tensor_npz": {name: str(path) for name, path in tensor_npz.items()},
                "embed_csv": {name: str(path) for name, path in embed_csv.items()},
                "hybrid_csv": {name: str(path) for name, path in hybrid_csv.items()},
            }
            pipeline_manifest_path = processed_dir / "pipeline_manifest.json"
            pipeline_manifest_path.write_text(json.dumps(pipeline_manifest, indent=2), encoding="utf-8")
            mlflow_log_params(
                {
                    "split_dir": str(split_dir),
                    "processed_dir": str(processed_dir),
                    "tcn_model_dir": str(model_dir),
                    "ag_model_dir": str(ag_model_dir),
                }
            )
            mlflow_log_artifacts([pipeline_manifest_path], artifact_path="pipeline")

            missing_splits = [name for name, path in split_csv.items() if not path.exists()]
            if use_existing_split:
                if missing_splits:
                    missing = ", ".join(f"{name}.csv" for name in missing_splits)
                    raise Exception(f"Existing split dir is missing: {missing}")
                job_store.update(job_id, progress=8, message="Using existing train/valid/test split...")
            else:
                if dataset_path is None:
                    raise Exception("dataset_path is required when split_dir is not provided")
                job_store.update(job_id, progress=8, message="Splitting dataset into train/valid/test...")
                run_cmd([
                    sys.executable, str(SCRIPT_PATHS["split_csv_train_test"]),
                    "--in", str(dataset_path),
                    "--outdir", str(split_dir),
                    "--train-frac", "0.8",
                    "--valid-frac", "0.1",
                    "--test-frac", "0.1",
                    "--seed", "42",
                    "--stratify-cols", "type",
                    "--label-col", target_col,
                    "--target-bins", "6",
                ])
            mlflow_log_artifacts([split_dir / "split_manifest.csv"], artifact_path="splits")

            for idx, split_name in enumerate(["train", "valid", "test"]):
                base_progress = 12 + idx * 12
                job_store.update(job_id, progress=base_progress, message=f"Extracting {split_name} features...")
                run_cmd([
                    sys.executable, str(SCRIPT_PATHS["extract_features"]),
                    "--mode", "train",
                    "--in", str(split_csv[split_name]),
                    "--out", str(feature_csv[split_name]),
                ])

                job_store.update(job_id, progress=base_progress + 6, message=f"Building {split_name} tensor...")
                run_cmd([
                    sys.executable, str(SCRIPT_PATHS["make_wave_tensor"]),
                    "--in", str(split_csv[split_name]),
                    "--out", str(tensor_npz[split_name]),
                    "--target-len", "1000",
                    "--label-col", target_col,
                ])

            if train_new_tcn:
                job_store.update(job_id, progress=52, message="Training TCN on train split only...")
                tcn_cmd = [
                    sys.executable, str(SCRIPT_PATHS["train_tcn_encoder"]),
                    "--waves", str(tensor_npz["train"]),
                    "--valid-waves", str(tensor_npz["valid"]),
                    "--out", str(model_dir),
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--lr", str(lr),
                    "--embedding-dim", str(embedding_dim),
                    "--seed", "42",
                    "--fast-ms", str(fast_ms),
                    "--fast-weight", str(fast_weight),
                    "--early-stopping-patience", str(early_stopping_patience),
                    "--log-target",
                ]
                if tcn_augment:
                    tcn_cmd.extend([
                        "--augment",
                        "--noise-std", str(tcn_noise_std),
                        "--scale-jitter", str(tcn_scale_jitter),
                        "--time-shift", str(tcn_time_shift),
                    ])
                run_cmd(tcn_cmd)
            else:
                job_store.update(job_id, progress=52, message=f"Using existing TCN: {tcn_model_name}")

            overfitting_history_path = model_dir / "train_history.json"
            overfitting = analyze_overfitting(overfitting_history_path)
            tcn_history = load_tcn_history(overfitting_history_path)
            mlflow_log_metrics(overfitting, prefix="tcn.overfitting")
            mlflow_log_artifacts(
                [
                    model_dir / "config.json",
                    model_dir / "train_history.json",
                    model_dir / "learning_curve.png",
                ],
                artifact_path="tcn",
            )
            if log_model_dirs_enabled():
                mlflow_log_artifacts([model_dir], artifact_path="models/tcn")

            for idx, split_name in enumerate(["train", "valid", "test"]):
                base_progress = 62 + idx * 7
                job_store.update(job_id, progress=base_progress, message=f"Exporting {split_name} TCN embeddings...")
                run_cmd([
                    sys.executable, str(SCRIPT_PATHS["export_tcn_encoder"]),
                    "--model", str(model_dir),
                    "--waves", str(tensor_npz[split_name]),
                    "--out", str(embed_csv[split_name]),
                ])

                job_store.update(job_id, progress=base_progress + 3, message=f"Merging {split_name} hybrid features...")
                run_cmd([
                    sys.executable, str(SCRIPT_PATHS["merge_features_and_embeddings"]),
                    "--features", str(feature_csv[split_name]),
                    "--embeddings", str(embed_csv[split_name]),
                    "--out", str(hybrid_csv[split_name]),
                ])

            job_store.update(job_id, progress=86, message="Training AutoGluon on train split...")
            run_cmd([
                sys.executable, str(SCRIPT_PATHS["train_ag_1stage"]),
                "--train", str(hybrid_csv["train"]),
                "--valid", str(hybrid_csv["valid"]),
                "--test", str(hybrid_csv["test"]),
                "--label", target_col,
                "--model-dir", str(ag_model_dir),
                "--model-name", ag_model_name,
                "--time-limit", str(time_limit),
                "--presets", ag_presets,
                "--weight-fast-ms", str(fast_ms),
                "--fast-weight", str(fast_weight),
                "--seed", "42",
                "--log-target",
            ])
            ag_metrics_path = ag_model_dir / "metrics.json"
            if ag_metrics_path.exists():
                ag_metrics = json.loads(ag_metrics_path.read_text(encoding="utf-8"))
                mlflow_log_metrics(ag_metrics, prefix="ag")
            mlflow_log_artifacts(
                [
                    ag_model_dir / "feature_cols.json",
                    ag_model_dir / "meta.json",
                    ag_model_dir / "metrics.json",
                    ag_model_dir / f"valid_predictions_{ag_model_name}.csv",
                    ag_model_dir / f"test_predictions_{ag_model_name}.csv",
                    ag_model_dir / f"feature_importance_{ag_model_name}.csv",
                ],
                artifact_path="autogluon",
            )
            if log_model_dirs_enabled():
                mlflow_log_artifacts([ag_model_dir], artifact_path="models/autogluon")

            # --- Step 7: Analyze results ---
            job_store.update(job_id, progress=92, message="Analyzing results...")

            analysis_dir = ANALYSIS_DIR / ag_model_name
            analysis_dir.mkdir(parents=True, exist_ok=True)

            val_pred_csv = ag_model_dir / f"test_predictions_{ag_model_name}.csv"
            fi_csv       = ag_model_dir / f"feature_importance_{ag_model_name}.csv"
            waveform_plot_dir = PLOTS_DIR / job_id
            waveform_plot_dir.mkdir(parents=True, exist_ok=True)
            analysis_manifest = []
            analysis_images = []
            total_waves = 0

            # วิเคราะห์ regression predictions (scatter, residual, histogram)
            if val_pred_csv.exists():
                run_cmd([
                    sys.executable, str(SCRIPT_PATHS["analyze_regression_preds"]),
                    "--in",      str(val_pred_csv),
                    "--outdir",  str(analysis_dir),
                    "--fast-ms", str(fast_ms),
                ])
                job_store.update(job_id, progress=94, message="Generating training waveform gallery...")
                run_cmd([
                    sys.executable, str(SCRIPT_PATHS["plot_pred_on_waveforms"]),
                    "--raw",    str(split_csv["test"]),
                    "--pred",   str(val_pred_csv),
                    "--outdir", str(waveform_plot_dir),
                    "--topk",   "30",
                    "--mode",   "first",
                ])
                analysis_manifest = build_analysis_manifest(val_pred_csv, waveform_plot_dir, job_id, "plots")
                analysis_images = list_pngs(waveform_plot_dir, job_id, "plots")
                try:
                    total_waves = int(pd.read_csv(val_pred_csv)["wave_id"].nunique())
                except Exception:
                    total_waves = len(analysis_manifest)

            # วิเคราะห์ feature importance แยก group
            if fi_csv.exists():
                fi_analysis_dir = ANALYSIS_DIR / f"feature_importance_{ag_model_name}"
                fi_analysis_dir.mkdir(parents=True, exist_ok=True)
                run_cmd([
                    sys.executable, str(SCRIPT_PATHS["features"]),
                    "--in",     str(fi_csv),
                    "--outdir", str(fi_analysis_dir),
                    "--topn",   "30",
                ])

            # อ่าน feature importance summary (ถ้ามี)
            feature_summary = None
            fi_analysis_dir = ANALYSIS_DIR / f"feature_importance_{ag_model_name}"
            full_csv        = fi_analysis_dir / "feature_importance_full_sorted.csv"
            if full_csv.exists():
                feature_summary = parse_feature_summary(fi_analysis_dir, topn=30)
                mlflow_log_metrics(feature_summary, prefix="feature_importance")

            # อ่าน metrics จาก summary.txt ที่ analyze_regression_preds.py สร้างไว้
            metrics      = {}
            summary_path = analysis_dir / "summary.txt"
            if summary_path.exists():
                for line in summary_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("MAE(all):"):
                        metrics["mae_all"]        = float(line.split(":")[1].strip())
                    elif line.startswith("RMSE:"):
                        metrics["rmse"]           = float(line.split(":")[1].strip())
                    elif "Fast precision" in line:
                        metrics["fast_precision"] = float(line.split(":")[1].strip())
                    elif "Fast recall" in line:
                        metrics["fast_recall"]    = float(line.split(":")[1].strip())
                    elif line.startswith("MAE(fast"):
                        metrics["mae_fast"]       = float(line.split(":")[1].strip())
                    elif line.startswith("MAE(slow"):
                        metrics["mae_slow"]       = float(line.split(":")[1].strip())
            mlflow_log_metrics(metrics, prefix="analysis")
            mlflow_log_artifacts([analysis_dir], artifact_path="analysis/regression")
            mlflow_log_artifacts([fi_analysis_dir], artifact_path="analysis/feature_importance")
            evaluation = load_ag_evaluation_data(ag_model_dir)

            # เขียน model_meta.json เก็บ metadata รวมถึง metrics และ plot paths
            meta = {
                "tcn_name": tcn_model_name,
                "tcn_path": str(model_dir),
                "ag_name":  ag_model_name,
                "ag_path":  str(ag_model_dir),
                "train_new_tcn": train_new_tcn,
                "split_dir": str(split_dir),
                "use_existing_split": use_existing_split,
                "mlflow": mlflow_info,
                "result": {
                    "metrics":             metrics,
                    "history":             tcn_history,
                    "evaluation":          evaluation,
                    "analysis_manifest":   analysis_manifest,
                    "analysis_images":     analysis_images,
                    "total_waves":         total_waves,
                    "_dataset_path":        str(split_csv["test"]),
                    "_pred_csv":            str(val_pred_csv),
                    "feature_summary":     feature_summary,
                    "overfitting_summary": overfitting,
                    "plots": {
                        "learning_curve":      f"/api/files/tcn/{tcn_model_name}/learning_curve.png",
                        "loss_curve":          f"/api/files/analysis/{ag_model_name}/abs_error_hist.png",
                        "actual_vs_pred":      f"/api/files/analysis/{ag_model_name}/scatter_true_vs_pred.png",
                        "error_histogram":     f"/api/files/analysis/{ag_model_name}/residual_plot.png",
                        "target_distribution": f"/api/files/analysis/{ag_model_name}/dist_true_vs_pred.png",
                        "feature_importance":  f"/api/files/analysis/feature_importance_{ag_model_name}/top_30_feature_importance.png",
                        "feature_group":       f"/api/files/analysis/feature_importance_{ag_model_name}/feature_group_importance_sum.png",
                        "feature_count":       f"/api/files/analysis/feature_importance_{ag_model_name}/feature_group_top_30_count.png",
                    },
                }
            }
            (ag_model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
            registry_package = {
                "model_type": "tcn_autogluon_hybrid",
                "ag_model_name": ag_model_name,
                "tcn_model_name": tcn_model_name,
                "ag_model_dir": str(ag_model_dir),
                "tcn_model_dir": str(model_dir),
                "split_dir": str(split_dir),
                "metrics": metrics,
                "feature_summary": feature_summary,
                "overfitting_summary": overfitting,
                "artifacts": {
                    "model_meta": "model_meta.json",
                    "ag_metrics": "metrics.json",
                    "feature_cols": "feature_cols.json",
                    "tcn_history": str(model_dir / "train_history.json"),
                    "tcn_checkpoint": str(model_dir / "tcn_encoder.pt"),
                },
            }
            registry_package_path = ag_model_dir / "registry_model_package.json"
            registry_package_path.write_text(json.dumps(registry_package, indent=2), encoding="utf-8")
            mlflow_log_artifacts([ag_model_dir / "model_meta.json"], artifact_path="autogluon")
            mlflow_log_artifacts(
                [
                    ag_model_dir / "model_meta.json",
                    registry_package_path,
                    ag_model_dir / "metrics.json",
                    ag_model_dir / "feature_cols.json",
                    model_dir / "config.json",
                    model_dir / "train_history.json",
                ],
                artifact_path="registered_model",
            )
            registry_info = mlflow_register_model_version(
                run_id=mlflow_info.get("run_id"),
                source_artifact_path="registered_model",
                tags={
                    "ag_model_name": ag_model_name,
                    "tcn_model_name": tcn_model_name,
                    "train_new_tcn": train_new_tcn,
                    "split_source": "existing" if use_existing_split else "generated",
                },
            )
            mlflow_info["registry"] = registry_info
            meta["mlflow"] = mlflow_info
            (ag_model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            mlflow_log_artifacts([ag_model_dir / "model_meta.json"], artifact_path="autogluon")
            mlflow_log_artifacts([ag_model_dir / "model_meta.json"], artifact_path="registered_model")
            mlflow_set_tags({"status": "completed"})
            if mlflow_session is not None:
                mlflow_session.end("FINISHED")

            job_store.update(
                job_id,
                status="completed",
                progress=100,
                message="Training completed",
                result={
                    "tcn_model":      tcn_model_name,
                    "ag_model":       ag_model_name,
                    "train_new_tcn":  train_new_tcn,
                    "tcn_model_dir":  str(model_dir),
                    "ag_model_dir":   str(ag_model_dir),
                    "split_dir":      str(split_dir),
                    "processed_dir":  str(processed_dir),
                    "use_existing_split": use_existing_split,
                    "mlflow":         mlflow_info,
                    "overfitting_summary": overfitting,
                    "metrics":        metrics,
                    "history":        tcn_history,
                    "evaluation":     evaluation,
                    "analysis_manifest": analysis_manifest,
                    "analysis_images": analysis_images,
                    "total_waves":    total_waves,
                    "_dataset_path":   str(split_csv["test"]),
                    "_pred_csv":       str(val_pred_csv),
                    "feature_summary": feature_summary,
                    "plots": {
                        "learning_curve":      f"/api/files/tcn/{tcn_model_name}/learning_curve.png",
                        "loss_curve":          f"/api/files/analysis/{ag_model_name}/abs_error_hist.png",
                        "actual_vs_pred":      f"/api/files/analysis/{ag_model_name}/scatter_true_vs_pred.png",
                        "error_histogram":     f"/api/files/analysis/{ag_model_name}/residual_plot.png",
                        "target_distribution": f"/api/files/analysis/{ag_model_name}/dist_true_vs_pred.png",
                        "feature_importance":  f"/api/files/analysis/feature_importance_{ag_model_name}/top_30_feature_importance.png",
                        "feature_group":       f"/api/files/analysis/feature_importance_{ag_model_name}/feature_group_importance_sum.png",
                        "feature_count":       f"/api/files/analysis/feature_importance_{ag_model_name}/feature_group_top_30_count.png",
                    },
                    "params": {
                        "epochs":        epochs,
                        "batch_size":    batch_size,
                        "lr":            lr,
                        "embedding_dim": embedding_dim,
                        "target_col":    target_col,
                        "fast_ms":       fast_ms,
                        "fast_weight":   fast_weight,
                        "time_limit":    time_limit,
                        "ag_presets":    ag_presets,
                        "tcn_augment":   tcn_augment,
                        "early_stopping_patience": early_stopping_patience,
                    },
                    "pipeline": {  # debug info
                        "used_existing_tcn": not train_new_tcn,
                        "tcn_source":        tcn_model_name,
                        "split_source":      "existing" if use_existing_split else "generated",
                    }
                },
            )

        except subprocess.CalledProcessError as e:
            err = e.stderr or e.stdout or str(e)
            removed_dirs = cleanup_failed_model_dirs(cleanup_model_dirs)
            if removed_dirs:
                err += "\n\n[cleanup] Removed failed model artifacts:\n" + "\n".join(removed_dirs)
            mlflow_set_tags({"status": "failed", "failure_type": "subprocess", "error": err[:500]})
            if mlflow_session is not None:
                mlflow_session.end("FAILED")
            # Script subprocess ล้มเหลว — เก็บ stderr/stdout เพื่อ debug
            job_store.update(
                job_id,
                status="failed",
                progress=100,
                message="Training failed",
                error=err,
            )
        except Exception as e:
            err = str(e)
            removed_dirs = cleanup_failed_model_dirs(cleanup_model_dirs)
            if removed_dirs:
                err += "\n\n[cleanup] Removed failed model artifacts:\n" + "\n".join(removed_dirs)
            mlflow_set_tags({"status": "failed", "failure_type": "exception", "error": err[:500]})
            if mlflow_session is not None:
                mlflow_session.end("FAILED")
            job_store.update(
                job_id,
                status="failed",
                progress=100,
                message="Training failed",
                error=err,
            )


# ---------------------------------------------------------------------------
# Prediction Service
# ---------------------------------------------------------------------------

class PredictionService:
    """
    จัดการ pipeline การ predict แบบ async (background thread)

    Pipeline:
        1. extract_features      → handcrafted features จาก raw waveform
        2. make_wave_tensor      → แปลง waveform เป็น tensor
        3. export_tcn_encoder    → export embedding จาก TCN ที่ train ไว้แล้ว
        4. merge_features        → รวม handcrafted + TCN embedding
        5. predict_ag_1stage     → run inference ด้วย AutoGluon
        6. plot_pred_on_waveforms → สร้างภาพ waveform + annotation ผล predict
    """

    @staticmethod
    def cleanup_old_prediction_jobs(max_age_hours: int = 24):
        """
        ตรวจสอบและทำความสะอาดเฉพาะ Prediction Jobs ที่มีอายุมากกว่า max_age_hours ชั่วโมง:
        1. ลบไฟล์ทำนายต้นฉบับใน uploads/
        2. ลบโฟลเดอร์ intermediate files ใน data/processed/prediction/<job_id>
        3. ลบโฟลเดอร์ผลลัพธ์ใน results/<job_id>
        4. ลบโฟลเดอร์กราฟพล็อตใน plots/<job_id>
        5. ลบประวัติงานใน SQLite (jobs table)
        """
        import time
        import shutil
        import sqlite3
        from .job_store import DB_PATH, job_store

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        if not RESULTS_DIR.exists():
            return

        for job_dir in RESULTS_DIR.iterdir():
            if not job_dir.is_dir():
                continue

            job_id = job_dir.name

            try:
                # ตรวจสอบอายุไดเรกทอรี (อ้างอิง mtime หรือ ctime)
                mtime = job_dir.stat().st_mtime
                if (current_time - mtime) > max_age_seconds:
                    # ดึงข้อมูลจากฐานข้อมูล SQLite เพื่อหาไฟล์ต้นทาง
                    job_data = job_store.as_dict(job_id)
                    if job_data and job_data.get("job_type") == "predict":
                        result = job_data.get("result")
                        if result:
                            # 1. ลบไฟล์อัปโหลดดิบใน uploads/
                            source_path = result.get("_source_dataset_path")
                            if source_path:
                                p = Path(source_path)
                                if p.exists() and p.is_file():
                                    p.unlink()

                            # 2. ลบไฟล์ intermediate ใน data/processed/prediction/<job_id>
                            processed_path = result.get("_dataset_path")
                            if processed_path:
                                p = Path(processed_path).parent
                                try:
                                    p.resolve().relative_to((DATA_DIR / "processed" / "prediction").resolve())
                                except ValueError:
                                    p = None
                                if p is not None and p.exists() and p.is_dir():
                                    shutil.rmtree(p, ignore_errors=True)

                        # 3. ลบโฟลเดอร์ผลลัพธ์ใน results/<job_id>
                        shutil.rmtree(job_dir, ignore_errors=True)

                        # 4. ลบโฟลเดอร์กราฟพล็อตใน plots/<job_id>
                        plot_dir = PLOTS_DIR / job_id
                        if plot_dir.exists() and plot_dir.is_dir():
                            shutil.rmtree(plot_dir, ignore_errors=True)

                        # 5. ลบประวัติงานใน SQLite
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            except Exception:
                # ป้องกันข้อผิดพลาดระหว่างการทำความสะอาดขัดขวางการทำงานหลัก
                pass

    @staticmethod
    def start_prediction(payload: dict) -> str:
        """
        เริ่ม prediction job ใหม่ใน background thread
        คืนค่า job_id สำหรับ polling status
        """
        job_id = str(uuid.uuid4())
        job_store.create(job_id, "predict")
        job_queue.submit(job_id, "predict", PredictionService._run_prediction, payload)
        return job_id

    @staticmethod
    def _run_prediction(job_id: str, payload: dict):
        """
        ฟังก์ชันหลักสำหรับ run prediction pipeline ใน background thread
        อ่าน TCN path จาก model_meta.json ถ้ามี (fallback เป็น TCN_DIR/<model_name>)
        """
        try:
            job_store.update(job_id, status="running", progress=5, message="Start prediction")

            dataset_path = Path(payload["dataset_path"])

            # เตรียม directory โครงสร้าง
            processed_dir  = DATA_DIR / "processed"
            prediction_dir = processed_dir / "prediction" / job_id
            processed_dir.mkdir(parents=True, exist_ok=True)
            prediction_dir.mkdir(parents=True, exist_ok=True)

            # Intermediate file paths
            signal_csv   = prediction_dir / "infer_signal_long.csv"
            features_csv = prediction_dir / "infer_features.csv"
            tensor_npz   = prediction_dir / "infer_wave_tensor.npz"
            embed_csv    = prediction_dir / "infer_tcn_embed.csv"
            hybrid_csv   = prediction_dir / "infer_hybrid.csv"

            # Output prediction CSV แยกตาม job_id เพื่อไม่ชนกัน
            result_dir = RESULTS_DIR / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            pred_csv = result_dir / "pred_1stage_hybrid.csv"

            # ตรวจสอบชื่อโมเดลและ resolve path
            model_name = sanitize_model_name(payload.get("model_name") or DEFAULT_MODEL_NAME)
            if not model_name:
                raise Exception("model_name is required")

            ag_model_dir = AUTOGLUON_DIR / model_name
            meta_file    = ag_model_dir / "model_meta.json"

            if meta_file.exists():
                # อ่าน TCN path จาก metadata ที่บันทึกตอน train
                meta      = json.loads(meta_file.read_text())
                model_dir = Path(meta["tcn_path"])
            else:
                model_dir = TCN_DIR / model_name  # fallback ถ้าไม่มี metadata

            # Directory สำหรับเก็บภาพ waveform
            plot_dir = PLOTS_DIR / job_id
            plot_dir.mkdir(parents=True, exist_ok=True)

            # --- Step 1: Normalize uploaded signal table ---
            job_store.update(job_id, progress=10, message="Preparing prediction dataset...")
            run_cmd([
                sys.executable, str(SCRIPT_PATHS["convert_signal_csv"]),
                "--in", str(dataset_path),
                "--out", str(signal_csv),
            ])

            # --- Step 2: Extract features ---
            job_store.update(job_id, progress=20, message="Extracting features...")
            run_cmd([
                sys.executable, str(SCRIPT_PATHS["extract_features"]),
                "--mode", "pred",
                "--in", str(signal_csv),
                "--out", str(features_csv),
            ])

            # --- Step 3: Build waveform tensor ---
            job_store.update(job_id, progress=32, message="Building tensor...")
            run_cmd([
                sys.executable, str(SCRIPT_PATHS["make_wave_tensor"]),
                "--in", str(signal_csv),
                "--out", str(tensor_npz),
                "--target-len", "1000",
            ])

            # --- Step 4: Export TCN embeddings ---
            job_store.update(job_id, progress=48, message="Extracting embeddings...")
            run_cmd([
                sys.executable, str(SCRIPT_PATHS["export_tcn_encoder"]),
                "--model", str(model_dir),
                "--waves", str(tensor_npz),
                "--out", str(embed_csv),
            ])

            # --- Step 5: Merge features + embeddings ---
            job_store.update(job_id, progress=64, message="Merging features...")
            run_cmd([
                sys.executable, str(SCRIPT_PATHS["merge_features_and_embeddings"]),
                "--features", str(features_csv),
                "--embeddings", str(embed_csv),
                "--out", str(hybrid_csv),
            ])

            # --- Step 6: Run AutoGluon inference ---
            job_store.update(job_id, progress=78, message="Running prediction...")
            run_cmd([
                sys.executable, str(SCRIPT_PATHS["predict_ag_1stage"]),
                "--model-path", str(ag_model_dir),
                "--in", str(hybrid_csv),
                "--out", str(pred_csv),
            ])

            # --- Step 6: Plot waveforms พร้อม annotation ผล predict ---
            job_store.update(job_id, progress=90, message="Generating waveform plots...")
            run_cmd([
                sys.executable, str(SCRIPT_PATHS["plot_pred_on_waveforms"]),
                "--raw",    str(signal_csv),
                "--pred",   str(pred_csv),
                "--outdir", str(plot_dir),
                "--topk",   "30",    # plot 30 waveforms แรก
                "--mode",   "first",
            ])

            # สร้าง preview และ manifest สำหรับ frontend
            preview_predictions = []
            analysis_manifest   = []

            if pred_csv.exists():
                pred_df             = pd.read_csv(pred_csv)
                preview_predictions = pred_df.head(20).fillna("").to_dict(orient="records")
                analysis_manifest   = build_analysis_manifest(pred_csv, plot_dir, job_id, "plots")

            job_store.update(
                job_id,
                status="completed",
                progress=100,
                message="Prediction done",
                result={
                    "_dataset_path":       str(signal_csv),
                    "_source_dataset_path": str(dataset_path),
                    "_pred_csv":           str(pred_csv),
                    "total_waves":         len(pred_df) if pred_csv.exists() else 0,
                    "predictions_csv":     f"/api/files/results/{job_id}/pred_1stage_hybrid.csv",
                    "preview_predictions": preview_predictions,    # 20 แถวแรกสำหรับ preview
                    "analysis_manifest":   analysis_manifest,      # waveform + pred สำหรับ gallery
                    "analysis_images":     list_pngs(plot_dir, job_id, "plots"),
                },
            )

        except subprocess.CalledProcessError as e:
            # Script subprocess ล้มเหลว — เก็บ stderr/stdout เพื่อ debug
            job_store.update(
                job_id,
                status="failed",
                progress=100,
                message="Prediction failed",
                error=e.stderr or e.stdout or str(e),
            )
        except Exception as e:
            job_store.update(
                job_id,
                status="failed",
                progress=100,
                message="Prediction failed",
                error=str(e),
            )
