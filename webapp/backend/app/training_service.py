from __future__ import annotations

import subprocess
import threading
import uuid
import sys
import pandas as pd
import json

from pathlib import Path
from .config import ANALYSIS_DIR, PLOTS_DIR, DATA_DIR, MODELS_DIR, TCN_DIR, AUTOGLUON_DIR, RESULTS_DIR, SCRIPTS_DIR
from .job_store import job_store


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str]):
    """รัน subprocess command แล้วคืนค่า stdout; raise CalledProcessError ถ้าล้มเหลว"""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


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
                for col in ["wait_time_ms", "true"]:
                    if col in pred_df.columns:
                        item["true"] = float(row[col])
                        break
                item["wave_id"] = row["wave_id"]

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

        thread = threading.Thread(
            target=TrainingService._run_training,
            args=(job_id, payload),
            daemon=True,
        )
        thread.start()
        return job_id

    @staticmethod
    def _run_training(job_id: str, payload: dict):
        """
        ฟังก์ชันหลักที่รันใน background thread
        อัปเดต progress ผ่าน job_store ในแต่ละขั้นตอน
        ถ้าล้มเหลวจะ catch error แล้ว set status = "failed"
        """
        try:
            job_store.update(job_id, status="running", progress=5, message="Start training pipeline")

            dataset_path = Path(payload["dataset_path"])

            # ดึง hyperparameter จาก payload (มี default value)
            epochs        = payload.get("epochs", 30)
            batch_size    = payload.get("batch_size", 64)
            lr            = payload.get("lr", 0.001)
            embedding_dim = payload.get("embedding_dim", 64)
            fast_ms       = payload.get("fast_ms", 0.1)
            target_col    = payload.get("target_col", "wait_time_ms")

            # เตรียม directory สำหรับ intermediate files
            processed_dir = DATA_DIR / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Intermediate file paths
            features_csv = processed_dir / "train_features.csv"
            tensor_npz   = processed_dir / "train_wave_tensor.npz"
            embed_csv    = processed_dir / "train_tcn_embed.csv"
            hybrid_csv   = processed_dir / "train_hybrid.csv"

            # ตั้งชื่อโมเดล AutoGluon และ TCN
            requested_ag_name = payload.get("model_name", "")
            ag_model_name     = sanitize_model_name(requested_ag_name or f"model_{job_id[:8]}")

            # ตัดสินใจว่าจะ train TCN ใหม่ หรือ reuse ของเดิม
            train_new_tcn         = bool(payload.get("train_new_tcn", True))
            existing_tcn_name_raw = payload.get("existing_tcn_name") or ""
            existing_tcn_name     = sanitize_model_name(existing_tcn_name_raw)

            if train_new_tcn:
                tcn_model_name = ag_model_name          # ใช้ชื่อเดียวกับ AG model
            else:
                if not existing_tcn_name:
                    raise Exception("existing_tcn_name is required when train_new_tcn=False")
                tcn_model_name = existing_tcn_name      # reuse TCN ที่มีอยู่แล้ว

            model_dir    = TCN_DIR / tcn_model_name
            ag_model_dir = AUTOGLUON_DIR / ag_model_name

            # Guard: ป้องกัน overwrite โมเดลที่มีอยู่แล้ว
            if ag_model_dir.exists():
                raise Exception(f"AutoGluon model name already exists: {ag_model_name}")
            if train_new_tcn and model_dir.exists():
                raise Exception(f"TCN model name already exists: {tcn_model_name}")
            if not train_new_tcn and not model_dir.exists():
                raise Exception(f"TCN model not found: {tcn_model_name}")

            # --- Step 1: Extract handcrafted features ---
            job_store.update(job_id, progress=10, message="Extracting features...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "extract_features.py"),
                "--mode", "train",
                "--in", str(dataset_path),
                "--out", str(features_csv),
            ])

            # --- Step 2: Build waveform tensor ---
            job_store.update(job_id, progress=20, message="Building wave tensor...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "make_wave_tensor.py"),
                "--in", str(dataset_path),
                "--out", str(tensor_npz),
                "--target-len", "1000",
                "--label-col", target_col,
            ])

            # --- Step 3: Train หรือ reuse TCN encoder ---
            if train_new_tcn:
                job_store.update(job_id, progress=40, message="Training TCN...")
                run_cmd([
                    sys.executable, str(SCRIPTS_DIR / "train_tcn_encoder.py"),
                    "--waves", str(tensor_npz),
                    "--out", str(model_dir),
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--lr", str(lr),
                    "--embedding-dim", str(embedding_dim),
                    "--log-target",
                ])
            else:
                job_store.update(job_id, progress=40, message=f"Using existing TCN: {tcn_model_name}")

            # วิเคราะห์ overfitting จาก training history
            overfitting_history_path = model_dir / "train_history.json" if train_new_tcn \
                else TCN_DIR / tcn_model_name / "train_history.json"
            overfitting = analyze_overfitting(overfitting_history_path)

            # --- Step 4: Export TCN embeddings ---
            job_store.update(job_id, progress=55, message="Exporting embeddings...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "export_tcn_encoder.py"),
                "--model", str(model_dir),
                "--waves", str(tensor_npz),
                "--out", str(embed_csv),
            ])

            # --- Step 5: Merge handcrafted features + TCN embeddings ---
            job_store.update(job_id, progress=70, message="Merging features...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "merge_features_and_embeddings.py"),
                "--features", str(features_csv),
                "--embeddings", str(embed_csv),
                "--out", str(hybrid_csv),
            ])

            # --- Step 6: Train AutoGluon tabular model ---
            job_store.update(job_id, progress=90, message="Training AutoGluon...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "train_ag_1stage.py"),
                "--data", str(hybrid_csv),
                "--label", target_col,
                "--model-dir", str(ag_model_dir),
                "--model-name", ag_model_name,
                "--time-limit", "300",
                "--log-target",
            ])

            # --- Step 7: Analyze results ---
            job_store.update(job_id, progress=92, message="Analyzing results...")

            analysis_dir = ANALYSIS_DIR / ag_model_name
            analysis_dir.mkdir(parents=True, exist_ok=True)

            val_pred_csv = ag_model_dir / f"test_predictions_{ag_model_name}.csv"
            fi_csv       = ag_model_dir / f"feature_importance_{ag_model_name}.csv"

            # วิเคราะห์ regression predictions (scatter, residual, histogram)
            if val_pred_csv.exists():
                run_cmd([
                    sys.executable, str(SCRIPTS_DIR / "analyze_regression_preds.py"),
                    "--in",      str(val_pred_csv),
                    "--outdir",  str(analysis_dir),
                    "--fast-ms", str(fast_ms),
                ])

            # วิเคราะห์ feature importance แยก group
            if fi_csv.exists():
                fi_analysis_dir = ANALYSIS_DIR / f"feature_importance_{ag_model_name}"
                fi_analysis_dir.mkdir(parents=True, exist_ok=True)
                run_cmd([
                    sys.executable, str(SCRIPTS_DIR / "features.py"),
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

            # เขียน model_meta.json เก็บ metadata รวมถึง metrics และ plot paths
            meta = {
                "tcn_name": tcn_model_name,
                "tcn_path": str(model_dir),
                "ag_name":  ag_model_name,
                "ag_path":  str(ag_model_dir),
                "train_new_tcn": train_new_tcn,
                "result": {
                    "metrics":             metrics,
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
                    "overfitting_summary": overfitting,
                    "metrics":        metrics,
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
                    },
                    "pipeline": {  # debug info
                        "used_existing_tcn": not train_new_tcn,
                        "tcn_source":        tcn_model_name,
                    }
                },
            )

        except subprocess.CalledProcessError as e:
            # Script subprocess ล้มเหลว — เก็บ stderr/stdout เพื่อ debug
            job_store.update(
                job_id,
                status="failed",
                progress=100,
                message="Training failed",
                error=e.stderr or e.stdout or str(e),
            )
        except Exception as e:
            job_store.update(
                job_id,
                status="failed",
                progress=100,
                message="Training failed",
                error=str(e),
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
    def start_prediction(payload: dict) -> str:
        """
        เริ่ม prediction job ใหม่ใน background thread
        คืนค่า job_id สำหรับ polling status
        """
        job_id = str(uuid.uuid4())
        job_store.create(job_id, "predict")

        thread = threading.Thread(
            target=PredictionService._run_prediction,
            args=(job_id, payload),
            daemon=True,
        )
        thread.start()
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
            prediction_dir = processed_dir / "prediction"
            processed_dir.mkdir(parents=True, exist_ok=True)
            prediction_dir.mkdir(parents=True, exist_ok=True)

            # Intermediate file paths
            features_csv = processed_dir / "infer_features.csv"
            tensor_npz   = processed_dir / "infer_wave_tensor.npz"
            embed_csv    = processed_dir / "infer_tcn_embed.csv"
            hybrid_csv   = processed_dir / "infer_hybrid.csv"

            # Output prediction CSV แยกตาม job_id เพื่อไม่ชนกัน
            result_dir = RESULTS_DIR / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            pred_csv = result_dir / "pred_1stage_hybrid.csv"

            # ตรวจสอบชื่อโมเดลและ resolve path
            model_name = sanitize_model_name(payload.get("model_name", ""))
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

            # --- Step 1: Extract features ---
            job_store.update(job_id, progress=10, message="Extracting features...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "extract_features.py"),
                "--mode", "pred",
                "--in", str(dataset_path),
                "--out", str(features_csv),
            ])

            # --- Step 2: Build waveform tensor ---
            job_store.update(job_id, progress=20, message="Building tensor...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "make_wave_tensor.py"),
                "--in", str(dataset_path),
                "--out", str(tensor_npz),
                "--target-len", "1000",
            ])

            # --- Step 3: Export TCN embeddings ---
            job_store.update(job_id, progress=40, message="Extracting embeddings...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "export_tcn_encoder.py"),
                "--model", str(model_dir),
                "--waves", str(tensor_npz),
                "--out", str(embed_csv),
            ])

            # --- Step 4: Merge features + embeddings ---
            job_store.update(job_id, progress=60, message="Merging features...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "merge_features_and_embeddings.py"),
                "--features", str(features_csv),
                "--embeddings", str(embed_csv),
                "--out", str(hybrid_csv),
            ])

            # --- Step 5: Run AutoGluon inference ---
            job_store.update(job_id, progress=75, message="Running prediction...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "predict_ag_1stage.py"),
                "--model-path", str(ag_model_dir),
                "--in", str(hybrid_csv),
                "--out", str(pred_csv),
            ])

            # --- Step 6: Plot waveforms พร้อม annotation ผล predict ---
            job_store.update(job_id, progress=90, message="Generating waveform plots...")
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "plot_pred_on_waveforms.py"),
                "--raw",    str(dataset_path),
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
                    "_dataset_path":       str(dataset_path),
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