from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory

from .config import ANALYSIS_DIR, PLOTS_DIR, RESULTS_DIR, UPLOAD_DIR, AUTOGLUON_DIR, TCN_DIR
from .data_service import build_preview, load_dataset, save_uploaded_file
from .job_store import job_store
from .training_service import PredictionService, TrainingService, run_cmd, SCRIPTS_DIR, list_available_models

api = Blueprint("api", __name__)


@api.route("/health", methods=["GET"])
def health_check():
    return jsonify({"ok": True})


@api.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]
    try:
        path = save_uploaded_file(file, UPLOAD_DIR)
        df = load_dataset(path)
        preview = build_preview(df)
        preview["dataset_path"] = str(path)
        return jsonify(preview)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@api.route("/train", methods=["POST"])
def start_train():
    payload = request.get_json(force=True)
    job_id = TrainingService.start_training(payload)
    return jsonify({"job_id": job_id})


@api.route("/predict", methods=["POST"])
def start_predict():
    payload = request.get_json(force=True)
    job_id = PredictionService.start_prediction(payload)
    return jsonify({"job_id": job_id})


@api.route("/jobs", methods=["GET"])
def list_jobs():
    return jsonify({"jobs": job_store.list_ids()})


@api.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    data = job_store.as_dict(job_id)
    if data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(data)

@api.route("/files/analysis/<model_name>/<path:filename>", methods=["GET"])
def serve_analysis(model_name: str, filename: str):
    base_dir    = ANALYSIS_DIR / model_name
    target_path = base_dir / filename
    if not target_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(base_dir, filename)

@api.route("/files/<category>/<job_id>/<path:filename>", methods=["GET"])
def serve_artifacts(category: str, job_id: str, filename: str):
    base_map = {
        "plots":    PLOTS_DIR / job_id,
        "results":  RESULTS_DIR / job_id,
    }
    if category not in base_map:
        return jsonify({"error": "Invalid artifact category"}), 404

    base_dir   = base_map[category]
    target_dir = base_dir / Path(filename).parent
    target_name = Path(filename).name

    if not (target_dir / target_name).exists():
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(target_dir, target_name)


@api.route("/plot-wave", methods=["POST"])
def plot_wave_on_demand():
    body = request.get_json(force=True)
    print(f"DEBUG /plot-wave called: {body}")
    wave_id = (body.get("wave_id") or "").strip()
    job_id  = (body.get("job_id")  or "").strip()

    if not wave_id or not job_id:
        return jsonify({"error": "wave_id and job_id are required"}), 400

    job = job_store.as_dict(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    result       = job.get("result") or {}
    dataset_path = result.get("_dataset_path")
    pred_csv     = result.get("_pred_csv")

    if not dataset_path or not pred_csv:
        return jsonify({"error": "Job is missing dataset or prediction paths"}), 400

    plot_dir = PLOTS_DIR / job_id
    plot_dir.mkdir(parents=True, exist_ok=True)

    out_png = plot_dir / f"{wave_id}.png"

    # ถ้ายังไม่มีรูป → plot เดี๋ยวนี้เลย
    if not out_png.exists():
        try:
            run_cmd([
                sys.executable, str(SCRIPTS_DIR / "plot_pred_on_waveforms.py"),
                "--raw",     str(dataset_path),
                "--pred",    str(pred_csv),
                "--outdir",  str(plot_dir),
                "--wave-id", wave_id,
            ])
        except Exception as e:
            return jsonify({"error": f"Plot failed: {str(e)}"}), 500

    if not out_png.exists():
        return jsonify({"error": f"{wave_id} not found in dataset"}), 404

    # ดึง pred / true จาก csv
    pred_val = None
    true_val = None
    try:
        df  = pd.read_csv(pred_csv)
        row = df[df["wave_id"].astype(str) == wave_id]
        if not row.empty:
            for col in ["pred", "prediction", "pred_wait_time_ms", "pred_wait_time"]:
                if col in df.columns:
                    pred_val = float(row.iloc[0][col])
                    break
            for col in ["wait_time_ms", "true"]:
                if col in df.columns:
                    true_val = float(row.iloc[0][col])
                    break
    except Exception:
        pass

    return jsonify({
        "wave_id": wave_id,
        "image":   f"/api/files/plots/{job_id}/{wave_id}.png",
        "pred":    pred_val,
        "true":    true_val,
    })

@api.route("/models", methods=["GET"])
def get_models():
    return jsonify({
        "models": list_available_models()
    })

@api.route("/models/<model_name>", methods=["GET"])
def get_model(model_name: str):
    ag_dir    = AUTOGLUON_DIR / model_name
    meta_file = ag_dir / "model_meta.json"
    if not meta_file.exists():
        return jsonify({"error": "Model not found"}), 404
    import json
    meta = json.loads(meta_file.read_text())
    return jsonify(meta)

@api.route("/tcn-models", methods=["GET"])
def get_tcn_models():
    from .training_service import list_available_tcn_models

    models = list_available_tcn_models()
    return {"result": True, "data": models}

@api.route("/files/tcn/<model_name>/<path:filename>", methods=["GET"])
def serve_tcn(model_name: str, filename: str):
    base_dir = TCN_DIR / model_name
    target   = base_dir / filename
    if not target.exists():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(str(base_dir), filename)

@api.route("/", methods=["GET"])
def api_root():
    return jsonify({"message": "API is running"})