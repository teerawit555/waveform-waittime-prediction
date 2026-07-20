# ==============================================================================
# This file handles backend API routes and rate limiting.
# ==============================================================================
from __future__ import annotations

import sys
import json
import time
import shutil
from collections import defaultdict, deque
from hmac import compare_digest
import pandas as pd
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import safe_join

from .config import (
    ADMIN_TOKEN,
    ANALYSIS_DIR,
    AUTOGLUON_DIR,
    DATA_DIR,
    IS_PRODUCTION,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PLOTS_DIR,
    RATE_LIMIT_DEFAULT,
    RATE_LIMIT_PREDICT,
    RATE_LIMIT_TRAIN,
    RATE_LIMIT_UPLOAD,
    RATE_LIMIT_WINDOW_SECONDS,
    RESULTS_DIR,
    TCN_DIR,
    TRAINING_ENABLED,
    UPLOAD_DIR,
    resolve_default_model_name,
)
from .data_service import build_preview_from_file, save_uploaded_file
from .job_queue import job_queue
from .job_store import job_store
from .mlflow_service import get_mlflow_runtime_status, get_model_registry_summary, get_tracking_summary
from .training_service import PredictionService, TrainingService, build_analysis_manifest, run_cmd, SCRIPT_PATHS, list_available_models, load_ag_evaluation_data, load_tcn_history
from .inference_service import inference_service

api = Blueprint("api", __name__)
_rate_limit_hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def _format_upload_limit(max_mb) -> str:
    try:
        max_mb = int(max_mb)
    except (TypeError, ValueError):
        return str(max_mb)
    if max_mb >= 1024 and max_mb % 1024 == 0:
        return f"{max_mb // 1024} GB"
    return f"{max_mb} MB"


def _client_ip() -> str:
    return (request.access_route[0] if request.access_route else request.remote_addr) or "unknown"


def _rate_limit_for_path(path: str) -> int:
    if path.endswith("/upload"):
        return RATE_LIMIT_UPLOAD
    if path.endswith("/predict") or path.endswith("/plot-wave"):
        return RATE_LIMIT_PREDICT
    if path.endswith("/train"):
        return RATE_LIMIT_TRAIN
    return RATE_LIMIT_DEFAULT


@api.before_request
def apply_rate_limit():
    if request.method == "OPTIONS" or request.endpoint == "api.health_check":
        return None

    limit = _rate_limit_for_path(request.path)
    if limit <= 0:
        return None

    now = time.monotonic()
    key = (_client_ip(), request.path)
    hits = _rate_limit_hits[key]
    while hits and now - hits[0] > RATE_LIMIT_WINDOW_SECONDS:
        hits.popleft()

    if len(hits) >= limit:
        return jsonify({"error": "Too many requests. Please wait and try again."}), 429

    hits.append(now)
    return None


@api.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@api.errorhandler(RequestEntityTooLarge)
def upload_too_large(_exc):
    max_mb = current_app.config.get("MAX_UPLOAD_MB", "configured")
    return jsonify({"error": f"Uploaded file is too large. Maximum size is {_format_upload_limit(max_mb)}."}), 413


def _safe_path(base_dir: Path, filename: str) -> Path | None:
    joined = safe_join(str(base_dir), filename)
    if not joined:
        return None
    try:
        path = Path(joined).resolve()
        path.relative_to(base_dir.resolve())
        return path
    except ValueError:
        return None


def _json_payload() -> dict:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object request body")
    return payload


def _resolve_upload_id(upload_id: str | None) -> Path:
    if not upload_id:
        raise ValueError("upload_id is required")
    safe_name = "".join(c for c in upload_id if c.isalnum() or c in ("-", "_", "."))
    if not safe_name or safe_name != upload_id:
        raise ValueError("Invalid upload_id format")

    path = UPLOAD_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise ValueError("Uploaded dataset not found")
    if path.suffix.lower() not in (".csv", ".xlsx"):
        raise ValueError("Uploaded dataset must be a CSV or Excel (.xlsx) file")
    return path


def _resolve_existing_split_dir(split_dir: str | None) -> Path | None:
    if not split_dir:
        return None

    path = Path(split_dir)
    if not path.is_absolute():
        path = DATA_DIR.parent / path
    path = path.resolve()
    try:
        path.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise ValueError("split_dir must be inside the project data directory") from exc

    expected_sets = [
        ("train.csv", "train_hybrid.csv", "train_features.csv"),
        ("valid.csv", "valid_hybrid.csv", "valid_features.csv"),
        ("test.csv", "test_hybrid.csv", "test_features.csv"),
    ]
    missing = []
    for candidates in expected_sets:
        if not any((path / c).exists() for c in candidates):
            missing.append(candidates[0])
    if missing:
        raise ValueError(f"split_dir is missing required files: {', '.join(missing)}")
    return path


def _get_admin_token() -> str:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.headers.get("X-Admin-Token", "").strip()


def is_admin_request() -> bool:
    token = _get_admin_token()
    return bool(ADMIN_TOKEN and token and compare_digest(token, ADMIN_TOKEN))


def require_admin():
    if not ADMIN_TOKEN:
        return jsonify({"error": "Admin token is not configured. Set NEUROSETTLE_ADMIN_TOKEN on the backend."}), 503

    if is_admin_request():
        return None

    return jsonify({"error": "Admin access required"}), 401


@api.route("/health", methods=["GET"])
def health_check():
    return jsonify({"ok": True})


@api.route("/mlflow", methods=["GET"])
def get_mlflow_config():
    return jsonify(get_tracking_summary())


@api.route("/mlflow/runtime", methods=["GET"])
def get_mlflow_runtime():
    return jsonify(get_mlflow_runtime_status())


@api.route("/mlflow/model-registry", methods=["GET"])
def get_mlflow_model_registry():
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    model_name = request.args.get("name") or None
    return jsonify(get_model_registry_summary(registered_model_name=model_name) if model_name else get_model_registry_summary())


@api.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]
    try:
        path = save_uploaded_file(file, UPLOAD_DIR)
        preview = build_preview_from_file(path)
        preview["upload_id"] = path.name
        return jsonify(preview)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@api.route("/train", methods=["POST"])
def start_train():
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    if not TRAINING_ENABLED:
        message = "Training is disabled in this environment."
        if IS_PRODUCTION:
            message += " Set ENABLE_TRAINING=1 only on a protected admin server if training is required."
        return jsonify({"error": message}), 403

    try:
        payload = _json_payload()
        split_dir = _resolve_existing_split_dir(payload.get("split_dir"))
        if split_dir is not None:
            payload["split_dir"] = str(split_dir)
        else:
            dataset_path = _resolve_upload_id(payload.get("upload_id"))
            payload["dataset_path"] = str(dataset_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = TrainingService.start_training(payload)
    return jsonify({"job_id": job_id})


@api.route("/predict", methods=["POST"])
def start_predict():
    try:
        payload = _json_payload()
        dataset_path = _resolve_upload_id(payload.get("upload_id"))
        payload["dataset_path"] = str(dataset_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = PredictionService.start_prediction(payload)
    return jsonify({"job_id": job_id})


@api.route("/predict-sync", methods=["POST"])
def predict_sync():
    try:
        payload = _json_payload()
        result = inference_service.predict_single(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api.route("/predict-batch-sync", methods=["POST"])
def predict_batch_sync():
    try:
        payload = _json_payload()
        result = inference_service.predict_batch(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api.route("/jobs", methods=["GET"])
def list_jobs():
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    return jsonify({"jobs": job_store.list_ids()})


@api.route("/jobs/queue", methods=["GET"])
def get_job_queue():
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    return jsonify(job_queue.stats())


@api.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    data = job_store.as_dict(job_id)
    if data is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(data)

@api.route("/files/analysis/<model_name>/<path:filename>", methods=["GET"])
def serve_analysis(model_name: str, filename: str):
    safe_model_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name.strip()).strip("_")
    if not safe_model_name:
        return jsonify({"error": "Invalid model name"}), 400

    base_dir = ANALYSIS_DIR / safe_model_name
    target_path = _safe_path(base_dir, filename)
    if target_path is None or not target_path.exists():
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

    base_dir = base_map[category]
    target_path = _safe_path(base_dir, filename)

    if target_path is None or not target_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(base_dir, filename)


@api.route("/plot-wave", methods=["POST"])
def plot_wave_on_demand():
    try:
        body = _json_payload()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

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
                sys.executable, str(SCRIPT_PATHS["plot_pred_on_waveforms"]),
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
    ready_models = [model for model in list_available_models() if model.get("ready")]
    default_model = resolve_default_model_name()
    models = ready_models
    return jsonify({
        "default_model": default_model,
        "models": models,
        "training_enabled": TRAINING_ENABLED,
    })

@api.route("/models/<model_name>", methods=["GET"])
def get_model(model_name: str):
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    ag_dir    = AUTOGLUON_DIR / model_name
    meta_file = ag_dir / "model_meta.json"
    if not meta_file.exists():
        return jsonify({"error": "Model not found"}), 404
    meta = json.loads(meta_file.read_text())
    result = meta.get("result") or {}
    
    modified = False

    if "history" not in result:
        history_path = None
        tcn_path = meta.get("tcn_path")
        if tcn_path:
            history_path = Path(tcn_path) / "train_history.json"
        elif (meta.get("artifacts") or {}).get("tcn_history"):
            history_path = Path(meta["artifacts"]["tcn_history"])

        if history_path is not None:
            result["history"] = load_tcn_history(history_path)
            meta["result"] = result
            modified = True

    if "evaluation" not in result:
        ag_path = meta.get("ag_path")
        if ag_path:
            result["evaluation"] = load_ag_evaluation_data(Path(ag_path))
            meta["result"] = result
            modified = True

    # Add check for analysis_manifest to display audit plots in training section
    if "analysis_manifest" not in result or not result["analysis_manifest"]:
        safe_model_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name.strip()).strip("_")
        dataset_path = Path(result.get("_dataset_path") or "")
        pred_csv = Path(result.get("_pred_csv") or "")

        if not dataset_path.is_file():
            split_dir = Path(meta.get("split_dir") or "")
            for candidate_name in ("test.csv", "valid.csv", "train.csv"):
                candidate = split_dir / candidate_name
                if candidate.is_file():
                    dataset_path = candidate
                    break

        if not pred_csv.is_file():
            for candidate in (
                ag_dir / f"test_predictions_{safe_model_name}.csv",
                ag_dir / f"valid_predictions_{safe_model_name}.csv",
            ):
                if candidate.is_file():
                    pred_csv = candidate
                    break

        if dataset_path.is_file() and pred_csv.is_file():
            audit_id = f"audit_{safe_model_name}"
            plot_dir = PLOTS_DIR / audit_id
            plot_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Generate up to 30 waves (same default as training)
                if not list(plot_dir.glob("*.png")):
                    run_cmd([
                        sys.executable, str(SCRIPT_PATHS["plot_pred_on_waveforms"]),
                        "--raw", str(dataset_path),
                        "--pred", str(pred_csv),
                        "--outdir", str(plot_dir),
                        "--topk", "30",
                    ])
                
                manifest = build_analysis_manifest(pred_csv, plot_dir, audit_id, "plots")
                result["analysis_manifest"] = manifest
                result["total_waves"] = len(manifest)
                meta["result"] = result
                modified = True
            except Exception as e:
                print(f"[WARN] Failed to generate on-demand audit manifest: {e}")

    if modified:
        try:
            meta_file.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            print(f"[WARN] Failed to write updated model meta: {e}")

    return jsonify(meta)


@api.route("/models/<model_name>/leaderboard", methods=["GET"])
def get_model_leaderboard(model_name: str):
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    meta_file = AUTOGLUON_DIR / model_name / "model_meta.json"
    if not meta_file.exists():
        return jsonify({"error": "Model not found"}), 404

    meta = json.loads(meta_file.read_text())
    ag_path = meta.get("ag_path")
    if not ag_path or not Path(ag_path).exists():
        return jsonify({"error": "AutoGluon model path not found"}), 404

    try:
        from autogluon.tabular import TabularPredictor
        predictor = TabularPredictor.load(ag_path)
        lb = predictor.leaderboard()
        records = lb.to_dict(orient="records")
        return jsonify({
            "leaderboard": records,
            "best_model": predictor.model_best,
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load leaderboard: {e}"}), 500


def _safe_model_name(model_name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name.strip()).strip("_")


def _delete_dir_inside(base_dir: Path, target: Path, deleted_paths: list[str]) -> None:
    try:
        resolved_base = base_dir.resolve()
        resolved_target = target.resolve()
        resolved_target.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete path outside {base_dir}") from exc

    if resolved_target.exists():
        shutil.rmtree(resolved_target)
        deleted_paths.append(str(resolved_target))


def _delete_mlflow_versions_for_model(model_name: str) -> list[str]:
    deleted_versions: list[str] = []
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MLFLOW_REGISTERED_MODEL_NAME}'")
        for version in versions:
            version_tags = dict(getattr(version, "tags", {}) or {})
            run_name = None
            run_tags: dict[str, str] = {}
            run_id = getattr(version, "run_id", None)
            if run_id:
                try:
                    run = client.get_run(run_id)
                    run_tags = dict(getattr(getattr(run, "data", None), "tags", {}) or {})
                    run_name = run_tags.get("mlflow.runName")
                except Exception:
                    pass

            linked_name = (
                version_tags.get("ag_model_name")
                or run_tags.get("ag_model_name")
                or run_name
                or ""
            )
            if linked_name == model_name:
                client.delete_model_version(
                    name=MLFLOW_REGISTERED_MODEL_NAME,
                    version=str(getattr(version, "version")),
                )
                deleted_versions.append(str(getattr(version, "version")))
    except Exception as exc:
        print(f"[WARN] MLflow model version delete skipped/failed for {model_name}: {exc}")
    return deleted_versions


def _is_tcn_model_shared(tcn_path: Path, current_model_name: str) -> bool:
    if not AUTOGLUON_DIR.exists():
        return False

    try:
        target = tcn_path.resolve()
    except Exception:
        return False

    for ag_dir in AUTOGLUON_DIR.iterdir():
        if not ag_dir.is_dir() or ag_dir.name == current_model_name:
            continue
        meta_file = ag_dir / "model_meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            other_tcn = Path(meta.get("tcn_path") or "").resolve()
            if other_tcn == target:
                return True
        except Exception:
            continue
    return False


@api.route("/models/<model_name>", methods=["DELETE"])
def delete_model(model_name: str):
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    safe_model_name = _safe_model_name(model_name)
    if not safe_model_name or safe_model_name != model_name:
        return jsonify({"error": "Invalid model name"}), 400

    if safe_model_name == resolve_default_model_name():
        return jsonify({"error": "Default model cannot be deleted"}), 400

    ag_dir = AUTOGLUON_DIR / safe_model_name
    meta_file = ag_dir / "model_meta.json"
    if not ag_dir.exists():
        return jsonify({"error": "Model not found"}), 404

    meta = {}
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    deleted_paths: list[str] = []
    skipped_paths: list[str] = []

    tcn_path = Path(meta.get("tcn_path") or (TCN_DIR / safe_model_name))
    try:
        if tcn_path.exists() and not _is_tcn_model_shared(tcn_path, safe_model_name):
            _delete_dir_inside(TCN_DIR, tcn_path, deleted_paths)
        elif tcn_path.exists():
            skipped_paths.append(str(tcn_path))

        _delete_dir_inside(AUTOGLUON_DIR, ag_dir, deleted_paths)
        _delete_dir_inside(ANALYSIS_DIR, ANALYSIS_DIR / safe_model_name, deleted_paths)
        _delete_dir_inside(ANALYSIS_DIR, ANALYSIS_DIR / f"feature_importance_{safe_model_name}", deleted_paths)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    deleted_versions = _delete_mlflow_versions_for_model(safe_model_name)

    return jsonify({
        "deleted": True,
        "model_name": safe_model_name,
        "deleted_paths": deleted_paths,
        "skipped_paths": skipped_paths,
        "deleted_mlflow_versions": deleted_versions,
    })


@api.route("/models/<model_name>/waveform-audit", methods=["POST"])
def get_model_waveform_audit(model_name: str):
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    safe_model_name = _safe_model_name(model_name)
    if not safe_model_name or safe_model_name != model_name:
        return jsonify({"error": "Invalid model name"}), 400

    ag_dir = AUTOGLUON_DIR / safe_model_name
    meta_file = ag_dir / "model_meta.json"
    if not meta_file.exists():
        return jsonify({"error": "Model not found"}), 404
    try:
        payload = request.get_json(silent=True) or {}
        topk = int(payload.get("topk") or 4)
        topk = max(1, min(topk, 100))
    except Exception:
        topk = 4

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        result = meta.get("result") if isinstance(meta.get("result"), dict) else {}

        manifest = result.get("analysis_manifest") if isinstance(result.get("analysis_manifest"), list) else []
        if manifest:
            return jsonify({
                "model_name": safe_model_name,
                "dataset_path": result.get("_dataset_path") or "",
                "predictions_csv": result.get("_pred_csv") or "",
                "total_waves": result.get("total_waves") or len(manifest),
                "analysis_manifest": manifest[:topk],
            })

        dataset_path = Path(result.get("_dataset_path") or "")
        pred_csv = Path(result.get("_pred_csv") or "")

        if not dataset_path.is_file():
            split_dir = Path(meta.get("split_dir") or "")
            for candidate_name in ("test.csv", "valid.csv", "train.csv"):
                candidate = split_dir / candidate_name
                if candidate.is_file():
                    dataset_path = candidate
                    break

        if not pred_csv.is_file():
            for candidate in (
                ag_dir / f"test_predictions_{safe_model_name}.csv",
                ag_dir / f"valid_predictions_{safe_model_name}.csv",
            ):
                if candidate.is_file():
                    pred_csv = candidate
                    break

        if not dataset_path.is_file() or not pred_csv.is_file():
            return jsonify({"error": "Model audit source files are missing"}), 404

        audit_id = f"audit_{safe_model_name}"
        plot_dir = PLOTS_DIR / audit_id
        plot_dir.mkdir(parents=True, exist_ok=True)

        if not list(plot_dir.glob("*.png")):
            run_cmd([
                sys.executable, str(SCRIPT_PATHS["plot_pred_on_waveforms"]),
                "--raw", str(dataset_path),
                "--pred", str(pred_csv),
                "--outdir", str(plot_dir),
                "--topk", str(topk),
            ])

        manifest = build_analysis_manifest(pred_csv, plot_dir, audit_id, "plots")
        return jsonify({
            "model_name": safe_model_name,
            "dataset_path": str(dataset_path),
            "predictions_csv": str(pred_csv),
            "total_waves": len(manifest),
            "analysis_manifest": manifest[:topk],
        })
    except Exception as exc:
        return jsonify({"error": f"Failed to load model audit: {exc}"}), 500

@api.route("/tcn-models", methods=["GET"])
def get_tcn_models():
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    from .training_service import list_available_tcn_models

    models = list_available_tcn_models()
    return {"result": True, "data": models}

@api.route("/files/tcn/<model_name>/<path:filename>", methods=["GET"])
def serve_tcn(model_name: str, filename: str):
    auth_error = require_admin()
    if auth_error is not None:
        return auth_error

    safe_model_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name.strip()).strip("_")
    if not safe_model_name:
        return jsonify({"error": "Invalid model name"}), 400

    base_dir = TCN_DIR / safe_model_name
    target = _safe_path(base_dir, filename)
    if target is None or not target.exists():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(str(base_dir), filename)

@api.route("/", methods=["GET"])
def api_root():
    return jsonify({"message": "API is running"})
