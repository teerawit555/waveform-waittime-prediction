from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import (
    AUTOGLUON_DIR,
    DEFAULT_MODEL_NAME,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_LOG_MODEL_DIRS,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)


MLFLOW_DISABLED_VALUES = {"1", "true", "yes", "on"}
_LAST_IMPORT_ERROR: str | None = None


@dataclass
class MlflowSession:
    mlflow: Any | None
    run: Any | None
    info: dict[str, Any]

    @property
    def run_id(self) -> str | None:
        if self.run is None:
            return None
        return self.run.info.run_id

    def end(self, status: str) -> None:
        if self.mlflow is None or self.run is None:
            return
        try:
            self.mlflow.set_tag("run_status", status)
            self.mlflow.end_run(status=status)
        except Exception as exc:
            print(f"[WARN] MLflow end_run failed: {exc}")


def _is_disabled() -> bool:
    return os.getenv("MLFLOW_DISABLED", "").lower() in MLFLOW_DISABLED_VALUES


def _import_mlflow() -> Any | None:
    global _LAST_IMPORT_ERROR
    _LAST_IMPORT_ERROR = None
    if _is_disabled():
        return None
    try:
        import mlflow
    except Exception:
        _LAST_IMPORT_ERROR = traceback.format_exc()
        return None
    return mlflow


def get_mlflow_runtime_status() -> dict[str, Any]:
    mlflow = _import_mlflow()
    return {
        "enabled": mlflow is not None,
        "disabled_by_env": _is_disabled(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "mlflow_version": getattr(mlflow, "__version__", None) if mlflow is not None else None,
        "mlflow_import_error": _LAST_IMPORT_ERROR,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "reason": None if mlflow is not None else (
            "disabled" if _is_disabled() else "mlflow package is not installed in this backend Python environment"
        ),
    }


def _stringify_param(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


def _flatten_metrics(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_metrics(value, next_prefix))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def begin_training_run(run_name: str, tags: dict[str, Any]) -> MlflowSession:
    mlflow = _import_mlflow()
    if mlflow is None:
        reason = "disabled" if _is_disabled() else "mlflow package is not installed"
        return MlflowSession(
            mlflow=None,
            run=None,
            info={
                "enabled": False,
                "reason": reason,
                "tracking_uri": MLFLOW_TRACKING_URI,
                "experiment_name": MLFLOW_EXPERIMENT_NAME,
            },
        )

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        run = mlflow.start_run(
            run_name=run_name,
            tags={key: str(value) for key, value in tags.items() if value is not None},
        )
        return MlflowSession(
            mlflow=mlflow,
            run=run,
            info={
                "enabled": True,
                "run_id": run.info.run_id,
                "tracking_uri": MLFLOW_TRACKING_URI,
                "experiment_name": MLFLOW_EXPERIMENT_NAME,
                "log_model_dirs": MLFLOW_LOG_MODEL_DIRS,
            },
        )
    except Exception as exc:
        print(f"[WARN] MLflow setup failed: {exc}")
        return MlflowSession(
            mlflow=None,
            run=None,
            info={
                "enabled": False,
                "reason": str(exc),
                "tracking_uri": MLFLOW_TRACKING_URI,
                "experiment_name": MLFLOW_EXPERIMENT_NAME,
            },
        )


def log_params(params: dict[str, Any]) -> None:
    mlflow = _import_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    try:
        clean = {key: _stringify_param(value) for key, value in params.items()}
        mlflow.log_params(clean)
    except Exception as exc:
        print(f"[WARN] MLflow log_params failed: {exc}")


def log_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    mlflow = _import_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    try:
        flat = _flatten_metrics(metrics, prefix)
        if flat:
            mlflow.log_metrics(flat)
    except Exception as exc:
        print(f"[WARN] MLflow log_metrics failed: {exc}")


def log_artifacts(paths: Iterable[Path | str], artifact_path: str | None = None) -> None:
    mlflow = _import_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as exc:
            print(f"[WARN] MLflow log_artifact failed for {path}: {exc}")


def set_tags(tags: dict[str, Any]) -> None:
    mlflow = _import_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    try:
        mlflow.set_tags({key: str(value) for key, value in tags.items() if value is not None})
    except Exception as exc:
        print(f"[WARN] MLflow set_tags failed: {exc}")


def log_model_dirs_enabled() -> bool:
    return MLFLOW_LOG_MODEL_DIRS


def register_model_version(
    run_id: str | None,
    source_artifact_path: str,
    tags: dict[str, Any],
    registered_model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
) -> dict[str, Any]:
    mlflow = _import_mlflow()
    if mlflow is None or not run_id:
        return {
            "enabled": False,
            "reason": "mlflow package is not installed" if mlflow is None else "missing run_id",
            "registered_model_name": registered_model_name,
        }

    try:
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        try:
            client.create_registered_model(registered_model_name)
        except Exception as exc:
            if "RESOURCE_ALREADY_EXISTS" not in str(exc) and "already exists" not in str(exc).lower():
                raise

        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri.rstrip("/")
        source = f"{artifact_uri}/{source_artifact_path.strip('/')}"
        version = client.create_model_version(
            name=registered_model_name,
            source=source,
            run_id=run_id,
            tags={key: str(value) for key, value in tags.items() if value is not None},
        )
        alias = "candidate"
        try:
            client.set_registered_model_alias(registered_model_name, alias, version.version)
        except Exception as exc:
            print(f"[WARN] MLflow set alias failed: {exc}")

        return {
            "enabled": True,
            "registered_model_name": registered_model_name,
            "model_version": version.version,
            "model_version_status": version.status,
            "model_source": source,
            "alias": alias,
        }
    except Exception as exc:
        print(f"[WARN] MLflow register model version failed: {exc}")
        return {
            "enabled": False,
            "reason": str(exc),
            "registered_model_name": registered_model_name,
        }


def _format_mlflow_time(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        timestamp = int(value)
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _pick_metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[WARN] Failed to read JSON {path}: {exc}")
    return {}


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _local_metrics(ag_dir: Path, meta: dict[str, Any]) -> dict[str, float | None]:
    result = meta.get("result") if isinstance(meta.get("result"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}

    if not metrics:
        raw_metrics = _read_json(ag_dir / "metrics.json")
        test_metrics = raw_metrics.get("test") if isinstance(raw_metrics.get("test"), dict) else raw_metrics
        metrics = {
            "mae_all": test_metrics.get("mae"),
            "rmse": test_metrics.get("rmse"),
        }

    overfitting = result.get("overfitting_summary") if isinstance(result.get("overfitting_summary"), dict) else {}
    return {
        "mae_all": _pick_metric(metrics, "mae_all", "mae"),
        "rmse": _pick_metric(metrics, "rmse"),
        "fast_precision": _pick_metric(metrics, "fast_precision"),
        "fast_recall": _pick_metric(metrics, "fast_recall"),
        "mae_fast": _pick_metric(metrics, "mae_fast"),
        "mae_slow": _pick_metric(metrics, "mae_slow"),
        "best_epoch": _pick_metric(overfitting, "best_epoch"),
        "final_gap": _pick_metric(overfitting, "gap_final", "final_gap"),
    }


def _local_model_version_to_dict(ag_dir: Path, version: int) -> dict[str, Any] | None:
    if not ag_dir.is_dir():
        return None

    meta = _read_json(ag_dir / "model_meta.json")
    metrics = _local_metrics(ag_dir, meta)
    run_name = meta.get("ag_name") or ag_dir.name
    updated_at = _mtime_iso(ag_dir / "model_meta.json") or _mtime_iso(ag_dir)
    aliases = ["default"] if ag_dir.name == DEFAULT_MODEL_NAME else []

    return {
        "name": MLFLOW_REGISTERED_MODEL_NAME,
        "version": str(version),
        "status": "READY" if meta else "LOCAL",
        "current_stage": "Local",
        "aliases": aliases,
        "run_id": (meta.get("mlflow") or {}).get("run_id") if isinstance(meta.get("mlflow"), dict) else None,
        "run_name": run_name,
        "run_status": "FINISHED",
        "source": str(ag_dir),
        "creation_timestamp": None,
        "last_updated_timestamp": None,
        "created_at": updated_at,
        "updated_at": updated_at,
        "metrics": metrics,
        "all_metrics": metrics,
        "params": (meta.get("result") or {}).get("params", {}) if isinstance(meta.get("result"), dict) else {},
        "tags": {
            "registry_backend": "local",
            "ag_model_name": ag_dir.name,
            "tcn_model_name": str(meta.get("tcn_name") or ""),
        },
        "version_tags": {},
    }


def get_local_model_registry_summary(
    registered_model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
    reason: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    rows = []
    if AUTOGLUON_DIR.exists():
        model_dirs = sorted(
            [path for path in AUTOGLUON_DIR.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
        )
        for index, ag_dir in enumerate(model_dirs, start=1):
            row = _local_model_version_to_dict(ag_dir, index)
            if row is not None:
                rows.append(row)

    rows.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    rows = rows[:max_results]

    best_by_mae = min(
        (row for row in rows if row["metrics"].get("mae_all") is not None),
        key=lambda row: row["metrics"]["mae_all"],
        default=None,
    )
    candidate = rows[0] if rows else None
    production = next((row for row in rows if "default" in row.get("aliases", [])), None)

    return {
        "enabled": True,
        "registry_backend": "local",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "registered_model_name": registered_model_name,
        "versions": rows,
        "version_count": len(rows),
        "latest_versions_count": len(rows),
        "candidate_version": candidate["version"] if candidate else None,
        "production_version": production["version"] if production else None,
        "best_version": best_by_mae["version"] if best_by_mae else None,
        "best_mae_all": best_by_mae["metrics"]["mae_all"] if best_by_mae else None,
        "reason": reason or "MLflow registry is unavailable; showing local trained model artifacts.",
    }


def _model_version_to_dict(client: Any, version: Any) -> dict[str, Any]:
    run = None
    run_id = getattr(version, "run_id", None)
    if run_id:
        try:
            run = client.get_run(run_id)
        except Exception as exc:
            print(f"[WARN] MLflow get_run failed for {run_id}: {exc}")

    metrics = dict(getattr(getattr(run, "data", None), "metrics", {}) or {})
    params = dict(getattr(getattr(run, "data", None), "params", {}) or {})
    tags = dict(getattr(getattr(run, "data", None), "tags", {}) or {})
    version_tags = dict(getattr(version, "tags", {}) or {})

    selected_metrics = {
        "mae_all": _pick_metric(metrics, "analysis.mae_all", "mae_all", "ag.mae_all"),
        "rmse": _pick_metric(metrics, "analysis.rmse", "rmse", "ag.rmse"),
        "fast_precision": _pick_metric(metrics, "analysis.fast_precision", "fast_precision"),
        "fast_recall": _pick_metric(metrics, "analysis.fast_recall", "fast_recall"),
        "mae_fast": _pick_metric(metrics, "analysis.mae_fast", "mae_fast"),
        "mae_slow": _pick_metric(metrics, "analysis.mae_slow", "mae_slow"),
        "best_epoch": _pick_metric(metrics, "tcn.overfitting.best_epoch", "best_epoch"),
        "final_gap": _pick_metric(metrics, "tcn.overfitting.gap_final", "gap_final"),
    }

    return {
        "name": getattr(version, "name", None),
        "version": str(getattr(version, "version", "")),
        "status": getattr(version, "status", None),
        "current_stage": getattr(version, "current_stage", None),
        "aliases": list(getattr(version, "aliases", []) or []),
        "run_id": run_id,
        "run_name": tags.get("mlflow.runName"),
        "run_status": getattr(getattr(run, "info", None), "status", None),
        "source": getattr(version, "source", None),
        "creation_timestamp": getattr(version, "creation_timestamp", None),
        "last_updated_timestamp": getattr(version, "last_updated_timestamp", None),
        "created_at": _format_mlflow_time(getattr(version, "creation_timestamp", None)),
        "updated_at": _format_mlflow_time(getattr(version, "last_updated_timestamp", None)),
        "metrics": selected_metrics,
        "all_metrics": metrics,
        "params": params,
        "tags": tags,
        "version_tags": version_tags,
    }


def get_model_registry_summary(
    registered_model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
    max_results: int = 50,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": not _is_disabled(),
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "registered_model_name": registered_model_name,
        "versions": [],
    }
    mlflow = _import_mlflow()
    if mlflow is None:
        reason = "disabled" if _is_disabled() else "mlflow package is not installed"
        return get_local_model_registry_summary(registered_model_name, reason=reason, max_results=max_results)

    try:
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        try:
            registered_model = client.get_registered_model(registered_model_name)
            summary["description"] = getattr(registered_model, "description", None)
            summary["latest_versions_count"] = len(getattr(registered_model, "latest_versions", []) or [])
        except Exception as exc:
            return get_local_model_registry_summary(
                registered_model_name,
                reason=f"registered model not found: {exc}",
                max_results=max_results,
            )

        versions = client.search_model_versions(f"name='{registered_model_name}'")
        rows = [_model_version_to_dict(client, version) for version in versions]
        rows.sort(key=lambda item: int(item["version"]) if str(item["version"]).isdigit() else 0, reverse=True)
        rows = rows[:max_results]
        if not rows:
            return get_local_model_registry_summary(
                registered_model_name,
                reason="MLflow has no registered versions; showing local model artifacts.",
                max_results=max_results,
            )

        best_by_mae = min(
            (row for row in rows if row["metrics"].get("mae_all") is not None),
            key=lambda row: row["metrics"]["mae_all"],
            default=None,
        )
        candidate = next((row for row in rows if "candidate" in row.get("aliases", [])), None)
        production = next((row for row in rows if "production" in row.get("aliases", [])), None)

        summary.update(
            {
                "versions": rows,
                "version_count": len(rows),
                "candidate_version": candidate["version"] if candidate else None,
                "production_version": production["version"] if production else None,
                "best_version": best_by_mae["version"] if best_by_mae else None,
                "best_mae_all": best_by_mae["metrics"]["mae_all"] if best_by_mae else None,
            }
        )
    except Exception as exc:
        summary["enabled"] = False
        summary["reason"] = str(exc)
    return summary


def get_tracking_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": not _is_disabled(),
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "log_model_dirs": MLFLOW_LOG_MODEL_DIRS,
        "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
    }
    mlflow = _import_mlflow()
    if mlflow is None:
        summary["enabled"] = False
        summary["reason"] = "disabled" if _is_disabled() else "mlflow package is not installed"
        return summary

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return summary
        summary["experiment_id"] = experiment.experiment_id
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
            output_format="list",
        )
        if runs:
            latest = runs[0]
            summary["latest_run_id"] = latest.info.run_id
            summary["latest_run_name"] = latest.data.tags.get("mlflow.runName")
            summary["latest_run_status"] = latest.info.status
    except Exception as exc:
        summary["enabled"] = False
        summary["reason"] = str(exc)
    return summary
