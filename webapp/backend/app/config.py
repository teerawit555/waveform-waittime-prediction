import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent


MODELS_DIR    = PROJECT_ROOT / "models"
TCN_DIR       = MODELS_DIR / "TCNModels"
AUTOGLUON_DIR = MODELS_DIR / "AutogluonModels"

DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" 
SCRIPT_DATA_DIR = SCRIPTS_DIR / "data"
SCRIPT_FEATURES_DIR = SCRIPTS_DIR / "features"
SCRIPT_TCN_DIR = SCRIPTS_DIR / "tcn"
SCRIPT_AUTOGLUON_DIR = SCRIPTS_DIR / "autogluon"
SCRIPT_ANALYSIS_DIR = SCRIPTS_DIR / "analysis"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri())
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "adaptive-wait-time")
MLFLOW_LOG_MODEL_DIRS = os.getenv("MLFLOW_LOG_MODEL_DIRS", "0").lower() in {"1", "true", "yes", "on"}
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "adaptive_wait_time_hybrid")
ADMIN_TOKEN = os.getenv("NEUROSETTLE_ADMIN_TOKEN") or os.getenv("ADMIN_TOKEN")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "TCN_aug_weighted_v1")
JOB_WORKERS = int(os.getenv("JOB_WORKERS", "1"))

NEUROSETTLE_ENV = os.getenv("NEUROSETTLE_ENV") or os.getenv("FLASK_ENV") or "development"
IS_PRODUCTION = NEUROSETTLE_ENV.lower() in {"prod", "production"}
TRAINING_ENABLED = os.getenv("ENABLE_TRAINING", "0" if IS_PRODUCTION else "1").lower() in {"1", "true", "yes", "on"}
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "NEUROSETTLE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if origin.strip()
]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_DEFAULT = int(os.getenv("RATE_LIMIT_DEFAULT", "120"))
RATE_LIMIT_UPLOAD = int(os.getenv("RATE_LIMIT_UPLOAD", "20"))
RATE_LIMIT_PREDICT = int(os.getenv("RATE_LIMIT_PREDICT", "30"))
RATE_LIMIT_TRAIN = int(os.getenv("RATE_LIMIT_TRAIN", "5"))

DEFAULT_TCN_AUGMENT = os.getenv("DEFAULT_TCN_AUGMENT", "0").lower() in {"1", "true", "yes", "on"}
DEFAULT_TCN_EPOCHS = int(os.getenv("DEFAULT_TCN_EPOCHS", "30"))
DEFAULT_TCN_NOISE_STD = float(os.getenv("DEFAULT_TCN_NOISE_STD", "0.015"))
DEFAULT_TCN_SCALE_JITTER = float(os.getenv("DEFAULT_TCN_SCALE_JITTER", "0.04"))
DEFAULT_TCN_TIME_SHIFT = int(os.getenv("DEFAULT_TCN_TIME_SHIFT", "8"))
DEFAULT_TCN_FAST_WEIGHT = float(os.getenv("DEFAULT_TCN_FAST_WEIGHT", "1.0"))
DEFAULT_TCN_EARLY_STOPPING_PATIENCE = int(os.getenv("DEFAULT_TCN_EARLY_STOPPING_PATIENCE", "5"))
DEFAULT_AG_PRESETS = os.getenv("DEFAULT_AG_PRESETS", "medium_quality")
DEFAULT_AG_TIME_LIMIT = int(os.getenv("DEFAULT_AG_TIME_LIMIT", "300"))

UPLOAD_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"
PLOTS_DIR = BACKEND_DIR / "plots"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
