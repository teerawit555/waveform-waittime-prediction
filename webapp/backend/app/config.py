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

UPLOAD_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"
PLOTS_DIR = BACKEND_DIR / "plots"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)