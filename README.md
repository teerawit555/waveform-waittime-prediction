### Verify available Python versions
```powershell
py -0
```
### Create a virtual environment using Python 3.11
```powershell
py -V:3.11 -m venv venv311
```
### Enable script execution (Required for PowerShell activation)
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
### Activate the virtual environment
```powershell
.\venv311\Scripts\Activate.ps1
```
### Verify the current Python version
```powershell
python --version
```
### Install dependencies
```powershell
pip install -r requirements.txt
```

### Configure environment
Copy the checked-in examples before starting the backend and frontend:

```powershell
Copy-Item .env.example .env
Copy-Item webapp\frontend\.env.example webapp\frontend\.env.local
```

The backend automatically loads the root `.env`. Values already set in the shell take precedence. Set a long, random `NEUROSETTLE_ADMIN_TOKEN` before using the admin console. Production startup fails when this token is missing.

For production, use at least:

```dotenv
NEUROSETTLE_ENV=production
NEUROSETTLE_ADMIN_TOKEN=replace-with-a-long-random-token
ENABLE_TRAINING=0
FLASK_DEBUG=0
```

`VITE_*` values are bundled into frontend JavaScript. Never put the admin token or another secret in the frontend environment file. See `.env.example` for every supported backend setting.

### MLflow tracking
Training jobs log to MLflow automatically when `mlflow` is installed.

Defaults:
- tracking URI: `mlruns/` at the project root
- experiment: `adaptive-wait-time`
- full model directory artifact logging: off by default

Useful environment variables:
```powershell
$env:MLFLOW_TRACKING_URI="postgresql://user:password@localhost:5432/mlflow"
$env:MLFLOW_EXPERIMENT_NAME="adaptive-wait-time"
$env:MLFLOW_LOG_MODEL_DIRS="1"
$env:MLFLOW_DISABLED="1"
$env:JOB_WORKERS="1"
$env:FLASK_DEBUG="0"
$env:DEFAULT_TCN_AUGMENT="0"
$env:DEFAULT_TCN_EPOCHS="30"
$env:DEFAULT_TCN_FAST_WEIGHT="1.0"
$env:DEFAULT_TCN_EARLY_STOPPING_PATIENCE="5"
$env:DEFAULT_AG_PRESETS="medium_quality"
$env:DEFAULT_AG_TIME_LIMIT="300"
```

Training defaults can also be overridden from the web UI for each run.

Run the local UI from the project root:
```powershell
mlflow ui --backend-store-uri .\mlruns
```

### Script layout
The `scripts/` directory is grouped by pipeline responsibility:

- `scripts/data/`: CSV splitting, signal conversion, waveform tensor creation
- `scripts/features/`: handcrafted feature extraction and feature/embedding merge
- `scripts/tcn/`: TCN encoder training and embedding export
- `scripts/autogluon/`: AutoGluon training and prediction
- `scripts/analysis/`: regression analysis, feature-importance plots, waveform plots
- `scripts/pipelines/`: end-to-end orchestration scripts
- `scripts/generate/`: synthetic data generation
