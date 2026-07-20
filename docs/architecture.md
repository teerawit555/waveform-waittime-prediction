# NEUROSETTLE Architecture

## System View

```text
Browser
  |
  v
React + Vite frontend
  |
  v
Flask API
  |-- uploads and preview
  |-- public prediction jobs
  |-- synchronous cached inference
  |-- admin training and model operations
  |
  +--> TCN encoder artifacts
  +--> AutoGluon predictor artifacts
  +--> job store, results, plots, and analysis
  +--> optional MLflow tracking and registry metadata
```

## Frontend

The frontend is a single React workspace with public Home, Prediction, and Workflow views. Admin unlock adds Training and Models. `src/App.tsx` owns view state and orchestration, while `src/lib/api.ts` defines the backend contract.

The public prediction selector uses `GET /api/models`, keeps internal model names as request values, and formats visible labels through `src/lib/modelDisplay.ts`.

## Backend

`webapp/backend/run.py` creates the Flask application and registers the API blueprint. `app/routes.py` defines the public and protected HTTP boundary. Long-running training and prediction work is represented as jobs through `job_queue.py` and `job_store.py`.

`inference_service.py` supports low-latency synchronous prediction by caching TCN and AutoGluon models in memory. Route imports can therefore trigger default-model prewarming.

## Model Package

A runnable prediction model consists of:

- an AutoGluon directory under `models/AutogluonModels/<model_name>`
- a TCN directory under `models/TCNModels/<model_name>` or a valid metadata reference
- metadata connecting the AutoGluon predictor and TCN encoder

Model directories are machine-local and ignored by Git. Migration code must tolerate absolute paths recorded on another machine and fall back to matching local artifacts.

The configured default model is used when it is ready. Otherwise, the backend selects the newest ready model. Public users can choose any ready model; protected endpoints expose deeper model operations.

## Prediction Flow

```text
upload dataset
  -> receive upload_id
  -> choose ready model
  -> create prediction job
  -> preprocess waveform
  -> extract handcrafted features
  -> generate TCN embedding
  -> run AutoGluon prediction
  -> write result and plot artifacts
  -> poll job status
```

Synchronous single and batch endpoints reuse cached model objects and skip the job queue.

## Training Flow

Training is admin-only and environment-controlled. A run may create data splits, train or reuse a TCN encoder, extract and merge features, train AutoGluon, generate analysis artifacts, and optionally log metadata to MLflow.

Production disables training by default. Enable it only on a protected server with appropriate compute and storage.

## Configuration

The backend loads the root `.env`; pre-existing shell variables win. Frontend configuration is separate and only public `VITE_*` values are bundled into the browser. See `.env.example`, `webapp/frontend/.env.example`, and `README.md`.

## Durable Decisions

- `decisions/001-model-version-naming.md`
- `decisions/002-public-admin-access.md`
