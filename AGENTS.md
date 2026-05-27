# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

NEUROSETTLE is a machine-learning web application for waveform wait-time prediction.

The system has two main parts:

- `webapp/frontend`: React + Vite + TypeScript UI.
- `webapp/backend`: Flask API that runs upload, prediction, training, MLflow registry, and artifact endpoints.

The ML pipeline combines:

- waveform preprocessing
- handcrafted feature extraction
- TCN encoder embeddings
- AutoGluon regression
- optional MLflow tracking/model registry

Prediction is intended to be public-facing. Training, model details, TCN model listing, and MLflow registry views are admin-only.

## Important Product Rules

- Keep the in-app ADI logo (`webapp/frontend/public/adi_logo.png`) as-is unless the user explicitly asks to change it.
- NEUROSETTLE branding can be used for the browser title/favicon and project visuals.
- The default prediction model is NS 1.3, backed by model name `TCN_aug_weighted_v1`.
- Public users should primarily see and use Prediction.
- Admin users can unlock Training, Models, and Workflow with `NEUROSETTLE_ADMIN_TOKEN`.
- Do not expose model detail/registry endpoints publicly unless the user explicitly changes the product requirements.

## Frontend Notes

Key files:

- `webapp/frontend/src/App.tsx`: main application state, view switching, admin unlock state, landing page, training and prediction orchestration.
- `webapp/frontend/src/styles.css`: global visual system and page/component styling.
- `webapp/frontend/src/lib/api.ts`: frontend API helpers.
- `webapp/frontend/src/lib/modelDisplay.ts`: user-facing model labels such as NS 1.3.
- `webapp/frontend/src/components/PredictionWorkspace.tsx`: prediction UI.
- `webapp/frontend/src/components/ModelRegistrySection.tsx`: admin model registry UI.
- `webapp/frontend/src/components/WorkflowSection.tsx`: ML workflow explanation UI.

Frontend commands:

```powershell
cd webapp\frontend
npm.cmd run build
npm.cmd run dev
```

Design expectations:

- Keep the theme in the ADI/NEUROSETTLE direction: white, navy, sky blue, precise engineering feel.
- Prefer compact, premium product UI over marketing-heavy decoration.
- Landing page should lead users into prediction and API usage.
- Training should feel like an admin console, not a public user workflow.
- Avoid oversized headers/navbars that push the hero too far down.
- Use `lucide-react` icons when adding buttons or navigation items.

## Backend Notes

Key files:

- `webapp/backend/run.py`: Flask app factory and local server entrypoint.
- `webapp/backend/app/routes.py`: API routes and admin checks.
- `webapp/backend/app/config.py`: directory paths and environment-driven defaults.
- `webapp/backend/app/training_service.py`: training and prediction job orchestration.
- `webapp/backend/app/job_queue.py`: background job queue.
- `webapp/backend/app/job_store.py`: job status/result store.
- `webapp/backend/app/mlflow_service.py`: MLflow tracking/registry helpers.

Important environment variables:

```powershell
$env:NEUROSETTLE_ADMIN_TOKEN="change-me"
$env:DEFAULT_MODEL_NAME="TCN_aug_weighted_v1"
$env:MLFLOW_TRACKING_URI="file:///path/to/mlruns"
$env:MLFLOW_EXPERIMENT_NAME="adaptive-wait-time"
$env:JOB_WORKERS="1"
```

Backend verification:

```powershell
python -m py_compile webapp\backend\app\config.py webapp\backend\app\routes.py webapp\backend\app\training_service.py
```

Local backend run:

```powershell
cd webapp\backend
python run.py
```

## API Access Model

Public:

- `POST /api/upload`
- `POST /api/predict`
- `GET /api/jobs/<job_id>`
- `GET /api/models`, but public output should be limited to the default NS 1.3 model.

Admin-only:

- `POST /api/train`
- `GET /api/models/<model_name>`
- `GET /api/tcn-models`
- `GET /api/mlflow/model-registry`

Admin token is accepted through:

- `X-Admin-Token: <token>`
- `Authorization: Bearer <token>`

## Model Naming

Internal model names can be technical, but UI labels should be human-readable.

Current important mapping:

- `TCN_aug_weighted_v1` -> `NS 1.3`
- `test_ml_flow` -> `NS 1.2`
- `wave_model_v_overfit_check` -> `NS 1.1`
- `ag_1stage_hybrid_v1` -> `NS 1.0`
- `tcn_v1` -> `NS 1.0 Encoder`

Use `webapp/frontend/src/lib/modelDisplay.ts` for frontend labels and notes.

## Repository Hygiene

- The working tree may already contain user changes. Do not revert unrelated files.
- Keep edits scoped to the user request.
- Use `rg` or `rg --files` for searches.
- Use `apply_patch` for manual code edits.
- Do not commit unless the user explicitly asks.
- Avoid writing generated logs into the repo. If temporary logs are created for preview/debugging, remove them before finishing.

## Verification Checklist

For frontend changes:

```powershell
cd webapp\frontend
npm.cmd run build
```

Then preview the UI if the change affects layout, routing, or interaction.

For backend changes:

```powershell
python -m py_compile webapp\backend\app\config.py webapp\backend\app\routes.py webapp\backend\app\training_service.py
```

For admin behavior changes, verify at least:

- public `/api/models` returns only the default NS 1.3 model
- admin `/api/models` returns all ready models
- public registry/model-detail/TCN endpoints return `401`
- `/api/train` requires admin token

## Common Local Paths

- Models: `models/`
- TCN models: `models/TCNModels/`
- AutoGluon models: `models/AutogluonModels/`
- MLflow runs: `mlruns/`
- Uploaded files: `webapp/backend/app/uploads/`
- Prediction/training results: `webapp/backend/app/results/`
- Plot artifacts: `webapp/backend/app/plots/`
- Generated NEUROSETTLE logo assets: `webapp/frontend/public/neurosettle-logo.png`, `webapp/frontend/public/neurosettle-icon.png`

## Final Response Style

When reporting work to the user:

- Be concise.
- Mention changed files.
- Mention verification commands that passed.
- Mention any blocker clearly.
- The user often writes Thai, so Thai summaries are welcome.
