# Backend Agent Guidance

Applies to `webapp/backend` in addition to the repository root `AGENTS.md`.

## Responsibilities

- Maintain Flask routes, access control, uploads, jobs, prediction, training, model discovery, and MLflow integration.
- Keep public prediction usable without exposing protected model operations.
- Preserve compatibility with machine-local model packages whose metadata may contain stale absolute paths.

## Access Boundary

Core public endpoints include:

- `GET /api/health`
- `POST /api/upload`
- `POST /api/predict`
- synchronous prediction endpoints
- `GET /api/jobs/<job_id>`
- `GET /api/models`, returning every ready prediction model

Protected operations include:

- `POST /api/train`
- model detail, leaderboard, audit, and deletion endpoints
- `GET /api/tcn-models`
- job-list and queue administration
- `GET /api/mlflow/model-registry`

Admin authentication accepts `X-Admin-Token` or `Authorization: Bearer <token>`. Keep checks server-side; frontend visibility is not authorization.

See `docs/decisions/002-public-admin-access.md`.

## Configuration

- `app/config.py` loads the root `.env` with `python-dotenv`.
- Shell values override `.env` values.
- Production requires `NEUROSETTLE_ADMIN_TOKEN`.
- Training defaults to disabled in production.
- `resolve_default_model_name()` uses the configured ready model or falls back to the newest ready model.
- Keep `.env.example`, README documentation, and config names synchronized.

## Model Discovery And Migration

- AutoGluon models live in `models/AutogluonModels/`.
- TCN models live in `models/TCNModels/`.
- A model is public-selectable only when discovery marks it ready.
- Imported `model_meta.json` files may reference another machine. Fall back to the matching local TCN directory when a recorded path does not exist.
- Do not invent public options for missing artifacts; selection must correspond to a runnable model package.

## Key Files

- `run.py`: Flask application and CORS setup
- `app/routes.py`: route definitions and authorization
- `app/config.py`: environment, paths, limits, and defaults
- `app/inference_service.py`: cached synchronous inference and prewarming
- `app/training_service.py`: model discovery and background pipeline orchestration
- `app/job_store.py`: job persistence
- `app/job_queue.py`: worker queue
- `app/mlflow_service.py`: tracking and registry integration

## Safety

- Sanitize model names and resolve paths inside approved base directories.
- Treat model deletion as an admin-only destructive operation.
- Never return secrets in API responses or logs.
- Avoid importing heavy ML libraries in modules that do not require them.
- Be aware that importing routes instantiates the inference service and may start default-model prewarming.

## Verification

```powershell
python -m py_compile app\config.py app\routes.py app\training_service.py
```

For route changes, use a Flask test client or a running local backend to verify both public and admin requests. Preserve the admin token when restarting a process that inherited it from a parent shell.
