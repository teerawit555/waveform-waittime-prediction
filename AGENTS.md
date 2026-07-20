# AGENTS.md

Guidance for coding agents working in the NEUROSETTLE repository.

## How Instructions Apply

- This root file applies to the whole repository.
- Before editing a directory, also read the nearest scoped `AGENTS.md`.
- Scoped files add domain-specific rules; they do not replace the product and security rules here.
- `README.md` is the operator guide. `docs/architecture.md` and `docs/decisions/` explain system structure and accepted decisions.
- Keep one source of truth. Link to existing guidance instead of copying it into another agent file.

Scoped guidance currently exists in:

- `webapp/frontend/AGENTS.md`
- `webapp/backend/AGENTS.md`
- `scripts/AGENTS.md`

## Project Overview

NEUROSETTLE predicts waveform settling wait time with a hybrid ML pipeline:

- waveform preprocessing and handcrafted features
- TCN encoder embeddings
- AutoGluon regression
- optional MLflow tracking and registry metadata

Main applications:

- `webapp/frontend`: React, Vite, and TypeScript
- `webapp/backend`: Flask API and background jobs
- `scripts`: data, feature, TCN, AutoGluon, and analysis CLIs

## Current Product Contract

- Home, Prediction, and Workflow are public-facing views.
- Training and Models are admin-only views.
- Public users may list and select every ready prediction model through `GET /api/models`.
- Model detail, leaderboard, deletion, audit, TCN listing, training, and MLflow registry operations remain admin-only.
- The backend default is the configured ready model; if it is unavailable, the backend resolves the newest ready model.
- Names ending in `vN`, such as `wave_model_v3`, display as `NS 1.(N-1)`, such as `NS 1.2`.
- Explicit legacy mappings in `webapp/frontend/src/lib/modelDisplay.ts` take precedence over the sequential rule.
- Keep `webapp/frontend/public/adi_logo.png` unchanged unless the user explicitly requests a brand change.

Accepted decisions are recorded in:

- `docs/decisions/001-model-version-naming.md`
- `docs/decisions/002-public-admin-access.md`

## Repository Map

- `webapp/frontend/src/App.tsx`: application state, navigation, admin unlock, and orchestration
- `webapp/frontend/src/lib/api.ts`: API client and `VITE_API_URL` handling
- `webapp/frontend/src/lib/modelDisplay.ts`: model labels and ordering
- `webapp/backend/run.py`: Flask application entrypoint
- `webapp/backend/app/routes.py`: routes and access checks
- `webapp/backend/app/config.py`: paths, `.env` loading, and runtime defaults
- `webapp/backend/app/inference_service.py`: cached synchronous inference
- `webapp/backend/app/training_service.py`: model discovery and training/prediction jobs
- `models/`: local TCN and AutoGluon artifacts; ignored by Git
- `mlruns/`: local MLflow state; ignored by Git

## Environment And Secrets

- The backend loads the root `.env` through `python-dotenv`.
- Existing shell variables take precedence over `.env` values.
- Use `.env.example` for backend configuration and `webapp/frontend/.env.example` for public frontend configuration.
- Never commit `.env`, credentials, tokens, model binaries, uploaded data, logs, or generated benchmark output.
- Never expose `NEUROSETTLE_ADMIN_TOKEN` through a `VITE_*` variable; Vite values are bundled into browser JavaScript.
- Production requires an admin token and disables training by default unless explicitly enabled.

## Common Commands

Backend setup and run from the repository root:

```powershell
python -m pip install -r requirements.txt
cd webapp\backend
python run.py
```

Frontend setup and run:

```powershell
cd webapp\frontend
npm install
npm run dev
```

Use `npm.cmd` only when PowerShell resolves `npm.ps1` and blocks it through execution policy.

## Verification

Backend syntax check:

```powershell
python -m py_compile webapp\backend\app\config.py webapp\backend\app\routes.py webapp\backend\app\training_service.py
```

Frontend build:

```powershell
cd webapp\frontend
npm run build
```

For access-control changes, verify at least:

- public `GET /api/models` returns all ready models
- public model-detail, registry, and TCN endpoints return `401`
- `POST /api/train` requires an admin token
- valid admin headers unlock protected endpoints

Scale verification to the blast radius. Do not run full model training merely to validate a narrow UI or route change.

## Engineering Rules

- Read the existing implementation before choosing an abstraction.
- Keep edits scoped and preserve unrelated user changes in dirty worktrees.
- Use `rg` or `rg --files` for searches and `apply_patch` for manual edits.
- Prefer structured parsers and existing project helpers over ad hoc text processing.
- Treat model artifacts and metadata as machine-local; recorded absolute paths may be stale after migration.
- Do not expose additional model-detail or admin endpoints without an explicit product decision.
- Do not commit or push unless the user explicitly asks.
- Remove temporary build and test artifacts created during the task.

## Multi-Agent Coordination

The parent agent owns integration and assigns non-overlapping file ownership. A useful default split is:

- Backend agent: routes, config, services, and backend tests
- Frontend agent: UI, model display, API client, and frontend tests
- Pipeline agent: scripts and artifact contracts
- Verification agent: read-only builds, API checks, and regression review

Subagents should not use committed Markdown files as a chat channel. Return a concise handoff to the parent agent with:

```markdown
Goal:
Files owned:
Assumptions:
Changes made:
Verification:
Risks or open questions:
```

Do not assign two agents to edit the same file concurrently. Store durable architectural reasoning in `docs/decisions/`, not in temporary chat transcripts or task logs.

## Final Response

- Be concise and lead with the outcome.
- Mention changed files and verification that passed.
- State blockers and unverified behavior clearly.
- Thai summaries are appropriate when the user writes in Thai.
