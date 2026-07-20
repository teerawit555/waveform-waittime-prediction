# Frontend Agent Guidance

Applies to `webapp/frontend` in addition to the repository root `AGENTS.md`.

## Responsibilities

- Build the React and Vite user experience.
- Keep public and admin navigation consistent with backend access control.
- Preserve the ADI and NEUROSETTLE visual direction: white, navy, sky blue, compact engineering UI.
- Use `lucide-react` for interface icons when an appropriate icon exists.

## Product Behavior

- Home, Prediction, and Workflow are public.
- Training and Models appear only after admin unlock.
- Prediction model choices come from public `GET /api/models`.
- Honor the backend `default_model` when selecting the initial prediction model.
- Do not fetch protected model details merely to render the public prediction selector.

## Model Display

`src/lib/modelDisplay.ts` is the source of truth for user-facing labels.

- Explicit entries in `modelDisplayMap` win first.
- Otherwise, a name ending in `vN` maps to `NS 1.(N-1)`.
- Examples: `wave_model_v1` -> `NS 1.0`, `wave_model_v3` -> `NS 1.2`.
- Sort sequential versions numerically so `v10` does not appear before `v2`.
- Keep internal model names as API values; format only the visible label.

See `docs/decisions/001-model-version-naming.md`.

## Environment

- Vite reads `VITE_API_URL` from frontend environment files.
- `VITE_API_URL` must include the `/api` suffix when explicitly set.
- Never place admin tokens, database credentials, or other secrets in `VITE_*` variables.
- The local fallback is `http://<browser-host>:5000/api`; deployed same-origin builds use `/api`.

## Key Files

- `src/App.tsx`: view state, navigation, model loading, and orchestration
- `src/lib/api.ts`: API requests and types
- `src/lib/modelDisplay.ts`: labels, notes, and model sorting
- `src/components/PredictionWorkspace.tsx`: public prediction workflow
- `src/components/ModelRegistrySection.tsx`: protected model registry
- `src/styles.css`: global visual system

## UI Rules

- Prefer compact product surfaces over marketing-heavy sections.
- Keep controls stable across loading, empty, success, and error states.
- Do not nest cards inside cards.
- Keep text and controls within their containers at desktop and mobile widths.
- Preserve existing interaction patterns unless the request changes them.

## Commands And Verification

```powershell
npm install
npm run dev
npm run build
```

Use `npm.cmd` only as a Windows PowerShell execution-policy fallback.

For model-selection changes, verify:

- public users receive every ready model
- labels and numeric ordering match the version rule
- backend `default_model` remains selected
- prediction requests send the internal model name
