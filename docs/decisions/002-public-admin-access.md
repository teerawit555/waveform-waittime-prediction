# 002: Public Prediction And Protected Model Operations

- Status: Accepted
- Date: 2026-07-20

## Context

Prediction is the primary user workflow and must work without an admin session. Users need to select among ready prediction models. Training, model inspection, deletion, audits, TCN artifacts, queue administration, and MLflow registry operations expose privileged or destructive capabilities.

## Decision

- Public users may upload supported datasets, run prediction, poll their job by ID, view Workflow, and list every ready prediction model.
- `GET /api/models` returns the ready options needed by the prediction UI without requiring admin authentication.
- Detailed model and registry operations remain protected by the backend.
- Training and Models navigation remains hidden until frontend admin unlock, but backend authorization is always the security boundary.
- Admin requests use `X-Admin-Token` or `Authorization: Bearer <token>`.
- Production requires an admin token and keeps training disabled unless explicitly enabled.

## Consequences

- Adding a ready local model makes it available to public prediction after model discovery refresh.
- Public visibility of a model name does not grant access to protected metadata or operations.
- Frontend hiding is a usability feature, not authorization.
- Access-control changes require tests for both anonymous and authenticated requests.
