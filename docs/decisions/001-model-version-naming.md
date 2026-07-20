# 001: Sequential Model Version Naming

- Status: Accepted
- Date: 2026-07-20

## Context

Model artifact names are technical identifiers used by the backend. Public users need stable, concise product labels, and future sequential models should not require a new hard-coded label for every release.

## Decision

The frontend keeps the backend model name as the API value and derives only the visible label.

1. An explicit entry in `modelDisplayMap` takes precedence for legacy or exceptional models.
2. Otherwise, a model name ending in `vN` maps to `NS 1.(N-1)`.
3. Sequential models are sorted by numeric `N`, not lexical string order.

Examples:

| Internal name | Public label |
|---|---|
| `wave_model_v1` | `NS 1.0` |
| `wave_model_v2` | `NS 1.1` |
| `wave_model_v3` | `NS 1.2` |
| `wave_model_v10` | `NS 1.9` |

## Consequences

- New sequential versions receive labels automatically.
- API payloads and artifact directories retain their technical names.
- Exceptional naming must be added explicitly to `modelDisplayMap`.
- Tests and UI review should cover numeric ordering when adding model-selection behavior.
