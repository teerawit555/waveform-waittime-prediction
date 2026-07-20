# ML Pipeline Agent Guidance

Applies to `scripts` in addition to the repository root `AGENTS.md`.

## Pipeline Areas

- `data/`: CSV conversion, splitting, waveform tensors, and normalization
- `features/`: handcrafted feature extraction and embedding merges
- `tcn/`: TCN training and encoder export
- `autogluon/`: regressor training and prediction
- `analysis/`: metrics, feature importance, residuals, and waveform plots
- `pipelines/`: composed prediction workflows

## Contracts

- Preserve existing CLI argument names unless all callers are updated together.
- Use structured CSV/JSON parsing and explicit column validation.
- Keep model metadata portable: prefer project-relative identifiers and include a local-path fallback.
- Write generated artifacts only to the established data, analysis, model, result, or plot directories.
- Do not commit model binaries, generated datasets, plots, MLflow runs, or benchmark output.

## Training Safety

- Do not launch full training or broad benchmark jobs unless the user requests them.
- Use small fixtures or existing artifacts for narrow verification.
- Record seeds and relevant hyperparameters for reproducible runs.
- Keep TCN and AutoGluon model names consistent in metadata.

## GPU Compatibility

Do not assume CUDA support from GPU detection alone. Verify the installed PyTorch build, CUDA runtime, supported architectures, and a real tensor operation:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list()); print((torch.ones(1, device='cuda') * 2).item() if torch.cuda.is_available() else 'CPU')"
```

The repository requirements and the installed GPU architecture must agree. When changing the standard PyTorch wheel, update `requirements.txt` and rerun the operation test.

## Verification

- Run `python -m py_compile` for changed Python scripts.
- Exercise `--help` when modifying CLI parsing.
- Validate output columns and metadata against downstream consumers.
- State clearly when full training was not run.
