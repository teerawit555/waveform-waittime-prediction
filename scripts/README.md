# Scripts

Pipeline scripts are grouped by responsibility:

- `data/`: split CSVs, convert raw signal CSVs, build waveform tensors
- `features/`: extract handcrafted features and merge them with TCN embeddings
- `tcn/`: train the TCN encoder and export embeddings
- `autogluon/`: train and run the AutoGluon tabular stage
- `analysis/`: evaluate predictions, feature importance, and waveform plots
- `pipelines/`: end-to-end orchestration helpers
- `generate/`: synthetic waveform data generation

The web backend resolves script paths through `SCRIPT_PATHS` in
`webapp/backend/app/training_service.py`.
