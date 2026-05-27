type ModelDisplayInfo = {
  label: string;
  note?: string;
};

const modelDisplayMap: Record<string, ModelDisplayInfo> = {
  ag_1stage_hybrid_v1: {
    label: 'NS 1.0',
    note: 'First clean-signal version without noise.',
  },
  tcn_v1: {
    label: 'NS 1.0 Encoder',
    note: 'TCN encoder used by the first clean-signal version.',
  },
  wave_model_v_overfit_check: {
    label: 'NS 1.1',
    note: 'Next version after NS 1.0 with overfit checking.',
  },
  test_ml_flow: {
    label: 'NS 1.2',
    note: 'Noise was added for model training and MLflow tracking is enabled.',
  },
  TCN_aug_weighted_v1: {
    label: 'NS 1.3',
    note: 'Noise and augmentation were used during training, with MLflow tracking.',
  },
};

const hiddenModelNames = new Set(['noise', 'noise_split_web_v1']);

export function normalizeModelName(name?: string | null) {
  return String(name ?? '').replace(/\.[^.]+$/, '');
}

export function shouldHideModel(name?: string | null) {
  return hiddenModelNames.has(normalizeModelName(name));
}

export function formatModelName(name?: string | null) {
  if (!name) return '';
  const normalized = normalizeModelName(name);
  const mapped = modelDisplayMap[normalized];
  if (mapped) return mapped.label;

  return normalized
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function getModelNote(name?: string | null) {
  if (!name) return '';
  return modelDisplayMap[normalizeModelName(name)]?.note ?? '';
}
