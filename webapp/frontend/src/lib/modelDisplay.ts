type ModelDisplayInfo = {
  label: string;
  note?: string;
};

export const DEFAULT_MODEL_NAME = 'TCN_aug_weighted_v1';

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
  [DEFAULT_MODEL_NAME]: {
    label: 'NS 1.3',
    note: 'Noise and augmentation were used during training, with MLflow tracking.',
  },
};

const hiddenModelNames = new Set(['noise', 'noise_split_web_v1']);
const sequentialModelPattern = /(?:^|[_\s-])v(\d+)$/i;
const modelLabelCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

export function normalizeModelName(name?: string | null) {
  return String(name ?? '').replace(/\.[^.]+$/, '');
}

export function shouldHideModel(name?: string | null) {
  return hiddenModelNames.has(normalizeModelName(name));
}

export function getModelVersion(name?: string | null) {
  const match = normalizeModelName(name).match(sequentialModelPattern);
  if (!match) return null;

  const version = Number.parseInt(match[1], 10);
  return Number.isSafeInteger(version) && version > 0 ? version : null;
}

export function formatModelName(name?: string | null) {
  if (!name) return '';
  const normalized = normalizeModelName(name);
  const mapped = modelDisplayMap[normalized];
  if (mapped) return mapped.label;

  const version = getModelVersion(normalized);
  if (version !== null) return `NS 1.${version - 1}`;

  return normalized
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function getModelNote(name?: string | null) {
  if (!name) return '';
  return modelDisplayMap[normalizeModelName(name)]?.note ?? '';
}

export function compareModelNames(a?: string | null, b?: string | null) {
  const aVersion = getModelVersion(a);
  const bVersion = getModelVersion(b);

  if (aVersion !== null && bVersion !== null && aVersion !== bVersion) {
    return aVersion - bVersion;
  }

  return modelLabelCollator.compare(formatModelName(a), formatModelName(b));
}
