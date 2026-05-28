type ModelDisplayInfo = {
  label: string;
  note?: string;
};

export const DEFAULT_MODEL_NAME = 'TCN_aug_weighted_v1';

const modelDisplayMap: Record<string, ModelDisplayInfo> = {
  [DEFAULT_MODEL_NAME]: {
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
