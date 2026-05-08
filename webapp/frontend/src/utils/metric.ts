export function formatMetric(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

export function formatSignedMetric(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }

  const num = Number(value);
  return `${num >= 0 ? "+" : ""}${num.toFixed(digits)}`;
}