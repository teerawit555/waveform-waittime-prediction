// const API_BASE = 'http://localhost:5000/api';

export const API_BASE = 'http://localhost:5000/api';

export type UploadResponse = {
  dataset_path: string;
  shape: [number, number];
  columns: string[];
  preview: Record<string, unknown>[];
  numeric_columns: string[];
  wave_count: number; 
  sample_count: number;
  missing_top20: { column: string; missing: number }[];
};

export type JobResponse = {
  job_id: string;
  job_type: 'train' | 'predict';
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  result?: any;
  error?: string | null;
};

export type ModelItem = {
  name: string;
  tcn_path: string;
  ag_path: string;
  ready: boolean;
};

export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error((await res.json()).error ?? 'Upload failed');
  return res.json();
}

export async function startTrain(payload: Record<string, unknown>) {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? 'Training request failed');
  return data as { job_id: string };
}

export async function startPredict(payload: Record<string, unknown>) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? 'Prediction request failed');
  return data as { job_id: string };
}

export async function getJob(jobId: string): Promise<JobResponse> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? 'Failed to fetch job');
  return data;
}

export async function getModels(): Promise<{ models: ModelItem[] }> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || "Failed to fetch models");
  }
  return res.json();
}

export function toFileUrl(relativePath?: string | null) {
  if (!relativePath) return '';
  if (relativePath.startsWith('http')) return relativePath;
  return `http://localhost:5000${relativePath}`;
}