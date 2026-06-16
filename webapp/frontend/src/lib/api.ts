// const API_BASE = 'http://localhost:5000/api';

const isLocal =
  window.location.hostname === 'localhost' ||
  window.location.hostname === '127.0.0.1' ||
  window.location.hostname.startsWith('10.') ||
  window.location.hostname.startsWith('192.168.') ||
  window.location.hostname.startsWith('172.') ||
  window.location.hostname.startsWith('127.') ||
  window.location.port === '5173' ||
  window.location.port === '4173';

const defaultApiBase = isLocal ? `http://${window.location.hostname}:5000/api` : `${window.location.origin}/api`;
export const API_BASE = ((import.meta as any).env?.VITE_API_URL as string) || defaultApiBase;

export type UploadResponse = {
  upload_id: string;
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

export type MlflowInfo = {
  enabled: boolean;
  tracking_uri: string;
  experiment_name: string;
  registered_model_name?: string;
  log_model_dirs?: boolean;
  run_id?: string;
  latest_run_id?: string;
  latest_run_name?: string;
  latest_run_status?: string;
  registry?: {
    enabled: boolean;
    registered_model_name?: string;
    model_version?: string;
    model_version_status?: string;
    model_source?: string;
    alias?: string;
    reason?: string;
  };
  reason?: string;
};

export type MlflowModelVersion = {
  name: string;
  version: string;
  status?: string | null;
  current_stage?: string | null;
  aliases: string[];
  run_id?: string | null;
  run_name?: string | null;
  run_status?: string | null;
  source?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  metrics: {
    mae_all?: number | null;
    rmse?: number | null;
    fast_precision?: number | null;
    fast_recall?: number | null;
    mae_fast?: number | null;
    mae_slow?: number | null;
    best_epoch?: number | null;
    final_gap?: number | null;
  };
  all_metrics?: Record<string, number>;
  params?: Record<string, string>;
  tags?: Record<string, string>;
  version_tags?: Record<string, string>;
};

export type MlflowModelRegistry = {
  enabled: boolean;
  registry_backend?: 'mlflow' | 'local';
  tracking_uri: string;
  experiment_name: string;
  registered_model_name: string;
  description?: string | null;
  version_count?: number;
  latest_versions_count?: number;
  candidate_version?: string | null;
  production_version?: string | null;
  best_version?: string | null;
  best_mae_all?: number | null;
  versions: MlflowModelVersion[];
  reason?: string;
};

export type ModelAuditResponse = {
  model_name: string;
  dataset_path: string;
  predictions_csv: string;
  total_waves: number;
  analysis_manifest: {
    wave_id: string | number;
    image: string;
    pred?: number | null;
    true?: number | null;
    error?: number | null;
    abs_error?: number | null;
  }[];
};

function adminHeaders(adminToken?: string) {
  return adminToken ? { 'X-Admin-Token': adminToken } : undefined;
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error((await res.json()).error ?? 'Upload failed');
  return res.json();
}

export async function startTrain(payload: Record<string, unknown>, adminToken?: string) {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(adminToken ? { 'X-Admin-Token': adminToken } : {}),
    },
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

export async function getModels(adminToken?: string): Promise<{ models: ModelItem[]; default_model?: string; training_enabled?: boolean }> {
  const res = await fetch(`${API_BASE}/models`, {
    headers: adminHeaders(adminToken),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || "Failed to fetch models");
  }
  return res.json();
}

export async function getMlflowConfig(): Promise<MlflowInfo> {
  const res = await fetch(`${API_BASE}/mlflow`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? 'Failed to fetch MLflow config');
  return data;
}

export async function getMlflowModelRegistry(adminToken?: string): Promise<MlflowModelRegistry> {
  const res = await fetch(`${API_BASE}/mlflow/model-registry`, {
    headers: adminHeaders(adminToken),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? 'Failed to fetch MLflow model registry');
  return data;
}

export async function deleteModel(modelName: string, adminToken?: string) {
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(modelName)}`, {
    method: 'DELETE',
    headers: adminHeaders(adminToken),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? 'Failed to delete model');
  return data as { deleted: boolean; model_name: string; deleted_paths: string[] };
}

export async function getModelAudit(modelName: string, adminToken?: string, topk = 4): Promise<ModelAuditResponse> {
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(modelName)}/waveform-audit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(adminToken ? { 'X-Admin-Token': adminToken } : {}),
    },
    body: JSON.stringify({ topk }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? 'Failed to load model audit plots');
  return data;
}

export function toFileUrl(relativePath?: string | null) {
  if (!relativePath) return '';
  if (relativePath.startsWith('http')) return relativePath;
  const base = API_BASE.endsWith('/api') ? API_BASE.slice(0, -4) : API_BASE;
  return `${base}${relativePath}`;
}
