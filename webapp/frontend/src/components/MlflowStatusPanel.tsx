import { Database } from 'lucide-react';
import { MlflowInfo } from '../lib/api';

type MlflowStatusPanelProps = {
  info?: MlflowInfo | null;
  mode?: 'current' | 'latest';
};

export default function MlflowStatusPanel({ info, mode = 'latest' }: MlflowStatusPanelProps) {
  if (!info) return null;

  const registry = info.registry;
  const visibleRunId = info.run_id ?? info.latest_run_id;
  const isCurrent = mode === 'current';

  const title = registry?.model_version
    ? (isCurrent ? 'Model version registered' : 'Latest model version')
    : visibleRunId
      ? (isCurrent ? 'Experiment run recorded' : 'Latest experiment run')
      : (isCurrent ? 'Tracking ready for the next run' : 'Tracking ready for training');

  const primaryLabel = registry?.model_version ? 'Model Version' : 'Experiment';
  const primaryValue = registry?.model_version ? `v${registry.model_version}` : info.experiment_name;

  const secondaryLabel = registry?.registered_model_name
    ? 'Registered Model'
    : visibleRunId
      ? (info.run_id ? 'Current Run' : 'Latest Run')
      : 'Tracking URI';

  const secondaryValue = registry?.registered_model_name
    || visibleRunId
    || info.tracking_uri
    || info.reason;

  const status = registry?.model_version_status
    || info.latest_run_status
    || (info.enabled ? 'Configured' : 'Inactive');

  return (
    <div className={`mlflow-panel ${info.enabled ? 'is-active' : 'is-inactive'}`}>
      <div className="mlflow-main">
        <div className="mlflow-kicker">
          <Database size={15} />
          <span>MLflow</span>
        </div>
        <div className="mlflow-title">{title}</div>
      </div>
      <div className="mlflow-meta">
        <div>
          <span>{primaryLabel}</span>
          <strong>{primaryValue}</strong>
        </div>
        <div>
          <span>{secondaryLabel}</span>
          <strong className="mono truncate">{secondaryValue}</strong>
        </div>
        <div>
          <span>Status</span>
          <strong>{status}</strong>
        </div>
      </div>
    </div>
  );
}
