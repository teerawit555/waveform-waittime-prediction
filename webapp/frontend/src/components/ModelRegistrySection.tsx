import { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, Database, GitCompareArrows, Images, RefreshCw, Trash2 } from 'lucide-react';
import { MlflowModelRegistry, MlflowModelVersion, ModelAuditResponse, toFileUrl } from '../lib/api';
import { formatModelName, getModelNote } from '../lib/modelDisplay';
import StatCard from './StatCard';

type ModelRegistrySectionProps = {
  registry: MlflowModelRegistry | null;
  loading: boolean;
  onRefresh: () => void;
  onDeleteModel: (modelName: string) => void;
  onLoadModelAudit: (modelName: string) => Promise<ModelAuditResponse>;
};

const DEFAULT_MODEL_NAME = 'TCN_aug_weighted_v1';

const metricLabels: Record<string, string> = {
  mae_all: 'MAE',
  rmse: 'RMSE',
  fast_precision: 'Fast Precision',
  fast_recall: 'Fast Recall',
  mae_fast: 'MAE Fast',
  mae_slow: 'MAE Slow',
};

const higherIsBetter = new Set(['fast_precision', 'fast_recall']);

function formatMetric(value?: number | null, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return Number(value).toFixed(digits);
}

function formatDate(value?: string | null) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toLocaleString();
}

function versionLabel(version: MlflowModelVersion) {
  const aliases = version.aliases?.length ? ` (${version.aliases.join(', ')})` : '';
  return `v${version.version}${aliases}`;
}

function versionDisplayLabel(version: MlflowModelVersion) {
  const runLabel = formatModelName(version.run_name);
  return runLabel ? `${versionLabel(version)} / ${runLabel}` : versionLabel(version);
}

function compactRunId(runId?: string | null) {
  if (!runId) return '-';
  if (runId.length <= 16) return runId;
  return `${runId.slice(0, 8)}...${runId.slice(-6)}`;
}

function formatRegistryName(name?: string | null) {
  if (!name) return '-';
  return name.replace(/[_-]+/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function getVersionModelName(version: MlflowModelVersion) {
  return version.tags?.ag_model_name || version.version_tags?.ag_model_name || version.run_name || '';
}

export default function ModelRegistrySection({ registry, loading, onRefresh, onDeleteModel, onLoadModelAudit }: ModelRegistrySectionProps) {
  const versions = registry?.versions ?? [];
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);
  const [auditModelName, setAuditModelName] = useState('');
  const [auditData, setAuditData] = useState<ModelAuditResponse | null>(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditError, setAuditError] = useState<string | null>(null);

  useEffect(() => {
    if (!versions.length) {
      setSelectedVersions([]);
      return;
    }
    setSelectedVersions((current) => {
      const valid = current.filter((version) => versions.some((item) => item.version === version));
      return valid.length ? valid : versions.slice(0, 2).map((item) => item.version);
    });
  }, [versions]);

  const selectedRows = useMemo(() => {
    return versions.filter((item) => selectedVersions.includes(item.version));
  }, [selectedVersions, versions]);

  const metricWinners = useMemo(() => {
    const winners: Record<string, string | null> = {};
    Object.keys(metricLabels).forEach((key) => {
      const scoredRows = selectedRows
        .map((version) => ({
          version: version.version,
          value: Number(version.metrics?.[key as keyof MlflowModelVersion['metrics']]),
        }))
        .filter((item) => Number.isFinite(item.value));

      if (!scoredRows.length) {
        winners[key] = null;
        return;
      }

      const bestValue = higherIsBetter.has(key)
        ? Math.max(...scoredRows.map((item) => item.value))
        : Math.min(...scoredRows.map((item) => item.value));
      const bestRows = scoredRows.filter((item) => item.value === bestValue);
      winners[key] = bestRows.length === 1 ? bestRows[0].version : null;
    });
    return winners;
  }, [selectedRows]);

  const toggleVersion = (version: string) => {
    setSelectedVersions((current) => {
      if (current.includes(version)) return current.filter((item) => item !== version);
      return [...current, version].slice(-3);
    });
  };

  const openModelAudit = async (modelName: string) => {
    if (!modelName) return;
    setAuditModelName(modelName);
    setAuditLoading(true);
    setAuditError(null);
    try {
      const data = await onLoadModelAudit(modelName);
      setAuditData(data);
    } catch (err: any) {
      setAuditData(null);
      setAuditError(err.message || 'Failed to load model audit plots');
    } finally {
      setAuditLoading(false);
    }
  };

  const bestVersion = registry?.best_version ? `v${registry.best_version}` : '-';
  const candidateVersion = registry?.candidate_version ? `v${registry.candidate_version}` : '-';
  const registryMode = registry?.registry_backend === 'local' ? 'Local Artifact Registry' : 'MLflow-lite Registry';
  const formatAuditMetric = (value: any) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(6) : '--';
  };
  const formatSignedAuditMetric = (value: any) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '--';
    return `${numeric > 0 ? '+' : ''}${numeric.toFixed(6)}`;
  };

  return (
    <section className="card model-registry-section">
      <div className="section-header">
        <div className="section-title">
          <Database size={18} />
          <span>Model Registry</span>
        </div>
        <button className="icon-btn" type="button" onClick={onRefresh} disabled={loading} title="Refresh model registry">
          <RefreshCw size={16} />
        </button>
      </div>

      {!registry ? (
        <div className="empty-state empty-action-state">
          <strong>Registry not loaded</strong>
          <span>Refresh MLflow registry metadata to compare trained model versions.</span>
          <button type="button" className="empty-state-action" onClick={onRefresh} disabled={loading}>
            Refresh registry
          </button>
        </div>
      ) : !registry.enabled ? (
        <div className="error-banner">{registry.reason || 'MLflow model registry is unavailable.'}</div>
      ) : versions.length === 0 ? (
        <div className="empty-state empty-action-state">
          <strong>No model versions yet</strong>
          <span>Train a model to create the first comparable version.</span>
          <button type="button" className="empty-state-action" onClick={onRefresh} disabled={loading}>
            Refresh registry
          </button>
        </div>
      ) : (
        <>
          {registry.reason ? (
            <div className="note-box registry-fallback-note">{registry.reason}</div>
          ) : null}

          <div className="grid four compact-gap">
            <StatCard title="Registered Model" value={formatRegistryName(registry.registered_model_name)} />
            <StatCard title="Versions" value={registry.version_count ?? versions.length} />
            <StatCard title="Candidate" value={candidateVersion} />
            <StatCard title="Best MAE" value={registry.best_mae_all != null ? `${bestVersion} / ${formatMetric(registry.best_mae_all)}` : '-'} />
          </div>

          <div className="registry-run-table-panel">
            <div className="registry-run-table-head">
              <div>
                <span className="view-kicker">{registryMode}</span>
                <h3>Scan model versions, run status, and deploy readiness</h3>
              </div>
              <span>{versions.length} version{versions.length === 1 ? '' : 's'}</span>
            </div>
            <div className="registry-run-table-wrap">
              <table className="registry-run-table">
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Alias / Status</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>Fast Recall</th>
                    <th>Run</th>
                    <th>Updated</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {versions.map((version) => {
                    const isBest = registry.best_version === version.version;
                    const aliasLabel = version.aliases?.length ? version.aliases.join(', ') : version.status || version.current_stage || '-';
                    const modelName = getVersionModelName(version);
                    const isDefaultModel = modelName === DEFAULT_MODEL_NAME;
                    return (
                      <tr key={`registry-table-${version.version}`}>
                        <td>
                          <strong>{versionLabel(version)}</strong>
                          {isBest ? <span className="registry-mini-badge">Best</span> : null}
                        </td>
                        <td>{aliasLabel}</td>
                        <td>{formatMetric(version.metrics?.mae_all)}</td>
                        <td>{formatMetric(version.metrics?.rmse)}</td>
                        <td>{formatMetric(version.metrics?.fast_recall, 4)}</td>
                        <td className="mono truncate" title={version.run_id || undefined}>{compactRunId(version.run_id)}</td>
                        <td>{formatDate(version.updated_at || version.created_at)}</td>
                        <td>
                          <div className="registry-row-actions">
                            <button
                              type="button"
                              className="registry-audit-btn"
                              onClick={() => openModelAudit(modelName)}
                              disabled={!modelName || auditLoading}
                              title={modelName ? `View waveform audit for ${formatModelName(modelName)}` : 'Audit plots unavailable'}
                            >
                              <Images size={14} />
                              <span>Audit</span>
                            </button>
                            <button
                              type="button"
                              className="registry-delete-btn"
                              onClick={() => onDeleteModel(modelName)}
                              disabled={!modelName || isDefaultModel}
                              title={isDefaultModel ? 'Default model cannot be deleted' : `Delete ${formatModelName(modelName)}`}
                            >
                              <Trash2 size={14} />
                              <span>Delete</span>
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {(auditLoading || auditError || auditData) ? (
            <div className="registry-audit-panel">
              <div className="registry-run-table-head">
                <div>
                  <span className="view-kicker">Waveform Audit</span>
                  <h3>{auditModelName ? formatModelName(auditModelName) : 'Model audit plots'}</h3>
                </div>
                {auditData ? <span>{auditData.total_waves} waves total</span> : null}
              </div>

              {auditLoading ? (
                <div className="empty-state">Generating audit plots...</div>
              ) : auditError ? (
                <div className="error-banner">{auditError}</div>
              ) : auditData ? (
                <>
                  <div className="training-waveform-legend" aria-label="Waveform audit legend">
                    <span><i className="legend-line is-waveform" /> Waveform</span>
                    <span><i className="legend-line is-label" /> Label</span>
                    <span><i className="legend-line is-prediction" /> Prediction</span>
                  </div>
                  <div className="training-waveform-grid registry-audit-grid">
                    {auditData.analysis_manifest.map((item) => {
                      const absError = item.abs_error ?? (
                        item.pred != null && item.true != null ? Math.abs(Number(item.pred) - Number(item.true)) : null
                      );

                      return (
                        <article className="training-waveform-card" key={`${item.wave_id}-${item.image}`}>
                          <img src={toFileUrl(item.image)} alt={`Waveform ${item.wave_id}`} />
                          <div className="training-waveform-meta">
                            <div className="training-waveform-title">
                              <Images size={15} />
                              <strong>Wave {item.wave_id}</strong>
                            </div>
                            <div className="training-waveform-values">
                              <div>
                                <span>Label</span>
                                <strong>{formatAuditMetric(item.true)}</strong>
                              </div>
                              <div>
                                <span>Prediction</span>
                                <strong>{formatAuditMetric(item.pred)}</strong>
                              </div>
                              <div>
                                <span>Error</span>
                                <strong>{formatSignedAuditMetric(item.error)}</strong>
                              </div>
                              <div>
                                <span>Abs Error</span>
                                <strong>{formatAuditMetric(absError)}</strong>
                              </div>
                            </div>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                </>
              ) : null}
            </div>
          ) : null}

          <div className="registry-layout">
            <div className="registry-version-list">
              {versions.map((version) => {
                const selected = selectedVersions.includes(version.version);
                const isBest = registry.best_version === version.version;
                return (
                  <button
                    key={version.version}
                    type="button"
                    className={`registry-version-row ${selected ? 'is-selected' : ''}`}
                    onClick={() => toggleVersion(version.version)}
                  >
                    <div className="registry-version-head">
                      <span className="registry-version-name">{versionDisplayLabel(version)}</span>
                      {isBest ? <span className="registry-chip is-best"><CheckCircle2 size={13} /> Best</span> : null}
                    </div>
                    {getModelNote(version.run_name) ? <div className="registry-version-note">{getModelNote(version.run_name)}</div> : null}
                    <div className="registry-version-meta">
                      <span>{version.status || '-'}</span>
                      <span>{formatDate(version.created_at)}</span>
                    </div>
                    <div className="registry-metric-strip">
                      <span>MAE {formatMetric(version.metrics?.mae_all)}</span>
                      <span>RMSE {formatMetric(version.metrics?.rmse)}</span>
                      <span>Recall {formatMetric(version.metrics?.fast_recall, 4)}</span>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="registry-compare-panel">
              <div className="registry-compare-title">
                <GitCompareArrows size={16} />
                <span>Compare Selected Versions</span>
              </div>

              {selectedRows.length === 0 ? (
                <div className="empty-state">Select model versions to compare.</div>
              ) : (
                <div className="registry-compare-table-wrap">
                  <table className="registry-compare-table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        {selectedRows.map((version) => (
                          <th key={version.version}>{versionDisplayLabel(version)}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(metricLabels).map(([key, label]) => (
                        <tr key={key}>
                          <td>{label}</td>
                          {selectedRows.map((version) => (
                            <td
                              key={`${version.version}-${key}`}
                              className={metricWinners[key] === version.version ? 'is-winner' : undefined}
                            >
                              {formatMetric(version.metrics?.[key as keyof MlflowModelVersion['metrics']])}
                            </td>
                          ))}
                        </tr>
                      ))}
                      <tr>
                        <td>Run</td>
                        {selectedRows.map((version) => (
                          <td
                            key={`${version.version}-run`}
                            className="mono truncate"
                            title={version.run_id || undefined}
                          >
                            {compactRunId(version.run_id)}
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
