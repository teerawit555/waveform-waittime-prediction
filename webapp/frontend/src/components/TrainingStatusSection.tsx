import { Activity, BarChart3, Database, Download, FileCode2, Images, LineChart, Target } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart as ReLineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { MlflowInfo, toFileUrl } from '../lib/api';
import MlflowStatusPanel from './MlflowStatusPanel';
import OverfittingCard from './OverfittingCard';
import PlotCard from './PlotCard';
import ProgressBar from './ProgressBar';
import SectionDivider from './SectionDivider';
import StatCard from './StatCard';

type TrainingStatusSectionProps = {
  trainJob: any;
  mlflowInfo?: MlflowInfo | null;
  trainMetrics: Record<string, any>;
  trainHistory: any[];
  overfittingSummary?: any;
};

export default function TrainingStatusSection({
  trainJob,
  mlflowInfo,
  trainMetrics,
  trainHistory,
  overfittingSummary,
}: TrainingStatusSectionProps) {
  const runStatus = trainJob?.status ?? 'idle';
  const hasMetrics = Object.keys(trainMetrics).length > 0;
  const hasHistory = trainHistory.length > 0;
  const evaluation = trainJob?.result?.evaluation ?? {};
  const evaluationPoints = evaluation.sample_points ?? [];
  const absErrorHist = evaluation.abs_error_hist ?? [];
  const distribution = evaluation.distribution ?? [];
  const worstCases = evaluation.worst_cases ?? [];
  const hasEvaluation = evaluationPoints.length > 0;
  const bestEpoch = hasHistory
    ? trainHistory.reduce((best, point) => (
        Number(point.val_loss ?? Infinity) < Number(best.val_loss ?? Infinity) ? point : best
      ), trainHistory[0])
    : null;
  const finalEpoch = hasHistory ? trainHistory[trainHistory.length - 1] : null;
  const mlflowRunId = mlflowInfo?.run_id || mlflowInfo?.latest_run_id || trainJob?.result?.mlflow?.run_id;
  const registryInfo = mlflowInfo?.registry || trainJob?.result?.mlflow?.registry;
  const plotArtifacts = trainJob?.result?.plots ? Object.keys(trainJob.result.plots).filter((key) => trainJob.result.plots[key]) : [];
  const resultArtifacts = trainJob?.result?.results ? Object.keys(trainJob.result.results).filter((key) => trainJob.result.results[key]) : [];
  const artifactCount = plotArtifacts.length + resultArtifacts.length;
  const waveformItems = trainJob?.result?.analysis_manifest ?? [];
  const waveformTotal = trainJob?.result?.total_waves ?? waveformItems.length;
  const displayedWaveforms = [...waveformItems].sort((a: any, b: any) => {
    const numA = parseInt(String(a.wave_id ?? '').replace(/\D/g, ''), 10);
    const numB = parseInt(String(b.wave_id ?? '').replace(/\D/g, ''), 10);
    if (!Number.isNaN(numA) && !Number.isNaN(numB)) return numA - numB;
    return String(a.wave_id ?? '').localeCompare(String(b.wave_id ?? ''));
  });
  const formatWaveMetric = (value: any) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(6) : '--';
  };
  const formatSignedWaveMetric = (value: any) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '--';
    return `${numeric > 0 ? '+' : ''}${numeric.toFixed(6)}`;
  };

  return (
    <section className="card analytics-section">
      <div className="section-header">
        <div className="section-title"><Activity size={18} /><span>Training Status</span></div>
      </div>

      {trainJob ? (
        <>
          <div className="run-overview">
            <div className="run-progress-panel">
              <div className="analytics-kicker">
                <LineChart size={15} />
                <span>Active Run</span>
              </div>
              <div className="run-overview-head">
                <div>
                  <h3>{runStatus === 'completed' ? 'Training completed' : 'Training pipeline'}</h3>
                  <p>{trainJob.message || 'Waiting for the next training event.'}</p>
                </div>
                <span className={`status-pill status-${runStatus}`}>{runStatus}</span>
              </div>
              <ProgressBar progress={trainJob.progress ?? 0} />
            </div>
            <MlflowStatusPanel info={mlflowInfo} mode="current" />
          </div>

          <div className="metrics-ribbon">
            <StatCard title="MAE (All)" value={hasMetrics ? Number(trainMetrics.mae_all ?? 0).toFixed(6) : '--'} />
            <StatCard title="RMSE" value={hasMetrics ? Number(trainMetrics.rmse ?? 0).toFixed(6) : '--'} />
            <StatCard title="Fast Precision" value={hasMetrics ? Number(trainMetrics.fast_precision ?? 0).toFixed(6) : '--'} />
            <StatCard title="Fast Recall" value={hasMetrics ? Number(trainMetrics.fast_recall ?? 0).toFixed(6) : '--'} />
          </div>

          <div className="run-history-panel">
            <div className="run-history-head">
              <div>
                <span className="view-kicker">MLflow-lite Monitor</span>
                <h3>Run history, artifacts, and registry readiness</h3>
              </div>
              <span className={`status-pill status-${runStatus}`}>{runStatus}</span>
            </div>
            <div className="run-history-grid">
              <div>
                <FileCode2 size={17} />
                <span>Run ID</span>
                <strong title={mlflowRunId || undefined}>{mlflowRunId || 'Not logged yet'}</strong>
              </div>
              <div>
                <Activity size={17} />
                <span>Events</span>
                <strong>{hasHistory ? `${trainHistory.length} epochs tracked` : 'Waiting for history'}</strong>
              </div>
              <div>
                <Download size={17} />
                <span>Artifacts</span>
                <strong>{artifactCount ? `${artifactCount} files exported` : 'No artifacts yet'}</strong>
              </div>
              <div>
                <Database size={17} />
                <span>Registry</span>
                <strong>{registryInfo?.model_version ? `v${registryInfo.model_version}` : registryInfo?.reason || 'Pending registration'}</strong>
              </div>
            </div>
          </div>

          <SectionDivider label="TCN Diagnostics" />
          <div className="analytics-grid">
            <div className="analytics-main">
              <div className="chart-panel chart-panel-hero">
                <div className="chart-panel-head">
                  <div>
                    <span>Encoder</span>
                    <h3>Learning Curve</h3>
                    <p>Train and validation loss behavior across epochs.</p>
                  </div>
                  {bestEpoch ? (
                    <div className="chart-callout">
                      <span>Best Epoch</span>
                      <strong>{bestEpoch.epoch}</strong>
                    </div>
                  ) : null}
                </div>
                {hasHistory ? (
                  <>
                    <div className="native-chart-wrap">
                      <ResponsiveContainer width="100%" height={300}>
                        <ReLineChart data={trainHistory} margin={{ top: 12, right: 20, left: 4, bottom: 8 }}>
                          <CartesianGrid stroke="rgba(143,163,189,.22)" vertical={false} />
                          <XAxis
                            dataKey="epoch"
                            tickLine={false}
                            axisLine={false}
                            tick={{ fill: '#4D6A8B', fontSize: 11 }}
                            label={{ value: 'Epoch', position: 'insideBottomRight', offset: -2, fill: '#8FA3BD', fontSize: 11 }}
                          />
                          <YAxis
                            tickLine={false}
                            axisLine={false}
                            tick={{ fill: '#4D6A8B', fontSize: 11 }}
                            width={52}
                          />
                          <Tooltip
                            contentStyle={{
                              border: '1px solid #D4DDE9',
                              borderRadius: 6,
                              boxShadow: '0 10px 28px rgba(0,58,112,.12)',
                              fontFamily: 'Barlow, sans-serif',
                            }}
                            labelFormatter={(value) => `Epoch ${value}`}
                          />
                          <Line
                            type="monotone"
                            dataKey="train_loss"
                            name="Train Loss"
                            stroke="#00528A"
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={{ r: 5 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="val_loss"
                            name="Validation Loss"
                            stroke="#00AEEF"
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={{ r: 5 }}
                          />
                        </ReLineChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="chart-micro-stats">
                      <div>
                        <span>Final Train</span>
                        <strong>{Number(finalEpoch?.train_loss ?? 0).toFixed(5)}</strong>
                      </div>
                      <div>
                        <span>Final Val</span>
                        <strong>{Number(finalEpoch?.val_loss ?? 0).toFixed(5)}</strong>
                      </div>
                      <div>
                        <span>Best Val</span>
                        <strong>{Number(bestEpoch?.val_loss ?? 0).toFixed(5)}</strong>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="chart-empty-state">
                    <BarChart3 size={28} />
                    <strong>Native learning curve unavailable</strong>
                    <span>Training history JSON is missing for this run. The exported PNG is available in Plot Artifacts below.</span>
                  </div>
                )}
              </div>
            </div>
            <div className="analytics-side">
              <OverfittingCard summary={overfittingSummary} />
            </div>
          </div>

          <SectionDivider label="AutoGluon Evaluation" />
          {hasEvaluation ? (
            <>
              <div className="evaluation-grid">
                <div className="chart-panel chart-panel-hero">
                  <div className="chart-panel-head">
                    <div>
                      <span>Calibration</span>
                      <h3>Actual vs Predicted</h3>
                      <p>Interactive scatter from prediction CSV, not exported PNG.</p>
                    </div>
                    <div className="chart-callout">
                      <span>Samples</span>
                      <strong>{evaluation.total_points ?? evaluationPoints.length}</strong>
                    </div>
                  </div>
                  <div className="native-chart-wrap">
                    <ResponsiveContainer width="100%" height={300}>
                      <ScatterChart margin={{ top: 12, right: 20, left: 4, bottom: 8 }}>
                        <CartesianGrid stroke="rgba(143,163,189,.18)" />
                        <XAxis
                          type="number"
                          dataKey="true"
                          name="Actual"
                          tickLine={false}
                          axisLine={false}
                          tick={{ fill: '#4D6A8B', fontSize: 11 }}
                        />
                        <YAxis
                          type="number"
                          dataKey="pred"
                          name="Predicted"
                          tickLine={false}
                          axisLine={false}
                          tick={{ fill: '#4D6A8B', fontSize: 11 }}
                          width={52}
                        />
                        <Tooltip
                          cursor={{ stroke: '#8FA3BD', strokeDasharray: '3 3' }}
                          contentStyle={{
                            border: '1px solid #D4DDE9',
                            borderRadius: 6,
                            boxShadow: '0 10px 28px rgba(0,58,112,.12)',
                            fontFamily: 'Barlow, sans-serif',
                          }}
                        />
                        <Scatter name="Prediction" data={evaluationPoints} fill="#0078C8" fillOpacity={0.72} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="chart-panel chart-panel-hero">
                  <div className="chart-panel-head">
                    <div>
                      <span>Residuals</span>
                      <h3>Error by Actual</h3>
                      <p>Residual spread across wait-time range.</p>
                    </div>
                  </div>
                  <div className="native-chart-wrap">
                    <ResponsiveContainer width="100%" height={300}>
                      <ScatterChart margin={{ top: 12, right: 20, left: 4, bottom: 8 }}>
                        <CartesianGrid stroke="rgba(143,163,189,.18)" />
                        <XAxis
                          type="number"
                          dataKey="true"
                          name="Actual"
                          tickLine={false}
                          axisLine={false}
                          tick={{ fill: '#4D6A8B', fontSize: 11 }}
                        />
                        <YAxis
                          type="number"
                          dataKey="error"
                          name="Residual"
                          tickLine={false}
                          axisLine={false}
                          tick={{ fill: '#4D6A8B', fontSize: 11 }}
                          width={52}
                        />
                        <ReferenceLine y={0} stroke="#00AEEF" strokeDasharray="4 4" />
                        <Tooltip
                          cursor={{ stroke: '#8FA3BD', strokeDasharray: '3 3' }}
                          contentStyle={{
                            border: '1px solid #D4DDE9',
                            borderRadius: 6,
                            boxShadow: '0 10px 28px rgba(0,58,112,.12)',
                            fontFamily: 'Barlow, sans-serif',
                          }}
                        />
                        <Scatter name="Residual" data={evaluationPoints} fill="#00528A" fillOpacity={0.68} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="chart-panel">
                  <div className="chart-panel-head">
                    <div>
                      <span>Error Distribution</span>
                      <h3>Absolute Error Histogram</h3>
                      <p>Count of samples by absolute prediction error.</p>
                    </div>
                  </div>
                  <div className="native-chart-wrap">
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={absErrorHist} margin={{ top: 12, right: 20, left: 0, bottom: 8 }}>
                        <CartesianGrid stroke="rgba(143,163,189,.18)" vertical={false} />
                        <XAxis dataKey="bin" tickLine={false} axisLine={false} tick={{ fill: '#4D6A8B', fontSize: 10 }} minTickGap={18} />
                        <YAxis tickLine={false} axisLine={false} tick={{ fill: '#4D6A8B', fontSize: 11 }} width={40} />
                        <Tooltip
                          contentStyle={{
                            border: '1px solid #D4DDE9',
                            borderRadius: 6,
                            boxShadow: '0 10px 28px rgba(0,58,112,.12)',
                            fontFamily: 'Barlow, sans-serif',
                          }}
                        />
                        <Bar dataKey="count" name="Count" fill="#0078C8" radius={[5, 5, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="chart-panel">
                  <div className="chart-panel-head">
                    <div>
                      <span>Target Shape</span>
                      <h3>True vs Pred Distribution</h3>
                      <p>Distribution overlay from prediction values.</p>
                    </div>
                  </div>
                  <div className="native-chart-wrap">
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={distribution} margin={{ top: 12, right: 20, left: 0, bottom: 8 }}>
                        <CartesianGrid stroke="rgba(143,163,189,.18)" vertical={false} />
                        <XAxis dataKey="bin" tickLine={false} axisLine={false} tick={{ fill: '#4D6A8B', fontSize: 10 }} minTickGap={18} />
                        <YAxis tickLine={false} axisLine={false} tick={{ fill: '#4D6A8B', fontSize: 11 }} width={40} />
                        <Tooltip
                          contentStyle={{
                            border: '1px solid #D4DDE9',
                            borderRadius: 6,
                            boxShadow: '0 10px 28px rgba(0,58,112,.12)',
                            fontFamily: 'Barlow, sans-serif',
                          }}
                        />
                        <Legend />
                        <Bar dataKey="true" name="True" fill="#00528A" radius={[5, 5, 0, 0]} />
                        <Bar dataKey="pred" name="Predicted" fill="#00AEEF" radius={[5, 5, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {worstCases.length ? (
                <div className="worst-cases-panel">
                  <div className="chart-panel-head">
                    <div>
                      <span>Audit Queue</span>
                      <h3>Worst Prediction Cases</h3>
                    </div>
                  </div>
                  <div className="worst-case-list">
                    {worstCases.map((item: any, index: number) => (
                      <div className="worst-case-row" key={`${item.wave_id ?? 'row'}-${index}`}>
                        <strong>{item.wave_id ?? `#${index + 1}`}</strong>
                        <span>Actual {Number(item.true).toFixed(5)}</span>
                        <span>Pred {Number(item.pred).toFixed(5)}</span>
                        <em>Error {Number(item.abs_error).toFixed(5)}</em>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </>
          ) : (
            <div className="chart-empty-state">
              <BarChart3 size={28} />
              <strong>Native AutoGluon evaluation unavailable</strong>
              <span>Prediction CSV data is missing for this model. Exported PNG artifacts are still available below.</span>
            </div>
          )}

          <SectionDivider label="Waveform Prediction Audit" />
          {displayedWaveforms.length ? (
            <div className="training-waveform-gallery">
              <div className="chart-panel-head">
                <div>
                  <span>Per-Wave Review</span>
                  <h3>Label vs Prediction Gallery</h3>
                  <p>Each exported waveform overlays the ground-truth label and model prediction in one plot.</p>
                </div>
                <div className="chart-callout">
                  <span>Test Waves</span>
                  <strong>{waveformTotal}</strong>
                </div>
              </div>
              <div className="training-waveform-legend" aria-label="Waveform plot legend">
                <span><i className="legend-line is-waveform" /> Waveform</span>
                <span><i className="legend-line is-label" /> Label</span>
                <span><i className="legend-line is-prediction" /> Prediction</span>
              </div>
              <div className="training-waveform-grid">
                {displayedWaveforms.map((item: any) => {
                  const error = item.abs_error ?? (
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
                            <strong>{formatWaveMetric(item.true)}</strong>
                          </div>
                          <div>
                            <span>Prediction</span>
                            <strong>{formatWaveMetric(item.pred)}</strong>
                          </div>
                          <div>
                            <span>Error</span>
                            <strong>{formatSignedWaveMetric(item.error)}</strong>
                          </div>
                          <div>
                            <span>Abs Error</span>
                            <strong>{formatWaveMetric(error)}</strong>
                          </div>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="chart-empty-state">
              <Images size={28} />
              <strong>No waveform audit plots yet</strong>
              <span>New training runs will export per-wave plots with label and prediction markers after AutoGluon evaluation.</span>
            </div>
          )}

          <details className="artifact-disclosure">
            <summary>
              <span>Plot Artifacts</span>
              <strong>Exported backend images for audit and debugging</strong>
            </summary>
          <div className="plot-dashboard">
            <PlotCard
              title="Loss Curve"
              eyebrow="Fit"
              description="Model loss trend from the training run."
              imageUrl={toFileUrl(trainJob.result?.plots?.loss_curve)}
            />
            <PlotCard
              title="Actual vs Predicted"
              eyebrow="Calibration"
              description="Prediction alignment against validation targets."
              imageUrl={toFileUrl(trainJob.result?.plots?.actual_vs_pred)}
            />
            <PlotCard
              title="Error Histogram"
              eyebrow="Residuals"
              description="Distribution of validation error."
              imageUrl={toFileUrl(trainJob.result?.plots?.error_histogram)}
            />
            <PlotCard
              title="Target Distribution"
              eyebrow="Dataset"
              description="Target shape used by the model."
              imageUrl={toFileUrl(trainJob.result?.plots?.target_distribution)}
            />
          </div>
          </details>

          {trainHistory.length ? (
            <div className="insight-strip">
              <Target size={16} />
              <span>Training history detected across <strong>{trainHistory.length}</strong> epochs.</span>
            </div>
          ) : null}
          {trainJob.result?.results?.validation_predictions_csv ? (
            <a className="ghost-btn" href={toFileUrl(trainJob.result.results.validation_predictions_csv)} target="_blank" rel="noreferrer">
              <Download size={15} />
              Download Validation Predictions CSV
            </a>
          ) : null}
        </>
      ) : (
        <>
          <MlflowStatusPanel info={mlflowInfo} />
          <div className="empty-state analytics-empty">
            <BarChart3 size={28} />
            <strong>No training run selected</strong>
            <span>Run training or load an existing model to populate diagnostics, metrics, and feature analysis.</span>
          </div>
        </>
      )}
    </section>
  );
}
