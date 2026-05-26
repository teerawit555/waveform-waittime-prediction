import { Layers3, Sparkles } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { toFileUrl } from '../lib/api';
import PlotCard from './PlotCard';
import StatCard from './StatCard';

type FeatureImportanceSectionProps = {
  featureSummary?: any;
  trainJob: any;
};

export default function FeatureImportanceSection({ featureSummary, trainJob }: FeatureImportanceSectionProps) {
  const colors = ['#00528A', '#00AEEF', '#8FA3BD'];
  const groupValues = [
    {
      label: 'TCN Embeddings',
      value: Number(featureSummary?.group_sum?.tcn_embedding ?? 0),
      count: featureSummary?.top30_count?.tcn_embedding ?? 0,
    },
    {
      label: 'Late Settle',
      value: Number(featureSummary?.group_sum?.late_settle ?? 0),
      count: featureSummary?.top30_count?.late_settle ?? 0,
    },
    {
      label: 'Other Signals',
      value: Number(featureSummary?.group_sum?.handcrafted_other ?? 0),
      count: featureSummary?.top30_count?.handcrafted_other ?? 0,
    },
  ];
  const maxGroupValue = Math.max(...groupValues.map((item) => item.value), 1);
  const chartData = groupValues.map((item) => ({
    name: item.label,
    importance: Number(item.value.toFixed(6)),
    top30: item.count,
  }));

  return (
    <section className="card analytics-section">
      <div className="section-header">
        <div className="section-title">
          <Sparkles size={18} />
          <span>Feature Importance Analysis</span>
        </div>
      </div>

      {featureSummary ? (
        <>
          <div className="feature-hero">
            <div>
              <div className="analytics-kicker">
                <Layers3 size={15} />
                <span>Signal Attribution</span>
              </div>
              <h3>Which signal families are driving the model?</h3>
              <p>Native charts show grouped importance directly in the app. Exported PNGs are kept below as artifacts.</p>
            </div>
            <div className="feature-rank-list">
              {groupValues.map((item) => (
                <div className="feature-rank-row" key={item.label}>
                  <div>
                    <strong>{item.label}</strong>
                    <span>{item.count} features in top 30</span>
                  </div>
                  <div className="feature-bar-track" aria-hidden="true">
                    <span style={{ width: `${Math.max(8, (item.value / maxGroupValue) * 100)}%` }} />
                  </div>
                  <em>{item.value.toFixed(4)}</em>
                </div>
              ))}
            </div>
          </div>

          <div className="feature-chart-grid">
            <div className="chart-panel">
              <div className="chart-panel-head">
                <div>
                  <span>Grouped Importance</span>
                  <h3>Model Signal Mix</h3>
                  <p>Contribution sum by feature family.</p>
                </div>
              </div>
              <div className="native-chart-wrap">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 28, left: 24, bottom: 8 }}>
                    <CartesianGrid stroke="rgba(143,163,189,.18)" horizontal={false} />
                    <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: '#4D6A8B', fontSize: 11 }} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tickLine={false}
                      axisLine={false}
                      tick={{ fill: '#1C3048', fontSize: 12 }}
                      width={118}
                    />
                    <Tooltip
                      cursor={{ fill: 'rgba(0,174,239,.08)' }}
                      contentStyle={{
                        border: '1px solid #D4DDE9',
                        borderRadius: 6,
                        boxShadow: '0 10px 28px rgba(0,58,112,.12)',
                        fontFamily: 'Barlow, sans-serif',
                      }}
                    />
                    <Bar dataKey="importance" name="Importance" radius={[0, 6, 6, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={entry.name} fill={colors[index % colors.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="chart-panel">
              <div className="chart-panel-head">
                <div>
                  <span>Top-N Coverage</span>
                  <h3>Top 30 Composition</h3>
                  <p>How many top features come from each family.</p>
                </div>
              </div>
              <div className="native-chart-wrap">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={chartData} margin={{ top: 12, right: 18, left: 0, bottom: 8 }}>
                    <CartesianGrid stroke="rgba(143,163,189,.18)" vertical={false} />
                    <XAxis
                      dataKey="name"
                      tickLine={false}
                      axisLine={false}
                      tick={{ fill: '#4D6A8B', fontSize: 11 }}
                      interval={0}
                    />
                    <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fill: '#4D6A8B', fontSize: 11 }} width={34} />
                    <Tooltip
                      cursor={{ fill: 'rgba(0,174,239,.08)' }}
                      contentStyle={{
                        border: '1px solid #D4DDE9',
                        borderRadius: 6,
                        boxShadow: '0 10px 28px rgba(0,58,112,.12)',
                        fontFamily: 'Barlow, sans-serif',
                      }}
                    />
                    <Bar dataKey="top30" name="Top 30 Count" radius={[6, 6, 0, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={entry.name} fill={colors[index % colors.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="metrics-ribbon">
            <StatCard title="Total Features" value={featureSummary.total_features ?? 0} />
            <StatCard title="Top-N Used" value={featureSummary.topn ?? 0} />
            <StatCard title="TCN (Top-30)" value={featureSummary.top30_count?.tcn_embedding ?? 0} />
            <StatCard title="Late Settle (Top-30)" value={featureSummary.top30_count?.late_settle ?? 0} />
          </div>

          <div className="grid three compact-gap feature-metrics-grid">
            <StatCard title="TCN Importance" value={Number(featureSummary.group_sum?.tcn_embedding ?? 0).toFixed(4)} />
            <StatCard title="Late Settle" value={Number(featureSummary.group_sum?.late_settle ?? 0).toFixed(4)} />
            <StatCard title="Other" value={Number(featureSummary.group_sum?.handcrafted_other ?? 0).toFixed(4)} />
          </div>

          <details className="artifact-disclosure">
            <summary>
              <span>Feature Artifacts</span>
              <strong>Exported backend plots for model audit</strong>
            </summary>
            <div className="plot-dashboard feature-plots-grid">
              <PlotCard
                title="Feature Importance"
                eyebrow="Ranking"
                description="Most influential individual features."
                imageUrl={toFileUrl(trainJob.result?.plots?.feature_importance)}
                variant="hero"
              />
              <PlotCard
                title="Feature Group"
                eyebrow="Families"
                description="Contribution by engineered signal family."
                imageUrl={toFileUrl(trainJob.result?.plots?.feature_group)}
              />
              <PlotCard
                title="Feature Count"
                eyebrow="Coverage"
                description="Feature family presence in the selected top-N set."
                imageUrl={toFileUrl(trainJob.result?.plots?.feature_count)}
              />
            </div>
          </details>

          <div className="insight-strip">
            <Sparkles size={16} />
            <span>Hybrid model detected: the model combines waveform shape representation with timing behavior.</span>
          </div>
        </>
      ) : (
        <div className="empty-state analytics-empty">
          <Layers3 size={28} />
          <strong>No feature analysis available</strong>
          <span>Feature attribution will appear after a completed training run with exported analysis plots.</span>
        </div>
      )}
    </section>
  );
}
