import { useEffect, useState } from 'react';
import { getModelLeaderboard, LeaderboardEntry } from '../lib/api';

type Props = {
  modelName: string;
  adminToken?: string;
};

export default function ModelLeaderboard({ modelName, adminToken }: Props) {
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!modelName) return;
    setLoading(true);
    setError('');
    getModelLeaderboard(modelName, adminToken)
      .then(({ leaderboard, best_model }) => {
        setEntries(leaderboard);
        setActiveModel(best_model ?? leaderboard[0]?.model ?? null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [modelName, adminToken]);

  if (loading) return <div className="stat-card">Loading leaderboard...</div>;
  if (error) return <div className="stat-card" style={{ color: 'var(--color-error)' }}>{error}</div>;
  if (!entries.length) return null;

  return (
    <div className="table-shell">
      <h3 style={{ margin: '0 0 0.5rem' }}>Model Leaderboard</h3>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Model</th>
              <th>MAE (score)</th>
              <th>Val Runtime (s)</th>
              <th>Train Runtime (s)</th>
              <th>Stack Level</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((row, i) => (
              <tr
                key={row.model}
                style={row.model === activeModel ? { background: 'var(--color-accent-muted, rgba(99,102,241,0.12))' } : undefined}
              >
                <td>{i + 1}</td>
                <td>
                  {row.model}
                  {row.model === activeModel && <span style={{ marginLeft: 6, fontSize: '0.75rem', opacity: 0.7 }}>● active</span>}
                </td>
                <td>{Math.abs(row.score_val).toFixed(6)}</td>
                <td>{row.pred_time_val.toFixed(3)}</td>
                <td>{row.fit_time.toFixed(1)}</td>
                <td>{row.stack_level}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
