import React from "react";
import { CheckCircle2, AlertTriangle, XCircle, Info } from 'lucide-react';
import { formatMetric, formatSignedMetric } from "../utils/metric";

export type OverfittingSummary = {
  status?: "good" | "mild" | "strong" | "unknown" | string;
  label?: string | null;
  best_epoch?: number | null;
  train_loss_best?: number | null;
  val_loss_best?: number | null;
  final_train_loss?: number | null;
  final_val_loss?: number | null;
  gap_best?: number | null;
  gap_final?: number | null;
  val_rise_after_best?: number | null;
  message?: string | null;
};

type OverfittingCardProps = {
  summary?: OverfittingSummary | null;
};

const StatusIcon = ({ status }: { status?: string }) => {
  switch (status) {
    case "good": return <CheckCircle2 size={16} className="text-success" />;
    case "mild": return <AlertTriangle size={16} className="text-warn" />;
    case "strong": return <XCircle size={16} className="text-error" />;
    default: return <Info size={16} />;
  }
};

export default function OverfittingCard({ summary }: OverfittingCardProps) {
  if (!summary) {
    return (
      <section className="plot-card overfit-card-match">
        <div className="plot-title">Overfitting Check</div>
        <div className="plot-empty">No overfitting summary available.</div>
      </section>
    );
  }

  return (
    <section className="plot-card overfit-card-match">
      <div className="plot-title">TCN Overfitting Check</div>

      {/* Status Monitor Section */}
      <div className={`overfit-status-monitor ${summary.status || ""}`}>
        <div className="status-info">
          <span className="status-led"></span>
          <div>
            <div className="overfit-status-label">Status</div>
            <div className="overfit-status-value">{summary.label || "N/A"}</div>
          </div>
        </div>
        <StatusIcon status={summary.status} />
      </div>

      {/* Metrics Grid Section */}
      <div className="overfit-metrics-grid">
        <div className="overfit-metric-item">
          <div className="metric-label">Best Epoch</div>
          <div className="metric-value">{summary.best_epoch ?? "0"}</div>
        </div>
        <div className="overfit-metric-item" style={{ borderLeftColor: 'var(--sky)' }}>
          <div className="metric-label">Final Gap</div>
          <div className="metric-value">{formatSignedMetric(summary.gap_final)}</div>
        </div>
        <div className="overfit-metric-item">
          <div className="metric-label">Train @ Best</div>
          <div className="metric-value">{formatMetric(summary.train_loss_best)}</div>
        </div>
        <div className="overfit-metric-item">
          <div className="metric-label">Val @ Best</div>
          <div className="metric-value">{formatMetric(summary.val_loss_best)}</div>
        </div>
        <div className="overfit-metric-item">
          <div className="metric-label">Final Train</div>
          <div className="metric-value">{formatMetric(summary.final_train_loss)}</div>
        </div>
        <div className="overfit-metric-item">
          <div className="metric-label">Final Val</div>
          <div className="metric-value">{formatMetric(summary.final_val_loss)}</div>
        </div>
      </div>

      {/* Interpretation Section */}
      <div className="overfit-interpretation">
        <div className="interpretation-header">
          <Info size={14} />
          <span>Analysis Manifest</span>
        </div>
        <div className="muted small" style={{ lineHeight: 1.5, color: 'var(--navy)' }}>
          {summary.message || "No automated interpretation generated for this model run."}
        </div>
      </div>
    </section>
  );
}