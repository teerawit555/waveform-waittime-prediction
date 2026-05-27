import { Activity, Database, FileUp, GitCompareArrows, Rocket, ShieldCheck, Sparkles } from 'lucide-react';
import MLPipelineWorkflow from './MLPipelineWorkflow';
import { HeroWaveformGraphic } from './HeroWaveformGraphic';

const lifecycleSteps = [
  {
    title: 'Dataset',
    body: 'Validate waveform schema, reuse split directories, or upload controlled training data.',
    icon: FileUp,
  },
  {
    title: 'Model Setup',
    body: 'Choose TCN strategy, presets, augmentation, and run naming before compute starts.',
    icon: GitCompareArrows,
  },
  {
    title: 'Train',
    body: 'Run TCN embeddings and AutoGluon regression with progress and MLflow run context.',
    icon: Activity,
  },
  {
    title: 'Evaluate',
    body: 'Review MAE, RMSE, fast-class metrics, residuals, waveform plots, and artifacts.',
    icon: ShieldCheck,
  },
  {
    title: 'Registry',
    body: 'Compare versions, identify candidates, and keep prediction users on the best model.',
    icon: Database,
  },
];

const explanationCards = [
  {
    title: 'Evaluate',
    body: 'Metrics and plots exist to explain model behavior before a run becomes a candidate.',
  },
  {
    title: 'Monitor',
    body: 'Prediction and training states stay visible through jobs, artifacts, registry metadata, and waveform galleries.',
  },
  {
    title: 'Deploy',
    body: 'The default NS 1.3 route gives API users one stable inference path while admins manage model changes.',
  },
  {
    title: 'Access',
    body: 'Prediction stays open for users; training, workflow operations, and registry review sit behind admin mode.',
  },
];

export default function WorkflowSection() {
  return (
    <section className="workflow-page">
      <MLPipelineWorkflow />

      <div className="workflow-lifecycle">
        <div className="workflow-lifecycle-head">
          <Rocket size={18} />
          <div>
            <span className="view-kicker">ML Lifecycle</span>
            <h3>From waveform dataset to monitored prediction gateway</h3>
          </div>
        </div>
        <div className="workflow-lifecycle-grid">
          {lifecycleSteps.map((step, index) => {
            const Icon = step.icon;
            return (
              <article key={step.title}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <Icon size={18} />
                <strong>{step.title}</strong>
                <p>{step.body}</p>
              </article>
            );
          })}
        </div>
      </div>

      <div className="workflow-explanation-split">
        <div className="workflow-explanation">
          <div className="workflow-explanation-intro">
            <Sparkles size={18} />
            <div>
              <span className="view-kicker">Workflow Notes</span>
              <h3>How this project turns waveform signals into reliable wait-time predictions</h3>
            </div>
          </div>
          <div className="workflow-explanation-grid">
            {explanationCards.map((card) => (
              <article key={card.title}>
                <strong>{card.title}</strong>
                <span>{card.body}</span>
              </article>
            ))}
          </div>
        </div>

        <div className="workflow-graphic-panel">
          <div className="graphic-panel-head">
            <span className="view-kicker">Signal Visualizer</span>
            <h4>Real-time Adaptive Wait-time Interpolation</h4>
          </div>
          <div className="graphic-canvas-wrapper">
            <HeroWaveformGraphic />
          </div>
        </div>
      </div>
    </section>
  );
}
