type WorkflowNodeProps = {
  className: string;
  title: string;
  subtitle?: string;
  label?: string;
  slanted?: boolean;
  model?: boolean;
};

function WorkflowNode({ className, title, subtitle, label, slanted = false, model = false }: WorkflowNodeProps) {
  return (
    <div className={`ml-workflow-node ${className}${slanted ? ' is-slanted' : ''}${model ? ' is-model' : ''}`}>
      {label ? <span className="ml-workflow-label">{label}</span> : null}
      <strong>{title}</strong>
      {subtitle ? <span>{subtitle}</span> : null}
    </div>
  );
}

export default function MLPipelineWorkflow() {
  return (
    <section className="ml-pipeline-card" aria-labelledby="ml-pipeline-title">
      <div className="ml-pipeline-header">
        <div>
          <span className="view-kicker">Method Map</span>
          <h3 id="ml-pipeline-title">Project workflow</h3>
        </div>
        <div className="ml-pipeline-summary">
          <span>Hybrid model pipeline</span>
          <strong>Waveform + handcrafted features + TCN embedding + AutoGluon</strong>
        </div>
      </div>

      <div className="ml-pipeline-scroll">
        <div className="ml-workflow-canvas" aria-label="Machine learning pipeline workflow">
          {/* <span className="ml-workflow-lane-title ml-workflow-lane-feature-title">Handcrafted Feature Branch</span>
          <span className="ml-workflow-lane-title ml-workflow-lane-tcn-title">TCN Deep Learning Embedding Branch</span> */}

          <svg className="ml-workflow-connectors" viewBox="0 0 940 360" preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <marker id="workflow-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
                <path className="ml-workflow-arrow-head" d="M1.3 1.4 L7.8 4.5 L1.3 7.6" />
              </marker>
            </defs>
            <path d="M136 155 C166 155 160 83 195 83" />
            <path d="M136 198 C166 198 138 303 170 303" />
            <path d="M359 83 C470 83 583 88 583 128" />
            <path d="M316 303 H350" />
            <path d="M478 303 H510" />
            <path d="M583 268 V202" />
            <path d="M656 163 H688" />
            <path d="M816 163 H838" />
          </svg>

          <span className="ml-workflow-edge-label ml-workflow-label-feature-csv">.csv</span>
          <span className="ml-workflow-edge-label ml-workflow-label-embed-csv">.csv</span>
          <span className="ml-workflow-edge-label ml-workflow-label-augment">Augment: noise / scale / shift</span>

          <WorkflowNode
            className="ml-workflow-input"
            title="Waveform"
            subtitle="1000 samples"
            slanted
          />
          <WorkflowNode
            className="ml-workflow-feature"
            title="Feature Engineering"
            subtitle="handcrafted features"
          />
          <WorkflowNode
            className="ml-workflow-tensor"
            title="Wave Tensor"
            subtitle=".npz artifact"
          />
          <WorkflowNode
            className="ml-workflow-encoder"
            title="TCN Encoder"
            label="Deep Learning"
            model
          />
          <WorkflowNode
            className="ml-workflow-embedding"
            title="TCN Embedding"
            subtitle="features"
          />
          <WorkflowNode
            className="ml-workflow-merge"
            title="Feature Concatenation"
            subtitle="merge"
          />
          <WorkflowNode
            className="ml-workflow-autogluon"
            title="AutoGluon Tabular"
            label="Machine Learning"
            model
          />
          <WorkflowNode
            className="ml-workflow-output"
            title="Predict wait time (ms)"
            slanted
          />
        </div>
      </div>
    </section>
  );
}
