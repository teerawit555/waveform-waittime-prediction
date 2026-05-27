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
            {/* Input to Feature Engineering (Top path): orthogonal rounded corner, starting from slanted edge */}
            <path d="M131 180 H144 V93 Q144 85, 152 85 H195" />
            {/* Input to Wave Tensor (Bottom path): orthogonal rounded corner, starting from slanted edge */}
            <path d="M131 180 H144 V267 Q144 275, 152 275 H170" />
            {/* Feature Engineering to Feature Concatenation (Top path down): orthogonal rounded corner */}
            <path d="M359 85 H575 Q583 85, 583 93 V145" />
            {/* Wave Tensor to TCN Encoder */}
            <path d="M316 275 H350" />
            {/* TCN Encoder to TCN Embedding */}
            <path d="M478 275 H510" />
            {/* TCN Embedding to Feature Concatenation */}
            <path d="M583 240 V215" />
            {/* Feature Concatenation to AutoGluon Tabular */}
            <path d="M656 180 H688" />
            {/* AutoGluon Tabular to Output */}
            <path d="M816 180 H838" />
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
