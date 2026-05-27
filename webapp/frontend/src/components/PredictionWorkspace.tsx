import { ChangeEvent, useMemo, useState } from 'react';
import { Activity, BrainCircuit, CheckCircle2, ChevronDown, Copy, Download, FileUp, Gauge, PlayCircle } from 'lucide-react';
import { API_BASE, ModelItem, toFileUrl, UploadResponse } from '../lib/api';
import { formatModelName, getModelNote } from '../lib/modelDisplay';
import DataTable from './DataTable';
import ProgressBar from './ProgressBar';

type PredictionWorkspaceProps = {
  models: ModelItem[];
  modelsLoading: boolean;
  selectedModel: string;
  setSelectedModel: (modelName: string) => void;
  predictFile: File | null;
  handlePredictFile: (event: ChangeEvent<HTMLInputElement>) => void;
  predictUpload: UploadResponse | null;
  runPredict: () => void;
  predictJob: any;
  predictPreview: any[];
};

type ApiExampleTab = 'request' | 'response';

function highlightCodeValue(value: string) {
  const pieces: JSX.Element[] = [];
  const tokenPattern = /(".*?"|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d+)?|[{}\[\],])/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenPattern.exec(value)) !== null) {
    if (match.index > cursor) {
      pieces.push(<span key={`plain-${cursor}`}>{value.slice(cursor, match.index)}</span>);
    }

    const token = match[0];
    let className = 'code-token code-punctuation';
    if (token.startsWith('"')) className = 'code-token code-string';
    else if (/^-?\d/.test(token)) className = 'code-token code-number';
    else if (token === 'true' || token === 'false' || token === 'null') className = 'code-token code-constant';

    pieces.push(<span key={`token-${match.index}`} className={className}>{token}</span>);
    cursor = match.index + token.length;
  }

  if (cursor < value.length) {
    pieces.push(<span key={`plain-${cursor}`}>{value.slice(cursor)}</span>);
  }

  return pieces;
}

function renderHighlightedApiExample(source: string, tab: ApiExampleTab) {
  return source.split('\n').map((line, index) => {
    if (!line) {
      return <span key={`line-${index}`} className="code-line">&nbsp;</span>;
    }

    if (tab === 'request' && index === 0) {
      const [method, ...endpointParts] = line.split(' ');
      return (
        <span key={`line-${index}`} className="code-line">
          <span className="code-token code-method">{method}</span>
          <span> </span>
          <span className="code-token code-url">{endpointParts.join(' ')}</span>
        </span>
      );
    }

    if (tab === 'request' && line.startsWith('Content-Type:')) {
      const [header, value] = line.split(': ');
      return (
        <span key={`line-${index}`} className="code-line">
          <span className="code-token code-header">{header}</span>
          <span className="code-token code-punctuation">: </span>
          <span className="code-token code-string">{value}</span>
        </span>
      );
    }

    const keyMatch = line.match(/^(\s*)("[^"]+")(:)(.*)$/);
    if (keyMatch) {
      return (
        <span key={`line-${index}`} className="code-line">
          <span>{keyMatch[1]}</span>
          <span className="code-token code-property">{keyMatch[2]}</span>
          <span className="code-token code-punctuation">{keyMatch[3]}</span>
          {highlightCodeValue(keyMatch[4])}
        </span>
      );
    }

    return (
      <span key={`line-${index}`} className="code-line">
        {highlightCodeValue(line)}
      </span>
    );
  });
}

export default function PredictionWorkspace({
  models,
  modelsLoading,
  selectedModel,
  setSelectedModel,
  predictFile,
  handlePredictFile,
  predictUpload,
  runPredict,
  predictJob,
  predictPreview,
}: PredictionWorkspaceProps) {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [apiExampleTab, setApiExampleTab] = useState<ApiExampleTab>('request');
  const openPredictionFilePicker = () => {
    document.getElementById('prediction-file-input')?.click();
  };
  const selectedModelNote = getModelNote(selectedModel);
  const selectedModelLabel = selectedModel ? formatModelName(selectedModel) : 'No model selected';
  const uploadedRows = predictUpload?.shape?.[0] ?? null;
  const uploadedColumns = predictUpload?.shape?.[1] ?? null;
  const jobStatus = predictJob?.status ?? 'ready';
  const isJobRunning = predictJob && !['completed', 'failed'].includes(predictJob.status);
  const hasResults = predictPreview.length > 0 || Boolean(predictJob?.result?.predictions_csv);
  const predictionEndpoint = `${API_BASE}/predict`;
  const requestExample = `POST ${predictionEndpoint}
Content-Type: application/json

{
  "upload_id": "${predictUpload?.upload_id ?? 'infer_12345.csv'}",
  "model_name": "${selectedModel || 'TCN_aug_weighted_v1'}"
}`;
  const responseExample = `{
  "job_id": "predict_...",
  "status": "queued",
  "result": {
    "predictions_csv": "/api/files/results/.../pred_1stage_hybrid.csv",
    "preview_predictions": []
  }
}`;
  const predictionStats = useMemo(() => {
    const candidateColumns = ['pred_wait_time_ms', 'pred_wait_time', 'prediction', 'pred', 'wait_time_ms'];
    const predColumn = candidateColumns.find((column) =>
      predictPreview.some((row) => Number.isFinite(Number(row?.[column]))),
    );

    if (!predColumn) {
      return {
        predColumn: 'prediction',
        average: null,
        min: null,
        max: null,
        count: predictPreview.length,
      };
    }

    const values = predictPreview
      .map((row) => Number(row?.[predColumn]))
      .filter((value) => Number.isFinite(value));

    if (!values.length) {
      return {
        predColumn,
        average: null,
        min: null,
        max: null,
        count: predictPreview.length,
      };
    }

    const total = values.reduce((sum, value) => sum + value, 0);
    return {
      predColumn,
      average: total / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length,
    };
  }, [predictPreview]);
  const formatMetric = (value: number | null) => (
    value == null ? 'N/A' : value.toLocaleString(undefined, { maximumFractionDigits: 4 })
  );
  const copyText = async (key: string, value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey(null), 1400);
    } catch {
      setCopiedKey(null);
    }
  };
  const steps = [
    {
      label: 'Model',
      description: 'Choose the inference model',
      complete: Boolean(selectedModel),
      active: !selectedModel,
    },
    {
      label: 'Upload',
      description: 'Select CSV input',
      complete: Boolean(predictUpload),
      active: Boolean(selectedModel) && !predictUpload,
    },
    {
      label: 'Run',
      description: 'Start prediction pipeline',
      complete: Boolean(predictJob),
      active: Boolean(predictUpload) && !predictJob,
    },
    {
      label: 'Review',
      description: 'Preview results and artifacts',
      complete: hasResults,
      active: Boolean(predictJob) && !hasResults,
    },
  ];

  return (
    <div className="prediction-workspace">
      <section className="prediction-flow" aria-label="Prediction workflow">
        <ol>
          {steps.map((step, index) => (
          <li
            key={step.label}
            className={`${step.complete ? 'is-complete' : ''} ${step.active ? 'is-active' : ''}`}
          >
            <span>{step.complete ? <CheckCircle2 size={13} /> : String(index + 1).padStart(2, '0')}</span>
            <div>
              <strong>{step.label}</strong>
              <small>{step.description}</small>
            </div>
          </li>
          ))}
        </ol>
      </section>

      <section className="card tall-card prediction-run-card">
          <div className="section-header">
            <div className="section-title"><BrainCircuit size={18} /><span>Predict New Dataset</span></div>
          </div>
          <div className="prediction-run-body">
            <div className="prediction-model-column">
              <label>
                <span>Select Model</span>
                <div className="select-shell">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    disabled={modelsLoading || models.length === 0}
                  >
                    {models.length === 0 ? (
                      <option value="">{modelsLoading ? 'Loading...' : 'No models found'}</option>
                    ) : (
                      models.map((m) => (
                        <option key={m.name} value={m.name}>{formatModelName(m.name)}</option>
                      ))
                    )}
                  </select>
                  <ChevronDown size={16} aria-hidden="true" />
                </div>
                {selectedModelNote ? <small className="model-note">{selectedModelNote}</small> : null}
              </label>
            </div>

            <div className="prediction-upload-column">
              <label className="upload-inline prediction-upload">
                <input id="prediction-file-input" type="file" accept=".csv,text/csv" onChange={handlePredictFile} />
                <span>
                  <FileUp size={18} />
                  {predictFile ? predictFile.name : 'Choose a CSV file for prediction'}
                </span>
              </label>
              {predictUpload ? (
                <div className="prediction-file-ready">
                  <CheckCircle2 size={15} />
                  <span>Dataset ready</span>
                  <strong>{uploadedRows?.toLocaleString()} rows{uploadedColumns ? ` / ${uploadedColumns} columns` : ''}</strong>
                </div>
              ) : null}
              <button className="primary-btn" onClick={runPredict} disabled={!predictUpload || !selectedModel || Boolean(isJobRunning)}>
                <PlayCircle size={16} />
                <span>{isJobRunning ? 'Running Prediction' : 'Run Prediction'}</span>
              </button>
              {predictJob ? (
                <ProgressBar
                  className="top-gap"
                  progress={predictJob.progress ?? 0}
                  message={predictJob.message}
                />
              ) : null}
              {predictJob?.result?.predictions_csv ? (
                <a className="ghost-btn" href={toFileUrl(predictJob.result.predictions_csv)} target="_blank" rel="noreferrer">
                  <Download size={15} />
                  Download Prediction CSV
                </a>
              ) : null}
            </div>
          </div>
      </section>

      <section className="prediction-docs-panel">
        <div className="prediction-docs-copy">
          <span className="view-kicker">API Docs</span>
          <h3>Prediction endpoint</h3>
          <p>
            Use the same inference contract from the console or client code. Upload data first,
            then start a prediction job and poll job status for artifacts.
          </p>

          <div className="prediction-endpoint-row">
            <span>POST</span>
            <code>/api/predict</code>
          </div>

          <dl className="prediction-docs-schema">
            <div>
              <dt>Input</dt>
              <dd>CSV waveform table</dd>
            </div>
            <div>
              <dt>Schema</dt>
              <dd>wave_id/sample/time_ms/value or Signal plus waveform columns</dd>
            </div>
            <div>
              <dt>Output</dt>
              <dd>Prediction CSV and waveform artifacts</dd>
            </div>
          </dl>
        </div>

        <div className="prediction-docs-example">
          <div className="prediction-docs-tabs">
            <div role="tablist" aria-label="Prediction API examples">
              <button
                type="button"
                className={apiExampleTab === 'request' ? 'is-active' : undefined}
                onClick={() => setApiExampleTab('request')}
              >
                Request
              </button>
              <button
                type="button"
                className={apiExampleTab === 'response' ? 'is-active' : undefined}
                onClick={() => setApiExampleTab('response')}
              >
                Response
              </button>
            </div>
            <button
              type="button"
              onClick={() => copyText(apiExampleTab, apiExampleTab === 'request' ? requestExample : responseExample)}
            >
              <Copy size={13} /> {copiedKey === apiExampleTab ? 'Copied' : 'Copy'}
            </button>
          </div>
          <pre>
            <code>{renderHighlightedApiExample(apiExampleTab === 'request' ? requestExample : responseExample, apiExampleTab)}</code>
          </pre>
        </div>
      </section>

      <section className="card prediction-preview-card">
        <div className="section-header">
          <div className="section-title"><Activity size={18} /><span>Prediction Preview</span></div>
          {hasResults ? (
            <div className="preview-count">{predictPreview.length} preview rows</div>
          ) : null}
        </div>

        {predictPreview.length
          ? (
            <>
              <div className="prediction-result-strip prediction-result-strip-wide">
                <div>
                  <span>Status</span>
                  <strong>{jobStatus}</strong>
                </div>
                <div>
                  <span>Model</span>
                  <strong>{selectedModelLabel}</strong>
                </div>
                <div>
                  <span>Rows</span>
                  <strong>{uploadedRows?.toLocaleString() ?? 'Ready'}</strong>
                </div>
                <div>
                  <span>Avg {predictionStats.predColumn}</span>
                  <strong>{formatMetric(predictionStats.average)}</strong>
                </div>
                <div>
                  <span>Min / Max</span>
                  <strong>{formatMetric(predictionStats.min)} / {formatMetric(predictionStats.max)}</strong>
                </div>
                <div>
                  <span>Preview Count</span>
                  <strong>{predictionStats.count.toLocaleString()}</strong>
                </div>
              </div>
              <div className="prediction-insight-note">
                <Gauge size={16} />
                <span>
                  Inspect the preview first, then use the CSV export and waveform gallery for row-level validation.
                </span>
              </div>
              <DataTable rows={predictPreview.slice(0, 20)} />
            </>
          )
          : (
            <div className="empty-state empty-action-state">
              <strong>No prediction results yet</strong>
              <span>Select a trained model, upload a CSV file, then run prediction to preview wait_time_ms outputs here.</span>
              <button type="button" className="empty-state-action" onClick={openPredictionFilePicker}>
                Choose prediction file
              </button>
            </div>
          )
        }
      </section>
    </div>
  );
}
