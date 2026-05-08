import { ChangeEvent, useEffect, useMemo, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Activity, BrainCircuit, FileUp, Search, Sparkles, Waves, X } from 'lucide-react';
import { getJob, getModels, startPredict, startTrain, toFileUrl, uploadFile, UploadResponse, API_BASE, ModelItem } from './lib/api';
import OverfittingCard from './components/OverfittingCard';
import StatCard from './components/StatCard';
import PlotCard from './components/PlotCard';
import DataTable from './components/DataTable';

function App() {
  const [dataset, setDataset] = useState<UploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [targetCol, setTargetCol] = useState('wait_time_ms');
  const [idCol, setIdCol] = useState('wave_id');
  const [wavePrefix, setWavePrefix] = useState('wave_');
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.001);
  const [embeddingDim, setEmbeddingDim] = useState(64);
  const [fastMs, setFastMs] = useState(0.1);
  const [trainJobId, setTrainJobId] = useState<string | null>(
    () => localStorage.getItem('trainJobId'));
  const [trainJob, setTrainJob] = useState<any>(null);
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predictUpload, setPredictUpload] = useState<UploadResponse | null>(null);
  const [predictJobId, setPredictJobId] = useState<string | null>(
    () => localStorage.getItem('predictJobId'));
  const [predictJob, setPredictJob] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [gallerySearch, setGallerySearch] = useState('');
  const [searchedItem, setSearchedItem] = useState<any>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [modelName, setModelName] = useState('wave_model_v1');
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selectedModel, setSelectedModel] = useState(''); // predict
  const [selectedTrainModel, setSelectedTrainModel] = useState(''); // train
  const [modelsLoading, setModelsLoading] = useState(false);
  // Select model version 
  const [trainNewTCN, setTrainNewTCN] = useState(true);
  const [tcnModels, setTcnModels] = useState<{ name: string; path: string; ready: boolean }[]>([]);
  const [selectedTCNModel, setSelectedTCNModel] = useState('');
  const [tcnModelsLoading, setTcnModelsLoading] = useState(false);
  // features 
  const featureSummary = trainJob?.result?.feature_summary;
  const overfittingSummary = trainJob?.result?.overfitting_summary;

  // Load model when open web
  const fetchModels = async () => {
    try {
      setModelsLoading(true);
      const res = await getModels();
      const readyModels = (res.models || []).filter((m) => m.ready);
      setModels(readyModels);

      if (!selectedModel && readyModels.length > 0) {
        setSelectedModel(readyModels[0].name);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setModelsLoading(false);
    }
  };

  const fetchTCNModels = async () => {
    try {
      setTcnModelsLoading(true);
      const res = await fetch(`${API_BASE}/tcn-models`);
      const json = await res.json();

      const readyModels = (json.data || []).filter((m: any) => m.ready);
      setTcnModels(readyModels);

      if (!selectedTCNModel && readyModels.length > 0) {
        setSelectedTCNModel(readyModels[0].name);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setTcnModelsLoading(false);
    }
  };

  const loadTrainResult = async (modelName: string) => {
    if (!modelName) {
      setTrainJob(null);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/models/${modelName}`);
      const data = await res.json();
      setTrainJob({
        status: 'completed',
        progress: 100,
        message: `Loaded model: ${modelName}`,
        result: data.result,
      });
    } catch (err: any) {
      setError(err.message);
    }
  };

useEffect(() => {
  fetchModels();
  fetchTCNModels();
}, []);

  const onDrop = async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;
    try {
      setError(null);
      setUploading(true);
      const res = await uploadFile(file);
      setDataset(res);
      if (res.columns.includes('wait_time_ms')) setTargetCol('wait_time_ms');
      if (res.columns.includes('wave_id')) setIdCol('wave_id');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
  });

  useEffect(() => {
    if (!trainJobId) return;
    const interval = setInterval(async () => {
      try {
        const job = await getJob(trainJobId);
        setTrainJob(job);
        if (job.status === 'completed' || job.status === 'failed') clearInterval(interval);
      } catch (err: any) {
        clearInterval(interval);
        setTrainJobId(null);
        localStorage.removeItem('trainJobId');
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [trainJobId]);

  useEffect(() => {
    if (!predictJobId) return;
    const interval = setInterval(async () => {
      try {
        const job = await getJob(predictJobId);
        setPredictJob(job);
        if (job.status === 'completed' || job.status === 'failed') clearInterval(interval);
      } catch (err: any) {
        clearInterval(interval);
        setPredictJobId(null);
        localStorage.removeItem('predictJobId');
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [predictJobId]);

  useEffect(() => {
    if (trainJobId) localStorage.setItem('trainJobId', trainJobId);
    else localStorage.removeItem('trainJobId');
  }, [trainJobId]);

  useEffect(() => {
    if (predictJobId) localStorage.setItem('predictJobId', predictJobId);
    else localStorage.removeItem('predictJobId');
  }, [predictJobId]);

  const startTraining = async () => {
    if (!dataset) return;

    if (!modelName.trim()) {
      setError('Please enter model name');
      return;
    }

    if (!trainNewTCN && !selectedTCNModel) {
      setError('Please select an existing TCN model');
      return;
    }

    try {
      setError(null);
      setTrainJob(null);

      const res = await startTrain({
        dataset_path: dataset.dataset_path,
        target_col: targetCol,
        id_col: idCol,
        wave_prefix: wavePrefix,
        epochs,
        batch_size: batchSize,
        lr: learningRate,
        embedding_dim: embeddingDim,
        fast_ms: fastMs,
        model_name: modelName.trim(),

        train_new_tcn: trainNewTCN,
        existing_tcn_name: trainNewTCN ? null : selectedTCNModel,
      });

      setTrainJobId(res.job_id);
    } catch (err: any) {
      setError(err.message);
    }
  };

  useEffect(() => {
    if (trainJob?.status === 'completed') {
      fetchModels();
      fetchTCNModels();

      if (trainJob?.result?.ag_model) {
        setSelectedModel(trainJob.result.ag_model);
      }
    }
  }, [trainJob]);

  const handlePredictFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      setError(null);
      setPredictFile(file);
      const res = await uploadFile(file);
      setPredictUpload(res);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const runPredict = async () => {
    if (!predictUpload) return;
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    try {
      setError(null);
      setPredictJob(null);

      const res = await startPredict({
        dataset_path: predictUpload.dataset_path,
        id_col: idCol,
        wave_prefix: wavePrefix,
        model_name: selectedModel,
      });

      setPredictJobId(res.job_id);
    } catch (err: any) {
      setError(err.message);
    }
  };

  // ── derived data ──────────────────────────────────────────
  const trainMetrics   = trainJob?.result?.metrics ?? {};
  const predictPreview = predictJob?.result?.preview_predictions ?? [];
  const analysisItems  = predictJob?.result?.analysis_manifest
                      ?? trainJob?.result?.analysis_manifest
                      ?? [];

  // total จาก backend (3000) — ถ้าไม่มีก็ใช้จำนวน items ที่มี (30)
  const totalWaves: number = predictJob?.result?.total_waves
                          ?? trainJob?.result?.total_waves
                          ?? analysisItems.length;

  const activeJobId: string | null = useMemo(() => {
    // ลอง predictJobId ก่อนเลย — มีอยู่แน่ถ้ายังไม่ refresh
    if (predictJobId) return predictJobId;
    // fallback: ดึงจาก image URL กรณี refresh หน้า
    const first = analysisItems[0];
    if (!first?.image) return null;
    const parts = String(first.image).split('/');
    const idx = parts.indexOf('plots');
    if (idx !== -1 && parts[idx + 1]) return parts[idx + 1];
    return null;
  }, [analysisItems, predictJobId]);

  // 30 แรก เรียง 1, 2, 3 ... ตาม numeric sort
  const displayedAnalysis = useMemo(() => {
    return [...analysisItems].sort((a: any, b: any) => {
      const numA = parseInt(String(a.wave_id ?? '').replace(/\D/g, ''), 10);
      const numB = parseInt(String(b.wave_id ?? '').replace(/\D/g, ''), 10);
      if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
      return String(a.wave_id ?? '').localeCompare(String(b.wave_id ?? ''));
    });
  }, [analysisItems]);

  const trainHistory = useMemo(() => {
    const history = trainJob?.result?.history;
    if (!history) return [];
    return history.train_loss.map((_: number, index: number) => ({
      epoch: index + 1,
      train_loss: history.train_loss[index],
      val_loss: history.val_loss[index],
    }));
  }, [trainJob]);

  // ── on-demand search ──

  const handleWaveSearch = async () => {
    const q = gallerySearch.trim();
    if (!q || !predictJobId) return;

    setSearchLoading(true);
    setSearchError(null);
    setSearchedItem(null);

    try {
      const res = await fetch(`${API_BASE}/plot-wave`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wave_id: q, job_id: predictJobId }),  // ใช้ predictJobId ตรงๆ
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Not found: ${q}`);
      }
      const data = await res.json();
      setSearchedItem(data);
    } catch (err: any) {
      setSearchError(err.message);
    } finally {
      setSearchLoading(false);
    }
  };

  const clearSearch = () => {
    setGallerySearch('');
    setSearchedItem(null);
    setSearchError(null);
  };

  // ─────────────────────────────────────────────────────────
  return (
    <div className="page-shell">
      <header className="hero">
        <div className="hero-copy">
          <div className="badge">
            <Sparkles size={16} />
            <span>TCN + AutoGluon Regression Pipeline</span>
          </div>
          <h1>TTR Tool AI QOET</h1>
          <p className="hero-text">
            Upload CSV files, train your waveform model, run predictions, inspect metrics,
            and browse waveform analysis images from one clean dashboard.
          </p>
          <div className="hero-tags">
            <span>TCN Embeddings</span>
            <span>AutoGluon</span>
            <span>Prediction Dashboard</span>
            <span>Waveform Gallery</span>
          </div>
        </div>
        <div className="hero-orb-wrap">
          <div className="hero-orb" />
          <div className="hero-orb hero-orb-small" />
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="grid two">
        <section className="card tall-card">
          <label style={{ marginBottom: 16 }}>
            <span>Load Existing Model</span>
            <select
              value={selectedTrainModel}
              onChange={(e) => {
                setSelectedTrainModel(e.target.value);
                loadTrainResult(e.target.value);
              }}
              disabled={modelsLoading || models.length === 0}
            >
              <option value="">— Train new model —</option>
              {models.map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </label>
          <div className="section-header">
            <div className="section-title"><FileUp size={18} /><span>Training Dataset</span></div>
          </div>
          <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
            <input {...getInputProps()} />
            <Waves size={30} />
            <div className="dropzone-title">
              {uploading ? 'Uploading dataset...' : 'Drag and drop a CSV file here'}
            </div>
            <small>Or click to browse. Expected waveform columns: wave_0 to wave_999</small>
          </div>
          {dataset ? (
            <>
              <div className="grid four compact-gap">
                <StatCard title="Rows" value={dataset.shape[0].toLocaleString()} />
                <StatCard title="Columns" value={dataset.shape[1]} />
                <StatCard title="Waves"   value={dataset.wave_count.toLocaleString() ?? 0} />
                <StatCard title="Samples" value={dataset.sample_count ?? 0} />
              </div>
              <div className="form-grid">
                <label>
                  <span>Training Mode</span>
                  <select
                    value={trainNewTCN ? 'new' : 'existing'}
                    onChange={(e) => {
                      const isNew = e.target.value === 'new';
                      setTrainNewTCN(isNew);
                      setError(null);
                      if (isNew) {
                        setSelectedTCNModel('');
                      }
                    }}
                  >
                    <option value="new">Train new TCN + AutoGluon</option>
                    <option value="existing">Use existing TCN, train AutoGluon only</option>
                  </select>
                </label>

                {!trainNewTCN && (
                  <label>
                    <span>Existing TCN Model</span>
                    <select
                      value={selectedTCNModel}
                      onChange={(e) => setSelectedTCNModel(e.target.value)}
                      disabled={tcnModelsLoading || tcnModels.length === 0}
                    >
                      {tcnModels.length === 0 ? (
                        <option value="">
                          {tcnModelsLoading ? 'Loading...' : 'No TCN models found'}
                        </option>
                      ) : (
                        tcnModels.map((m) => (
                          <option key={m.name} value={m.name}>
                            {m.name}
                          </option>
                        ))
                      )}
                    </select>
                  </label>
                )}
                <label><span>Model Name</span>
                  <input
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    placeholder="e.g. wave_model_v2"
                  />
                </label>
                {/* <label><span>ID Column</span>
                  <select value={idCol} onChange={(e) => setIdCol(e.target.value)}>
                    {dataset.columns.map((c) => <option key={c}>{c}</option>)}
                  </select>
                </label>
                <label><span>Wave Prefix</span>
                  <input value={wavePrefix} onChange={(e) => setWavePrefix(e.target.value)} />
                </label> */}
                <label><span>Fast Threshold (ms)</span>
                  <input type="number" step="0.001" value={fastMs} onChange={(e) => setFastMs(Number(e.target.value))} />
                </label>
                <label><span>Epochs</span>
                  <input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
                </label>
                <label><span>Batch Size</span>
                  <input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
                </label>
                <label><span>Learning Rate</span>
                  <input type="number" step="0.0001" value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))} />
                </label>
                <label><span>Embedding Dimension</span>
                  <input type="number" value={embeddingDim} onChange={(e) => setEmbeddingDim(Number(e.target.value))} />
                </label>
              </div>
              <button className="primary-btn" onClick={startTraining}>
                <BrainCircuit size={16} /><span>Start Training</span>
              </button>
            </>
          ) : null}
        </section>

        <section className="card tall-card">
          <div className="section-header">
            <div className="section-title"><Activity size={18} /><span>Dataset Preview</span></div>
          </div>
          {dataset
            ? <DataTable rows={dataset.preview} />
            : <div className="empty-state">Upload a dataset to see a preview table here.</div>
          }
        </section>
      </div>

      <section className="card">
        <div className="section-header">
          <div className="section-title"><Activity size={18} /><span>Training Status</span></div>
        </div>
        {trainJob ? (
          <>
            <div className="progress-wrap">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${trainJob.progress ?? 0}%` }} />
              </div>
              <div className="muted small">{trainJob.message}</div>
            </div>
            {/* ── TCN Section ── */}
            <div className="section-divider">
              <span>TCN</span>
            </div>
            <div className="grid two compact-gap">
              <PlotCard
                title="Learning Curve"
                imageUrl={toFileUrl(trainJob.result?.plots?.learning_curve)}
              />
              <OverfittingCard summary={overfittingSummary} />
            </div>

            {/* ── AutoGluon Section ── */}
            <div className="section-divider">
              <span>AutoGluon</span>
            </div>
            <div className="grid four compact-gap">
              <StatCard title="MAE (All)"       value={Number(trainMetrics.mae_all        ?? 0).toFixed(6)} />
              <StatCard title="RMSE"            value={Number(trainMetrics.rmse           ?? 0).toFixed(6)} />
              <StatCard title="Fast Precision"  value={Number(trainMetrics.fast_precision ?? 0).toFixed(6)} />
              <StatCard title="Fast Recall"     value={Number(trainMetrics.fast_recall    ?? 0).toFixed(6)} />
            </div>
            <div className="grid two compact-gap">
              <PlotCard title="Loss Curve"          imageUrl={toFileUrl(trainJob.result?.plots?.loss_curve)} />
              <PlotCard title="Actual vs Predicted" imageUrl={toFileUrl(trainJob.result?.plots?.actual_vs_pred)} />
              <PlotCard title="Error Histogram"     imageUrl={toFileUrl(trainJob.result?.plots?.error_histogram)} />
              <PlotCard title="Target Distribution" imageUrl={toFileUrl(trainJob.result?.plots?.target_distribution)} />
            </div>

            {trainHistory.length ? (
              <div className="note-box">Training history detected: {trainHistory.length} epochs</div>
            ) : null}
            {trainJob.result?.results?.validation_predictions_csv ? (
              <a className="ghost-btn" href={toFileUrl(trainJob.result.results.validation_predictions_csv)} target="_blank" rel="noreferrer">
                Download Validation Predictions CSV
              </a>
            ) : null}
          </>
        ) : (
          <div className="empty-state">No training job has been started yet.</div>
        )}
      </section>
      {/* ── Feature Importance Analysis ── */}
      <section className="card">
        <div className="section-header">
          <div className="section-title">
            <Sparkles size={18} />
            <span>Feature Importance Analysis</span>
          </div>
        </div>

        {featureSummary ? (
          <>
            <div className="grid four compact-gap">
              <StatCard title="Total Features" value={featureSummary.total_features ?? 0}/>
              <StatCard title="Top-N Used" value={featureSummary.topn ?? 0}/>
              <StatCard title="TCN (Top-30)" value={featureSummary.top30_count?.tcn_embedding ?? 0}/>
              <StatCard title="Late Settle (Top-30)" value={featureSummary.top30_count?.late_settle ?? 0}/>
            </div>

            <div className="grid three compact-gap" style={{ marginTop: 12 }}>
              <StatCard title="TCN Importance" value={Number(featureSummary.group_sum?.tcn_embedding ?? 0).toFixed(4)}/>
              <StatCard title="Late Settle" value={Number(featureSummary.group_sum?.late_settle ?? 0).toFixed(4)}/>
              <StatCard title="Other" value={Number(featureSummary.group_sum?.handcrafted_other ?? 0).toFixed(4)}/>
            </div>

            {/* ── Plots ── */}
            <div className="grid two compact-gap " style={{ marginTop: 20 }}>
              <PlotCard title="Feature Importance" imageUrl={toFileUrl(trainJob.result?.plots?.feature_importance)} />
              <PlotCard title="Feature Group" imageUrl={toFileUrl(trainJob.result?.plots?.feature_group)} />
              <PlotCard title="Feature Count" imageUrl={toFileUrl(trainJob.result?.plots?.feature_count)} />
            </div>

            {/* ── Interpretation ── */}
            <div className="note-box">
              <div className="note-title">Hybrid model detected</div>
              <ul>
                <li>TCN embeddings dominate representation</li>
                <li>Late-settle features still influence prediction</li>
                <li>Model learns both shape + timing behavior</li>
              </ul>
            </div>
          </>
        ) : (
          <div className="empty-state">No feature analysis available yet.</div>
        )}
      </section>

      <div className="grid two">
        <section className="card tall-card">
          <div className="section-header">
            <div className="section-title"><BrainCircuit size={18} /><span>Predict New Dataset</span></div>
          </div>
          <label>
            <span>Select Model</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={modelsLoading || models.length === 0}
            >
              {models.length === 0 ? (
                <option value="">{modelsLoading ? 'Loading...' : 'No models found'}</option>
              ) : (
                models.map((m) => (
                  <option key={m.name} value={m.name}>{m.name}</option>
                ))
              )}
            </select>
          </label>
          <label className="upload-inline">
            <input type="file" accept=".csv" onChange={handlePredictFile} />
            <span>{predictFile ? predictFile.name : 'Choose a CSV file for prediction'}</span>
          </label>
          {predictUpload ? (
            <div className="muted">Prediction dataset ready: {predictUpload.shape[0]} rows</div>
          ) : null}
          <button className="primary-btn" onClick={runPredict} disabled={!predictUpload || !selectedModel}>
            <span>Run Prediction</span>
          </button>
          {predictJob ? (
            <div className="progress-wrap top-gap">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${predictJob.progress ?? 0}%` }} />
              </div>
              <div className="muted small">{predictJob.message}</div>
            </div>
          ) : null}
          {predictJob?.result?.predictions_csv ? (
            <a className="ghost-btn" href={toFileUrl(predictJob.result.predictions_csv)} target="_blank" rel="noreferrer">
              Download Prediction CSV
            </a>
          ) : null}
        </section>

        <section className="card tall-card">
          <div className="section-header">
            <div className="section-title"><Activity size={18} /><span>Prediction Preview</span></div>
          </div>
          {predictPreview.length
            ? <DataTable rows={predictPreview.slice(0, 20)} />
            : <div className="empty-state">Prediction results will appear here after the job finishes.</div>
          }
        </section>
      </div>

      <section className="card">
        <div className="section-header">
          <div className="section-title"><Waves size={18} /><span>Waveform Analysis Gallery</span></div>
        </div>

        <p className="gallery-text">
          Showing the first 30 waves by default. Search by wave_id to load any wave on demand.
        </p>

        {analysisItems.length > 0 && (
          <div className="gallery-search-wrap">
            <div className="gallery-search-input-wrap">
              <Search size={14} />
              <input
                className="gallery-search"
                type="text"
                placeholder="e.g. 1000"
                value={gallerySearch}
                onChange={(e) => { setGallerySearch(e.target.value); setSearchedItem(null); setSearchError(null); }}
                onKeyDown={(e) => { if (e.key === 'Enter') handleWaveSearch(); }}
              />
            </div>

            <button
              className="primary-btn"
              style={{ width: 'auto', padding: '9px 20px' }}
              onClick={handleWaveSearch}
              disabled={searchLoading || !gallerySearch.trim() || !predictJobId}
            >
              {searchLoading ? 'Loading…' : 'Search'}
            </button>

            {gallerySearch && (
              <button className="gallery-search-clear" onClick={clearSearch}>
                <X size={12} style={{ display: 'inline', marginRight: 4 }} />
                Clear
              </button>
            )}

            {/* แสดง totalWaves จาก backend (3000) ไม่ใช่ 30 */}
            <span className="gallery-count">
              <strong>{totalWaves}</strong> waves total
            </span>
          </div>
        )}

        {searchError && (
          <div className="error-banner" style={{ marginBottom: 16 }}>{searchError}</div>
        )}

        {searchedItem && (
          <div style={{ marginBottom: 24 }}>
            <div className="note-box" style={{ marginBottom: 12 }}>
              Search result for <strong>{searchedItem.wave_id}</strong>
            </div>
            <div className="analysis-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))' }}>
              <div className="analysis-card">
                <img src={toFileUrl(searchedItem.image)} alt={searchedItem.wave_id} />
                <div className="analysis-meta">
                  <strong>{searchedItem.wave_id}</strong>
                  <span>Pred: {Number(searchedItem.pred ?? 0).toFixed(6)}</span>
                  {searchedItem.true != null && (
                    <span>True: {Number(searchedItem.true).toFixed(6)}</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Default grid — 30 แรก เรียง 1, 2, 3 ... */}
        <div className="analysis-grid">
          {analysisItems.length === 0 ? (
            <div className="empty-state">No waveform analysis images are available yet.</div>
          ) : (
            displayedAnalysis.map((item: any) => (
              <div className="analysis-card" key={`${item.wave_id}-${item.image}`}>
                <img src={toFileUrl(item.image)} alt={item.wave_id} />
                <div className="analysis-meta">
                  <strong>{item.wave_id}</strong>
                  <span>Pred: {Number(item.pred ?? 0).toFixed(6)}</span>
                  {item.true != null && (
                    <span>True: {Number(item.true).toFixed(6)}</span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}

export default App;