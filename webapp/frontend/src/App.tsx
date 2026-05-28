import { ChangeEvent, FormEvent, Suspense, lazy, useEffect, useMemo, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Activity, ArrowRight, BookOpen, BrainCircuit, CheckCircle2, Database, FileCode2, FileUp, Home, KeyRound, LineChart, Lock, LogOut, PlayCircle, Rocket, Server, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, TerminalSquare, Waves } from 'lucide-react';
import { deleteModel, getJob, getMlflowConfig, getMlflowModelRegistry, getModelAudit, getModels, startPredict, startTrain, uploadFile, UploadResponse, API_BASE, ModelItem, MlflowInfo, MlflowModelRegistry, MlflowModelVersion } from './lib/api';
import StatCard from './components/StatCard';
import DataTable from './components/DataTable';
import { DEFAULT_MODEL_NAME, formatModelName, getModelNote, shouldHideModel } from './lib/modelDisplay';
import { HeroWaveformGraphic } from './components/HeroWaveformGraphic';

const FeatureImportanceSection = lazy(() => import('./components/FeatureImportanceSection'));
const ModelRegistrySection = lazy(() => import('./components/ModelRegistrySection'));
const PredictionWorkspace = lazy(() => import('./components/PredictionWorkspace'));
const TrainingStatusSection = lazy(() => import('./components/TrainingStatusSection'));
const WaveformGallery = lazy(() => import('./components/WaveformGallery'));
const WorkflowSection = lazy(() => import('./components/WorkflowSection'));

type WorkspaceView = 'home' | 'prediction' | 'training' | 'registry' | 'workflow';
type TrainingPresetKey = 'fast' | 'balanced' | 'best';

const ADMIN_ONLY_VIEWS = new Set<WorkspaceView>(['registry']);
const workspaceViews: WorkspaceView[] = ['home', 'prediction', 'training', 'registry', 'workflow'];
const trainingPresets: Record<TrainingPresetKey, {
  title: string;
  description: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  embeddingDim: number;
  earlyStoppingPatience: number;
  timeLimit: number;
  agPresets: string;
  tcnAugment: boolean;
}> = {
  fast: {
    title: 'Fast Check',
    description: 'Short run for pipeline validation.',
    epochs: 12,
    batchSize: 96,
    learningRate: 0.0012,
    embeddingDim: 48,
    earlyStoppingPatience: 3,
    timeLimit: 120,
    agPresets: 'medium_quality',
    tcnAugment: false,
  },
  balanced: {
    title: 'Balanced',
    description: 'Default quality and runtime tradeoff.',
    epochs: 30,
    batchSize: 64,
    learningRate: 0.001,
    embeddingDim: 64,
    earlyStoppingPatience: 5,
    timeLimit: 300,
    agPresets: 'good_quality',
    tcnAugment: true,
  },
  best: {
    title: 'Best Quality',
    description: 'Longer run for candidate models.',
    epochs: 60,
    batchSize: 48,
    learningRate: 0.0008,
    embeddingDim: 96,
    earlyStoppingPatience: 8,
    timeLimit: 900,
    agPresets: 'best_quality',
    tcnAugment: true,
  },
};

function getInitialWorkspaceView(): WorkspaceView {
  const stored = localStorage.getItem('workspaceView') as WorkspaceView | null;
  return stored && workspaceViews.includes(stored) ? stored : 'home';
}

function getModelNameFromPath(path?: string | null): string {
  if (!path) return '';
  const parts = String(path).split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || '';
}

function getRegistryVersionModelName(version?: MlflowModelVersion | null) {
  return version?.tags?.ag_model_name || version?.version_tags?.ag_model_name || version?.run_name || '';
}

function App() {
  const [activeView, setActiveView] = useState<WorkspaceView>(getInitialWorkspaceView);
  const [dataset, setDataset] = useState<UploadResponse | null>(null);
  const [trainingSource, setTrainingSource] = useState<'upload' | 'split'>('split');
  const [existingSplitDir, setExistingSplitDir] = useState('data/split_noise_10000/splits');
  const [uploading, setUploading] = useState(false);
  const [targetCol, setTargetCol] = useState('wait_time_ms');
  const [idCol, setIdCol] = useState('wave_id');
  const [wavePrefix, setWavePrefix] = useState('wave_');
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.001);
  const [embeddingDim, setEmbeddingDim] = useState(64);
  const [fastMs, setFastMs] = useState(0.1);
  const [fastWeight, setFastWeight] = useState(1);
  const [timeLimit, setTimeLimit] = useState(300);
  const [agPresets, setAgPresets] = useState('medium_quality');
  const [tcnAugment, setTcnAugment] = useState(false);
  const [tcnNoiseStd, setTcnNoiseStd] = useState(0.015);
  const [tcnScaleJitter, setTcnScaleJitter] = useState(0.04);
  const [tcnTimeShift, setTcnTimeShift] = useState(8);
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(5);
  const [trainingPreset, setTrainingPreset] = useState<TrainingPresetKey>('balanced');
  const [trainJobId, setTrainJobId] = useState<string | null>(
    () => localStorage.getItem('trainJobId'));
  const [trainJob, setTrainJob] = useState<any>(null);
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predictUpload, setPredictUpload] = useState<UploadResponse | null>(null);
  const [predictJobId, setPredictJobId] = useState<string | null>(
    () => localStorage.getItem('predictJobId'));
  const [predictJob, setPredictJob] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [adminToken, setAdminToken] = useState(() => localStorage.getItem('neurosettleAdminToken') || '');
  const [adminTokenInput, setAdminTokenInput] = useState('');
  const [adminLoginError, setAdminLoginError] = useState<string | null>(null);
  const [isAdminValidating, setIsAdminValidating] = useState(false);
  const [isAdminUnlocked, setIsAdminUnlocked] = useState(() => Boolean(localStorage.getItem('neurosettleAdminToken')));
  const [gallerySearch, setGallerySearch] = useState('');
  const [searchedItem, setSearchedItem] = useState<any>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [modelName, setModelName] = useState('wave_model_v1');
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selectedModel, setSelectedModel] = useState(''); // predict
  const [selectedTrainModel, setSelectedTrainModel] = useState(''); // train
  const [modelsLoading, setModelsLoading] = useState(false);
  const [isTrainingEnabled, setIsTrainingEnabled] = useState(true);
  // Select model version 
  const [trainNewTCN, setTrainNewTCN] = useState(true);
  const [tcnModels, setTcnModels] = useState<{ name: string; path: string; ready: boolean }[]>([]);
  const [selectedTCNModel, setSelectedTCNModel] = useState('');
  const [tcnModelsLoading, setTcnModelsLoading] = useState(false);
  const [mlflowConfig, setMlflowConfig] = useState<MlflowInfo | null>(null);
  const [modelRegistry, setModelRegistry] = useState<MlflowModelRegistry | null>(null);
  const [modelRegistryLoading, setModelRegistryLoading] = useState(false);
  // features 
  const featureSummary = trainJob?.result?.feature_summary;
  const overfittingSummary = trainJob?.result?.overfitting_summary;
  const jobMlflowInfo = trainJob?.result?.mlflow;
  const mlflowInfo = (jobMlflowInfo?.run_id || jobMlflowInfo?.latest_run_id)
    ? jobMlflowInfo
    : (mlflowConfig ?? jobMlflowInfo);
  const readyModelNames = useMemo(() => new Set(models.map((model) => model.name)), [models]);
  const currentRegistryVersions = useMemo(() => {
    return (modelRegistry?.versions ?? []).filter((version) => {
      const modelName = getRegistryVersionModelName(version);
      return modelName && readyModelNames.has(modelName) && !shouldHideModel(modelName);
    });
  }, [modelRegistry, readyModelNames]);
  const canConfigureTraining = trainingSource === 'split' || !!dataset;
  const isTerminalJob = (job: any) => job?.status === 'completed' || job?.status === 'failed';

  // Load model when open web
  const fetchModels = async () => {
    try {
      setModelsLoading(true);
      const res = await getModels(isAdminUnlocked ? adminToken : undefined);
      
      const trainingEnabled = res.training_enabled !== undefined ? res.training_enabled : true;
      setIsTrainingEnabled(trainingEnabled);

      const readyModels = (res.models || [])
        .filter((m) => m.ready && !shouldHideModel(m.name))
        .sort((a, b) => {
          if (a.name === DEFAULT_MODEL_NAME) return -1;
          if (b.name === DEFAULT_MODEL_NAME) return 1;
          return formatModelName(a.name).localeCompare(formatModelName(b.name));
        });
      setModels(readyModels);

      setSelectedModel((current) => {
        if (current && readyModels.some((model) => model.name === current)) return current;
        return readyModels.find((model) => model.name === DEFAULT_MODEL_NAME)?.name ?? readyModels[0]?.name ?? '';
      });

      // Auto-redirect if on training page but training is disabled
      if (isAdminUnlocked && !trainingEnabled && activeView === 'training') {
        setActiveView('registry');
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
      const res = await fetch(`${API_BASE}/tcn-models`, {
        headers: adminToken ? { 'X-Admin-Token': adminToken } : undefined,
      });
      if (!res.ok) {
        if (res.status === 401) {
          lockAdminConsole();
          throw new Error('Your admin session has expired or is invalid. Please log in again.');
        }
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to fetch TCN models');
      }
      const json = await res.json();

      const readyModels = (json.data || []).filter((m: any) => m.ready && !shouldHideModel(m.name));
      setTcnModels(readyModels);

      setSelectedTCNModel((current) => {
        if (current && readyModels.some((model: any) => model.name === current)) return current;
        return readyModels[0]?.name ?? '';
      });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setTcnModelsLoading(false);
    }
  };

  const fetchMlflowConfig = async () => {
    try {
      const res = await getMlflowConfig();
      setMlflowConfig(res);
    } catch (err: any) {
      setMlflowConfig({
        enabled: false,
        tracking_uri: '',
        experiment_name: 'adaptive-wait-time',
        reason: err.message,
      });
    }
  };

  const fetchModelRegistry = async () => {
    try {
      setModelRegistryLoading(true);
      const res = await getMlflowModelRegistry(isAdminUnlocked ? adminToken : undefined);
      setModelRegistry(res);
    } catch (err: any) {
      if (err.message?.includes('Admin access required')) {
        lockAdminConsole();
        setError('Your admin session has expired or is invalid. Please log in again.');
      } else {
        setModelRegistry({
          enabled: false,
          tracking_uri: '',
          experiment_name: 'adaptive-wait-time',
          registered_model_name: mlflowConfig?.registered_model_name || 'adaptive_wait_time_hybrid',
          versions: [],
          reason: err.message,
        });
      }
    } finally {
      setModelRegistryLoading(false);
    }
  };

  const loadTrainResult = async (modelName: string) => {
    if (!modelName) {
      setTrainJob(null);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/models/${modelName}`, {
        headers: adminToken ? { 'X-Admin-Token': adminToken } : undefined,
      });
      if (!res.ok) {
        if (res.status === 401) {
          lockAdminConsole();
          throw new Error('Your admin session has expired or is invalid. Please log in again.');
        }
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to load model');
      }
      const data = await res.json();
      const result = data.result ?? {};
      let history = result.history;

      if (!history?.length) {
        const tcnModelName = data.tcn_name || getModelNameFromPath(data.tcn_path);
        if (tcnModelName) {
          const historyRes = await fetch(`${API_BASE}/files/tcn/${encodeURIComponent(tcnModelName)}/train_history.json`, {
            headers: adminToken ? { 'X-Admin-Token': adminToken } : undefined,
          });
          if (historyRes.ok) {
            history = await historyRes.json();
          }
        }
      }

      setTrainJob({
        status: 'completed',
        progress: 100,
        message: `Loaded model: ${modelName}`,
        result: { ...result, history, mlflow: data.mlflow },
      });
    } catch (err: any) {
      setError(err.message);
    }
  };

useEffect(() => {
  fetchModels();
  fetchMlflowConfig();
}, [isAdminUnlocked, adminToken]);

useEffect(() => {
  if (!isAdminUnlocked) {
    setTcnModels([]);
    setSelectedTCNModel('');
    setModelRegistry(null);
    if (ADMIN_ONLY_VIEWS.has(activeView)) setActiveView('home');
    return;
  }

  fetchTCNModels();
  fetchModelRegistry();
}, [isAdminUnlocked, adminToken]);

useEffect(() => {
  if (!isAdminUnlocked && ADMIN_ONLY_VIEWS.has(activeView)) {
    setActiveView('home');
  }
}, [activeView, isAdminUnlocked]);

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

  const unlockAdminConsole = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const token = adminTokenInput.trim();
    if (!token) {
      setAdminLoginError('Enter the admin access key to open training controls.');
      return;
    }

    setIsAdminValidating(true);
    setAdminLoginError(null);
    try {
      const res = await fetch(`${API_BASE}/tcn-models`, {
        headers: { 'X-Admin-Token': token },
      });
      if (!res.ok) {
        if (res.status === 503) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.error || 'Admin console is currently unavailable. Please make sure NEUROSETTLE_ADMIN_TOKEN is configured on the backend.');
        }
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Invalid admin access key.');
      }
      localStorage.setItem('neurosettleAdminToken', token);
      setAdminToken(token);
      setAdminTokenInput('');
      setAdminLoginError(null);
      setIsAdminUnlocked(true);
    } catch (err: any) {
      setAdminLoginError(err.message || 'Validation failed. Please verify your token.');
    } finally {
      setIsAdminValidating(false);
    }
  };

  const lockAdminConsole = () => {
    localStorage.removeItem('neurosettleAdminToken');
    setAdminToken('');
    setAdminTokenInput('');
    setAdminLoginError(null);
    setIsAdminUnlocked(false);
  };

  const applyTrainingPreset = (presetKey: TrainingPresetKey) => {
    const preset = trainingPresets[presetKey];
    setTrainingPreset(presetKey);
    setEpochs(preset.epochs);
    setBatchSize(preset.batchSize);
    setLearningRate(preset.learningRate);
    setEmbeddingDim(preset.embeddingDim);
    setEarlyStoppingPatience(preset.earlyStoppingPatience);
    setTimeLimit(preset.timeLimit);
    setAgPresets(preset.agPresets);
    setTcnAugment(preset.tcnAugment);
  };

  useEffect(() => {
    if (!trainJobId) return;
    let cancelled = false;
    let interval: number | undefined;

    const pollTrainJob = async () => {
      try {
        const job = await getJob(trainJobId);
        if (cancelled) return;
        setTrainJob(job);
        setError(null);
        if (isTerminalJob(job)) {
          if (interval) window.clearInterval(interval);
          setTrainJobId(null);
          localStorage.removeItem('trainJobId');
        }
      } catch (err: any) {
        if (!cancelled) {
          setError(`Training status refresh failed: ${err.message}`);
        }
      }
    };

    pollTrainJob();
    interval = window.setInterval(pollTrainJob, 1500);
    return () => {
      cancelled = true;
      if (interval) window.clearInterval(interval);
    };
  }, [trainJobId]);

  useEffect(() => {
    if (!predictJobId) return;
    let cancelled = false;
    let interval: number | undefined;

    const pollPredictJob = async () => {
      try {
        const job = await getJob(predictJobId);
        if (cancelled) return;
        setPredictJob(job);
        setError(null);
        if (isTerminalJob(job)) {
          if (interval) window.clearInterval(interval);
          setPredictJobId(null);
          localStorage.removeItem('predictJobId');
        }
      } catch (err: any) {
        if (!cancelled) {
          setError(`Prediction status refresh failed: ${err.message}`);
        }
      }
    };

    pollPredictJob();
    interval = window.setInterval(pollPredictJob, 1500);
    return () => {
      cancelled = true;
      if (interval) window.clearInterval(interval);
    };
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
    if (!isAdminUnlocked || !adminToken) {
      setError('Admin access is required before starting a training job.');
      return;
    }

    const useExistingSplit = trainingSource === 'split';
    if (!dataset && !useExistingSplit) {
      setError('Please upload a training CSV or use an existing split directory');
      return;
    }

    if (useExistingSplit && !existingSplitDir.trim()) {
      setError('Please enter split directory path');
      return;
    }

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
        upload_id: useExistingSplit ? null : dataset?.upload_id,
        split_dir: useExistingSplit ? existingSplitDir.trim() : null,
        target_col: targetCol,
        id_col: idCol,
        wave_prefix: wavePrefix,
        epochs,
        batch_size: batchSize,
        lr: learningRate,
        embedding_dim: embeddingDim,
        fast_ms: fastMs,
        fast_weight: fastWeight,
        model_name: modelName.trim(),
        time_limit: timeLimit,
        ag_presets: agPresets,
        tcn_augment: tcnAugment,
        tcn_noise_std: tcnNoiseStd,
        tcn_scale_jitter: tcnScaleJitter,
        tcn_time_shift: tcnTimeShift,
        early_stopping_patience: earlyStoppingPatience,

        train_new_tcn: trainNewTCN,
        existing_tcn_name: trainNewTCN ? null : selectedTCNModel,
      }, adminToken);

      setTrainJob({
        job_id: res.job_id,
        job_type: 'train',
        status: 'queued',
        progress: 0,
        message: 'Queued training job...',
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
      fetchMlflowConfig();
      fetchModelRegistry();

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
        upload_id: predictUpload.upload_id,
        id_col: idCol,
        wave_prefix: wavePrefix,
        model_name: selectedModel,
      });

      setPredictJobId(res.job_id);
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Derived data used by the dashboard sections.
  const trainMetrics   = trainJob?.result?.metrics ?? {};
  const predictPreview = predictJob?.result?.preview_predictions ?? [];
  const analysisItems  = predictJob?.result?.analysis_manifest
                      ?? trainJob?.result?.analysis_manifest
                      ?? [];

  // Prefer the backend total, then fall back to the loaded preview count.
  const totalWaves: number = predictJob?.result?.total_waves
                          ?? trainJob?.result?.total_waves
                          ?? analysisItems.length;

  const activeJobId: string | null = useMemo(() => {
    if (predictJobId) return predictJobId;

    // Recover the job id from plot image URLs after a page refresh.
    const first = analysisItems[0];
    if (!first?.image) return null;
    const parts = String(first.image).split('/');
    const idx = parts.indexOf('plots');
    if (idx !== -1 && parts[idx + 1]) return parts[idx + 1];
    return null;
  }, [analysisItems, predictJobId]);

  // Keep the first preview waves in numeric order when IDs contain numbers.
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

    if (Array.isArray(history)) {
      return history.map((point: any, index: number) => ({
        epoch: Number(point.epoch ?? index + 1),
        train_loss: Number(point.train_loss ?? 0),
        val_loss: Number(point.val_loss ?? point.valid_loss ?? 0),
      }));
    }

    const trainLoss = Array.isArray(history.train_loss) ? history.train_loss : [];
    const valLoss = Array.isArray(history.val_loss)
      ? history.val_loss
      : (Array.isArray(history.valid_loss) ? history.valid_loss : []);

    return trainLoss.map((value: number, index: number) => ({
      epoch: index + 1,
      train_loss: Number(value ?? 0),
      val_loss: Number(valLoss[index] ?? 0),
    }));
  }, [trainJob]);

  const handleWaveSearch = async () => {
    const q = gallerySearch.trim();
    if (!q || !activeJobId) return;

    setSearchLoading(true);
    setSearchError(null);
    setSearchedItem(null);

    try {
      const res = await fetch(`${API_BASE}/plot-wave`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wave_id: q, job_id: activeJobId }),
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

  const handleDeleteModel = async (modelToDelete: string) => {
    if (!isAdminUnlocked || !adminToken) {
      setError('Admin access is required before deleting a model.');
      return;
    }

    if (modelToDelete === DEFAULT_MODEL_NAME) {
      setError('Default model cannot be deleted. Promote another model before removing it.');
      return;
    }

    const confirmed = window.confirm(`Delete model "${formatModelName(modelToDelete)}"? This removes its local AutoGluon and unused TCN artifacts.`);
    if (!confirmed) return;

    try {
      setError(null);
      await deleteModel(modelToDelete, adminToken);

      if (selectedModel === modelToDelete) {
        const fallbackModel = models.find((model) => model.name !== modelToDelete)?.name ?? '';
        setSelectedModel(fallbackModel);
      }
      if (selectedTrainModel === modelToDelete) {
        setSelectedTrainModel('');
        setTrainJob(null);
      }

      await Promise.all([
        fetchModels(),
        fetchTCNModels(),
        fetchModelRegistry(),
      ]);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleLoadModelAudit = async (modelToAudit: string) => {
    if (!isAdminUnlocked || !adminToken) {
      throw new Error('Admin access is required before loading model audit plots.');
    }

    return getModelAudit(modelToAudit, adminToken);
  };

  useEffect(() => {
    localStorage.setItem('workspaceView', activeView);
  }, [activeView]);

  useEffect(() => {
    const viewTitleMap: Record<WorkspaceView, string> = {
      home: 'API Gateway',
      prediction: 'Prediction',
      training: 'Training',
      registry: 'Models',
      workflow: 'Workflow',
    };
    document.title = `NEUROSETTLE - ${viewTitleMap[activeView]}`;
  }, [activeView]);

  useEffect(() => {
    const root = document.querySelector('.app-main');
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!root || reduceMotion) return undefined;

    document.body.classList.add('reveal-motion-ready');

    const revealSelector = [
      '.workspace-panel > .view-heading',
      '.workspace-panel > section',
      '.landing-page > section',
      '.prediction-workspace > section',
      '.waveform-gallery-card',
      '.workflow-page > .ml-pipeline-card',
      '.workflow-page > .workflow-lifecycle',
      '.workflow-page > .workflow-explanation',
      '.workflow-lifecycle-grid > article',
      '.workflow-explanation-grid > article',
      '.landing-feature-card',
      '.analysis-card',
      '.card',
    ].join(',');

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add('is-revealed');
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -8% 0px' },
    );

    const revealIfVisible = (element: HTMLElement) => {
      const rect = element.getBoundingClientRect();
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
      if (rect.top < viewportHeight + 120 && rect.bottom > -80) {
        element.classList.add('is-revealed');
        observer.unobserve(element);
      }
    };

    const revealVisibleTargets = () => {
      root.querySelectorAll('.scroll-reveal:not(.is-revealed)').forEach((element) => {
        if (element instanceof HTMLElement) revealIfVisible(element);
      });
    };

    let revealFrame: number | null = null;
    const scheduleRevealCheck = () => {
      if (revealFrame !== null) return;
      revealFrame = window.requestAnimationFrame(() => {
        revealFrame = null;
        revealVisibleTargets();
      });
    };

    const registerRevealTargets = () => {
      root.querySelectorAll(revealSelector).forEach((element, index) => {
        if (!(element instanceof HTMLElement)) return;
        if (element.classList.contains('scroll-reveal')) return;

        element.classList.add('scroll-reveal');
        element.style.setProperty('--reveal-delay', `${Math.min((index % 6) * 40, 200)}ms`);
        observer.observe(element);
      });
      scheduleRevealCheck();
    };

    registerRevealTargets();

    const mutationObserver = new MutationObserver(registerRevealTargets);
    mutationObserver.observe(root, { childList: true, subtree: true });
    window.addEventListener('scroll', scheduleRevealCheck, { passive: true });
    window.addEventListener('resize', scheduleRevealCheck);

    const revealFallback = window.setInterval(revealVisibleTargets, 250);
    const stopRevealFallback = window.setTimeout(() => {
      window.clearInterval(revealFallback);
    }, 3000);

    return () => {
      if (revealFrame !== null) window.cancelAnimationFrame(revealFrame);
      window.clearInterval(revealFallback);
      window.clearTimeout(stopRevealFallback);
      window.removeEventListener('scroll', scheduleRevealCheck);
      window.removeEventListener('resize', scheduleRevealCheck);
      mutationObserver.disconnect();
      observer.disconnect();
      document.body.classList.remove('reveal-motion-ready');
    };
  }, []);

  const trainStatusLabel = trainJob?.status
    ? `${trainJob.status}${trainJob.progress != null ? ` / ${trainJob.progress}%` : ''}`
    : 'Ready';
  const predictStatusLabel = predictJob?.status
    ? `${predictJob.status}${predictJob.progress != null ? ` / ${predictJob.progress}%` : ''}`
    : 'Ready';
  const bestRegistryVersion = currentRegistryVersions.find((version) => version.version === modelRegistry?.best_version)
    ?? currentRegistryVersions[0];
  const hasReadyModels = models.length > 0;
  const bestRegistryModelName = getRegistryVersionModelName(bestRegistryVersion) || selectedModel;
  const bestModelLabel = hasReadyModels && (bestRegistryVersion || selectedModel) ? formatModelName(bestRegistryModelName) : 'None';
  const selectedModelLabel = hasReadyModels && selectedModel ? formatModelName(selectedModel) : 'No model available';
  const selectedTrainModelNote = getModelNote(selectedTrainModel);
  const predictEndpoint = `${API_BASE}/predict`;
  const adminStatusLabel = isAdminUnlocked ? 'Admin Unlocked' : 'Admin Locked';

  return (
    <div className={`page-shell ${activeView === 'home' ? 'is-home' : ''}`}>
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark">
            <img src="/adi_logo.png" alt="ADI logo" className="brand-logo" />
          </div>
          <div className="brand-wordmark">
            <h1>NEUROSETTLE</h1>
            <span>ADI product workspace</span>
          </div>
        </div>

        <nav className="app-navbar" aria-label="Primary workspace navigation">
          <button
            type="button"
            className={activeView === 'home' ? 'is-active' : ''}
            onClick={() => setActiveView('home')}
          >
            <Home size={17} />
            <span>Home</span>
          </button>
          <button
            type="button"
            className={activeView === 'prediction' ? 'is-active' : ''}
            onClick={() => setActiveView('prediction')}
          >
            <PlayCircle size={17} />
            <span>Prediction</span>
          </button>
          {isAdminUnlocked ? (
            <>
              {isTrainingEnabled && (
                <button
                  type="button"
                  className={activeView === 'training' ? 'is-active' : ''}
                  onClick={() => setActiveView('training')}
                >
                  <BrainCircuit size={17} />
                  <span>Training</span>
                </button>
              )}
              <button
                type="button"
                className={activeView === 'registry' ? 'is-active' : ''}
                onClick={() => setActiveView('registry')}
              >
                <Database size={17} />
                <span>Models</span>
              </button>
            </>
          ) : null}
          <button
            type="button"
            className={activeView === 'workflow' ? 'is-active' : ''}
            onClick={() => setActiveView('workflow')}
          >
            <BookOpen size={17} />
            <span>Workflow</span>
          </button>
        </nav>

        <div className="header-actions">
          <div className="header-model-pill" title={selectedModel || undefined}>
            <Sparkles size={15} />
            <span>Default</span>
            <strong>{selectedModelLabel}</strong>
          </div>
          {isAdminUnlocked ? (
            <button type="button" className="header-auth-btn" onClick={lockAdminConsole}>
              <LogOut size={15} />
              <span>Lock</span>
            </button>
          ) : (
            <button type="button" className="header-auth-btn" onClick={() => setActiveView('training')}>
              <ShieldCheck size={15} />
              <span>Admin Login</span>
            </button>
          )}
        </div>
      </header>

      <div className="app-frame">
        <div className="app-main">
          {activeView === 'training' ? (
          <section className="workspace-strip">
            <div className="workspace-summary">
              <div>
                <span>{isAdminUnlocked ? 'Train' : 'API'}</span>
                <strong>{isAdminUnlocked ? trainStatusLabel : 'Ready'}</strong>
              </div>
              <div>
                <span>Predict</span>
                <strong>{predictStatusLabel}</strong>
              </div>
              <div>
                <span>Ready Models</span>
                <strong>{models.length}</strong>
              </div>
              <div>
                <span>Best Model</span>
                <strong title={selectedModel || undefined}>{bestModelLabel}</strong>
              </div>
            </div>
          </section>
          ) : null}

      {error ? <div className="error-banner">{error}</div> : null}

      {activeView === 'home' ? (
        <main className="workspace-panel landing-page">
          <section className="landing-hero">
            <div className="landing-hero-copy">
              {/* Commented out brand logo badge:
              <div className="hero-brand-badge">
                <img src="/neurosettle-icon.png" alt="NEUROSETTLE" className="hero-brand-logo" />
                <span className="landing-kicker"><ShieldCheck size={15} /> ML Platform</span>
              </div>
              */}
              <span className="landing-kicker"><ShieldCheck size={15} /> NEUROSETTLE ML Platform</span>
              <h2>Predict settling time from raw data with confidence.</h2>
              <p>
                Run NS 1.3 inference through an API-ready workspace, inspect waveform-level
                results, and keep training plus model operations protected for admins.
              </p>
              <div className="landing-actions">
                <button type="button" className="primary-btn" onClick={() => setActiveView('prediction')}>
                  <PlayCircle size={17} /><span>Start Prediction</span><ArrowRight size={16} />
                </button>
                <button type="button" className="ghost-btn" onClick={() => setActiveView('training')}>
                  <Lock size={16} /><span>Admin Console</span>
                </button>
              </div>
            </div>
            <div className="landing-hero-visual">
              {/* Commented out dynamic waveform graph:
              <HeroWaveformGraphic />
              */}
              <img src="/neurosettle-logo.png" alt="NEUROSETTLE waveform prediction platform" className="hero-logo-card" loading="lazy" />
            </div>
          </section>

          <section className="landing-surface-section">
            <div className="surface-copy">
              <span className="view-kicker">Live Surface</span>
              <h2>API status, default model, and access mode in one operational strip.</h2>
              <p>
                Monitor your prediction gateway, active model, and access permissions from one unified dashboard.
              </p>
            </div>
            <div className="landing-product-panel" aria-label="NEUROSETTLE runtime summary">
              <div className="product-panel-top">
                <div>
                  <span>Live Surface</span>
                  <strong>Prediction Gateway</strong>
                </div>
                <CheckCircle2 size={18} />
              </div>
              <div className="product-panel-screen">
                <div className="screen-row is-live">
                  <Server size={16} />
                  <span>POST /api/predict</span>
                  <strong>Public</strong>
                </div>
                <div className="screen-row">
                  <BrainCircuit size={16} />
                  <span>{selectedModelLabel}</span>
                  <strong>Default</strong>
                </div>
                <div className="screen-row">
                  <Rocket size={16} />
                  <span>{predictStatusLabel}</span>
                  <strong>Inference</strong>
                </div>
              </div>
              <div className="landing-hero-metrics">
                <div>
                  <span>Ready Models</span>
                  <strong>{models.length}</strong>
                </div>
                <div>
                  <span>Best Model</span>
                  <strong>{bestModelLabel}</strong>
                </div>
                <div>
                  <span>Admin</span>
                  <strong>{adminStatusLabel}</strong>
                </div>
              </div>
            </div>
          </section>

          <section className="landing-api-panel">
            <div className="landing-section-heading">
              <span className="view-kicker">Developer Entry</span>
              <h2>Prediction first, training protected.</h2>
            </div>
            <div className="landing-api-grid">
              <article className="landing-api-copy">
                <div className="landing-api-icon"><FileCode2 size={20} /></div>
                <h3>Public inference workspace</h3>
                <p>
                  Upload waveform CSVs, select a ready model, run inference, and inspect waveform-level plots.
                  This is the default path for API users and non-admin operators.
                </p>
                <button type="button" className="ghost-btn" onClick={() => setActiveView('prediction')}>
                  <PlayCircle size={16} /><span>Run Prediction</span>
                </button>
              </article>
              <article className="landing-code-card">
                <div className="landing-code-head">
                  <TerminalSquare size={18} />
                  <span>API Quickstart</span>
                </div>
                <pre><code>{`POST ${predictEndpoint}
Content-Type: application/json

{
  "upload_id": "infer.csv",
  "model_name": "${selectedModel || 'your_model_name'}",
  "id_col": "wave_id",
  "wave_prefix": "wave_"
}`}</code></pre>
              </article>
            </div>
          </section>

          <section className={`landing-capabilities ${isAdminUnlocked ? 'is-admin' : 'is-public'}`}>
            <div className="landing-section-heading">
              <span className="view-kicker">Workspace Map</span>
              <h2>{isAdminUnlocked ? 'One product surface for admins and model review.' : 'A focused product surface for API users.'}</h2>
            </div>
            <div className="landing-feature-grid">
              <button type="button" className="landing-feature-card" onClick={() => setActiveView('prediction')}>
                <PlayCircle size={20} />
                <span>Prediction</span>
                <strong>Run API-facing inference and review outputs.</strong>
              </button>
              {isAdminUnlocked ? (
                <>
                  {isTrainingEnabled && (
                    <button type="button" className="landing-feature-card" onClick={() => setActiveView('training')}>
                      <BrainCircuit size={20} />
                      <span>Admin Training</span>
                      <strong>Start controlled TCN + AutoGluon training runs.</strong>
                    </button>
                  )}
                  <button type="button" className="landing-feature-card" onClick={() => setActiveView('registry')}>
                    <Database size={20} />
                    <span>Models</span>
                    <strong>Review model registry and candidate metrics.</strong>
                  </button>
                  <button type="button" className="landing-feature-card" onClick={() => setActiveView('workflow')}>
                    <BookOpen size={20} />
                    <span>Workflow</span>
                    <strong>Trace the feature, TCN, and AutoGluon pipeline.</strong>
                  </button>
                </>
              ) : (
                <>
                  <button type="button" className="landing-feature-card" onClick={() => setActiveView('workflow')}>
                    <BookOpen size={20} />
                    <span>Workflow</span>
                    <strong>Trace the feature, TCN, and AutoGluon pipeline.</strong>
                  </button>
                </>
              )}
            </div>
          </section>
        </main>
      ) : null}

      {activeView === 'training' ? (
        <main className="workspace-panel training-page">
          <div className="view-heading">
            <div>
              <span className="view-kicker">Training Lab</span>
              <h2>Build and evaluate waveform models</h2>
            </div>
            <div className="view-chip">
              {isAdminUnlocked ? <ShieldCheck size={15} /> : <Lock size={15} />}
              {isAdminUnlocked ? trainStatusLabel : 'Admin access required'}
            </div>
          </div>

          {!isAdminUnlocked ? (
            <section className="admin-gate">
              <div className="admin-gate-copy">
                <span className="landing-kicker"><KeyRound size={15} /> Admin Console</span>
                <h2>Training is reserved for model owners.</h2>
                <p>
                  Prediction remains open for API users. Training creates new artifacts, updates model candidates,
                  and can consume compute, so this console is separated behind an admin access key.
                </p>
                <div className="admin-gate-points">
                  <span><ShieldCheck size={15} /> Backend train endpoint supports admin token enforcement</span>
                  <span><Database size={15} /> Model registry opens only after admin unlock</span>
                </div>
              </div>
               <form className="admin-login-card" onSubmit={unlockAdminConsole}>
                <label>
                  <span>Admin Access Key</span>
                  <input
                    type="password"
                    value={adminTokenInput}
                    onChange={(e) => {
                      setAdminTokenInput(e.target.value);
                      setAdminLoginError(null);
                    }}
                    placeholder="Enter training key"
                    autoComplete="current-password"
                    disabled={isAdminValidating}
                  />
                </label>
                {adminLoginError ? <div className="admin-login-error">{adminLoginError}</div> : null}
                <button type="submit" className="primary-btn" disabled={isAdminValidating}>
                  <KeyRound size={16} />
                  <span>{isAdminValidating ? 'Verifying...' : 'Unlock Training'}</span>
                </button>
                <small>Set NEUROSETTLE_ADMIN_TOKEN on the backend before deploying.</small>
              </form>
            </section>
          ) : (
          <>
          <section className="training-flow" aria-label="Training workflow steps">
            <ol>
            <li>
              <span>01</span>
              <div>
                <strong>Dataset</strong>
                <small>Choose split or upload CSV</small>
              </div>
            </li>
            <li>
              <span>02</span>
              <div>
                <strong>Model Setup</strong>
                <small>Name the run and choose TCN strategy</small>
              </div>
            </li>
            <li>
              <span>03</span>
              <div>
                <strong>Train</strong>
                <small>Tune essentials, then start pipeline</small>
              </div>
            </li>
            <li>
              <span>04</span>
              <div>
                <strong>Evaluate</strong>
                <small>Review metrics, plots, and artifacts</small>
              </div>
            </li>
            <li>
              <span>05</span>
              <div>
                <strong>Registry</strong>
                <small>Promote candidates and compare versions</small>
              </div>
            </li>
            </ol>
          </section>

          <div className="admin-session-bar">
            <span><ShieldCheck size={15} /> Admin training controls unlocked for this browser.</span>
            <button type="button" className="ghost-btn" onClick={lockAdminConsole}>
              <Lock size={15} /><span>Lock Console</span>
            </button>
          </div>

          <div className="grid two training-grid">
            <section className="card tall-card">
              <label className="load-model-label">
                <span>Load Existing Model</span>
                <select
                  value={selectedTrainModel}
                  onChange={(e) => {
                    setSelectedTrainModel(e.target.value);
                    loadTrainResult(e.target.value);
                  }}
                  disabled={modelsLoading || models.length === 0}
                >
                  <option value="">-- Train new model --</option>
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>{formatModelName(m.name)}</option>
                  ))}
                </select>
                {selectedTrainModelNote ? <small className="model-note">{selectedTrainModelNote}</small> : null}
              </label>
              <div className="section-header">
                <div className="section-title"><FileUp size={18} /><span className="step-number">01</span><span>Training Dataset</span></div>
              </div>
              <label>
                <span>Training Source</span>
                <select
                  value={trainingSource}
                  onChange={(e) => {
                    setTrainingSource(e.target.value as 'upload' | 'split');
                    setError(null);
                  }}
                >
                  <option value="split">Use existing train/valid/test split</option>
                  <option value="upload">Upload raw CSV and split automatically</option>
                </select>
              </label>

              {trainingSource === 'split' ? (
                <label>
                  <span>Split Directory</span>
                  <input
                    value={existingSplitDir}
                    onChange={(e) => setExistingSplitDir(e.target.value)}
                    placeholder="data/split_noise_10000/splits"
                  />
                </label>
              ) : (
                <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                  <input {...getInputProps()} />
                  <Waves size={30} />
                  <div className="dropzone-title">
                    {uploading ? 'Uploading dataset...' : 'Drag and drop a CSV file here'}
                  </div>
                  <small>Expected columns: wave_id, sample, time_ms, value, wait_time_ms</small>
                </div>
              )}

              {dataset && trainingSource === 'upload' ? (
                <div className="grid four compact-gap">
                  <StatCard title="Rows" value={dataset.shape[0].toLocaleString()} />
                  <StatCard title="Columns" value={dataset.shape[1]} />
                  <StatCard title="Waves"   value={dataset.wave_count.toLocaleString() ?? 0} />
                  <StatCard title="Samples" value={dataset.sample_count ?? 0} />
                </div>
              ) : null}

              {canConfigureTraining ? (
                <>
                  <div className="config-stack">
                    <div className="config-group">
                      <div className="config-group-head">
                        <div>
                          <span className="config-title-line">
                            <span className="step-number">02</span>
                            <span className="config-group-title">Basic Setup</span>
                          </span>
                          <strong>Choose strategy and name this run</strong>
                        </div>
                        <Settings2 size={18} />
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
                                    {formatModelName(m.name)}
                                  </option>
                                ))
                              )}
                            </select>
                            {getModelNote(selectedTCNModel) ? <small className="model-note">{getModelNote(selectedTCNModel)}</small> : null}
                          </label>
                        )}
                        <label><span>Model Name</span>
                          <input
                            value={modelName}
                            onChange={(e) => setModelName(e.target.value)}
                            placeholder="e.g. wave_model_v2"
                          />
                        </label>
                      </div>

                      <div className="preset-panel">
                        <div className="preset-panel-head">
                          <span>Quick Preset</span>
                          <strong>Choose a reliable starting point, then override only when needed</strong>
                        </div>
                        <div className="preset-grid" role="group" aria-label="Training presets">
                          {(Object.entries(trainingPresets) as [TrainingPresetKey, typeof trainingPresets[TrainingPresetKey]][]).map(([key, preset]) => (
                            <button
                              key={key}
                              type="button"
                              className={`preset-option ${trainingPreset === key ? 'is-active' : ''}`}
                              onClick={() => applyTrainingPreset(key)}
                            >
                              <span>{preset.title}</span>
                              <strong>{preset.description}</strong>
                              <small>{preset.epochs} epochs / {preset.agPresets.replace('_', ' ')}</small>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <details className="config-disclosure">
                      <summary>
                        <span className="config-summary-copy">
                          <span className="config-title-line">
                            <span className="step-number">03</span>
                            <span className="config-disclosure-title">Training Overrides</span>
                          </span>
                          <strong>Optional knobs for expert tuning</strong>
                        </span>
                      </summary>
                      <div className="form-grid">
                        <label><span>Epochs</span>
                          <input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
                        </label>
                        <label><span>Batch Size</span>
                          <input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
                        </label>
                        <label><span>Learning Rate</span>
                          <input type="number" step="0.0001" value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))} />
                        </label>
                        <label><span>Early Stop Patience</span>
                          <input type="number" min="1" value={earlyStoppingPatience} onChange={(e) => setEarlyStoppingPatience(Number(e.target.value))} />
                        </label>
                        <label><span>Fast Threshold (ms)</span>
                          <input type="number" step="0.001" value={fastMs} onChange={(e) => setFastMs(Number(e.target.value))} />
                        </label>
                        <label><span>Fast Weight</span>
                          <input type="number" step="0.1" min="1" value={fastWeight} onChange={(e) => setFastWeight(Number(e.target.value))} />
                        </label>
                      </div>
                    </details>

                    <details className="config-disclosure">
                      <summary>
                        <span className="config-disclosure-title">Advanced Controls</span>
                        <strong>TCN embedding, augmentation, and AutoGluon budget</strong>
                      </summary>
                      <div className="form-grid">
                        <label><span>Embedding Dimension</span>
                          <input type="number" value={embeddingDim} onChange={(e) => setEmbeddingDim(Number(e.target.value))} />
                        </label>
                        <label><span>AutoGluon Preset</span>
                          <select value={agPresets} onChange={(e) => setAgPresets(e.target.value)}>
                            <option value="medium_quality">medium_quality</option>
                            <option value="good_quality">good_quality</option>
                            <option value="high_quality">high_quality</option>
                            <option value="best_quality">best_quality</option>
                          </select>
                        </label>
                        <label><span>AutoGluon Time Limit (sec)</span>
                          <input type="number" min="30" value={timeLimit} onChange={(e) => setTimeLimit(Number(e.target.value))} />
                        </label>
                        <label className={`checkbox-label ${tcnAugment ? 'is-checked' : ''}`}>
                          <input
                            type="checkbox"
                            checked={tcnAugment}
                            onChange={(e) => setTcnAugment(e.target.checked)}
                          />
                          <span className="checkbox-box" aria-hidden="true" />
                          <span className="checkbox-copy">TCN augmentation</span>
                        </label>
                        {tcnAugment ? (
                          <>
                            <label><span>Aug Noise Std</span>
                              <input type="number" step="0.001" min="0" value={tcnNoiseStd} onChange={(e) => setTcnNoiseStd(Number(e.target.value))} />
                            </label>
                            <label><span>Aug Scale Jitter</span>
                              <input type="number" step="0.01" min="0" value={tcnScaleJitter} onChange={(e) => setTcnScaleJitter(Number(e.target.value))} />
                            </label>
                            <label><span>Aug Time Shift</span>
                              <input type="number" min="0" value={tcnTimeShift} onChange={(e) => setTcnTimeShift(Number(e.target.value))} />
                            </label>
                          </>
                        ) : null}
                      </div>
                    </details>
                  </div>
                  <button className="primary-btn" onClick={startTraining}>
                    <SlidersHorizontal size={16} /><span>Start Training Pipeline</span>
                  </button>
                </>
              ) : null}
            </section>

            <section className="card tall-card">
              <div className="section-header">
                <div className="section-title"><Activity size={18} /><b>Preview</b><span>Dataset Preview</span></div>
              </div>
              {dataset
                ? <DataTable rows={dataset.preview} />
                : (
                  <div className="empty-state empty-action-state">
                    <strong>Preview appears after upload</strong>
                    <span>Upload mode shows a sample table here. Existing split mode trains directly from train.csv, valid.csv, and test.csv.</span>
                    <button type="button" className="empty-state-action" onClick={() => setTrainingSource('upload')}>
                      Switch to upload mode
                    </button>
                  </div>
                )
              }
            </section>
          </div>

          <Suspense fallback={<div className="empty-state">Loading training analytics...</div>}>
            <TrainingStatusSection
              trainJob={trainJob}
              mlflowInfo={mlflowInfo}
              trainMetrics={trainMetrics}
              trainHistory={trainHistory}
              overfittingSummary={overfittingSummary}
            />

            <FeatureImportanceSection featureSummary={featureSummary} trainJob={trainJob} />
          </Suspense>
          </>
          )}
        </main>
      ) : null}

      {activeView === 'prediction' ? (
        <main className="workspace-panel prediction-page">
          <div className="view-heading">
            <div>
              <span className="view-kicker">Prediction Workspace</span>
              <h2>Run inference with a trained model</h2>
            </div>
            <div className="view-chip" title={selectedModel || undefined}><LineChart size={15} /> {selectedModel ? formatModelName(selectedModel) : 'No model selected'}</div>
          </div>

          <Suspense fallback={<div className="empty-state">Loading prediction workspace...</div>}>
            <PredictionWorkspace
              models={models}
              modelsLoading={modelsLoading}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
              predictFile={predictFile}
              handlePredictFile={handlePredictFile}
              predictUpload={predictUpload}
              runPredict={runPredict}
              predictJob={predictJob}
              predictPreview={predictPreview}
            />

            <WaveformGallery
              analysisItems={analysisItems}
              displayedAnalysis={displayedAnalysis}
              totalWaves={totalWaves}
              gallerySearch={gallerySearch}
              setGallerySearch={(value) => {
                setGallerySearch(value);
                setSearchedItem(null);
              }}
              searchedItem={searchedItem}
              searchLoading={searchLoading}
              searchError={searchError}
              predictJobId={activeJobId}
              handleWaveSearch={handleWaveSearch}
              clearSearch={clearSearch}
              clearSearchError={() => setSearchError(null)}
            />
          </Suspense>
        </main>
      ) : null}

      {activeView === 'registry' && isAdminUnlocked ? (
        <main className="workspace-panel registry-page">
          <div className="view-heading">
            <div>
              <span className="view-kicker">Models & Runs</span>
              <h2>Monitor the model registry</h2>
            </div>
            <div className="view-chip"><Database size={15} /> {modelRegistry?.registered_model_name || 'Registry'}</div>
          </div>

          <Suspense fallback={<div className="empty-state">Loading model registry...</div>}>
            <ModelRegistrySection
              registry={modelRegistry}
              loading={modelRegistryLoading}
              onRefresh={() => {
                fetchMlflowConfig();
                fetchModelRegistry();
              }}
              onDeleteModel={handleDeleteModel}
              onLoadModelAudit={handleLoadModelAudit}
            />
          </Suspense>
        </main>
      ) : null}

      {activeView === 'workflow' ? (
        <main className="workspace-panel workflow-view">
          <div className="view-heading">
            <div>
              <span className="view-kicker">Workflow</span>
              <h2>Understand the hybrid ML pipeline</h2>
            </div>
            <div className="view-chip"><BookOpen size={15} /> Method Notes</div>
          </div>

          <Suspense fallback={<div className="empty-state">Loading workflow...</div>}>
            <WorkflowSection />
          </Suspense>
        </main>
      ) : null}
        </div>
      </div>

      <footer className="app-footer">
        <div className="footer-divider-glow"></div>
        <div className="footer-container">
          <div className="footer-brand-section">
            <div className="footer-logo-lockup">
              <img src="/neurosettle-icon.png" alt="NEUROSETTLE" className="footer-mini-logo" />
              <div className="footer-brand-text">
                <span className="footer-brand-name">NEUROSETTLE</span>
                <span className="footer-brand-sub">Waveform Analytics Engine</span>
              </div>
            </div>
            <p className="footer-description">
              High-precision settling time prediction and analysis platform for ADI engineering workflows.
            </p>
          </div>

          <div className="footer-meta-section">
            <div className="footer-status-pills">
              <span className="meta-pill">
                <span>Inference Active</span>
              </span>
              <span className="meta-pill version">v1.3</span>
              <span className="meta-pill environment">{isTrainingEnabled ? 'Local Workspace' : 'Production Hub'}</span>
            </div>
            
            <div className="footer-credits">
              <span>Developed by <strong>Teerawit Pongkunawut</strong> and <strong>Sukit Saelao</strong></span>
              <span className="footer-copyright">© {new Date().getFullYear()} Analog Devices, Inc. All rights reserved.</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
