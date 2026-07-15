# ==============================================================================
# This file is responsible for managing the in-memory cache of PyTorch (TCN) 
# and AutoGluon models and performing fast synchronous predictions.
# ==============================================================================
from __future__ import annotations

import os
import json
import time
import datetime
import threading
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path for absolute imports of scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.features.extract_features import extract_one_wave
from scripts.data.make_wave_tensor import resample_wave, normalize_wave
from scripts.tcn.train_tcn_encoder import TCNRegressor
from autogluon.tabular import TabularPredictor

from .config import (
    AUTOGLUON_DIR,
    TCN_DIR,
    resolve_default_model_name,
)

class InferenceService:
    def __init__(self) -> None:
        self._tcn_cache: dict[str, Any] = {}
        self._ag_cache: dict[str, Any] = {}
        self._meta_cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._log_path = Path(__file__).resolve().parent.parent / "inference_latency.log"

        # Pre-warm/preload the default model in a background thread to avoid cold start latency
        threading.Thread(
            target=self._prewarm_default_model,
            daemon=True,
            name="ModelPrewarmThread"
        ).start()

    def _prewarm_default_model(self) -> None:
        try:
            model_name = resolve_default_model_name()
            self._get_models(model_name)
            print(f"[INFO] Successfully pre-warmed default model: {model_name}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to pre-warm default model: {e}", file=sys.stderr)

    def _log_latency(self, model_name: str, endpoint: str, count: int, latency_ms: float) -> None:
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_line = f"[{now}] Model: {model_name} | Endpoint: {endpoint} | Waveforms: {count} | Latency: {latency_ms:.2f}ms\n"
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            print(f"[WARN] Failed to write to inference_latency.log: {e}", file=sys.stderr)

    def _sanitize_name(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
        return safe.strip("_")

    def _get_models(self, model_name: str) -> tuple[Any, Any, dict]:
        model_name = self._sanitize_name(model_name or resolve_default_model_name())
        
        with self._lock:
            # Check cache first
            if model_name in self._tcn_cache and model_name in self._ag_cache:
                return self._tcn_cache[model_name], self._ag_cache[model_name], self._meta_cache[model_name]

            ag_model_dir = AUTOGLUON_DIR / model_name
            meta_file    = ag_model_dir / "model_meta.json"

            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                tcn_path = Path(meta["tcn_path"])
            else:
                tcn_path = TCN_DIR / model_name
                meta = {}

            if not ag_model_dir.exists():
                raise FileNotFoundError(f"AutoGluon model folder not found at {ag_model_dir}")

            # Load PyTorch TCN Encoder
            ckpt_path = tcn_path / "tcn_encoder.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"TCN encoder checkpoint not found at {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu")
            embedding_dim = int(ckpt["embedding_dim"])

            tcn_model = TCNRegressor(embedding_dim=embedding_dim)
            tcn_model.load_state_dict(ckpt["state_dict"])
            tcn_model.eval()

            # Load AutoGluon TabularPredictor
            predictor = TabularPredictor.load(str(ag_model_dir))
            try:
                predictor.persist()
            except Exception as e:
                print(f"[WARN] Failed to persist AutoGluon models: {e}", file=sys.stderr)

            # Store in cache
            self._tcn_cache[model_name] = tcn_model
            self._ag_cache[model_name] = predictor
            self._meta_cache[model_name] = meta

            return tcn_model, predictor, meta

    def predict_single(self, payload: dict) -> dict:
        t0 = time.perf_counter()
        
        waveform_raw = payload.get("waveform")
        if not waveform_raw or not isinstance(waveform_raw, list):
            raise ValueError("waveform (list of floats) is required")

        dt_ms = float(payload.get("dt_ms", 0.01))
        model_name = payload.get("model_name") or resolve_default_model_name()

        waveform = np.array([float(x) for x in waveform_raw], dtype=float)
        n_samples = len(waveform)
        if n_samples < 20:
            raise ValueError("Waveform too short (minimum 20 samples required)")

        # 1. Load models (reused from cache)
        tcn_model, predictor, meta = self._get_models(model_name)

        # 2. Extract handcrafted features
        df_single = pd.DataFrame({
            "wave_id": ["single_test"] * n_samples,
            "sample": np.arange(1, n_samples + 1),
            "time_ms": np.round(np.arange(1, n_samples + 1) * dt_ms, 6),
            "value": waveform
        })
        feats_dict = extract_one_wave(df_single, mode="pred")
        df_feat = pd.DataFrame([feats_dict])

        # 3. Extract TCN embedding
        t_ms = df_single["time_ms"].to_numpy(dtype=float)
        _, x_rs = resample_wave(t_ms, waveform, target_len=1000)
        x_rs = normalize_wave(x_rs, mode="robust")

        xb = torch.tensor(x_rs, dtype=torch.float32).unsqueeze(0).unsqueeze(1) # shape: (1, 1, 1000)
        with torch.no_grad():
            _, emb = tcn_model(xb)
        emb_np = emb.squeeze(0).numpy()

        cols = [f"tcn_embed_{i:02d}" for i in range(emb.shape[1])]
        df_emb = pd.DataFrame([emb_np], columns=cols)
        df_emb.insert(0, "wave_id", "single_test")

        # 4. Merge
        df_feat["wave_id"] = df_feat["wave_id"].astype(str)
        df_emb["wave_id"] = df_emb["wave_id"].astype(str)
        df_hybrid = df_feat.merge(df_emb, on="wave_id", how="inner")

        # 5. Load feature column schema configuration
        feature_cols_path = Path(predictor.path) / "feature_cols.json"
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        COLS_TO_DROP = ["force_mA", "range_V", "temp_C", "type"]
        LABEL_LEAK_COLS = ["wait_time_ms", "wait_time_log", "is_fast", "is_zero"]
        df_clean = df_hybrid.drop(columns=COLS_TO_DROP + LABEL_LEAK_COLS, errors="ignore").copy()

        if "wave_id" in df_clean.columns:
            df_clean["wave_id"] = np.zeros(len(df_clean), dtype=float)

        missing = [c for c in feature_cols if c not in df_clean.columns]
        for c in missing:
            df_clean[c] = 0.0
        X = df_clean[feature_cols]

        # 6. Predict
        pred_fit = predictor.predict(X)
        pred_fit = float(pred_fit.iloc[0])
        pred_ms = np.expm1(pred_fit) if bool(meta.get("log_target", False)) else pred_fit
        pred_ms = float(np.clip(pred_ms, 0.0, None))

        latency_ms = (time.perf_counter() - t0) * 1000
        self._log_latency(model_name, "/predict-sync", 1, latency_ms)

        return {
            "pred_wait_time_ms": pred_ms,
            "pred_is_fast_at_0p1": 1 if pred_ms <= 0.1 + 1e-12 else 0
        }

    def predict_batch(self, payload: dict) -> dict:
        t0 = time.perf_counter()

        waveforms = payload.get("waveforms", [])
        if not waveforms:
            raise ValueError("waveforms (list of dicts containing wave_id and data) is required")

        dt_ms = float(payload.get("dt_ms", 0.01))
        model_name = payload.get("model_name") or resolve_default_model_name()

        # 1. Load models (reused from cache)
        tcn_model, predictor, meta = self._get_models(model_name)

        rows_feat = []
        resampled_list = []
        wave_ids = []

        for idx, w in enumerate(waveforms):
            wid = w.get("wave_id") or f"wave_{idx}"
            data_raw = w.get("data")
            if not data_raw or not isinstance(data_raw, list):
                raise ValueError(f"Invalid or missing data for waveform: {wid}")

            waveform = np.array([float(x) for x in data_raw], dtype=float)
            n_samples = len(waveform)
            if n_samples < 20:
                raise ValueError(f"Waveform too short for {wid} (minimum 20 samples required)")

            # Feature extraction dataframe
            df_single = pd.DataFrame({
                "wave_id": [wid] * n_samples,
                "sample": np.arange(1, n_samples + 1),
                "time_ms": np.round(np.arange(1, n_samples + 1) * dt_ms, 6),
                "value": waveform
            })

            feats_dict = extract_one_wave(df_single, mode="pred")
            rows_feat.append(feats_dict)

            # TCN prep
            t_ms = df_single["time_ms"].to_numpy(dtype=float)
            _, x_rs = resample_wave(t_ms, waveform, target_len=1000)
            x_rs = normalize_wave(x_rs, mode="robust")
            resampled_list.append(x_rs.astype(np.float32))
            wave_ids.append(wid)

        df_feat = pd.DataFrame(rows_feat)

        # 2. Extract batch TCN embeddings
        xb = torch.tensor(np.array(resampled_list), dtype=torch.float32).unsqueeze(1) # shape: (B, 1, 1000)
        with torch.no_grad():
            _, emb = tcn_model(xb)
        emb_np = emb.numpy()

        cols = [f"tcn_embed_{i:02d}" for i in range(emb.shape[1])]
        df_emb = pd.DataFrame(emb_np, columns=cols)
        df_emb.insert(0, "wave_id", wave_ids)

        # 3. Merge
        df_feat["wave_id"] = df_feat["wave_id"].astype(str)
        df_emb["wave_id"] = df_emb["wave_id"].astype(str)
        df_hybrid = df_feat.merge(df_emb, on="wave_id", how="inner")

        # 4. Load column schema
        feature_cols_path = Path(predictor.path) / "feature_cols.json"
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        COLS_TO_DROP = ["force_mA", "range_V", "temp_C", "type"]
        LABEL_LEAK_COLS = ["wait_time_ms", "wait_time_log", "is_fast", "is_zero"]
        df_clean = df_hybrid.drop(columns=COLS_TO_DROP + LABEL_LEAK_COLS, errors="ignore").copy()

        original_wave_ids = df_clean["wave_id"].tolist()
        if "wave_id" in df_clean.columns:
            df_clean["wave_id"] = np.zeros(len(df_clean), dtype=float)

        missing = [c for c in feature_cols if c not in df_clean.columns]
        for c in missing:
            df_clean[c] = 0.0
        X = df_clean[feature_cols]

        # 5. Predict
        pred_fit = predictor.predict(X).to_numpy()
        pred_ms = np.expm1(pred_fit) if bool(meta.get("log_target", False)) else pred_fit
        pred_ms = np.clip(pred_ms, 0.0, None)

        predictions = []
        for wid, val in zip(original_wave_ids, pred_ms):
            val_float = float(val)
            predictions.append({
                "wave_id": wid,
                "pred_wait_time_ms": val_float,
                "pred_is_fast_at_0p1": 1 if val_float <= 0.1 + 1e-12 else 0
            })

        latency_ms = (time.perf_counter() - t0) * 1000
        self._log_latency(model_name, "/predict-batch-sync", len(waveforms), latency_ms)

        return {"predictions": predictions}

# Global singleton instance
inference_service = InferenceService()
