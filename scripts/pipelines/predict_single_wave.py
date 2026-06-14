from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.features.extract_features import extract_one_wave
from scripts.data.make_wave_tensor import resample_wave, normalize_wave
from scripts.tcn.train_tcn_encoder import TCNRegressor
from autogluon.tabular import TabularPredictor


def main() -> None:
    ap = argparse.ArgumentParser("predict_single_wave")
    ap.add_argument("--model-path", required=True, help="AutoGluon model directory")
    ap.add_argument("--tcn-path", required=True, help="TCN encoder model directory")
    ap.add_argument("--waveform", required=True, help="Comma-separated float values of the waveform")
    ap.add_argument("--dt-ms", type=float, default=0.01, help="Sampling interval in ms")
    args = ap.parse_args()

    # 1. Parse waveform
    try:
        waveform = np.array([float(x.strip()) for x in args.waveform.split(",")], dtype=float)
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse waveform values: {e}"}))
        sys.exit(1)

    n_samples = len(waveform)
    if n_samples < 20:
        print(json.dumps({"error": "Waveform too short (minimum 20 samples required)"}))
        sys.exit(1)

    # 2. Extract handcrafted features
    df = pd.DataFrame({
        "wave_id": "single_test",
        "sample": np.arange(1, n_samples + 1),
        "time_ms": np.round(np.arange(1, n_samples + 1) * args.dt_ms, 6),
        "value": waveform
    })

    try:
        feats_dict = extract_one_wave(df, mode="pred")
    except Exception as e:
        print(json.dumps({"error": f"Feature extraction failed: {e}"}))
        sys.exit(1)

    df_feat = pd.DataFrame([feats_dict])

    # 3. Extract TCN embedding
    try:
        # Load TCN checkpoint
        ckpt_path = os.path.join(args.tcn_path, "tcn_encoder.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"TCN checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        embedding_dim = int(ckpt["embedding_dim"])

        model = TCNRegressor(embedding_dim=embedding_dim)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # Resample and normalize waveform
        t_ms = df["time_ms"].to_numpy(dtype=float)
        _, x_rs = resample_wave(t_ms, waveform, target_len=1000)
        x_rs = normalize_wave(x_rs, mode="robust")

        # Extract embedding
        xb = torch.tensor(x_rs, dtype=torch.float32).unsqueeze(0).unsqueeze(1) # shape: (1, 1, 1000)
        with torch.no_grad():
            _, emb = model(xb)
        emb_np = emb.squeeze(0).numpy()

        # Create embedding DataFrame
        cols = [f"tcn_embed_{i:02d}" for i in range(len(emb_np))]
        df_emb = pd.DataFrame([emb_np], columns=cols)
        df_emb.insert(0, "wave_id", "single_test")
    except Exception as e:
        print(json.dumps({"error": f"TCN embedding extraction failed: {e}"}))
        sys.exit(1)

    # 4. Merge handcrafted features and embeddings
    df_feat["wave_id"] = df_feat["wave_id"].astype(str)
    df_emb["wave_id"] = df_emb["wave_id"].astype(str)
    df_hybrid = df_feat.merge(df_emb, on="wave_id", how="inner")

    if df_hybrid.empty:
        print(json.dumps({"error": "Failed to merge features and embeddings"}))
        sys.exit(1)

    # 5. Predict with AutoGluon
    try:
        # Load AutoGluon model info
        meta_path = os.path.join(args.model_path, "meta.json")
        feature_cols_path = os.path.join(args.model_path, "feature_cols.json")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        predictor = TabularPredictor.load(args.model_path)

        # Align columns
        COLS_TO_DROP = ["force_mA", "range_V", "temp_C", "type"]
        LABEL_LEAK_COLS = ["wait_time_ms", "wait_time_log", "is_fast", "is_zero"]
        df_clean = df_hybrid.drop(columns=COLS_TO_DROP + LABEL_LEAK_COLS, errors="ignore").copy()

        # Handle wave_id numeric type mismatch
        if "wave_id" in df_clean.columns:
            try:
                pd.to_numeric(df_clean["wave_id"], errors="raise")
            except (ValueError, TypeError):
                df_clean["wave_id"] = np.zeros(len(df_clean), dtype=float)

        # Align columns
        missing = [c for c in feature_cols if c not in df_clean.columns]
        for c in missing:
            df_clean[c] = 0.0
        X = df_clean[feature_cols]

        # Predict
        pred_fit = predictor.predict(X)
        pred_fit = float(pred_fit.iloc[0])
        pred_ms = np.expm1(pred_fit) if bool(meta.get("log_target", False)) else pred_fit
        pred_ms = float(np.clip(pred_ms, 0.0, None))

        print(json.dumps({
            "pred_wait_time_ms": pred_ms,
            "pred_is_fast_at_0p1": 1 if pred_ms <= 0.1 + 1e-12 else 0
        }))

    except Exception as e:
        print(json.dumps({"error": f"AutoGluon prediction failed: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
