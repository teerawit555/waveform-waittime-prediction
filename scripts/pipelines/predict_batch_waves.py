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
    ap = argparse.ArgumentParser("predict_batch_waves")
    ap.add_argument("--model-path", required=True, help="AutoGluon model directory")
    ap.add_argument("--tcn-path", required=True, help="TCN encoder model directory")
    ap.add_argument("--input-json", required=True, help="Path to input JSON file containing waveforms")
    ap.add_argument("--dt-ms", type=float, default=0.01, help="Sampling interval in ms")
    args = ap.parse_args()

    # 1. Load input JSON
    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to read input JSON file: {e}"}))
        sys.exit(1)

    waveforms = payload.get("waveforms", [])
    if not waveforms:
        print(json.dumps({"error": "No waveforms provided in input JSON"}))
        sys.exit(1)

    # 2. Extract handcrafted features & prepare TCN waveforms
    rows_feat = []
    resampled_list = []
    wave_ids = []

    for idx, w in enumerate(waveforms):
        wid = w.get("wave_id") or f"wave_{idx}"
        data_raw = w.get("data")
        if not data_raw or not isinstance(data_raw, list):
            print(json.dumps({"error": f"Invalid or missing data for waveform: {wid}"}))
            sys.exit(1)

        try:
            waveform = np.array([float(x) for x in data_raw], dtype=float)
        except Exception as e:
            print(json.dumps({"error": f"Failed to parse waveform data for {wid}: {e}"}))
            sys.exit(1)

        n_samples = len(waveform)
        if n_samples < 20:
            print(json.dumps({"error": f"Waveform too short for {wid} (minimum 20 samples required)"}))
            sys.exit(1)

        # 2a. Handcrafted features dataframe
        df_single = pd.DataFrame({
            "wave_id": [wid] * n_samples,
            "sample": np.arange(1, n_samples + 1),
            "time_ms": np.round(np.arange(1, n_samples + 1) * args.dt_ms, 6),
            "value": waveform
        })

        try:
            feats_dict = extract_one_wave(df_single, mode="pred")
            rows_feat.append(feats_dict)
        except Exception as e:
            print(json.dumps({"error": f"Feature extraction failed for {wid}: {e}"}))
            sys.exit(1)

        # 2b. TCN resample and normalize
        try:
            t_ms = df_single["time_ms"].to_numpy(dtype=float)
            _, x_rs = resample_wave(t_ms, waveform, target_len=1000)
            x_rs = normalize_wave(x_rs, mode="robust")
            resampled_list.append(x_rs.astype(np.float32))
            wave_ids.append(wid)
        except Exception as e:
            print(json.dumps({"error": f"TCN pre-processing failed for {wid}: {e}"}))
            sys.exit(1)

    df_feat = pd.DataFrame(rows_feat)

    # 3. Batch extract TCN embeddings
    try:
        ckpt_path = os.path.join(args.tcn_path, "tcn_encoder.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"TCN checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        embedding_dim = int(ckpt["embedding_dim"])

        model = TCNRegressor(embedding_dim=embedding_dim)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        xb = torch.tensor(np.array(resampled_list), dtype=torch.float32).unsqueeze(1) # shape: (B, 1, 1000)
        with torch.no_grad():
            _, emb = model(xb)
        emb_np = emb.numpy()

        cols = [f"tcn_embed_{i:02d}" for i in range(len(emb_np[0]))]
        df_emb = pd.DataFrame(emb_np, columns=cols)
        df_emb.insert(0, "wave_id", wave_ids)
    except Exception as e:
        print(json.dumps({"error": f"Batch TCN embedding extraction failed: {e}"}))
        sys.exit(1)

    # 4. Merge handcrafted features and embeddings
    df_feat["wave_id"] = df_feat["wave_id"].astype(str)
    df_emb["wave_id"] = df_emb["wave_id"].astype(str)
    df_hybrid = df_feat.merge(df_emb, on="wave_id", how="inner")

    if len(df_hybrid) != len(waveforms):
        print(json.dumps({"error": f"Merged rows count mismatch: got {len(df_hybrid)} instead of {len(waveforms)}"}))
        sys.exit(1)

    # 5. Predict with AutoGluon
    try:
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

        # Handle wave_id type mismatch
        original_wave_ids = df_clean["wave_id"].tolist()
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
        pred_fit = predictor.predict(X).to_numpy()
        if bool(meta.get("log_target", False)):
            pred_ms = np.expm1(pred_fit)
        else:
            pred_ms = pred_fit
        pred_ms = np.clip(pred_ms, 0.0, None)

        predictions = []
        for wid, val in zip(original_wave_ids, pred_ms):
            val_float = float(val)
            predictions.append({
                "wave_id": wid,
                "pred_wait_time_ms": val_float,
                "pred_is_fast_at_0p1": 1 if val_float <= 0.1 + 1e-12 else 0
            })

        print(json.dumps({"predictions": predictions}))

    except Exception as e:
        print(json.dumps({"error": f"AutoGluon prediction failed: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
