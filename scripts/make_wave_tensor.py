from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# small constant to avoid division by zero
EPS = 1e-12


def resample_wave(t: np.ndarray, x: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample waveform to fixed length using linear interpolation.

    Input
    -----
    t : time array
    x : signal value array

    Output
    ------
    t_new : resampled time
    x_new : resampled waveform (length = target_len)

    Reason
    ------
    Waveforms may have different number of samples.
    Deep learning models require fixed-length inputs.
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    # edge case: waveform has fewer than 2 samples
    if len(t) < 2:
        t_new = np.linspace(0.0, 1.0, target_len)
        x_new = np.full(target_len, float(x[0]) if len(x) else 0.0, dtype=float)
        return t_new, x_new

    # ensure time is sorted
    order = np.argsort(t)
    t = t[order]
    x = x[order]

    t_min = float(t[0])
    t_max = float(t[-1])

    # edge case: constant time
    if t_max <= t_min:
        t_new = np.linspace(t_min, t_min + 1.0, target_len)
        x_new = np.full(target_len, float(np.median(x)), dtype=float)
        return t_new, x_new

    # create evenly spaced time grid
    t_new = np.linspace(t_min, t_max, target_len)

    # interpolate signal
    x_new = np.interp(t_new, t, x)

    return t_new, x_new


def normalize_wave(x: np.ndarray, mode: str = "robust") -> np.ndarray:
    """
    Normalize waveform amplitude.

    Modes
    -----
    robust : robust scaling using median and percentiles
    zscore : standard normalization
    none   : no normalization
    """

    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return x

    if mode == "zscore":

        # standard normalization
        mu = float(np.mean(x))
        sd = float(np.std(x))
        sd = max(sd, EPS)

        return (x - mu) / sd

    # robust normalization (less sensitive to outliers)
    med = float(np.median(x))
    q95 = float(np.percentile(x, 95))
    q05 = float(np.percentile(x, 5))

    scale = max(q95 - q05, EPS)

    return (x - med) / scale


def extract_wave_label(g: pd.DataFrame, label_col: str | None) -> float | None:
    """
    Extract regression label for a waveform.

    Each wave_id may have many rows (samples).
    We assume the label is identical across the group.
    """

    if label_col is None:
        return None

    if label_col not in g.columns:
        raise KeyError(f"Missing label column: {label_col}")

    vals = g[label_col].dropna().to_numpy(dtype=float)

    if len(vals) == 0:
        return None

    return float(vals[0])


def main() -> None:
    """
    Convert waveform CSV data into tensor dataset (.npz).

    Output
    ------
    X : waveform tensor (N, L)
    y : labels
    wave_id : identifiers
    """

    ap = argparse.ArgumentParser("make_wave_tensor")

    # input CSV file
    ap.add_argument("--in", dest="in_path", required=True)

    # output tensor file
    ap.add_argument("--out", dest="out_path", required=True)

    # target waveform length after resampling
    ap.add_argument("--target-len", type=int, default=1000)

    # optional time window cutoff
    ap.add_argument("--window-ms", type=float, default=None)

    # minimum points required for waveform
    ap.add_argument("--min-pts", type=int, default=20)

    # label column (optional)
    ap.add_argument("--label-col", default=None)

    # normalization method
    ap.add_argument("--normalize", choices=["robust", "zscore", "none"], default="robust")

    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================
    # Load CSV dataset
    # =========================

    df = pd.read_csv(in_path)

    required = ["wave_id", "sample", "time_ms", "value"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # sort waveform samples
    df = df.sort_values(["wave_id", "sample"]).copy()

    # optional time window truncation
    if args.window_ms is not None:
        df = df[df["time_ms"].astype(float) <= float(args.window_ms)].copy()

    waves = []
    wave_ids = []
    labels = []

    # =========================
    # Process each waveform
    # =========================

    for wave_id, g in df.groupby("wave_id", sort=False):

        # skip short signals
        if len(g) < int(args.min_pts):
            continue

        t = g["time_ms"].to_numpy(dtype=float)
        x = g["value"].to_numpy(dtype=float)

        # resample waveform to fixed length
        _, x_rs = resample_wave(t, x, target_len=int(args.target_len))

        # normalize signal
        if args.normalize != "none":
            x_rs = normalize_wave(x_rs, mode=args.normalize)

        waves.append(x_rs.astype(np.float32))
        wave_ids.append(int(wave_id))

        # extract label
        y = extract_wave_label(g, args.label_col)

        labels.append(np.nan if y is None else y)

    if len(waves) == 0:
        raise ValueError("No valid waves found")

    # stack waveform tensor
    X = np.stack(waves, axis=0)  # shape: (N, L)

    wave_ids_arr = np.asarray(wave_ids, dtype=np.int64)
    y_arr = np.asarray(labels, dtype=np.float32)

    # =========================
    # Save dataset
    # =========================

    np.savez_compressed(
        out_path,
        X=X,
        wave_id=wave_ids_arr,
        y=y_arr,
    )

    print(f"Saved {out_path} | X={X.shape} | labeled={np.isfinite(y_arr).sum()}")


if __name__ == "__main__":
    main()