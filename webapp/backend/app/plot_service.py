from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_loss_curve(history: dict[str, list[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(history.get("train_loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_scatter_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true, y_pred, s=16, alpha=0.7)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([low, high], [low, high])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_error_histogram(errors: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4.5))
    plt.hist(errors, bins=40)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_target_distribution(series: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4.5))
    plt.hist(series.astype(float), bins=40)
    plt.xlabel(series.name or "target")
    plt.ylabel("Count")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_waveform_analysis(
    row_id: str,
    waveform: np.ndarray,
    pred_value: float,
    true_value: float | None,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 4.8))
    plt.plot(waveform)
    title = f"Waveform {row_id} | pred={pred_value:.6f}"
    if true_value is not None:
        title += f" | true={true_value:.6f} | err={pred_value - true_value:.6f}"
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
