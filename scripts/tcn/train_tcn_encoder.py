from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt


# small constant for numerical stability
EPS = 1e-12


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    Ensures experiments produce the same results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WaveDataset(Dataset):
    """
    PyTorch Dataset for waveform regression.

    Input
    -----
    X : waveform signals (N, L)
    y : regression target (wait_time)

    Output shape
    ------------
    X -> (N, 1, L)   # channel dimension added for Conv1D
    y -> (N)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wave_id: np.ndarray,
        sample_weight: np.ndarray | None = None,
        augment: bool = False,
        noise_std: float = 0.0,
        scale_jitter: float = 0.0,
        time_shift: int = 0,
    ):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.wave_id = torch.tensor(wave_id)
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        self.sample_weight = torch.tensor(sample_weight, dtype=torch.float32)
        self.augment = augment
        self.noise_std = float(noise_std)
        self.scale_jitter = float(scale_jitter)
        self.time_shift = int(time_shift)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()
        if self.augment:
            if self.scale_jitter > 0:
                scale = 1.0 + torch.empty(1).uniform_(-self.scale_jitter, self.scale_jitter).item()
                x = x * scale
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            if self.time_shift > 0:
                shift = int(torch.randint(-self.time_shift, self.time_shift + 1, (1,)).item())
                if shift:
                    x = torch.roll(x, shifts=shift, dims=-1)
        return x, self.y[idx], self.wave_id[idx], self.sample_weight[idx]


class Chomp1d(nn.Module):
    """
    Removes extra padding on the right side.

    In causal convolution we pad on the left, but PyTorch Conv1D pads
    both sides. Chomp1d removes the extra elements to enforce causality.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """
    Core building block of the TCN.

    Structure
    ---------
    Conv1D
    -> ReLU
    -> Dropout
    -> Conv1D
    -> ReLU
    -> Dropout
    + Residual connection

    dilation increases exponentially to enlarge receptive field.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()

        # padding needed for causal convolution
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),   # enforce causal convolution
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # if input/output channels differ, use 1x1 convolution to match dimensions
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)

        # residual connection
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TCNRegressor(nn.Module):
    """
    Temporal Convolutional Network for waveform regression.

    Pipeline
    --------
    Waveform
    -> TCN feature extractor
    -> Global pooling
    -> Embedding layer
    -> Linear regression head
    """

    def __init__(
        self,
        input_channels: int = 1,
        channels: list[int] | None = None,
        kernel_size: int = 5,
        dropout: float = 0.1,
        embedding_dim: int = 64,
    ):
        super().__init__()

        # default TCN channel sizes
        if channels is None:
            channels = [32, 64, 64]

        layers = []
        in_ch = input_channels

        # build stacked TCN layers
        for i, out_ch in enumerate(channels):

            # exponential dilation
            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )

            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        # global average pooling over time dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # embedding layer (feature representation of waveform)
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, embedding_dim),
            nn.ReLU(),
        )

        # final regression head
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # extract temporal features
        z = self.tcn(x)

        # pool across time
        z = self.pool(z)

        # embedding representation
        emb = self.embed(z)

        # regression output
        y = self.head(emb).squeeze(-1)

        return y, emb


@dataclass
class TrainConfig:
    """
    Training configuration parameters.
    """

    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    log_target: bool = True
    seed: int = 42


def train_one_epoch(model, loader, optimizer, criterion, device: str) -> float:
    """
    Run one training epoch.
    """

    model.train()
    total = 0.0
    count = 0

    for xb, yb, _, wb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        wb = wb.to(device)

        optimizer.zero_grad()
        pred, _ = model(xb)
        loss = (criterion(pred, yb) * wb).mean()
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * len(xb)
        count += len(xb)

    return total / max(count, 1)


def eval_one_epoch(model, loader, criterion, epoch: int, device: str, log_target: bool) -> float:
    """
    Evaluate model on validation set.
    """

    model.eval()

    total = 0.0
    count = 0

    with torch.no_grad():
        for i, (xb, yb, _, _) in enumerate(loader):

            xb = xb.to(device)
            yb = yb.to(device)

            pred, _ = model(xb)

            # Convert back to real wait time when training with log target.
            pred_real = torch.expm1(pred) if log_target else pred
            true_real = torch.expm1(yb) if log_target else yb

            # Print one preview batch only, to keep the training log readable.
            if i == 0 and epoch == 1:
                print("pred:", pred_real[:10].cpu().numpy())
                print("true:", true_real[:10].cpu().numpy())
                print("-----")

            loss = criterion(pred, yb).mean()

            total += float(loss.item()) * len(xb)
            count += len(xb)

    return total / max(count, 1)


# ADD : print predict wait time from Test and cal MAE,RMSE 
def plot_training_curve(history, out_dir: str) -> None:
    """
    Plot training vs validation loss and save figure.
    """

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    valid_loss = [h["valid_loss"] for h in history]

    best_epoch = epochs[valid_loss.index(min(valid_loss))]

    plt.figure(figsize=(8,5))

    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, valid_loss, marker="o", label="Validation Loss")

    plt.axvline(best_epoch, linestyle="--", label=f"Best Epoch ({best_epoch})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    save_path = os.path.join(out_dir, "learning_curve.png")
    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Saved learning curve to {save_path}")

# Early Stop training


# Prediction : Legacy
def predict(model, loader, device, out_dir, name="test"):
    model.eval()

    all_pred = []
    all_true = []
    wave_ids = []

    with torch.no_grad():
        for xb, yb, wid, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred, _ = model(xb)

            pred_real = torch.expm1(pred)
            true_real = torch.expm1(yb)

            all_pred.append(pred_real.cpu())
            all_true.append(true_real.cpu())
            wave_ids.append(wid.cpu())

    pred_all = torch.cat(all_pred)
    true_all = torch.cat(all_true)
    wave_ids = torch.cat(wave_ids)

    mae = torch.mean(torch.abs(pred_all - true_all))
    rmse = torch.sqrt(torch.mean((pred_all - true_all)**2))

    print(f"[{name}] MAE: {mae.item():.6f}")
    print(f"[{name}] RMSE: {rmse.item():.6f}")

    df = pd.DataFrame({
        "wave_id": wave_ids.numpy(),
        "true_wait_time": true_all.numpy(),
        "pred_wait_time": pred_all.numpy(),
        "error": (pred_all - true_all).numpy()
    })

    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{name}_predictions.csv")
    df.to_csv(save_path, index=False)

    print(f"Saved {save_path}") 


def sample_weights(y_real: np.ndarray, fast_ms: float, fast_weight: float) -> np.ndarray:
    weights = np.ones(len(y_real), dtype=np.float32)
    if fast_ms > 0 and fast_weight > 1:
        weights[y_real <= fast_ms] = float(fast_weight)
    return weights


def load_tensor(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    wave_id = d["wave_id"].astype(np.int64)
    mask = np.isfinite(y)
    return X[mask], y[mask], wave_id[mask]


def main() -> None:

    # command line arguments
    ap = argparse.ArgumentParser("train_tcn_encoder")

    ap.add_argument("--waves", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid-waves", default=None)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Prediction
    # ap.add_argument("--test-waves", required=True)

    ap.add_argument("--embedding-dim", type=int, default=64)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    ap.add_argument("--valid-frac", type=float, default=0.2)

    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--noise-std", type=float, default=0.015)
    ap.add_argument("--scale-jitter", type=float, default=0.04)
    ap.add_argument("--time-shift", type=int, default=8)
    ap.add_argument("--fast-ms", type=float, default=0.1)
    ap.add_argument("--fast-weight", type=float, default=3.0)
    ap.add_argument("--early-stopping-patience", type=int, default=8)

    args = ap.parse_args()

    set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    X, y, wave_id = load_tensor(args.waves)

    if args.valid_waves:
        X_valid, y_valid_real, wave_id_valid = load_tensor(args.valid_waves)
        X_train, y_train_real, wave_id_train = X, y, wave_id
    else:
        n = len(X)
        idx = np.arange(n)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(idx)
        n_valid = int(round(n * float(args.valid_frac)))
        valid_idx = idx[:n_valid]
        train_idx = idx[n_valid:]
        X_train, y_train_real, wave_id_train = X[train_idx], y[train_idx], wave_id[train_idx]
        X_valid, y_valid_real, wave_id_valid = X[valid_idx], y[valid_idx], wave_id[valid_idx]

    if args.log_target:
        y_train = np.log1p(np.clip(y_train_real, 0.0, None))
        y_valid = np.log1p(np.clip(y_valid_real, 0.0, None))
    else:
        y_train = y_train_real.copy()
        y_valid = y_valid_real.copy()

    train_weight = sample_weights(y_train_real, args.fast_ms, args.fast_weight)
    valid_weight = np.ones(len(y_valid), dtype=np.float32)

    # dataset FIX : add wave_id to get the pred data
    # train_ds = WaveDataset(X_train, y_train)
    # valid_ds = WaveDataset(X_valid, y_valid)

    train_ds = WaveDataset(
        X_train,
        y_train,
        wave_id_train,
        sample_weight=train_weight,
        augment=bool(args.augment),
        noise_std=args.noise_std,
        scale_jitter=args.scale_jitter,
        time_shift=args.time_shift,
    )
    valid_ds = WaveDataset(X_valid, y_valid, wave_id_valid, sample_weight=valid_weight)

    # dataloader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)

    # create model
    model = TCNRegressor(
        embedding_dim=int(args.embedding_dim)
    ).to(device)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    # regression loss
    criterion = nn.SmoothL1Loss(reduction="none")

    best_val = float("inf")
    best_state = None
    history = []
    epochs_without_improvement = 0

    # training loop
    for epoch in range(1, args.epochs + 1):

        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        va = eval_one_epoch(
            model,
            valid_loader,
            criterion,
            epoch,
            device,
            bool(args.log_target),
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr,
            "valid_loss": va
        })

        print(f"epoch={epoch:03d} train={tr:.6f} valid={va:.6f}")

        # save best model
        if va < best_val:
            best_val = va
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(args.early_stopping_patience):
                print(f"Early stopping at epoch={epoch} best_valid={best_val:.6f}")
                break

    if best_state is None:
        raise RuntimeError("No best model state captured")

    model.load_state_dict(best_state)

    # === RUN TEST PREDICTION ===
    # predict(model, test_loader, device, args.out, name="test")

    # save model checkpoint
    ckpt = {
        "state_dict": best_state,
        "embedding_dim": int(args.embedding_dim),
        "log_target": bool(args.log_target),
        "seed": int(args.seed),
        "augment": bool(args.augment),
        "fast_ms": float(args.fast_ms),
        "fast_weight": float(args.fast_weight),
    }

    torch.save(ckpt, os.path.join(args.out, "tcn_encoder.pt"))

    # save training history
    with open(os.path.join(args.out, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # plot learning curve
    plot_training_curve(history, args.out)

    # save config
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "embedding_dim": args.embedding_dim,
                "seed": args.seed,
                "log_target": bool(args.log_target),
                "valid_waves": args.valid_waves,
                "augment": bool(args.augment),
                "noise_std": args.noise_std,
                "scale_jitter": args.scale_jitter,
                "time_shift": args.time_shift,
                "fast_ms": args.fast_ms,
                "fast_weight": args.fast_weight,
                "early_stopping_patience": args.early_stopping_patience,
            },
            f,
            indent=2,
        )

    print(f"Saved model to {args.out} | best_valid={best_val:.6f}")


if __name__ == "__main__":
    main()
