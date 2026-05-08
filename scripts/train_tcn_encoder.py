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

    def __init__(self, X: np.ndarray, y: np.ndarray,wave_id: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.wave_id = torch.tensor(wave_id)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.wave_id[idx]


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

    for xb, yb , _  in loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred, _ = model(xb)

        loss = criterion(pred, yb)

        loss.backward()

        optimizer.step()

        total += float(loss.item()) * len(xb)
        count += len(xb)

    return total / max(count, 1)


def eval_one_epoch(model, loader, criterion, epoch: int, device: str) -> float:
    """
    Evaluate model on validation set.
    """

    model.eval()

    total = 0.0
    count = 0

    with torch.no_grad():
        for i, (xb, yb, _) in enumerate(loader):  # รับ 3 ค่า

            xb = xb.to(device)
            yb = yb.to(device)

            pred, _ = model(xb)

            # แปลงกลับเป็นค่า real (ถ้าใช้ log target)
            pred_real = torch.expm1(pred)
            true_real = torch.expm1(yb)

            # print แค่ครั้งเดียว (กัน log ระเบิด)
            if i == 0 and epoch == 1:
                print("pred:", pred_real[:10].cpu().numpy())
                print("true:", true_real[:10].cpu().numpy())
                print("-----")

            loss = criterion(pred, yb)

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


# Prediction
def predict(model, loader, device, out_dir, name="test"):
    model.eval()

    all_pred = []
    all_true = []
    wave_ids = []

    with torch.no_grad():
        for xb, yb, wid in loader:
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


def main() -> None:

    # command line arguments
    ap = argparse.ArgumentParser("train_tcn_encoder")

    ap.add_argument("--waves", required=True)
    ap.add_argument("--out", required=True)

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

    args = ap.parse_args()

    set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    # load waveform dataset
    d = np.load(args.waves)

    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    wave_id = d["wave_id"].astype(np.int64)

    # remove invalid targets
    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]
    wave_id = wave_id[mask]

    # optional log transform of target
    if args.log_target:
        y_train_all = np.log1p(np.clip(y, 0.0, None))
    else:
        y_train_all = y.copy()

    # split train / validation
    n = len(X)

    idx = np.arange(n)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    n_valid = int(round(n * float(args.valid_frac)))

    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    X_train, y_train = X[train_idx], y_train_all[train_idx]
    X_valid, y_valid = X[valid_idx], y_train_all[valid_idx]

    # dataset FIX : add wave_id to get the pred data
    # train_ds = WaveDataset(X_train, y_train)
    # valid_ds = WaveDataset(X_valid, y_valid)

    train_ds = WaveDataset(X_train, y_train, wave_id[train_idx])
    valid_ds = WaveDataset(X_valid, y_valid, wave_id[valid_idx])

    # dataloader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    # ===== TEST LOADER =====
    # d_test = np.load(args.test_waves)

    # X_test = d_test["X"].astype(np.float32)
    # y_test = d_test["y"].astype(np.float32)
    # wave_id_test = d_test["wave_id"].astype(np.int64)

    # if args.log_target:
    #     y_test = np.log1p(np.clip(y_test, 0.0, None))

    # test_ds = WaveDataset(X_test, y_test, wave_id_test)
    # test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

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
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    history = []

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
            device
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

    if best_state is None:
        raise RuntimeError("No best model state captured")

    model.load_state_dict(best_state)

    # === RUN TEST PREDICTION ===
    #predict(model, test_loader, device, args.out, name="test")

    # save model checkpoint
    ckpt = {
        "state_dict": best_state,
        "embedding_dim": int(args.embedding_dim),
        "log_target": bool(args.log_target),
        "seed": int(args.seed),
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
            },
            f,
            indent=2,
        )

    print(f"Saved model to {args.out} | best_valid={best_val:.6f}")


if __name__ == "__main__":
    main()