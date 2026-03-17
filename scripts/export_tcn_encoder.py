from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class WaveOnlyDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRegressor(nn.Module):
    def __init__(self, input_channels: int = 1, channels: list[int] | None = None, kernel_size: int = 5, dropout: float = 0.1, embedding_dim: int = 64):
        super().__init__()
        if channels is None:
            channels = [32, 64, 64]
        layers = []
        in_ch = input_channels
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, embedding_dim), nn.ReLU())
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.tcn(x)
        z = self.pool(z)
        emb = self.embed(z)
        y = self.head(emb).squeeze(-1)
        return y, emb


def main() -> None:
    ap = argparse.ArgumentParser("export_tcn_embeddings")
    ap.add_argument("--model", required=True)
    ap.add_argument("--waves", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(os.path.join(args.model, "tcn_encoder.pt"), map_location="cpu")
    embedding_dim = int(ckpt["embedding_dim"])

    model = TCNRegressor(embedding_dim=embedding_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(args.device)

    d = np.load(args.waves)
    X = d["X"].astype(np.float32)
    wave_id = d["wave_id"].astype(np.int64)

    ds = WaveOnlyDataset(X)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_emb = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(args.device)
            _, emb = model(xb)
            all_emb.append(emb.cpu().numpy())

    E = np.concatenate(all_emb, axis=0)
    cols = [f"tcn_embed_{i:02d}" for i in range(E.shape[1])]
    out = pd.DataFrame(E, columns=cols)
    out.insert(0, "wave_id", wave_id)
    out.to_csv(args.out, index=False)
    print(f"✅ Saved {args.out} | rows={len(out)} cols={len(out.columns)}")


if __name__ == "__main__":
    main()