#!/usr/bin/env python3
"""Core VAE and MLP components for instrument-specific MS QC models."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters shared by the published instrument-specific models."""

    latent_dim: int = 8
    vae_epochs: int = 200
    mlp_epochs: int = 200
    batch_size: int = 16
    vae_learning_rate: float = 1e-3
    mlp_learning_rate: float = 1e-3
    seed: int = 100


def set_seed(seed: int) -> None:
    """Set random seeds and deterministic CUDA settings."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VAE(nn.Module):
    """Variational autoencoder used to represent QC missingness profiles."""

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(256, latent_dim)
        self.z_log_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        mu = self.z_mean(encoded)
        logvar = self.z_log_var(encoded)
        reconstruction = self.decoder(self.reparameterize(mu, logvar))
        return reconstruction, mu, logvar


class ClassifierMLP(nn.Module):
    """Binary classifier operating on the VAE latent mean."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def vae_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Return the reconstruction plus KL-divergence loss."""

    mse = ((reconstruction - x) ** 2).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return mse + kl


def best_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Select the threshold that maximizes Youden's J statistic."""

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, scores)
    return float(thresholds[int(np.argmax(true_positive_rate - false_positive_rate))])


def train_vae_mlp(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    config: TrainConfig,
    log_prefix: str = "",
    reset_seed: bool = True,
) -> tuple[VAE, ClassifierMLP, np.ndarray]:
    """Train a VAE on good samples and an MLP on all latent representations."""

    if reset_seed:
        set_seed(config.seed)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    good_x = torch.tensor(x[y == 0], dtype=torch.float32, device=device)
    if good_x.shape[0] == 0:
        raise ValueError("At least one good sample is required to train the VAE.")

    vae = VAE(input_dim=x.shape[1], latent_dim=config.latent_dim).to(device)
    vae_optimizer = optim.Adam(vae.parameters(), lr=config.vae_learning_rate)
    for epoch in range(config.vae_epochs):
        vae.train()
        vae_optimizer.zero_grad()
        reconstruction, mu, logvar = vae(good_x)
        loss = vae_loss(reconstruction, good_x, mu, logvar)
        loss.backward()
        vae_optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"{log_prefix} VAE epoch {epoch + 1}/{config.vae_epochs}: "
                f"loss={loss.item():.6f}",
                flush=True,
            )

    vae.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        _, latent_mean, _ = vae(x_tensor)

    latent = latent_mean.detach()
    y_tensor = torch.tensor(y.astype(np.float32), device=device).reshape(-1, 1)
    loader = DataLoader(
        TensorDataset(latent, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    mlp = ClassifierMLP(config.latent_dim).to(device)
    mlp_optimizer = optim.Adam(mlp.parameters(), lr=config.mlp_learning_rate)
    criterion = nn.BCELoss()
    for epoch in range(config.mlp_epochs):
        mlp.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            mlp_optimizer.zero_grad()
            predictions = mlp(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            mlp_optimizer.step()
            epoch_loss += loss.item() * batch_x.shape[0]
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"{log_prefix} MLP epoch {epoch + 1}/{config.mlp_epochs}: "
                f"loss={epoch_loss / len(y):.6f}",
                flush=True,
            )

    mlp.eval()
    with torch.no_grad():
        scores = mlp(latent).cpu().numpy().ravel()
    return vae, mlp, scores


def predict_scores(
    vae: VAE,
    mlp: ClassifierMLP,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict bad-sample scores from a standardized feature matrix."""

    vae.eval()
    mlp.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(np.asarray(x, dtype=np.float32), device=device)
        _, latent_mean, _ = vae(x_tensor)
        return mlp(latent_mean).cpu().numpy().ravel()


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def save_checkpoint(
    path: Path,
    vae: VAE,
    mlp: ClassifierMLP,
    threshold: float,
    feature_columns: list[str],
    instrument: str,
    preprocessing_profile: str,
    config: TrainConfig,
) -> None:
    """Save one self-describing checkpoint for training and inference."""

    checkpoint: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "instrument": instrument,
        "preprocessing_profile": preprocessing_profile,
        "feature_layout": "clean_missing_block_then_raw_missing_block",
        "feature_columns": feature_columns,
        "input_dim": len(feature_columns),
        "latent_dim": config.latent_dim,
        "threshold": float(threshold),
        "train_config": asdict(config),
        "vae_state_dict": _cpu_state_dict(vae),
        "mlp_state_dict": _cpu_state_dict(mlp),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[VAE, ClassifierMLP, dict[str, Any]]:
    """Load a published checkpoint and reconstruct both neural networks."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "vae_state_dict",
        "mlp_state_dict",
        "input_dim",
        "latent_dim",
        "threshold",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")

    vae = VAE(checkpoint["input_dim"], checkpoint["latent_dim"]).to(device)
    mlp = ClassifierMLP(checkpoint["latent_dim"]).to(device)
    vae.load_state_dict(checkpoint["vae_state_dict"])
    mlp.load_state_dict(checkpoint["mlp_state_dict"])
    vae.eval()
    mlp.eval()
    return vae, mlp, checkpoint
