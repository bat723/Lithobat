"""
Training and inference utilities for the wafer anomaly detection autoencoder.

Design
------
The autoencoder is trained to reconstruct **clean** images only.  During
inference, high per-pixel reconstruction error indicates a defect — the
model cannot faithfully reconstruct patterns it was never trained on.

Anomaly scoring
---------------
Rather than using global MSE (which is diluted by the large background),
we compute the mean of the top-*k* pixel errors in the centre of the
image (cropping border artefacts).  This is the "localised anomaly score"
reported in PROLITH-style inspection workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean batch loss.

    Parameters
    ----------
    model : nn.Module
        Autoencoder model.
    dataloader : DataLoader
        Training dataloader (clean images only).
    optimizer : Optimizer
        Gradient-descent optimizer.
    criterion : nn.Module
        Reconstruction loss (e.g. ``nn.MSELoss()``).
    device : torch.device
        Compute device.

    Returns
    -------
    float
        Mean loss over all mini-batches.
    """
    model.train()
    total_loss = 0.0
    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on a held-out validation set.

    Parameters
    ----------
    model : nn.Module
        Autoencoder model.
    dataloader : DataLoader
        Validation dataloader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.

    Returns
    -------
    float
        Mean validation loss.
    """
    model.eval()
    total_loss = 0.0
    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        recon = model(images)
        total_loss += criterion(recon, images).item()
    return total_loss / max(len(dataloader), 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    save_dir: Path = Path("models"),
    model_name: str = "autoencoder",
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Full training loop with early stopping and best-model checkpointing.

    Parameters
    ----------
    model : nn.Module
        Autoencoder to train.
    train_loader, val_loader : DataLoader
        Training and validation data.
    epochs : int
        Maximum training epochs.
    lr : float
        Initial learning rate for Adam.
    save_dir : Path
        Directory for model checkpoints.
    model_name : str
        Checkpoint filename prefix.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    dict
        ``{"train_loss": [...], "val_loss": [...]}`` history lists.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    # No ``verbose`` — torch removed the argument in 2.x, and this crashed
    # on any current install until mypy flagged it.
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_ckpt = save_dir / f"{model_name}_best.pth"
    patience_counter = 0
    early_stop_patience = 15

    logger.info(
        "Training: device=%s, epochs=%d, lr=%.5f, "
        "train=%d batches, val=%d batches",
        device, epochs, lr, len(train_loader), len(val_loader),
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            "Epoch [%3d/%d]  train=%.5f  val=%.5f",
            epoch, epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)
            patience_counter = 0
            logger.debug("  → New best val loss: %.5f – checkpoint saved.", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping at epoch %d (patience=%d).", epoch, early_stop_patience)
                break

    # Save final model
    torch.save(model.state_dict(), save_dir / f"{model_name}_final.pth")
    logger.info("Training complete. Best val loss: %.5f", best_val_loss)
    return history

