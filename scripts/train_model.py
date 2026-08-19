#!/usr/bin/env python3
"""
Script 2: Train the convolutional autoencoder on clean wafer images.

Usage
-----
    python scripts/train_model.py [--data data/raw_synthetic/clean]
                                  [--model-type conv] [--latent-dim 64]
                                  [--epochs 50] [--batch-size 32]
                                  [--output models/]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.utils import setup_logging
from litho_sim.ml.dataset import create_dataloaders
from litho_sim.ml.models import build_model
from litho_sim.ml.train import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train wafer anomaly detection autoencoder.")
    p.add_argument("--data", type=Path, default=Path("data/raw_synthetic/clean"))
    p.add_argument("--model-type", default="conv", choices=["conv", "simple"])
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("models"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build model
    model = build_model(model_type=args.model_type, latent_dim=args.latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %s (latent_dim=%d) | %s parameters",
        args.model_type, args.latent_dim, f"{n_params:,}",
    )

    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_folder=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.output,
        model_name=f"autoencoder_{args.model_type}",
        device=device,
    )

    # Print summary
    best_epoch = int(torch.tensor(history["val_loss"]).argmin()) + 1
    logger.info(
        "Training complete. Best epoch: %d/%d, val_loss=%.5f",
        best_epoch, len(history["val_loss"]),
        min(history["val_loss"]),
    )


if __name__ == "__main__":
    main()

