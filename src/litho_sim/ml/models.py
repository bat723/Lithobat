"""
Autoencoder architectures for wafer image anomaly detection.

Both models are trained on **clean** images only.  At inference,
reconstruction error (MSE) is used as the anomaly score — defective
images reconstruct poorly, yielding high error.

Architectures
-------------
ConvAutoencoder  – full-capacity model  (256×256 → latent_dim → 256×256)
SimpleAutoencoder – tiny bottleneck model for stronger compression
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for 256×256 greyscale images.

    Parameters
    ----------
    latent_dim : int
        Bottleneck latent-space dimensionality.

    Attributes
    ----------
    encoder : nn.Sequential
        Convolutional encoder: 256×256 → ``latent_dim``.
    decoder_fc : nn.Linear
        Linear projection from latent space back to feature maps.
    decoder : nn.Sequential
        Transposed-convolutional decoder: feature maps → 256×256.
    """

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 256→128
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 128→64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 64→32
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, latent_dim),
        )

        self.decoder_fc = nn.Linear(latent_dim, 32 * 32 * 32)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32→64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # 64→128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),    # 128→256
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: encode then decode.

        Parameters
        ----------
        x : Tensor
            Input batch, shape ``(B, 1, 256, 256)``.

        Returns
        -------
        Tensor
            Reconstructed batch, same shape as *x*.
        """
        z = self.encoder(x)
        z = self.decoder_fc(z)
        z = z.view(-1, 32, 32, 32)
        return self.decoder(z)

    @torch.no_grad()
    def get_reconstruction_error(self, x: Tensor) -> Tensor:
        """Compute per-image mean-squared reconstruction error.

        Parameters
        ----------
        x : Tensor
            Input batch, shape ``(B, 1, H, W)``.

        Returns
        -------
        Tensor
            1-D tensor of per-image MSE values, shape ``(B,)``.
        """
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2, 3))


class SimpleAutoencoder(nn.Module):
    """Ultra-compact autoencoder with extreme bottleneck compression.

    Suitable for detecting gross defects due to its very limited
    representational capacity.  Trains faster and overfits less.

    Parameters
    ----------
    latent_dim : int
        Bottleneck dimension (default 16).
    """

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),   # 256→128
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # 128→64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 64→32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), # 32→16
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, latent_dim),
        )

        self.decoder_fc = nn.Linear(latent_dim, 32 * 16 * 16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32→64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),   # 64→128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),    # 128→256
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode and decode *x*."""
        z = self.encoder(x)
        z = self.decoder_fc(z)
        z = z.view(-1, 32, 16, 16)
        return self.decoder(z)

    @torch.no_grad()
    def get_reconstruction_error(self, x: Tensor) -> Tensor:
        """Per-image MSE reconstruction error."""
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2, 3))


def build_model(
    model_type: str = "conv",
    latent_dim: int = 64,
) -> nn.Module:
    """Factory function for creating an autoencoder by name.

    Parameters
    ----------
    model_type : str
        ``"conv"`` → :class:`ConvAutoencoder`;
        ``"simple"`` → :class:`SimpleAutoencoder`.
    latent_dim : int
        Bottleneck dimensionality.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        For an unknown *model_type*.
    """
    if model_type == "conv":
        return ConvAutoencoder(latent_dim=latent_dim)
    elif model_type == "simple":
        return SimpleAutoencoder(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'conv' or 'simple'.")

