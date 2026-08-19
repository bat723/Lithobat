"""
Post-exposure bake diffusion models.

A single Gaussian blur of the PAC/latent image — the linear, single-species
approximation of acid diffusion. ``apply_peb`` acts on a 2-D map,
``apply_peb_3d`` on a (nz, ny, nx) volume with an optionally different
vertical length. Physically this is what washes out standing waves; see the
vault's Standing Waves note.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from litho_sim.core.config import GridConfig, ResistConfig

logger = logging.getLogger(__name__)


def apply_peb(
    pac_map: NDArray[np.float64],
    diffusion_sigma: float,
    pixel_size: float,
) -> NDArray[np.float64]:
    """Simulate PEB acid diffusion via Gaussian blur.

    Parameters
    ----------
    pac_map : NDArray
        PAC concentration after exposure, shape ``(ny, nx)``.
    diffusion_sigma : float
        Acid diffusion length [m] (1-σ of Gaussian kernel).
    pixel_size : float
        Physical pixel size [m].

    Returns
    -------
    NDArray[np.float64]
        Latent image after PEB, values in [0, 1].
    """
    sigma_px = diffusion_sigma / pixel_size
    if sigma_px < 1e-3:
        logger.debug("PEB: σ_px=%.4f too small – skipping diffusion.", sigma_px)
        return pac_map.copy()
    diffused = gaussian_filter(pac_map.astype(np.float64), sigma=sigma_px)
    diffused = np.clip(diffused, 0.0, 1.0)
    logger.debug(
        "PEB diffusion: σ=%.1f nm (%.2f px)",
        diffusion_sigma * 1e9, sigma_px,
    )
    return diffused


def apply_peb_3d(
    latent: NDArray[np.float64],
    resist: ResistConfig,
    grid: GridConfig,
    sigma_z: float | None = None,
) -> NDArray[np.float64]:
    """Diffuse the latent image in 3-D.

    Parameters
    ----------
    latent : NDArray
        ``(nz, ny, nx)`` PAC concentration.
    resist : ResistConfig
        Supplies ``diffusion_sigma`` [m].
    grid : GridConfig
        Supplies ``pixel_size`` and ``dz``.
    sigma_z : float, optional
        Vertical diffusion length [m].  Defaults to the lateral value; acid
        diffusion is essentially isotropic.

    Returns
    -------
    NDArray
        Diffused latent image.
    """
    s_lat = resist.diffusion_sigma
    s_ver = s_lat if sigma_z is None else sigma_z
    if s_lat <= 0 and s_ver <= 0:
        return latent.copy()
    sigma = (s_ver / grid.dz, s_lat / grid.pixel_size, s_lat / grid.pixel_size)
    out = gaussian_filter(latent, sigma=sigma, mode="nearest")
    return np.clip(out, 0.0, 1.0)
