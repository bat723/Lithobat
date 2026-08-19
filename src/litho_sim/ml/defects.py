"""
Defect injection utilities for synthetic dataset generation.

Provides:
* Particle / blob defects on aerial images.
* Stochastic line-edge roughness (LER) on aerial images.
* Bridging defects between adjacent features.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def add_particle_defect(
    image: NDArray[np.float64],
    particle_radius_px: float = 6.0,
    intensity: float = 1.0,
    position: tuple[int, int] | None = None,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Add a circular particle (blob) defect to an image.

    Parameters
    ----------
    image : NDArray
        2-D image to augment (values in [0, 1]).
    particle_radius_px : float
        Particle radius in pixels.
    intensity : float
        Defect intensity (0–1).  1.0 = fully opaque particle.
    position : (row, col), optional
        Centre of the particle.  Random if not given.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Augmented image, same shape and dtype.
    """
    rng = np.random.default_rng(seed)
    ny, nx = image.shape
    result = image.copy()

    if position is None:
        margin = int(np.ceil(particle_radius_px * 2))
        row_c = int(rng.integers(margin, ny - margin))
        col_c = int(rng.integers(margin, nx - margin))
    else:
        row_c, col_c = int(position[0]), int(position[1])

    y_idx, x_idx = np.ogrid[:ny, :nx]
    dist = np.sqrt((y_idx - row_c) ** 2 + (x_idx - col_c) ** 2)
    mask = dist <= particle_radius_px

    result[mask] = np.clip(result[mask] + intensity * (1.0 - result[mask]), 0.0, 1.0)
    logger.debug(
        "Particle defect: centre=(%d,%d), r=%.1f px, I=%.2f",
        row_c, col_c, particle_radius_px, intensity,
    )
    return result.astype(np.float64)


def add_line_roughness(
    image: NDArray[np.float64],
    roughness_amplitude: float = 0.1,
    correlation_length_px: float = 8.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Add correlated LER-style noise to an aerial image.

    Generates spatially correlated Gaussian noise (via Gaussian blur of
    white noise) and adds it to the image.

    Parameters
    ----------
    image : NDArray
        2-D aerial image.
    roughness_amplitude : float
        Amplitude of the added noise (fraction of peak intensity).
    correlation_length_px : float
        Spatial correlation length of the noise [pixels].
    seed : int, optional
        RNG seed.

    Returns
    -------
    NDArray[np.float64]
        Augmented image.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(image.shape)
    sigma_blur = max(correlation_length_px / 2.355, 0.5)  # FWHM → sigma
    correlated = gaussian_filter(white, sigma=sigma_blur)
    correlated /= max(correlated.std(), 1e-12)  # unit variance
    result = np.clip(image + roughness_amplitude * correlated, 0.0, 1.0)
    logger.debug(
        "Line roughness: amplitude=%.3f, corr_len=%.1f px",
        roughness_amplitude, correlation_length_px,
    )
    return result.astype(np.float64)


def add_bridge_defect(
    image: NDArray[np.float64],
    row: int | None = None,
    bridge_width_px: int = 4,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Add a horizontal bridging defect across a dark space in the image.

    Useful for simulating photoresist bridging between adjacent lines.

    Parameters
    ----------
    image : NDArray
        2-D aerial image.
    row : int, optional
        Row index for the bridge.  Defaults to the centre row.
    bridge_width_px : int
        Vertical extent of the bridge in pixels.
    seed : int, optional
        RNG seed for randomising column extent.

    Returns
    -------
    NDArray[np.float64]
        Augmented image.
    """
    rng = np.random.default_rng(seed)
    result = image.copy()
    ny, nx = image.shape

    if row is None:
        row = ny // 2

    # Randomise horizontal extent: bridge spans some fraction of the width
    span_start = int(rng.integers(0, nx // 3))
    span_end = int(rng.integers(2 * nx // 3, nx))
    half = bridge_width_px // 2

    r0 = max(0, row - half)
    r1 = min(ny, row + half + 1)

    result[r0:r1, span_start:span_end] = np.clip(
        result[r0:r1, span_start:span_end] + 0.5, 0.0, 1.0
    )
    logger.debug(
        "Bridge defect: row=%d±%d, col=[%d,%d]",
        row, half, span_start, span_end,
    )
    return result.astype(np.float64)

