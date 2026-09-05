"""
Interface-level tests for the aerial-image subfeature of ``litho_sim.expose``.

These exercise :func:`compute_aerial_image` and :func:`extract_cross_section`
strictly through the public package surface (``litho_sim.expose``,
``litho_sim.core``, ``litho_sim.mask``) the way the CLI and app consume them —
happy-path physical sanity plus the validation the code actually performs.
Deep physics coverage lives in ``tests/test_aerial_image.py`` and
``tests/test_vector_imaging.py``; nothing here duplicates it.

Grids are kept small (64 px, coarse source) so the whole file runs in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.core import GridConfig, OpticsConfig
from litho_sim.expose import (
    compute_aerial_image,
    extract_cross_section,
    normalisation_scale,
)
from litho_sim.mask import lines_and_spaces

# ---------------------------------------------------------------------------
# Fixtures — one cheap scalar image shared across the file
# ---------------------------------------------------------------------------

N = 64
PIXEL = 4e-9  # 256 nm field


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=N, pixel_size=PIXEL)


@pytest.fixture(scope="module")
def optics() -> OpticsConfig:
    # Coarse source grid: the Abbe sum costs one FFT per source point, and
    # interface checks do not need illumination fidelity.
    return OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.85, source_grid=7)


@pytest.fixture(scope="module")
def mask(grid: GridConfig) -> np.ndarray:
    return lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=200e-9, cd=100e-9)


@pytest.fixture(scope="module")
def aerial(mask, optics, grid) -> np.ndarray:
    return compute_aerial_image(mask, optics, grid, dose=1.0)


# ---------------------------------------------------------------------------
# Happy path: config → mask → compute_aerial_image
# ---------------------------------------------------------------------------


def test_scalar_image_shape_dtype_and_finiteness(aerial, grid):
    """The image matches the grid, is float64, and contains no NaN/inf."""
    assert aerial.shape == (grid.n_pixels, grid.n_pixels)
    assert aerial.dtype == np.float64
    assert np.all(np.isfinite(aerial))


def test_scalar_image_is_nonnegative_and_peak_normalised(aerial):
    """Intensity is physical (>= 0) and peak-normalised to dose=1."""
    assert float(aerial.min()) >= -1e-12
    assert abs(float(aerial.max()) - 1.0) < 1e-9


def test_clear_regions_print_brighter_than_opaque(aerial, mask):
    """Mean intensity under clear mask openings beats mean under chrome."""
    clear_mean = float(aerial[mask > 0.5].mean())
    dark_mean = float(aerial[mask < 0.5].mean())
    assert clear_mean > 2.0 * dark_mean, (
        f"No contrast: clear={clear_mean:.3f}, dark={dark_mean:.3f}"
    )


def test_dose_scales_the_image(mask, optics, grid, aerial):
    """Dose is a pure multiplier on the normalised image."""
    dosed = compute_aerial_image(mask, optics, grid, dose=1.5)
    assert np.allclose(dosed, 1.5 * aerial, rtol=1e-10, atol=1e-12)


def test_return_raw_reconstructs_the_finished_image(mask, optics, grid, aerial):
    """raw × normalisation_scale reproduces the normal path — the documented
    contract that lets sweeps re-dose without repeating the Abbe sum."""
    raw, peak, clear = compute_aerial_image(mask, optics, grid, return_raw=True)
    assert raw.shape == aerial.shape
    assert peak > 0.0 and clear > 0.0
    scale = normalisation_scale(optics.normalisation, peak, clear, dose=1.0)
    assert np.allclose(raw * scale, aerial, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# extract_cross_section
# ---------------------------------------------------------------------------


def test_cross_section_default_is_centre_column(aerial):
    profile = extract_cross_section(aerial)
    assert profile.shape == (aerial.shape[0],)
    assert np.array_equal(profile, aerial[:, aerial.shape[1] // 2])


def test_cross_section_axis_and_index_selection(aerial):
    row = extract_cross_section(aerial, axis=0, index=5)
    col = extract_cross_section(aerial, axis=1, index=5)
    assert np.array_equal(row, aerial[5, :])
    assert np.array_equal(col, aerial[:, 5])


def test_cross_section_across_lines_shows_modulation(aerial):
    """A profile cut across the L/S pattern must swing between dark and bright."""
    profile = extract_cross_section(aerial, axis=0)  # vary along x, across lines
    assert float(profile.max()) - float(profile.min()) > 0.5


# ---------------------------------------------------------------------------
# Vector imaging path (selected via OpticsConfig.imaging_model)
# ---------------------------------------------------------------------------


def test_vector_path_runs_and_is_physical(mask, grid):
    optics = OpticsConfig(
        wavelength=193e-9,
        NA=1.35,
        n_immersion=1.44,
        sigma_outer=0.85,
        source_grid=5,
        imaging_model="vector",
    )
    aerial = compute_aerial_image(mask, optics, grid, dose=1.0)
    assert aerial.shape == (grid.n_pixels, grid.n_pixels)
    assert np.all(np.isfinite(aerial))
    assert float(aerial.min()) >= -1e-12
    assert abs(float(aerial.max()) - 1.0) < 1e-9  # peak-normalised


# ---------------------------------------------------------------------------
# Validation the code actually performs
# ---------------------------------------------------------------------------


def test_unknown_imaging_model_is_rejected():
    """Refused when the config is built, not several calls later."""
    with pytest.raises(ValueError, match="imaging_model must be"):
        OpticsConfig(source_grid=5, imaging_model="rigorous")


def test_unknown_normalisation_is_rejected():
    with pytest.raises(ValueError, match="normalisation must be"):
        OpticsConfig(source_grid=5, normalisation="bogus")


def test_unknown_polarisation_is_rejected_on_vector_path():
    with pytest.raises(ValueError, match="polarisation must be"):
        OpticsConfig(source_grid=5, imaging_model="vector", polarisation="circular-ish")


def test_na_above_the_image_index_is_rejected():
    """NA = n sin θ, so a dry lens cannot have NA 1.3 — the pair is refused."""
    with pytest.raises(ValueError, match="sin θ > 1"):
        OpticsConfig(NA=1.3)
    OpticsConfig(NA=1.3, n_immersion=1.44)                       # reachable in water


def test_normalisation_scale_rejects_unknown_mode():
    with pytest.raises(ValueError, match="normalisation must be"):
        normalisation_scale("banana", peak=1.0, clear=1.0)
