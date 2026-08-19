"""
Unit tests for the aerial image computation module.

Tests verify physical correctness and numerical sanity, not pixel-exact
values (which depend on discretisation).  Run with::

    pytest tests/test_aerial_image.py -v
"""

from __future__ import annotations

# Ensure src/ is on the path when running tests directly
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, OpticsConfig, SimulationConfig
from litho_sim.expose.aerial_image import compute_aerial_image, extract_cross_section
from litho_sim.mask.patterns import lines_and_spaces

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg() -> SimulationConfig:
    """Small-grid config for fast unit tests."""
    return SimulationConfig(
        optics=OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.85),
        grid=GridConfig(n_pixels=128, pixel_size=4e-9),
    )


@pytest.fixture
def ls_mask(small_cfg: SimulationConfig) -> np.ndarray:
    cfg = small_cfg
    return lines_and_spaces(
        cfg.grid.n_pixels,
        cfg.grid.pixel_size,
        pitch=200e-9,
        cd=100e-9,
    )


# ---------------------------------------------------------------------------
# Shape / dtype tests
# ---------------------------------------------------------------------------


def test_aerial_image_shape(small_cfg, ls_mask):
    """Output shape must match the grid."""
    n = small_cfg.grid.n_pixels
    aerial = compute_aerial_image(ls_mask, small_cfg.optics, small_cfg.grid)
    assert aerial.shape == (n, n), f"Expected ({n},{n}), got {aerial.shape}"


def test_aerial_image_dtype(small_cfg, ls_mask):
    """Output must be float64."""
    aerial = compute_aerial_image(ls_mask, small_cfg.optics, small_cfg.grid)
    assert aerial.dtype == np.float64


def test_aerial_image_range(small_cfg, ls_mask):
    """Intensity must be in [0, dose]."""
    dose = 1.2
    aerial = compute_aerial_image(ls_mask, small_cfg.optics, small_cfg.grid, dose=dose)
    assert float(aerial.min()) >= -1e-9, "Negative intensities found."
    assert float(aerial.max()) <= dose + 1e-9, f"Max > dose ({dose}): {aerial.max()}"


def test_aerial_image_normalised_at_dose_one(small_cfg, ls_mask):
    """Peak intensity should be ≈ dose (normalised to peak)."""
    aerial = compute_aerial_image(ls_mask, small_cfg.optics, small_cfg.grid, dose=1.0)
    assert abs(float(aerial.max()) - 1.0) < 0.05, "Peak deviates more than 5% from dose."


# ---------------------------------------------------------------------------
# Defocus effect tests
# ---------------------------------------------------------------------------


def test_defocus_reduces_contrast(small_cfg, ls_mask):
    """Large defocus should reduce aerial image contrast."""
    cfg_focus = small_cfg
    aerial_focus = compute_aerial_image(ls_mask, cfg_focus.optics, cfg_focus.grid)

    import copy
    defocused_optics = copy.replace(cfg_focus.optics, defocus=300e-9)
    aerial_defocus = compute_aerial_image(ls_mask, defocused_optics, cfg_focus.grid)

    contrast_focus = float(aerial_focus.max() - aerial_focus.min())
    contrast_defocus = float(aerial_defocus.max() - aerial_defocus.min())
    assert contrast_defocus < contrast_focus, (
        f"Defocused contrast ({contrast_defocus:.4f}) should be < "
        f"in-focus contrast ({contrast_focus:.4f})."
    )


# ---------------------------------------------------------------------------
# Cross-section extraction
# ---------------------------------------------------------------------------


def test_extract_cross_section_shape(small_cfg, ls_mask):
    n = small_cfg.grid.n_pixels
    aerial = compute_aerial_image(ls_mask, small_cfg.optics, small_cfg.grid)
    cs = extract_cross_section(aerial, axis=1)
    assert cs.shape == (n,), f"Expected ({n},), got {cs.shape}"


def test_cross_section_axis0_axis1_differ():
    """Cross-sections along different axes differ for non-symmetric masks."""
    cfg = SimulationConfig(
        optics=OpticsConfig(wavelength=193e-9, NA=0.93),
        grid=GridConfig(n_pixels=64, pixel_size=4e-9),
    )
    mask = lines_and_spaces(64, 4e-9, pitch=200e-9, cd=100e-9, orientation="vertical")
    aerial = compute_aerial_image(mask, cfg.optics, cfg.grid)

    cs0 = extract_cross_section(aerial, axis=0)  # row at centre → varies for vertical L/S
    cs1 = extract_cross_section(aerial, axis=1)  # col at centre → constant for vertical L/S

    # Vertical lines vary along x, so the centre *row* is the one that carries
    # the pattern; the centre column cuts along a single line and is flat.
    assert float(np.std(cs0)) > float(np.std(cs1)), (
        "Expected the axis=0 (row) cross-section to vary more for vertical lines."
    )



# ---------------------------------------------------------------------------
# Illumination tilt sign convention
# ---------------------------------------------------------------------------


def test_a_source_point_tilts_the_mask_the_other_way():
    """A source point at pupil +σ illuminates the mask with tilt **−**σ·NA/λ.

    ``_pupil_geometry`` evaluates the pupil at ``f − f_s`` while leaving the
    mask spectrum alone.  Shifting the pupil by ``+f_s`` is equivalent to
    shifting the *spectrum* by ``−f_s``, i.e. to illuminating the mask with a
    plane wave ``exp(−i2π f_s·x)`` — the opposite handedness to the usual
    textbook statement, and to this module's own docstring derivation.

    It is invisible today: every built-in source is centrosymmetric, so the
    Abbe sum pairs ``+f_s`` with ``−f_s`` and the handedness cancels.  It stops
    being invisible under a thick-mask model, where the near field is computed
    *at* an incidence angle and the sign sets which way the printed feature
    shifts.  So it is pinned here, against the real engine, rather than
    rediscovered later from a placement error of the wrong sign.

    The offset is a whole number of frequency bins so the tilt is exactly
    representable on the grid, and the mask is asymmetric so the two signs are
    genuinely distinguishable.  The field is deliberately wide (2 µm): the
    pupil radius is ``NA/λ`` in frequency, so a narrow field puts fewer than
    one frequency bin inside the aperture and every image comes out flat.
    """
    n, dx = 64, 32e-9  # 2 µm field → pupil radius ≈ 9.9 bins
    rng = np.random.default_rng(1)
    mask = rng.random((n, n))  # asymmetric: handedness is observable

    # One source point, placed off-axis on both axes by a whole number of bins.
    shift_x, shift_y = 3, -2
    optics = OpticsConfig(
        wavelength=193e-9, NA=0.93, source_grid=3, normalisation="none",
    )
    lam, NA = optics.wavelength, optics.NA
    fs_x, fs_y = shift_x / (n * dx), shift_y / (n * dx)

    # Drive the engine's own pupil, so this tests the shipped code path.
    from litho_sim.core.utils import make_freq_grid
    from litho_sim.expose.aerial_image import _build_fft_pupil, _pupil_geometry

    FX, FY = make_freq_grid(n, dx)
    RHO, PHI, inside = _pupil_geometry(FX, FY, lam, NA, fs_x, fs_y, need_phi=True)
    pupil = _build_fft_pupil(
        RHO, PHI, inside, lam, NA, optics.defocus, optics.image_index,
        {4: 0.05}, exact_defocus=True,   # an asymmetric-in-ρ aberration, for bite
    )
    engine = np.abs(np.fft.ifft2(np.fft.fft2(mask) * pupil)) ** 2

    # The direct formulation: tilt the mask, hold the pupil on axis.
    ix = np.arange(n)
    X, Y = np.meshgrid(ix, ix)
    RHO0, PHI0, inside0 = _pupil_geometry(FX, FY, lam, NA, 0.0, 0.0, need_phi=True)
    on_axis = _build_fft_pupil(
        RHO0, PHI0, inside0, lam, NA, optics.defocus, optics.image_index,
        {4: 0.05}, exact_defocus=True,
    )

    def imaged(sign: int) -> np.ndarray:
        tilt = np.exp(sign * 2j * np.pi * (shift_x * X + shift_y * Y) / n)
        return np.abs(np.fft.ifft2(np.fft.fft2(mask * tilt) * on_axis)) ** 2

    scale = float(engine.max())
    assert np.abs(engine - imaged(-1)).max() < 1e-12 * scale, (
        "The engine's source offset must correspond to a mask tilt of "
        "exp(-i2pi f_s.x)."
    )
    assert np.abs(engine - imaged(+1)).max() > 1e-3 * scale, (
        "The two tilt signs must be distinguishable, or this test proves nothing."
    )
