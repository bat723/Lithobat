"""Interface-level tests for the PEB diffusion subfeature.

Everything here goes through the public package surface — ``from
litho_sim.bake import apply_peb, apply_peb_3d`` — and mirrors how the live
consumers call it:

* ``develop/resist.py`` (mack path) and ``analysis/process_window.py``:
  ``apply_peb(pac, cfg.diffusion_sigma, grid.pixel_size)``
* ``develop/resist3d.py``, ``app/pipeline.py``, ``patterning/steps.py``:
  ``apply_peb_3d(pac, resist_cfg, grid_cfg)``

Covered: the standing-wave-smoothing happy path in 2-D and 3-D, the
sigma-to-pixel conversion (more physical sigma on the same grid smooths
more; equal sigma/pixel ratios at different physical scales agree), the
anisotropic ``sigma_z`` path, the zero-sigma no-op-copy contract, and input
validation.  Unit-level PEB tests (whole-field mean conservation on random
noise) already live in ``tests/test_resist.py`` and are not repeated here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import litho_sim.bake as bake_pkg
from litho_sim.bake import apply_peb, apply_peb_3d
from litho_sim.core.config import GridConfig, ResistConfig

GRID = GridConfig(n_pixels=64, pixel_size=4e-9, dz=2e-9)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ripple_2d(n: int = 64, period_px: int = 8, axis: int = 0) -> np.ndarray:
    """A standing-wave-like latent image: cosine ripple along one axis.

    Values stay inside [0.2, 0.8] so the [0, 1] clip in the bake is inert
    and dose bookkeeping is exact.
    """
    wave = 0.5 + 0.3 * np.cos(2.0 * np.pi * np.arange(n) / period_px)
    if axis == 0:
        return np.tile(wave[:, None], (1, n))
    return np.tile(wave[None, :], (n, 1))


def ripple_3d_z(nz: int = 56, nxy: int = 16, period_vox: int = 28) -> np.ndarray:
    """A volume with vertical standing-wave ripple, laterally uniform.

    period_vox=28 at dz=2 nm is the ~57 nm lambda/(2 n) period at 193 nm.
    """
    wave = 0.5 + 0.3 * np.cos(2.0 * np.pi * np.arange(nz) / period_vox)
    return np.tile(wave[:, None, None], (1, nxy, nxy))


def amp(profile: np.ndarray) -> float:
    """Peak-to-peak amplitude of a 1-D profile."""
    return float(profile.max() - profile.min())


# ---------------------------------------------------------------------------
# Wiring: the public package surface
# ---------------------------------------------------------------------------


def test_public_interface_exports_peb():
    """Both PEB entry points are part of the declared bake package API."""
    assert "apply_peb" in bake_pkg.__all__
    assert "apply_peb_3d" in bake_pkg.__all__
    assert callable(bake_pkg.apply_peb)
    assert callable(bake_pkg.apply_peb_3d)


# ---------------------------------------------------------------------------
# Happy path: standing-wave smoothing, consumer-shaped calls
# ---------------------------------------------------------------------------


def test_peb_smooths_standing_wave_ripple_2d():
    """A 32 nm-period ripple is nearly erased by a 20 nm diffusion length,
    while interior dose is conserved — the physical job of the bake."""
    pac = ripple_2d(n=64, period_px=8, axis=0)
    cfg = ResistConfig(diffusion_sigma=20e-9)

    out = apply_peb(pac, cfg.diffusion_sigma, GRID.pixel_size)

    assert out.shape == pac.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    # amplitude measured on the interior (>= 3 sigma_px from the edges) so
    # boundary-mode transients don't count as ripple
    prof_before = pac.mean(axis=1)[16:48]
    prof_after = out.mean(axis=1)[16:48]
    assert amp(prof_after) < 0.05 * amp(prof_before), (
        "20 nm PEB left visible standing waves"
    )
    interior = np.s_[16:48, 16:48]
    assert out[interior].mean() == pytest.approx(pac[interior].mean(), abs=0.01)


def test_larger_physical_sigma_smooths_more():
    """Same grid, growing diffusion length: residual ripple must shrink
    monotonically — the sigma-to-pixel conversion is doing its job."""
    pac = ripple_2d(n=64, period_px=8, axis=1)
    amps = []
    for sigma in (2e-9, 8e-9, 32e-9):
        out = apply_peb(pac, sigma, GRID.pixel_size)
        amps.append(amp(out.mean(axis=0)))
    assert amps[0] > amps[1] > amps[2], f"ripple did not shrink with sigma: {amps}"
    assert amps[0] < amp(pac.mean(axis=0))  # even the smallest sigma smooths some


def test_sigma_pixel_scaling_is_consistent():
    """The blur depends only on sigma/pixel_size: the same latent image at
    twice the physical scale (double sigma, double pixel) bakes identically."""
    pac = ripple_2d(n=64, period_px=8, axis=0)
    out_fine = apply_peb(pac, 16e-9, 4e-9)
    out_coarse = apply_peb(pac, 32e-9, 8e-9)
    np.testing.assert_allclose(out_coarse, out_fine, rtol=1e-9, atol=1e-12)


def test_peb_3d_smooths_standing_waves_in_z():
    """The 3-D bake washes out the vertical standing-wave ripple."""
    vol = ripple_3d_z(nz=56, nxy=16, period_vox=28)
    cfg = ResistConfig(diffusion_sigma=20e-9)
    grid = GridConfig(n_pixels=16, pixel_size=4e-9, dz=2e-9)

    out = apply_peb_3d(vol, cfg, grid)

    assert out.shape == vol.shape
    z_before = vol.mean(axis=(1, 2))
    z_after = out.mean(axis=(1, 2))
    # interior slices: keep clear of the mode="nearest" boundary
    assert amp(z_after[10:46]) < 0.25 * amp(z_before[10:46])
    # dose conservation, measured over exactly one full ripple period so the
    # pre-bake window mean equals the field mean
    assert out[14:42].mean() == pytest.approx(vol[14:42].mean(), abs=0.02)


# ---------------------------------------------------------------------------
# Anisotropy: sigma_z decouples vertical from lateral diffusion
# ---------------------------------------------------------------------------


def test_sigma_z_zero_preserves_vertical_ripple():
    """Lateral-only bake (sigma_z=0) of a laterally uniform volume is an
    identity: the z standing waves survive untouched."""
    vol = ripple_3d_z(nz=56, nxy=16, period_vox=28)
    cfg = ResistConfig(diffusion_sigma=20e-9)
    grid = GridConfig(n_pixels=16, pixel_size=4e-9, dz=2e-9)

    out = apply_peb_3d(vol, cfg, grid, sigma_z=0.0)

    np.testing.assert_allclose(out, vol, atol=1e-9)


def test_lateral_zero_with_sigma_z_smooths_only_z():
    """The complementary case: lateral sigma 0 with sigma_z > 0 leaves a
    z-uniform lateral ripple untouched, while an isotropic bake smooths it."""
    n = 64
    wave = 0.5 + 0.3 * np.cos(2.0 * np.pi * np.arange(n) / 8)
    vol = np.tile(wave[None, :, None], (8, 1, 8))  # ripple along y only
    grid = GridConfig(n_pixels=n, pixel_size=4e-9, dz=2e-9)

    out_z_only = apply_peb_3d(vol, ResistConfig(diffusion_sigma=0.0), grid, sigma_z=40e-9)
    np.testing.assert_allclose(out_z_only, vol, atol=1e-9)

    out_iso = apply_peb_3d(vol, ResistConfig(diffusion_sigma=20e-9), grid)
    # interior of the y-profile, clear of the mode="nearest" edge transients
    y_prof = out_iso[4].mean(axis=1)[20:44]
    assert amp(y_prof) < 0.05 * amp(wave[20:44])


# ---------------------------------------------------------------------------
# Zero-sigma no-op contract
# ---------------------------------------------------------------------------


def test_zero_sigma_is_noop_copy_2d():
    """sigma=0 returns the input values unchanged — and as a *copy*, which
    app/compute.py relies on to draw the pre/post-bake panes independently."""
    pac = ripple_2d(n=32, period_px=8)
    keep = pac.copy()

    out = apply_peb(pac, 0.0, GRID.pixel_size)

    assert np.array_equal(out, pac)
    assert out is not pac
    assert np.array_equal(pac, keep)  # input not mutated

    # sub-milli-pixel sigma is treated as zero too (the documented cutoff)
    tiny = apply_peb(pac, 1e-15, GRID.pixel_size)
    assert np.array_equal(tiny, pac)


def test_zero_sigma_is_noop_copy_3d():
    vol = ripple_3d_z(nz=20, nxy=8, period_vox=10)
    cfg = ResistConfig(diffusion_sigma=0.0)
    grid = GridConfig(n_pixels=8, pixel_size=4e-9, dz=2e-9)

    out = apply_peb_3d(vol, cfg, grid)

    assert np.array_equal(out, vol)
    assert out is not vol


# ---------------------------------------------------------------------------
# Validation: garbage in, ValueError out
# ---------------------------------------------------------------------------


def test_apply_peb_rejects_bad_inputs():
    pac2d = ripple_2d(n=16, period_px=8)
    vol = np.zeros((4, 16, 16))

    with pytest.raises(ValueError, match="2-D"):
        apply_peb(vol, 20e-9, 4e-9)  # wrong dimensionality
    with pytest.raises(ValueError, match="diffusion_sigma"):
        apply_peb(pac2d, -20e-9, 4e-9)  # negative diffusion length
    with pytest.raises(ValueError, match="pixel_size"):
        apply_peb(pac2d, 20e-9, 0.0)  # degenerate grid
    with pytest.raises(ValueError, match="pixel_size"):
        apply_peb(pac2d, 20e-9, -4e-9)


def test_apply_peb_3d_rejects_bad_inputs():
    pac2d = ripple_2d(n=16, period_px=8)
    vol = np.zeros((4, 16, 16))
    grid = GridConfig(n_pixels=16, pixel_size=4e-9, dz=2e-9)

    with pytest.raises(ValueError, match="3-D"):
        apply_peb_3d(pac2d, ResistConfig(diffusion_sigma=20e-9), grid)
    with pytest.raises(ValueError, match="3-D"):
        # must raise even on the zero-sigma path, not silently no-op
        apply_peb_3d(pac2d, ResistConfig(diffusion_sigma=0.0), grid)
    with pytest.raises(ValueError, match="diffusion lengths"):
        apply_peb_3d(vol, ResistConfig(diffusion_sigma=-1e-9), grid)
    with pytest.raises(ValueError, match="diffusion lengths"):
        apply_peb_3d(vol, ResistConfig(diffusion_sigma=20e-9), grid, sigma_z=-1e-9)
    with pytest.raises(ValueError, match="spacings"):
        apply_peb_3d(vol, ResistConfig(diffusion_sigma=20e-9),
                     GridConfig(n_pixels=16, pixel_size=4e-9, dz=0.0))
    with pytest.raises(ValueError, match="spacings"):
        apply_peb_3d(vol, ResistConfig(diffusion_sigma=20e-9),
                     GridConfig(n_pixels=16, pixel_size=0.0, dz=2e-9))
