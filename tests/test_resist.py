"""
Unit tests for the resist module.

These cover the pipeline that :func:`litho_sim.develop.resist.simulate_resist`
orchestrates.  Before the Phase-0 repair that function referenced four
private helpers that existed nowhere in the tree, so *every* call raised
``NameError`` — these tests exist to keep that from regressing.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.bake.peb import apply_peb
from litho_sim.core.config import GridConfig, SimulationConfig
from litho_sim.develop.resist import (
    dill_exposure,
    mack_development_rate,
    measure_cd_1d,
    simulate_resist,
    threshold_development,
)
from litho_sim.expose.aerial_image import compute_aerial_image
from litho_sim.mask.patterns import lines_and_spaces


@pytest.fixture(scope="module")
def cfg() -> SimulationConfig:
    c = SimulationConfig.from_tech_node("ArF")
    c.grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    return c


@pytest.fixture(scope="module")
def aerial(cfg) -> np.ndarray:
    mask = lines_and_spaces(
        cfg.grid.n_pixels, cfg.grid.pixel_size, pitch=200e-9, cd=100e-9
    )
    return compute_aerial_image(mask, cfg.optics, cfg.grid, dose=1.0)


# ---------------------------------------------------------------------------
# simulate_resist — the previously dead orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["threshold", "mack", "car"])
def test_simulate_resist_runs(aerial, cfg, model):
    """Both development models must run and return three same-shaped arrays."""
    pac_exp, pac_peb, resist = simulate_resist(
        aerial, cfg.resist, cfg.grid, model=model
    )
    for arr in (pac_exp, pac_peb, resist):
        assert arr.shape == aerial.shape
        assert np.isfinite(arr).all()


@pytest.mark.parametrize("model", ["threshold", "mack", "car"])
def test_simulate_resist_is_binary(aerial, cfg, model):
    """The developed image is binary: 1 = resist remaining, 0 = cleared."""
    _, _, resist = simulate_resist(aerial, cfg.resist, cfg.grid, model=model)
    assert set(np.unique(resist)).issubset({0.0, 1.0})


@pytest.mark.parametrize("model", ["threshold", "mack", "car"])
def test_simulate_resist_prints_a_pattern(aerial, cfg, model):
    """A resolvable L/S must actually pattern — not clear or retain everywhere.

    The Mack path in particular is degenerate unless the Dill exposure gets a
    dose in real mJ/cm² units (``ResistConfig.dose_nominal``); without it the
    default C of 0.04 barely bleaches and nothing ever develops out.
    """
    _, _, resist = simulate_resist(aerial, cfg.resist, cfg.grid, model=model)
    remaining = float(resist.mean())
    assert 0.05 < remaining < 0.95, (
        f"{model}: {remaining:.1%} of the field remains — expected a pattern."
    )


def test_simulate_resist_rejects_unknown_model(aerial, cfg):
    with pytest.raises(ValueError, match="Unknown resist model"):
        simulate_resist(aerial, cfg.resist, cfg.grid, model="nope")


def test_simulate_resist_tone_inverts(aerial, cfg):
    """Negative tone is the complement of positive tone."""
    import dataclasses

    pos = dataclasses.replace(cfg.resist, tone="positive")
    neg = dataclasses.replace(cfg.resist, tone="negative")
    _, _, r_pos = simulate_resist(aerial, pos, cfg.grid, model="threshold")
    _, _, r_neg = simulate_resist(aerial, neg, cfg.grid, model="threshold")
    assert np.array_equal(r_pos, 1.0 - r_neg)


def test_peb_diffusion_actually_diffuses(aerial, cfg):
    """A non-zero diffusion length must change the latent image.

    The Streamlit app has always exposed a 'PEB Diffusion' slider that fed a
    ResistConfig which the resist step then ignored, making the control a
    no-op. This asserts the library path does not have that hole.
    """
    import dataclasses

    sharp = dataclasses.replace(cfg.resist, diffusion_sigma=0.0)
    blurred = dataclasses.replace(cfg.resist, diffusion_sigma=30e-9)
    _, peb_sharp, _ = simulate_resist(aerial, sharp, cfg.grid, model="mack")
    _, peb_blur, _ = simulate_resist(aerial, blurred, cfg.grid, model="mack")
    assert np.std(peb_blur) < np.std(peb_sharp), "PEB diffusion did not smooth the latent image."


# ---------------------------------------------------------------------------
# Individual physics steps
# ---------------------------------------------------------------------------


def test_dill_exposure_bleaches_monotonically():
    """More light must leave less photoactive compound."""
    intensity = np.linspace(0.0, 1.0, 32).reshape(1, -1)
    M = dill_exposure(intensity, dose=30.0, dill_C=0.04)
    assert np.all(np.diff(M[0]) <= 0.0)
    assert M.max() <= 1.0 and M.min() >= 0.0


def test_apply_peb_conserves_mean():
    """Gaussian diffusion redistributes acid but does not create or destroy it."""
    rng = np.random.default_rng(0)
    pac = rng.random((64, 64))
    out = apply_peb(pac, diffusion_sigma=20e-9, pixel_size=4e-9)
    assert out.mean() == pytest.approx(pac.mean(), rel=1e-3)


def test_apply_peb_noop_for_zero_sigma():
    pac = np.random.default_rng(1).random((16, 16))
    assert np.array_equal(apply_peb(pac, 0.0, 4e-9), pac)


def test_mack_rate_increases_with_exposure():
    """Dissolution rate rises as PAC is consumed (M → 0)."""
    M = np.linspace(1.0, 0.0, 32).reshape(1, -1)
    R = mack_development_rate(M, Rmax=100.0, Rmin=0.01, Mth=0.5, n=4)
    assert np.all(np.diff(R[0]) >= 0.0)


def test_threshold_development_tones():
    latent = np.array([[0.1, 0.9]])
    pos = threshold_development(latent, threshold=0.5, tone="positive")
    neg = threshold_development(latent, threshold=0.5, tone="negative")
    assert np.array_equal(pos, [[1.0, 0.0]])
    assert np.array_equal(neg, [[0.0, 1.0]])
    with pytest.raises(ValueError):
        threshold_development(latent, 0.5, tone="sideways")


# ---------------------------------------------------------------------------
# CD measurement
# ---------------------------------------------------------------------------


def test_measure_cd_1d_is_exact():
    """A feature spanning exactly k pixels must measure exactly k pixels.

    np.diff indices name the gap *before* a transition, so the width is
    (falling - rising), not (falling - rising + 1). The old +1 made every CD
    one pixel too wide.
    """
    for width in (11, 40, 41):
        profile = np.zeros(100)
        start = (100 - width) // 2
        profile[start:start + width] = 1.0
        cd = measure_cd_1d(profile, pixel_size=2e-9)
        assert cd == pytest.approx(width * 2e-9), f"width={width}"
