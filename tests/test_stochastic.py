"""
Stochastic printing: correlated roughness, Monte-Carlo trials, and metrics.

The roughness generator is checked for the property that justified replacing
the old edge-flip model: the σ you ask for is the σ you measure, and the
edges carry a correlation length. The trial runner is checked for the
physics headline — EUV trials scatter, ArF trials don't — and the metrics
against constructions whose answers are known by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.analysis.stochastics import (
    count_defects,
    edge_positions,
    failure_rate,
    measure_lcdu,
    measure_line_roughness,
)
from litho_sim.core.config import GridConfig, ResistConfig
from litho_sim.develop.stochastic import (
    add_edge_roughness,
    correlated_noise,
    stochastic_trials,
)

PX = 4e-9


def _stripe(n_rows: int = 128, n_cols: int = 64, width: int = 24) -> np.ndarray:
    """A vertical resist line, centred."""
    img = np.zeros((n_rows, n_cols))
    lo = (n_cols - width) // 2
    img[:, lo:lo + width] = 1.0
    return img


# ---------------------------------------------------------------------------
# Correlated noise and cosmetic roughness
# ---------------------------------------------------------------------------


def test_correlated_noise_is_unit_variance():
    rng = np.random.default_rng(0)
    f = correlated_noise((256, 256), corr_length_px=6.0, rng=rng)
    assert float(f.std()) == pytest.approx(1.0, rel=1e-6)
    assert float(f.mean()) == pytest.approx(0.0, abs=0.05)


def test_correlated_noise_actually_correlates():
    """Neighbouring samples of filtered noise agree; white noise's don't."""
    rng = np.random.default_rng(1)
    f = correlated_noise((256, 256), corr_length_px=6.0, rng=rng)
    w = correlated_noise((256, 256), corr_length_px=0.0, rng=rng)
    corr_f = float(np.mean(f[:, :-1] * f[:, 1:]))
    corr_w = float(np.mean(w[:, :-1] * w[:, 1:]))
    assert corr_f > 0.8
    assert abs(corr_w) < 0.05


def test_edge_roughness_delivers_requested_sigma():
    """σ in, σ out — the property the edge-flip model never had.

    128 correlated rows carry ~25 independent samples, so the estimate
    itself is ~15 % uncertain; the tolerance covers estimator noise plus
    half-pixel quantisation of the binary edge.
    """
    sigma, xi = 8e-9, 20e-9
    rough = add_edge_roughness(_stripe(), sigma, xi, PX, seed=42)
    stats = measure_line_roughness(rough, PX)
    assert stats.ler == pytest.approx(sigma, rel=0.4)
    assert stats.corr_length == pytest.approx(xi, rel=0.75)


def test_edge_roughness_noop_cases():
    stripe = _stripe()
    assert add_edge_roughness(stripe, 0.0, 20e-9, PX) is stripe
    uniform = np.ones((32, 32))
    assert add_edge_roughness(uniform, 5e-9, 20e-9, PX) is uniform


def test_edge_roughness_is_binary_and_reproducible():
    a = add_edge_roughness(_stripe(), 6e-9, 20e-9, PX, seed=7)
    b = add_edge_roughness(_stripe(), 6e-9, 20e-9, PX, seed=7)
    assert set(np.unique(a)).issubset({0.0, 1.0})
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Metrics on constructions with known answers
# ---------------------------------------------------------------------------


def test_edge_positions_flags_broken_rows():
    stripe = _stripe()
    stripe[50, :] = 0.0  # a broken cutline
    left, _right = edge_positions(stripe, PX)
    assert np.isnan(left[50])
    assert np.isfinite(left[49])


def test_perfect_line_has_zero_roughness():
    stats = measure_line_roughness(_stripe(), PX)
    # Interpolated crossings agree to float rounding, not bit-exactly.
    assert stats.ler == pytest.approx(0.0, abs=1e-15)
    assert stats.lwr == pytest.approx(0.0, abs=1e-15)
    assert stats.cd_mean == pytest.approx(24 * PX)
    assert stats.n_rows == 128


def test_lcdu_of_known_widths():
    trials = np.stack([_stripe(width=w) for w in (20, 22, 24)])
    u = measure_lcdu(trials, PX)
    assert u.cd_mean == pytest.approx(22 * PX)
    assert u.cd_sigma == pytest.approx(float(np.std([20, 22, 24], ddof=1)) * PX)
    assert u.n_failed == 0
    assert u.lcdu == pytest.approx(3.0 * u.cd_sigma)


def test_lcdu_counts_vanished_features_as_failures():
    trials = np.stack([_stripe(), np.zeros((128, 64))])
    u = measure_lcdu(trials, PX)
    assert u.n_failed == 1
    assert u.cd_mean == pytest.approx(24 * PX)


def test_defect_counting_sees_bridges_and_breaks():
    # Reference: two vertical lines.
    ref = np.zeros((64, 64))
    ref[:, 10:20] = 1.0
    ref[:, 40:50] = 1.0

    bridged = ref.copy()
    bridged[30:34, 20:40] = 1.0  # rung between the lines: splits the space
    d = count_defects(bridged, ref)
    assert d.bridges == 1 and d.breaks == 0

    broken = ref.copy()
    broken[30:34, 10:20] = 0.0  # gap in one line: splits the resist
    d = count_defects(broken, ref)
    assert d.breaks == 1 and d.bridges == 0

    stats = failure_rate(np.stack([ref, bridged, broken]), ref)
    assert stats.n_failed == 2
    assert stats.rate == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# The physical chain, end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def car_cfg() -> ResistConfig:
    """A CAR tuned for test speed: short bake, gentle quench kinetics."""
    return ResistConfig(
        dill_A=0.0, dill_B=4.0, dill_C=0.05, thickness=50e-9,
        dose_nominal=30.0, pag_density=3.0e26, quencher_ratio=0.15,
        bake_time=8.0, D_acid=3.0e-18, k_quench=5.0, k_amp=0.4,
        mack_Mth=0.6, mack_n=8, mack_Rmax=150.0, mack_Rmin=0.001,
        develop_time=5.0,
    )


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=48, pixel_size=4e-9)


@pytest.fixture(scope="module")
def soft_aerial(grid) -> np.ndarray:
    """A smooth line/space-like image with finite log-slope at the edges."""
    x = np.arange(grid.n_pixels) * grid.pixel_size
    profile = 0.5 + 0.45 * np.cos(2 * np.pi * x / (grid.n_pixels * grid.pixel_size / 2))
    return np.tile(profile, (grid.n_pixels, 1))


def test_stochastic_trials_shapes_and_binariness(soft_aerial, car_cfg, grid):
    res = stochastic_trials(soft_aerial, car_cfg, grid, wavelength=13.5e-9,
                            trials=4, seed=0)
    assert res.resist.shape == (4, 48, 48)
    assert res.trials == 4
    assert set(np.unique(res.resist)).issubset({0.0, 1.0})
    assert res.protected.shape == (48, 48)
    assert res.sample.mean_pag > 10


def test_stochastic_trials_reproducible(soft_aerial, car_cfg, grid):
    a = stochastic_trials(soft_aerial, car_cfg, grid, 13.5e-9, trials=2, seed=3)
    b = stochastic_trials(soft_aerial, car_cfg, grid, 13.5e-9, trials=2, seed=3)
    assert np.array_equal(a.resist, b.resist)


def test_euv_trials_scatter_and_arf_trials_dont(soft_aerial, car_cfg, grid):
    """The headline: identical chemistry and dose, photon count decides.

    Trial-to-trial disagreement (fraction of pixels differing between two
    realisations) must be markedly larger at 13.5 nm than at 193 nm.  PAG
    loading is raised so the shared molecular-counting floor doesn't blur
    the comparison — this test is about photons.
    """
    import dataclasses

    dense = dataclasses.replace(car_cfg, pag_density=3.0e27)
    # This is a statement about photon counting, so the development noise —
    # which is wavelength-blind by construction — is switched off for it.
    dense = dataclasses.replace(dense, dissolution_sigma=0.0)
    euv = stochastic_trials(soft_aerial, dense, grid, 13.5e-9, trials=2, seed=9)
    arf = stochastic_trials(soft_aerial, dense, grid, 193e-9, trials=2, seed=9)

    def disagreement(r):
        return float(np.mean(r.resist[0] != r.resist[1]))

    assert disagreement(euv) > 3.0 * disagreement(arf)
