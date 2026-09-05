"""
Interface tests: roughness metrics through the ``litho_sim.analysis`` API.

Imports come from ``litho_sim.analysis`` (and ``litho_sim.develop`` for the
end-to-end hop) — never the ``stochastics`` module directly.  Synthetic
images with known answers pin the measurement contracts: a straight line has
zero roughness, a planted wiggle measures at its injected size, planted
bridges and breaks count exactly, and featureless input degrades to the
documented NaN/zero-row behaviour instead of a wrong number.  Unit-level
statistics live in ``test_stochastic.py``; this file is about the interface.
"""

from __future__ import annotations

import numpy as np
import pytest

import litho_sim.analysis
from litho_sim.analysis import (
    CDUniformity,
    LineRoughness,
    count_defects,
    edge_positions,
    failure_rate,
    measure_lcdu,
    measure_line_roughness,
)
from litho_sim.core.config import GridConfig, ResistConfig
from litho_sim.develop import stochastic_trials

PIXEL = 4e-9


def straight_line(ny: int = 32, nx: int = 32, lo: int = 12, hi: int = 20) -> np.ndarray:
    img = np.zeros((ny, nx))
    img[:, lo:hi] = 1.0
    return img


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_symbols_exported_at_package_level():
    for name in (
        "measure_line_roughness", "LineRoughness", "measure_lcdu",
        "CDUniformity", "edge_positions", "count_defects", "failure_rate",
    ):
        assert hasattr(litho_sim.analysis, name), f"litho_sim.analysis.{name} missing"
        assert name in litho_sim.analysis.__all__, f"{name} not in __all__"


# ---------------------------------------------------------------------------
# Known-answer geometry
# ---------------------------------------------------------------------------


def test_edge_positions_of_straight_line():
    left, right = edge_positions(straight_line(), PIXEL)
    assert left.shape == right.shape == (32,)
    assert not np.any(np.isnan(left)) and not np.any(np.isnan(right))
    # Constant down the line, sub-pixel interpolated, 8 px apart.
    assert np.allclose(left, left[0]) and np.allclose(right, right[0])
    assert right[0] - left[0] == pytest.approx(8 * PIXEL, abs=0.1 * PIXEL)
    # The 0.5-crossing sits between the last 0 and first 1 pixel centres.
    assert 11.0 * PIXEL < left[0] < 12.0 * PIXEL
    assert 19.0 * PIXEL < right[0] < 20.0 * PIXEL


def test_straight_line_has_zero_roughness():
    r = measure_line_roughness(straight_line(), PIXEL)
    assert isinstance(r, LineRoughness)
    assert r.n_rows == 32
    assert r.ler == pytest.approx(0.0, abs=1e-15)
    assert r.lwr == pytest.approx(0.0, abs=1e-15)
    assert r.cd_mean == pytest.approx(8 * PIXEL, abs=0.1 * PIXEL)
    assert r.ler_3s == 3.0 * r.ler and r.lwr_3s == 3.0 * r.lwr


def test_rigid_wiggle_gives_ler_but_no_lwr():
    """Shift the whole line sideways block-by-block: edges move, width doesn't."""
    img = np.zeros((32, 32))
    for i in range(32):
        shift = 1 if (i // 8) % 2 else 0
        img[i, 12 + shift:20 + shift] = 1.0
    r = measure_line_roughness(img, PIXEL)
    # Positions alternate +-0.5 px around the mean: sigma ~0.5 px.
    assert 0.3 * PIXEL < r.ler < 0.7 * PIXEL
    assert r.lwr == pytest.approx(0.0, abs=1e-15)
    assert r.sigma_left == pytest.approx(r.sigma_right, rel=1e-9)


def test_width_wiggle_gives_lwr():
    """Alternate the width between 8 and 10 px: LWR ~1 px."""
    img = np.zeros((32, 32))
    for i in range(32):
        if (i // 8) % 2:
            img[i, 11:21] = 1.0
        else:
            img[i, 12:20] = 1.0
    r = measure_line_roughness(img, PIXEL)
    assert 0.5 * PIXEL < r.lwr < 1.5 * PIXEL
    assert r.cd_mean == pytest.approx(9 * PIXEL, abs=0.2 * PIXEL)


def test_lcdu_of_known_widths_and_failed_trial():
    trials = np.stack([
        straight_line(lo=12, hi=20),   # 8 px
        straight_line(lo=11, hi=21),   # 10 px
        straight_line(lo=10, hi=22),   # 12 px
        np.zeros((32, 32)),            # nothing printed at all
    ])
    u = measure_lcdu(trials, PIXEL)
    assert isinstance(u, CDUniformity)
    assert u.n_trials == 4
    assert u.n_failed == 1                      # the vanished feature
    assert u.cds.shape == (4,)
    assert u.cd_mean == pytest.approx(10 * PIXEL, abs=0.2 * PIXEL)
    # std of {8, 10, 12} px with ddof=1 is 2 px; lcdu is the 3-sigma value.
    assert u.cd_sigma == pytest.approx(2 * PIXEL, abs=0.3 * PIXEL)
    assert u.lcdu == pytest.approx(3.0 * u.cd_sigma, rel=1e-12)


# ---------------------------------------------------------------------------
# Topological failures
# ---------------------------------------------------------------------------


def test_planted_bridge_and_break_count_exactly():
    ref = np.zeros((48, 48))
    ref[:, 10:18] = 1.0
    ref[:, 30:38] = 1.0

    bridged = ref.copy()
    bridged[22:26, 18:30] = 1.0     # rung between the lines splits the space
    broken = ref.copy()
    broken[22:26, 10:18] = 0.0      # gap in one line splits the resist

    assert not count_defects(ref, ref).any
    d = count_defects(bridged, ref)
    assert (d.bridges, d.breaks) == (1, 0)
    d = count_defects(broken, ref)
    assert (d.bridges, d.breaks) == (0, 1)

    stats = failure_rate(np.stack([ref, bridged, broken]), ref)
    assert stats.n_trials == 3 and stats.n_failed == 2
    assert stats.bridges_total == 1 and stats.breaks_total == 1
    assert stats.rate == pytest.approx(2.0 / 3.0)


def test_all_cleared_trial_is_lcdu_failure_not_defect():
    """The documented split: a vanished feature is measure_lcdu's failure;
    count_defects only sees component *increases* (bridges/breaks)."""
    ref = np.zeros((48, 48))
    ref[:, 10:18] = 1.0
    ref[:, 30:38] = 1.0
    cleared = np.zeros_like(ref)

    assert not count_defects(cleared, ref).any
    stats = failure_rate(np.stack([cleared]), ref)
    assert stats.n_failed == 0          # topologically quiet...
    u = measure_lcdu(np.stack([cleared]), PIXEL)
    assert u.n_failed == 1              # ...but caught here, by contract


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_roughness_of_featureless_image_is_nan_not_wrong():
    for img in (np.zeros((32, 32)), np.ones((32, 32))):
        r = measure_line_roughness(img, PIXEL)
        assert r.n_rows == 0
        for value in (r.ler, r.lwr, r.cd_mean, r.corr_length):
            assert np.isnan(value)
    left, right = edge_positions(np.zeros((32, 32)), PIXEL)
    assert np.all(np.isnan(left)) and np.all(np.isnan(right))


def test_failure_rate_of_empty_batch_is_nan():
    ref = straight_line()
    stats = failure_rate(np.zeros((0, 32, 32)), ref)
    assert stats.n_trials == 0
    assert np.isnan(stats.rate)


# ---------------------------------------------------------------------------
# End to end: develop-package output feeds analysis-package input
# ---------------------------------------------------------------------------


def test_metrics_on_real_stochastic_trials():
    cfg = ResistConfig(
        dill_A=0.0, dill_B=4.0, dill_C=0.05, thickness=50e-9,
        dose_nominal=30.0, pag_density=3.0e26, quencher_ratio=0.15,
        bake_time=8.0, D_acid=3.0e-18, k_quench=5.0, k_amp=0.4,
        mack_Mth=0.6, mack_n=8, mack_Rmax=150.0, mack_Rmin=0.001,
        develop_time=5.0,
    )
    grid = GridConfig(n_pixels=48, pixel_size=4e-9)
    x = np.arange(grid.n_pixels) * grid.pixel_size
    period = grid.n_pixels * grid.pixel_size / 2
    aerial = np.tile(0.5 + 0.45 * np.cos(2 * np.pi * x / period),
                     (grid.n_pixels, 1))

    res = stochastic_trials(aerial, cfg, grid, 13.5e-9, trials=3, seed=7)

    u = measure_lcdu(res.resist, grid.pixel_size)
    assert u.n_trials == 3
    assert u.n_failed + int(np.sum(u.cds > 0.0)) == 3
    assert np.isfinite(u.cd_mean) and 0.0 < u.cd_mean < grid.grid_size

    r = measure_line_roughness(res.resist[0], grid.pixel_size)
    assert r.n_rows >= 2
    assert np.isfinite(r.ler) and r.ler >= 0.0
    assert np.isfinite(r.lwr) and r.lwr >= 0.0

    stats = failure_rate(res.resist, res.resist[0])
    assert stats.n_trials == 3
    assert 0.0 <= stats.rate <= 1.0
