"""
Interface tests: stochastic printing through the ``litho_sim.develop`` API.

Imports come from ``litho_sim.develop`` (and ``litho_sim.analysis`` for the
end-to-end hop) — never the ``stochastic`` module directly — so these tests
pin the public surface: ``stochastic_trials`` reachable and reproducible from
package level, ``add_edge_roughness`` honouring its no-op contracts, and
``simulate_resist``'s ``use_stochastic`` knob actually consumed on both the
``"threshold"`` and ``"car"`` paths.  Unit-level statistics (noise spectra,
EUV-vs-ArF scatter) are covered in ``test_stochastic.py``.

Parameters are tuned for speed: 48 px grids, <=3 trials, an 8 s bake with
gentle quench kinetics.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import litho_sim.develop
from litho_sim.analysis import measure_lcdu
from litho_sim.core.config import GridConfig, ResistConfig
from litho_sim.develop import (
    StochasticResult,
    add_edge_roughness,
    simulate_resist,
    stochastic_trials,
)

EUV = 13.5e-9


@pytest.fixture(scope="module")
def car_cfg() -> ResistConfig:
    """A CAR tuned for suite speed: short bake, gentle quench kinetics."""
    return ResistConfig(
        dill_A=0.0, dill_B=4.0, dill_C=0.05, thickness=50e-9,
        dose_nominal=30.0, pag_density=3.0e26, quencher_ratio=0.15,
        bake_time=8.0, D_acid=3.0e-18, k_quench=5.0, k_amp=0.4,
        mack_Mth=0.6, mack_n=8, mack_Rmax=150.0, mack_Rmin=0.001,
        develop_time=5.0, threshold=0.5,
    )


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=48, pixel_size=4e-9)


@pytest.fixture(scope="module")
def aerial(grid) -> np.ndarray:
    """Two-line/space cosine image with finite log-slope at the edges."""
    x = np.arange(grid.n_pixels) * grid.pixel_size
    period = grid.n_pixels * grid.pixel_size / 2
    profile = 0.5 + 0.45 * np.cos(2 * np.pi * x / period)
    return np.tile(profile, (grid.n_pixels, 1))


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_symbols_exported_at_package_level():
    for name in ("stochastic_trials", "StochasticResult", "add_edge_roughness"):
        assert hasattr(litho_sim.develop, name), f"litho_sim.develop.{name} missing"
        assert name in litho_sim.develop.__all__, f"{name} not in __all__"


# ---------------------------------------------------------------------------
# stochastic_trials: the physical Monte-Carlo chain, end to end
# ---------------------------------------------------------------------------


def test_trials_happy_path_through_package_api(aerial, car_cfg, grid):
    """Aerial image -> sampled chemistry -> bake -> develop -> metrics."""
    res = stochastic_trials(aerial, car_cfg, grid, EUV, trials=3, seed=7)
    assert isinstance(res, StochasticResult)
    assert res.resist.shape == (3, grid.n_pixels, grid.n_pixels)
    assert res.trials == 3
    # Developed images are strictly binary; the latent one is a fraction.
    assert set(np.unique(res.resist)).issubset({0.0, 1.0})
    assert np.all(np.isfinite(res.protected))
    assert np.all((res.protected >= 0.0) & (res.protected <= 1.0))
    # The counts that decide the noise are carried out for inspection.
    assert res.sample.mean_pag == pytest.approx(240.0, rel=1e-9)
    assert res.sample.mean_photons > 0.0
    # Something printed, and something cleared.
    assert 0.0 < float(res.resist.mean()) < 1.0
    # The develop-package output feeds the analysis package unmodified.
    u = measure_lcdu(res.resist, grid.pixel_size)
    assert u.n_trials == 3
    assert u.n_failed + int(np.sum(u.cds > 0.0)) == 3
    assert np.isfinite(u.cd_mean) and 0.0 < u.cd_mean < grid.grid_size


def test_trials_seeded_reproducibility(aerial, car_cfg, grid):
    a = stochastic_trials(aerial, car_cfg, grid, EUV, trials=2, seed=11)
    b = stochastic_trials(aerial, car_cfg, grid, EUV, trials=2, seed=11)
    c = stochastic_trials(aerial, car_cfg, grid, EUV, trials=2, seed=12)
    assert np.array_equal(a.resist, b.resist)
    assert np.array_equal(a.protected, b.protected)
    assert not np.array_equal(a.resist, c.resist)
    # Trials within a batch are independent realisations, not copies.
    assert not np.array_equal(a.resist[0], a.resist[1])


# ---------------------------------------------------------------------------
# add_edge_roughness: contracts of the cosmetic model
# ---------------------------------------------------------------------------


def test_edge_roughness_zero_sigma_is_noop():
    line = np.zeros((32, 32))
    line[:, 12:20] = 1.0
    assert np.array_equal(add_edge_roughness(line, 0.0, 25e-9, 4e-9, seed=0), line)
    assert np.array_equal(add_edge_roughness(line, -1e-9, 25e-9, 4e-9, seed=0), line)


def test_edge_roughness_uniform_image_is_noop():
    """No edges, nothing to roughen — all-resist and all-clear pass through."""
    ones = np.ones((16, 16))
    zeros = np.zeros((16, 16))
    assert np.array_equal(add_edge_roughness(ones, 2e-9, 25e-9, 4e-9, seed=0), ones)
    assert np.array_equal(add_edge_roughness(zeros, 2e-9, 25e-9, 4e-9, seed=0), zeros)


def test_edge_roughness_perturbs_binary_and_reproduces():
    line = np.zeros((48, 48))
    line[:, 20:28] = 1.0
    a = add_edge_roughness(line, 4e-9, 25e-9, 4e-9, seed=3)
    b = add_edge_roughness(line, 4e-9, 25e-9, 4e-9, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)).issubset({0.0, 1.0})
    assert np.any(a != line)  # 1-px sigma must move some edge pixels
    # The correlation-length argument is consumed: same seed, different xi,
    # different pattern.
    c = add_edge_roughness(line, 4e-9, 80e-9, 4e-9, seed=3)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# simulate_resist wiring: use_stochastic and the "car" model interplay
# ---------------------------------------------------------------------------


def test_use_stochastic_flag_is_consumed_on_threshold_path(aerial, car_cfg, grid):
    noisy_cfg = dataclasses.replace(
        car_cfg, use_stochastic=True,
        stochastic_sigma=4e-9, stochastic_corr_length=25e-9,
    )
    clean_cfg = dataclasses.replace(noisy_cfg, use_stochastic=False)
    _, _, noisy = simulate_resist(aerial, noisy_cfg, grid, model="threshold")
    _, _, clean = simulate_resist(aerial, clean_cfg, grid, model="threshold")
    assert set(np.unique(noisy)).issubset({0.0, 1.0})
    changed = float(np.mean(noisy != clean))
    assert 0.0 < changed < 0.5  # roughness moved edges, not the whole image


def test_use_stochastic_composes_with_car_model(aerial, car_cfg, grid):
    """The cosmetic knob stacks on the deterministic CAR print."""
    noisy_cfg = dataclasses.replace(
        car_cfg, use_stochastic=True,
        stochastic_sigma=4e-9, stochastic_corr_length=25e-9,
    )
    _, _, det = simulate_resist(aerial, car_cfg, grid, model="car")
    _, _, noisy = simulate_resist(aerial, noisy_cfg, grid, model="car")
    for image in (det, noisy):
        assert set(np.unique(image)).issubset({0.0, 1.0})
    assert 0.0 < float(det.mean()) < 1.0
    assert np.any(noisy != det)
    # The deterministic CAR print is the natural reference for failure
    # statistics — same chain stochastic_trials runs, minus the sampling.
    assert det.shape == (grid.n_pixels, grid.n_pixels)
