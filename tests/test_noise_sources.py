"""
The two noise sources added to the stochastic model, and the scaling law.

* Photon statistics obey the inverse square root of dose once the molecular
  floor is suppressed and development noise is off — on a constant-slope
  image, so the edge does not slide onto a different part of the image as
  dose changes.
* The acid-yield term adds variance and leaves the mean alone.
* Dissolution noise adds roughness that no photon count removes, with the
  granule scatter averaged over the granules a voxel spans.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from litho_sim.analysis import edge_positions, trial_roughness
from litho_sim.core.config import GridConfig, ResistConfig
from litho_sim.develop import stochastic_trials
from litho_sim.develop.stochastic import dissolution_scatter
from litho_sim.expose import generate_acid, sample_species
from litho_sim.expose.photochem import acids_per_photon

EUV = 13.5e-9
PX = 2e-9
N = 96


@pytest.fixture
def ramp():
    """A trapezoid-wave aerial image: the same slope at every edge, at every dose.

    Flat valleys 12 pixels wide at 0.02, 8-pixel linear ramps, 4-pixel flat
    tops at 1.0, on a 32-pixel period. The valleys are wide enough that the
    electron blur does not lift their floor — a triangle wave's sharp
    valleys rose to a third of the peak and swallowed the edge level.
    """
    period = np.concatenate([
        np.full(12, 0.02), np.linspace(0.02, 1.0, 10)[1:-1], np.full(4, 1.0),
        np.linspace(1.0, 0.02, 10)[1:-1],
    ])
    assert period.size == 32
    row = np.tile(period, N // 32)
    return np.broadcast_to(row, (N, N)).copy()


@pytest.fixture
def resist():
    return ResistConfig(
        tone="positive", dill_A=0.0, dill_B=4.0, dill_C=0.05, thickness=50e-9,
        mack_Rmax=150.0, mack_Rmin=0.02, mack_Mth=0.6, mack_n=12, develop_time=5.0,
        pag_density=3.0e26, quencher_ratio=0.2, bake_time=60.0, D_acid=3.0e-19,
        k_quench=20.0, k_amp=0.05, electron_blur_sigma=4e-9,
        dissolution_sigma=0.0, acid_yield_noise=True,
    )


def _ler(aerial, cfg, dose, seed=1):
    res = stochastic_trials(aerial, cfg, GridConfig(n_pixels=N, pixel_size=PX), EUV,
                            dose=dose, trials=6, seed=seed)
    return trial_roughness(res, PX).ler


def test_edge_noise_falls_as_the_inverse_square_root_of_dose(ramp, resist):
    """Photon-limited regime: PAG loading ×100 removes the molecular floor.

    The law is for a *fixed edge*: the exposure at the edge scales with the
    dose, so its photon count does, and the image slope scales with dose too
    — σ_x ∝ √dose / dose. So the edge is read off the sampled acid field at
    a level that scales with dose, which keeps it at the same place on the
    ramp; a developed edge in a quencher resist would instead slide to the
    same exposure at every dose and see the same photons every time.
    """
    photon_limited = dataclasses.replace(resist, pag_density=resist.pag_density * 100)
    grid = GridConfig(n_pixels=N, pixel_size=PX)

    def edge_sigma(dose: float, draws: int = 6) -> float:
        # The level where the ramp's exposure is 3 mJ/cm² × dose, in acid —
        # above the ramp's valleys, which the electron blur lifts to about
        # 0.9 mJ/cm² × dose, and below its peaks.
        e_edge = 3.0 * dose
        level = 1.0 - np.exp(-photon_limited.dill_C * e_edge)
        sig = []
        for k in range(draws):
            acid = sample_species(ramp, photon_limited, EUV, PX, dose=dose,
                                  rng=np.random.default_rng(k)).acid
            left, _right = edge_positions(acid, grid.pixel_size, threshold=level)
            sig.append(np.nanstd(left, ddof=1))
        return float(np.mean(sig))

    low, high = edge_sigma(0.25), edge_sigma(1.0)
    assert low > 0 and high > 0
    assert low / high == pytest.approx(2.0, rel=0.35), (low * 1e9, high * 1e9)


def test_yield_noise_adds_variance_and_keeps_the_mean(ramp, resist):
    # The mean, at a heavy loading where every count is large.
    heavy = dataclasses.replace(resist, pag_density=resist.pag_density * 100)
    draws = [sample_species(ramp, heavy, EUV, PX, rng=np.random.default_rng(k)).acid
             for k in range(12)]
    mean = generate_acid(ramp, heavy, PX)
    bright = ramp > 0.5
    assert np.mean(draws, axis=0)[bright].mean() == pytest.approx(mean[bright].mean(), rel=0.05)
    # The variance, at the preset's own loading: the yield term is 1/φ of the
    # photon term and φ is 5.5 here, so it must show as at least a tenth.
    assert acids_per_photon(resist, EUV) > 1.0     # the EUV regime: several acids per photon
    off = dataclasses.replace(resist, acid_yield_noise=False)
    var_with = np.mean([np.var([sample_species(ramp, resist, EUV, PX,
                                               rng=np.random.default_rng(k)).acid[bright]
                                for k in range(16)], axis=0)])
    var_without = np.mean([np.var([sample_species(ramp, off, EUV, PX,
                                                  rng=np.random.default_rng(k)).acid[bright]
                                   for k in range(16)], axis=0)])
    assert var_with > 1.1 * var_without, (var_with, var_without)


def test_dissolution_noise_adds_roughness_no_photon_count_removes(ramp, resist):
    quiet = dataclasses.replace(resist, pag_density=resist.pag_density * 1000)
    base = _ler(ramp, quiet, 1.6)
    noisy = _ler(ramp, dataclasses.replace(quiet, dissolution_sigma=1.5), 1.6)
    assert noisy > 1.2 * base, (base * 1e9, noisy * 1e9)


def test_dissolution_scatter_averages_over_the_granules_a_voxel_spans():
    cfg = ResistConfig(dissolution_sigma=1.0, dissolution_corr_length=5e-9, thickness=100e-9)
    rng = np.random.default_rng(0)
    column = dissolution_scatter((64, 64), cfg, (4e-9, 4e-9), rng, column_height=100e-9)
    voxel = dissolution_scatter((16, 64, 64), cfg, (4e-9, 4e-9, 4e-9), rng)
    assert column is not None and voxel is not None
    # Unit mean either way, and the column's log-scatter is the granule's
    # over sqrt(100 nm / 5 nm).
    assert column.mean() == pytest.approx(1.0, rel=0.05)
    assert voxel.mean() == pytest.approx(1.0, rel=0.05)
    assert np.log(column).std() == pytest.approx(1.0 / np.sqrt(20.0), rel=0.2)
    assert np.log(voxel).std() == pytest.approx(1.0, rel=0.2)
    assert dissolution_scatter((8, 8), dataclasses.replace(cfg, dissolution_sigma=0.0),
                               (4e-9, 4e-9), rng) is None
