"""
Exposure photochemistry: photon counting and acid statistics.

The constants are checked against hand arithmetic, and the sampled chemistry
against its own mean field — the design claim of ``photochem`` is that
``sample_species`` converges to ``generate_acid`` as counts grow, so that is
what gets asserted. The EUV/ArF comparison is the physics headline: same
dose, ~14x fewer photons, visibly noisier acid, from constants alone.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from litho_sim.core.config import RESIST_LIBRARY, ResistConfig
from litho_sim.expose.photochem import (
    absorbed_fraction,
    generate_acid,
    mean_absorbed_photons,
    photon_energy,
    sample_species,
)

ARF = 193e-9
EUV = 13.5e-9
PX = 4e-9


@pytest.fixture()
def resist() -> ResistConfig:
    return ResistConfig(dill_A=0.0, dill_B=4.0, dill_C=0.05, thickness=50e-9,
                        dose_nominal=30.0, pag_density=3.0e26)


def test_photon_energy_known_values():
    """hc/λ: 6.42 eV at ArF, 91.8 eV at EUV — the 14.3x that makes EUV loud."""
    e_arf = photon_energy(ARF)
    e_euv = photon_energy(EUV)
    assert e_arf == pytest.approx(1.029e-18, rel=1e-3)
    assert e_euv == pytest.approx(1.471e-17, rel=1e-3)
    assert e_euv / e_arf == pytest.approx(ARF / EUV)


def test_absorbed_fraction_is_beer_lambert(resist):
    """A+B per µm through the thickness, nothing more."""
    expected = 1.0 - np.exp(-4.0e6 * 50e-9)
    assert absorbed_fraction(resist) == pytest.approx(expected)


def test_mean_absorbed_photons_hand_check(resist):
    """Uniform unit aerial: dose × area × f_abs / E_photon, in SI."""
    aerial = np.ones((8, 8))
    n = mean_absorbed_photons(aerial, resist, EUV, PX)
    expected = (30.0 * 10.0 / photon_energy(EUV)) * PX * PX * absorbed_fraction(resist)
    assert n == pytest.approx(np.full((8, 8), expected))
    # And the number itself is small — tens of photons — which is the point.
    assert 10 < n[0, 0] < 100


def test_euv_gets_arf_photon_ratio(resist):
    """Equal dose and absorbance: photon counts scale with wavelength."""
    aerial = np.ones((4, 4))
    n_euv = mean_absorbed_photons(aerial, resist, EUV, PX)
    n_arf = mean_absorbed_photons(aerial, resist, ARF, PX)
    assert n_arf[0, 0] / n_euv[0, 0] == pytest.approx(ARF / EUV)


def test_generate_acid_is_first_order_photolysis(resist):
    """h = 1 − exp(−C·E), monotonic, zero in the dark."""
    aerial = np.linspace(0.0, 1.0, 16).reshape(1, -1)
    h = generate_acid(aerial, resist, PX)
    expected = 1.0 - np.exp(-0.05 * aerial * 30.0)
    assert h == pytest.approx(expected)
    assert h[0, 0] == 0.0
    assert (np.diff(h[0]) > 0).all()


def test_sample_species_mean_matches_deterministic(resist):
    """Averaged over many voxels, the sampled acid sits on the mean field.

    Uniform aerial so every voxel is a repeat of the same experiment; ArF so
    the counts are large and the concave-conversion (Jensen) bias is small.
    """
    aerial = np.full((64, 64), 0.7)
    rng = np.random.default_rng(7)
    s = sample_species(aerial, resist, ARF, PX, rng=rng)
    h_det = generate_acid(aerial, resist, PX)[0, 0]
    assert float(s.acid.mean()) == pytest.approx(h_det, rel=0.02)
    assert float(s.quencher.mean()) == pytest.approx(resist.quencher_ratio, rel=0.05)


def test_photon_shot_noise_is_poisson(resist):
    """Relative fluctuation of the photon draw is 1/sqrt(n̄)."""
    aerial = np.full((128, 128), 1.0)
    rng = np.random.default_rng(11)
    s = sample_species(aerial, resist, EUV, PX, rng=rng)
    n_mean = mean_absorbed_photons(aerial, resist, EUV, PX)[0, 0]
    rel = float(s.photons.std() / s.photons.mean())
    assert rel == pytest.approx(1.0 / np.sqrt(n_mean), rel=0.1)


def test_euv_acid_is_noisier_than_arf(resist):
    """Same dose, same chemistry, same seed procedure — EUV must be louder.

    At realistic PAG loadings both wavelengths share a *molecular* noise
    floor (~240 PAG per voxel → ~5 % acid σ either way), so the photon
    contrast only shows once that floor is pushed down: 10x the PAG isolates
    photon statistics, and there EUV must be severalfold noisier.
    """
    dense = dataclasses.replace(resist, pag_density=3.0e27)
    aerial = np.full((64, 64), 0.7)
    s_euv = sample_species(aerial, dense, EUV, PX, rng=np.random.default_rng(3))
    s_arf = sample_species(aerial, dense, ARF, PX, rng=np.random.default_rng(3))
    assert s_euv.acid.std() > 2.0 * s_arf.acid.std()


def test_electron_blur_smooths_but_conserves(resist):
    """The cascade moves deposited energy around; it does not create any."""
    blurred_cfg = dataclasses.replace(resist, electron_blur_sigma=4e-9)
    aerial = np.full((64, 64), 1.0)
    s_sharp = sample_species(aerial, resist, EUV, PX, rng=np.random.default_rng(5))
    s_blur = sample_species(aerial, blurred_cfg, EUV, PX, rng=np.random.default_rng(5))
    assert s_blur.photons.std() < s_sharp.photons.std()
    assert s_blur.photons.mean() == pytest.approx(s_sharp.photons.mean(), rel=1e-6)


def test_acid_never_exceeds_local_pag(resist):
    """Binomial conversion saturates: a voxel cannot yield acid it has no PAG for."""
    hot = np.full((32, 32), 50.0)  # absurd over-exposure
    s = sample_species(hot, resist, EUV, PX, rng=np.random.default_rng(1))
    # In PAG₀ units the acid is n_acid / mean_pag; with p→1 it equals the
    # sampled PAG count, whose fluctuation is Poisson — nothing unphysical.
    assert float(s.acid.mean()) == pytest.approx(1.0, rel=0.05)


def test_euv_preset_is_loadable():
    """The library entry must construct — field names are checked by dataclass."""
    cfg = ResistConfig(**RESIST_LIBRARY["EUV CAR (Positive)"])
    assert cfg.dill_A == 0.0
    assert cfg.electron_blur_sigma > 0.0
