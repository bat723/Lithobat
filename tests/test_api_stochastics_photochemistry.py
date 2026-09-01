"""
Interface tests: photochemistry through the ``litho_sim.expose`` package API.

Everything here imports from ``litho_sim.expose`` — never the ``photochem``
module directly — so these tests pin the *public* surface: the exports exist
at package level, their contracts hold when called the way a consumer calls
them, and the sampled chemistry is reproducible under a seeded generator.
Unit-level physics (Poisson statistics, Jensen bias, blur conservation) is
covered in ``test_photochem.py``; this file is about the interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import litho_sim.expose
from litho_sim.core.config import ResistConfig
from litho_sim.expose import (
    SpeciesSample,
    absorbed_fraction,
    generate_acid,
    mean_absorbed_photons,
    photon_energy,
    sample_species,
)

EUV = 13.5e-9
ARF = 193e-9
PIXEL = 4e-9


@pytest.fixture(scope="module")
def euv_resist() -> ResistConfig:
    """An EUV-CAR-like film: no bleaching, B carries the absorbance."""
    return ResistConfig(
        dill_A=0.0, dill_B=4.0, dill_C=0.05,
        thickness=50e-9, dose_nominal=30.0,
        pag_density=3.0e26, quencher_ratio=0.15,
    )


@pytest.fixture(scope="module")
def aerial() -> np.ndarray:
    """A smooth 32x32 line/space-like image in [0.05, 0.95]."""
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    profile = 0.5 + 0.45 * np.cos(x)
    return np.tile(profile, (32, 1))


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_all_symbols_exported_at_package_level():
    for name in (
        "photon_energy", "absorbed_fraction", "mean_absorbed_photons",
        "generate_acid", "sample_species", "SpeciesSample",
    ):
        assert hasattr(litho_sim.expose, name), f"litho_sim.expose.{name} missing"
        assert name in litho_sim.expose.__all__, f"{name} not in __all__"


# ---------------------------------------------------------------------------
# Deterministic layer: hand-checkable numbers
# ---------------------------------------------------------------------------


def test_photon_energy_euv_vs_arf():
    e_euv = photon_energy(EUV)
    e_arf = photon_energy(ARF)
    # hc/lambda: 1.47e-17 J at 13.5 nm (~92 eV), ratio is lambda ratio.
    assert e_euv == pytest.approx(1.4714e-17, rel=1e-3)
    assert e_euv / e_arf == pytest.approx(ARF / EUV, rel=1e-12)


def test_absorbed_fraction_is_bounded_beer_lambert(euv_resist):
    f = absorbed_fraction(euv_resist)
    # 4 /um over 50 nm: 1 - exp(-0.2)
    assert f == pytest.approx(1.0 - np.exp(-0.2), rel=1e-9)
    assert 0.0 < f < 1.0


def test_mean_absorbed_photons_hand_check_and_scaling(euv_resist):
    ones = np.ones((8, 8))
    n = mean_absorbed_photons(ones, euv_resist, EUV, PIXEL)
    # 30 mJ/cm2 = 300 J/m2; / E_photon; * voxel area; * absorbed fraction.
    expected = (
        300.0 / photon_energy(EUV) * PIXEL * PIXEL * absorbed_fraction(euv_resist)
    )
    assert np.allclose(n, expected)
    assert np.all(np.isfinite(n)) and np.all(n >= 0.0)
    # Linear in relative dose.
    n2 = mean_absorbed_photons(ones, euv_resist, EUV, PIXEL, dose=2.0)
    assert np.allclose(n2, 2.0 * n)


def test_generate_acid_bounds_and_monotonicity(euv_resist, aerial):
    acid = generate_acid(aerial, euv_resist, PIXEL)
    assert acid.shape == aerial.shape
    assert np.all((acid >= 0.0) & (acid <= 1.0))
    # More dose, more acid everywhere light lands.
    hot = generate_acid(aerial, euv_resist, PIXEL, dose=2.0)
    assert np.all(hot >= acid)
    # Dark voxels make no acid.
    assert np.allclose(generate_acid(np.zeros((8, 8)), euv_resist, PIXEL), 0.0)


# ---------------------------------------------------------------------------
# Sampled layer: reproducibility and sanity through the public API
# ---------------------------------------------------------------------------


def test_sample_species_happy_path(euv_resist, aerial):
    s = sample_species(aerial, euv_resist, EUV, PIXEL, rng=np.random.default_rng(0))
    assert isinstance(s, SpeciesSample)
    for field in (s.acid, s.quencher, s.photons):
        assert field.shape == aerial.shape
        assert np.all(np.isfinite(field))
        assert np.all(field >= 0.0)
    # 3e26 /m3 in a (4 nm)^2 x 50 nm voxel: 240 molecules.
    assert s.mean_pag == pytest.approx(240.0, rel=1e-9)
    assert s.mean_photons > 0.0
    # Sampled acid tracks the mean field loosely (finite-count Jensen bias
    # pushes it slightly low; the interface promise is "same chemistry").
    det = generate_acid(aerial, euv_resist, PIXEL)
    assert abs(float(s.acid.mean()) - float(det.mean())) < 0.1


def test_sample_species_seeded_reproducibility(euv_resist, aerial):
    a = sample_species(aerial, euv_resist, EUV, PIXEL, rng=np.random.default_rng(42))
    b = sample_species(aerial, euv_resist, EUV, PIXEL, rng=np.random.default_rng(42))
    c = sample_species(aerial, euv_resist, EUV, PIXEL, rng=np.random.default_rng(43))
    assert np.array_equal(a.acid, b.acid)
    assert np.array_equal(a.quencher, b.quencher)
    assert np.array_equal(a.photons, b.photons)
    assert not np.array_equal(a.acid, c.acid)


def test_quencher_knob_is_consumed(euv_resist, aerial):
    """quencher_ratio flows from ResistConfig into the sampled field."""
    import dataclasses

    s = sample_species(aerial, euv_resist, EUV, PIXEL, rng=np.random.default_rng(1))
    assert float(s.quencher.mean()) == pytest.approx(0.15, rel=0.10)
    none = dataclasses.replace(euv_resist, quencher_ratio=0.0)
    s0 = sample_species(aerial, none, EUV, PIXEL, rng=np.random.default_rng(1))
    assert np.allclose(s0.quencher, 0.0)


def test_electron_blur_knob_is_consumed(euv_resist, aerial):
    """electron_blur_sigma spreads the photon field; without it counts stay
    integers (a pure Poisson draw)."""
    import dataclasses

    sharp = sample_species(aerial, euv_resist, EUV, PIXEL,
                           rng=np.random.default_rng(5))
    assert np.allclose(sharp.photons, np.round(sharp.photons))
    blurred_cfg = dataclasses.replace(euv_resist, electron_blur_sigma=4e-9)
    blurred = sample_species(aerial, blurred_cfg, EUV, PIXEL,
                             rng=np.random.default_rng(5))
    assert not np.allclose(blurred.photons, np.round(blurred.photons))
    # Blur redistributes, it does not create photons.
    assert float(blurred.photons.sum()) == pytest.approx(
        float(sharp.photons.sum()), rel=1e-6
    )


def test_zero_dose_edge_case(euv_resist, aerial):
    """No light: no photons, no acid — the chain degrades gracefully."""
    s = sample_species(aerial, euv_resist, EUV, PIXEL, dose=0.0,
                       rng=np.random.default_rng(2))
    assert np.allclose(s.photons, 0.0)
    assert np.allclose(s.acid, 0.0)
    # The formulated-in quencher is still there — it is not photogenerated.
    assert float(s.quencher.mean()) > 0.0
