"""
Interface-level tests for the coat TMM optics seam.

Everything here goes through the public surface — ``litho_sim.coat`` for the
transfer-matrix computation itself, and the ``litho_sim.develop`` dispatch
that consumes it when ``ResistConfig.optical_model="tmm"``. The unit-level
physics validation (Fresnel closed forms, BARC design sweeps) lives in
``tests/test_films.py``; these tests instead pin down the *contract*:

- the four public names are importable from ``litho_sim.coat``;
- amplitude and intensity reflectance agree, at angle and for both
  polarisations;
- the swing curve and the in-resist standing wave carry the textbook
  ``λ/(2·n·cosθ)`` period *off normal incidence* (angle handling);
- the standing-wave contrast matches the closed form ``2|r|/(1+|r|²)``;
- config records flow from ``ResistConfig.film_stack`` through the dispatch
  into the stack, and material names resolve on the way;
- nonsense input (empty stack, unknown material, negative thickness,
  non-positive wavelength) raises instead of returning NaN or unphysical
  reflectance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import litho_sim.coat as coat
from litho_sim.coat import (
    Film,
    FilmStack,
    film_stack_from_records,
    resist_index_from_dill,
)
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.develop import apply_vertical_interference, tmm_standing_waves

LAM = 193e-9
N_RESIST_REAL = 1.70


@pytest.fixture(scope="module")
def n_resist() -> complex:
    return resist_index_from_dill(N_RESIST_REAL, 0.8, 0.05, LAM)


@pytest.fixture(scope="module")
def bare(n_resist) -> FilmStack:
    """Resist straight on silicon — the worst case a BARC exists to fix."""
    return film_stack_from_records([], 120e-9, n_resist)


@pytest.fixture(scope="module")
def with_barc(n_resist) -> FilmStack:
    """A 30 nm n=1.8, k=0.5 organic BARC — near the null found by design."""
    return film_stack_from_records(
        [{"n": 1.80, "k": 0.50, "thickness": 30e-9, "name": "BARC"}],
        120e-9,
        n_resist,
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_names_are_exported():
    """The package namespace, not films.py, is the supported import path."""
    expected = {"Film", "FilmStack", "film_stack_from_records", "resist_index_from_dill"}
    assert expected == set(coat.__all__)
    for name in expected:
        assert getattr(coat, name) is not None


def test_amplitude_reflection_squares_to_reflectance(bare):
    """|amplitude|² and intensity reflectance must be the same quantity."""
    for theta0 in (0.0, 0.45):
        for pol in ("s", "p"):
            r = bare.amplitude_reflection(LAM, theta0, pol)
            R = bare.reflectance(LAM, theta0, pol)
            assert abs(r) ** 2 == pytest.approx(R, rel=1e-12)
            assert 0.0 <= R <= 1.0


# ---------------------------------------------------------------------------
# Happy-path physics through the public API
# ---------------------------------------------------------------------------


def test_designed_barc_kills_the_substrate_reflection(bare, with_barc):
    """Absolute numbers, not just a ratio: |r| ≈ 0.76 bare, < 0.12 coated.

    ``substrate_reflection`` is the quantity that drives standing waves, so
    it — not the total stack reflectance — is what a BARC must suppress.
    """
    r_bare = abs(bare.substrate_reflection(LAM))
    r_barc = abs(with_barc.substrate_reflection(LAM))
    assert r_bare > 0.70
    assert r_barc < 0.12
    # And it holds off-normal too, where the null detunes but must not vanish.
    assert abs(with_barc.substrate_reflection(LAM, theta0=0.3)) < 0.5 * abs(
        bare.substrate_reflection(LAM, theta0=0.3)
    )


def test_swing_curve_period_off_normal(bare):
    """At incidence θ₀ the period is λ/(2·n·cosθ_resist), θ from Snell.

    The normal-incidence period is pinned by the unit tests; this checks the
    angle handling the illumination average in the consumer relies on.
    """
    theta0 = 0.6
    ts = np.linspace(100e-9, 400e-9, 601)
    sw = bare.swing_curve(LAM, ts, theta0=theta0)
    assert np.all(sw >= 0.0) and np.all(sw <= 1.0)

    freqs = np.fft.rfftfreq(len(ts), d=ts[1] - ts[0])
    k = int(np.argmax(np.abs(np.fft.rfft(sw - sw.mean()))))
    period = 1.0 / freqs[k]
    cos_r = np.sqrt(1.0 - (np.sin(theta0) / N_RESIST_REAL) ** 2)
    expected = LAM / (2.0 * N_RESIST_REAL * cos_r)
    bin_width = abs(1.0 / freqs[k] - 1.0 / freqs[k + 1])
    assert period == pytest.approx(expected, abs=bin_width), (
        f"oblique swing period {period * 1e9:.1f} nm != "
        f"λ/(2n·cosθ) = {expected * 1e9:.1f} nm"
    )


def test_field_profile_contrast_matches_closed_form(n_resist):
    """Two-beam interference gives contrast exactly 2|r|/(1+|r|²).

    A closed form the envelope must reproduce once the resist is thick
    enough to sample whole fringes — a sharper check than "there is a
    fringe", and one that ties the envelope amplitude to the *same*
    ``substrate_reflection`` the reflectance path computes.
    """
    stack = film_stack_from_records([], 600e-9, n_resist)
    z = np.linspace(0.0, 600e-9, 2401)
    env = stack.field_profile(LAM, z)
    assert np.all(env >= 0.0)
    assert env.mean() == pytest.approx(1.0)

    r = abs(stack.substrate_reflection(LAM))
    contrast = (env.max() - env.min()) / (env.max() + env.min())
    assert contrast == pytest.approx(2.0 * r / (1.0 + r * r), abs=1e-3)


# ---------------------------------------------------------------------------
# The consumer seam: ResistConfig.optical_model="tmm" reaches this code
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=16, pixel_size=4e-9, dz=2e-9)


@pytest.fixture(scope="module")
def optics() -> OpticsConfig:
    return OpticsConfig(wavelength=LAM, NA=0.93, sigma_outer=0.8, source_grid=11)


def test_dispatch_routes_tmm_to_the_film_stack(grid, optics):
    """``optical_model="tmm"`` must actually switch models at the seam.

    With every default left alone (``substrate_reflectance=0``) the legacy
    two-beam path is a no-op; flipping only ``optical_model`` must produce
    the bare-silicon fringe — proof the config field reaches the TMM code.
    """
    intensity = np.ones((60, 16, 16))
    twobeam = ResistConfig(thickness=120e-9, n_resist=1.7)
    tmm = ResistConfig(thickness=120e-9, n_resist=1.7, optical_model="tmm")

    out_twobeam = apply_vertical_interference(intensity, twobeam, optics, grid)
    out_tmm = apply_vertical_interference(intensity, tmm, optics, grid)

    assert np.array_equal(out_twobeam, intensity)  # legacy default: no-op
    trace = out_tmm[:, 8, 8]
    assert (trace.max() - trace.min()) / (trace.max() + trace.min()) > 0.3
    # The dispatch must be exactly the public tmm_standing_waves computation.
    assert np.array_equal(out_tmm, tmm_standing_waves(intensity, tmm, optics, grid))


def test_material_records_flow_from_config_to_the_stack(grid, optics):
    """A ``{"material": ...}`` record in ResistConfig must resolve and act.

    The library SiARC is a real anti-reflective film, so routing it through
    ``film_stack`` must visibly flatten the fringe versus bare silicon —
    which exercises record parsing, material lookup, and stack assembly in
    one pass through the public consumer path.
    """
    intensity = np.ones((60, 16, 16))

    def contrast(cfg: ResistConfig) -> float:
        t = apply_vertical_interference(intensity, cfg, optics, grid)[:, 8, 8]
        return (t.max() - t.min()) / (t.max() + t.min())

    bare_cfg = ResistConfig(thickness=120e-9, n_resist=1.7, optical_model="tmm")
    siarc_cfg = ResistConfig(
        thickness=120e-9,
        n_resist=1.7,
        optical_model="tmm",
        film_stack=[{"material": "SiARC", "thickness": 38e-9}],
    )
    # Library SiARC at 38 nm measures ~0.64x of the bare-Si contrast (0.60
    # vs 0.95); anything under 0.75x proves the record parsed and acted.
    assert contrast(siarc_cfg) < 0.75 * contrast(bare_cfg)


# ---------------------------------------------------------------------------
# Errors and edges the interface guards
# ---------------------------------------------------------------------------


def test_empty_stack_is_rejected():
    with pytest.raises(ValueError, match="at least an ambient"):
        FilmStack([])


def test_unknown_material_is_rejected(n_resist):
    with pytest.raises(KeyError, match="unobtainium"):
        film_stack_from_records(
            [{"material": "unobtainium", "thickness": 38e-9}], 120e-9, n_resist
        )


def test_record_missing_thickness_is_rejected(n_resist):
    with pytest.raises(KeyError, match="thickness"):
        film_stack_from_records([{"n": 1.8, "k": 0.4}], 120e-9, n_resist)


def test_negative_thickness_is_rejected(n_resist):
    """A negative layer would run the matrix backwards and amplify."""
    with pytest.raises(ValueError, match="negative thickness"):
        film_stack_from_records(
            [{"n": 1.8, "k": 0.4, "thickness": -38e-9}], 120e-9, n_resist
        )
    with pytest.raises(ValueError):
        film_stack_from_records([], -120e-9, n_resist)
    with pytest.raises(ValueError, match="negative thickness"):
        FilmStack(
            [
                Film("air", 0.0, 1 + 0j),
                Film("resist", -120e-9, 1.7 + 0j),
                Film("Si", 0.0, 0.883 + 2.778j),
            ]
        )


def test_nonpositive_wavelength_is_rejected(bare):
    """λ ≤ 0 divides by zero in δ = 2πnd/λ; NaN out is worse than raising."""
    z = np.linspace(0.0, 120e-9, 61)
    for bad in (0.0, -193e-9, float("nan")):
        with pytest.raises(ValueError, match="wavelength"):
            bare.reflectance(bad)
        with pytest.raises(ValueError, match="wavelength"):
            bare.field_profile(bad, z)
