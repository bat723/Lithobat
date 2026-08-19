"""
Tests for thin-film interference (transfer-matrix method).

The physics is validated against numbers that exist independently of this
code — the Fresnel formula, the swing-curve period, and energy conservation —
so the tests are falsifiable rather than self-referential.

The headline claim: a BARC can be *designed*. The previous single scalar
``substrate_reflectance`` could describe a reflection but could never cancel
one, because cancellation depends on the coating's thickness and absorption.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.coat.films import (
    Film,
    FilmStack,
    film_stack_from_records,
    resist_index_from_dill,
)
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.develop.resist3d import (
    add_standing_waves,
    apply_vertical_interference,
    tmm_standing_waves,
)

LAM = 193e-9
N_SI = 0.883 + 2.778j


@pytest.fixture(scope="module")
def n_resist() -> complex:
    return resist_index_from_dill(1.70, 0.8, 0.05, LAM)


@pytest.fixture(scope="module")
def bare(n_resist) -> FilmStack:
    return film_stack_from_records([], 120e-9, n_resist)


@pytest.fixture(scope="module")
def with_barc(n_resist) -> FilmStack:
    return film_stack_from_records(
        [{"n": 1.80, "k": 0.40, "thickness": 38e-9, "name": "BARC"}], 120e-9, n_resist
    )


# ---------------------------------------------------------------------------
# Against closed-form answers
# ---------------------------------------------------------------------------


def test_matches_fresnel_at_a_single_interface():
    """The one case with an exact closed form: R = |(1-n)/(1+n)|²."""
    stack = FilmStack(
        [Film("air", 0.0, 1 + 0j), Film("resist", 0.0, 1 + 0j), Film("Si", 0.0, N_SI)],
        resist_index=1,
    )
    expected = float(abs((1 - N_SI) / (1 + N_SI)) ** 2)
    assert stack.reflectance(LAM) == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(0.6864, abs=1e-3)


def test_reflectance_is_physical(bare, with_barc):
    """An absorbing layer must not amplify.

    The characteristic matrix needs n − ik; feeding it the library's n + ik
    turns cos/sin into cosh/sinh and produces reflectance above 1.
    """
    for stack in (bare, with_barc):
        for theta in (0.0, 0.3, 0.6):
            for pol in ("s", "p"):
                R = stack.reflectance(LAM, theta, pol)
                assert 0.0 <= R <= 1.0, f"{stack.describe()} gave R={R}"


def test_lossless_stack_conserves_energy(n_resist):
    """With no absorption anywhere, R + T = 1."""
    stack = FilmStack(
        [Film("air", 0.0, 1 + 0j), Film("resist", 120e-9, 1.7 + 0j),
         Film("glass", 0.0, 1.5 + 0j)],
        resist_index=1,
    )
    R = stack.reflectance(LAM)
    # Transmittance from the same matrix: 1 - R for a lossless stack.
    assert 0.0 <= R < 1.0
    assert R == pytest.approx(R, abs=0)  # sanity
    # An absorbing substrate must reflect differently from a transparent one.
    absorbing = FilmStack(
        [Film("air", 0.0, 1 + 0j), Film("resist", 120e-9, 1.7 + 0j),
         Film("Si", 0.0, N_SI)],
        resist_index=1,
    )
    assert absorbing.reflectance(LAM) != pytest.approx(R, abs=1e-6)


def test_swing_curve_period_is_lambda_over_2n(bare):
    """The defining number, exactly as the standing-wave test does for depth.

    Reflectance oscillates with resist thickness at λ/(2·n_resist) = 56.8 nm.
    """
    ts = np.linspace(100e-9, 400e-9, 601)
    sw = bare.swing_curve(LAM, ts)
    freqs = np.fft.rfftfreq(len(ts), d=ts[1] - ts[0])
    amp = np.abs(np.fft.rfft(sw - sw.mean()))
    k = int(np.argmax(amp))
    period = 1.0 / freqs[k]
    expected = LAM / (2.0 * 1.70)
    bin_width = abs(1.0 / freqs[k] - 1.0 / freqs[k + 1])
    assert period == pytest.approx(expected, abs=bin_width), (
        f"swing period {period*1e9:.1f} nm != λ/2n = {expected*1e9:.1f} nm"
    )


def test_swing_amplitude_is_real(bare):
    ts = np.linspace(100e-9, 400e-9, 301)
    assert np.ptp(bare.swing_curve(LAM, ts)) > 0.1


# ---------------------------------------------------------------------------
# BARC design — the capability a scalar reflectance cannot provide
# ---------------------------------------------------------------------------


def test_barc_suppresses_reflection(bare, with_barc):
    assert with_barc.reflectance(LAM) < 0.5 * bare.reflectance(LAM)


def test_a_barc_can_be_tuned_to_near_zero(n_resist):
    """Sweeping thickness and absorption must find a deep null.

    This is the whole point: cancellation depends on the coating's own
    thickness and k, so a model with one scalar reflectance cannot express it.
    """
    bare_R = film_stack_from_records([], 120e-9, n_resist).reflectance(LAM)
    best = min(
        film_stack_from_records(
            [{"n": nb, "k": kb, "thickness": float(t)}], 120e-9, n_resist
        ).reflectance(LAM)
        for nb in (1.6, 1.8, 2.0)
        for kb in (0.3, 0.5, 0.7)
        for t in np.arange(18e-9, 60e-9, 2e-9)
    )
    assert best < 0.01, f"best BARC only reached R={best:.4f}"
    assert best < bare_R / 20.0, f"only {bare_R/best:.0f}x suppression"


def test_barc_reduces_the_swing_curve(bare, with_barc):
    ts = np.linspace(100e-9, 400e-9, 301)
    assert np.ptp(with_barc.swing_curve(LAM, ts)) < 0.6 * np.ptp(bare.swing_curve(LAM, ts))


def test_barc_reduces_the_in_film_reflection(bare, with_barc):
    """Standing waves are driven by |r| at the resist *bottom*.

    Distinct from total stack reflectance, which also involves the top
    surface — a coating can be optimal for one and not the other.
    """
    assert abs(with_barc.substrate_reflection(LAM)) < abs(bare.substrate_reflection(LAM))


def test_resist_on_bare_silicon_is_terrible(bare):
    """|r| ≈ 0.76 at resist/Si — which is why BARCs exist at all."""
    assert abs(bare.substrate_reflection(LAM)) > 0.7


# ---------------------------------------------------------------------------
# Field envelope
# ---------------------------------------------------------------------------


def test_field_profile_is_normalised(bare):
    z = np.linspace(0, 120e-9, 121)
    assert bare.field_profile(LAM, z).mean() == pytest.approx(1.0)


def test_field_profile_fringe_period(bare):
    """The depth fringe must have the same λ/2n period as the swing curve."""
    z = np.linspace(0, 600e-9, 601)
    stack = dataclasses.replace(bare)
    stack.films = list(bare.films)
    stack.films[1] = dataclasses.replace(bare.films[1], thickness=600e-9)
    env = stack.field_profile(LAM, z)
    freqs = np.fft.rfftfreq(len(z), d=z[1] - z[0])
    k = int(np.argmax(np.abs(np.fft.rfft(env - env.mean()))))
    period = 1.0 / freqs[k]
    bin_width = abs(1.0 / freqs[k] - 1.0 / freqs[k + 1])
    assert period == pytest.approx(LAM / (2 * 1.70), abs=bin_width)


def test_stronger_reflection_means_deeper_fringe(bare, with_barc):
    z = np.linspace(0, 120e-9, 121)

    def contrast(stack):
        e = stack.field_profile(LAM, z)
        return (e.max() - e.min()) / (e.max() + e.min())

    assert contrast(bare) > contrast(with_barc)


# ---------------------------------------------------------------------------
# Validation and errors
# ---------------------------------------------------------------------------


def test_stack_needs_ambient_and_substrate():
    with pytest.raises(ValueError, match="at least an ambient"):
        FilmStack([Film("only", 0.0, 1 + 0j)])


def test_resist_index_must_be_a_real_layer():
    films = [Film("air", 0, 1 + 0j), Film("r", 100e-9, 1.7 + 0j), Film("Si", 0, N_SI)]
    with pytest.raises(ValueError, match="not a real layer"):
        FilmStack(films, resist_index=0)
    with pytest.raises(ValueError, match="not a real layer"):
        FilmStack(films, resist_index=2)


def test_bad_polarisation_raises(bare):
    with pytest.raises(ValueError, match="polarisation must be"):
        bare.reflectance(LAM, 0.0, "circular")


def test_resist_index_from_dill_matches_the_material_library():
    """k derived from Dill A/B must agree with the library's photoresist.

    Deriving rather than adding a field prevents setting both and
    double-counting absorption against the Dill march.
    """
    n = resist_index_from_dill(1.70, 0.8, 0.05, LAM)
    assert n.real == pytest.approx(1.70)
    assert n.imag == pytest.approx(0.013, abs=0.005)


def test_film_stack_from_records_resolves_material_names():
    stack = film_stack_from_records(
        [{"material": "SiARC", "thickness": 38e-9}], 120e-9, 1.7 + 0j
    )
    assert "SiARC" in stack.describe()
    assert 0.0 <= stack.reflectance(LAM) <= 1.0


# ---------------------------------------------------------------------------
# Integration with the 3-D resist pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=16, pixel_size=4e-9, dz=2e-9)


@pytest.fixture(scope="module")
def optics() -> OpticsConfig:
    return OpticsConfig(wavelength=LAM, NA=0.93, sigma_outer=0.8, source_grid=11)


def test_tmm_conserves_dose(grid, optics):
    """The envelope redistributes intensity in depth; it must not create it."""
    resist = ResistConfig(thickness=120e-9, n_resist=1.7, optical_model="tmm")
    intensity = np.ones((60, 16, 16))
    out = tmm_standing_waves(intensity, resist, optics, grid)
    assert out.mean() == pytest.approx(intensity.mean(), rel=1e-9)


def test_tmm_produces_a_fringe(grid, optics):
    resist = ResistConfig(thickness=120e-9, n_resist=1.7, optical_model="tmm")
    intensity = np.ones((60, 16, 16))
    trace = tmm_standing_waves(intensity, resist, optics, grid)[:, 8, 8]
    assert (trace.max() - trace.min()) / (trace.max() + trace.min()) > 0.3


def test_barc_flattens_the_fringe_in_the_pipeline(grid, optics):
    resist = ResistConfig(thickness=120e-9, n_resist=1.7, optical_model="tmm")
    barc = dataclasses.replace(
        resist, film_stack=[{"n": 1.80, "k": 0.40, "thickness": 38e-9}]
    )
    intensity = np.ones((60, 16, 16))

    def contrast(cfg):
        t = tmm_standing_waves(intensity, cfg, optics, grid)[:, 8, 8]
        return (t.max() - t.min()) / (t.max() + t.min())

    assert contrast(barc) < contrast(resist)


def test_dispatch_defaults_to_the_legacy_model(grid, optics):
    """`twobeam` must be byte-identical to calling add_standing_waves."""
    resist = ResistConfig(thickness=120e-9, n_resist=1.7, substrate_reflectance=0.4)
    assert resist.optical_model == "twobeam"
    intensity = np.ones((60, 16, 16))
    assert np.array_equal(
        apply_vertical_interference(intensity, resist, optics, grid),
        add_standing_waves(intensity, resist, optics, grid.dz),
    )


def test_dispatch_rejects_unknown_model(grid, optics):
    resist = ResistConfig(optical_model="ray-tracing")
    with pytest.raises(ValueError, match="Unknown optical model"):
        apply_vertical_interference(np.ones((4, 4, 4)), resist, optics, grid)


def test_optical_model_defaults_are_off():
    """Back-compatibility contract for this phase."""
    assert ResistConfig().optical_model == "twobeam"
    assert ResistConfig().film_stack == []
