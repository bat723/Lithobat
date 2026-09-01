"""
Interface tests for the CAR reaction–diffusion bake, through the public
package surface.

``tests/test_reaction.py`` checks the module against its analytic limits via
deep imports.  These tests instead exercise ``litho_sim.bake`` the way the
live consumers do — ``develop.resist.simulate_resist(model="car")`` with a
scalar quencher loading, and ``develop.stochastic.stochastic_trials`` with a
sampled per-voxel quencher field — and pin down the behaviours those
pipelines lean on: dose monotonicity of deprotection, the quencher acting as
a threshold rather than a blur, per-voxel loadings steering the result
spatially, stability-limited stepping, and input validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.bake import bake_reaction_diffusion, diffusion_length

PX = 4e-9

# The shape of the call simulate_resist(model="car") makes: acid from
# exposure, then chemistry rates from the resist config.
CAR_KWARGS = dict(
    pixel_size=PX,
    bake_time=20.0,
    D_acid=1.5e-17,
    D_quencher=1e-18,
    k_quench=50.0,
    k_loss=0.005,
    k_amp=5.0,
)


def _latent_image(n=96, rows=8):
    """A sinusoidal acid latent image in [0, 1], like a line/space exposure."""
    x = np.linspace(0.0, 1.0, n)
    return (0.5 * (1.0 + np.sin(2 * np.pi * 3 * x)))[None, :].repeat(rows, 0)


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------

def test_car_bake_returns_the_contracted_fields():
    """The dict contract the develop step consumes: acid, quencher,
    protected and steps, all field-shaped, protected in [0, 1]."""
    acid = _latent_image()
    out = bake_reaction_diffusion(acid, quencher=0.3, **CAR_KWARGS)

    assert set(out) == {"acid", "quencher", "protected", "steps"}
    for key in ("acid", "quencher", "protected"):
        assert out[key].shape == acid.shape
        assert np.all(np.isfinite(out[key]))
        assert np.all(out[key] >= 0.0)
    assert 0.0 <= out["protected"].max() <= 1.0
    assert out["steps"] >= 1
    # The integration must not write into the caller's latent image.
    assert np.array_equal(acid, _latent_image())


def test_deprotection_is_monotone_in_dose():
    """More exposure -> more acid -> more deprotection, dose after dose.
    This is the axis the process-window analysis sweeps."""
    base = _latent_image()
    deprotected = []
    for dose in (0.5, 1.0, 2.0, 4.0):
        out = bake_reaction_diffusion(base * dose, quencher=0.2, **CAR_KWARGS)
        deprotected.append(float((1.0 - out["protected"]).mean()))
    assert all(b > a for a, b in zip(deprotected, deprotected[1:])), deprotected


def test_scalar_quencher_suppresses_low_dose_more_than_high_dose():
    """The threshold behaviour the ``"car"`` model exists for, seen in the
    *protected* channel the develop step actually consumes: base annihilates
    sub-threshold acid, so quenched deprotection is suppressed far more at
    low dose than at high dose."""
    base = _latent_image()

    def mean_deprotection(dose, quencher):
        out = bake_reaction_diffusion(base * dose, quencher=quencher, **CAR_KWARGS)
        return float((1.0 - out["protected"]).mean())

    # Quencher present must always cost deprotection.
    assert mean_deprotection(1.0, 0.5) < mean_deprotection(1.0, 0.0)

    ratio_lo = mean_deprotection(0.4, 0.5) / mean_deprotection(0.4, 0.0)
    ratio_hi = mean_deprotection(3.0, 0.5) / mean_deprotection(3.0, 0.0)
    assert ratio_hi > ratio_lo + 0.1, (
        f"quencher suppression should be dose-asymmetric, got "
        f"lo={ratio_lo:.3f} hi={ratio_hi:.3f}"
    )


def test_pervoxel_quencher_field_steers_deprotection_spatially():
    """The stochastic path's seam: a quencher *field* loaded on one side
    must protect that side.  Uniform acid, base only on the left."""
    acid = np.full((16, 64), 0.8)
    quencher = np.zeros_like(acid)
    quencher[:, :32] = 0.6

    out = bake_reaction_diffusion(
        acid, PX, 10.0, 1e-18, quencher=quencher, k_quench=50.0, k_amp=0.2,
    )
    left = out["protected"][:, :16].mean()   # quenched quarter
    right = out["protected"][:, -16:].mean()  # unquenched quarter
    assert left > right + 0.1, (left, right)


def test_scalar_and_uniform_field_quencher_agree_in_3d():
    """The scalar/per-voxel seam again, but on the 3-D anisotropic grid the
    resist-volume path uses — the two parameterisations must be the same
    bake."""
    rng = np.random.default_rng(7)
    acid = rng.uniform(0.0, 1.0, (10, 14, 14))
    kwargs = dict(
        pixel_size=PX, bake_time=2.0, D_acid=2e-18,
        spacing=(2e-9, 4e-9, 4e-9), k_quench=5.0, k_amp=0.2,
    )
    scalar = bake_reaction_diffusion(acid, quencher=0.3, **kwargs)
    field = bake_reaction_diffusion(
        acid, quencher=np.full(acid.shape, 0.3), **kwargs
    )
    for key in ("acid", "quencher", "protected"):
        assert np.array_equal(scalar[key], field[key])
    assert scalar["steps"] == field["steps"]


# ---------------------------------------------------------------------------
# diffusion_length
# ---------------------------------------------------------------------------

def test_diffusion_length_scales_as_sqrt_of_D_and_t():
    """sigma = sqrt(2 D t): quadrupling either argument doubles it."""
    D, t = 2e-17, 15.0
    sigma = diffusion_length(D, t)
    assert sigma > 0.0
    assert diffusion_length(D, 4.0 * t) == pytest.approx(2.0 * sigma)
    assert diffusion_length(4.0 * D, t) == pytest.approx(2.0 * sigma)
    assert diffusion_length(2.0 * D, 2.0 * t) == pytest.approx(2.0 * sigma)


# ---------------------------------------------------------------------------
# Stability-limited stepping
# ---------------------------------------------------------------------------

def test_large_D_t_is_stability_limited_not_unstable():
    """A diffusion length far beyond the grid must cost steps, not blow up:
    the stepper picks dt from the stability limit, so the field relaxes to
    its (conserved) mean instead of oscillating or going NaN."""
    rng = np.random.default_rng(3)
    field = rng.uniform(0.0, 1.0, (32, 32))
    D, t = 4e-16, 60.0  # sigma ~ 220 nm on a 128 nm grid
    out = bake_reaction_diffusion(field, PX, t, D)

    assert out["steps"] > 100, "large D*t should force many small steps"
    assert np.all(np.isfinite(out["acid"]))
    assert np.all(out["acid"] >= 0.0)
    # Zero-flux edges + no sinks: acid conserved, and this much diffusion
    # flattens the field onto its mean.
    assert out["acid"].sum() == pytest.approx(field.sum(), rel=1e-9)
    assert np.allclose(out["acid"], field.mean(), atol=1e-6)


def test_fast_kinetics_stay_finite_and_bounded():
    """Aggressive rate constants shrink dt via the reaction timescale; the
    result must remain finite with concentrations that never go negative."""
    acid = _latent_image(n=64, rows=4)
    out = bake_reaction_diffusion(
        acid, PX, 30.0, 1.5e-17,
        quencher=0.5, k_quench=500.0, k_loss=0.1, k_amp=50.0,
    )
    for key in ("acid", "quencher", "protected"):
        assert np.all(np.isfinite(out[key]))
        assert np.all(out[key] >= 0.0)
    assert np.all(out["protected"] <= 1.0)


def test_reaction_limited_cap_stays_finite():
    """Capping at max_steps when only the *reaction* timescale wants more
    steps is a documented accuracy trade — the result must stay finite and
    non-negative, never NaN."""
    out = bake_reaction_diffusion(
        np.full((8, 8), 1.0), PX, 10.0, 0.0,
        quencher=0.5, k_quench=1e5, max_steps=50,
    )
    assert out["steps"] == 50
    for key in ("acid", "quencher", "protected"):
        assert np.all(np.isfinite(out[key]))
        assert np.all(out[key] >= 0.0)


def test_diffusion_unstable_cap_raises_instead_of_returning_nan():
    """When max_steps cannot accommodate the diffusion stability limit the
    explicit stepper would diverge to NaN — the call must refuse."""
    with pytest.raises(ValueError, match="stability"):
        bake_reaction_diffusion(
            np.ones((32, 32)), PX, 60.0, 1e-14, max_steps=200
        )


# ---------------------------------------------------------------------------
# Edge and error cases
# ---------------------------------------------------------------------------

def test_zero_bake_time_is_the_identity():
    """t = 0 is a well-defined limit: nothing diffuses, nothing reacts."""
    rng = np.random.default_rng(11)
    acid = rng.random((12, 12))
    quencher = rng.random((12, 12))
    out = bake_reaction_diffusion(acid, PX, 0.0, 1e-17,
                                  quencher=quencher, k_quench=50.0, k_amp=1.0)
    assert out["steps"] == 0
    assert np.array_equal(out["acid"], acid)
    assert np.array_equal(out["quencher"], quencher)
    assert np.all(out["protected"] == 1.0)
    # Returned fields are copies, not views of caller memory.
    out["acid"][0, 0] = -99.0
    assert acid[0, 0] != -99.0


def test_negative_bake_time_raises():
    with pytest.raises(ValueError, match="bake_time"):
        bake_reaction_diffusion(np.ones((8, 8)), PX, -5.0, 1e-17)


def test_mismatched_quencher_shape_raises():
    """A per-voxel loading that cannot be broadcast onto the acid grid is a
    caller bug, not a bake."""
    with pytest.raises(ValueError):
        bake_reaction_diffusion(
            np.ones((8, 8)), PX, 1.0, 1e-17, quencher=np.ones((5, 7))
        )


def test_short_spacing_tuple_raises():
    """3-D acid with a 2-entry spacing — the under-specified twin of the
    over-specified case the unit tests pin."""
    with pytest.raises(ValueError, match="spacing has"):
        bake_reaction_diffusion(
            np.ones((4, 4, 4)), PX, 1.0, 1e-17, spacing=(1e-9, 1e-9)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(pixel_size=0.0),
        dict(pixel_size=-PX),
        dict(pixel_size=PX, spacing=(4e-9, -4e-9)),
    ],
)
def test_non_positive_voxel_sizes_raise(kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        bake_reaction_diffusion(
            np.ones((8, 8)), bake_time=1.0, D_acid=1e-17, **kwargs
        )
