"""
Post-exposure bake as reaction–diffusion.

The parts with analytic answers are checked against them: pure diffusion of a
Gaussian is a wider Gaussian, first-order loss is an exponential, and with
every reaction switched off the whole thing must conserve acid exactly. What is
left over — the quencher — is checked for the behaviour that motivates it, that
it makes the bake non-linear in dose.
"""

import numpy as np
import pytest

from litho_sim.bake.reaction import bake_reaction_diffusion, diffusion_length


def _gaussian_2d(n, sigma_px, centre=None):
    c = (n - 1) / 2 if centre is None else centre
    g = np.arange(n) - c
    r2 = g[:, None] ** 2 + g[None, :] ** 2
    return np.exp(-r2 / (2 * sigma_px ** 2))


def test_diffusion_length_is_the_usual_definition():
    assert diffusion_length(2e-17, 50.0) == pytest.approx(np.sqrt(2 * 2e-17 * 50.0))
    assert diffusion_length(0.0, 60.0) == 0.0


def test_pure_diffusion_of_a_gaussian_is_a_wider_gaussian():
    """The one case with a closed form: widths add in quadrature.

    Done on a compact blob far from the edges, so the boundary condition —
    zero-flux here, which is what a bake on a wafer does — plays no part.
    """
    n, px = 96, 4e-9
    sigma0_px = 6.0
    field = _gaussian_2d(n, sigma0_px)

    D, t = 1.2e-17, 40.0
    sigma_add = diffusion_length(D, t) / px
    out = bake_reaction_diffusion(field, px, t, D)["acid"]

    expected_px = np.sqrt(sigma0_px ** 2 + sigma_add ** 2)
    # Recover the width from the second moment rather than by fitting.
    g = np.arange(n) - (n - 1) / 2
    w = out / out.sum()
    var = (w * (g[:, None] ** 2 + g[None, :] ** 2)).sum() / 2.0
    assert np.sqrt(var) == pytest.approx(expected_px, rel=0.02)


def test_acid_is_conserved_without_loss_or_quencher():
    rng = np.random.default_rng(0)
    field = _gaussian_2d(64, 8.0) * rng.uniform(0.5, 1.0, (64, 64))
    out = bake_reaction_diffusion(field, 4e-9, 30.0, 1e-17)["acid"]
    assert out.sum() == pytest.approx(field.sum(), rel=1e-9)


def test_first_order_loss_is_exponential():
    field = np.full((16, 16), 1.0)
    k, t = 0.02, 60.0
    out = bake_reaction_diffusion(field, 4e-9, t, 0.0, k_loss=k)["acid"]
    assert out.mean() == pytest.approx(np.exp(-k * t), rel=1e-3)


def test_zero_everything_leaves_the_field_alone():
    rng = np.random.default_rng(1)
    field = rng.random((32, 32))
    out = bake_reaction_diffusion(field, 4e-9, 10.0, 0.0)["acid"]
    assert np.allclose(out, field)


def test_the_quencher_makes_the_bake_non_linear_in_dose():
    """The whole reason this module exists.

    A Gaussian blur is a convolution, so scaling the input scales the output
    by exactly the same factor — the surviving fraction cannot depend on dose.
    Neutralisation annihilates acid below the quencher level and spares acid
    above it, which is a threshold, and thresholds are where a real resist's
    contrast comes from.
    """
    x = np.linspace(0, 1, 128)
    pattern = (0.5 * (1 + np.sin(2 * np.pi * 3 * x)))[None, :].repeat(8, 0)

    def surviving_fraction(scale, quencher):
        out = bake_reaction_diffusion(
            pattern * scale, 4e-9, 60.0, 1.5e-17,
            quencher=quencher, k_quench=50.0,
        )["acid"]
        return out.sum() / (pattern.sum() * scale)

    linear_lo = surviving_fraction(0.8, 0.0)
    linear_hi = surviving_fraction(1.25, 0.0)
    assert linear_lo == pytest.approx(linear_hi, rel=1e-6), (
        "with no quencher this must be exactly linear, like the blur it replaces"
    )

    quenched_lo = surviving_fraction(0.8, 0.5)
    quenched_hi = surviving_fraction(1.25, 0.5)
    assert quenched_hi - quenched_lo > 0.1, (
        "quencher present but the bake is still dose-linear — the "
        "neutralisation term is not doing anything"
    )


def test_quencher_is_consumed_where_the_acid_is():
    x = np.linspace(0, 1, 64)
    pattern = (0.5 * (1 + np.sin(2 * np.pi * 2 * x)))[None, :].repeat(4, 0)
    out = bake_reaction_diffusion(
        pattern, 4e-9, 30.0, 0.0, quencher=0.4, k_quench=80.0
    )
    left = out["quencher"]
    bright = pattern[0] > 0.75
    dim = pattern[0] < 0.25
    assert left[:, bright].mean() < left[:, dim].mean()


def test_deprotection_follows_the_acid():
    x = np.linspace(0, 1, 64)
    pattern = (0.5 * (1 + np.sin(2 * np.pi * 2 * x)))[None, :].repeat(4, 0)
    out = bake_reaction_diffusion(pattern, 4e-9, 30.0, 0.0, k_amp=0.1)
    M = out["protected"]
    assert M.max() <= 1.0 and M.min() >= 0.0
    assert M[0].argmin() == pattern[0].argmax(), "deprotected away from the acid"


def test_anisotropic_spacing_is_respected():
    """3-D here has dz != pixel_size, so this is the normal case."""
    field = np.zeros((21, 21, 21))
    field[10, 10, 10] = 1.0
    out = bake_reaction_diffusion(
        field, 4e-9, 20.0, 5e-18, spacing=(1e-9, 8e-9, 8e-9)
    )["acid"]

    def spread(profile):
        """Second moment in voxels — how far it actually got, not how tall."""
        g = np.arange(profile.size) - 10
        w = profile / profile.sum()
        return float(np.sqrt((w * g * g).sum()))

    # Same physical diffusion length; the axis with the smaller voxels
    # therefore covers more voxels.
    assert spread(out[:, 10, 10]) > spread(out[10, 10, :])


def test_spacing_length_is_checked():
    with pytest.raises(ValueError, match="spacing has"):
        bake_reaction_diffusion(np.ones((4, 4)), 4e-9, 1.0, 1e-17,
                                spacing=(1e-9, 1e-9, 1e-9))


def test_array_quencher_matches_uniform_scalar():
    """A quencher *field* that happens to be uniform must reproduce the scalar
    path bit-for-bit — the seam the stochastic sampler feeds through."""
    field = _gaussian_2d(48, 5.0)
    kwargs = dict(pixel_size=4e-9, bake_time=2.0, D_acid=2e-18,
                  k_quench=5.0, k_amp=0.2)
    scalar = bake_reaction_diffusion(field, quencher=0.3, **kwargs)
    array = bake_reaction_diffusion(field, quencher=np.full_like(field, 0.3),
                                    **kwargs)
    for key in ("acid", "quencher", "protected"):
        assert np.array_equal(scalar[key], array[key])
