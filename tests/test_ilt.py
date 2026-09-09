"""
Tests for inverse lithography.

Five contracts are pinned:

* **The adjoint is the gradient.** Both the bare SOCS adjoint and the full
  objective — penalties included — agree with a central finite difference to
  the difference's own noise floor. This is the test that matters: a wrong
  gradient still descends, just to the wrong place, and every other symptom it
  produces looks like a tuning problem.
* **The forward pass is the engine's.** ``socs_fields`` reproduces
  ``SOCSKernels.raw`` exactly, so the thing being differentiated is the thing
  the rest of the engine prints.
* **The solve corrects.** On a line/space target it drives the pattern error
  down by an order of magnitude, and the mask it returns is curvilinear —
  which is the whole point, and something no edge-based correction can do.
* **Assist features emerge.** Mask energy appears in empty field beside the
  drawn features, placed by the gradient rather than by the rule in
  ``opc.assist``.
* **A degenerate print is refused.** A mask that prints nothing has no contour
  and no gradient, so the solve raises instead of reporting that it converged.
* **The curvature limit is a curve, and still a gradient.** Drawing the mask
  through a Gaussian removes the pixel-scale structure the sharpening would
  otherwise freeze into the outline, and the chain rule through it agrees with
  finite differences — at the field's edges too, where the boundary handling
  would show if the blur were not its own adjoint.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.expose.hopkins import SOCSKernels
from litho_sim.mask.patterns import contact_array, lines_and_spaces
from litho_sim.opc.ilt import (
    ILTFrame,
    _grad_theta,
    _mask_of,
    _Objective,
    dose_for_target,
    print_mask,
    run_ilt,
    socs_fields,
    socs_gradient,
)
from litho_sim.opc.model import PrintModel


@pytest.fixture(scope="module")
def small():
    """A grid small enough to finite-difference on."""
    grid = GridConfig(n_pixels=32, pixel_size=8e-9)
    return grid, SOCSKernels.build(OpticsConfig(), grid, tol=1e-6)


def _model(n=128, px=4e-9, **kw):
    grid = GridConfig(n_pixels=n, pixel_size=px)
    return PrintModel(
        optics=OpticsConfig(), resist=ResistConfig(), grid=grid, tone="clear",
        model="threshold", normalisation="clear", imaging="socs", **kw,
    )


# ---------------------------------------------------------------------------
# The forward pass is the engine's own
# ---------------------------------------------------------------------------


def test_socs_fields_reproduces_the_kernels_raw_image(small):
    grid, kernels = small
    rng = np.random.default_rng(1)
    mask = rng.random((grid.n_pixels, grid.n_pixels))

    _, intensity = socs_fields(kernels, mask)
    raw, _peak, _clear = kernels.raw(mask)

    assert np.allclose(intensity, raw, rtol=0, atol=1e-12)


def test_socs_fields_rejects_a_mask_of_the_wrong_shape(small):
    _grid, kernels = small
    with pytest.raises(ValueError, match="mask must be"):
        socs_fields(kernels, np.zeros((8, 8)))


# ---------------------------------------------------------------------------
# The adjoint is the gradient
# ---------------------------------------------------------------------------


def test_socs_adjoint_matches_finite_differences(small):
    """``socs_gradient`` is d/dm of ``Σ_x sensitivity(x)·I(x)``."""
    grid, kernels = small
    n = grid.n_pixels
    rng = np.random.default_rng(2)
    mask = rng.random((n, n))
    sens = rng.standard_normal((n, n))

    fields, _ = socs_fields(kernels, mask)
    analytic = socs_gradient(kernels, fields, sens)

    def cost(m):
        _, intensity = socs_fields(kernels, m)
        return float(np.sum(sens * intensity))

    h = 1e-6
    for i, j in rng.integers(0, n, size=(8, 2)):
        up, dn = mask.copy(), mask.copy()
        up[i, j] += h
        dn[i, j] -= h
        fd = (cost(up) - cost(dn)) / (2 * h)
        assert fd == pytest.approx(analytic[i, j], rel=1e-5, abs=1e-9)


def test_full_objective_gradient_matches_finite_differences(small):
    """Resist relaxation, PEB blur and both penalties, all differentiated."""
    grid, kernels = small
    n = grid.n_pixels
    rng = np.random.default_rng(3)
    mask = rng.random((n, n))
    target = (rng.random((n, n)) > 0.5).astype(float)

    obj = _Objective(
        kernels=kernels, target=target, scale=1.7, level=0.3, sigma_px=1.5,
        sign=1.0, steepness=12.0, weight_tv=1e-3, weight_discrete=1e-3,
    )
    _cost, analytic, _signed, _pe = obj(mask)

    h = 1e-6
    for i, j in rng.integers(0, n, size=(8, 2)):
        up, dn = mask.copy(), mask.copy()
        up[i, j] += h
        dn[i, j] -= h
        fd = (obj(up)[0] - obj(dn)[0]) / (2 * h)
        assert fd == pytest.approx(analytic[i, j], rel=1e-5, abs=1e-12)


@pytest.mark.parametrize("tone", ["clear", "dark"])
def test_gradient_is_correct_for_either_mask_tone(small, tone):
    grid, kernels = small
    n = grid.n_pixels
    rng = np.random.default_rng(4)
    mask = rng.random((n, n))
    target = (rng.random((n, n)) > 0.5).astype(float)

    obj = _Objective(
        kernels=kernels, target=target, scale=2.0, level=0.3, sigma_px=0.0,
        sign=1.0 if tone == "clear" else -1.0, steepness=20.0,
        weight_tv=0.0, weight_discrete=0.0,
    )
    _c, analytic, _s, _pe = obj(mask)

    # A central difference of a unit-scale cost at h=1e-6 has a roundoff floor
    # near 1e-10, so the absolute tolerance is what governs the small gradient
    # components and the relative one governs the rest.
    h = 1e-6
    for i, j in rng.integers(0, n, size=(6, 2)):
        up, dn = mask.copy(), mask.copy()
        up[i, j] += h
        dn[i, j] -= h
        fd = (obj(up)[0] - obj(dn)[0]) / (2 * h)
        assert fd == pytest.approx(analytic[i, j], rel=1e-4, abs=1e-9)


def test_the_parameterisation_gradient_matches_finite_differences(small):
    """``σ(β·G θ)`` end to end: the sigmoid's slope and the Gaussian's adjoint."""
    grid, kernels = small
    n = grid.n_pixels
    rng = np.random.default_rng(5)
    theta = 0.5 * rng.standard_normal((n, n))
    target = (rng.random((n, n)) > 0.5).astype(float)
    obj = _Objective(
        kernels=kernels, target=target, scale=1.7, level=0.3, sigma_px=1.0,
        sign=1.0, steepness=12.0, weight_tv=1e-3, weight_discrete=0.0,
    )
    beta, smooth_px = 6.0, 2.0
    mask = _mask_of(theta, beta, smooth_px)
    _cost, grad_m, _signed, _pe = obj(mask)
    analytic = _grad_theta(grad_m, mask, beta, smooth_px)

    def cost(th):
        return obj(_mask_of(th, beta, smooth_px))[0]

    # Interior pixels, and the field's corners and edges on purpose.
    pixels = [(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1), (0, 5), (n - 1, 9)]
    pixels += [tuple(ij) for ij in rng.integers(1, n - 1, size=(6, 2))]
    h = 1e-6
    for i, j in pixels:
        up, dn = theta.copy(), theta.copy()
        up[i, j] += h
        dn[i, j] -= h
        fd = (cost(up) - cost(dn)) / (2 * h)
        assert fd == pytest.approx(analytic[i, j], rel=1e-4, abs=1e-9), (i, j)


def test_no_smoothing_is_the_plain_sigmoid():
    theta = np.linspace(-2.0, 2.0, 16).reshape(4, 4)
    assert np.allclose(_mask_of(theta, 3.0, 0.0), 1.0 / (1.0 + np.exp(-3.0 * theta)))


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def line_solve():
    """One real solve, shared — it is the expensive fixture in this file."""
    model = _model()
    target = lines_and_spaces(128, 4e-9, pitch=128e-9, cd=64e-9)
    model = dataclasses.replace(model, dose=dose_for_target(target, model))
    before = int(np.count_nonzero((print_mask(target, model).signed > 0) != (target > 0.5)))
    result = run_ilt(
        target, model, max_iter=150, step=0.03, beta=4.0, beta_final=16.0,
        initial_grey=0.99, steepness=25.0, weight_tv=1e-3,
    )
    return model, target, before, result


def test_ilt_reduces_pattern_error_by_an_order_of_magnitude(line_solve):
    _model_, _target, before, result = line_solve
    after = int(result.history.iloc[-1].pattern_error)
    assert after < before / 10


def test_the_corrected_mask_is_curvilinear(line_solve):
    """Grey pixels are edges the solve placed between the drawn ones.

    An edge-based correction can only return the drawn polygon with its
    vertices moved; every pixel is 0 or 1. A mask with intermediate values
    along its boundaries is one no fragment offset could have produced.
    """
    _model_, _target, _before, result = line_solve
    grey = np.count_nonzero((result.mask > 0.05) & (result.mask < 0.95))
    assert grey > 0
    assert result.mask.min() < 0.05
    assert result.mask.max() > 0.95


def test_assist_features_appear_in_empty_field(line_solve):
    """Mask energy off the drawn features, placed by the gradient alone."""
    _model_, target, _before, result = line_solve
    outside = result.mask[target < 0.5]
    assert outside.max() > 0.25


def test_thresholding_the_mask_keeps_most_of_the_correction(line_solve):
    """The continuous mask is state; a writer takes the binary one.

    At its own dose. Thresholding moves the mask's total transmission and with
    it every printed edge, so re-sizing is part of taking the binary mask, not
    a correction to it — see :meth:`ILTResult.binary`.
    """
    model, target, before, result = line_solve
    binary = result.binary()
    sized = dataclasses.replace(model, dose=dose_for_target(target, model, mask=binary))
    printed = print_mask(binary, sized)
    after = int(np.count_nonzero((printed.signed > 0) != (target > 0.5)))
    assert after < before / 5


def test_the_binary_mask_needs_its_own_dose(line_solve):
    """Pins the trap: the same mask at the continuous solve's dose is far worse."""
    model, target, _before, result = line_solve
    binary = result.binary()
    at_solve_dose = int(np.count_nonzero(
        (print_mask(binary, model).signed > 0) != (target > 0.5)))
    sized = dataclasses.replace(model, dose=dose_for_target(target, model, mask=binary))
    at_own_dose = int(np.count_nonzero(
        (print_mask(binary, sized).signed > 0) != (target > 0.5)))
    assert at_own_dose < at_solve_dose / 4


def test_history_and_frames_cover_every_iteration(line_solve):
    _model_, _target, _before, result = line_solve
    h = result.history
    assert list(h.iteration) == list(range(len(h)))
    assert h.cost.iloc[-1] < h.cost.iloc[0]


def test_progress_streams_a_frame_per_iteration():
    """What the live viewer consumes — including one before the first step."""
    model = _model(n=64, px=8e-9)
    target = lines_and_spaces(64, 8e-9, pitch=128e-9, cd=64e-9)
    model = dataclasses.replace(model, dose=dose_for_target(target, model))

    frames: list[ILTFrame] = []
    result = run_ilt(target, model, max_iter=6, step=0.03, progress=frames.append)

    assert len(frames) == result.iterations + 1
    assert frames[0].iteration == 0
    assert [f.iteration for f in frames] == list(range(len(frames)))
    for f in frames:
        assert f.mask.shape == (64, 64)
        assert f.signed.shape == (64, 64)
        assert 0.0 <= f.mask.min() and f.mask.max() <= 1.0

    # The frames are copies: the solver keeps stepping its own arrays.
    assert not np.shares_memory(frames[0].mask, frames[-1].mask)
    assert not np.allclose(frames[0].mask, frames[-1].mask)


# ---------------------------------------------------------------------------
# Sizing, and the failure that used to look like success
# ---------------------------------------------------------------------------


def test_dose_for_target_makes_the_drawn_mask_print_its_own_area():
    model = _model()
    target = contact_array(128, 4e-9, pitch_x=240e-9, pitch_y=240e-9, cd_x=90e-9)
    sized = dataclasses.replace(model, dose=dose_for_target(target, model))
    printed = print_mask(target, sized)
    assert int(printed.printed.sum()) == pytest.approx(int(target.sum()), rel=0.02)


def test_an_undosed_target_that_prints_nothing_is_refused():
    """The failure mode this guard exists for.

    At dose 1 the contact array's image never reaches the resist level, so
    nothing prints, the relaxed resist saturates, and the gradient is zero
    everywhere — a solve that stops immediately and calls itself converged.
    """
    model = _model()
    target = contact_array(128, 4e-9, pitch_x=160e-9, pitch_y=160e-9, cd_x=70e-9)
    assert int(print_mask(target, model).printed.sum()) == 0
    with pytest.raises(ValueError, match="prints nothing"):
        run_ilt(target, model, max_iter=5)


def test_peak_normalisation_is_refused():
    grid = GridConfig(n_pixels=64, pixel_size=8e-9)
    model = PrintModel(
        optics=OpticsConfig(), resist=ResistConfig(), grid=grid, tone="clear",
        model="threshold", normalisation="peak", imaging="socs",
    )
    target = lines_and_spaces(64, 8e-9, pitch=128e-9, cd=64e-9)
    with pytest.raises(ValueError, match="mask-independent normalisation"):
        run_ilt(target, model, max_iter=2)


def test_a_thick_mask_process_is_refused():
    """The adjoint has no thick-mask form; saying so beats approximating."""
    grid = GridConfig(n_pixels=64, pixel_size=8e-9)
    optics = dataclasses.replace(OpticsConfig(), mask_model="fdtd")
    model = PrintModel(
        optics=optics, resist=ResistConfig(), grid=grid, tone="clear",
        model="threshold", normalisation="clear", imaging="auto",
    )
    target = lines_and_spaces(64, 8e-9, pitch=128e-9, cd=64e-9)
    with pytest.raises(ValueError, match="SOCS kernels"):
        run_ilt(target, model, max_iter=2)


def test_target_shape_must_match_the_grid():
    model = _model(n=64, px=8e-9)
    with pytest.raises(ValueError, match="target must be"):
        run_ilt(np.zeros((32, 32)), model, max_iter=2)


# ---------------------------------------------------------------------------
# The curvature limit
# ---------------------------------------------------------------------------


def test_the_curvature_limit_bounds_feature_size_by_construction():
    """White noise in θ is speckle when the mask is free per pixel, and blobs
    no finer than the Gaussian when it is not. A morphological opening two
    pixels wide destroys the one and leaves the other almost untouched — the
    property the outline's smoothness rests on, independent of any solve."""
    from scipy.ndimage import binary_opening

    rng = np.random.default_rng(6)
    theta = rng.standard_normal((96, 96))
    y, x = np.ogrid[-2:3, -2:3]
    disc = (x ** 2 + y ** 2) <= 4

    def survives(mask):
        cut = mask >= 0.5
        return np.count_nonzero(binary_opening(cut, structure=disc)) / np.count_nonzero(cut)

    assert survives(_mask_of(theta, 8.0, 0.0)) < 0.05
    assert survives(_mask_of(theta, 8.0, 3.0)) > 0.9


def test_a_smoothed_solve_still_corrects():
    """The curvature limit costs the correction little on a target it can
    represent: a line/space array, where the free solve reaches zero error."""
    model = _model(n=64, px=8e-9)
    target = lines_and_spaces(64, 8e-9, pitch=128e-9, cd=64e-9)
    model = dataclasses.replace(model, dose=dose_for_target(target, model))
    drawn = int(np.count_nonzero((print_mask(target, model).signed > 0) != (target > 0.5)))
    kw = dict(max_iter=60, step=0.03, weight_tv=1e-3)
    free = int(run_ilt(target, model, **kw).history.pattern_error.iloc[-1])
    curved = int(run_ilt(target, model, smooth=16e-9, **kw).history.pattern_error.iloc[-1])
    assert curved < drawn
    assert curved <= free + 8


def test_a_negative_curvature_limit_is_refused():
    model = _model(n=64, px=8e-9)
    target = lines_and_spaces(64, 8e-9, pitch=128e-9, cd=64e-9)
    with pytest.raises(ValueError, match="cannot be negative"):
        run_ilt(target, model, max_iter=2, smooth=-1e-9)
