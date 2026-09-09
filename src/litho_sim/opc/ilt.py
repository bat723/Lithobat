"""
Inverse lithography: the mask solved for, not edited.

Model-based OPC in :mod:`litho_sim.opc.correct` starts from the drawn polygon
and asks where its edges should move. That question carries an assumption
worth naming: the answer is a polygon, with the edges the designer drew. A
fragment slides along its normal, and no sequence of such slides ever produces
a curve — only a Manhattan outline with more and more vertices in it.

Inverse lithography drops the polygon. The mask becomes a *field* ``m(x)`` on
the imaging grid, every pixel free, and the correction becomes a minimisation::

    minimise   J(m) = ‖ resist(I[m]) − target ‖²  +  penalties
    over       m ∈ [0, 1]^(n×n)

What comes out is curvilinear because nothing ever constrained it not to be.
Sub-resolution assist features come out too, in the empty field beside a
feature, nucleated by the gradient rather than placed by the rule in
:mod:`litho_sim.opc.assist` — which is the interesting comparison, and the
reason both live in this package.

Curvilinear and *smooth* are different things, though. Free per pixel, the
mask can carry structure the optics cannot see, and nothing in the objective
minds; the sharpening at the end of a solve then snaps every grey pixel to
whichever level it is nearer, and the outline comes out as a staircase with
bumps on it — rectangles with noise, to the eye, however curved the physics
wanted it. ``smooth`` draws the mask through a Gaussian of that width::

    m = σ( β · G_s ∗ θ )

which is a minimum radius of curvature imposed by construction: nothing in the
mask is finer than ``s``, or turns more sharply. A Gaussian is its own adjoint,
so the gradient stays exact, and the thresholded mask keeps more of the
correction because fewer pixels are left sitting on the fence at 0.5.

Why the gradient is cheap
-------------------------
The forward model is the sum of coherent systems (:mod:`litho_sim.expose.hopkins`)::

    E_k = F⁻¹( φ_k · F(m) ),     I = Σ_k λ_k |E_k|²

Each ``E_k`` is a *convolution* of the mask, so the operator ``L_k : m ↦ E_k``
is diagonal in frequency, and the adjoint of a diagonal operator is its
conjugate::

    L_kᴴ(v) = F⁻¹( φ_k* · F(v) )

The DFT normalisation cancels exactly — ``Fᴴ = N²F⁻¹`` and ``(F⁻¹)ᴴ = F/N²`` —
so no scale factor survives into the adjoint. Writing ``G = ∂J/∂I`` for the
(real) sensitivity of the cost to intensity, the chain rule gives

.. math::

    \\nabla_m J = 2 \\sum_k \\lambda_k \\,
        \\mathrm{Re}\\left\\{ \\mathcal{F}^{-1}\\!\\left(
        \\phi_k^{*} \\cdot \\mathcal{F}(G \\cdot E_k) \\right) \\right\\}

which costs one transform pair per kernel — the same order as the forward
image. A whole gradient is about three prints' work, *not* the ``n²`` prints a
finite-difference gradient over every mask pixel would cost. That ratio is the
only reason inverse lithography is tractable at all, and it is worth
internalising before trusting any of the numbers below.

The gradient is verified against finite differences in the test suite. Do that
first whenever this file is edited: a wrong gradient still descends, just to
the wrong place, and it will look like a tuning problem for days.

Watching it converge
--------------------
:func:`run_ilt` takes a ``progress`` callback and hands it an :class:`ILTFrame`
every iteration — the current mask and the signed field its printed contour is
the zero set of. That is enough to draw the whole state of the solve, and it
is how the app animates a correction live rather than showing a spinner. The
callback is on the caller's thread and is called synchronously, so a slow one
slows the solve; keep it to a queued signal emit.

At the engine's default ArF preset one iteration is about 66 ms on a 128-pixel
field and about 0.5 s on a 256-pixel one, so 128 px animates at roughly 15 fps
and 256 px at about 2.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from litho_sim.expose.aerial_image import normalisation_scale
from litho_sim.expose.hopkins import SOCSKernels
from litho_sim.opc.model import PrintedImage, PrintModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# What the caller sees
# ---------------------------------------------------------------------------


@dataclass
class ILTFrame:
    """One iteration of a solve, handed to a live viewer.

    Everything needed to draw the state and nothing that costs extra to
    produce: the mask and the signed field are already in hand when the
    gradient is formed, so streaming them is free.

    Attributes
    ----------
    iteration : int
        0-based. Iteration 0 is the *initial* mask, emitted before the first
        step, so a viewer opens on the uncorrected print — the design drawn
        through the curvature limit, when there is one, so its corners are
        already rounded to what the parameterisation can represent.
    mask : NDArray
        The current continuous mask, ``(n, n)`` in ``[0, 1]``. A view of the
        solver's own array is never handed out — this is a copy, safe to keep.
    signed : NDArray
        The signed resist field, positive inside the printed feature. Its
        zero contour *is* the printed edge, by the same definition
        :class:`~litho_sim.opc.model.PrintedImage` uses, so a viewer can draw
        the contour without knowing anything about resist models.
    cost : float
        The objective, including penalties.
    pattern_error : int
        Pixels where the print disagrees with the target. The standard
        inverse-lithography scalar, and the one to watch: the cost can fall
        while this sits still, which means the contour is sharpening rather
        than moving.
    beta : float
        The current mask-sharpening parameter — see :func:`run_ilt`.
    """

    iteration: int
    mask: NDArray[np.float64]
    signed: NDArray[np.float64]
    cost: float
    pattern_error: int
    beta: float


@dataclass
class ILTResult:
    """A finished solve.

    Attributes
    ----------
    mask : NDArray
        The optimised continuous mask, ``(n, n)`` in ``[0, 1]``. Continuous
        because the solve is; see :meth:`binary` for the manufacturable form.
    theta : NDArray
        The unconstrained latent variable behind it, ``mask = σ(β·G(theta))``
        with ``G`` the curvature smoothing (the identity when ``smooth`` is
        0). Kept so a solve can be resumed or annealed further.
    target : NDArray
        The binary design the solve was aiming at.
    printed : PrintedImage
        The final print, for contours and CD measurement.
    history : pandas.DataFrame
        One row per iteration: ``iteration``, ``cost``, ``pattern_error``,
        ``beta``, ``grad_norm``.
    iterations : int
        Steps actually taken.
    converged : bool
        Whether the cost stopped improving by more than ``tol`` before the
        iteration cap. False means the cap was hit, and the mask is whatever
        the last step produced — still usable, just not settled.
    """

    mask: NDArray[np.float64]
    theta: NDArray[np.float64]
    target: NDArray[np.bool_]
    printed: PrintedImage
    history: pd.DataFrame = field(repr=False)
    iterations: int = 0
    converged: bool = False

    def binary(self, level: float = 0.5) -> NDArray[np.float64]:
        """The mask thresholded to two levels — what a writer would take.

        The continuous mask is the solve's internal state, not a mask order.
        Thresholding it costs some of the correction back, and how much is the
        honest measure of whether the anneal ran long enough.

        **Re-size the dose before judging it.** Thresholding changes the mask's
        total transmission, which moves every printed edge together — on a
        128 nm-pitch line/space that alone is worth a factor of eight in
        pattern error (2048 px against 256), and it is a uniform CD bias, not
        a failure of the correction. A mask and its dose are sized together::

            binary = result.binary()
            dose   = dose_for_target(result.target, model, mask=binary)
            print  = print_mask(binary, dataclasses.replace(model, dose=dose))
        """
        return (np.asarray(self.mask) >= level).astype(np.float64)


# ---------------------------------------------------------------------------
# Forward and adjoint
# ---------------------------------------------------------------------------


def socs_fields(
    kernels: SOCSKernels, mask: NDArray[np.float64]
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """The per-kernel coherent fields and the raw intensity they sum to.

    :meth:`~litho_sim.expose.hopkins.SOCSKernels.raw` throws the fields away
    once it has squared them. The adjoint needs them back — ``∇J`` is built
    from ``G·E_k``, not from ``I`` — so this is that method with the
    intermediate kept, and it is the one place the two must agree.

    Returns
    -------
    (fields, intensity) : tuple
        ``fields`` is ``(K, n, n)`` complex; ``intensity`` is the same
        un-normalised, un-dosed sum ``raw`` returns first.
    """
    n = kernels.grid.n_pixels
    if mask.shape != (n, n):
        raise ValueError(f"mask must be {(n, n)} for these kernels, got {mask.shape}")
    spec = np.fft.fft2(np.asarray(mask, dtype=np.complex128))
    fields = np.fft.ifft2(kernels.kernels * spec[None], axes=(-2, -1))
    intensity = np.tensordot(
        kernels.weights, fields.real ** 2 + fields.imag ** 2, axes=1
    )
    return fields, intensity


def socs_gradient(
    kernels: SOCSKernels,
    fields: NDArray[np.complex128],
    sensitivity: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``∂/∂m`` of ``Σ_x sensitivity(x)·I(x)``, for the raw intensity.

    The adjoint sweep: one transform pair per kernel, conjugating the kernel
    the forward pass multiplied by. Real on return because the mask is real —
    the imaginary part is the component of the gradient that would take the
    mask off the real axis, and dropping it is the projection onto the space
    the mask actually lives in, not an approximation.

    Parameters
    ----------
    kernels : SOCSKernels
        The same decomposition :func:`socs_fields` used.
    fields : NDArray
        ``(K, n, n)`` coherent fields from :func:`socs_fields`.
    sensitivity : NDArray
        ``∂J/∂I``, real, ``(n, n)``.

    Returns
    -------
    NDArray[np.float64]
        ``(n, n)`` real gradient with respect to the mask.
    """
    g = np.asarray(sensitivity, dtype=np.float64)
    back = np.fft.fft2(g[None] * fields)
    adj = np.fft.ifft2(np.conj(kernels.kernels) * back, axes=(-2, -1))
    weighted = np.tensordot(kernels.weights, adj, axes=1)
    return 2.0 * np.real(weighted)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def _blur(a: NDArray[np.float64], sigma_px: float) -> NDArray[np.float64]:
    """The PEB blur, unclipped.

    :func:`litho_sim.bake.peb.apply_peb` clips its result to ``[0, 1]``, which
    is right for a PAC concentration and wrong for an aerial image, whose
    bright regions routinely exceed 1 under clear-field normalisation. Clipping
    is also flat, so it would zero the gradient exactly where the solve needs
    it most. This is the same Gaussian without that step.

    A Gaussian blur is linear and its kernel is symmetric, so this function is
    its own adjoint — which is why the chain rule below can simply call it
    again on the way back.
    """
    if sigma_px < 1e-3:
        return a
    return gaussian_filter(a, sigma=sigma_px)


@dataclass
class _Objective:
    """The cost, its gradient, and the state a viewer wants — in one pass.

    Bundled rather than split into ``cost()`` and ``grad()`` because they share
    the forward image, and computing it twice would double the cost of the
    solve for no gain.
    """

    kernels: SOCSKernels
    target: NDArray[np.float64]
    scale: float
    level: float
    sigma_px: float
    sign: float
    steepness: float
    weight_tv: float
    weight_discrete: float

    def __call__(
        self, mask: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64], int]:
        fields, raw = socs_fields(self.kernels, mask)
        aerial = raw * self.scale
        fld = _blur(aerial, self.sigma_px)
        signed = self.sign * (fld - self.level)

        # A relaxed resist. The engine's own criterion is a hard crossing of
        # `level`, which has no useful derivative; a logistic of the same
        # signed field agrees with it wherever the contour is and is smooth
        # everywhere else. `steepness` is how sharply the two are made to
        # agree, and it is the one knob that trades the solve's stability
        # against how literally it believes the resist edge.
        a = self.steepness / max(abs(self.level), 1e-12)
        z = 1.0 / (1.0 + np.exp(-np.clip(a * signed, -60.0, 60.0)))

        resid = z - self.target
        npix = float(resid.size)
        cost = float(np.sum(resid ** 2) / npix)

        # dJ/dz -> dz/dsigned -> dsigned/dfield -> dfield/dI (blur is
        # self-adjoint) -> dI/dm (the SOCS adjoint).
        dz = (2.0 / npix) * resid * (a * z * (1.0 - z))
        d_field = self.sign * dz
        d_intensity = _blur(d_field, self.sigma_px) * self.scale
        grad = socs_gradient(self.kernels, fields, d_intensity)

        if self.weight_tv > 0.0:
            # Mask complexity, as the squared gradient of the mask. Its own
            # derivative is the Laplacian, which is what stops the solve
            # answering with single-pixel speckle that images as nothing.
            lap = (
                np.roll(mask, 1, 0) + np.roll(mask, -1, 0)
                + np.roll(mask, 1, 1) + np.roll(mask, -1, 1)
                - 4.0 * mask
            )
            gx = mask - np.roll(mask, 1, 1)
            gy = mask - np.roll(mask, 1, 0)
            cost += self.weight_tv * float(np.sum(gx ** 2 + gy ** 2) / npix)
            grad += self.weight_tv * (-2.0 / npix) * lap

        if self.weight_discrete > 0.0:
            # Pushes the mask off the fence toward 0 or 1, so that the
            # thresholded mask in `ILTResult.binary` costs less to take.
            cost += self.weight_discrete * float(np.sum(mask * (1.0 - mask)) / npix)
            grad += self.weight_discrete * (1.0 - 2.0 * mask) / npix

        pattern_error = int(np.count_nonzero((signed > 0.0) != (self.target > 0.5)))
        return cost, grad, signed, pattern_error


# ---------------------------------------------------------------------------
# The parameterisation
# ---------------------------------------------------------------------------


def _mask_of(
    theta: NDArray[np.float64], beta: float, smooth_px: float
) -> NDArray[np.float64]:
    """``σ(β · G(θ))`` — the mask a latent field describes.

    The Gaussian is what bounds the outline's curvature; the sigmoid is what
    holds the mask in ``[0, 1]`` without clipping, so the gradient stays alive
    at the bounds.
    """
    field = _blur(theta, smooth_px)
    return 1.0 / (1.0 + np.exp(-np.clip(beta * field, -60.0, 60.0)))


def _grad_theta(
    grad_mask: NDArray[np.float64],
    mask: NDArray[np.float64],
    beta: float,
    smooth_px: float,
) -> NDArray[np.float64]:
    """``dJ/dθ`` from ``dJ/dm``: back through the sigmoid's slope, then the
    Gaussian, which is its own adjoint."""
    return _blur(grad_mask * (beta * mask * (1.0 - mask)), smooth_px)


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def run_ilt(
    target: NDArray[np.float64],
    model: PrintModel,
    *,
    initial: NDArray[np.float64] | None = None,
    max_iter: int = 120,
    step: float = 0.4,
    tol: float = 1e-7,
    beta: float = 4.0,
    beta_final: float | None = 12.0,
    initial_grey: float = 0.9,
    steepness: float = 12.0,
    weight_tv: float = 0.0,
    weight_discrete: float = 0.0,
    smooth: float = 0.0,
    progress: Callable[[ILTFrame], None] | None = None,
) -> ILTResult:
    """Solve for the mask that prints *target*.

    Adam on an unconstrained latent ``theta``, with the mask itself held in
    ``[0, 1]`` by ``mask = σ(β·G(theta))``, ``G`` a Gaussian of width *smooth*.
    Constraining by parameterisation rather than by clipping keeps the
    gradient alive at the bounds — a clipped pixel that wants to leave its
    bound is stuck there, and a mask full of stuck pixels stops improving long
    before it is right.

    Parameters
    ----------
    target : NDArray
        The design, ``(n, n)``, 1 inside the features that should print. This
        is what the wafer should look like, *not* a mask — rasterise the drawn
        layout with :meth:`~litho_sim.mask.layout.Layout.rasterize` and pass
        that.
    model : PrintModel
        The process. Must image through the SOCS kernels — the adjoint above
        is written against that decomposition, so a vector or thick-mask
        process is refused rather than silently approximated.
    initial : NDArray, optional
        Starting mask in ``[0, 1]``. Defaults to the target, which is the
        sensible start: OPC is a correction, and the drawn shape is the
        zeroth-order answer.
    max_iter : int
        Iteration cap. Each iteration is about three prints' work.
    step : float
        Adam's learning rate, in ``theta``.
    tol : float
        Stop once the cost improves by less than this over a whole iteration.
    beta, beta_final : float
        Mask sharpening, annealed geometrically from *beta* to *beta_final*
        across the solve. A low ``β`` early lets grey mask move freely and
        find the right topology — where the assist features want to be — and
        raising it late drives the result toward the two levels a writer can
        actually make. ``beta_final=None`` holds ``β`` fixed.
    initial_grey : float
        How dark the starting mask's bright pixels are, in ``(0.5, 1)``. Not
        cosmetic: the step is taken in ``theta``, where the gradient carries a
        factor ``m(1−m)``, so a mask that starts at 0 and 1 exactly starts with
        no gradient anywhere and the solve reports itself converged without
        having moved. 0.9 leaves the mask room to breathe; the anneal sharpens
        it back up.
    steepness : float
        Sharpness of the relaxed resist, relative to the print level.
    weight_tv : float
        Penalty on mask complexity. 0 leaves the solve free to answer with
        whatever it likes, speckle included; a small value (~1e-3) is what
        makes the result look like a mask.
    weight_discrete : float
        Penalty on grey mask pixels.
    smooth : float
        Curvature limit [m]: the width of the Gaussian the mask is drawn
        through. Nothing in the mask is finer than this or turns on a smaller
        radius, so the outline is a curve by construction rather than a
        staircase. 0 leaves every pixel free. A quarter of the feature size is
        a sensible start; it costs the continuous solve almost nothing and
        the thresholded mask gains.
    progress : callable, optional
        Called with an :class:`ILTFrame` before the first step and after every
        one thereafter. Synchronous and on this thread — see the module
        docstring.

    Returns
    -------
    ILTResult

    Raises
    ------
    ValueError
        If *model* cannot image through SOCS kernels, if *target* is not
        square and matching the model's grid, if *smooth* is negative, or if
        the model normalises on the image peak — under peak normalisation the dose means something
        different for every mask the solve tries, so the objective would move
        under its own feet. Use ``normalisation="clear"``.
    """
    kernels = model.kernels()
    if kernels is None:
        raise ValueError(
            "inverse lithography images through the SOCS kernels; this model "
            f"images through the Abbe sum (imaging={model.imaging!r}, "
            f"imaging_model={model.optics.imaging_model!r}, "
            f"mask_model={model.optics.mask_model!r}). The adjoint is written "
            "against the kernel decomposition and has no thick-mask form."
        )
    if model.optics.normalisation.lower() == "peak":
        raise ValueError(
            "inverse lithography needs a mask-independent normalisation; "
            "'peak' rescales the image by whatever the current mask's brightest "
            "point happens to be, which moves the objective every step. "
            "Use normalisation='clear'."
        )

    n = model.grid.n_pixels
    tgt = np.asarray(target, dtype=np.float64)
    if tgt.shape != (n, n):
        raise ValueError(f"target must be {(n, n)} for this grid, got {tgt.shape}")
    if smooth < 0.0:
        raise ValueError(f"smooth is a length and cannot be negative, got {smooth:g}")
    smooth_px = float(smooth / model.grid.pixel_size)

    scale = normalisation_scale(
        model.optics.normalisation, 1.0, kernels.clear, model.dose
    )
    level = float(model.resist.threshold)
    sigma_px = float(model.resist.diffusion_sigma / model.grid.pixel_size)
    sign = 1.0 if model.tone == "clear" else -1.0

    objective = _Objective(
        kernels=kernels, target=tgt, scale=scale, level=level, sigma_px=sigma_px,
        sign=sign, steepness=steepness, weight_tv=weight_tv,
        weight_discrete=weight_discrete,
    )

    p = float(np.clip(initial_grey, 0.5 + 1e-3, 1.0 - 1e-6))
    if initial is None:
        m0 = 0.5 + (np.clip(tgt, 0.0, 1.0) - 0.5) * (2.0 * p - 1.0)
    else:
        m0 = np.clip(np.asarray(initial, dtype=np.float64), 1.0 - p, p)
    theta = np.log(m0 / (1.0 - m0)) / beta

    growth = (
        1.0 if beta_final is None or max_iter <= 1
        else (beta_final / beta) ** (1.0 / (max_iter - 1))
    )

    # Adam. The moments live in theta, so annealing beta rescales the
    # objective under them; that is deliberate and mild, and resetting them
    # on every anneal step costs more convergence than it buys.
    mom1 = np.zeros_like(theta)
    mom2 = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    rows: list[dict] = []
    beta_k = float(beta)
    prev_cost = np.inf
    converged = False
    taken = 0

    mask = _mask_of(theta, beta_k, smooth_px)
    cost, grad_m, signed, perr = objective(mask)

    # A print that is entirely dark or entirely clear has no contour, and a
    # relaxed resist saturated everywhere has no gradient either — so the solve
    # would take a few zero-sized steps and report itself converged, which
    # looks like success and is not. Nearly always the dose: the drawn mask has
    # to roughly print before a correction to it means anything. Caught here
    # rather than left to the caller because the symptom points nowhere near
    # the cause.
    lit = int(np.count_nonzero(signed > 0.0))
    if lit == 0 or lit == signed.size:
        peak = float(np.max(np.abs(signed) + level))
        raise ValueError(
            f"the initial mask prints {'nothing' if lit == 0 else 'everywhere'} "
            f"— no contour exists, so the objective has no gradient. The resist "
            f"level is {level:g} and the field reaches {peak:.4g} at dose "
            f"{model.dose:g}. Size the dose first, e.g. "
            f"model.dose = model.dose_to_size(layout, target_cd_nm) or "
            f"litho_sim.opc.ilt.dose_for_target(target, model)."
        )

    if progress is not None:
        progress(ILTFrame(0, mask.copy(), signed.copy(), cost, perr, beta_k))
    rows.append(
        {"iteration": 0, "cost": cost, "pattern_error": perr,
         "beta": beta_k, "grad_norm": float(np.linalg.norm(grad_m))}
    )

    for it in range(1, max_iter + 1):
        # Back through the parameterisation: the sigmoid's slope, then the
        # Gaussian — its own adjoint, so the same blur the forward pass took.
        grad_t = _grad_theta(grad_m, mask, beta_k, smooth_px)

        mom1 = b1 * mom1 + (1.0 - b1) * grad_t
        mom2 = b2 * mom2 + (1.0 - b2) * grad_t ** 2
        m_hat = mom1 / (1.0 - b1 ** it)
        v_hat = mom2 / (1.0 - b2 ** it)
        theta = theta - step * m_hat / (np.sqrt(v_hat) + eps)

        beta_k *= growth
        mask = _mask_of(theta, beta_k, smooth_px)
        cost, grad_m, signed, perr = objective(mask)
        taken = it

        rows.append(
            {"iteration": it, "cost": cost, "pattern_error": perr,
             "beta": beta_k, "grad_norm": float(np.linalg.norm(grad_m))}
        )
        if progress is not None:
            progress(ILTFrame(it, mask.copy(), signed.copy(), cost, perr, beta_k))

        if abs(prev_cost - cost) < tol:
            converged = True
            logger.debug("run_ilt: cost settled after %d iterations", it)
            break
        prev_cost = cost

    if not converged:
        logger.info(
            "run_ilt: %d iterations without settling (cost %.4g, pattern error %d)",
            taken, cost, perr,
        )

    printed = print_mask(mask, model)
    return ILTResult(
        mask=mask, theta=theta, target=tgt > 0.5, printed=printed,
        history=pd.DataFrame(rows), iterations=taken, converged=converged,
    )


def print_mask(mask: NDArray[np.float64], model: PrintModel) -> PrintedImage:
    """Image a mask *array* through a print model.

    :meth:`~litho_sim.opc.model.PrintModel.print` takes a
    :class:`~litho_sim.mask.layout.Layout` and rasterises it. An inverse
    lithography mask never was a layout, so this is the same print from the
    array inward — and it is what makes an ILT result comparable to an
    edge-based one, since both end in the same
    :class:`~litho_sim.opc.model.PrintedImage`.
    """
    from litho_sim.develop.resist import develop_field
    from litho_sim.expose.aerial_image import compute_aerial_image

    kernels = model.kernels()
    if kernels is not None:
        raw, peak, clear = kernels.raw(mask)
    else:
        raw, peak, clear = compute_aerial_image(
            mask, model.optics, model.grid, return_raw=True
        )
    aerial = raw * normalisation_scale(
        model.optics.normalisation, peak, clear, model.dose
    )
    fld, level = develop_field(aerial, model.resist, model.grid, model=model.model)
    return PrintedImage(mask, aerial, fld, level, model.tone, model.grid)


def dose_for_target(
    target: NDArray[np.float64],
    model: PrintModel,
    mask: NDArray[np.float64] | None = None,
) -> float:
    """The dose at which *mask* prints the same **area** as *target*.

    :meth:`~litho_sim.opc.model.PrintModel.dose_to_size` sizes a CD along a
    cut, which needs a feature the cut crosses and an axis to cross it on. An
    inverse lithography target is an arbitrary 2-D field — a contact array, a
    corner, a piece of real layout — and often has neither. Matching printed
    area instead asks the same question without needing a cut, and it is the
    right anchor for a solve whose whole cost is an area disagreement.

    Cheap enough to call freely: the raw image is computed once, and the whole
    dose dependence is the scalar in front of it, so the search is arithmetic
    on an array that never gets re-imaged.

    Parameters
    ----------
    target : NDArray
        The design, ``(n, n)``, non-zero inside the features.
    model : PrintModel
        The process. Its ``dose`` is read only as a fallback and is not
        modified — assign the result yourself.
    mask : NDArray, optional
        The mask to size on. Defaults to *target*, i.e. the drawn shape.

    Returns
    -------
    float
        The dose. Returns ``model.dose`` unchanged if the target is empty or
        full, where matching areas says nothing.
    """
    n = model.grid.n_pixels
    tgt = np.asarray(target, dtype=np.float64)
    m = tgt if mask is None else np.asarray(mask, dtype=np.float64)
    if m.shape != (n, n):
        raise ValueError(f"mask must be {(n, n)} for this grid, got {m.shape}")

    want = int(np.count_nonzero(tgt > 0.5))
    if want == 0 or want == tgt.size:
        return float(model.dose)

    kernels = model.kernels()
    if kernels is None:
        raise ValueError("dose_for_target needs a model that images through SOCS kernels")

    raw, _peak, clear = kernels.raw(m)
    sigma_px = float(model.resist.diffusion_sigma / model.grid.pixel_size)
    # field is linear in dose, so one blur of the unit-dose image is the whole
    # dose dependence; everything below is a comparison against a scalar.
    unit = _blur(raw * normalisation_scale(model.optics.normalisation, 1.0, clear, 1.0),
                 sigma_px)
    level = float(model.resist.threshold)
    sign = 1.0 if model.tone == "clear" else -1.0

    def area(dose: float) -> int:
        return int(np.count_nonzero(sign * (dose * unit - level) > 0.0))

    lo, hi = 1e-3, 1e3
    # Printed area is monotone in dose — rising for a clear-tone mask, falling
    # for a dark one — so a bisection is well posed either way.
    rising = area(hi) > area(lo)
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if (area(mid) < want) == rising:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


__all__ = [
    "ILTFrame", "ILTResult", "dose_for_target", "print_mask", "run_ilt",
    "socs_fields", "socs_gradient",
]
