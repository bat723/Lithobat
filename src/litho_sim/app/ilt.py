"""
The inverse-lithography computation behind the app's ILT tab.

Qt-free, like ``fem`` and ``stochastics``: the tab hands an :class:`ILTRequest`
to the worker thread, :func:`compute_ilt` runs the solve, and every iteration
comes back through a callback while it runs. Tests drive this module headless.

What makes this tab different from every other one is that the *intermediate*
states are the point. A process window is worth showing once it exists; an ILT
solve is worth watching, because what it does on the way — the mask going grey
and re-forming, assist features nucleating in empty field where nothing was
drawn — is the thing that explains what inverse lithography is. So the
progress callback here carries a whole frame rather than a percentage, and the
tab draws every one.

The budget that makes that honest: one iteration is about three aerial images
— the forward print, and one transform pair per kernel again for the adjoint.
What that costs is set by the *kernel count* as much as by the pixel count,
and the kernel count follows the field: a wider field samples the pupil more
finely and keeps more of the source. On the app's default 128-pixel, 512 nm
field it is 61 kernels and about 70 ms an iteration (about 45 ms with numpy
held to one thread), so around 15 frames a second. On a 256-pixel, 2048 nm
field — where the isolated designs live — it is 150 to 200 kernels and about
0.65 s an iteration: a slideshow, not an animation. The tab reports what it
actually measured rather than promising a rate.

Frames are dropped, not queued. The solver runs on the worker thread and emits
into the GUI thread's event loop; if the GUI cannot keep up, holding every
frame would grow an unbounded backlog and the animation would lag further and
further behind a solve that has already finished. :class:`ILTRequest` carries
``min_frame_interval`` for that, and the last frame is always delivered.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from litho_sim.app.opc import AppPrintModel, app_layout, mask_from_layout, print_model
from litho_sim.app.params import ParameterModel
from litho_sim.expose.hopkins import SOCSKernels
from litho_sim.opc.ilt import ILTFrame, ILTResult, dose_for_target, print_mask, run_ilt

logger = logging.getLogger(__name__)


@dataclass
class ILTRequest:
    """Everything one press of Run asks for.

    Attributes
    ----------
    params : ParameterModel
        A snapshot of the dock, deep-copied at Run-press time so the worker
        never races the sliders.
    max_iter : int
        Iteration cap.
    step : float
        Adam's learning rate in the latent variable. The default is the one
        that works across the presets; an order of magnitude larger overshoots
        into an all-dark mask on the first step, which the solve then reports
        as convergence.
    beta, beta_final : float
        Mask sharpening, annealed across the solve.
    steepness : float
        Sharpness of the relaxed resist.
    weight_tv : float
        Mask-complexity penalty — what keeps the answer looking like a mask.
    smooth_nm : float
        Curvature limit [nm]: the mask is drawn through a Gaussian this wide,
        so its outline cannot turn on a smaller radius. What makes the answer
        a curve rather than a staircase with bumps on it once sharpened; 0
        leaves every pixel free.
    size_dose : bool
        Size the dose to the target's own area before solving. Almost always
        wanted: at the engine's nominal dose a contact array prints nothing at
        all, and a solve against a blank print has no gradient to follow.
    min_frame_interval : float
        Seconds between streamed frames [s]. 0 streams every iteration.
    """

    params: ParameterModel
    max_iter: int = 120
    step: float = 0.03
    beta: float = 4.0
    beta_final: float = 16.0
    steepness: float = 25.0
    weight_tv: float = 1e-3
    smooth_nm: float = 24.0
    size_dose: bool = True
    min_frame_interval: float = 0.0


@dataclass
class ILTResultBundle:
    """A finished solve, with the before-and-after the tab reports.

    Attributes
    ----------
    result : ILTResult
        The solve itself — mask, history, final print.
    target : NDArray
        The design it aimed at.
    dose : float
        The dose it solved at, after sizing.
    pattern_error_before : int
        Pixels wrong printing the *drawn* mask, the baseline any correction
        has to beat.
    pattern_error_after : int
        The same count for the continuous ILT mask.
    pattern_error_binary : int
        And for the thresholded mask at its own re-sized dose — the number
        that corresponds to a mask someone could actually write.
    dose_binary : float
        That re-sized dose. Thresholding moves the mask's total transmission
        and with it every printed edge, so the binary mask does not print
        properly at the dose the continuous one was solved at.
    seconds : float
        Wall-clock for the solve.
    """

    result: ILTResult
    target: NDArray[np.float64]
    dose: float
    pattern_error_before: int
    pattern_error_after: int
    pattern_error_binary: int
    dose_binary: float
    seconds: float
    history: object = field(default=None, repr=False)


def ilt_print_model(params: ParameterModel) -> AppPrintModel:
    """The app's print model, normalised the way a solve needs.

    :func:`~litho_sim.app.opc.print_model` deliberately keeps the user's
    normalisation so the OPC loop prints exactly what Print prints; under peak
    normalisation each move re-scales the dose a little and the adaptive gain
    absorbs it.

    A gradient cannot absorb it. Peak normalisation divides the image by
    whatever the current mask's brightest point happens to be, so the dose —
    and with it the level the contour sits at — is a function of the very
    variable being optimised. The objective would move under its own feet
    every step, and the ``∂I/∂m`` above does not account for it. So this
    forces ``"clear"``, which is mask-independent, and the tab says so rather
    than quietly printing something Print would not.
    """
    model = print_model(params)
    if model.optics.normalisation.lower() == "clear":
        return model
    return dataclasses.replace(model, normalisation="clear")


def images_through_kernels(model: AppPrintModel) -> bool:
    """Whether *model* will image through the SOCS kernels — without building them.

    The two tests :meth:`~litho_sim.opc.model.PrintModel.kernels` makes before
    it decomposes anything. Asked on every dock change, so it must not cost a
    decomposition: at 256 pixels that is seconds, on the GUI thread.
    """
    if model.imaging == "abbe":
        return False
    return model.imaging != "auto" or SOCSKernels.supports(model.optics)


def ilt_availability(params: ParameterModel) -> str | None:
    """Why inverse lithography cannot run on this configuration, or ``None``.

    Mirrors :func:`~litho_sim.app.opc.opc_availability`: the tab asks before
    enabling Run, so the reason is a sentence in the panel rather than a
    traceback after a click. Cheap, because the main window asks again on
    every control that moves.
    """
    try:
        model = ilt_print_model(params)
    except ValueError as exc:
        return str(exc)
    if not images_through_kernels(model):
        return (
            "Inverse lithography images through the SOCS kernels, which need "
            "the scalar thin-mask process. Set the imaging model to scalar "
            "and the mask model to thin."
        )
    return None


def normalisation_note(params: ParameterModel) -> str | None:
    """A line for the panel when the solve will not print the way Print does."""
    if str(params.optics().normalisation).lower() == "clear":
        return None
    return (
        "Solving under clear-field normalisation — the dock is set to "
        f"'{params.optics().normalisation}', which rescales the dose by the "
        "current mask and would move the objective every step."
    )


def target_for(params: ParameterModel, model: AppPrintModel) -> NDArray[np.float64]:
    """The design the solve aims at, as a binary field on the imaging grid.

    The same vector rasterisation the OPC tab corrects against
    (:func:`~litho_sim.app.opc.mask_from_layout`), so a correction computed
    here and one computed there are aiming at the same edges — the app's one
    definition of the drawn shape.

    Binary regardless of the dock's mask type: an attenuated PSM is a way of
    *making* the image, not a different design, and the solve is aiming at
    where the resist edge should land.
    """
    layout = app_layout(params)
    drawn = mask_from_layout(layout, model.grid, "binary", oversample=model.oversample)
    return (np.real(np.asarray(drawn)) > 0.5).astype(np.float64)


def compute_ilt(
    req: ILTRequest,
    progress: Callable[[ILTFrame], None] | None = None,
) -> ILTResultBundle:
    """Run one solve, streaming frames as it goes.

    Parameters
    ----------
    req : ILTRequest
        The snapshot to solve.
    progress : callable, optional
        Called with each :class:`~litho_sim.opc.ilt.ILTFrame` the solver
        produces, thinned to ``req.min_frame_interval``. The first and last
        frames are always delivered — the first because it is the uncorrected
        print the animation opens on, the last because it is the answer.

    Returns
    -------
    ILTResultBundle
    """
    reason = ilt_availability(req.params)
    if reason is not None:
        raise ValueError(reason)

    model = ilt_print_model(req.params)
    target = target_for(req.params, model)

    if req.size_dose:
        model = dataclasses.replace(model, dose=dose_for_target(target, model))

    before = int(np.count_nonzero(
        (print_mask(target, model).signed > 0.0) != (target > 0.5)
    ))

    last_emit = [0.0]

    def relay(frame: ILTFrame) -> None:
        if progress is None:
            return
        now = time.perf_counter()
        first = frame.iteration == 0
        due = (now - last_emit[0]) >= req.min_frame_interval
        if first or due:
            last_emit[0] = now
            progress(frame)

    t0 = time.perf_counter()
    result = run_ilt(
        target, model,
        max_iter=req.max_iter, step=req.step,
        beta=req.beta, beta_final=req.beta_final,
        steepness=req.steepness, weight_tv=req.weight_tv,
        smooth=req.smooth_nm * 1e-9,
        progress=relay,
    )
    seconds = time.perf_counter() - t0

    # The solver's own last frame may have been thinned away; the answer never
    # is. Re-emitting it costs nothing and spares the tab from showing a state
    # the solve had already moved past.
    if progress is not None:
        h = result.history.iloc[-1]
        progress(ILTFrame(
            int(h.iteration), result.mask.copy(), result.printed.signed.copy(),
            float(h.cost), int(h.pattern_error), float(h.beta),
        ))

    binary = result.binary()
    dose_binary = dose_for_target(target, model, mask=binary)
    printed_binary = print_mask(binary, dataclasses.replace(model, dose=dose_binary))
    after_binary = int(np.count_nonzero(
        (printed_binary.signed > 0.0) != (target > 0.5)
    ))

    logger.info(
        "ILT: %d iterations in %.2f s (%.0f ms/iter), pattern error %d -> %d "
        "(%d thresholded)",
        result.iterations, seconds,
        1e3 * seconds / max(result.iterations, 1),
        before, int(result.history.iloc[-1].pattern_error), after_binary,
    )

    return ILTResultBundle(
        result=result,
        target=target,
        dose=float(model.dose),
        pattern_error_before=before,
        pattern_error_after=int(result.history.iloc[-1].pattern_error),
        pattern_error_binary=after_binary,
        dose_binary=float(dose_binary),
        seconds=seconds,
        history=result.history,
    )


def estimate_ilt_cost_ms(params: ParameterModel, image_ms: float) -> float:
    """Roughly what one iteration will cost [ms], for the tab's frame-rate note.

    One iteration is a forward image plus its adjoint, and the adjoint is one
    transform pair per kernel against the forward's one — so about three
    images' work, which is what the measured 66 ms on a 128-pixel field against
    a 22 ms image says.
    """
    return 3.0 * float(image_ms)


__all__ = [
    "ILTRequest", "ILTResultBundle", "compute_ilt", "estimate_ilt_cost_ms",
    "ilt_availability", "ilt_print_model", "images_through_kernels",
    "normalisation_note", "target_for",
]
