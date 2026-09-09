"""
Optical proximity correction in the app: the drawn pattern as geometry, a
print model that images it exactly as Print does, and the correction.

The app's patterns are pixel arrays (:mod:`litho_sim.mask.patterns`); OPC
moves the edges of polygons (:mod:`litho_sim.opc`). This module is the
bridge. :func:`app_layout` draws every pattern the Mask tab offers as a
:class:`~litho_sim.mask.layout.Layout`, placed where the pixel builders
place it; :func:`mask_from_layout` rasterises a layout back into the array
the imaging consumes, with the same att-PSM conversion the pixel path
applies; :class:`AppPrintModel` is the engine's print model with that
rasteriser, so the mask the loop corrects against is the mask Print images.

Two facts shape the geometry. The imaging is periodic over the field, so a
line drawn through the whole field has no ends — but a polygon does, and
the loop would read its window edges as line ends and pull them in. Shapes
that run out of the field are therefore drawn *past* it, and the loop is
told the field (``clip``) so the fragments outside it are held. And the
pixel builders sample pixel centres, which lands their edges half a pixel
from where the geometry's are; the vector raster is the geometry. The two
agree to a pixel, and the OPC path uses the vector raster throughout — for
the design it corrects, and for the corrected mask Print then images.

Qt-free, like the rest of the app's core.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from litho_sim.app.params import ParameterModel
from litho_sim.core.config import GridConfig
from litho_sim.expose.hopkins import SOCSKernels
from litho_sim.mask import Contact, Layout, Polygon, Rect
from litho_sim.opc import (
    OPCResult,
    PrintModel,
    add_scattering_bars,
    assist_features_printed,
    default_fragment_length,
    run_opc,
)

#: The layer the design is drawn on; scattering bars go on ``"sraf"``.
LAYER = "main"

#: Patterns drawn to their own scale rather than the field's: isolated
#: designs with empty field around them, the case inverse lithography is
#: usually shown on. See :func:`field_fit` for what "empty field" costs.
ISOLATED = ("frame and bars", "isolated contacts")

#: How much of the field's width or height an isolated design may take up.
#: The imaging wraps at the field edge, so a design that fills the field meets
#: its own copy through the seam and prints as a dense array of itself.
_FILL = 0.8

#: Amplitude of an attenuated-PSM film: 6 % intensity at 180°. The same
#: constants :func:`~litho_sim.mask.patterns.to_attenuated_psm` uses.
_ATTENUATED = math.sqrt(0.06) * np.exp(1j * math.pi)


def field_box(grid: GridConfig) -> tuple[float, float, float, float]:
    """The imaged field ``(x0, y0, x1, y1)`` [m], at the pixel *edges*.

    Pixel centres sit at ``(i − n//2) · px``; the field runs half a pixel
    past the first and last of them, so a shape drawn to this box covers
    every raster sample and a control point inside it is genuinely imaged.
    """
    n, px = grid.n_pixels, grid.pixel_size
    lo = -(n // 2) * px - 0.5 * px
    hi = (n - n // 2) * px - 0.5 * px
    return (lo, lo, hi, hi)


def _periodic_centres(pitch: float, cd: float, lo: float, hi: float) -> list[float]:
    """Feature centres of the pixel builders' periodic patterns within ``[lo, hi]``.

    :func:`~litho_sim.mask.patterns.lines_and_spaces` opens a feature where
    ``(x + pitch/2) mod pitch < cd``, which puts the centres at
    ``k · pitch − (pitch − cd) / 2``; :func:`contact_array` uses the same
    rule on both axes.
    """
    c0 = -0.5 * (pitch - cd)
    k_lo = math.floor((lo - c0) / pitch)
    k_hi = math.ceil((hi - c0) / pitch)
    return [c0 + k * pitch for k in range(k_lo, k_hi + 1) if lo <= c0 + k * pitch <= hi]


def _frame_and_bars(cd: float) -> list:
    """A frame with three bars inside, drawn in units of the CD.

    The design inverse lithography is usually shown on: isolated, with empty
    field around it where the solve nucleates assist rings nobody drew. The
    frame is *one* polygon, broken at the bottom by a gap of two CDs — a
    closed ring would need a hole, and a frame of four abutting rectangles
    would hand the edge loop shared edges to correct as two facing ones (see
    :func:`~litho_sim.opc.correct.run_opc`). The break is a tip-to-tip pair in
    its own right. Counter-clockwise, as :class:`~litho_sim.mask.Rect` draws.
    """
    c = cd
    frame = Polygon(LAYER, (
        (c, -5 * c), (6.5 * c, -5 * c), (6.5 * c, 5 * c), (-6.5 * c, 5 * c),
        (-6.5 * c, -5 * c), (-c, -5 * c), (-c, -4 * c), (-5.5 * c, -4 * c),
        (-5.5 * c, 4 * c), (5.5 * c, 4 * c), (5.5 * c, -4 * c), (c, -4 * c),
    ))
    bars = [
        Rect(LAYER, -2.4 * c, 0.0, 1.2 * c, 4.0 * c),
        Rect(LAYER, 0.0, 0.0, 0.6 * c, 4.0 * c),
        Rect(LAYER, 2.4 * c, 0.0, 1.2 * c, 4.0 * c),
    ]
    return [frame, *bars]


def _isolated_contacts(pitch: float, cd: float) -> list:
    """A 2 × 2 group of contacts at the drawn pitch, and a lone one beside it."""
    group = [
        Contact(LAYER, x, y, cd, "square")
        for x in (-1.5 * pitch, -0.5 * pitch) for y in (-0.5 * pitch, 0.5 * pitch)
    ]
    return [*group, Contact(LAYER, 1.5 * pitch, 0.0, cd, "square")]


def design_extent(params: ParameterModel) -> tuple[float, float] | None:
    """Width and height [m] of an isolated design; ``None`` for a periodic one."""
    pitch, cd = float(params.si("pitch")), float(params.si("cd"))
    pattern = params["pattern"]
    if pattern == "frame and bars":
        return 13.0 * cd, 10.0 * cd
    if pattern == "isolated contacts":
        return 3.0 * pitch + cd, pitch + cd
    return None


def field_fit(params: ParameterModel) -> str | None:
    """Why an isolated design does not fit the field, or ``None`` if it does.

    An isolated design is only isolated with empty field around it: it may
    take up :data:`_FILL` of the field's width and height, and beyond that the
    correction pages refuse to run — and say which grid to set — rather than
    correct a design that is imaging its own wrapped copy. Print itself never
    refuses; it draws whatever fits, so the Mask tab always shows something.
    """
    extent = design_extent(params)
    if extent is None:
        return None
    w, h = extent
    grid = params.grid()
    field = grid.n_pixels * grid.pixel_size
    need = max(w, h) / _FILL
    if need <= field:
        return None
    # The nearest setting of the dock's own steps that would do — 8 nm pixels
    # first, since that keeps a 100 nm feature a dozen pixels wide.
    n = int(math.ceil(need / 8e-9 / 32.0)) * 32
    if n <= 256:
        what = f"Grid {n} and Pixel size 8 nm ({n * 8} nm field)"
    else:
        px = math.ceil(need / 256 / 0.5e-9) * 0.5
        what = f"Grid 256 and Pixel size {px:g} nm ({256 * px:g} nm field)"
    return (
        f"'{params['pattern']}' is {w * 1e9:.0f} × {h * 1e9:.0f} nm at CD "
        f"{params['cd']:g} nm and needs empty field around it, but the field is "
        f"{field * 1e9:.0f} nm. Set {what}, or lower the CD."
    )


def app_layout(params: ParameterModel) -> Layout:
    """The drawn pattern as vector geometry, on layer ``"main"``.

    Every pattern the Mask tab offers, placed where :func:`build_mask`
    draws it. Lines run a quarter-field past each edge of the field so the
    loop, told the field, holds their window ends; *line ends* is the
    dense array with every line broken by a gap of one drawn CD, a quarter
    of the way up the field — the tip-to-tip test, and the geometry
    proximity correction exists for. The gap sits off the centre row so
    the row the CD is read on still crosses the lines. The two isolated
    designs (:data:`ISOLATED`) are drawn to their own scale, centred, and
    simply clipped by a field too small for them; :func:`field_fit` is
    what says so.
    """
    grid = params.grid()
    x0, y0, x1, y1 = field_box(grid)
    pitch, cd = float(params.si("pitch")), float(params.si("cd"))
    pattern = params["pattern"]
    overhang = 0.25 * (y1 - y0)
    tall = (y1 - y0) + 2.0 * overhang
    shapes: list = []

    if pattern in ("lines and spaces", "line ends"):
        for x in _periodic_centres(pitch, cd, x0 - 0.5 * cd, x1 + 0.5 * cd):
            if pattern == "lines and spaces":
                shapes.append(Rect(LAYER, x, 0.0, cd, tall))
            else:
                gap_at = 0.25 * (y1 - y0)
                top, bottom = y1 + overhang, y0 - overhang
                upper = top - (gap_at + 0.5 * cd)
                lower = (gap_at - 0.5 * cd) - bottom
                shapes.append(Rect(LAYER, x, top - 0.5 * upper, cd, upper))
                shapes.append(Rect(LAYER, x, bottom + 0.5 * lower, cd, lower))
    elif pattern == "isolated line":
        shapes.append(Rect(LAYER, 0.0, 0.0, cd, tall))
    elif pattern == "contacts":
        xs = _periodic_centres(pitch, cd, x0 - 0.5 * cd, x1 + 0.5 * cd)
        ys = _periodic_centres(pitch, cd, y0 - 0.5 * cd, y1 + 0.5 * cd)
        shapes = [Contact(LAYER, x, y, cd, "square") for y in ys for x in xs]
    elif pattern == "checkerboard":
        # patterns.checkerboard lays its cells from the field's first pixel
        # centre, not from the field centre, and opens the hole in every
        # cell whose index sum is even.
        n, px = grid.n_pixels, grid.pixel_size
        origin = -(n // 2) * px
        cells = int(math.ceil(n * px / pitch)) + 1
        for j in range(cells):
            for i in range(cells):
                if (i + j) % 2 == 0:
                    shapes.append(Contact(
                        LAYER, origin + (i + 0.5) * pitch, origin + (j + 0.5) * pitch,
                        cd, "square",
                    ))
    elif pattern == "frame and bars":
        shapes = _frame_and_bars(cd)
    elif pattern == "isolated contacts":
        shapes = _isolated_contacts(pitch, cd)
    else:
        raise ValueError(f"unknown pattern '{pattern}'")
    # Only what the field sees. A periodic builder overruns the field by a
    # cell; a shape wholly outside it would be fragmented, held, and drawn
    # for nothing.
    def seen(s) -> bool:
        bx0, by0, bx1, by1 = s.bounds()
        return bx1 > x0 and bx0 < x1 and by1 > y0 and by0 < y1

    return Layout([s for s in shapes if seen(s)], name=pattern)


def mask_from_layout(
    layout: Layout,
    grid: GridConfig,
    mask_type: str = "binary",
    *,
    layer: str | None = None,
    oversample: int = 4,
) -> NDArray:
    """Rasterise a layout into the transmittance array the imaging consumes.

    Anti-aliased, so a quarter-pixel edge move is represented — which is
    what lets a sub-nanometre correction act on a 4 nm grid at all. For an
    attenuated PSM the drawn feature becomes the 6 %, 180° film and the
    background clears, exactly as :func:`~litho_sim.mask.patterns.
    to_attenuated_psm` converts a hard raster; blended by coverage here so
    the edge move survives the conversion. On a binary raster the two are
    identical.
    """
    cov = layout.rasterize(grid, layer=layer, tone="clear", oversample=oversample)
    if mask_type == "att-psm":
        return cov * _ATTENUATED + (1.0 - cov) * (1.0 + 0.0j)
    return cov


@dataclasses.dataclass
class AppPrintModel(PrintModel):
    """The engine's print model, rasterising as the app does.

    Same optics, resist, dose and resist model as Print; the only
    difference from :class:`~litho_sim.opc.model.PrintModel` is that the
    layout is drawn by :func:`mask_from_layout`, so an attenuated PSM on
    the Mask tab is an attenuated PSM in the correction. The mask polarity
    follows the mask type: the drawn feature is the bright one on a binary
    mask and the attenuated one on an att-PSM.
    """

    mask_type: str = "binary"

    def rasterize(self, layout: Layout, layer: str | None = None) -> NDArray:
        return mask_from_layout(
            layout, self.grid, self.mask_type, layer=layer, oversample=self.oversample
        )


def opc_availability(params: ParameterModel) -> str | None:
    """Why the correction cannot run at these settings, or ``None`` if it can.

    Shared by the OPC page and, through :func:`~litho_sim.app.ilt.
    ilt_availability`, by the ILT tab: neither correction is offered on a
    design that is imaging its own wrapped copy.
    """
    if params["mask_model"] == "fdtd":
        return (
            "OPC images through the thin or multilayer mask model. The FDTD "
            "near-field library is solved for the drawn line geometry and would "
            "not see a corrected edge — set Mask model to thin."
        )
    return field_fit(params)


def print_model(params: ParameterModel, cache: dict | None = None) -> AppPrintModel:
    """The process at the current settings, as a print model.

    Keeps the user's normalisation rather than forcing the engine's
    ``"clear"`` default, so what the loop prints is what Print prints —
    under peak normalisation every move re-scales the dose a little, which
    the adaptive gain absorbs. *cache* is a previous model's kernel cache,
    handed on so the SOCS kernels survive a change of resist or dose.
    """
    reason = opc_availability(params)
    if reason is not None:
        raise ValueError(reason)
    mask_type = str(params["mask_type"])
    model = AppPrintModel(
        params.optics(), params.resist(), params.grid(),
        dose=params.dose,
        tone="dark" if mask_type == "att-psm" else "clear",
        model=params.resist_model,
        normalisation=None,
        mask_type=mask_type,
    )
    if cache is not None:
        model._raw_cache = cache
    return model


def correction_window(
    params: ParameterModel, model: PrintModel
) -> tuple[float, float, float, float]:
    """The part of the field the loop owns: the field, less a guard band.

    The image wraps at the field edge, and the field is rarely a whole
    number of pitches — at the defaults it is 2.56 — so the last feature
    on one side meets the first on the other through the seam. A fragment
    whose search line crosses the seam reads the neighbour's wrapped image
    as its own edge, and a loop that trusts it pulls the outer lines apart.
    The guard band is the search reach, so no live fragment can look past
    the edge; what it holds is exactly what the field cannot judge.
    """
    x0, y0, x1, y1 = field_box(params.grid())
    reach = 2.0 * default_fragment_length(model)
    return (x0 + reach, y0 + reach, x1 - reach, y1 - reach)


def compute_opc(
    params: ParameterModel,
    model: AppPrintModel | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> OPCResult:
    """Correct the drawn pattern for the process on the step tabs.

    Scattering bars, if asked for, are placed first at the dense pitch's
    space — beside every edge with room, which in a dense array is none of
    them — and left alone by the loop. The loop is told the
    :func:`correction_window`, so the fragments of a line that runs out of
    the field, and those within a search reach of its edge, are held.
    """
    t0 = time.perf_counter()
    model = model if model is not None else print_model(params)
    design = app_layout(params)
    layout = design
    if params["opc_sraf"]:
        pitch, cd = float(params.si("pitch")), float(params.si("cd"))
        gap = pitch - cd
        if gap > 0.0:
            layout = add_scattering_bars(design, gap=gap, width=0.4 * cd)
    result = run_opc(
        layout, model,
        layer=LAYER,
        max_iter=int(params["opc_iterations"]),
        corner_radius=0.0 if params["opc_corners"] == "sharp" else None,
        clip=correction_window(params, model),
        progress=progress,
    )
    result.settings["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return result


def dose_to_size(params: ParameterModel) -> float:
    """The dose at which the dense array of the drawn pitch and CD prints to size.

    The anchor every real process is set up on, and what the engine's own
    OPC demo does before correcting. Sized on the app's own raster of the
    *lines and spaces* pattern — not on a layout of its own — because the
    field wraps and a differently placed array meets itself through the
    seam, which under peak normalisation moves the dose. The drawn feature
    is what is sized: the bright line of a binary mask, the attenuated one
    of an att-PSM. NaN when no dose in the bracket prints it to size.
    """
    from litho_sim.analysis.process_window import calibrate_dose_to_size
    from litho_sim.app.compute import build_mask

    reason = opc_availability(params)
    if reason is not None:
        raise ValueError(reason)
    anchor = ParameterModel(dict(params.values))
    anchor.set("pattern", "lines and spaces")
    feature = "below" if params["mask_type"] == "att-psm" else "above"
    return float(calibrate_dose_to_size(
        build_mask(anchor), params.optics(), params.grid(), params.resist(),
        float(params["cd"]), model=params.resist_model, normalisation=None,
        feature=feature, axis=1,
    ))


def opc_summary(result: OPCResult) -> str:
    """The one line a correction is judged by: the error before and after,
    the iterations it took, and whether the assist features stayed dark."""
    sb, sa = result.epe_before.stats(), result.epe_after.stats()
    n_it = len(result.history) - 1
    text = (
        f"max |EPE| {sb['max_abs_epe_nm']:.1f} to {sa['max_abs_epe_nm']:.1f} nm    "
        f"rms {sb['rms_epe_nm']:.1f} to {sa['rms_epe_nm']:.1f} nm    "
        f"{n_it} iteration{'s' if n_it != 1 else ''}"
        + ("" if result.converged else " (not converged)")
    )
    if sa["n_failed"]:
        text += f"    {sa['n_failed']} edge(s) not found"
    if "sraf" in result.corrected.layers():
        flags = assist_features_printed(result.after, result.corrected)
        text += f"    {sum(flags)} of {len(flags)} bars printed"
    elapsed = result.settings.get("elapsed_ms")
    if elapsed is not None:
        text += f"    [{elapsed:.0f} ms]"
    return text


def estimate_opc_cost_ms(params: ParameterModel, image_ms: float) -> float:
    """Rough cost of the correction, given the cost of one image.

    One print per iteration plus the first and the kept one. Through the
    SOCS kernels a print is a third of the Abbe sum — 61 kernels against
    197 source points at the defaults — and the kernels themselves are
    built once and kept.
    """
    prints = int(params["opc_iterations"]) + 2
    per = image_ms * (0.35 if SOCSKernels.supports(params.optics()) else 1.0)
    return float(prints * per)


__all__ = [
    "ISOLATED", "LAYER", "AppPrintModel", "app_layout", "compute_opc",
    "correction_window", "design_extent", "dose_to_size", "estimate_opc_cost_ms",
    "field_box", "field_fit", "mask_from_layout", "opc_availability",
    "opc_summary", "print_model",
]
