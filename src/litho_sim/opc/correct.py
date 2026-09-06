"""
Model-based OPC — the iteration that moves mask edges until the wafer agrees.

The loop is the textbook one, and it is short:

1. Fragment the drawn layout's edges (:mod:`litho_sim.opc.fragments`).
2. Print the current mask through the full engine
   (:class:`~litho_sim.opc.model.PrintModel`).
3. Measure the edge placement error at each fragment's control point,
   against the *drawn* edge — the target never moves.
4. Move each fragment along its normal by ``−gain × EPE``, clamped.
5. Repeat until the largest |EPE| is under tolerance.

What makes it work is that a mask edge moved by δ moves the printed edge
by roughly ``MEEF × δ`` with MEEF near 1 for well-resolved features, so a
gain below 1 is a damped Newton step on a nearly linear problem and
converges in a handful of iterations. What makes it interesting is where
that stops being true — line ends, whose response is sub-linear and whose
EPE starts at tens of nanometres, and dense pitches near the resolution
limit where MEEF climbs and neighbouring fragments couple. Those are the
places the ``history`` shows converging last, if at all.

Nothing here knows the difference between a serif, a hammerhead and a
bias. All three are just what the offsets look like afterwards.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from litho_sim.mask.geometry import shapes_overlap_distance
from litho_sim.mask.layout import Layout
from litho_sim.opc.fragments import (
    FragmentedShape,
    facing_distances,
    fragment_layout,
    rebuild_layout,
)
from litho_sim.opc.model import EPE, PrintedImage, PrintModel, measure_epe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults tied to the optics
# ---------------------------------------------------------------------------


def default_fragment_length(model: PrintModel) -> float:
    """A quarter of the optical resolution ``λ/NA``, but never under two pixels.

    52 nm at ArF dry, 36 nm at ArF immersion, 10 nm at EUV. Shorter
    fragments give the loop more freedom than the optics can distinguish
    and the anti-aliased raster can represent; longer ones cannot bend
    around a corner.
    """
    optical = 0.25 * model.optics.wavelength / model.optics.NA
    return float(max(optical, 2.0 * model.grid.pixel_size))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class OPCResult:
    """Everything an OPC run produced, before and after.

    Attributes
    ----------
    design : Layout
        The drawn layout — the target.
    corrected : Layout
        The layout to put on the mask. Polygons on the design's layers;
        rasterises, serialises and exposes like any other layout.
    fragmented : list of FragmentedShape
        The fragments with their final offsets. ``offsets`` is the
        correction itself, fragment by fragment.
    before, after : PrintedImage
        The design and the corrected layout, printed.
    epe_before, epe_after : EPE
        Edge placement errors of each, at the same control points.
    history : DataFrame
        One row per iteration: ``iteration, max_abs_epe_nm, rms_epe_nm,
        n_failed, max_abs_offset_nm``.
    epe_trace, offset_trace : (n_iterations + 1, M) arrays
        Every fragment's EPE and offset at every iteration [m] — the
        convergence, fragment by fragment. Plot a column to watch a line
        end being fixed.
    converged : bool
        Whether the loop stopped on tolerance rather than on the cap.
        ``corrected``/``after``/``epe_after`` are the kept iterate (see
        ``keep``), which is the last one when it converged.
    settings : dict
        The parameters the run used, resolved from their defaults.
    """

    design: Layout
    corrected: Layout
    fragmented: list[FragmentedShape]
    before: PrintedImage
    after: PrintedImage
    epe_before: EPE
    epe_after: EPE
    history: pd.DataFrame
    epe_trace: NDArray[np.float64]
    offset_trace: NDArray[np.float64]
    converged: bool
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def offsets(self) -> NDArray[np.float64]:
        """All fragment offsets [m], in fragment order."""
        if not self.fragmented:
            return np.zeros(0)
        return np.concatenate([fs.offsets for fs in self.fragmented])

    def summary(self) -> pd.DataFrame:
        """One row per fragment — where it is, what it was, what it became.

        Columns: ``shape, edge, role, x_nm, y_nm, nx, ny, epe_before_nm,
        epe_after_nm, offset_nm, status``. ``x, y`` and ``nx, ny`` are the
        measurement site and direction — the retargeted corner for a corner
        fragment. Sort by ``epe_before_nm`` to see which parts of a layout
        needed the correction — it is always the line ends and the corners,
        and the table says by how much.
        """
        rows = []
        k = 0
        for si, fs in enumerate(self.fragmented):
            for fi, f in enumerate(fs.fragments):
                c = f.control_point
                rows.append(dict(
                    shape=si, edge=f.edge, role=f.role,
                    x_nm=c[0] * 1e9, y_nm=c[1] * 1e9,
                    nx=f.control_normal[0], ny=f.control_normal[1],
                    epe_before_nm=self.epe_before.values[k] * 1e9,
                    epe_after_nm=self.epe_after.values[k] * 1e9,
                    offset_nm=fs.offsets[fi] * 1e9,
                    status=self.epe_after.status[k],
                ))
                k += 1
        return pd.DataFrame(rows)

    def stats(self) -> pd.DataFrame:
        """The headline numbers before and after, as a two-row table."""
        return pd.DataFrame(
            [self.epe_before.stats(), self.epe_after.stats()], index=["before", "after"]
        )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_opc(
    layout: Layout,
    model: PrintModel,
    *,
    layer: str | None = None,
    fragment_length: float | None = None,
    corner_length: float | None = None,
    corner_radius: float | None = None,
    max_iter: int = 12,
    tol: float = 1e-9,
    corner_tol: float | None = None,
    gain: float = 0.7,
    adaptive: bool = True,
    max_step: float | None = None,
    max_offset: float | None = None,
    min_space: float | None = None,
    search: float | None = None,
    keep: str = "best",
    clip: tuple[float, float, float, float] | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> OPCResult:
    """Correct a layout so it prints as drawn under *model*.

    Parameters
    ----------
    layout : Layout
        The design. Shapes on *layer* are corrected; every other layer
        (assist features, say) is imaged as-is and left alone. Shapes must
        not abut or overlap — a T is one polygon, not a line and a bar —
        since a shared edge would be corrected as two facing edges.
    model : PrintModel
        The process to correct for, at its dose.
    layer : str, optional
        Which layer to correct. ``None`` corrects every shape.
    fragment_length, corner_length : float, optional
        Fragmentation [m]; see :func:`default_fragment_length`. The corner
        length defaults to half the fragment length.
    corner_radius : float, optional
        Radius convex corners are retargeted to [m]; defaults to the
        fragment length. Ask for sharp corners (``0``) and the corner
        fragments chase an EPE no mask can remove, all the way to the
        mask-rule cap.
    max_iter : int
        Iteration cap. Each iteration is one aerial image.
    tol : float
        Stop when every edge fragment's |EPE| is under this [m].
    corner_tol : float, optional
        Tolerance for the retargeted corner sites [m]; defaults to twice
        *tol*. A corner's contour is a compromise between two edges and
        the smallest radius the optics can draw, so it settles a little
        further from its target than an edge does — and a corner held to
        the edge tolerance keeps the loop chasing a residual that only a
        larger ``corner_radius`` can remove.
    gain : float
        Fraction of the EPE each move takes back, to start with. 1 is a
        pure Newton step at MEEF 1 and diverges on coupled fragments
        without adaptation; 0.7 converges with or without it. Every
        fragment then keeps its own gain, halved whenever its EPE changes
        sign or fails to follow its move, restored while moves keep
        working.
    adaptive : bool
        Turn the per-fragment gain adaptation off to run the plain
        fixed-gain iteration — the textbook loop, and the one to watch
        first.
    max_step : float, optional
        Largest single-iteration move of a fragment [m]; defaults to a
        quarter fragment length. Also the move a failed fragment makes.
    max_offset : float, optional
        Largest total offset of a fragment [m]; defaults to the fragment
        length. A correction bigger than this is a design-rule problem,
        not an OPC one.
    min_space : float, optional
        The mask rule check [m]: no fragment may move so far outward that
        the space to the feature it faces drops below this, nor so far
        inward that its own feature gets narrower than this. Each fragment
        is allowed half of what is available, so two facing fragments can
        both move and still respect the rule. Defaults to half the fragment
        length. This is what stops a corner fragment beside a pulled-back
        line end from growing a serif until the dense array bridges — the
        failure the loop finds within five iterations without it.
    search : float, optional
        EPE search range along the normal [m]; defaults to twice the
        fragment length.
    keep : str
        ``"best"`` returns the iterate with the fewest failed sites and
        then the smallest worst-case |EPE| — the loop is not monotone
        once fragments sit on a cap, and the mask worth keeping is the
        best one it saw. ``"last"`` returns the final iterate.
    clip : (x0, y0, x1, y1), optional
        The imaged field [m]. Fragments whose control point lies outside
        it are held fixed — not measured, not moved — so a shape that
        runs out of the field is corrected only where it is imaged. The
        imaging is periodic, and without this a line drawn through the
        whole field reads its window edges as line ends. The layout is
        rasterised on the model's grid regardless; this only says which
        fragments the loop owns.
    progress : callable, optional
        Called ``progress(iteration, max_abs_epe_nm)`` after each print.

    Returns
    -------
    OPCResult
    """
    frag_len = fragment_length if fragment_length is not None else default_fragment_length(model)
    corner = corner_length if corner_length is not None else 0.5 * frag_len
    step_cap = max_step if max_step is not None else 0.25 * frag_len
    total_cap = max_offset if max_offset is not None else frag_len
    reach = search if search is not None else 2.0 * frag_len
    if not 0.0 < gain <= 2.0:
        raise ValueError(f"gain must be in (0, 2], got {gain}")

    rule = min_space if min_space is not None else 0.5 * frag_len
    tol_corner = corner_tol if corner_tol is not None else 2.0 * tol

    radius = corner_radius if corner_radius is not None else frag_len

    targets = layout.shapes if layer is None else layout.on_layer(layer)
    _warn_if_touching(targets)
    fragmented = fragment_layout(
        layout, frag_len, corner, layer=layer, corner_radius=radius, clip=clip
    )
    untouched = [] if layer is None else [s for s in layout.shapes if s.layer != layer]
    n_frag = sum(len(fs) for fs in fragmented)
    held = np.array([f.fixed for fs in fragmented for f in fs.fragments], dtype=bool)

    # Mask rule check, from the drawn geometry once: each fragment's share of
    # the space in front of it and of the width behind it. A held fragment's
    # share is nothing: it stays where it was drawn.
    space = facing_distances(fragmented, +1.0)
    width = facing_distances(fragmented, -1.0)
    hi = np.minimum(total_cap, np.maximum(0.5 * (space - rule), 0.0))
    lo = -np.minimum(total_cap, np.maximum(0.5 * (width - rule), 0.0))
    hi[held] = 0.0
    lo[held] = 0.0
    logger.info(
        "OPC: %d shapes, %d fragments (%.1f nm, %d held), tol %.2f nm, up to %d iterations",
        len(fragmented), n_frag, frag_len * 1e9, int(held.sum()), tol * 1e9, max_iter,
    )

    def current() -> Layout:
        lay = rebuild_layout(fragmented, name=f"{layout.name}+opc")
        lay.shapes.extend(untouched)
        return lay

    rows: list[dict[str, float]] = []
    epe_trace: list[NDArray[np.float64]] = []
    offset_trace: list[NDArray[np.float64]] = []
    first: tuple[PrintedImage, EPE] | None = None
    printed, epe = None, None
    converged = False
    # Per-fragment gain: starts at *gain*, halves whenever a fragment's EPE
    # changes sign between iterations (it overshot: coupled to a neighbour,
    # or on a non-linear response) or fails to follow its own move (it is
    # not the fragment that owns this edge — a side fragment beside a line
    # end, whose contour the end fragments decide), and creeps back up
    # while moves keep working.
    gains = np.full(n_frag, float(gain))
    prev = np.full(n_frag, np.nan)
    prev_offs = np.zeros(n_frag)
    # A move smaller than the raster's sub-pixel step is invisible to the
    # model, so it says nothing about sensitivity.
    min_move = model.grid.pixel_size / max(int(model.oversample), 1)
    weak_sensitivity = 0.25          # d(EPE)/d(offset) below this is "not my edge"
    # A retargeted fragment moves along its edge normal but is measured along
    # the arc normal at its site, so only the projection of a move shows up
    # in its EPE. Dividing by that projection keeps the effective gain the
    # same for every fragment.
    proj = np.concatenate([
        [max(float(f.normal @ f.control_normal), 0.5) for f in fs.fragments] for fs in fragmented
    ]) if fragmented else np.zeros(0)
    tolerances = np.concatenate([
        [tol_corner if f.retargeted else tol for f in fs.fragments] for fs in fragmented
    ]) if fragmented else np.zeros(0)

    for it in range(max_iter + 1):
        corrected = current()
        printed = model.print(corrected)
        epe = measure_epe(printed, fragmented, reach)
        if first is None:
            first = (printed, epe)
        offs = np.concatenate([fs.offsets for fs in fragmented]) if fragmented else np.zeros(0)
        epe_trace.append(epe.values.copy())
        offset_trace.append(offs.copy())
        rows.append(dict(
            iteration=it,
            max_abs_epe_nm=epe.max_abs * 1e9,
            rms_epe_nm=epe.rms * 1e9,
            n_failed=epe.n_failed,
            max_abs_offset_nm=float(np.abs(offs).max()) * 1e9 if offs.size else 0.0,
        ))
        logger.info(
            "  iteration %d: max |EPE| %.2f nm, rms %.2f nm, %d failed",
            it, epe.max_abs * 1e9, epe.rms * 1e9, epe.n_failed,
        )
        if progress is not None:
            progress(it, epe.max_abs * 1e9)

        ok = epe.ok
        if epe.n_failed == 0 and bool(np.all(np.abs(epe.values[ok]) < tolerances[ok])):
            converged = True
            break
        if it == max_iter:
            break

        # The move. A fragment whose edge was not found makes the largest
        # allowed move in the direction that recovers it.
        finite = np.isfinite(prev) & np.isfinite(epe.values)
        flipped = finite & (np.sign(prev) != np.sign(epe.values))
        d_off = offs - prev_offs
        moved = finite & (np.abs(d_off) > min_move)
        with np.errstate(divide="ignore", invalid="ignore"):
            sensitivity = np.where(moved, (epe.values - prev) / d_off, np.nan)
        weak = moved & (sensitivity < weak_sensitivity)
        strong = moved & (sensitivity >= weak_sensitivity) & ~flipped
        if adaptive:
            gains = np.where(flipped | weak, np.maximum(0.5 * gains, 0.02), gains)
            gains = np.where(strong, np.minimum(1.2 * gains, gain), gains)
        prev = epe.values.copy()
        prev_offs = offs.copy()
        move = -gains * epe.values / proj
        for i, st in enumerate(epe.status):
            if st == "missing":
                move[i] = +step_cap
            elif st == "merged":
                move[i] = -step_cap
        move[held] = 0.0
        move = np.clip(move, -step_cap, step_cap)
        k = 0
        for fs in fragmented:
            m = len(fs)
            fs.offsets[:] = np.clip(fs.offsets + move[k:k + m], lo[k:k + m], hi[k:k + m])
            k += m

    assert first is not None and printed is not None and epe is not None
    if keep not in ("best", "last"):
        raise ValueError(f"keep must be 'best' or 'last', got '{keep}'")
    if keep == "best" and len(rows) > 1:
        ranked = sorted(
            range(len(rows)),
            key=lambda k: (rows[k]["n_failed"], rows[k]["max_abs_epe_nm"], k),
        )
        best = ranked[0]
        if best != len(rows) - 1:
            logger.info("OPC: keeping iteration %d, the best of %d", best, len(rows) - 1)
            k = 0
            for fs in fragmented:
                m = len(fs)
                fs.offsets[:] = offset_trace[best][k:k + m]
                k += m
            printed = model.print(current())
            epe = measure_epe(printed, fragmented, reach)
    result = OPCResult(
        design=layout,
        corrected=current(),
        fragmented=fragmented,
        before=first[0],
        after=printed,
        epe_before=first[1],
        epe_after=epe,
        history=pd.DataFrame(rows),
        epe_trace=np.asarray(epe_trace).reshape(len(rows), n_frag),
        offset_trace=np.asarray(offset_trace).reshape(len(rows), n_frag),
        converged=converged,
        settings=dict(
            fragment_length=frag_len, corner_length=corner, max_iter=max_iter, tol=tol,
            corner_radius=radius, corner_tol=tol_corner, gain=gain, adaptive=adaptive,
            keep=keep, max_step=step_cap,
            max_offset=total_cap, min_space=rule, search=reach, layer=layer,
            clip=clip, dose=model.dose, tone=model.tone, model=model.model,
        ),
    )
    logger.info(
        "OPC %s after %d iteration(s): max |EPE| %.2f → %.2f nm",
        "converged" if converged else "stopped", len(rows) - 1,
        result.epe_before.max_abs * 1e9, result.epe_after.max_abs * 1e9,
    )
    return result


def _warn_if_touching(shapes) -> None:
    """Shapes that abut or overlap are one feature drawn as two.

    OPC fragments each outline on its own, so the shared edge of two
    abutting rectangles is treated as two edges facing each other inside
    a printed feature — every fragment on it reads ``"merged"`` and
    retreats, and the two halves pull apart. There are no polygon
    booleans in this engine, so the fix is upstream: draw a T as one
    :class:`~litho_sim.mask.geometry.Polygon`.
    """
    touching = [
        (i, j)
        for i in range(len(shapes))
        for j in range(i + 1, len(shapes))
        if shapes_overlap_distance(shapes[i], shapes[j]) <= 0.0
    ]
    if touching:
        logger.warning(
            "OPC: %d pair(s) of shapes abut or overlap (first: %s); their shared "
            "edges will be corrected as if they were feature edges. Merge them "
            "into one Polygon first.", len(touching), touching[0],
        )


def verify(
    layout: Layout,
    model: PrintModel,
    *,
    layer: str | None = None,
    fragment_length: float | None = None,
    corner_length: float | None = None,
    corner_radius: float | None = None,
    search: float | None = None,
    clip: tuple[float, float, float, float] | None = None,
) -> tuple[PrintedImage, EPE, list[FragmentedShape]]:
    """Print a layout once and measure its EPE — OPC's step 2 and 3 with no step 4.

    The tool for the question "how far off is this design as drawn?", and
    for checking a corrected layout at a *different* condition from the one
    it was corrected at — through focus, at the edges of the dose window —
    which is where a correction shows whether it was robust or merely
    exact. *clip* is the imaged field, as for :func:`run_opc`.
    """
    frag_len = fragment_length if fragment_length is not None else default_fragment_length(model)
    corner = corner_length if corner_length is not None else 0.5 * frag_len
    reach = search if search is not None else 2.0 * frag_len
    radius = corner_radius if corner_radius is not None else frag_len
    fragmented = fragment_layout(
        layout, frag_len, corner, layer=layer, corner_radius=radius, clip=clip
    )
    printed = model.print(layout)
    return printed, measure_epe(printed, fragmented, reach), fragmented


__all__ = ["OPCResult", "run_opc", "verify", "default_fragment_length"]
