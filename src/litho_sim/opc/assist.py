"""
Sub-resolution assist features — scattering bars placed by rule.

Model-based OPC moves edges that exist. It cannot give an isolated line
what a dense line has for free: neighbours. The diffraction orders of a
dense pitch are what hold its image contrast through focus; an isolated
edge has no pitch, its spectrum is broad, and it prints with a worse
depth of focus and a different bias than the same width in an array —
the iso-dense difference that every process window shows.

A scattering bar is a narrow feature drawn beside an isolated edge, at
about the dense pitch's spacing, too thin to print itself. It gives the
edge a neighbour in frequency space without giving it one on the wafer.
Placement is rule-based — a gap and a width, applied wherever there is
room — because the rule *is* the physics: the gap is the pitch you are
imitating, and the width is whatever stays below the resist threshold
at dose.

Bars land on their own layer, so :func:`~litho_sim.opc.correct.run_opc`
leaves them alone while correcting the design, and so the check that
they did not print can find them again.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from matplotlib.path import Path as MplPath

from litho_sim.mask.geometry import Rect, Shape, polygon_distance
from litho_sim.mask.layout import Layout
from litho_sim.opc.model import PrintedImage

logger = logging.getLogger(__name__)


def _ccw(points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return points if area2 >= 0 else points[::-1]


def _inside_any(point: np.ndarray, shapes: list[Shape]) -> bool:
    xy = (float(point[0]), float(point[1]))
    return any(MplPath(s.polygon()).contains_point(xy) for s in shapes)


def add_scattering_bars(
    layout: Layout,
    *,
    gap: float,
    width: float,
    clearance: float | None = None,
    min_length: float | None = None,
    pullback: float | None = None,
    layer: str = "sraf",
    beside: str | None = None,
) -> Layout:
    """Place a scattering bar beside every design edge that has room for one.

    Parameters
    ----------
    layout : Layout
        The design. Returned layout is a new object; the design shapes are
        shared, not copied.
    gap : float
        Edge-to-bar space [m] — typically the dense pitch's space, so the
        bar sits where a neighbouring line would.
    width : float
        Bar width [m]. Sub-resolution: a third to a half of the minimum
        printable width is usual, and :func:`assist_features_printed` is
        the check.
    clearance : float, optional
        Minimum distance from a bar to any design shape and to any other
        bar [m]. Defaults to *gap*. A candidate that violates it is dropped
        — that is how dense edges get no bar: there is no room.
    min_length : float, optional
        Shortest edge that gets a bar [m]; defaults to ``2 × gap``. Rules
        out line ends and contact sides.
    pullback : float, optional
        How much shorter than its edge a bar is, at each end [m]; defaults
        to *width*. Keeps bars from poking past corners.
    layer : str
        Layer the bars are drawn on.
    beside : str, optional
        Only design shapes on this layer get bars. ``None`` means all of
        them (excluding any already on *layer*).

    Returns
    -------
    Layout
        The design plus its bars, named ``"<design>+sraf"``.
    """
    if gap <= 0 or width <= 0:
        raise ValueError("gap and width must be positive")
    clear = gap if clearance is None else clearance
    shortest = 2.0 * gap if min_length is None else min_length
    pull = width if pullback is None else pullback

    design = [s for s in layout.shapes if s.layer != layer]
    targets = design if beside is None else [s for s in design if s.layer == beside]
    bars: list[Shape] = []
    n_considered = 0

    for s in targets:
        pts = _ccw(np.asarray(s.polygon(), dtype=np.float64))
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            d = b - a
            L = float(np.hypot(*d))
            if L < shortest or L - 2.0 * pull <= 0.0:
                continue
            n_considered += 1
            t = d / L
            normal = np.array([t[1], -t[0]])
            centre = 0.5 * (a + b) + (gap + 0.5 * width) * normal
            bar = Rect(
                layer, float(centre[0]), float(centre[1]), L - 2.0 * pull, width,
                math.degrees(math.atan2(t[1], t[0])),
            )
            if _inside_any(centre, design):
                continue
            if any(polygon_distance(bar, o) < clear - 1e-15 for o in design):
                continue
            if any(polygon_distance(bar, o) < clear - 1e-15 for o in bars):
                continue
            if _inside_any(centre, bars):
                continue
            bars.append(bar)

    logger.info(
        "Scattering bars: %d placed beside %d candidate edges (gap %.0f nm, width %.0f nm)",
        len(bars), n_considered, gap * 1e9, width * 1e9,
    )
    out = Layout(list(layout.shapes) + bars, name=f"{layout.name}+sraf")
    return out


def assist_features_printed(
    printed: PrintedImage,
    layout: Layout,
    layer: str = "sraf",
    samples: int = 7,
) -> list[bool]:
    """Which assist features printed — one flag per bar on *layer*.

    Samples the signed latent field along each bar's centreline; a bar has
    printed if the field is above the resist edge level anywhere along it.
    A bar that prints is a defect, and the fix is a narrower bar or a
    smaller gap, not a different OPC.
    """
    flags = []
    for s in layout.on_layer(layer):
        poly = np.asarray(s.polygon(), dtype=np.float64)
        if isinstance(s, Rect):
            t = math.radians(s.rotation)
            axis = np.array([math.cos(t), math.sin(t)])
            half = 0.5 * max(s.w - 2.0 * printed.grid.pixel_size, 0.0)
            line = np.array([s.cx, s.cy]) + np.linspace(-half, half, samples)[:, None] * axis
        else:
            line = poly.mean(axis=0, keepdims=True)
        flags.append(bool((printed.sample(line) > 0.0).any()))
    return flags


__all__ = ["add_scattering_bars", "assist_features_printed"]
