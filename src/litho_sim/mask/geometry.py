"""
Mask geometry primitives.

The engine consumes masks as ``(n, n)`` float arrays, which is fine for the
five hard-coded patterns it shipped with but useless for designing a real
device layer.  This module adds analytic shapes — rectangles, polygons,
paths, contacts — that rasterise *into that same array format*, so nothing
downstream has to change.

Coordinates are in metres, measured from the **centre of the field**, which
is the convention :func:`~litho_sim.mask.patterns.lines_and_spaces` and friends
already use and the origin the FFT assumes.

Anti-aliasing
-------------
Rasterisation supersamples and averages, producing fractional edge pixels.
This is not cosmetic. At a 4 nm pixel, a 1 nm overlay shift is a quarter of
a pixel; with hard binary rasterisation that shift is invisible, so every
overlay and pitch-walking result would quantise to the pixel grid and the
whole multi-patterning story would be a staircase. Grey edges also improve
aerial-image fidelity, since the real mask edge does not sit on our grid.

No Shapely
----------
Polygon fill uses ``matplotlib.path.Path``, and Matplotlib is already a hard
dependency. Raster booleans cover the rest.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig

logger = logging.getLogger(__name__)

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """Base class for mask geometry. Coordinates are metres from field centre."""

    layer: str = "main"

    def polygon(self) -> NDArray[np.float64]:
        """Return the shape as an ``(N, 2)`` closed polygon in metres."""
        raise NotImplementedError

    def translated(self, dx: float, dy: float) -> Shape:
        """Return a copy shifted by (dx, dy) metres.

        Applied analytically before rasterisation, so sub-pixel overlay is
        represented exactly rather than snapped to the pixel grid.
        """
        raise NotImplementedError

    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box ``(xmin, ymin, xmax, ymax)`` in metres."""
        p = self.polygon()
        return (float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = type(self).__name__
        return d


@dataclass(frozen=True)
class Rect(Shape):
    """An axis-aligned or rotated rectangle, specified by its centre."""

    cx: float = 0.0
    cy: float = 0.0
    w: float = 100e-9
    h: float = 100e-9
    rotation: float = 0.0  # degrees, counter-clockwise

    def polygon(self) -> NDArray[np.float64]:
        hw, hh = self.w / 2.0, self.h / 2.0
        pts = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)], dtype=np.float64)
        if self.rotation:
            t = math.radians(self.rotation)
            c, s = math.cos(t), math.sin(t)
            pts = pts @ np.array([[c, s], [-s, c]], dtype=np.float64)
        return pts + np.array([self.cx, self.cy])

    def translated(self, dx: float, dy: float) -> Rect:
        return Rect(self.layer, self.cx + dx, self.cy + dy, self.w, self.h, self.rotation)


@dataclass(frozen=True)
class Polygon(Shape):
    """An arbitrary closed polygon."""

    points: tuple[Point, ...] = ()

    def polygon(self) -> NDArray[np.float64]:
        if len(self.points) < 3:
            raise ValueError(f"Polygon needs at least 3 points, got {len(self.points)}")
        return np.asarray(self.points, dtype=np.float64)

    def translated(self, dx: float, dy: float) -> Polygon:
        return Polygon(self.layer, tuple((x + dx, y + dy) for x, y in self.points))


@dataclass(frozen=True)
class PathShape(Shape):
    """A polyline stroked to a given width — the natural way to draw a wire."""

    points: tuple[Point, ...] = ()
    width: float = 40e-9

    def polygon(self) -> NDArray[np.float64]:
        pts = np.asarray(self.points, dtype=np.float64)
        if len(pts) < 2:
            raise ValueError("PathShape needs at least 2 points")
        hw = self.width / 2.0
        left, right = [], []
        for i in range(len(pts) - 1):
            d = pts[i + 1] - pts[i]
            n = np.hypot(*d)
            if n < 1e-18:
                continue
            nx, ny = -d[1] / n, d[0] / n
            for p in (pts[i], pts[i + 1]):
                left.append((p[0] + nx * hw, p[1] + ny * hw))
                right.append((p[0] - nx * hw, p[1] - ny * hw))
        if not left:
            raise ValueError("PathShape has zero length")
        return np.array(left + right[::-1], dtype=np.float64)

    def translated(self, dx: float, dy: float) -> PathShape:
        return PathShape(
            self.layer, tuple((x + dx, y + dy) for x, y in self.points), self.width
        )


@dataclass(frozen=True)
class Contact(Shape):
    """A contact or via — square, round, or octagonal."""

    cx: float = 0.0
    cy: float = 0.0
    d: float = 40e-9
    shape: str = "square"  # "square" | "round" | "octagon"

    def polygon(self) -> NDArray[np.float64]:
        # `d` is the flat-to-flat width, so the circumradius has to be scaled
        # up by 1/cos(pi/n_sides) for every polygon that is not a circle.
        r = self.d / 2.0
        if self.shape == "square":
            n_sides, phase = 4, math.pi / 4
            r = r / math.cos(math.pi / 4)
        elif self.shape == "octagon":
            n_sides, phase = 8, math.pi / 8
            r = r / math.cos(math.pi / 8)
        elif self.shape == "round":
            n_sides, phase = 32, 0.0
        else:
            raise ValueError(
                f"Contact shape must be 'square', 'round', or 'octagon', got '{self.shape}'"
            )
        a = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + phase
        return np.stack([self.cx + r * np.cos(a), self.cy + r * np.sin(a)], axis=1)

    def translated(self, dx: float, dy: float) -> Contact:
        return Contact(self.layer, self.cx + dx, self.cy + dy, self.d, self.shape)


_SHAPE_KINDS = {c.__name__: c for c in (Rect, Polygon, PathShape, Contact)}


def shape_from_dict(d: dict[str, Any]) -> Shape:
    """Reconstruct a shape from :meth:`Shape.to_dict` output."""
    data = dict(d)
    kind = data.pop("kind")
    if kind not in _SHAPE_KINDS:
        raise ValueError(f"Unknown shape kind '{kind}'. Known: {sorted(_SHAPE_KINDS)}")
    cls = _SHAPE_KINDS[kind]
    if "points" in data and data["points"] is not None:
        data["points"] = tuple(tuple(p) for p in data["points"])
    return cls(**data)


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------


def _axis_coords(grid: GridConfig) -> NDArray[np.float64]:
    """Pixel-centre coordinates in metres, centred on the field."""
    return ((np.arange(grid.n_pixels) - grid.n_pixels // 2) * grid.pixel_size).astype(np.float64)


def rasterize_shapes(
    shapes: Sequence[Shape],
    grid: GridConfig,
    oversample: int = 4,
    dx: float = 0.0,
    dy: float = 0.0,
) -> NDArray[np.float64]:
    """Rasterise shapes into an ``(n, n)`` coverage array in [0, 1].

    Parameters
    ----------
    shapes : sequence of Shape
        Geometry to draw.  Overlapping shapes union.
    grid : GridConfig
        Target grid.
    oversample : int
        Supersampling factor per axis.  4 gives 16 samples per pixel, which
        resolves overlay shifts down to a quarter pixel.  Set to 1 for hard
        binary edges.
    dx, dy : float
        Rigid translation applied to every shape before rasterising [m].
        This is how overlay error enters, and it is applied analytically so
        sub-pixel shifts survive.

    Returns
    -------
    NDArray[np.float64]
        Coverage fraction per pixel, shape ``(n_pixels, n_pixels)``.
    """
    from matplotlib.path import Path as MplPath

    n = grid.n_pixels
    if not shapes:
        return np.zeros((n, n), dtype=np.float64)

    k = max(int(oversample), 1)
    px = grid.pixel_size
    sub = px / k
    # Sub-pixel sample centres spanning the same physical field.
    base = _axis_coords(grid)
    fine = (base[:, None] + (np.arange(k) - (k - 1) / 2.0) * sub).ravel()

    acc = np.zeros((n * k, n * k), dtype=bool)

    for sh in shapes:
        s = sh.translated(dx, dy) if (dx or dy) else sh
        poly = s.polygon()
        # Only rasterise inside the shape's bounding box.
        xmin, ymin, xmax, ymax = (
            poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()
        )
        ix = np.nonzero((fine >= xmin - sub) & (fine <= xmax + sub))[0]
        iy = np.nonzero((fine >= ymin - sub) & (fine <= ymax + sub))[0]
        if ix.size == 0 or iy.size == 0:
            continue

        XX, YY = np.meshgrid(fine[ix], fine[iy])
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        inside = MplPath(poly).contains_points(pts).reshape(YY.shape)
        acc[np.ix_(iy, ix)] |= inside

    # Block-average the supersampled grid back down.
    if k == 1:
        return acc.astype(np.float64)
    return acc.reshape(n, k, n, k).mean(axis=(1, 3)).astype(np.float64)


def shapes_overlap_distance(a: Shape, b: Shape) -> float:
    """Bounding-box gap between two shapes [m]; 0 if the boxes touch.

    The fast path for the multi-patterning conflict graph. It underestimates
    the true edge-to-edge distance for rotated shapes, which errs toward
    *adding* conflicts — the safe direction for decomposition. When the exact
    distance matters, :func:`polygon_distance` is the one to call.
    """
    ax0, ay0, ax1, ay1 = a.bounds()
    bx0, by0, bx1, by1 = b.bounds()
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        gap_x = max(bx0 - ax1, ax0 - bx1, 0.0)
        gap_y = max(by0 - ay1, ay0 - by1, 0.0)
        return float(math.hypot(gap_x, gap_y))
    return 0.0


def _segment_distance(p: NDArray, a: NDArray, b: NDArray) -> NDArray:
    """Distance from points *p* to segment *a*–*b*."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-30:
        return np.linalg.norm(p - a, axis=-1)
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def polygon_distance(a: Shape, b: Shape) -> float:
    """Exact minimum distance between two shape outlines [m].

    Slower than :func:`shapes_overlap_distance` but correct for rotated and
    non-rectangular geometry.
    """
    pa, pb = a.polygon(), b.polygon()
    best = np.inf
    for poly, other in ((pa, pb), (pb, pa)):
        for i in range(len(other)):
            seg_a, seg_b = other[i], other[(i + 1) % len(other)]
            best = min(best, float(_segment_distance(poly, seg_a, seg_b).min()))
    return float(best)
