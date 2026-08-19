"""
Mask layouts and multi-patterning decomposition.

A :class:`Layout` is a named collection of geometry layers that rasterises
to the ``(n, n)`` float array the imaging engine already consumes. On top of
that sits the piece multi-patterning actually needs: splitting one layout
that is too dense to print in a single exposure into two or three that are
not.

Decomposition
-------------
Two features closer than the single-exposure minimum spacing *conflict* and
must go to different exposures. Building a graph of those conflicts and
colouring it is exactly graph colouring, solved here with DSATUR plus
bounded backtracking.

The interesting case is when it **fails**. An odd cycle of mutual conflicts
cannot be 2-coloured, no matter how clever the algorithm — that is the real
reason LELE has design rules, and why LE³ exists. So
:func:`decompose` reports the conflicts it could not resolve instead of
raising, letting a UI draw them in red.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig
from litho_sim.core.utils import JsonFileMixin
from litho_sim.mask.geometry import (
    Contact,
    Rect,
    Shape,
    polygon_distance,
    rasterize_shapes,
    shape_from_dict,
    shapes_overlap_distance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass
class Layout(JsonFileMixin):
    """A named set of mask shapes on one or more layers.

    Parameters
    ----------
    shapes : list of Shape
        All geometry.  Each shape carries its own ``layer`` name.
    name : str
        Identifier, used as a recipe key.
    """

    shapes: list[Shape] = field(default_factory=list)
    name: str = "layout"

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------

    def layers(self) -> list[str]:
        """Layer names present, in first-seen order."""
        seen: list[str] = []
        for s in self.shapes:
            if s.layer not in seen:
                seen.append(s.layer)
        return seen

    def on_layer(self, layer: str) -> list[Shape]:
        return [s for s in self.shapes if s.layer == layer]

    def add(self, *shapes: Shape) -> Layout:
        """Append shapes and return self, so calls chain."""
        self.shapes.extend(shapes)
        return self

    # ------------------------------------------------------------------
    # Rasterisation
    # ------------------------------------------------------------------

    def rasterize(
        self,
        grid: GridConfig,
        layer: str | None = None,
        tone: str = "clear",
        oversample: int = 4,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> NDArray[np.float64]:
        """Render to a mask array the imaging engine can consume directly.

        Parameters
        ----------
        grid : GridConfig
            Target grid.
        layer : str, optional
            Restrict to one layer.  Defaults to all layers unioned.
        tone : str
            ``"clear"`` — drawn shapes are transparent (1 = light passes).
            ``"dark"`` — drawn shapes are opaque; the field is clear.
        oversample : int
            Anti-aliasing factor; see :func:`~litho_sim.mask.geometry.rasterize_shapes`.
        dx, dy : float
            Overlay offset applied to every shape [m].

        Returns
        -------
        NDArray[np.float64]
            ``(n_pixels, n_pixels)`` transmittance in [0, 1].
        """
        shapes = self.shapes if layer is None else self.on_layer(layer)
        cov = rasterize_shapes(shapes, grid, oversample=oversample, dx=dx, dy=dy)
        if tone == "clear":
            return cov
        if tone == "dark":
            return 1.0 - cov
        raise ValueError(f"tone must be 'clear' or 'dark', got '{tone}'")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shapes": [s.to_dict() for s in self.shapes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Layout:
        return cls(
            shapes=[shape_from_dict(d) for d in data.get("shapes", [])],
            name=data.get("name", "layout"),
        )

    # ------------------------------------------------------------------
    # Editing helpers (these back the table editor in the UI)
    # ------------------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        """Flatten to table rows with nanometre units, for a data editor."""
        rows = []
        for s in self.shapes:
            x0, y0, x1, y1 = s.bounds()
            rows.append({
                "kind": type(s).__name__,
                "layer": s.layer,
                "cx_nm": round((x0 + x1) / 2 * 1e9, 3),
                "cy_nm": round((y0 + y1) / 2 * 1e9, 3),
                "w_nm": round((x1 - x0) * 1e9, 3),
                "h_nm": round((y1 - y0) * 1e9, 3),
            })
        return rows

    def __len__(self) -> int:
        return len(self.shapes)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Layout({self.name!r}, {len(self.shapes)} shapes, layers={self.layers()})"


# ---------------------------------------------------------------------------
# Parametric builders
# ---------------------------------------------------------------------------


def line_array(
    n_lines: int,
    pitch: float,
    cd: float,
    length: float,
    layer: str = "main",
    orientation: str = "vertical",
    centre: tuple[float, float] = (0.0, 0.0),
) -> list[Shape]:
    """Build an evenly spaced array of lines, centred on the field.

    Parameters
    ----------
    n_lines : int
        Number of lines.
    pitch : float
        Centre-to-centre spacing [m].
    cd : float
        Line width [m].
    length : float
        Line length [m].
    layer : str
        Layer name.
    orientation : str
        ``"vertical"`` (lines run along y) or ``"horizontal"``.
    centre : tuple
        Centre of the whole array [m].
    """
    span = (n_lines - 1) * pitch
    offs = np.linspace(-span / 2, span / 2, n_lines) if n_lines > 1 else np.array([0.0])
    out: list[Shape] = []
    for o in offs:
        if orientation == "vertical":
            out.append(Rect(layer, centre[0] + float(o), centre[1], cd, length))
        elif orientation == "horizontal":
            out.append(Rect(layer, centre[0], centre[1] + float(o), length, cd))
        else:
            raise ValueError(
                f"orientation must be 'vertical' or 'horizontal', got '{orientation}'"
            )
    return out


def contact_grid(
    nx: int,
    ny: int,
    pitch_x: float,
    pitch_y: float,
    d: float,
    layer: str = "main",
    shape: str = "square",
) -> list[Shape]:
    """Build a centred rectangular grid of contacts."""
    xs = (np.arange(nx) - (nx - 1) / 2) * pitch_x
    ys = (np.arange(ny) - (ny - 1) / 2) * pitch_y
    return [
        Contact(layer, float(x), float(y), d, shape)
        for y in ys for x in xs
    ]


def cut_bar(
    x: float, y: float, w: float, h: float, layer: str = "cut"
) -> Shape:
    """A cut/block bar — the standard way one drawn line becomes two gates."""
    return Rect(layer, x, y, w, h)


# ---------------------------------------------------------------------------
# Multi-patterning decomposition
# ---------------------------------------------------------------------------


@dataclass
class Decomposition:
    """Result of colouring a layout for multi-patterning.

    Attributes
    ----------
    colors : list of int
        Colour index per shape, in the order they were passed in.
    conflicts : list of tuple
        Shape index pairs that still conflict — i.e. ended up the same
        colour despite being too close.  Empty means a clean decomposition.
    n_colors : int
        Number of exposures requested.
    """

    colors: list[int]
    conflicts: list[tuple[int, int]]
    n_colors: int

    @property
    def ok(self) -> bool:
        return not self.conflicts

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {c: [] for c in range(self.n_colors)}
        for i, c in enumerate(self.colors):
            out.setdefault(c, []).append(i)
        return out


def conflict_graph(
    shapes: Sequence[Shape], min_spacing: float, exact: bool = False
) -> dict[int, set[int]]:
    """Build the adjacency of features too close to print in one exposure.

    Parameters
    ----------
    shapes : sequence of Shape
        Features to compare.
    min_spacing : float
        Minimum printable edge-to-edge space in a single exposure [m].
        Anything tighter must be split across exposures.
    exact : bool
        Use exact polygon-to-polygon distance instead of the fast
        bounding-box estimate.  Needed for rotated or non-rectangular
        geometry.

    Returns
    -------
    dict
        ``{index: {neighbour indices}}``.
    """
    dist = polygon_distance if exact else shapes_overlap_distance
    adj: dict[int, set[int]] = {i: set() for i in range(len(shapes))}
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if dist(shapes[i], shapes[j]) < min_spacing:
                adj[i].add(j)
                adj[j].add(i)
    n_edges = sum(len(v) for v in adj.values()) // 2
    logger.debug("Conflict graph: %d shapes, %d edges", len(shapes), n_edges)
    return adj


def _dsatur(adj: dict[int, set[int]], n_colors: int) -> list[int]:
    """Colour a graph with DSATUR (most-saturated-first greedy)."""
    n = len(adj)
    colors = [-1] * n
    sat: list[set[int]] = [set() for _ in range(n)]

    for _ in range(n):
        # Pick the uncoloured vertex with the most distinctly-coloured
        # neighbours, breaking ties on raw degree.
        best, best_key = -1, None
        for v in range(n):
            if colors[v] != -1:
                continue
            key = (len(sat[v]), len(adj[v]))
            if best_key is None or key > best_key:
                best, best_key = v, key
        if best == -1:
            break

        for c in range(n_colors):
            if c not in sat[best]:
                colors[best] = c
                break
        else:
            colors[best] = 0  # over-subscribed; recorded as a conflict later

        for u in adj[best]:
            sat[u].add(colors[best])
    return colors


def _remaining_conflicts(
    adj: dict[int, set[int]], colors: Sequence[int]
) -> list[tuple[int, int]]:
    out = []
    for i, neigh in adj.items():
        for j in neigh:
            if i < j and colors[i] == colors[j]:
                out.append((i, j))
    return out


def decompose(
    shapes: Sequence[Shape],
    min_spacing: float,
    n_colors: int = 2,
    exact: bool = False,
    max_backtrack: int = 200_000,
) -> Decomposition:
    """Split a layout across *n_colors* exposures.

    Tries DSATUR first, then a bounded backtracking search if that leaves
    conflicts.  Whatever remains unresolved is reported rather than raised —
    an odd conflict cycle genuinely cannot be 2-coloured, and seeing which
    features form it is the useful output.

    Parameters
    ----------
    shapes : sequence of Shape
        Features to split.
    min_spacing : float
        Single-exposure minimum spacing [m].
    n_colors : int
        Number of exposures (2 for LELE, 3 for LE³).
    exact : bool
        Exact polygon distance for the conflict test.
    max_backtrack : int
        Node budget for the exhaustive search.

    Returns
    -------
    Decomposition
    """
    adj = conflict_graph(shapes, min_spacing, exact=exact)
    colors = _dsatur(adj, n_colors)
    conflicts = _remaining_conflicts(adj, colors)

    if conflicts:
        exact_colors = _backtrack_color(adj, n_colors, max_backtrack)
        if exact_colors is not None:
            colors = exact_colors
            conflicts = _remaining_conflicts(adj, colors)

    if conflicts:
        logger.warning(
            "Decomposition into %d colours left %d conflict(s); "
            "the layout is not %d-colourable at %.0f nm spacing.",
            n_colors, len(conflicts), n_colors, min_spacing * 1e9,
        )
    else:
        logger.info(
            "Decomposed %d shapes into %d colours cleanly.", len(shapes), n_colors
        )
    return Decomposition(colors=list(colors), conflicts=conflicts, n_colors=n_colors)


def _backtrack_color(
    adj: dict[int, set[int]], n_colors: int, budget: int
) -> list[int] | None:
    """Exhaustive colouring within a node budget; None if it gives up."""
    n = len(adj)
    colors = [-1] * n
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    steps = 0

    def go(k: int) -> bool:
        nonlocal steps
        if k == n:
            return True
        steps += 1
        if steps > budget:
            return False
        v = order[k]
        used = {colors[u] for u in adj[v] if colors[u] != -1}
        for c in range(n_colors):
            if c in used:
                continue
            colors[v] = c
            if go(k + 1):
                return True
            colors[v] = -1
        return False

    return colors if go(0) else None


def split_by_color(
    layout: Layout,
    decomposition: Decomposition,
    layer: str | None = None,
    prefix: str = "color",
) -> dict[str, Layout]:
    """Split a layout into one :class:`Layout` per exposure colour.

    Returns
    -------
    dict
        ``{"colorA": Layout, "colorB": Layout, ...}`` — the keys a
        multi-patterning recipe expects.
    """
    shapes = layout.shapes if layer is None else layout.on_layer(layer)
    out: dict[str, Layout] = {}
    for c in range(decomposition.n_colors):
        key = f"{prefix}{chr(ord('A') + c)}"
        picked = [
            s for s, col in zip(shapes, decomposition.colors) if col == c
        ]
        out[key] = Layout(shapes=list(picked), name=f"{layout.name}:{key}")
    return out
