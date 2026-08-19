"""
Shared plotting layer for the wafer-metrology engine.

Every public function returns a ``(fig, axes)`` tuple; use :func:`save_figure`
to write to disk with consistent DPI and padding.  A single house style is
applied on import by :func:`apply_style`.

Colour policy
-------------
* **Signed** quantities (height deviation from a reference plane, measurement
  error) use a diverging map with a neutral mid-point, always normalised
  symmetrically about zero so that "zero" is the neutral colour.
* **Magnitude** quantities (SFQR, |residual|, fringe intensity) use a single-hue
  perceptually-uniform sequential map.
* **Categorical** series use :data:`PALETTE` in fixed order -- an Okabe-Ito
  ordering validated for deuteranopia/protanopia separation.  Series are always
  given a legend *and* distinct markers, never colour alone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style definition
# ---------------------------------------------------------------------------

_METROLOGY_RCPARAMS = {
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,   # keep the grid behind bars and markers
    "grid.color": "#DDDDDD",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "image.cmap": "viridis",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
}

PALETTE: Tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#009E73",  # bluish green
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#666666",  # grey
)
"""Categorical hues in fixed assignment order (Okabe-Ito, CVD-validated)."""

MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X")
"""Marker cycle paired with :data:`PALETTE` so identity is never colour-alone."""

CMAP_SIGNED = "RdBu_r"
"""Diverging map for signed height / error maps (neutral mid-point)."""

CMAP_MAGNITUDE = "viridis"
"""Single-hue sequential map for non-negative magnitudes."""

INK_PRIMARY = "#222222"
INK_MUTED = "#666666"


def apply_style() -> None:
    """Apply the wafer-metrology matplotlib style globally."""
    matplotlib.rcParams.update(_METROLOGY_RCPARAMS)


apply_style()  # auto-apply on import


def series_style(idx: int) -> dict:
    """Return colour + marker keyword arguments for categorical series *idx*.

    Hues are assigned in fixed order and never cycled beyond the palette
    length; a caller needing more than ``len(PALETTE)`` series should facet
    instead.

    Parameters
    ----------
    idx : int
        Zero-based series index.

    Returns
    -------
    dict
        ``{"color": ..., "marker": ...}`` ready to splat into a plot call.
    """
    return {
        "color": PALETTE[idx % len(PALETTE)],
        "marker": MARKERS[idx % len(MARKERS)],
    }


# ---------------------------------------------------------------------------
# Figure I/O
# ---------------------------------------------------------------------------


def save_figure(fig: Figure, path: str | Path, dpi: int = 150, close: bool = True) -> Path:
    """Write *fig* to *path*, creating parent directories as needed.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    path : str or Path
        Destination file path.
    dpi : int
        Output resolution.
    close : bool
        Close the figure afterwards to free memory (default ``True``).

    Returns
    -------
    Path
        The resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    logger.info("Wrote %s", out)
    return out


# ---------------------------------------------------------------------------
# Wafer maps
# ---------------------------------------------------------------------------


def _extent_mm(x: NDArray[np.float64], y: NDArray[np.float64]) -> Tuple[float, float, float, float]:
    """Return an imshow ``extent`` in millimetres for coordinate grids *x*, *y*."""
    return (
        float(np.nanmin(x)) * 1e3,
        float(np.nanmax(x)) * 1e3,
        float(np.nanmin(y)) * 1e3,
        float(np.nanmax(y)) * 1e3,
    )


def wafer_imshow(
    ax: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    *,
    scale: float = 1e6,
    label: str = "Height [µm]",
    signed: bool = True,
    cmap: Optional[str] = None,
    vlim: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
    title: str = "",
    colorbar: bool = True,
) -> object:
    """Draw a wafer map on *ax* with physical axes in millimetres.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x, y : NDArray
        Coordinate grids [m], as returned by :func:`~wafer_metrology.synthesize.make_wafer_grid`.
    z : NDArray
        Height map [m] with ``NaN`` outside the wafer.
    scale : float
        Multiplier applied to *z* before plotting (``1e6`` -> µm, ``1e9`` -> nm).
    label : str
        Colour-bar label, including the post-scaling unit.
    signed : bool
        If ``True`` use the diverging map normalised symmetrically about zero;
        otherwise use the sequential magnitude map.
    cmap : str, optional
        Explicit colormap override.
    vlim : (float, float), optional
        Colour limits **after** scaling.  Defaults to the data range (symmetric
        about zero when *signed*).
    radius : float, optional
        Wafer radius [m].  When given, the aperture outline is drawn.
    title : str
        Axes title.
    colorbar : bool
        Attach a colour bar to the parent figure.

    Returns
    -------
    AxesImage
        The image artist, so callers can attach further colour bars.
    """
    zs = np.asarray(z, dtype=np.float64) * scale
    if cmap is None:
        cmap = CMAP_SIGNED if signed else CMAP_MAGNITUDE

    if vlim is not None:
        vmin, vmax = vlim
    elif signed:
        span = float(np.nanmax(np.abs(zs))) if np.isfinite(zs).any() else 1.0
        span = span if span > 0 else 1.0
        vmin, vmax = -span, span
    else:
        vmin = float(np.nanmin(zs)) if np.isfinite(zs).any() else 0.0
        vmax = float(np.nanmax(zs)) if np.isfinite(zs).any() else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if (signed and vmin < 0 < vmax) else None
    im = ax.imshow(
        zs,
        origin="lower",
        extent=_extent_mm(x, y),
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
        **({"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}),
    )

    if radius is not None:
        ax.add_patch(
            plt.Circle((0, 0), radius * 1e3, fill=False, color=INK_MUTED, lw=0.8, ls="--")
        )

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    ax.grid(False)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label=label, shrink=0.82)
    return im


def plot_wafer_surface(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    *,
    radius: Optional[float] = None,
    title: str = "Wafer surface",
    scale: float = 1e6,
    label: str = "Height [µm]",
    signed: bool = True,
    figsize: Tuple[float, float] = (6.0, 5.0),
) -> Tuple[Figure, Axes]:
    """Plot a single wafer height map.

    Parameters
    ----------
    x, y, z : NDArray
        Coordinate grids and height map [m].
    radius : float, optional
        Wafer radius [m] for the aperture outline.
    title : str
        Figure title.
    scale : float
        Height multiplier (``1e6`` -> µm).
    label : str
        Colour-bar label.
    signed : bool
        Use the diverging colour map about zero.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (Figure, Axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    wafer_imshow(
        ax, x, y, z, scale=scale, label=label, signed=signed, radius=radius, title=title
    )
    fig.tight_layout()
    return fig, ax


def plot_wafer_panels(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    maps: Sequence[NDArray[np.float64]],
    titles: Sequence[str],
    *,
    scale: float = 1e6,
    label: str = "Height [µm]",
    signed: bool = True,
    radius: Optional[float] = None,
    shared_scale: bool = True,
    suptitle: str = "",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, NDArray]:
    """Plot several wafer maps side by side on a common colour scale.

    Parameters
    ----------
    x, y : NDArray
        Coordinate grids [m].
    maps : sequence of NDArray
        Height maps [m] to display.
    titles : sequence of str
        Per-panel titles; must match ``len(maps)``.
    scale : float
        Height multiplier.
    label : str
        Colour-bar label.
    signed : bool
        Use the diverging colour map about zero.
    radius : float, optional
        Wafer radius [m] for the aperture outline.
    shared_scale : bool
        Put every panel on one colour scale so panels are directly comparable.
    suptitle : str
        Overall figure title.
    figsize : tuple, optional
        Figure size; defaults to ``(4.6 * n, 4.4)``.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    n = len(maps)
    if len(titles) != n:
        raise ValueError(f"titles has length {len(titles)}, expected {n}")
    if figsize is None:
        figsize = (4.6 * n, 4.4)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    vlim: Optional[Tuple[float, float]] = None
    if shared_scale:
        finite = [m[np.isfinite(m)] * scale for m in maps if np.isfinite(m).any()]
        if finite:
            allv = np.concatenate(finite)
            if signed:
                span = float(np.max(np.abs(allv))) or 1.0
                vlim = (-span, span)
            else:
                vlim = (float(allv.min()), float(allv.max()))

    for ax, zmap, ttl in zip(axes, maps, titles):
        wafer_imshow(
            ax, x, y, zmap,
            scale=scale, label=label, signed=signed, vlim=vlim, radius=radius, title=ttl,
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def annotate_stats(ax: Axes, text: str, loc: str = "lower left") -> None:
    """Place a monospace statistics box on *ax*.

    Parameters
    ----------
    ax : Axes
        Target axes.
    text : str
        Multi-line text to display.
    loc : str
        ``"lower left"``, ``"lower right"``, ``"upper left"`` or ``"upper right"``.
    """
    positions = {
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
    }
    px, py, ha, va = positions[loc]
    ax.text(
        px, py, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=8.5,
        family="monospace",
        color=INK_PRIMARY,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.4"},
    )
