"""
Visualisation module for LithoPy.

All public functions return a ``(fig, axes)`` tuple.  Use
:func:`save_figure` to write to disk with consistent DPI and padding.

Consistent plot style is applied by calling :func:`apply_style` once
(automatically called on module import).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style definition
# ---------------------------------------------------------------------------

_LITHO_RCPARAMS = {
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": "#CCCCCC",
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
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "image.cmap": "inferno",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
}

# Discrete color palette for multi-dose / multi-pitch curves
_PALETTE = [
    "#1F77B4",  # blue
    "#FF7F0E",  # orange
    "#2CA02C",  # green
    "#D62728",  # red
    "#9467BD",  # purple
    "#8C564B",  # brown
    "#E377C2",  # pink
    "#7F7F7F",  # grey
]


def apply_style() -> None:
    """Apply the LithoPy matplotlib style globally."""
    matplotlib.rcParams.update(_LITHO_RCPARAMS)


apply_style()  # auto-apply on import


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


def _label_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)


# ---------------------------------------------------------------------------
# Aerial image & resist
# ---------------------------------------------------------------------------


def plot_aerial_image(
    aerial: NDArray[np.float64],
    grid: GridConfig,
    title: str = "Aerial Image",
    ax: plt.Axes | None = None,
) -> tuple[Figure, plt.Axes]:
    """Plot a 2-D aerial image with a physical-units axis.

    Parameters
    ----------
    aerial : NDArray
        2-D intensity array (values in [0, 1]).
    grid : GridConfig
        Grid configuration (used for axis scaling).
    title : str
        Figure title.
    ax : Axes, optional
        Existing axes to draw on.  A new figure is created if not given.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    half_nm = grid.grid_size / 2 * 1e9
    extent = [-half_nm, half_nm, -half_nm, half_nm]

    im = ax.imshow(
        aerial,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=aerial.max() or 1.0,
        cmap="inferno",
        aspect="equal",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Intensity [a.u.]", shrink=0.85)
    _label_axes(ax, "x [nm]", "y [nm]", title)
    return fig, ax


def plot_resist_profile(
    aerial: NDArray[np.float64],
    resist_image: NDArray[np.float64],
    grid: GridConfig,
    threshold: float = 0.5,
    ax: plt.Axes | None = None,
) -> tuple[Figure, plt.Axes]:
    """Overlay the 1-D aerial-image cross-section with the resist edge.

    Parameters
    ----------
    aerial : NDArray
        2-D aerial image.
    resist_image : NDArray
        2-D binary resist image.
    grid : GridConfig
        Grid configuration.
    threshold : float
        Resist development threshold drawn as a horizontal dashed line.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = ax.figure

    n = grid.n_pixels
    x_nm = (np.arange(n) - n // 2) * grid.pixel_size * 1e9

    profile_aerial = aerial[n // 2, :]
    profile_resist = resist_image[n // 2, :]

    ax.plot(x_nm, profile_aerial, color=_PALETTE[0], label="Aerial image")
    ax.fill_between(x_nm, 0, profile_resist, alpha=0.30, color=_PALETTE[2],
                    label="Resist (remains)")
    ax.axhline(threshold, color="k", linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold:.2f}")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper right")
    _label_axes(ax, "x [nm]", "Intensity / Resist", "Aerial Image & Resist Profile")
    return fig, ax


# ---------------------------------------------------------------------------
# Bossung curves
# ---------------------------------------------------------------------------


def plot_bossung_curves(
    bossung_df: pd.DataFrame,
    target_cd_nm: float,
    tolerance_pct: float = 10.0,
    ax: plt.Axes | None = None,
    highlight_best_focus: bool = True,
    best_focus_nm: float | None = None,
) -> tuple[Figure, plt.Axes]:
    """Plot Bossung curves: CD vs. defocus for each dose level.

    Parameters
    ----------
    bossung_df : pd.DataFrame
        Output of ``analysis.sweep_dose_focus``.
    target_cd_nm : float
        Target CD drawn as a solid horizontal line [nm].
    tolerance_pct : float
        Tolerance band drawn as dashed horizontal lines [%].
    ax : Axes, optional
        Existing axes.
    highlight_best_focus : bool
        Draw a vertical dashed line at the best focus.
    best_focus_nm : float, optional
        Where that line goes — pass ``compute_process_window``'s
        ``best_focus_nm``. Defaults to 0 for callers that predate the best
        focus being computed.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    doses = sorted(bossung_df["dose"].unique())
    for idx, dose in enumerate(doses):
        sub = bossung_df[np.isclose(bossung_df["dose"], dose)].sort_values("defocus_nm")
        ax.plot(
            sub["defocus_nm"],
            sub["cd_nm"],
            color=_pick_color(idx),
            marker="o",
            markersize=4,
            label=f"Dose = {dose:.2f}",
        )

    # Tolerance band
    lo = target_cd_nm * (1 - tolerance_pct / 100)
    hi = target_cd_nm * (1 + tolerance_pct / 100)
    ax.axhline(target_cd_nm, color="black", linewidth=1.5, linestyle="-",
               label=f"Target {target_cd_nm:.0f} nm")
    ax.axhline(lo, color="black", linewidth=0.9, linestyle="--",
               label=f"±{tolerance_pct:.0f}% band")
    ax.axhline(hi, color="black", linewidth=0.9, linestyle="--")
    ax.fill_between(
        ax.get_xlim() or [-500, 500],
        lo, hi,
        color="gray", alpha=0.08,
    )

    if highlight_best_focus:
        bf = 0.0 if best_focus_nm is None else float(best_focus_nm)
        if np.isfinite(bf):
            ax.axvline(bf, color="#888888", linewidth=0.8, linestyle=":",
                       label=f"Best focus ({bf:.0f} nm)")

    ax.legend(loc="upper right", ncol=2)
    _label_axes(ax, "Defocus [nm]", "Printed CD [nm]", "Bossung Curves")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    return fig, ax


# ---------------------------------------------------------------------------
# Process window
# ---------------------------------------------------------------------------


def plot_process_window(
    pw_results: dict,
    target_cd_nm: float,
    ax: plt.Axes | None = None,
) -> tuple[Figure, plt.Axes]:
    """Plot the 2-D lithographic process window as a dose vs. defocus map.

    Parameters
    ----------
    pw_results : dict
        Output of ``analysis.compute_process_window``.
    target_cd_nm : float
        Used for the plot title annotation.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    df = pw_results["window_df"]
    in_spec_df = df[df["in_spec"]]
    out_spec_df = df[~df["in_spec"]]

    ax.scatter(
        out_spec_df["defocus_nm"],
        out_spec_df["dose"],
        c="#DDDDDD",
        s=30,
        marker="o",
        label="Out of spec",
        zorder=2,
    )
    ax.scatter(
        in_spec_df["defocus_nm"],
        in_spec_df["dose"],
        c=_PALETTE[0],
        s=60,
        marker="o",
        edgecolors="white",
        linewidths=0.4,
        label="In spec",
        zorder=3,
    )

    # Mark best focus / best dose
    bfn = pw_results["best_focus_nm"]
    bd = pw_results["best_dose"]
    ax.axvline(bfn, color="#FF7F0E", linestyle="--", linewidth=1.2,
               label=f"Best focus = {bfn:.0f} nm")
    ax.axhline(bd, color="#2CA02C", linestyle="--", linewidth=1.2, label=f"Best dose = {bd:.3f}")

    el = pw_results["EL_pct"]
    dof = pw_results["DOF_nm"]
    ax.set_title(
        f"Process Window  |  Target CD={target_cd_nm:.0f} nm\n"
        f"EL={el:.1f}%   DOF={dof:.0f} nm   Area={pw_results['area']:.0f} %·nm",
        pad=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    _label_axes(ax, "Defocus [nm]", "Normalised Dose", "")
    return fig, ax


def plot_el_dof_curve(
    el_dof_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> tuple[Figure, plt.Axes]:
    """Plot one process's EL-vs-DOF trade-off curve.

    Draws the falling curve a single sweep traces as the focus-window demand
    widens — the shape a process engineer trades along.

    Parameters
    ----------
    el_dof_df : pd.DataFrame
        Output of ``analysis.compute_el_dof_curve`` — columns ``dof_nm``,
        ``el_pct``.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    if len(el_dof_df):
        ax.plot(
            el_dof_df["dof_nm"], el_dof_df["el_pct"],
            color=_pick_color(0), marker="o", markersize=4, zorder=3,
        )
        ax.set_ylim(bottom=0.0)
    else:
        ax.text(0.5, 0.5, "no in-spec window", transform=ax.transAxes,
                ha="center", va="center", color="#7F7F7F")

    _label_axes(ax, "Depth of Focus [nm]", "Exposure Latitude [%]",
                "EL available at each DOF demand")
    return fig, ax


def plot_nils_through_focus(
    bossung_df: pd.DataFrame,
    dose: float | None = None,
    ax: plt.Axes | None = None,
) -> tuple[Figure, plt.Axes]:
    """Plot NILS against defocus at one dose, with the NILS = 2 rule drawn in.

    Parameters
    ----------
    bossung_df : pd.DataFrame
        Output of ``analysis.sweep_dose_focus`` (needs its ``nils`` column).
    dose : float, optional
        Dose slice to show. Defaults to the sweep dose nearest 1.0 — the
        nominal, or the dose-to-size anchor if the sweep was centred on one.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    available = np.sort(bossung_df["dose"].unique())
    want = 1.0 if dose is None else float(dose)
    at = float(available[np.argmin(np.abs(available - want))])
    slice_df = bossung_df[np.isclose(bossung_df["dose"], at)].sort_values("defocus_nm")

    ax.plot(
        slice_df["defocus_nm"], slice_df["nils"],
        color=_pick_color(0), marker="o", markersize=4, zorder=3,
    )
    ax.axhline(2.0, color="#7F7F7F", linestyle="--", linewidth=1.0, zorder=2)
    ax.annotate("NILS 2 — robust printing", xy=(0.02, 2.0),
                xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 4),
                fontsize=8, color="#7F7F7F")
    ax.set_ylim(bottom=0.0)
    _label_axes(ax, "Defocus [nm]", "NILS",
                f"NILS through focus  (dose {at:.3f})")
    return fig, ax


# ---------------------------------------------------------------------------
# CD heatmap
# ---------------------------------------------------------------------------


def plot_cd_heatmap(
    bossung_df: pd.DataFrame,
    target_cd_nm: float,
    tolerance_pct: float = 10.0,
    ax: plt.Axes | None = None,
) -> tuple[Figure, plt.Axes]:
    """Plot a colour-coded CD heatmap over the dose × defocus grid.

    Parameters
    ----------
    bossung_df : pd.DataFrame
        Output of ``analysis.sweep_dose_focus``.
    target_cd_nm : float
        Target CD for centre-relative colour scaling [nm].
    tolerance_pct : float
        Tolerance band highlighted in green.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    from litho_sim.analysis.data_utils import pivot_cd_matrix

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    pivot = pivot_cd_matrix(bossung_df)
    defoci = pivot.index.to_numpy()
    doses = pivot.columns.to_numpy()

    data = pivot.to_numpy()
    vmax = target_cd_nm * (1 + tolerance_pct / 100 * 2)
    vmin = target_cd_nm * (1 - tolerance_pct / 100 * 2)

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[doses[0], doses[-1], defoci[0], defoci[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="RdYlGn_r",
        interpolation="bilinear",
    )
    fig.colorbar(im, ax=ax, label="CD [nm]", shrink=0.85)
    _label_axes(ax, "Normalised Dose", "Defocus [nm]", f"CD Map  (target={target_cd_nm:.0f} nm)")
    return fig, ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_figure(
    fig: Figure,
    path: str | Path,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    path : str or Path
        Destination path (extension determines format: PNG, PDF, SVG…).
    dpi : int
        Output resolution.
    close : bool
        Close the figure after saving to free memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved → %s", path)
    if close:
        plt.close(fig)

