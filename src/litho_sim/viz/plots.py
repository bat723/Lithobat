"""
Static figures: aerial images, resist profiles, Bossung curves, process windows.

Every public function returns a ``(fig, ax)`` tuple and accepts an ``ax`` to
draw into. The look comes from :mod:`litho_sim.viz.theme`; nothing here
knows a colour by its hex. Importing this module changes no global state —
call :func:`apply_style` (or :func:`theme.apply`) once per process, which the
desktop app and the CLI both do. A function that has to *create* a figure
applies the theme first, since the figure is then this module's to style.

Use :func:`save_figure` to write to disk with consistent DPI and padding.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig
from litho_sim.viz import theme
from litho_sim.viz.theme import AXIS, CMAP, INK2, MUTED, SERIES, WASH

logger = logging.getLogger(__name__)


def apply_style() -> None:
    """Install the LithoPy figure theme. Alias of :func:`litho_sim.viz.theme.apply`."""
    theme.apply()


def _own_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, making a themed figure when none was given."""
    if ax is not None:
        fig = ax.figure
        return (fig if isinstance(fig, Figure) else fig.figure), ax
    apply_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


# ---------------------------------------------------------------------------
# Aerial image & resist
# ---------------------------------------------------------------------------


def plot_aerial_image(
    aerial: NDArray[np.float64],
    grid: GridConfig,
    title: str = "Aerial image",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a 2-D aerial image on physical axes with a labelled intensity scale.

    Parameters
    ----------
    aerial : NDArray
        2-D intensity array (values in [0, 1]).
    grid : GridConfig
        Grid configuration (used for axis scaling).
    title : str
        Figure title — a name, not a readout.
    ax : Axes, optional
        Existing axes to draw on.  A new figure is created if not given.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    fig, ax = _own_axes(ax, (5.2, 4.4))
    half_nm = grid.grid_size / 2 * 1e9
    theme.physical_image(
        ax, aerial, (-half_nm, half_nm, -half_nm, half_nm), CMAP.intensity,
        vmin=0.0, vmax=float(aerial.max()) or 1.0, cbar_label="intensity [a.u.]",
    )
    theme.title(ax, title)
    return fig, ax


def plot_resist_profile(
    aerial: NDArray[np.float64],
    resist_image: NDArray[np.float64],
    grid: GridConfig,
    threshold: float = 0.5,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
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
        Resist development threshold, drawn as a labelled rule.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    fig, ax = _own_axes(ax, (7.5, 3.4))

    n = grid.n_pixels
    x_nm = (np.arange(n) - n // 2) * grid.pixel_size * 1e9
    profile_aerial = aerial[n // 2, :]
    profile_resist = resist_image[n // 2, :]

    ax.fill_between(x_nm, 0, profile_resist, color=SERIES.resist, alpha=0.18,
                    lw=0, label="resist remains")
    ax.plot(x_nm, profile_aerial, color=SERIES.aerial, label="aerial image")
    theme.rule(ax, y=threshold, text=f"threshold {threshold:.2f}",
               color=SERIES.threshold, ls="--", lw=1.0)
    ax.set_xlim(float(x_nm[0]), float(x_nm[-1]))
    ax.set_ylim(-0.02, 1.12)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("intensity")
    theme.grid(ax)
    theme.legend(ax, where="top")
    theme.title(ax, "Aerial image and resist profile")
    return fig, ax


# ---------------------------------------------------------------------------
# Bossung curves
# ---------------------------------------------------------------------------


def _spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Push label heights apart so none overlaps, keeping their order."""
    order = np.argsort(ys)
    placed: list[float] = []
    out = [0.0] * len(ys)
    for k in order:
        y = ys[k]
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        out[k] = y
    return out


def plot_bossung_curves(
    bossung_df: pd.DataFrame,
    target_cd_nm: float,
    tolerance_pct: float = 10.0,
    ax: Axes | None = None,
    highlight_best_focus: bool = True,
    best_focus_nm: float | None = None,
) -> tuple[Figure, Axes]:
    """Plot Bossung curves: CD vs. defocus for each dose level.

    Doses step along one hue, light to dark, and each curve is named where it
    ends rather than in a legend box. A CD of zero means nothing printed and
    is drawn as a gap, not as a line falling to the axis.

    Parameters
    ----------
    bossung_df : pd.DataFrame
        Output of ``analysis.sweep_dose_focus``.
    target_cd_nm : float
        Target CD drawn as a labelled rule [nm].
    tolerance_pct : float
        Tolerance band drawn as a wash behind the curves [%].
    ax : Axes, optional
        Existing axes.
    highlight_best_focus : bool
        Draw a vertical rule at the best focus.
    best_focus_nm : float, optional
        Where that rule goes — pass ``compute_process_window``'s
        ``best_focus_nm``. Defaults to 0 for callers that predate the best
        focus being computed.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    fig, ax = _own_axes(ax, (8, 5))

    doses = sorted(bossung_df["dose"].unique())
    colors = theme.ordinal_colors(len(doses))
    ends: list[tuple[float, float, str, str]] = []
    for color, dose in zip(colors, doses):
        sub = bossung_df[np.isclose(bossung_df["dose"], dose)].sort_values("defocus_nm")
        x = sub["defocus_nm"].to_numpy(dtype=float)
        y = sub["cd_nm"].to_numpy(dtype=float)
        y = np.where(y > 0.0, y, np.nan)
        ax.plot(x, y, color=color, lw=1.8)
        valid = np.flatnonzero(np.isfinite(y))
        if valid.size:
            i = valid[-1]
            ends.append((float(x[i]), float(y[i]), f"{dose:.2f}", color))

    lo = target_cd_nm * (1 - tolerance_pct / 100)
    hi = target_cd_nm * (1 + tolerance_pct / 100)
    ax.axhspan(lo, hi, color=WASH, lw=0, zorder=0)
    theme.rule(ax, y=target_cd_nm, color=MUTED, lw=1.0,
               text=f"target {target_cd_nm:.0f} nm ±{tolerance_pct:.0f} %")

    if highlight_best_focus:
        bf = 0.0 if best_focus_nm is None else float(best_focus_nm)
        if np.isfinite(bf):
            theme.rule(ax, x=bf, color=MUTED, lw=0.8, text=f"best focus {bf:.0f} nm")

    if ends:
        x_all = bossung_df["defocus_nm"].to_numpy(dtype=float)
        ax.set_xlim(float(x_all.min()), float(x_all.max()))
        y0, y1 = ax.get_ylim()
        ys = _spread_labels([e[1] for e in ends], 0.045 * (y1 - y0))
        x_pad = 0.012 * (x_all.max() - x_all.min())
        for (x, _y, text, color), y in zip(ends, ys):
            ax.text(x + x_pad, y, text, color=color, fontsize=8, ha="left",
                    va="center", clip_on=False)
        ax.annotate("dose", xy=(1.0, 1.0), xycoords="axes fraction",
                    xytext=(4, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8, color=MUTED)

    ax.set_xlabel("defocus [nm]")
    ax.set_ylabel("printed CD [nm]")
    theme.grid(ax)
    theme.title(ax, "Bossung curves")
    return fig, ax


# ---------------------------------------------------------------------------
# Process window
# ---------------------------------------------------------------------------


def plot_process_window(
    pw_results: dict,
    target_cd_nm: float,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the 2-D lithographic process window as a dose vs. defocus map.

    Parameters
    ----------
    pw_results : dict
        Output of ``analysis.compute_process_window``.
    target_cd_nm : float
        Named in the caption.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    fig, ax = _own_axes(ax, (7, 5))

    df = pw_results["window_df"]
    in_spec_df = df[df["in_spec"]]
    out_spec_df = df[~df["in_spec"]]

    ax.scatter(out_spec_df["defocus_nm"], out_spec_df["dose"], c=AXIS, s=18,
               lw=0, label="out of spec", zorder=2)
    ax.scatter(in_spec_df["defocus_nm"], in_spec_df["dose"], c=SERIES.aerial,
               s=46, edgecolors=theme.SURFACE, linewidths=1.2, label="in spec",
               zorder=3)

    bfn = float(pw_results["best_focus_nm"])
    bd = float(pw_results["best_dose"])
    theme.rule(ax, x=bfn, color=MUTED, lw=0.8, text=f"best focus {bfn:.0f} nm")
    theme.rule(ax, y=bd, color=MUTED, lw=0.8, text=f"best dose {bd:.3f}")

    el = pw_results["EL_pct"]
    dof = pw_results["DOF_nm"]
    ax.set_xlabel("defocus [nm]")
    ax.set_ylabel("normalised dose")
    theme.legend(ax, where="top")
    theme.title(
        ax, "Process window",
        caption_text=(f"target {target_cd_nm:.0f} nm  ·  EL {el:.1f} %  ·  "
                      f"DOF {dof:.0f} nm  ·  area {pw_results['area']:.0f} %·nm"),
    )
    return fig, ax


def plot_el_dof_curve(
    el_dof_df: pd.DataFrame,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
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
    fig, ax = _own_axes(ax, (6, 5))

    if len(el_dof_df):
        ax.plot(el_dof_df["dof_nm"], el_dof_df["el_pct"], color=SERIES.aerial, zorder=3)
        ax.set_ylim(bottom=0.0)
    else:
        theme.placeholder(ax, "no in-spec window")

    ax.set_xlabel("depth of focus [nm]")
    ax.set_ylabel("exposure latitude [%]")
    theme.grid(ax)
    theme.title(ax, "Exposure latitude against depth of focus")
    return fig, ax


def plot_nils_through_focus(
    bossung_df: pd.DataFrame,
    dose: float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
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
    fig, ax = _own_axes(ax, (6, 5))

    available = np.sort(bossung_df["dose"].unique())
    want = 1.0 if dose is None else float(dose)
    at = float(available[np.argmin(np.abs(available - want))])
    slice_df = bossung_df[np.isclose(bossung_df["dose"], at)].sort_values("defocus_nm")

    ax.plot(slice_df["defocus_nm"], slice_df["nils"], color=SERIES.aerial, zorder=3)
    theme.rule(ax, y=2.0, color=MUTED, lw=1.0, text="NILS 2 — robust printing")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("defocus [nm]")
    ax.set_ylabel("NILS")
    theme.grid(ax)
    theme.title(ax, "NILS through focus", caption_text=f"dose {at:.3f}")
    return fig, ax


# ---------------------------------------------------------------------------
# CD heatmap
# ---------------------------------------------------------------------------


def plot_cd_heatmap(
    bossung_df: pd.DataFrame,
    target_cd_nm: float,
    tolerance_pct: float = 10.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the CD map over the dose × defocus grid.

    Diverging about the target — thin in blue, on target in the neutral
    wash, fat in red — with the in-spec window outlined by one hairline.

    Parameters
    ----------
    bossung_df : pd.DataFrame
        Output of ``analysis.sweep_dose_focus``.
    target_cd_nm : float
        Target CD, the colour scale's midpoint [nm].
    tolerance_pct : float
        Tolerance band, outlined [%].
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    from litho_sim.analysis.data_utils import pivot_cd_matrix

    fig, ax = _own_axes(ax, (8, 5))

    pivot = pivot_cd_matrix(bossung_df)
    defoci = pivot.index.to_numpy(dtype=float)
    doses = pivot.columns.to_numpy(dtype=float)
    data = pivot.to_numpy(dtype=float)

    span = target_cd_nm * tolerance_pct / 100 * 2
    # Cell edges, so each sweep point owns the cell centred on it.
    def _edges(v: np.ndarray) -> tuple[float, float]:
        if v.size < 2:
            return float(v[0]) - 0.5, float(v[0]) + 0.5
        d = float(v[1] - v[0])
        return float(v[0]) - d / 2, float(v[-1]) + d / 2

    x0, x1 = _edges(doses)
    y0, y1 = _edges(defoci)
    theme.physical_image(
        ax, data, (x0, x1, y0, y1), CMAP.cd,
        vmin=target_cd_nm - span, vmax=target_cd_nm + span,
        cbar_label="CD [nm]", aspect="auto",
        xlabel="normalised dose", ylabel="defocus [nm]",
    )
    lo = target_cd_nm * (1 - tolerance_pct / 100)
    hi = target_cd_nm * (1 + tolerance_pct / 100)
    if doses.size > 1 and defoci.size > 1 and np.isfinite(data).any():
        ax.contour(doses, defoci, np.nan_to_num(data), levels=[lo, hi],
                   colors=[INK2], linewidths=0.8)
    theme.title(ax, "CD map",
                caption_text=f"target {target_cd_nm:.0f} nm ±{tolerance_pct:.0f} %")
    return fig, ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OPC
# ---------------------------------------------------------------------------


def _outline(ax, layout, color, lw, label=None, layer=None):
    """Draw every shape of a layout as a closed outline, in nanometres."""
    shapes = layout.shapes if layer is None else layout.on_layer(layer)
    first = True
    for sh in shapes:
        p = np.asarray(sh.polygon()) * 1e9
        p = np.vstack([p, p[:1]])
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, label=label if first else None,
                solid_joinstyle="miter")
        first = False


def _contours(ax, printed, color, lw, label=None):
    first = True
    for line in printed.contours():
        line = line * 1e9
        ax.plot(line[:, 0], line[:, 1], color=color, lw=lw, label=label if first else None)
        first = False


def plot_opc(result, axes=None, clip_nm: tuple[float, float, float, float] | None = None):
    """The three pictures an OPC run is judged by.

    Left: the drawn design over the corrected mask, so the jogs, serifs and
    hammerheads the loop grew are visible against what was asked for.
    Middle: the design with what it printed before correction and after.
    Right: the edge placement error at every fragment, before and after,
    sorted — the distribution the loop shrank, with the tolerance drawn in.

    Parameters
    ----------
    result : litho_sim.opc.OPCResult
        A finished run.
    axes : sequence of three Axes, optional
        Existing axes to draw into; a new figure otherwise.
    clip_nm : (x0, x1, y0, y1), optional
        Axis limits for the two layout panels [nm]; defaults to the
        design's bounds with a margin.

    Returns
    -------
    fig, axes : Figure, tuple of Axes
    """
    if axes is None:
        apply_style()
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    else:
        fig = axes[0].figure
    ax_mask, ax_print, ax_epe = axes

    design, corrected = result.design, result.corrected
    if clip_nm is None:
        b = np.array([sh.bounds() for sh in design.shapes]) * 1e9
        x0, y0 = b[:, 0].min(), b[:, 1].min()
        x1, y1 = b[:, 2].max(), b[:, 3].max()
        m = 0.08 * max(x1 - x0, y1 - y0)
        clip_nm = (x0 - m, x1 + m, y0 - m, y1 + m)

    # --- the mask ---------------------------------------------------------
    layer = result.settings.get("layer")
    _outline(ax_mask, design, SERIES.reference, 0.9, "design", layer=layer)
    if "sraf" in corrected.layers():
        _outline(ax_mask, corrected, MUTED, 0.9, "assist features", layer="sraf")
    _outline(ax_mask, corrected, SERIES.warn, 1.0, "corrected mask", layer=layer)
    ax_mask.set_aspect("equal")
    ax_mask.set_xlim(clip_nm[0], clip_nm[1])
    ax_mask.set_ylim(clip_nm[2], clip_nm[3])
    ax_mask.set_xlabel("x [nm]")
    ax_mask.set_ylabel("y [nm]")
    offs = np.abs(result.offsets) * 1e9
    theme.legend(ax_mask, where="top", ncol=2)
    theme.title(
        ax_mask, "Corrected mask",
        caption_text=f"{len(offs)} fragments · largest move {offs.max():.1f} nm",
    )

    # --- what printed -----------------------------------------------------
    _outline(ax_print, design, SERIES.reference, 0.9, "design", layer=layer)
    _contours(ax_print, result.before, SERIES.warn, 1.0, "printed, uncorrected")
    _contours(ax_print, result.after, SERIES.resist, 1.3, "printed, corrected")
    ax_print.set_aspect("equal")
    ax_print.set_xlim(clip_nm[0], clip_nm[1])
    ax_print.set_ylim(clip_nm[2], clip_nm[3])
    ax_print.set_xlabel("x [nm]")
    ax_print.set_ylabel("y [nm]")
    theme.legend(ax_print, where="top", ncol=1)
    s = result.settings
    theme.title(
        ax_print, "Printed contour",
        caption_text=f"dose {s.get('dose', float('nan')):.3f} · {s.get('model', '')} resist · "
                     f"{len(result.history) - 1} iterations"
                     + ("" if result.converged else " (not converged)"),
    )

    # --- the EPE distribution ---------------------------------------------
    eb = np.abs(result.epe_before.values) * 1e9
    ea = np.abs(result.epe_after.values) * 1e9
    eb = np.sort(eb[np.isfinite(eb)])
    ea = np.sort(ea[np.isfinite(ea)])
    ax_epe.plot(np.arange(len(eb)), eb, color=SERIES.warn, lw=1.2, label="before")
    ax_epe.plot(np.arange(len(ea)), ea, color=SERIES.resist, lw=1.4, label="after")
    theme.rule(ax_epe, y=s.get("tol", 1e-9) * 1e9, text="tolerance", color=SERIES.threshold)
    ax_epe.set_xlabel("fragment, sorted by |EPE|")
    ax_epe.set_ylabel("|EPE| [nm]")
    ax_epe.set_ylim(bottom=0)
    theme.grid(ax_epe)
    theme.legend(ax_epe, where="top")
    sb, sa = result.epe_before.stats(), result.epe_after.stats()
    theme.title(
        ax_epe, "Edge placement error",
        caption_text=f"max {sb['max_abs_epe_nm']:.1f} to {sa['max_abs_epe_nm']:.1f} nm · "
                     f"rms {sb['rms_epe_nm']:.1f} to {sa['rms_epe_nm']:.1f} nm · "
                     f"edges not found {sb['n_failed']} to {sa['n_failed']}",
    )
    return fig, tuple(axes)


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
