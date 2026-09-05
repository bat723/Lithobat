"""Pictures of OPC in action — the mechanism, the iterations, the light, the numbers.

`demo_opc.py` writes the three-panel verdict. This writes the pictures that
show *how* the verdict came about, on the same layout at the same dose:

* ``opc_mechanism.png`` — one line end: the fragments, their measurement
  sites (retargeted at the corners), the edge placement error of each as an
  arrow to the printed edge, and the hammerhead the loop grew from them.
* ``opc_iterations.png`` — the tip-to-tip gap at iterations 0, 1, 2, 3, 6
  and 12: the mask and what it prints, converging.
* ``opc_aerial.png`` — the aerial image of the uncorrected and the corrected
  mask, with the design and the printed contour on each.
* ``opc_convergence.png`` — worst and RMS EPE per iteration, and four named
  fragments' own EPE traces.

    python scripts/opc_gallery.py            # → results/opc_*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from litho_sim.cli.opc import build_layout  # noqa: E402
from litho_sim.core.config import GridConfig, SimulationConfig  # noqa: E402
from litho_sim.mask import Layout, line_array  # noqa: E402
from litho_sim.opc import PrintModel, run_opc  # noqa: E402
from litho_sim.opc.fragments import rebuild_layout  # noqa: E402
from litho_sim.viz import theme  # noqa: E402
from litho_sim.viz.plots import _contours, _outline, save_figure  # noqa: E402
from litho_sim.viz.theme import BLUE, CMAP, INK2, ORANGE, SERIES, VIOLET  # noqa: E402

ROLE_COLOUR = {"body": BLUE, "corner": ORANGE, "end": VIOLET}


def setup(pitch_nm: float = 200.0, cd_nm: float = 100.0, pixels: int = 256):
    cfg = SimulationConfig.from_tech_node("ArF")
    px = 6.4 * pitch_nm * 1e-9 / pixels
    grid = GridConfig(n_pixels=pixels, pixel_size=px)
    pitch, cd = pitch_nm * 1e-9, cd_nm * 1e-9
    design = build_layout(pitch, cd, 0.7 * grid.grid_size)
    model = PrintModel(cfg.optics, cfg.resist, grid, tone="dark")
    anchor = Layout(line_array(7, pitch=pitch, cd=cd, length=0.9 * grid.grid_size))
    model.dose = model.dose_to_size(anchor, cd_nm)
    return design, model, run_opc(design, model, layer="main")


def _layout_at(result, k: int) -> Layout:
    """The corrected layout as it stood at iteration *k*."""
    offs = result.offset_trace[k]
    frags = result.fragmented
    i = 0
    for fs in frags:
        fs.offsets[:] = offs[i:i + len(fs)]
        i += len(fs)
    lay = rebuild_layout(frags, name=f"iteration {k}")
    # Restore the final offsets so the result object is unchanged afterwards.
    final = result.offset_trace[-1]
    i = 0
    for fs in frags:
        fs.offsets[:] = final[i:i + len(fs)]
        i += len(fs)
    return lay


def _window(ax, x0, x1, y0, y1):
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")


# ---------------------------------------------------------------------------
# 1. The mechanism, on one line end
# ---------------------------------------------------------------------------


def fig_mechanism(result, shape_index: int, out: Path) -> None:
    fs = result.fragmented[shape_index]
    start = sum(len(f) for f in result.fragmented[:shape_index])
    epe = result.epe_before.values[start:start + len(fs)]
    x0, y0, x1, y1 = fs.source.bounds()
    # Zoom on the top end of this shape.
    cx = 0.5 * (x0 + x1) * 1e9
    top = y1 * 1e9
    win = (cx - 110, cx + 110, top - 120, top + 70)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)

    # --- left: fragments, sites, EPE arrows, against the uncorrected print
    ax = axes[0]
    _contours(ax, result.before, SERIES.warn, 1.1, "printed, as drawn")
    seen = set()
    for i, f in enumerate(fs.fragments):
        p0, p1 = f.p0 * 1e9, f.p1 * 1e9
        label = None if f.role in seen else f"{f.role} fragment"
        seen.add(f.role)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=ROLE_COLOUR[f.role], lw=2.6,
                solid_capstyle="butt", label=label)
        site = f.control_point * 1e9
        n = f.control_normal
        if f.retargeted:
            mid = f.midpoint * 1e9
            ax.plot([mid[0], site[0]], [mid[1], site[1]], color=INK2, lw=0.6, ls=":")
        ax.plot(site[0], site[1], "o", ms=3.2, color=INK2, zorder=5)
        e = epe[i]
        if np.isfinite(e) and abs(e) > 0.5e-9:
            ax.annotate(
                "", xy=(site[0] + n[0] * e * 1e9, site[1] + n[1] * e * 1e9), xytext=site,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0),
                zorder=6,
            )
    # Small perpendicular ticks between fragments make the cuts visible.
    _window(ax, *win)
    theme.legend(ax, where="below", ncol=3)
    n_ok = int(np.isfinite(epe).sum())
    theme.title(
        ax, "Fragments, sites and their EPE",
        caption_text=f"{len(fs)} fragments on this shape · corner sites on the arc · "
                     f"arrows: site to printed edge, worst {np.nanmax(np.abs(epe)) * 1e9:.1f} nm, "
                     f"{n_ok} of {len(fs)} found",
    )

    # --- right: the correction that came out of it
    ax = axes[1]
    _outline(ax, result.design, SERIES.reference, 1.0, "design", layer="main")
    _contours(ax, result.before, SERIES.warn, 0.9, "printed, as drawn")
    _outline(ax, result.corrected, SERIES.warn, 1.6, "corrected mask", layer="main")
    _contours(ax, result.after, SERIES.resist, 1.6, "printed, corrected")
    _window(ax, *win)
    theme.legend(ax, where="below", ncol=4)
    offs = np.abs(fs.offsets) * 1e9
    after = result.epe_after.values[start:start + len(fs)]
    theme.title(
        ax, "The hammerhead the loop grew",
        caption_text=f"largest move on this shape {offs.max():.1f} nm · "
                     f"worst EPE here {np.nanmax(np.abs(epe)) * 1e9:.1f} to "
                     f"{np.nanmax(np.abs(after)) * 1e9:.1f} nm",
    )
    save_figure(fig, out)


# ---------------------------------------------------------------------------
# 2. Iterations, on the tip-to-tip gap
# ---------------------------------------------------------------------------


def fig_iterations(result, model, centre_x_nm: float, out: Path,
                   ks=(0, 1, 2, 3, 6, 12)) -> None:
    ks = [k for k in ks if k < len(result.offset_trace)]
    fig, axes = plt.subplots(1, len(ks), figsize=(2.6 * len(ks), 4.4), constrained_layout=True)
    win = (centre_x_nm - 120, centre_x_nm + 120, -170, 170)
    for ax, k in zip(axes, ks):
        lay = _layout_at(result, k)
        printed = model.print(lay)
        _outline(ax, result.design, SERIES.reference, 0.9, "design", layer="main")
        _outline(ax, lay, SERIES.warn, 1.3, "mask", layer="main")
        _contours(ax, printed, SERIES.resist, 1.4, "printed")
        _window(ax, *win)
        if k != ks[0]:
            ax.set_ylabel("")
        row = result.history.iloc[k]
        theme.title(ax, f"Iteration {k}",
                    caption_text=f"worst {row.max_abs_epe_nm:.1f} nm · rms {row.rms_epe_nm:.1f} nm")
    theme.legend(axes[len(axes) // 2], where="below", ncol=3)
    theme.caption(fig, "the tip-to-tip gap, one CD wide: the mask (red) and what it prints (green) "
                       "against the design (grey), as the loop runs")
    save_figure(fig, out)


# ---------------------------------------------------------------------------
# 3. The aerial images
# ---------------------------------------------------------------------------


def fig_aerial(result, out: Path) -> None:
    grid = result.before.grid
    half = grid.grid_size / 2 * 1e9
    vmax = max(float(result.before.aerial.max()), float(result.after.aerial.max()))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    for ax, printed, name in ((axes[0], result.before, "Aerial image, mask as drawn"),
                              (axes[1], result.after, "Aerial image, corrected mask")):
        theme.physical_image(ax, printed.aerial, (-half, half, -half, half), CMAP.intensity,
                             vmin=0.0, vmax=vmax, cbar_label="intensity, clear field = 1")
        _outline(ax, result.design, SERIES.reference, 0.8, "design", layer="main")
        _contours(ax, printed, SERIES.resist, 1.1, "printed edge")
        stats = (result.epe_before if printed is result.before else result.epe_after).stats()
        theme.title(ax, name, caption_text=f"dose {result.settings['dose']:.3f} · "
                                           f"worst EPE {stats['max_abs_epe_nm']:.1f} nm · "
                                           f"rms {stats['rms_epe_nm']:.1f} nm")
    theme.legend(axes[1], where="top")
    save_figure(fig, out)


# ---------------------------------------------------------------------------
# 4. Convergence
# ---------------------------------------------------------------------------


def _pick(df, **query):
    sub = df
    for k, v in query.items():
        sub = sub[v(sub[k])]
    return int(sub.index[0]) if len(sub) else None


def fig_convergence(result, out: Path, tip_x_nm: float, t_x_nm: float) -> None:
    h = result.history
    df = result.summary()
    # Shapes: 0-2 the dense lines, 3 the upper and 4 the lower tip-to-tip
    # segment, 5 the T. The lower segment's top edge is the facing tip.
    picks = {
        "line-end body (tip)": _pick(df, shape=lambda s: s == 4, role=lambda r: r == "body",
                                     ny=lambda v: v > 0.5),
        "line-end corner (tip)": _pick(df, shape=lambda s: s == 4, role=lambda r: r == "corner",
                                       y_nm=lambda v: v > -80),
        "dense side body": _pick(df, shape=lambda s: s == 1, role=lambda r: r == "body",
                                 nx=lambda v: v > 0.99, y_nm=lambda v: abs(v) < 60),
        "T inner corner": _pick(df, shape=lambda s: s == 5, role=lambda r: r == "corner",
                                x_nm=lambda v: v > t_x_nm, y_nm=lambda v: abs(v) < 80),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(h.iteration, h.max_abs_epe_nm, color=SERIES.warn, lw=1.4, marker="o", ms=3,
            label="worst |EPE|")
    ax.plot(h.iteration, h.rms_epe_nm, color=SERIES.resist, lw=1.4, marker="o", ms=3,
            label="rms EPE")
    theme.rule(ax, y=result.settings["tol"] * 1e9, text="edge tolerance", color=SERIES.threshold)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("EPE [nm]")
    theme.grid(ax)
    theme.legend(ax, where="top")
    verdict = "converged" if result.converged else "stopped on the cap"
    theme.title(ax, "Convergence", caption_text=f"{len(h) - 1} iterations · {verdict}")

    ax = axes[1]
    colours = iter(theme.CATEGORICAL)
    for name, idx in picks.items():
        if idx is None:
            continue
        ax.plot(h.iteration, result.epe_trace[:, idx] * 1e9, lw=1.3, marker="o", ms=2.6,
                color=next(colours), label=name)
    theme.rule(ax, y=0.0, color=SERIES.reference)
    ax.set_xlabel("iteration")
    ax.set_ylabel("EPE [nm], signed")
    theme.grid(ax)
    theme.legend(ax, where="below", ncol=4)
    theme.title(ax, "Four fragments, followed",
                caption_text="negative: prints short of the drawn edge · positive: beyond it")
    save_figure(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pictures of OPC in action.")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    theme.apply()
    design, model, result = setup()
    p = 200.0
    x_pair, x_t = 0.9 * p, 2.1 * p
    print(result.stats().round(2).to_string())
    fig_mechanism(result, shape_index=4, out=a.out / "opc_mechanism.png")
    fig_iterations(result, model, x_pair, a.out / "opc_iterations.png")
    fig_aerial(result, a.out / "opc_aerial.png")
    fig_convergence(result, a.out / "opc_convergence.png", x_pair, x_t + 50)
    for name in ("opc_mechanism", "opc_iterations", "opc_aerial", "opc_convergence"):
        print(a.out / f"{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
