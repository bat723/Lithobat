"""
End-to-end demo: 3-D printing, overlay-driven pitch walking, and SADP.

    litho-sim multipatterning                 # results/multipatterning_demo.png
                                              # and results/sadp_cut_demo.png

(``scripts/demo_multipatterning.py`` is the same command.)

Everything shown here is emergent. Nothing in the engine computes "pitch
walk" or knows that a spacer doubles a pattern — those follow from imaging
shifted geometry and from a conformal film having two sidewalls.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from litho_sim.cli.common import add_output_args
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask.layout import Layout, cut_bar, line_array
from litho_sim.patterning import lele, sadp
from litho_sim.viz.viz3d import cross_section_figure, stack_figure_mpl


def sadp_cut_demo(optics: OpticsConfig, resist: ResistConfig, out: Path) -> None:
    """SADP with a cut mask: one drawn line becomes several gates.

    Three wafers side by side — no cut, litho cut (a real second print), and
    ideal cut (geometry-gated etch) — plus the 3-D result. The litho cut's
    slot prints thinner than drawn (isolated feature at a dense-calibrated
    dose), which is visible in the comparison and is the point of having
    both styles.
    """
    grid = GridConfig(n_pixels=128, pixel_size=4e-9, dz=4e-9, n_z_slices=9)
    mandrel = Layout(line_array(3, pitch=150e-9, cd=70e-9, length=460e-9),
                     name="mandrel")
    bar = Layout([cut_bar(0.0, 60e-9, 160e-9, 80e-9)], name="cut")

    print("\n=== SADP with cut mask ===")
    runs = {}
    cases: list[tuple[str, dict[str, Any]]] = [
        ("no cut", {}),
        ("litho cut", {"cut_layout": bar, "cut_style": "litho"}),
        ("ideal cut", {"cut_layout": bar, "cut_style": "ideal"}),
    ]
    for label, kwargs in cases:
        t0 = time.perf_counter()
        res = sadp(mandrel, grid, optics, resist, **kwargs).run(snapshot=False)
        runs[label] = res.stack
        m = res.measurements.iloc[-1]
        print(f"  {label:<10} {m['n_lines']:>2} features on the centre row   "
              f"[{time.perf_counter()-t0:.1f}s]")

    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    side = grid.n_pixels * grid.pixel_size * 1e9
    extent = (0.0, side, 0.0, side)
    bx, by, bw, bh = 0.0, 60.0, 160.0, 80.0  # drawn bar, nm, field-centred
    half = grid.n_pixels * grid.pixel_size * 1e9 / 2

    for i, label in enumerate(("no cut", "litho cut", "ideal cut")):
        ax = fig.add_subplot(1, 4, i + 1)
        th = np.asarray(runs[label].thickness_of("poly-Si")) * 1e9
        ax.imshow(th, origin="lower", extent=extent, cmap="cividis",
                  vmin=0, vmax=40)
        if label != "no cut":
            ax.add_patch(plt.Rectangle(
                (half + bx - bw / 2, half + by - bh / 2), bw, bh,
                fill=False, edgecolor="#d97f9b", lw=1.6, ls="--",
            ))
        ax.set(title=f"SADP, {label}", xlabel="x [nm]",
               ylabel="y [nm]" if i == 0 else None)

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    stack_figure_mpl(runs["litho cut"], z_exaggeration=2.5, downsample=2, ax=ax)
    ax.view_init(elev=32, azim=-55)
    ax.set_title("litho cut, 3-D")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


def add_parser(subs) -> None:
    ap = subs.add_parser("multipatterning",
                         help="LELE pitch walking, SADP, and SADP with a cut mask",
                         description=(__doc__ or "").split("\n\n")[0])
    add_output_args(ap)
    ap.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output)
    grid = GridConfig(n_pixels=128, pixel_size=4e-9, dz=4e-9, n_z_slices=9)
    optics = OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=15)
    resist = ResistConfig(
        n_resist=1.7, dill_A=0.8, dill_B=0.05, dill_C=0.04,
        dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9,
    )

    # A 6-line, 80 nm-pitch gate layer. Too dense for one ArF exposure.
    target = Layout(line_array(6, pitch=80e-9, cd=40e-9, length=600e-9), name="gates")

    print("\n=== LELE overlay sweep ===")
    overlays_nm = [0.0, 3.0, 6.0, 9.0, 12.0]
    walks, stacks = [], {}
    for ov in overlays_nm:
        t0 = time.perf_counter()
        flow = lele(target, grid, optics, resist, min_spacing=70e-9,
                    overlay=(ov * 1e-9, 0.0))
        res = flow.run(snapshot=False)
        m = res.measurements.iloc[-1]
        walks.append(m["pitch_walk_nm"])
        stacks[ov] = res.stack
        print(f"  overlay {ov:4.1f} nm -> CDs {[round(v) for v in m['lines_nm']]} nm, "
              f"pitch walk {m['pitch_walk_nm']:5.1f} nm   [{time.perf_counter()-t0:.1f}s]")

    print("\n=== SADP ===")
    sadp_grid = GridConfig(n_pixels=160, pixel_size=4e-9, dz=4e-9, n_z_slices=9)
    mandrel = Layout(line_array(4, pitch=160e-9, cd=80e-9, length=700e-9), name="mandrel")
    flow = sadp(mandrel, sadp_grid, optics, resist, spacer_thickness=20e-9)
    res_sadp = flow.run(snapshot=True)
    for _, row in res_sadp.measurements.iterrows():
        print(f"  {row['name']:<9} {row['n_lines']:>2} features, "
              f"CD {row['mean_line_nm']:.0f} nm")

    # Index of the snapshot just after the mandrel pull, which is the moment
    # the spacers become freestanding.
    pull_idx = next(
        i for i, s in enumerate(res_sadp.snapshots)
        if s.volume_fraction("a-C") == 0.0 and s.volume_fraction("spacer-oxide") > 0
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(overlays_nm, walks, "o-", color="#c8913a", lw=2, ms=7, label="simulated")
    ax.plot(overlays_nm, [2 * o for o in overlays_nm], "--", color="#7f7f7f",
            label="2 × overlay (theory)")
    ax.set(xlabel="Overlay error [nm]", ylabel="Pitch walk [nm]",
           title="LELE: pitch walking emerges from overlay")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for ov, color in [(0.0, "#4fd97f"), (12.0, "#d97f9b")]:
        prof = stacks[ov].line_profile("poly-Si", z_frac=0.9)
        x = np.arange(len(prof)) * grid.pixel_size * 1e9
        ax.step(x, prof.astype(float) + (0 if ov == 0 else 1.4), where="mid",
                color=color, lw=1.8, label=f"overlay {ov:.0f} nm")
    ax.set(xlabel="x [nm]", yticks=[], title="Printed gates: uniform vs. walking")
    ax.legend(fontsize=9, loc="upper right")

    ax = fig.add_subplot(gs[1, 0])
    cross_section_figure(stacks[0.0], ax=ax, title="LELE, perfect overlay")
    ax = fig.add_subplot(gs[1, 1])
    cross_section_figure(stacks[12.0], ax=ax, title="LELE, 12 nm overlay")

    ax = fig.add_subplot(gs[2, 0])
    cross_section_figure(
        res_sadp.snapshots[pull_idx], ax=ax,
        title="SADP after mandrel pull — freestanding spacers, pitch halved",
    )

    ax = fig.add_subplot(gs[2, 1], projection="3d")
    stack_figure_mpl(res_sadp.snapshots[pull_idx], z_exaggeration=2.5, downsample=2, ax=ax)
    ax.view_init(elev=26, azim=-60)
    ax.set_title("3-D printed stack")

    out = out_dir / "multipatterning_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nWrote {out}")

    sadp_cut_demo(optics, resist, out_dir / "sadp_cut_demo.png")
    if not args.no_show:
        plt.show()
    return 0
