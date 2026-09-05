"""Optical proximity correction on a layout that has every classic problem.

Four kinds of feature, one field, one dose: a dense array of lines, an
isolated line, line ends facing each other across a gap, and a T-junction.
Printed at the dose that sizes the dense array, each of them prints wrong
in its own way — the isolated line off-size, every line end pulled back,
the inner corner of the T filled in. That is proximity effect, and it is
what OPC exists to remove.

The script prints the design as drawn, measures the edge placement error
at every fragment, runs the model-based correction, measures again, and
writes one figure with the corrected mask, the printed contours before and
after, and the EPE distribution. Nothing in it is special-cased for the
demo: ``run_opc`` is the same call a device flow would make.

    litho-sim opc                        # results/opc_demo.png
    litho-sim opc --sraf                 # add scattering bars first
    litho-sim opc --node EUV --pitch 40 --cd 20
    litho-sim opc --sharp-corners        # what happens without retargeting

(``scripts/demo_opc.py`` is the same command.) The pictures of *how* — the
fragments and their EPE on one line end, the mask iteration by iteration, the
aerial image before and after, the convergence traces — are
``scripts/opc_gallery.py``, which reuses this layout and run.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from litho_sim.cli.common import add_node_arg, resist_for
from litho_sim.core.config import GridConfig, SimulationConfig
from litho_sim.mask import Layout, Polygon, Rect, line_array
from litho_sim.opc import PrintModel, add_scattering_bars, assist_features_printed, run_opc
from litho_sim.viz import plot_opc, save_figure


def build_layout(pitch: float, cd: float, length: float) -> Layout:
    """Dense lines, a tip-to-tip gap and a T, all at one CD, in a 6.4-pitch field.

    Positions are in pitches so the layout scales with the node. The T is
    one polygon: OPC fragments outlines, and a line with a bar abutting it
    would have a shared edge corrected as two edges facing each other.
    """
    p = pitch
    shapes = line_array(3, pitch=p, cd=cd, length=length, centre=(-1.4 * p, 0))
    # Two collinear segments with a gap of one CD between their ends — an
    # isolated line, and two line ends facing each other.
    x_pair = 0.9 * p
    seg = 0.5 * (length - cd)
    shapes += [
        Rect("main", x_pair, +0.5 * (cd + seg), cd, seg),
        Rect("main", x_pair, -0.5 * (cd + seg), cd, seg),
    ]
    # A T: a vertical line with a 0.6-pitch bar off its middle.
    xt, bar = 2.1 * p, 0.6 * p
    x0, x1 = xt - 0.5 * cd, xt + 0.5 * cd
    shapes.append(Polygon("main", (
        (x0, -0.5 * length), (x1, -0.5 * length), (x1, -0.5 * cd), (x1 + bar, -0.5 * cd),
        (x1 + bar, 0.5 * cd), (x1, 0.5 * cd), (x1, 0.5 * length), (x0, 0.5 * length),
    )))
    return Layout(shapes, name="proximity-test")


def add_parser(subs) -> None:
    ap = subs.add_parser("opc", help="model-based OPC on a dense/iso/tip-to-tip/T layout",
                         description=(__doc__ or "").split("\n\n")[0])

    add_node_arg(ap)
    ap.add_argument("--pitch", type=float, default=200.0, help="dense pitch [nm]")
    ap.add_argument("--cd", type=float, default=100.0, help="drawn CD of every feature [nm]")
    ap.add_argument("--pixels", type=int, default=256, help="grid side [px]")
    ap.add_argument("--pixel-nm", type=float, default=None,
                    help="pixel size [nm]; default makes the field 6.4 pitches wide")
    ap.add_argument("--resist-model", default="threshold", choices=["threshold", "mack", "car"],
                    help="resist model the print model corrects against")
    ap.add_argument("--diffusion-nm", type=float, default=None,
                    help="PEB diffusion length [nm]; the preset's 20 nm swallows a 40 nm "
                         "pitch, so an EUV run wants ~5")
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--sraf", action="store_true", help="place scattering bars before correcting")
    ap.add_argument("--sharp-corners", action="store_true",
                    help="target the drawn corners instead of a printable radius")
    ap.add_argument("--output", default="results/opc_demo.png")
    ap.add_argument("--no-show", action="store_true")
    ap.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:

    cfg = SimulationConfig.from_tech_node(args.node, resist_name=resist_for(args))
    if args.diffusion_nm is not None:
        cfg.resist.diffusion_sigma = args.diffusion_nm * 1e-9
    px = args.pixel_nm * 1e-9 if args.pixel_nm else 6.4 * args.pitch * 1e-9 / args.pixels
    grid = GridConfig(n_pixels=args.pixels, pixel_size=px)
    pitch, cd = args.pitch * 1e-9, args.cd * 1e-9
    length = 0.7 * grid.grid_size

    design = build_layout(pitch, cd, length)
    model = PrintModel(cfg.optics, cfg.resist, grid, tone="dark", model=args.resist_model)

    # Dose-to-size on the dense array — the anchor every real process uses.
    anchor = Layout(line_array(7, pitch=pitch, cd=cd, length=0.9 * grid.grid_size))
    t0 = time.perf_counter()
    dose = model.dose_to_size(anchor, args.cd)
    print(f"{args.node}: λ {cfg.optics.wavelength * 1e9:.1f} nm, NA {cfg.optics.NA}, "
          f"{grid.n_pixels} px × {px * 1e9:.2f} nm; dense {args.pitch:.0f}/{args.cd:.0f} nm, "
          f"PEB diffusion {cfg.resist.diffusion_sigma * 1e9:.0f} nm")
    if not np.isfinite(dose):
        print(
            f"the dense array never prints {args.cd:.0f} nm at any dose: the pitch is "
            f"below what this resist resolves (k1 = "
            f"{0.5 * args.pitch * 1e-9 * cfg.optics.NA / cfg.optics.wavelength:.2f}, "
            f"PEB diffusion {cfg.resist.diffusion_sigma * 1e9:.0f} nm). Try --diffusion-nm 5."
        )
        return 1
    model.dose = dose
    print(f"dose-to-size on the dense array: {model.dose:.4f}  "
          f"({time.perf_counter() - t0:.2f} s)")

    layout = design
    if args.sraf:
        layout = add_scattering_bars(design, gap=pitch - cd, width=0.4 * cd)
        print(f"scattering bars: {len(layout.on_layer('sraf'))} placed")

    t0 = time.perf_counter()
    result = run_opc(
        layout, model, layer="main", max_iter=args.iterations,
        corner_radius=0.0 if args.sharp_corners else None,
    )
    elapsed = time.perf_counter() - t0
    print(f"OPC: {len(result.offsets)} fragments, {len(result.history) - 1} iterations, "
          f"{elapsed:.1f} s, {'converged' if result.converged else 'not converged'}")
    print(result.stats().round(2).to_string())
    print()
    print("per iteration:")
    print(result.history.round(2).to_string(index=False))

    df = result.summary()
    print()
    print("where the correction went (mean |offset| by fragment role, nm):")
    print(df.groupby("role")[["epe_before_nm", "epe_after_nm", "offset_nm"]]
          .agg(lambda v: float(np.nanmean(np.abs(v)))).round(2).to_string())
    print()
    print("the six worst sites as drawn:")
    cols = ["shape", "role", "x_nm", "y_nm", "epe_before_nm", "epe_after_nm", "offset_nm"]
    print(df.reindex(df.epe_before_nm.abs().sort_values(ascending=False).index)
          .head(6)[cols].round(1).to_string(index=False))

    if args.sraf:
        flags = assist_features_printed(result.after, result.corrected)
        print(f"\nassist features printed after correction: {sum(flags)} of {len(flags)}")

    fig, _ = plot_opc(result)
    save_figure(fig, args.output)
    print(f"\nfigure → {args.output}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    return 0
