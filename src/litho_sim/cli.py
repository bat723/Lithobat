"""
LithoPy command-line interface.

Provides a CLI for running lithography simulations, generating Bossung
curves, and computing the process window.

Usage
-----
    # Quick demo of the full pipeline (writes the 2-D footprint AND the
    # 3-D resist profile: aerial_image.png, resist_profile.png,
    # resist_profile_3d.png)
    python -m litho_sim demo --node ArF --pitch 200 --cd 100

    # 3-D profile options: finite-rate develop, standing waves, or skip
    python -m litho_sim demo --develop-model mack
    python -m litho_sim demo --standing-waves     # needs substrate_reflectance > 0
    python -m litho_sim demo --no-3d

    # Bossung curve sweep and plot
    python -m litho_sim bossung --node ArF --pitch 200 --cd 100 --n-doses 5

    # Full process window analysis
    python -m litho_sim window --node ArF --pitch 200 --cd 100 --tolerance 10

(Installed via pyproject, the same commands are available as ``litho-sim …``.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from litho_sim.core.utils import setup_logging

logger = logging.getLogger(__name__)


def _apply_common_overrides(cfg, args: argparse.Namespace):
    """Fold shared CLI knobs into a freshly built config."""
    import dataclasses

    if getattr(args, "threshold", None) is not None:
        cfg.resist = dataclasses.replace(cfg.resist, threshold=args.threshold)
        logger.info("Resist threshold overridden → %.2f", args.threshold)
    return cfg


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _run_demo(args: argparse.Namespace) -> None:
    """Run a single-point simulation and display key results."""
    import matplotlib.pyplot as plt

    from litho_sim.analysis import compute_nils
    from litho_sim.core.config import SimulationConfig
    from litho_sim.develop.resist import measure_cd_2d, simulate_resist
    from litho_sim.expose.aerial_image import compute_aerial_image
    from litho_sim.mask.patterns import lines_and_spaces
    from litho_sim.viz.plots import (
        plot_aerial_image,
        plot_resist_profile,
        save_figure,
    )

    logger.info("=== LithoPy Demo ===")
    logger.info("Node=%s  pitch=%.0f nm  CD=%.0f nm", args.node, args.pitch, args.cd)

    cfg = SimulationConfig.from_tech_node(args.node, name="demo")
    cfg = _apply_common_overrides(cfg, args)
    mask = lines_and_spaces(
        cfg.grid.n_pixels, cfg.grid.pixel_size,
        pitch=args.pitch * 1e-9,
        cd=args.cd * 1e-9,
    )

    # Aerial image
    aerial = compute_aerial_image(mask, cfg.optics, cfg.grid, dose=1.0)

    # Resist
    _, _, resist_img = simulate_resist(aerial, cfg.resist, cfg.grid, model="threshold")

    # CD
    cd_m = measure_cd_2d(resist_img, cfg.grid.pixel_size)
    logger.info("Printed CD = %.1f nm  (target = %.0f nm)", cd_m * 1e9, args.cd)

    # NILS
    n = cfg.grid.n_pixels
    nils = compute_nils(
        aerial[n // 2, :],
        cfg.grid.pixel_size,
        threshold=cfg.resist.threshold,
        nominal_cd=args.cd * 1e-9,
    )
    logger.info("NILS = %.2f  (>2 = good printability)", nils)

    # Plots
    out = Path(args.output)
    fig1, _ = plot_aerial_image(aerial, cfg.grid, title=f"Aerial Image – {args.node}")
    save_figure(fig1, out / "aerial_image.png")

    fig2, _ = plot_resist_profile(aerial, resist_img, cfg.grid, threshold=cfg.resist.threshold)
    save_figure(fig2, out / "resist_profile.png")

    # 3-D resist profile — the depth-resolved solid, not just the footprint.
    if not args.no_3d:
        import dataclasses

        from litho_sim.develop.resist3d import print_resist_3d, sidewall_angle
        from litho_sim.viz.viz3d import resist_profile_3d_figure

        resist3d_cfg = cfg.resist
        if args.standing_waves is not None:
            # The flag must carry a reflectance: with the preset's default of
            # 0 (a perfect BARC), standing waves would be a silent no-op.
            resist3d_cfg = dataclasses.replace(
                cfg.resist, substrate_reflectance=args.standing_waves
            )
        res3 = print_resist_3d(
            mask, cfg.optics, cfg.grid, resist3d_cfg,
            dose=1.0,
            standing_waves=args.standing_waves is not None,
            develop_model=args.develop_model,
        )
        angle = sidewall_angle(res3["remaining"], cfg.grid)
        logger.info(
            "3-D profile: %.1f%% of film remaining, sidewall %.1f°",
            100.0 * float(res3["remaining"].mean()), angle,
        )
        fig3 = resist_profile_3d_figure(
            res3, cfg.grid,
            title=f"3-D resist profile – {args.node}, "
                  f"pitch {args.pitch:.0f} nm / CD {args.cd:.0f} nm "
                  f"({args.develop_model} develop)",
        )
        save_figure(fig3, out / "resist_profile_3d.png")

    logger.info("Saved plots → %s", out)

    if not args.no_show:
        plt.show()


def _run_bossung(args: argparse.Namespace) -> None:
    """Run a Bossung curve sweep and save results + plot."""
    import matplotlib.pyplot as plt
    import numpy as np

    from litho_sim.analysis import sweep_dose_focus
    from litho_sim.analysis.data_utils import save_results_csv
    from litho_sim.core.config import SimulationConfig
    from litho_sim.mask.patterns import lines_and_spaces
    from litho_sim.viz.plots import plot_bossung_curves, save_figure

    logger.info("=== Bossung Curve Sweep ===")
    cfg = SimulationConfig.from_tech_node(args.node, name="bossung")
    cfg = _apply_common_overrides(cfg, args)
    mask = lines_and_spaces(
        cfg.grid.n_pixels, cfg.grid.pixel_size,
        pitch=args.pitch * 1e-9,
        cd=args.cd * 1e-9,
    )

    doses = list(np.linspace(0.7, 1.3, args.n_doses))
    defoci = list(np.linspace(-args.defocus_range, args.defocus_range, args.n_defoci))

    bossung_df = sweep_dose_focus(
        mask=mask,
        optics=cfg.optics,
        grid=cfg.grid,
        resist=cfg.resist,
        doses=doses,
        defoci_nm=defoci,
        model="threshold",
        target_cd_nm=args.cd,
    )

    out = Path(args.output)
    save_results_csv(bossung_df, out / "bossung_data.csv")

    fig, _ = plot_bossung_curves(bossung_df, target_cd_nm=args.cd, tolerance_pct=10.0)
    save_figure(fig, out / "bossung_curves.png")

    logger.info("CD range: %.1f – %.1f nm", bossung_df["cd_nm"].min(), bossung_df["cd_nm"].max())
    logger.info("NILS range: %.2f – %.2f", bossung_df["nils"].min(), bossung_df["nils"].max())
    logger.info("Saved → %s", out)

    if not args.no_show:
        plt.show()


def _run_process_window(args: argparse.Namespace) -> None:
    """Run a full process window analysis."""
    import matplotlib.pyplot as plt

    from litho_sim.analysis import compute_el_dof_curve, run_full_analysis
    from litho_sim.analysis.data_utils import save_results_csv
    from litho_sim.core.config import SimulationConfig
    from litho_sim.viz.plots import (
        plot_bossung_curves,
        plot_cd_heatmap,
        plot_el_dof_curve,
        plot_process_window,
        save_figure,
    )

    logger.info("=== Process Window Analysis ===")
    cfg = SimulationConfig.from_tech_node(args.node, name="process_window")
    cfg = _apply_common_overrides(cfg, args)

    bossung_df, pw = run_full_analysis(
        cfg=cfg,
        pitch_nm=args.pitch,
        target_cd_nm=args.cd,
        n_doses=args.n_doses,
        n_defoci=args.n_defoci,
        defocus_range_nm=args.defocus_range,
        tolerance_pct=args.tolerance,
        model="threshold",
    )

    out = Path(args.output)
    save_results_csv(bossung_df, out / "process_window_data.csv")

    fig1, _ = plot_bossung_curves(
        bossung_df, target_cd_nm=args.cd, tolerance_pct=args.tolerance,
        best_focus_nm=pw["best_focus_nm"] if pw["prints"] else None,
    )
    save_figure(fig1, out / "bossung_curves.png")

    fig3, _ = plot_cd_heatmap(bossung_df, target_cd_nm=args.cd, tolerance_pct=args.tolerance)
    save_figure(fig3, out / "cd_heatmap.png")

    if pw["prints"]:
        fig2, _ = plot_process_window(pw, target_cd_nm=args.cd)
        save_figure(fig2, out / "process_window.png")

        el_dof = compute_el_dof_curve(
            bossung_df, target_cd_nm=args.cd, tolerance_pct=args.tolerance
        )
        save_results_csv(el_dof, out / "el_dof_curve.csv")
        fig4, _ = plot_el_dof_curve(el_dof)
        save_figure(fig4, out / "el_dof_curve.png")
    else:
        logger.warning(
            "Nothing printed robustly anywhere in the sweep — the window and "
            "EL-DOF plots are skipped. Doses are clear-field normalised; the "
            "printable regime may sit outside 0.7–1.3."
        )

    logger.info("─" * 50)
    logger.info("Process Window Summary")
    logger.info("  Node         : %s", args.node)
    logger.info("  Pitch        : %.0f nm", args.pitch)
    logger.info("  Target CD    : %.0f nm  (±%.0f%%)", args.cd, args.tolerance)
    if pw["prints"]:
        logger.info("  EL           : %.1f %%  (of nominal dose %.2f)",
                    pw["EL_pct"], pw["nominal_dose"])
        logger.info("  DOF          : %.0f nm", pw["DOF_nm"])
        logger.info("  Window area  : %.0f %%·nm", pw["area"])
        logger.info("  Best focus   : %.0f nm", pw["best_focus_nm"])
        logger.info("  Best dose    : %.3f", pw["best_dose"])
    else:
        logger.info("  No process window: nothing printed in spec.")
    logger.info("─" * 50)
    logger.info("Saved → %s", out)

    if not args.no_show:
        plt.show()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litho_sim",
        description="LithoPy – Photolithography Simulation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    subs = parser.add_subparsers(dest="command", metavar="COMMAND")
    subs.required = True

    # ---- shared arguments ----
    def _add_litho_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--node", default="ArF",
                       choices=["i-line", "KrF", "ArF", "ArF_immersion", "EUV"])
        p.add_argument("--pitch", type=float, default=200.0, help="Pattern pitch [nm]")
        p.add_argument("--cd", type=float, default=100.0, help="Target CD [nm]")
        p.add_argument("--threshold", type=float, default=None, metavar="T",
                       help="Resist clearing threshold, 0–1 (default: the resist "
                            "preset's own value). This is the CD calibration knob: "
                            "a higher threshold prints a wider feature. The demo "
                            "image is peak-normalised; the bossung/window sweeps "
                            "normalise to the clear field instead, so a dose means "
                            "the same energy at every focus — the same threshold "
                            "therefore prints differently in the two commands.")
        p.add_argument("--output", type=Path, default=Path("results"),
                       help="Output directory for plots and CSV")
        p.add_argument("--no-show", action="store_true", help="Don't open interactive plot window")

    def _add_sweep_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--n-doses", type=int, default=5, help="Number of dose levels")
        p.add_argument("--n-defoci", type=int, default=11, help="Number of defocus steps")
        p.add_argument("--defocus-range", type=float, default=300.0,
                       help="Half-range of defocus sweep [nm]")

    # -- demo --
    p_demo = subs.add_parser("demo", help="Single-point simulation demo")
    _add_litho_args(p_demo)
    p_demo.add_argument("--no-3d", action="store_true",
                        help="Skip the 3-D resist profile (2-D footprint only)")
    p_demo.add_argument("--standing-waves", nargs="?", const=0.35, default=None,
                        type=float, metavar="R",
                        help="Include substrate-reflection standing waves in the "
                             "3-D profile, with substrate reflectance R "
                             "(default 0.35 when the flag is given bare). "
                             "Without this flag the preset's reflectance — "
                             "usually 0, a perfect BARC — applies.")
    p_demo.add_argument("--develop-model", default="threshold",
                        choices=["threshold", "mack"],
                        help="3-D develop model (mack = finite-rate ray develop)")

    # -- bossung --
    p_bos = subs.add_parser("bossung", help="Generate Bossung curves")
    _add_litho_args(p_bos)
    _add_sweep_args(p_bos)

    # -- window --
    p_win = subs.add_parser("window", help="Full process window analysis")
    _add_litho_args(p_win)
    _add_sweep_args(p_win)
    p_win.add_argument("--tolerance", type=float, default=10.0, help="CD tolerance [%%]")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    from litho_sim.viz.plots import apply_style

    apply_style()

    dispatch = {
        "demo": _run_demo,
        "bossung": _run_bossung,
        "window": _run_process_window,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()

