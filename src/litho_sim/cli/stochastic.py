"""``litho-sim stochastic`` — print the same exposure many times and measure the scatter.

    litho-sim stochastic                       # EUV, 64 nm pitch, 2-D trials
    litho-sim stochastic --profile             # add 3-D trials: roughness through the film
    litho-sim stochastic --node ArF_immersion --pitch 120 --cd 60

The 2-D batch runs the chemically amplified chain with sampled photons and
molecules (:func:`~litho_sim.develop.stochastic.stochastic_trials`) and
reports LER, LWR, correlation length, LCDU and the failure rate, every
geometric number read off the continuous develop-depth field with sub-pixel
crossings. ``--profile`` calibrates dose and develop time for the film,
then runs the 3-D trials (:func:`~litho_sim.develop.stochastic.stochastic_trials_3d`)
on a coarser grid and draws line-width roughness against height, the
top-loss scatter and a cross-section of one trial.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from litho_sim.analysis import (
    failure_rate,
    measure_lcdu,
    profile_roughness,
    trial_roughness,
)
from litho_sim.cli.common import add_node_arg, add_output_args, resist_for
from litho_sim.core.config import GridConfig, SimulationConfig
from litho_sim.develop import (
    arrival_field,
    calibrate_profile,
    print_resist_3d,
    simulate_resist,
    stochastic_trials,
    stochastic_trials_3d,
)
from litho_sim.expose import compute_aerial_image
from litho_sim.mask import lines_and_spaces
from litho_sim.viz import theme
from litho_sim.viz.plots import save_figure


def add_parser(subs) -> None:
    ap = subs.add_parser("stochastic", help="stochastic printing: LER, LWR, LCDU, failures, "
                                            "and with --profile the roughness through the film",
                         description=(__doc__ or "").split("\n\n")[0])
    add_node_arg(ap, default="EUV")
    ap.add_argument("--pitch", type=float, default=64.0, help="pitch [nm]")
    ap.add_argument("--cd", type=float, default=32.0, help="drawn line width [nm]")
    ap.add_argument("--pixel-nm", type=float, default=2.0, help="2-D grid pixel [nm]")
    ap.add_argument("--pixels", type=int, default=96, help="2-D grid side [px]")
    ap.add_argument("--dose", type=float, default=None,
                    help="relative dose for the 2-D batch (default: dose-to-size of the line)")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true",
                    help="also run 3-D trials on a coarse grid and draw roughness through the film")
    ap.add_argument("--profile-pixel-nm", type=float, default=4.0)
    ap.add_argument("--profile-trials", type=int, default=8)
    add_output_args(ap)
    ap.set_defaults(func=run)


def _dose_to_size(mask, optics, grid, resist, target_nm: float) -> float:
    from litho_sim.analysis import calibrate_dose_to_size

    dose = calibrate_dose_to_size(mask, optics, grid, resist, target_nm, model="car",
                                  normalisation="clear")
    return float(dose) if np.isfinite(dose) else 1.0


def run(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    cfg = SimulationConfig.from_tech_node(args.node, resist_name=resist_for(args))
    optics = dataclasses.replace(cfg.optics, normalisation="clear")
    px = args.pixel_nm * 1e-9
    grid = GridConfig(n_pixels=args.pixels, pixel_size=px)
    mask = lines_and_spaces(args.pixels, px, pitch=args.pitch * 1e-9, cd=args.cd * 1e-9)
    dose = args.dose if args.dose is not None else _dose_to_size(mask, optics, grid, cfg.resist,
                                                                  args.cd)
    aerial = compute_aerial_image(mask, optics, grid, dose=dose)
    _, _, reference = simulate_resist(aerial, cfg.resist, grid, model="car")
    res = stochastic_trials(aerial, cfg.resist, grid, optics.wavelength,
                            trials=args.trials, seed=args.seed)
    rough = trial_roughness(res, px)
    lcdu = measure_lcdu(res.depth, px, threshold=res.level, feature=res.feature)
    fails = failure_rate(res.resist, reference)
    print(f"{args.node}: {args.pitch:.0f}/{args.cd:.0f} nm, dose {dose:.3f}, {args.trials} trials, "
          f"{res.sample.mean_photons:.1f} absorbed photons and {res.sample.mean_pag:.0f} PAG per "
          f"{args.pixel_nm:.0f} nm voxel")
    print(f"  LER 3σ {rough.ler_3s * 1e9:.2f} nm   LWR 3σ {rough.lwr_3s * 1e9:.2f} nm   "
          f"ξ {rough.corr_length * 1e9:.1f} nm   CD {lcdu.cd_mean * 1e9:.1f} nm   "
          f"LCDU {lcdu.lcdu * 1e9:.2f} nm   fails {fails.n_failed}/{fails.n_trials} "
          f"(bridges {fails.bridges_total}, breaks {fails.breaks_total}; residue specks "
          f"{fails.residue_total}, {fails.residue_pixels * (args.pixel_nm ** 2):.0f} nm²)")

    profile = None
    if args.profile:
        ppx = args.profile_pixel_nm * 1e-9
        n3 = max(int(round(args.pixels * px / ppx)), 32)
        grid3 = GridConfig(n_pixels=n3, pixel_size=ppx, dz=ppx, n_z_slices=5)
        mask3 = lines_and_spaces(n3, ppx, pitch=args.pitch * 1e-9, cd=args.cd * 1e-9)
        cal = calibrate_profile(mask3, optics, grid3, cfg.resist, args.cd, bake="car")
        det = print_resist_3d(mask3, optics, grid3, cal.resist, dose=cal.dose,
                              standing_waves=False, develop_model="mack", bake="car")
        ref_T, _, _ = arrival_field(det["latent"], cal.resist, grid3, "mack")
        res3 = stochastic_trials_3d(mask3, optics, grid3, cal.resist, dose=cal.dose,
                                    trials=args.profile_trials, seed=args.seed)
        profile = profile_roughness(res3, grid3, reference=ref_T)
        s = profile.summary()
        print(f"  3-D ({n3}×{n3}×{res3.arrival.shape[1]} voxels, dose {cal.dose:.3f}, develop "
              f"{cal.develop_time:.2f} s): LWR 3σ bottom/mid/top "
              f"{s['lwr_3s_bottom_nm']:.2f}/{s['lwr_3s_mid_nm']:.2f}/{s['lwr_3s_top_nm']:.2f} nm, "
              f"top loss {s['top_loss_mean_nm']:.1f} ± {s['top_loss_sigma_nm']:.1f} nm, "
              f"footing {s['footing_mean_nm']:.1f} nm, residue at the base "
              f"{s['residue_base_pct']:.1f}% of the open area, bridges {profile.bridges_base}")

    # ---- figure ----
    theme.apply()
    rows = 2 if profile is not None else 1
    fig = plt.figure(figsize=(13, 4.2 * rows), constrained_layout=True)
    gs = fig.add_gridspec(rows, 3)
    side = args.pixels * px * 1e9
    extent = (0.0, side, 0.0, side)
    resist_c = theme.material_colour("photoresist")
    bin_cmap, bin_norm = theme.binary_cmap(resist_c)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(reference, origin="lower", extent=extent, cmap=bin_cmap, norm=bin_norm,
              interpolation="nearest")
    theme.title(ax, "Deterministic print", caption_text=f"car model · dose {dose:.3f}")
    ax.set(xlabel="x [nm]", ylabel="y [nm]")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(res.resist[0], origin="lower", extent=extent, cmap=bin_cmap, norm=bin_norm,
              interpolation="nearest")
    theme.title(ax, "One trial", caption_text=(
        f"LER 3σ {rough.ler_3s * 1e9:.2f} nm · ξ {rough.corr_length * 1e9:.0f} nm"))
    ax.set(xlabel="x [nm]")

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(res.resist.mean(axis=0), origin="lower", extent=extent,
                   cmap=theme.CMAP.intensity, vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, shrink=0.85, label="resist probability")
    theme.title(ax, f"{args.trials} trials", caption_text=(
        f"LWR 3σ {rough.lwr_3s * 1e9:.2f} nm · LCDU {lcdu.lcdu * 1e9:.2f} nm · "
        f"fails {fails.n_failed}/{fails.n_trials}"))
    ax.set(xlabel="x [nm]")

    if profile is not None:
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(profile.lwr_3s * 1e9, profile.z_nm, color=theme.SERIES.aerial, lw=2, label="LWR 3σ")
        ax.plot(profile.ler_3s * 1e9, profile.z_nm, color=theme.SERIES.latent, lw=2, label="LER 3σ")
        ax.set(xlabel="roughness [nm]", ylabel="height above substrate [nm]", xlim=(0, None))
        theme.legend(ax)
        theme.title(ax, "Roughness through the film",
                    caption_text=f"{res3.trials} trials · {res3.sample.mean_pag:.0f} PAG per voxel")

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(profile.cd * 1e9, profile.z_nm, color=theme.SERIES.aerial, lw=2)
        ax.axvline(args.cd, color=theme.MUTED, lw=1)
        ax.set(xlabel="line width [nm]", ylabel="height above substrate [nm]")
        theme.title(ax, "Profile: mean width by height", caption_text=(
            f"top loss {profile.summary()['top_loss_mean_nm']:.1f} nm · "
            f"footing {profile.summary()['footing_mean_nm']:.1f} nm"))

        ax = fig.add_subplot(gs[1, 2])
        rem = res3.remaining[0]
        nz, ny, nx = rem.shape
        ax.imshow(rem[:, ny // 2, :], origin="lower", aspect="auto", cmap=bin_cmap, norm=bin_norm,
                  extent=(0.0, nx * grid3.pixel_size * 1e9, 0.0, nz * grid3.dz * 1e9),
                  interpolation="nearest")
        ax.set(xlabel="x [nm]", ylabel="z [nm]")
        theme.title(ax, "One trial, cleaved", caption_text=f"{args.profile_pixel_nm:.0f} nm voxels")

    out_dir = Path(args.output)
    out = out_dir / "stochastic_demo.png"
    save_figure(fig, out)
    print(f"figure → {out}   [{time.perf_counter() - t0:.1f} s]")

    if profile is not None:
        from litho_sim.viz import render

        if render.available():
            # One trial, cleaved and put under the microscope: the arrival
            # field gives the surface to sub-voxel accuracy, so the
            # roughness on the walls is the trial's, not the grid's.
            from litho_sim.viz.plots import plot_tilt_sem

            sem, buffers = render.tilt_sem_of_profile(
                None, grid3, field=res3.arrival[0], level=float(res3.level),
                film_nm=float(cal.resist.thickness) * 1e9,
            )
            fig_sem = plot_tilt_sem(
                sem, buffers,
                title=(f"{args.node}  ·  {args.pitch:.0f} / {args.cd:.0f} nm L/S  ·  "
                       f"{resist_for(args)}, {cal.resist.thickness * 1e9:.0f} nm  ·  "
                       f"stochastic trial  ·  cleaved"),
                caption=f"{args.profile_pixel_nm:g} nm voxels",
            )
            out_sem = out_dir / "stochastic_profile_sem.png"
            fig_sem.savefig(out_sem, dpi=150, facecolor=fig_sem.get_facecolor())
            print(f"micrograph → {out_sem}")
        else:
            print("  (pyvista not installed: no tilt-SEM micrograph of the trial)")
    if not args.no_show:
        plt.show()
    return 0
