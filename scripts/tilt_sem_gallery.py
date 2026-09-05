"""The tilt-SEM view in pictures: what the beam sees, the model one term at a
time, the terrace fix, the stage, the instrument, the projection, the cost.

    python scripts/tilt_sem_gallery.py            # → results/tilt_sem_gallery_*.png

Two profiles are computed once and cached in ``results/tilt_sem_gallery.npz``
(about two minutes the first time): the deterministic ArF-immersion
128 / 64 nm print at 4 nm voxels, calibrated to dose-to-size with the front
developer and the acid/quencher bake, and one stochastic trial of the same
pattern at 3 nm voxels. Every figure is formed from those two fields.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litho_sim.core.config import GridConfig, SimulationConfig  # noqa: E402
from litho_sim.develop import (  # noqa: E402
    calibrate_profile,
    print_resist_3d,
    stochastic_trials_3d,
)
from litho_sim.mask import lines_and_spaces  # noqa: E402
from litho_sim.metrology.sem import SEMConfig, tilt_sem, tilt_signal  # noqa: E402
from litho_sim.viz import render  # noqa: E402
from litho_sim.viz.plots import _stretch  # noqa: E402

BG, INK, INK2, MONO = "#101010", "#e6e6e6", "#9a9a9a", "DejaVu Sans Mono"


def profiles(cache: Path) -> dict:
    """The two develop fields the gallery is drawn from, computed once."""
    if cache.exists():
        c = np.load(cache)
        return {k: c[k] for k in c.files}
    logging.disable(logging.CRITICAL)
    cfg = SimulationConfig.from_tech_node("ArF_immersion", resist_name="CAR (Positive)")
    # Deterministic: the demo's own grid, 4 nm pixels and 2 nm rows.
    g4 = cfg.grid
    mask4 = lines_and_spaces(g4.n_pixels, g4.pixel_size, pitch=128e-9, cd=64e-9)
    cal = calibrate_profile(mask4, cfg.optics, g4, cfg.resist, 64.0,
                            develop_model="front", bake="car")
    res = print_resist_3d(mask4, cfg.optics, g4, cal.resist, dose=cal.dose,
                          develop_model="front", bake="car")
    # One stochastic trial at 3 nm, on a field of three pitches.
    optics = dataclasses.replace(cfg.optics, source_grid=11, normalisation="clear")
    g3 = GridConfig(n_pixels=128, pixel_size=3e-9, dz=2.5e-9, n_z_slices=9)
    mask3 = lines_and_spaces(128, 3e-9, pitch=128e-9, cd=64e-9)
    cal3 = calibrate_profile(mask3, optics, g3, cfg.resist, 64.0, bake="car", overdevelop=2.0)
    sto = stochastic_trials_3d(mask3, optics, g3, cal3.resist, dose=cal3.dose, trials=1,
                               seed=7, develop_model="front")
    out = {
        "field4": res["field"], "level4": np.float64(res["level"]), "rem4": res["remaining"],
        "field3": sto.arrival[0], "level3": np.float64(sto.level),
    }
    np.savez_compressed(cache, **out)
    return out


def dark_fig(w, h, title, sub=None):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    fig.text(0.02, 0.975, title, color=INK, fontsize=12, family=MONO, va="top")
    if sub:
        fig.text(0.02, 0.945, sub, color=INK2, fontsize=9, family=MONO, va="top")
    return fig


def show(ax, img, cmap="gray", vmin=0, vmax=1, title=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=INK, fontsize=9.5, family=MONO, loc="left", pad=4)


def save(fig, out: Path, name: str, dpi=110) -> None:
    path = out / f"tilt_sem_gallery_{name}.png"
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print(f"  {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", default="results", help="where the figures go")
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    p = profiles(out / "tilt_sem_gallery.npz")
    T4, lvl4, rem4 = p["field4"], float(p["level4"]), p["rem4"]
    T3, lvl3 = p["field3"], float(p["level3"])
    g3 = GridConfig(n_pixels=128, pixel_size=3e-9, dz=2.5e-9, n_z_slices=9)

    # -- B: the geometry buffers -------------------------------------------
    surf4 = render.profile_surface(field=T4, level=lvl4, spacing_nm=(2, 4, 4))
    buf4 = render.render_buffers(surf4, (512, 512), 100.0, render.Stage(), voxel_nm=(4, 4))
    sem4 = tilt_sem(buf4, SEMConfig())
    fig = dark_fig(17, 5.4, "The geometry pass: what the beam sees before any physics",
                   "two off-screen VTK passes with lighting off — normals into the colour "
                   "buffer + the depth buffer, then a flat material id — under a parallel camera")
    gs = fig.add_gridspec(1, 4, left=0.01, right=0.99, top=0.86, bottom=0.03, wspace=0.03)
    show(fig.add_subplot(gs[0]), 0.5 * (np.nan_to_num(buf4.normals) + 1.0),
         title="normals  (xyz → rgb)")
    d = np.where(buf4.seen, buf4.depth, np.nan)
    show(fig.add_subplot(gs[1]), d, cmap="magma", vmin=np.nanmin(d), vmax=np.nanmax(d),
         title="depth along the beam  [nm]")
    show(fig.add_subplot(gs[2]), buf4.material, cmap="viridis", vmin=0, vmax=2,
         title="material  (0 vacuum · 1 substrate · 2 resist)")
    show(fig.add_subplot(gs[3]), _stretch(sem4.image), title="→ the micrograph  (tilt_sem)")
    save(fig, out, "buffers")

    # -- C: the terms, one at a time ---------------------------------------
    surf3 = render.profile_surface(field=T3, level=lvl3, spacing_nm=(2.5, 3, 3))
    buf3 = render.render_buffers(surf3, (384, 384), 100.0,
                                 render.Stage(tilt=40, azimuth=20, pixel_nm=0.4),
                                 voxel_nm=(3, 3))
    steps = [
        ("1 · material only", dict(secant_power=0.0, edge_yield=0.0, escape_length=0.0,
                                   directionality=0.0, shadowing=0.0, beam_fwhm=0.0)),
        ("2 · + tilt yield  sec^0.8 θ", dict(edge_yield=0.0, escape_length=0.0,
                                             directionality=0.0, shadowing=0.0,
                                             beam_fwhm=0.0)),
        ("3 · + silhouette bloom", dict(directionality=0.0, shadowing=0.0, beam_fwhm=0.0)),
        ("4 · + detector side", dict(shadowing=0.0, beam_fwhm=0.0)),
        ("5 · + trench shadow", dict(beam_fwhm=0.0)),
        ("6 · + beam 3 nm, 400 e⁻/px", dict()),
    ]
    fig = dark_fig(18, 5.5, "The SE model, one term at a time  (tilt_signal)",
                   "the same stochastic trial; each panel adds the next term. Same grey scale "
                   "throughout: 0 → 3 SE per primary, then the noisy frame")
    gs = fig.add_gridspec(1, 6, left=0.005, right=0.995, top=0.82, bottom=0.02,
                          wspace=0.02)
    rows, cols = buf3.shape
    crop = (slice(int(rows * 0.05), int(rows * 0.75)),
            slice(int(cols * 0.15), int(cols * 0.62)))
    for i, (name, kw) in enumerate(steps):
        cfg = SEMConfig(**kw)
        img = tilt_sem(buf3, cfg).image if i == 5 else tilt_signal(buf3, cfg)
        show(fig.add_subplot(gs[i]), img[crop], vmin=0.0, vmax=3.0, title=name)
    save(fig, out, "terms")

    # -- D: the terraces ---------------------------------------------------
    surf_b = render.profile_surface(rem4, spacing_nm=(2, 4, 4))
    stage_fine = render.Stage(pixel_nm=0.4)
    buf_b = render.render_buffers(surf_b, (512, 512), 100.0, stage_fine, voxel_nm=(4, 4))
    buf_f = render.render_buffers(surf4, (512, 512), 100.0, stage_fine, voxel_nm=(4, 4))
    quiet = SEMConfig(electrons_per_pixel=400, frames=4)
    im_b = _stretch(tilt_sem(buf_b, quiet).image)
    im_f = _stretch(tilt_sem(buf_f, quiet).image)
    rows, cols = buf_b.shape
    crop = (slice(int(rows * 0.45), int(rows * 0.95)),
            slice(int(cols * 0.05), int(cols * 0.55)))
    nz, ny, _nx = rem4.shape
    r = ny // 2
    z = (np.arange(nz) + 0.5) * 2.0
    x_bool, x_field = [], []
    for iz in range(nz):
        row = rem4[iz, r]
        edge = np.nonzero(np.diff(row.astype(int)) == -1)[0]
        i = int(edge[0]) if edge.size else 0
        x_bool.append((i + 1) * 4.0 if edge.size else np.nan)
        a, b = T4[iz, r, i], T4[iz, r, i + 1]
        x_field.append((i + 0.5 + (a - lvl4) / (a - b)) * 4.0 if a != b else np.nan)
    fig = dark_fig(17, 6.6, "Why the develop step returns its field  "
                            "(develop_surface → field / level / feature)",
                   "an 84° wall steps sideways one 4 nm voxel every ~25 nm, and a boolean volume "
                   "can only step; the micrograph drew each step as a ledge")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.9], left=0.01, right=0.99,
                          top=0.86, bottom=0.1, wspace=0.08)
    show(fig.add_subplot(gs[0]), im_b[crop],
         title="surface from the boolean volume  (remaining)")
    show(fig.add_subplot(gs[1]), im_f[crop],
         title="surface from the develop field  (arrival time at the develop time)")
    ax = fig.add_subplot(gs[2])
    ax.set_facecolor(BG)
    ax.step(x_bool, z, where="mid", color="#e0a060", lw=2, label="boolean edge (voxel)")
    ax.plot(x_field, z, color="#80c8ff", lw=2, label="field crossing (sub-voxel)")
    ax.set_xlabel("wall x [nm]", color=INK2)
    ax.set_ylabel("z above substrate [nm]", color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.legend(loc="upper left", fontsize=8, facecolor=BG, edgecolor="#444", labelcolor=INK)
    ax.set_title("the same wall through the film, mid-field row", color=INK, fontsize=9.5,
                 family=MONO, loc="left")
    save(fig, out, "terraces")

    # -- E: the stage --------------------------------------------------------
    fig = dark_fig(18, 9.2, "The stage: two live knobs on the Develop tab  "
                            "(Stage.tilt, Stage.azimuth)",
                   "top: tilt 0° → 60° at rotation 20°   ·   bottom: rotation −30° → 45° at "
                   "tilt 40°. Same trial, same instrument; the frame is framed to the field "
                   "each time")
    gs = fig.add_gridspec(2, 4, left=0.005, right=0.995, top=0.88, bottom=0.02,
                          wspace=0.02, hspace=0.12)
    for j, tilt in enumerate((0.0, 20.0, 40.0, 60.0)):
        stage = render.Stage(tilt=tilt, azimuth=20.0, frame_px=700)
        sem, _ = render.tilt_sem_of_profile(None, g3, field=T3, level=lvl3, film_nm=100.0,
                                            stage=stage)
        show(fig.add_subplot(gs[0, j]), _stretch(sem.image),
             title=f"tilt {tilt:.0f}°, rotation 20°")
    for j, az in enumerate((-30.0, 0.0, 20.0, 45.0)):
        stage = render.Stage(tilt=40.0, azimuth=az, frame_px=700)
        sem, _ = render.tilt_sem_of_profile(None, g3, field=T3, level=lvl3, film_nm=100.0,
                                            stage=stage)
        show(fig.add_subplot(gs[1, j]), _stretch(sem.image),
             title=f"tilt 40°, rotation {az:+.0f}°")
    save(fig, out, "stage", dpi=100)

    # -- F: the instrument ---------------------------------------------------
    rows, cols = buf3.shape
    crop = (slice(int(rows * 0.08), int(rows * 0.62)),
            slice(int(cols * 0.18), int(cols * 0.58)))
    sweeps = [
        ("beam FWHM", [("1.5 nm", dict(beam_fwhm=1.5e-9)), ("3 nm", dict(beam_fwhm=3e-9)),
                       ("6 nm", dict(beam_fwhm=6e-9))]),
        ("electrons / px", [("25 × 4 frames", dict(electrons_per_pixel=25)),
                            ("100 × 4", dict(electrons_per_pixel=100)),
                            ("400 × 4", dict(electrons_per_pixel=400))]),
        ("detector side", [("0 (in-lens)", dict(directionality=0.0)),
                           ("0.5", dict(directionality=0.5)),
                           ("1.0", dict(directionality=1.0))]),
        ("trench shadow", [("0", dict(shadowing=0.0)), ("0.5", dict(shadowing=0.5)),
                           ("1.0", dict(shadowing=1.0))]),
    ]
    fig = dark_fig(9.5, 15, "The instrument: the SEM tab's knobs, applied to the tilt view",
                   "one SEMConfig field per row, the rest at default; geometry rendered once and "
                   "cached,\nonly the signal re-forms (~55 ms)")
    gs = fig.add_gridspec(4, 3, left=0.005, right=0.995, top=0.92, bottom=0.01,
                          wspace=0.02, hspace=0.14)
    for i, (label, cases) in enumerate(sweeps):
        for j, (val, kw) in enumerate(cases):
            img = _stretch(tilt_sem(buf3, SEMConfig(**kw)).image)
            show(fig.add_subplot(gs[i, j]), img[crop], title=f"{label} = {val}")
    save(fig, out, "instrument", dpi=100)

    # -- G: parallel projection ---------------------------------------------
    sem, buf = render.tilt_sem_of_profile(None, g3, field=T3, level=lvl3, film_nm=100.0,
                                          stage=render.Stage(frame_px=900))
    fig = dark_fig(11, 10.4, "Parallel projection: a 100 nm bar is the same pixels "
                             "wherever it lies",
                   "a SEM's scan angles are a fraction of a degree; bars drawn from world points "
                   "through GeometryBuffers.project, not typed")
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.88])
    show(ax, _stretch(sem.image))
    bars = ((-0.5, -30.0, "#ffffff"), (200.0, 100.0, "#ffd070"), (370.0, 100.0, "#80c8ff"))
    for (y, zz, c) in bars:
        pts = buf.project(np.array([[40.0, y, zz], [140.0, y, zz]]))
        length = np.linalg.norm(pts[1] - pts[0])
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=3)
        ax.annotate(f"100 nm along x at y = {y:.0f}, z = {zz:.0f}  →  {length:.1f} px",
                    xy=pts[1], xytext=(pts[1, 0] + 15, pts[1, 1] - 6), color=c, fontsize=9,
                    family=MONO, bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=2))
    save(fig, out, "parallel", dpi=100)

    # -- H: the pipeline -----------------------------------------------------
    fig = dark_fig(17, 6.2, "The path from the develop step to the picture, and who calls it",
                   "new or changed on 2026-09-05 in blue; unchanged engine in grey")
    ax = fig.add_axes([0, 0, 1, 0.9])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_axis_off()

    def box(x, y, w, h, text, new=True, small=None):
        col = "#80c8ff" if new else "#8a8a8a"
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=BG, ec=col, lw=1.6))
        ax.text(x + w / 2, y + h / 2 + (2.2 if small else 0), text, ha="center", va="center",
                color=INK, fontsize=9.5, family=MONO)
        if small:
            ax.text(x + w / 2, y + h / 2 - 2.6, small, ha="center", va="center", color=col,
                    fontsize=7.8, family=MONO)

    def arrow(x0, y0, x1, y1, text=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=14, color="#c0c0c0", lw=1.3))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.8, text, ha="center", color=INK2,
                    fontsize=7.8, family=MONO)

    box(1, 36, 16, 11, "print_resist_3d\n/ latent + develop_surface",
        small="+ field, level, feature")
    box(21, 36, 15, 11, "profile_surface", small="closed iso-surface, sub-voxel")
    box(40, 36, 16, 11, "render_buffers", small="VTK: normals · depth · material")
    box(60, 36, 14, 11, "tilt_signal / tilt_sem",
        small="secant · bloom · side · shadow · beam · Poisson")
    box(78, 36, 20, 11, "plot_tilt_sem", small="frame, projected bar, data bar")
    arrow(17, 41.5, 21, 41.5)
    arrow(36, 41.5, 40, 41.5)
    arrow(56, 41.5, 60, 41.5)
    arrow(74, 41.5, 78, 41.5)
    box(1, 6, 16, 11, "stochastic_trials_3d", new=False,
        small="arrival field per trial (existing)")
    arrow(17, 11.5, 28.5, 36, "field, level")
    box(24, 6, 22, 11, "Develop tab · 3-D radio",
        small="TiltSemView: stage knobs live, geometry cached")
    box(50, 6, 20, 11, "SEM tab · Instrument",
        small="+ Detector side, Trench shadow → re-forms the picture")
    box(74, 6, 24, 11, "litho-sim demo / stochastic",
        small="resist_profile_3d.png · stochastic_profile_sem.png")
    arrow(48, 36, 35, 17)
    arrow(67, 36, 60, 17)
    arrow(88, 36, 86, 17)
    ax.text(50, 54, "GeometryBuffers carries the camera axes and pixel pitch, so the physics "
                    "stays numpy and testable without VTK, and the figure can project a scale "
                    "bar onto the sample",
            ha="center", color=INK2, fontsize=8.6, family=MONO)
    save(fig, out, "pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
