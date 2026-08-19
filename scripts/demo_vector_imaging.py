"""
The vector effect: why hyper-NA lithography is polarised.

Run from the repository root::

    python scripts/demo_vector_imaging.py

Writes ``results/vector_effect.png``.

Nothing here computes "contrast loss". It falls out of one thing: the
electric field has to stay perpendicular to the ray, so when a lens bends two
rays towards each other their fields stop being parallel — unless the field
happens to point along the axis they rotate about, which is what s (TE)
polarisation means. The engine is only told to track three field components
instead of one; the cos 2θ law, the null at 45°, and the contrast inversion
past it are all consequences.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.core.utils import setup_logging
from litho_sim.expose.aerial_image import compute_aerial_image
from litho_sim.expose.pupil import jones_states, vector_coefficients
from litho_sim.mask.patterns import lines_and_spaces

OUT = Path("results/vector_effect.png")

TE_C, TM_C, UN_C, SC_C = "#4fd97f", "#d97f9b", "#c8913a", "#7f9fd9"


def two_beam_overlap(polarisation: str, sin_theta: float) -> float:
    """Interference term for two rays at ±θ, straight from the coefficients."""
    phi = np.array([0.0, np.pi])
    rho = np.array([1.0, 1.0])
    (jx, jy, _), = jones_states(polarisation, 0.0)
    V = np.array(vector_coefficients(rho, phi, jx, jy, sin_theta, obliquity=False))
    return float(
        np.vdot(V[:, 0], V[:, 1]).real
        / (np.linalg.norm(V[:, 0]) * np.linalg.norm(V[:, 1]))
    )


def contrast(a) -> float:
    return float((a.max() - a.min()) / (a.max() + a.min()))


def main() -> None:
    setup_logging()
    import logging

    logging.getLogger("litho_sim").setLevel(logging.WARNING)

    # ---- panel 1: the law, at the level where it is analytic -------------
    st = np.linspace(0.01, 0.995, 200)
    theta = np.arcsin(st)
    te_law = [two_beam_overlap("y", s) for s in st]
    tm_law = [two_beam_overlap("x", s) for s in st]

    # ---- panel 2: the same thing through the imaging engine --------------
    grid = GridConfig(n_pixels=128, pixel_size=4e-9)
    mask = lines_and_spaces(128, 4e-9, pitch=100e-9, cd=50e-9)
    nas = [0.60, 0.80, 0.95, 1.10, 1.25, 1.35]
    curves = {"te": [], "unpolarised": [], "tm": []}
    print("\n=== Contrast vs NA (x-dipole, dense 100 nm lines, in resist) ===")
    for NA in nas:
        row = []
        for pol in curves:
            o = OpticsConfig(
                NA=NA, n_immersion=1.44, n_image=1.70, source_grid=21,
                sigma_outer=0.9, sigma_inner=0.6, source_type="dipole",
                source_kwargs={"axis": "x"}, imaging_model="vector",
                polarisation=pol,
            )
            c = contrast(compute_aerial_image(mask, o, grid))
            curves[pol].append(c)
            row.append(c)
        print(f"  NA {NA:.2f}  TE {row[0]:.3f}   unpol {row[1]:.3f}   TM {row[2]:.3f}")

    # ---- panel 3: images at the immersion preset -------------------------
    imgs = {}
    for pol in ("te", "tm"):
        o = OpticsConfig(
            NA=1.35, n_immersion=1.44, n_image=1.70, source_grid=21,
            sigma_outer=0.9, sigma_inner=0.6, source_type="dipole",
            source_kwargs={"axis": "x"}, imaging_model="vector",
            polarisation=pol, normalisation="clear",
        )
        imgs[pol] = compute_aerial_image(mask, o, grid)

    # ---- panel 4: water vs resist ---------------------------------------
    print("\n=== Refraction into the resist protects contrast ===")
    media = []
    for n_img, label in ((1.44, "in water"), (1.70, "in resist")):
        vals = []
        for pol in ("te", "tm"):
            o = OpticsConfig(
                NA=1.35, n_immersion=1.44, n_image=n_img, source_grid=21,
                sigma_outer=0.9, sigma_inner=0.6, source_type="dipole",
                source_kwargs={"axis": "x"}, imaging_model="vector",
                polarisation=pol,
            )
            vals.append(contrast(compute_aerial_image(mask, o, grid)))
        media.append((label, 1.35 / n_img, vals))
        print(f"  {label:10s} sinθ={1.35/n_img:.3f}  TE {vals[0]:.3f}  TM {vals[1]:.3f}")

    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, :2])
    ax.axhline(0, color="#7f7f7f", lw=0.8)
    ax.plot(np.degrees(theta), te_law, color=TE_C, lw=2.4, label="TE (s) — engine")
    ax.plot(np.degrees(theta), tm_law, color=TM_C, lw=2.4, label="TM (p) — engine")
    ax.plot(np.degrees(theta), np.cos(2 * theta), "--", color="#111111", lw=1.3,
            dashes=(4, 4), label=r"analytic $\cos 2\theta$")
    ax.axvline(45, color="#d9a441", ls=":", lw=1.4)
    ax.annotate("TM null at 45°\nbeyond here contrast inverts", xy=(45, 0),
                xytext=(50, 0.45), fontsize=9, color="#d9a441",
                arrowprops=dict(arrowstyle="->", color="#d9a441", lw=1.2))
    for label, s, _ in media:
        ax.axvline(np.degrees(np.arcsin(s)), color="#7f9fd9", ls="-.", lw=1.0)
        ax.text(np.degrees(np.arcsin(s)) + 0.6, -0.85, label, fontsize=8,
                color="#7f9fd9", rotation=90, va="bottom")
    ax.set(xlabel="half-angle between the two beams, θ [°]",
           ylabel="interference term",
           title="The vector effect is one line of geometry: "
                 r"two p-polarised beams interfere as $\cos 2\theta$")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 2])
    for pol, c, lbl in (("te", TE_C, "TE"), ("unpolarised", UN_C, "unpolarised"),
                        ("tm", TM_C, "TM")):
        ax.plot(nas, curves[pol], "o-", color=c, lw=2, ms=5, label=lbl)
    ax.set(xlabel="NA", ylabel="image contrast",
           title="Through the engine\n(x-dipole, 100 nm pitch, in resist)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    extent = [0, 128 * 4e-9 * 1e9] * 2
    for i, pol in enumerate(("te", "tm")):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(imgs[pol], origin="lower", extent=extent, cmap="inferno")
        ax.set(title=f"NA 1.35 in resist — {pol.upper()}  "
                     f"(contrast {contrast(imgs[pol]):.2f})",
               xlabel="x [nm]", ylabel="y [nm]" if i == 0 else None)
        fig.colorbar(im, ax=ax, shrink=0.85, label="clear-field dose")

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [media[0][2][0], media[1][2][0]], w, color=TE_C, label="TE")
    ax.bar(x + w / 2, [media[0][2][1], media[1][2][1]], w, color=TM_C, label="TM")
    ax.set_xticks(x, [f"{m[0]}\nsinθ={m[1]:.2f}" for m in media])
    ax.set(ylabel="image contrast",
           title="Refraction into the resist\nprotects contrast")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
