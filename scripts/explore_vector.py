"""
Interactive vector-imaging bench — drag a slider, watch the physics move.

Run from the repository root::

    python scripts/explore_vector.py              # native window
    python scripts/explore_vector.py --selftest   # headless check, no window

Opens a native window (the macOS backend — no browser, no server, no extra
dependencies) with the four polarisation cases computed side by side, so the
TE/TM split is visible as you move NA rather than something you read off a
table afterwards.

Things worth trying:

* Drag **NA** up from 0.6. The four cross-sections sit on top of each other,
  then fan apart above about 0.9 — that is the vector effect switching on.
* Switch the **medium** between water and resist at NA 1.35. TM contrast
  roughly doubles, because refraction bends the rays back towards the normal
  before they interfere.
* Push **θ** past 45° (NA 1.2 in resist, or NA 1.02 in water) and watch the
  TM cross-section invert — bright where the others are dark.

Grid defaults are chosen so a vector update lands in ~70 ms all four ways;
raise them with --pixels / --source-grid if you want accuracy over feel.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.expose.aerial_image import compute_aerial_image
from litho_sim.mask.patterns import lines_and_spaces

# Order matters: this is the legend order and the bar order.
CASES = (
    ("scalar", "#7f9fd9"),
    ("te", "#4fd97f"),
    ("unpolarised", "#c8913a"),
    ("tm", "#d97f9b"),
)
MEDIA = {"dry (1.00)": 1.00, "water (1.44)": 1.44, "resist (1.70)": 1.70}
SOURCES = {"conventional": ("conventional", {}), "x-dipole": ("dipole", {"axis": "x"})}


def contrast(a: np.ndarray) -> float:
    return float((a.max() - a.min()) / (a.max() + a.min()))


class VectorBench:
    """Holds the parameter state and recomputes the four cases on demand."""

    def __init__(self, n_pixels: int, source_grid: int, pixel_size: float = 4e-9):
        self.grid = GridConfig(n_pixels=n_pixels, pixel_size=pixel_size)
        self.source_grid = source_grid
        self.NA = 1.10
        self.pitch_nm = 100.0
        self.defocus_nm = 0.0
        self.sigma_outer = 0.80
        self.n_image = 1.70
        self.source = "conventional"

    # -- physics ------------------------------------------------------
    def optics(self, case: str) -> OpticsConfig:
        stype, skw = SOURCES[self.source]
        sigma_inner = 0.6 * self.sigma_outer if stype == "dipole" else 0.0
        return OpticsConfig(
            wavelength=193e-9,
            NA=self.NA,
            n_immersion=min(self.n_image, 1.44),
            n_image=self.n_image,
            defocus=self.defocus_nm * 1e-9,
            sigma_outer=self.sigma_outer,
            sigma_inner=sigma_inner,
            source_type=stype,
            source_kwargs=skw,
            source_grid=self.source_grid,
            imaging_model="scalar" if case == "scalar" else "vector",
            polarisation="unpolarised" if case == "scalar" else case,
            normalisation="clear",
        )

    def mask(self) -> np.ndarray:
        n = self.grid.n_pixels
        return lines_and_spaces(
            n, self.grid.pixel_size,
            pitch=self.pitch_nm * 1e-9, cd=self.pitch_nm * 0.5e-9,
        )

    def compute(self) -> dict:
        mask = self.mask()
        out = {}
        for case, _ in CASES:
            out[case] = compute_aerial_image(mask, self.optics(case), self.grid)
        return out

    @property
    def sin_theta(self) -> float:
        return min(self.NA / self.n_image, 1.0)


def build(bench: VectorBench):
    """Wire the figure. Returns (figure, update-callable, widget refs)."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider

    fig = plt.figure(figsize=(13.5, 8.0))
    fig.canvas.manager.set_window_title("LithoPy — vector imaging bench")
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.35, 1.0, 0.9],
        left=0.24, right=0.97, top=0.94, bottom=0.08, hspace=0.45, wspace=0.30,
    )

    ax_img = fig.add_subplot(gs[0, 0])
    ax_cut = fig.add_subplot(gs[0, 1:])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_law = fig.add_subplot(gs[1, 1:])
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")

    # The analytic law is static — draw it once, move only the marker.
    th = np.linspace(0.001, np.pi / 2 - 0.001, 400)
    ax_law.plot(np.degrees(th), np.cos(2 * th), color="#444444", lw=1.4)
    ax_law.axhline(0, color="#999999", lw=0.8)
    ax_law.axvline(45, color="#d9a441", ls=":", lw=1.2)
    (marker,) = ax_law.plot([], [], "o", color="#d97f9b", ms=10, zorder=5)
    ax_law.set(xlabel="half-angle θ [°]", ylabel=r"$\cos 2\theta$",
               title="where this setup sits on the TM interference law",
               xlim=(0, 90), ylim=(-1.1, 1.1))
    ax_law.grid(alpha=0.25)

    # Build every artist once and mutate it in place. Clearing and rebuilding
    # the axes on each update costs several hundred milliseconds, which is the
    # difference between a slider that tracks the mouse and one that lurches.
    n = bench.grid.n_pixels
    extent = [0, n * bench.grid.pixel_size * 1e9] * 2
    x_nm = np.arange(n) * bench.grid.pixel_size * 1e9
    names = [c for c, _ in CASES]

    im = ax_img.imshow(np.zeros((n, n)), origin="lower", extent=extent,
                       cmap="inferno", vmin=0.0, vmax=1.0)
    ax_img.set(xlabel="x [nm]", ylabel="y [nm]")

    lines = {}
    for case, colour in CASES:
        (lines[case],) = ax_cut.plot(x_nm, np.zeros(n), color=colour, lw=2.0,
                                     label=case)
    ax_cut.set(xlabel="x [nm]", ylabel="intensity [clear-field dose]",
               title="cross-section — the four cases together", xlim=(x_nm[0], x_nm[-1]))
    ax_cut.legend(fontsize=8, ncol=4, loc="upper right")
    ax_cut.grid(alpha=0.25)

    bars = ax_bar.bar(range(len(names)), [0] * len(names),
                      color=[c for _, c in CASES])
    ax_bar.set_xticks(range(len(names)), names, fontsize=8, rotation=20)
    ax_bar.set(ylabel="contrast", ylim=(0, 1.05), title="image contrast")
    ax_bar.grid(alpha=0.25, axis="y")

    readout = ax_txt.text(0.0, 0.5, "", fontsize=11, va="center",
                          family="monospace")

    state = {"pol": "te"}

    def update(_=None):
        imgs = bench.compute()
        sel = state["pol"]
        mid = n // 2

        im.set_data(imgs[sel])
        im.set_clim(0.0, max(float(imgs[sel].max()), 1e-9))
        ax_img.set_title(f"aerial image — {sel}")

        top = 0.0
        for case in names:
            cut = imgs[case][mid, :]
            lines[case].set_ydata(cut)
            top = max(top, float(cut.max()))
        ax_cut.set_ylim(0.0, top * 1.25 + 1e-9)

        vals = [contrast(imgs[c]) for c in names]
        for rect, v in zip(bars, vals):
            rect.set_height(v)

        st = bench.sin_theta
        theta = float(np.degrees(np.arcsin(st)))
        marker.set_data([theta], [float(np.cos(2 * np.radians(theta)))])

        ratio = vals[names.index("tm")] / max(vals[names.index("te")], 1e-9)
        readout.set_text(
            f"NA {bench.NA:.2f}   n_image {bench.n_image:.2f}   "
            f"sin θ {st:.3f}   θ {theta:.1f}°   cos 2θ {np.cos(2*np.radians(theta)):+.3f}\n"
            f"pitch {bench.pitch_nm:.0f} nm   defocus {bench.defocus_nm:+.0f} nm   "
            f"σ {bench.sigma_outer:.2f}   {bench.source}\n"
            f"contrast   TE {vals[names.index('te')]:.3f}   "
            f"unpolarised {vals[names.index('unpolarised')]:.3f}   "
            f"TM {vals[names.index('tm')]:.3f}   →  TM/TE = {ratio:.3f}"
        )
        fig.canvas.draw_idle()

    # -- widgets -------------------------------------------------------
    def slider(rect, label, lo, hi, init, step=None, fmt="%.2f"):
        ax = fig.add_axes(rect)
        return Slider(ax, label, lo, hi, valinit=init, valstep=step, valfmt=fmt)

    s_na = slider([0.06, 0.86, 0.13, 0.02], "NA", 0.30, 1.35, bench.NA, 0.01)
    s_pitch = slider([0.06, 0.81, 0.13, 0.02], "pitch [nm]", 60, 400,
                     bench.pitch_nm, 5, "%.0f")
    s_def = slider([0.06, 0.76, 0.13, 0.02], "defocus [nm]", -200, 200,
                   bench.defocus_nm, 10, "%.0f")
    s_sig = slider([0.06, 0.71, 0.13, 0.02], "σ outer", 0.20, 0.95, bench.sigma_outer, 0.05)

    ax_pol = fig.add_axes([0.035, 0.47, 0.16, 0.19])
    ax_pol.set_title("show image", fontsize=9)
    r_pol = RadioButtons(ax_pol, [c for c, _ in CASES], active=1)

    ax_med = fig.add_axes([0.035, 0.28, 0.16, 0.15])
    ax_med.set_title("image medium", fontsize=9)
    r_med = RadioButtons(ax_med, list(MEDIA), active=2)

    ax_src = fig.add_axes([0.035, 0.14, 0.16, 0.11])
    ax_src.set_title("illumination", fontsize=9)
    r_src = RadioButtons(ax_src, list(SOURCES), active=0)

    state["pol"] = "te"

    def on_slider(_):
        bench.NA = float(s_na.val)
        bench.pitch_nm = float(s_pitch.val)
        bench.defocus_nm = float(s_def.val)
        bench.sigma_outer = float(s_sig.val)
        update()

    for s in (s_na, s_pitch, s_def, s_sig):
        s.on_changed(on_slider)

    r_pol.on_clicked(lambda label: (state.__setitem__("pol", label), update()))
    r_med.on_clicked(lambda label: (setattr(bench, "n_image", MEDIA[label]), update()))
    r_src.on_clicked(lambda label: (setattr(bench, "source", label), update()))

    widgets = (s_na, s_pitch, s_def, s_sig, r_pol, r_med, r_src)  # keep alive
    return fig, update, widgets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pixels", type=int, default=96, help="mask grid size")
    ap.add_argument("--source-grid", type=int, default=11,
                    help="illumination sampling; drives the cost")
    ap.add_argument("--selftest", action="store_true",
                    help="drive the callbacks headlessly and exit")
    args = ap.parse_args()

    # An INFO line per aerial image would mean ~50 lines a second while
    # dragging.
    logging.getLogger("litho_sim").setLevel(logging.WARNING)

    import matplotlib

    if args.selftest:
        matplotlib.use("Agg")

    bench = VectorBench(args.pixels, args.source_grid)
    fig, update, _widgets = build(bench)
    update()

    if args.selftest:
        import time

        for na in (0.6, 1.0, 1.35):
            bench.NA = na
            t0 = time.perf_counter()
            update()
            dt = (time.perf_counter() - t0) * 1000
            imgs = bench.compute()
            print(f"  NA {na:.2f}  redraw {dt:6.1f} ms   "
                  f"TE {contrast(imgs['te']):.3f}  TM {contrast(imgs['tm']):.3f}")
        for medium in MEDIA:
            bench.n_image = MEDIA[medium]
            update()
        for src in SOURCES:
            bench.source = src
            update()
        print("selftest OK — widgets wired, no window opened")
        return

    import matplotlib.pyplot as plt

    print(f"backend: {matplotlib.get_backend()}   "
          f"grid {args.pixels}px, source_grid {args.source_grid}")
    print("Drag NA up from 0.6 and watch the four cases fan apart.")
    plt.show()


if __name__ == "__main__":
    main()
