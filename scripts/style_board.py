"""Render every figure the theme governs onto one page, for looking at.

    python scripts/style_board.py                # writes results/style_board.png
    python scripts/style_board.py --out x.png

The board is how the figure style is judged: every ``viz.plots`` function and
one of each desktop-app panel, on synthetic data that exercises the awkward
cases (a dose that never prints, a caption beside a legend, a colourbar on a
square image). Check it against the rules in ``viz/theme.py``'s docstring —
nothing dashed, no numbers in a title, no legend over data, units on every
colour scale, every image in nanometres.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from litho_sim.core.config import GridConfig  # noqa: E402
from litho_sim.viz import plots, theme  # noqa: E402


def _synthetic_sweep(target: float = 100.0) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    doses = np.round(np.linspace(0.64, 1.18, 7), 2)
    defoci = np.linspace(-300, 300, 11)
    rows = []
    for d in doses:
        for f in defoci:
            # A Bossung: CD falls with dose, bows with focus, and a low dose
            # stops printing off-focus so one curve has a genuine gap.
            cd = target * (1.65 - 0.9 * d) - 1.2e-4 * f**2 * (1.4 - d)
            nils = 2.4 * np.exp(-(f / 260) ** 2)
            if cd < 15 or (d < 0.7 and abs(f) > 200):
                cd = 0.0
            rows.append({"dose": d, "defocus_nm": f, "cd_nm": cd, "nils": nils})
    df = pd.DataFrame(rows)
    lo, hi = target * 0.9, target * 1.1
    df["in_spec"] = (df["cd_nm"] >= lo) & (df["cd_nm"] <= hi)
    pw = {
        "window_df": df[["dose", "defocus_nm", "in_spec"]].copy(),
        "best_focus_nm": 0.0, "best_dose": 0.91, "EL_pct": 24.0,
        "DOF_nm": 480.0, "area": 11520.0, "prints": True,
    }
    el_dof = pd.DataFrame({
        "dof_nm": np.arange(0, 481, 60),
        "el_pct": [20, 20, 18, 15, 12, 10, 8, 4, 0],
    })
    return df, pw, el_dof


def _synthetic_print(n: int = 128, pitch_nm: float = 200.0, px_nm: float = 4.0):
    grid = GridConfig(n_pixels=n, pixel_size=px_nm * 1e-9)
    x = (np.arange(n) - n // 2) * px_nm
    aerial_1d = 0.5 + 0.42 * np.cos(2 * np.pi * x / pitch_nm)
    aerial = np.tile(aerial_1d, (n, 1))
    resist = (aerial < 0.45).astype(float)
    return grid, aerial, resist


def build(out: Path) -> Path:
    import matplotlib.pyplot as plt

    theme.apply()
    df, pw, el_dof = _synthetic_sweep()
    grid, aerial, resist = _synthetic_print()

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    plots.plot_aerial_image(aerial, grid, ax=fig.add_subplot(gs[0, 0]))
    plots.plot_resist_profile(aerial, resist, grid, threshold=0.45,
                              ax=fig.add_subplot(gs[0, 1:]))
    plots.plot_bossung_curves(df, 100.0, 10.0, best_focus_nm=0.0,
                              ax=fig.add_subplot(gs[1, 0]))
    plots.plot_cd_heatmap(df, 100.0, 10.0, ax=fig.add_subplot(gs[1, 1]))
    plots.plot_process_window(pw, 100.0, ax=fig.add_subplot(gs[1, 2]))
    plots.plot_el_dof_curve(el_dof, ax=fig.add_subplot(gs[2, 0]))
    plots.plot_nils_through_focus(df, dose=0.91, ax=fig.add_subplot(gs[2, 1]))

    # The material palette, as the sections and the solid view will show it.
    from litho_sim.wafer import MATERIAL_LIBRARY

    ax = fig.add_subplot(gs[2, 2])
    mats = [m for m in MATERIAL_LIBRARY.values() if m.name != "vacuum"]
    for i, m in enumerate(mats):
        ax.barh(i, 1.0, color=m.color, height=0.8, lw=0)
        ax.text(1.03, i, m.name, va="center", fontsize=8.5, color=theme.INK2)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(-0.6, len(mats) - 0.4)
    ax.invert_yaxis()
    ax.set_axis_off()
    theme.title(ax, "Material palette", caption_text=f"{len(mats)} materials")

    theme.caption(fig, "style board — synthetic data, every viz.plots figure on one page")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Render every themed figure onto one page.")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "style_board.png")
    a = ap.parse_args()
    print(build(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
