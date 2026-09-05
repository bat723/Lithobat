"""The static figures keep to the theme's rules.

These are not pixel tests. They assert the *structure* the rules produce —
where the numbers went, that a dose is named at its line and not in a box,
that a colour scale says its units — because those are the things that made
the earlier figures read badly and are the things a restyle quietly loses.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex, to_rgb

from litho_sim.core.config import GridConfig
from litho_sim.viz import plots, theme

TARGET = 100.0
DOSES = np.round(np.linspace(0.64, 1.18, 7), 2)


@pytest.fixture(autouse=True)
def _themed():
    theme.apply()
    yield
    plt.close("all")


@pytest.fixture
def sweep() -> pd.DataFrame:
    rows = []
    for d in DOSES:
        for f in np.linspace(-300, 300, 11):
            cd = TARGET * (1.65 - 0.9 * d) - 1.2e-4 * f**2 * (1.4 - d)
            if d < 0.7 and abs(f) > 200:      # the low dose stops printing off focus
                cd = 0.0
            rows.append({"dose": d, "defocus_nm": f,
                         "cd_nm": 0.0 if cd < 15 else cd,
                         "nils": 2.4 * np.exp(-(f / 260) ** 2)})
    return pd.DataFrame(rows)


@pytest.fixture
def window(sweep) -> dict:
    df = sweep.copy()
    df["in_spec"] = (df["cd_nm"] - TARGET).abs() <= 0.1 * TARGET
    return {"window_df": df, "best_focus_nm": 0.0, "best_dose": 0.91,
            "EL_pct": 24.0, "DOF_nm": 480.0, "area": 11520.0}


@pytest.fixture
def print_() -> tuple[GridConfig, np.ndarray, np.ndarray]:
    n = 64
    grid = GridConfig(n_pixels=n, pixel_size=4e-9)
    x = (np.arange(n) - n // 2) * 4.0
    aerial = np.tile(0.5 + 0.42 * np.cos(2 * np.pi * x / 200.0), (n, 1))
    return grid, aerial, (aerial < 0.45).astype(float)


def _every_figure(sweep, window, print_):
    grid, aerial, resist = print_
    el_dof = pd.DataFrame({"dof_nm": [0, 60, 120], "el_pct": [20.0, 10.0, 0.0]})
    yield plots.plot_aerial_image(aerial, grid)
    yield plots.plot_resist_profile(aerial, resist, grid, threshold=0.45)
    yield plots.plot_bossung_curves(sweep, TARGET, 10.0, best_focus_nm=0.0)
    yield plots.plot_cd_heatmap(sweep, TARGET, 10.0)
    yield plots.plot_process_window(window, TARGET)
    yield plots.plot_el_dof_curve(el_dof)
    yield plots.plot_nils_through_focus(sweep, dose=0.91)


def test_apply_is_idempotent_and_installs_a_solid_hairline_grid():
    theme.apply()
    before = {k: matplotlib.rcParams[k] for k in ("grid.linestyle", "axes.grid",
                                                   "axes.titlelocation")}
    theme.apply()
    after = {k: matplotlib.rcParams[k] for k in before}
    assert before == after
    assert after["grid.linestyle"] == "-", "gridlines are never dashed"
    assert after["axes.grid"] is False, "grid is opt-in, per axes"
    assert after["axes.titlelocation"] == "left"
    assert "litho.cd" in matplotlib.colormaps


def test_importing_the_theme_or_plots_changes_nothing():
    """The theme lands when asked for, not as an import side effect."""
    import importlib
    import subprocess
    import sys

    code = ("import matplotlib; a = matplotlib.rcParams['grid.linestyle'];"
            "import litho_sim.viz.plots, litho_sim.viz.theme;"
            "b = matplotlib.rcParams['grid.linestyle']; print(a == b)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "True"
    importlib.import_module("litho_sim.viz.plots")


def test_bossung_names_each_dose_at_its_line_not_in_a_legend(sweep):
    fig, ax = plots.plot_bossung_curves(sweep, TARGET, 10.0, best_focus_nm=0.0)
    assert ax.get_legend() is None
    texts = {t.get_text() for t in ax.texts}
    for d in DOSES:
        assert f"{d:.2f}" in texts, f"dose {d} is not named on the plot"
    # The colours step along one hue: lightness falls with dose.
    lines = [ln for ln in ax.get_lines() if np.size(ln.get_xdata()) > 2]
    lum = [np.mean(to_rgb(ln.get_color())) for ln in lines]
    assert lum == sorted(lum, reverse=True), "doses are an ordinal ramp, light to dark"


def test_bossung_draws_a_gap_where_nothing_printed(sweep):
    fig, ax = plots.plot_bossung_curves(sweep, TARGET, 10.0)
    lines = [ln for ln in ax.get_lines() if np.size(ln.get_xdata()) > 2]
    ys = [np.asarray(ln.get_ydata(), dtype=float) for ln in lines]
    assert any(np.isnan(y).any() for y in ys)
    assert not any((y == 0.0).any() for y in ys), \
        "a CD of zero is not a printed CD and must not be plotted as one"


def test_cd_map_colour_scale_carries_units_and_a_window_outline(sweep):
    fig, ax = plots.plot_cd_heatmap(sweep, TARGET, 10.0)
    im = ax.images[0]
    assert im.colorbar is not None
    assert "nm" in im.colorbar.ax.get_ylabel()
    assert ax.collections, "the in-spec window is outlined"
    assert ax.get_xlabel() and ax.get_ylabel()


def test_aerial_image_axes_are_physical_with_a_labelled_scale(print_):
    grid, aerial, _ = print_
    fig, ax = plots.plot_aerial_image(aerial, grid)
    im = ax.images[0]
    half = grid.grid_size / 2 * 1e9
    assert tuple(im.get_extent()) == pytest.approx((-half, half, -half, half))
    assert ax.get_xlabel() == "x [nm]"
    assert im.colorbar is not None and "intensity" in im.colorbar.ax.get_ylabel()


def test_titles_are_names_and_numbers_go_to_the_caption(sweep, window, print_):
    for fig, _ax in _every_figure(sweep, window, print_):
        for a in fig.axes:
            title = a.get_title(loc="left")
            assert not any(ch.isdigit() for ch in title), \
                f"a readout in a title: {title!r}"
    fig, ax = plots.plot_process_window(window, TARGET)
    cap = getattr(ax, "_litho_caption").get_text()
    assert "EL 24.0 %" in cap and "DOF 480 nm" in cap
    assert ax.get_title(loc="left") == "Process window"


def test_nothing_is_dashed_and_gridlines_are_hairlines(sweep, window, print_):
    for fig, _ax in _every_figure(sweep, window, print_):
        for a in fig.axes:
            for gl in a.get_xgridlines() + a.get_ygridlines():
                if gl.get_visible():
                    assert gl.get_linestyle() == "-"
                    assert gl.get_linewidth() <= 1.0


def test_a_legend_only_exists_for_two_series_and_sits_off_the_data(print_):
    grid, aerial, resist = print_
    fig, ax = plots.plot_resist_profile(aerial, resist, grid, threshold=0.45)
    leg = ax.get_legend()
    assert leg is not None
    fig.canvas.draw()
    lb, ab = leg.get_window_extent(), ax.get_window_extent()
    assert lb.y0 >= ab.y1 - 1.0, "the legend sits above the axes, not on the data"

    el_dof = pd.DataFrame({"dof_nm": [0, 60], "el_pct": [20.0, 0.0]})
    fig, ax = plots.plot_el_dof_curve(el_dof)
    assert ax.get_legend() is None, "one series is named by the title"


def test_el_dof_says_when_there_is_no_window():
    fig, ax = plots.plot_el_dof_curve(pd.DataFrame({"dof_nm": [], "el_pct": []}))
    assert any("no in-spec window" in t.get_text() for t in ax.texts)


def test_binary_cmap_is_two_flat_colours():
    cmap, norm = theme.binary_cmap("#1baf7a")
    assert cmap.N == 2
    assert to_hex(cmap(norm(1))) == "#1baf7a"
    assert to_hex(cmap(norm(0))) == theme.SURFACE


def test_ordinal_colours_step_one_hue_light_to_dark():
    cols = theme.ordinal_colors(5)
    lum = [np.mean(to_rgb(c)) for c in cols]
    assert lum == sorted(lum, reverse=True)
    assert len(set(cols)) == 5
