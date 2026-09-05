"""
Public-API tests for optical proximity correction.

Everything goes through ``litho_sim.opc``. Four contracts are pinned:

* **Fragmentation is exact.** Rebuilding a fragmented shape at zero offset
  gives the drawn polygon back, a uniform bias is an exact analytic offset
  (sub-pixel, unlike the pixel-based ``apply_bias``), and a single moved
  fragment produces the jog a corrected mask is made of.
* **EPE has the sign OPC needs.** A feature drawn too small measures
  negative, too big positive, on a field built by hand — so the loop's
  ``−gain × EPE`` move is in the right direction before any optics enter.
* **The loop reduces what it measures.** On a layout with a dense array,
  an isolated line and pulled-back line ends, model-based OPC lowers the
  worst and the RMS edge placement error, the mask rule holds, and the
  corrected layout is an ordinary ``Layout`` a flow can expose.
* **Assist features stay sub-resolution.** Scattering bars land only where
  there is room and do not print at the dose the design was corrected at.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.core.config import GridConfig, SimulationConfig
from litho_sim.mask import Contact, Layout, Rect, line_array
from litho_sim.opc import (
    EPE,
    PrintedImage,
    PrintModel,
    add_scattering_bars,
    assist_features_printed,
    bias_layout,
    fragment_layout,
    fragment_shape,
    measure_epe,
    rebuild_layout,
    run_opc,
    verify,
)
from litho_sim.opc.fragments import facing_distances
from litho_sim.patterning import Develop, Expose, Flow, SpinCoat
from litho_sim.wafer import Stack


def _area(points) -> float:
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=192, pixel_size=4e-9)


@pytest.fixture(scope="module")
def model(grid) -> PrintModel:
    cfg = SimulationConfig.from_tech_node("ArF")
    m = PrintModel(cfg.optics, cfg.resist, grid, tone="dark", model="threshold")
    anchor = Layout(line_array(5, pitch=200e-9, cd=100e-9, length=700e-9), name="anchor")
    m.dose = m.dose_to_size(anchor, 100.0)
    return m


def demo_layout() -> Layout:
    """Two dense lines, an isolated line, all with line ends inside the field."""
    dense = line_array(2, pitch=200e-9, cd=100e-9, length=500e-9, centre=(-200e-9, 0))
    iso = [Rect("main", 220e-9, 0, 100e-9, 500e-9)]
    return Layout(dense + iso, name="demo")


# ---------------------------------------------------------------------------
# Fragmentation and rebuild
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [
    Rect("main", 10e-9, -20e-9, 100e-9, 400e-9),
    Rect("main", 0, 0, 100e-9, 300e-9, rotation=30.0),
    Contact("main", 0, 0, 60e-9, "round"),
    Contact("main", 5e-9, 5e-9, 60e-9, "octagon"),
])
def test_rebuild_at_zero_offset_is_the_drawn_shape(shape):
    fs = fragment_shape(shape, 40e-9, 20e-9)
    rebuilt = fs.shape()
    assert rebuilt.layer == shape.layer
    assert _area(rebuilt.points) == pytest.approx(_area(shape.polygon()), rel=1e-9)
    assert rebuilt.bounds() == pytest.approx(shape.bounds(), abs=1e-15)


def test_fragments_have_outward_normals_that_cancel():
    fs = fragment_shape(Rect("main", 0, 0, 100e-9, 400e-9), 40e-9, 20e-9)
    # Weighted by length the normals of a closed outline sum to zero.
    total = sum(f.length * f.normal for f in fs.fragments)
    assert np.allclose(total, 0.0, atol=1e-18)
    # And a rectangle's bottom edge points down.
    bottom = [f for f in fs.fragments if f.edge == 0][0]
    assert np.allclose(bottom.normal, [0.0, -1.0])


def test_corner_fragments_bracket_every_edge():
    fs = fragment_shape(Rect("main", 0, 0, 100e-9, 400e-9), 40e-9, 20e-9)
    for e in range(4):
        roles = [f.role for f in fs.fragments if f.edge == e]
        assert roles[0] == "corner" and roles[-1] == "corner"
        assert all(r == "body" for r in roles[1:-1])
    short = fragment_shape(Rect("main", 0, 0, 30e-9, 400e-9), 40e-9, 20e-9)
    assert [f.role for f in short.fragments if f.edge == 0] == ["end"]


def test_corner_fragments_are_retargeted_and_body_is_not():
    fs = fragment_shape(Rect("main", 0, 0, 100e-9, 400e-9), 52e-9, 26e-9, corner_radius=52e-9)
    corners = [f for f in fs.fragments if f.role == "corner"]
    bodies = [f for f in fs.fragments if f.role == "body"]
    assert all(f.retargeted for f in corners)
    assert not any(f.retargeted for f in bodies)
    # A retargeted site sits inside the drawn outline, along the arc normal.
    f = corners[0]
    assert float((f.site - f.midpoint) @ f.normal) < 0.0
    assert float(f.site_normal @ f.normal) > 0.5
    # Two fragments meeting at a vertex get two different sites.
    at_vertex = [f for f in corners if np.allclose(np.abs(f.midpoint) > [30e-9, 180e-9], True)]
    sites = {tuple(np.round(f.site * 1e9, 3)) for f in at_vertex}
    assert len(sites) == len(at_vertex)
    # Sharp corners on request.
    sharp = fragment_shape(Rect("main", 0, 0, 100e-9, 400e-9), 52e-9, 26e-9, corner_radius=0.0)
    assert not any(f.retargeted for f in sharp.fragments)


def test_uniform_bias_is_exact_and_sub_pixel():
    lay = Layout([Rect("main", 0, 0, 100e-9, 400e-9), Contact("via", 200e-9, 0, 40e-9)])
    biased = bias_layout(lay, 0.7e-9)
    x0, y0, x1, y1 = biased.shapes[0].bounds()
    assert (x1 - x0) == pytest.approx(101.4e-9, abs=1e-15)
    assert (y1 - y0) == pytest.approx(401.4e-9, abs=1e-15)
    assert len(biased.shapes[0].points) == 4          # no jogs: every edge moved alike
    assert biased.shapes[1].layer == "via"
    shrunk = bias_layout(lay, -10e-9)
    x0, y0, x1, y1 = shrunk.shapes[0].bounds()
    assert (x1 - x0) == pytest.approx(80e-9, abs=1e-15)


def test_one_moved_fragment_makes_a_jog():
    fs = fragment_shape(Rect("main", 0, 0, 100e-9, 400e-9), 40e-9, 20e-9)
    body = [i for i, f in enumerate(fs.fragments) if f.edge == 1 and f.role == "body"]
    fs.offsets[body[0]] = 8e-9
    poly = fs.shape()
    assert len(poly.points) == 8                      # 4 corners + 4 jog vertices
    x0, _, x1, _ = poly.bounds()
    assert x1 == pytest.approx(58e-9, abs=1e-15)      # the jog sticks out by the offset
    assert x0 == pytest.approx(-50e-9, abs=1e-15)
    # And the rebuilt layout rasterises like any other.
    grid = GridConfig(n_pixels=128, pixel_size=4e-9)
    mask = rebuild_layout([fs]).rasterize(grid, tone="dark")
    assert mask.shape == (128, 128) and 0.0 <= mask.min() and mask.max() <= 1.0


def test_corrected_layout_round_trips_through_json(tmp_path):
    fs = fragment_shape(Rect("main", 0, 0, 100e-9, 400e-9), 40e-9, 20e-9)
    fs.offsets[:] = np.linspace(-5e-9, 5e-9, len(fs))
    lay = rebuild_layout([fs], name="corrected")
    path = tmp_path / "opc.json"
    lay.to_json(path)
    back = Layout.from_json(path)
    assert np.allclose(np.asarray(back.shapes[0].points), np.asarray(lay.shapes[0].points))


def test_facing_distances_see_the_neighbour_and_the_own_width():
    lay = Layout(line_array(2, pitch=200e-9, cd=100e-9, length=600e-9))
    frags = fragment_layout(lay, 52e-9, 26e-9)
    space = facing_distances(frags, +1.0)
    width = facing_distances(frags, -1.0)
    normals = np.concatenate([fs.normals for fs in frags])
    inward_x = np.abs(normals[:, 0]) > 0.5
    # Across the space between the two lines: 100 nm; on the outer sides: nothing.
    assert set(np.round(space[inward_x] * 1e9, 6)) == {100.0, np.inf}
    # Every side fragment looks across its own 100 nm line.
    assert np.allclose(width[inward_x], 100e-9)
    # Line ends face nothing and look down the 600 nm line.
    assert np.all(np.isinf(space[~inward_x]))
    assert np.allclose(width[~inward_x], 600e-9)


# ---------------------------------------------------------------------------
# EPE on a field built by hand
# ---------------------------------------------------------------------------


def _synthetic_print(grid: GridConfig, half_width: float, tone: str) -> PrintedImage:
    """A latent field whose printed feature is a vertical band of ±half_width."""
    n, px = grid.n_pixels, grid.pixel_size
    x = (np.arange(n) - n // 2) * px
    # A smooth trapezoid: 1 inside, 0 outside, edge width 20 nm.
    edge = 20e-9
    inside = np.clip((half_width - np.abs(x)) / edge + 0.5, 0.0, 1.0)
    field = np.tile(inside, (n, 1))
    if tone == "dark":
        field = 1.0 - field          # a dark-tone feature is the *low* side
    return PrintedImage(field, field, field, 0.5, tone, grid)


@pytest.mark.parametrize("tone", ["clear", "dark"])
def test_epe_sign_follows_the_printed_edge(tone):
    grid = GridConfig(n_pixels=128, pixel_size=4e-9)
    drawn = Layout([Rect("main", 0, 0, 100e-9, 300e-9)])
    frags = fragment_layout(drawn, 40e-9, 0.0, corner_radius=0.0)
    sides = np.abs(np.concatenate([fs.normals for fs in frags])[:, 0]) > 0.5
    for printed_half, expected in ((56e-9, +6e-9), (44e-9, -6e-9), (50e-9, 0.0)):
        epe = measure_epe(_synthetic_print(grid, printed_half, tone), frags, search=60e-9)
        assert np.allclose(epe.values[sides], expected, atol=0.2e-9)
        assert all(s == "ok" for s, side in zip(epe.status, sides) if side)
        # The band never ends, so the line-end fragments correctly report
        # a feature that runs past the search range.
        assert all(s == "merged" for s, side in zip(epe.status, sides) if not side)


def test_epe_reports_missing_and_merged():
    grid = GridConfig(n_pixels=128, pixel_size=4e-9)
    drawn = Layout([Rect("main", 0, 0, 100e-9, 300e-9)])
    frags = fragment_layout(drawn, 40e-9, 0.0, corner_radius=0.0)
    sides = np.abs(np.concatenate([fs.normals for fs in frags])[:, 0]) > 0.5
    gone = measure_epe(_synthetic_print(grid, 0.0, "clear"), frags, search=30e-9)
    assert all(s == "missing" for s, side in zip(gone.status, sides) if side)
    assert np.isnan(gone.values[sides]).all()
    huge = measure_epe(_synthetic_print(grid, 200e-9, "clear"), frags, search=30e-9)
    assert all(s == "merged" for s, side in zip(huge.status, sides) if side)
    assert gone.n_failed > 0 and huge.n_failed > 0


def test_epe_stats_ignore_failed_sites():
    e = EPE(np.array([1e-9, -3e-9, np.nan]), ["ok", "ok", "missing"], np.zeros(3))
    assert e.max_abs == pytest.approx(3e-9)
    assert e.rms == pytest.approx(np.sqrt(5.0) * 1e-9)
    assert e.stats()["n_failed"] == 1


# ---------------------------------------------------------------------------
# The print model
# ---------------------------------------------------------------------------


def test_print_model_forces_clear_field_normalisation(grid):
    cfg = SimulationConfig.from_tech_node("ArF")
    assert cfg.optics.normalisation == "peak"
    m = PrintModel(cfg.optics, cfg.resist, grid, tone="dark")
    assert m.optics.normalisation == "clear"
    kept = PrintModel(cfg.optics, cfg.resist, grid, tone="dark", normalisation=None)
    assert kept.optics.normalisation == "peak"
    with pytest.raises(ValueError):
        PrintModel(cfg.optics, cfg.resist, grid, tone="bright")


def test_dose_to_size_prints_the_drawn_feature_to_size(model, grid):
    anchor = Layout(line_array(5, pitch=200e-9, cd=100e-9, length=700e-9))
    printed = model.print(anchor)
    mid = grid.n_pixels // 2
    row = printed.printed[mid, :]
    # The centre line's printed width, in pixels, is the drawn 100 nm ± a pixel.
    centre = np.nonzero(row)[0]
    centre = centre[np.abs(centre - mid) < 25]
    assert len(centre) * grid.pixel_size == pytest.approx(100e-9, abs=grid.pixel_size)
    # And what printed is the drawn feature: a dark-tone line is the low side.
    assert printed.signed[mid, mid] > 0.0 and printed.signed[mid, mid + 25] < 0.0


def test_printed_contours_trace_the_feature(model):
    lay = Layout([Rect("main", 0, 0, 100e-9, 400e-9)])
    printed = model.print(lay)
    lines = printed.contours()
    assert lines, "no contour found"
    longest = max(lines, key=len)
    x0, y0 = longest.min(axis=0)
    x1, y1 = longest.max(axis=0)
    assert (x1 - x0) == pytest.approx(100e-9, abs=8e-9)
    assert (y1 - y0) < 400e-9                           # line-end pullback


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corrected(model):
    return run_opc(demo_layout(), model, max_iter=10)


def test_opc_reduces_the_error_it_measures(corrected):
    before, after = corrected.epe_before, corrected.epe_after
    assert after.n_failed == 0
    assert after.max_abs < 0.25 * before.max_abs
    assert after.rms < 0.25 * before.rms
    h = corrected.history
    assert list(h.columns) == [
        "iteration", "max_abs_epe_nm", "rms_epe_nm", "n_failed", "max_abs_offset_nm",
    ]
    assert h.max_abs_epe_nm.iloc[-1] < h.max_abs_epe_nm.iloc[0]
    assert corrected.epe_trace.shape == corrected.offset_trace.shape == (len(h), len(before.values))


def test_line_ends_needed_the_most_correction(corrected):
    df = corrected.summary()
    ends = df[(df.ny.abs() > 0.5) & (df.role == "body")]           # end-edge bodies
    sides = df[(df.nx.abs() > 0.99) & (df.role == "body")]         # long-edge bodies
    assert ends.epe_before_nm.mean() < -10.0                       # pulled back
    assert ends.offset_nm.mean() > 5.0                             # extended
    assert ends.offset_nm.abs().mean() > sides.offset_nm.abs().mean()


def test_the_mask_rule_holds(corrected, model):
    rule = corrected.settings["min_space"]
    frags = corrected.fragmented
    space = facing_distances(frags, +1.0)
    offs = corrected.offsets
    # Facing fragments each took at most their half of the space above the rule.
    finite = np.isfinite(space)
    assert np.all(offs[finite] <= 0.5 * (space[finite] - rule) + 1e-15)
    assert np.all(np.abs(offs) <= corrected.settings["max_offset"] + 1e-15)


def test_corrected_layout_exposes_in_a_flow(corrected, model, grid):
    lay = corrected.corrected
    assert lay.layers() == ["main"]
    assert all(type(s).__name__ == "Polygon" for s in lay.shapes)
    flow = Flow(
        steps=[SpinCoat(material="photoresist", thickness=100e-9),
               Expose(layout="opc", dose=model.dose, tone="dark"),
               Develop()],
        layouts={"opc": lay}, grid=grid, optics=model.optics, resist=model.resist,
        name="opc-print",
    )
    result = flow.run(Stack.blank(grid, dz=4e-9, headroom=200e-9))
    assert result.stack.mat.any()


def test_verify_matches_the_loops_own_first_measurement(model):
    lay = demo_layout()
    printed, epe, frags = verify(lay, model)
    res = run_opc(lay, model, max_iter=0)
    assert np.allclose(epe.values, res.epe_before.values, equal_nan=True)
    assert epe.status == res.epe_before.status
    assert not res.converged and len(res.history) == 1


def test_only_the_named_layer_is_corrected(model):
    lay = demo_layout()
    lay.add(Rect("cut", 0, 0, 40e-9, 40e-9))
    res = run_opc(lay, model, layer="main", max_iter=1)
    assert [s.layer for s in res.corrected.shapes].count("cut") == 1
    cut = [s for s in res.corrected.shapes if s.layer == "cut"][0]
    assert isinstance(cut, Rect)                       # untouched, not even rebuilt


# ---------------------------------------------------------------------------
# Assist features
# ---------------------------------------------------------------------------


def sraf_layout() -> Layout:
    """The dense pair with the isolated line 200 nm off its right edge."""
    dense = line_array(2, pitch=200e-9, cd=100e-9, length=500e-9, centre=(-200e-9, 0))
    return Layout(dense + [Rect("main", 200e-9, 0, 100e-9, 500e-9)], name="sraf-demo")


@pytest.fixture(scope="module")
def sraf_model() -> PrintModel:
    """A wider field: the outer bars of ``sraf_layout`` sit at ±470 nm."""
    cfg = SimulationConfig.from_tech_node("ArF")
    grid = GridConfig(n_pixels=256, pixel_size=4e-9)
    m = PrintModel(cfg.optics, cfg.resist, grid, tone="dark", model="threshold")
    m.dose = m.dose_to_size(Layout(line_array(5, pitch=200e-9, cd=100e-9, length=900e-9)), 100.0)
    return m


def test_scattering_bars_go_where_there_is_room():
    lay = sraf_layout()
    with_bars = add_scattering_bars(lay, gap=80e-9, width=40e-9)
    bars = with_bars.on_layer("sraf")
    # No room inside the dense pair's 100 nm space. The 200 nm space to the
    # isolated line fits exactly one bar at gap + width + clearance, and
    # the mirror-image candidate from the other edge lands on top of it and
    # is dropped. The two outer sides each get one. Line ends (100 nm) are
    # under the 2 × gap minimum length.
    assert sorted(round(b.cx * 1e9) for b in bars) == [-450, 50, 350]
    assert all(round(b.h * 1e9) == 40 and round(b.w * 1e9) == 420 for b in bars)
    assert len(with_bars.shapes) == len(lay.shapes) + 3
    assert with_bars.layers() == ["main", "sraf"]
    assert lay.layers() == ["main"]                    # the design is untouched


def test_scattering_bars_do_not_print_at_dose(sraf_model):
    lay = add_scattering_bars(sraf_layout(), gap=80e-9, width=40e-9)
    printed = sraf_model.print(lay)
    assert assist_features_printed(printed, lay) == [False] * 3
    # A bar nearly as wide as the line is a line: it prints.
    fat = add_scattering_bars(sraf_layout(), gap=80e-9, width=80e-9)
    assert len(fat.on_layer("sraf")) == 2
    assert assist_features_printed(sraf_model.print(fat), fat) == [True, True]


def test_opc_leaves_assist_features_alone(sraf_model):
    model = sraf_model
    lay = add_scattering_bars(sraf_layout(), gap=80e-9, width=40e-9)
    res = run_opc(lay, model, layer="main", max_iter=2)
    bars_in = lay.on_layer("sraf")
    bars_out = res.corrected.on_layer("sraf")
    assert len(bars_out) == len(bars_in)
    assert all(a is b for a, b in zip(bars_in, bars_out))
