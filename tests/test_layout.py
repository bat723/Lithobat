"""
Tests for mask geometry, rasterisation, and multi-patterning decomposition.

The two things that must not regress:

* **Sub-pixel rasterisation.** At a 4 nm pixel a 1 nm overlay is a quarter of
  a pixel. If rasterisation is binary, every overlay result snaps to the
  pixel grid and pitch walking becomes a staircase artefact instead of a
  measurement.
* **Honest decomposition failure.** An odd cycle of conflicts is not
  2-colourable. The code must say so rather than silently returning a
  colouring that still conflicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.core.config import GridConfig
from litho_sim.mask.geometry import (
    Contact,
    PathShape,
    Polygon,
    Rect,
    polygon_distance,
    rasterize_shapes,
    shape_from_dict,
)
from litho_sim.mask.layout import (
    Layout,
    conflict_graph,
    contact_grid,
    cut_bar,
    decompose,
    line_array,
    split_by_color,
)


@pytest.fixture
def grid() -> GridConfig:
    return GridConfig(n_pixels=64, pixel_size=4e-9)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_rect_polygon_is_centred():
    r = Rect("m", 0.0, 0.0, 100e-9, 40e-9)
    x0, y0, x1, y1 = r.bounds()
    assert (x0, x1) == pytest.approx((-50e-9, 50e-9))
    assert (y0, y1) == pytest.approx((-20e-9, 20e-9))


def test_rect_rotation_changes_bounds():
    r = Rect("m", 0, 0, 100e-9, 20e-9, rotation=90.0)
    x0, y0, x1, y1 = r.bounds()
    assert (x1 - x0) == pytest.approx(20e-9)
    assert (y1 - y0) == pytest.approx(100e-9)


def test_translate_is_analytic_not_snapped():
    """Sub-nanometre translation must survive exactly in the geometry."""
    r = Rect("m", 0, 0, 40e-9, 40e-9)
    t = r.translated(0.37e-9, 0.0)
    assert t.cx == pytest.approx(0.37e-9)


def test_contact_shapes():
    for shape in ("square", "round", "octagon"):
        c = Contact("m", 0, 0, 40e-9, shape)
        x0, y0, x1, y1 = c.bounds()
        assert (x1 - x0) == pytest.approx(40e-9, rel=0.05)
    with pytest.raises(ValueError):
        Contact("m", 0, 0, 40e-9, "trapezoid").polygon()


def test_polygon_needs_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        Polygon("m", ((0, 0), (1e-9, 0))).polygon()


def test_pathshape_width():
    p = PathShape("m", ((-50e-9, 0.0), (50e-9, 0.0)), width=20e-9)
    x0, y0, x1, y1 = p.bounds()
    assert (y1 - y0) == pytest.approx(20e-9)
    assert (x1 - x0) == pytest.approx(100e-9)


def test_pathshape_needs_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        PathShape("m", ((0, 0),), width=10e-9).polygon()


@pytest.mark.parametrize(
    "shape",
    [
        Rect("m", 1e-9, 2e-9, 30e-9, 40e-9, 15.0),
        Polygon("m", ((0, 0), (1e-8, 0), (0, 1e-8))),
        PathShape("m", ((0, 0), (1e-8, 1e-8)), 5e-9),
        Contact("m", 3e-9, 4e-9, 20e-9, "round"),
    ],
)
def test_shape_dict_round_trip(shape):
    assert shape_from_dict(shape.to_dict()) == shape


def test_shape_from_dict_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown shape kind"):
        shape_from_dict({"kind": "Blob", "layer": "m"})


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------


def test_rasterize_area_is_accurate(grid):
    """A rectangle must cover exactly its own area."""
    r = Rect("m", 0, 0, 40e-9, 200e-9)
    img = rasterize_shapes([r], grid, oversample=4)
    area = img.sum() * grid.pixel_size ** 2
    assert area == pytest.approx(40e-9 * 200e-9, rel=0.02)


def test_rasterize_is_bounded(grid):
    img = rasterize_shapes([Rect("m", 0, 0, 40e-9, 40e-9)], grid)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_rasterize_empty_is_blank(grid):
    img = rasterize_shapes([], grid)
    assert img.shape == (grid.n_pixels, grid.n_pixels)
    assert not img.any()


def test_subpixel_shift_moves_edge_proportionally(grid):
    """A quarter-pixel shift must change coverage by about a quarter.

    This is what makes 1 nm overlay measurable on a 4 nm grid.
    """
    r = Rect("m", 0, 0, 40e-9, 200e-9)
    base = rasterize_shapes([r], grid, oversample=4)
    shifted = rasterize_shapes([r], grid, oversample=4, dx=1e-9)
    delta = float(np.abs(base - shifted).max())
    assert delta == pytest.approx(0.25, abs=0.05), (
        f"quarter-pixel shift produced a {delta:.3f} change; anti-aliasing "
        "is not resolving sub-pixel overlay"
    )


def test_binary_rasterisation_only_moves_in_whole_pixels(grid):
    """Contrast case: with oversample=1 an edge can only ever jump a full pixel.

    Documents why anti-aliasing is load-bearing rather than cosmetic — a
    binary raster cannot represent a fractional edge position at all, so
    sub-pixel overlay is either ignored or exaggerated into a whole pixel.
    """
    r = Rect("m", 0, 0, 42e-9, 200e-9)
    base = rasterize_shapes([r], grid, oversample=1)
    shifted = rasterize_shapes([r], grid, oversample=1, dx=1e-9)
    deltas = set(np.unique(np.abs(base - shifted)).tolist())
    assert deltas <= {0.0, 1.0}, f"binary raster produced fractional edges: {deltas}"

    aa_base = rasterize_shapes([r], grid, oversample=4)
    aa_shift = rasterize_shapes([r], grid, oversample=4, dx=1e-9)
    aa_deltas = np.unique(np.abs(aa_base - aa_shift))
    assert ((aa_deltas > 0) & (aa_deltas < 1)).any(), (
        "anti-aliased raster should produce fractional edge changes"
    )


def test_overlapping_shapes_union(grid):
    a = Rect("m", -10e-9, 0, 40e-9, 40e-9)
    b = Rect("m", 10e-9, 0, 40e-9, 40e-9)
    img = rasterize_shapes([a, b], grid, oversample=2)
    assert img.max() <= 1.0, "overlap must union, not accumulate"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_layout_layers_and_filtering():
    lay = Layout([Rect("gate", 0, 0, 10e-9, 10e-9), Rect("cut", 0, 0, 10e-9, 10e-9)])
    assert lay.layers() == ["gate", "cut"]
    assert len(lay.on_layer("cut")) == 1


def test_layout_tone_inverts(grid):
    lay = Layout([Rect("m", 0, 0, 40e-9, 40e-9)])
    clear = lay.rasterize(grid, tone="clear")
    dark = lay.rasterize(grid, tone="dark")
    assert np.allclose(clear, 1.0 - dark)


def test_layout_rejects_bad_tone(grid):
    with pytest.raises(ValueError, match="tone must be"):
        Layout([]).rasterize(grid, tone="beige")


def test_layout_rasterize_matches_engine_format(grid):
    """The output must be exactly what compute_aerial_image already takes."""
    from litho_sim.core.config import OpticsConfig
    from litho_sim.expose.aerial_image import compute_aerial_image

    lay = Layout(line_array(3, pitch=160e-9, cd=80e-9, length=400e-9))
    mask = lay.rasterize(grid)
    assert mask.shape == (grid.n_pixels, grid.n_pixels)
    aerial = compute_aerial_image(mask, OpticsConfig(source_grid=11), grid)
    assert aerial.shape == mask.shape and np.isfinite(aerial).all()


def test_layout_json_round_trip(tmp_path):
    lay = Layout(
        line_array(4, 100e-9, 50e-9, 300e-9) + [cut_bar(0, 0, 400e-9, 30e-9)],
        name="gates",
    )
    p = tmp_path / "layout.json"
    lay.to_json(p)
    back = Layout.from_json(p)
    assert back.name == lay.name
    assert back.shapes == lay.shapes


def test_layout_from_json_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Layout.from_json(tmp_path / "nope.json")


def test_layout_records_are_in_nanometres():
    lay = Layout([Rect("m", 0, 0, 40e-9, 100e-9)])
    row = lay.to_records()[0]
    assert row["w_nm"] == pytest.approx(40.0)
    assert row["h_nm"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Parametric builders
# ---------------------------------------------------------------------------


def test_line_array_pitch_and_count():
    shapes = line_array(5, pitch=100e-9, cd=50e-9, length=400e-9)
    assert len(shapes) == 5
    centres = sorted(s.cx for s in shapes)
    assert np.allclose(np.diff(centres), 100e-9)


def test_line_array_is_centred():
    shapes = line_array(4, pitch=100e-9, cd=50e-9, length=400e-9)
    assert np.mean([s.cx for s in shapes]) == pytest.approx(0.0)


def test_line_array_orientation():
    v = line_array(3, 100e-9, 50e-9, 400e-9, orientation="vertical")
    h = line_array(3, 100e-9, 50e-9, 400e-9, orientation="horizontal")
    assert v[0].w < v[0].h and h[0].w > h[0].h
    with pytest.raises(ValueError):
        line_array(3, 100e-9, 50e-9, 400e-9, orientation="diagonal")


def test_contact_grid_count():
    assert len(contact_grid(3, 4, 80e-9, 80e-9, 40e-9)) == 12


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_conflict_graph_finds_close_neighbours():
    shapes = line_array(4, pitch=60e-9, cd=30e-9, length=200e-9)
    adj = conflict_graph(shapes, min_spacing=50e-9)
    # 30 nm space between 30 nm lines on a 60 nm pitch conflicts at 50 nm rule.
    assert all(len(v) > 0 for v in adj.values())


def test_no_conflicts_when_relaxed():
    shapes = line_array(4, pitch=300e-9, cd=30e-9, length=200e-9)
    adj = conflict_graph(shapes, min_spacing=50e-9)
    assert all(len(v) == 0 for v in adj.values())


def test_dense_lines_two_colour_alternately():
    """The canonical LELE case: alternate lines go to alternate exposures."""
    shapes = line_array(6, pitch=60e-9, cd=30e-9, length=200e-9)
    d = decompose(shapes, min_spacing=50e-9, n_colors=2)
    assert d.ok, f"dense L/S should be 2-colourable, got {d.conflicts}"
    ordered = [c for _, c in sorted(zip([s.cx for s in shapes], d.colors))]
    assert all(ordered[i] != ordered[i + 1] for i in range(len(ordered) - 1))


def test_odd_cycle_is_not_two_colourable():
    """Three mutually-conflicting features cannot be split across two masks.

    This is a real design-rule failure, not an algorithm weakness, and it is
    the reason LE-cubed exists. It must be reported, not hidden.
    """
    tri = [
        Rect("m", 0, 0, 20e-9, 20e-9),
        Rect("m", 30e-9, 0, 20e-9, 20e-9),
        Rect("m", 15e-9, 26e-9, 20e-9, 20e-9),
    ]
    d2 = decompose(tri, min_spacing=25e-9, n_colors=2)
    assert not d2.ok and d2.conflicts

    d3 = decompose(tri, min_spacing=25e-9, n_colors=3)
    assert d3.ok
    assert len(set(d3.colors)) == 3


def test_split_by_color_partitions_all_shapes():
    shapes = line_array(6, pitch=60e-9, cd=30e-9, length=200e-9)
    lay = Layout(shapes, name="dense")
    d = decompose(shapes, min_spacing=50e-9, n_colors=2)
    groups = split_by_color(lay, d)
    assert set(groups) == {"colorA", "colorB"}
    assert sum(len(g) for g in groups.values()) == len(shapes)
    assert all(g.name.startswith("dense:") for g in groups.values())


def test_split_by_color_three_way():
    shapes = line_array(9, pitch=40e-9, cd=20e-9, length=200e-9)
    d = decompose(shapes, min_spacing=70e-9, n_colors=3)
    groups = split_by_color(Layout(shapes), d, prefix="color")
    assert set(groups) == {"colorA", "colorB", "colorC"}


def test_exact_distance_handles_rotation():
    """Bounding boxes overestimate proximity for rotated shapes."""
    a = Rect("m", 0, 0, 100e-9, 10e-9, rotation=45.0)
    b = Rect("m", 90e-9, 90e-9, 100e-9, 10e-9, rotation=45.0)
    assert polygon_distance(a, b) > 0.0
