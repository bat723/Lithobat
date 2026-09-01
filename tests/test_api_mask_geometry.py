"""
API tests for the mask **geometry** subfeature.

Everything here goes through the public package interface —
``from litho_sim.mask import ...`` — never through deep module paths, so
these tests double as a guarantee that the geometry surface stays exported.
They build one of each shape kind, rasterise them into the ``(n, n)``
transmittance array the engine consumes, check that features land where the
coordinates say they should, and feed the raster to the real imaging engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.expose import compute_aerial_image
from litho_sim.mask import (
    Contact,
    PathShape,
    Polygon,
    Rect,
    Shape,
    rasterize_shapes,
)

N = 64
PX = 4e-9  # 4 nm pixels — the convention every process test uses


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=N, pixel_size=PX)


def idx(coord_m: float) -> int:
    """Index of the pixel whose centre sits at *coord_m* (field-centre metres)."""
    return round(coord_m / PX) + N // 2


# ---------------------------------------------------------------------------
# Happy path: all four shape kinds through the public API
# ---------------------------------------------------------------------------


def test_all_shape_kinds_rasterize_where_drawn(grid):
    """One shape per quadrant; each must light up its own quadrant only."""
    shapes = [
        Rect("m", -64e-9, -64e-9, 40e-9, 40e-9),
        Contact("m", 64e-9, -64e-9, 40e-9, "round"),
        Polygon("m", ((-84e-9, 44e-9), (-44e-9, 44e-9), (-64e-9, 84e-9))),
        PathShape("m", ((44e-9, 64e-9), (84e-9, 64e-9)), width=20e-9),
    ]
    assert all(isinstance(s, Shape) for s in shapes)

    img = rasterize_shapes(shapes, grid, oversample=4)

    assert img.shape == (N, N)
    assert img.dtype == np.float64
    assert img.min() >= 0.0 and img.max() <= 1.0

    # Interior pixels are fully covered, the empty field centre is not.
    assert img[idx(-64e-9), idx(-64e-9)] == 1.0   # Rect, lower-left
    assert img[idx(-64e-9), idx(64e-9)] == 1.0    # Contact, lower-right
    assert img[idx(52e-9), idx(-64e-9)] > 0.9     # Polygon, upper-left
    assert img[idx(64e-9), idx(64e-9)] == 1.0     # PathShape, upper-right
    assert img[N // 2, N // 2] == 0.0

    # Every quadrant carries flux — no shape got lost or misplaced.
    h = N // 2
    for quadrant in (img[:h, :h], img[:h, h:], img[h:, :h], img[h:, h:]):
        assert quadrant.sum() > 0.0


def test_rasterization_conserves_area(grid):
    """Anti-aliased coverage must integrate to the drawn area."""
    img = rasterize_shapes([Rect("m", 0, 0, 40e-9, 40e-9)], grid, oversample=4)
    expected_pixels = (40e-9 / PX) ** 2  # 100 pixels of area
    assert abs(img.sum() - expected_pixels) < 1.0


def test_subpixel_overlay_shifts_centroid(grid):
    """A 1 nm (quarter-pixel) dx must survive anti-aliased rasterisation."""
    r = Rect("m", 0, 0, 40e-9, 200e-9)
    base = rasterize_shapes([r], grid, oversample=4)
    shifted = rasterize_shapes([r], grid, oversample=4, dx=1e-9)

    coords = (np.arange(N) - N // 2) * PX
    cx_base = float(base.sum(axis=0) @ coords) / float(base.sum())
    cx_shift = float(shifted.sum(axis=0) @ coords) / float(shifted.sum())
    assert abs(cx_base) < 0.3e-9
    assert abs((cx_shift - cx_base) - 1e-9) < 0.3e-9


def test_overlapping_shapes_union_not_sum(grid):
    """Coverage is a union: overlap must never push transmittance above 1."""
    a = Rect("m", -8e-9, 0, 40e-9, 40e-9)
    b = Rect("m", 8e-9, 0, 40e-9, 40e-9)
    img = rasterize_shapes([a, b], grid, oversample=4)
    assert img.max() <= 1.0
    # Union area: 100 + 100 - (24 nm x 40 nm overlap = 60 px) = 140 px.
    assert abs(img.sum() - 140.0) < 2.0


def test_raster_feeds_the_imaging_engine(grid):
    """The geometry raster is a real mask: the aerial image must resolve it."""
    # A 100 nm line is comfortably resolvable at 193 nm / NA 0.93 (the scale
    # test_resist images); 40 nm dense lines would be below the k1 limit.
    line = Rect("m", 0.0, 0.0, 100e-9, N * PX)
    mask = rasterize_shapes([line], grid, oversample=4)

    optics = OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=11)
    aerial = compute_aerial_image(mask, optics, grid, dose=1.0)

    assert aerial.shape == mask.shape
    assert np.isfinite(aerial).all()
    assert aerial.min() >= 0.0
    # Bright where the drawn line is clear, dark 50 nm outside its edge.
    row = N // 2
    assert aerial[row, idx(0.0)] > 2.0 * aerial[row, idx(100e-9)]


def test_serialization_and_distance_helpers_are_public():
    """to_dict's inverse and the spacing helpers must be package-reachable."""
    from litho_sim.mask import (
        polygon_distance,
        shape_from_dict,
        shapes_overlap_distance,
    )

    shape = Contact("via1", 3e-9, -4e-9, 20e-9, "octagon")
    assert shape_from_dict(shape.to_dict()) == shape

    a = Rect("m", 0, 0, 40e-9, 40e-9)
    b = Rect("m", 100e-9, 0, 40e-9, 40e-9)  # 60 nm edge-to-edge gap
    assert shapes_overlap_distance(a, b) == pytest.approx(60e-9)
    assert polygon_distance(a, b) == pytest.approx(60e-9)


# ---------------------------------------------------------------------------
# Errors and edge cases
# ---------------------------------------------------------------------------


def test_degenerate_polygon_raises(grid):
    with pytest.raises(ValueError, match="at least 3 points"):
        rasterize_shapes([Polygon("m", ((0.0, 0.0), (1e-9, 0.0)))], grid)


def test_degenerate_path_raises(grid):
    with pytest.raises(ValueError, match="at least 2 points"):
        rasterize_shapes([PathShape("m", ((0.0, 0.0),), width=10e-9)], grid)
    with pytest.raises(ValueError, match="zero length"):
        rasterize_shapes(
            [PathShape("m", ((5e-9, 5e-9), (5e-9, 5e-9)), width=10e-9)], grid
        )


def test_unknown_contact_style_raises(grid):
    with pytest.raises(ValueError, match="square.*round.*octagon"):
        rasterize_shapes([Contact("m", 0, 0, 40e-9, "hexagon")], grid)


def test_empty_shape_list_is_dark_field(grid):
    img = rasterize_shapes([], grid)
    assert img.shape == (N, N)
    assert img.dtype == np.float64
    assert not img.any()


def test_shape_entirely_off_field_is_dark(grid):
    img = rasterize_shapes([Rect("m", 1e-6, 1e-6, 40e-9, 40e-9)], grid)
    assert not img.any()


def test_zero_size_rect_covers_nothing(grid):
    img = rasterize_shapes([Rect("m", 0, 0, 0.0, 0.0)], grid, oversample=4)
    assert not img.any()


def test_oversample_one_gives_hard_binary_edges(grid):
    img = rasterize_shapes([Rect("m", 0, 0, 42e-9, 42e-9)], grid, oversample=1)
    assert set(np.unique(img)).issubset({0.0, 1.0})
    assert img.any()
