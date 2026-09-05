"""
Public-API tests for mask layouts and decomposition.

Everything here goes through the package surface — ``litho_sim.mask`` for
the layout API, ``litho_sim.patterning`` for the consumer — never the deep
module paths. Two contracts are pinned:

* **The multi-patterning invariant.** Splitting a layout by colour must
  partition it exactly: every shape lands in exactly one exposure, and the
  colour rasters recompose to the full-layout raster pixel for pixel.
* **The consumer chain.** :func:`litho_sim.patterning.lele` is the real
  customer of ``decompose``/``split_by_color``; building its flow from a
  dense layout must wire one exposure per colour, and a non-colourable
  layout must be refused with the conflicts named.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask import (
    Layout,
    Rect,
    contact_grid,
    cut_bar,
    decompose,
    line_array,
    split_by_color,
)
from litho_sim.patterning import Expose, lele


@pytest.fixture
def grid() -> GridConfig:
    return GridConfig(n_pixels=128, pixel_size=4e-9)


def dense_lines() -> list:
    """Six lines at 60 nm pitch / 30 nm CD: prints only as LELE at a
    50 nm single-exposure rule."""
    return line_array(6, pitch=60e-9, cd=30e-9, length=200e-9)


# ---------------------------------------------------------------------------
# Happy path: build -> decompose -> split -> rasterize
# ---------------------------------------------------------------------------


def test_builders_compose_one_layout():
    lay = Layout(dense_lines(), name="device")
    lay.add(cut_bar(0.0, 0.0, 300e-9, 30e-9))
    lay.add(*contact_grid(2, 2, 120e-9, 120e-9, 40e-9, layer="via"))
    assert len(lay) == 6 + 1 + 4
    assert lay.layers() == ["main", "cut", "via"]
    assert len(lay.on_layer("via")) == 4


def test_split_partitions_shapes_and_raster(grid):
    """Colour layouts must recompose to the full layout exactly.

    The shapes are disjoint, so the union raster equals the sum of the
    per-colour rasters — any double-assigned or dropped shape breaks this.
    """
    lay = Layout(dense_lines(), name="dense")
    dec = decompose(lay.shapes, min_spacing=50e-9, n_colors=2)
    assert dec.ok, f"dense L/S must be 2-colourable, got {dec.conflicts}"

    groups = split_by_color(lay, dec)
    assert set(groups) == {"colorA", "colorB"}
    assert sum(len(g) for g in groups.values()) == len(lay)
    # No shape may appear in both colours.
    seen = [s for g in groups.values() for s in g.shapes]
    assert len(seen) == len({id(s) for s in seen})

    full = lay.rasterize(grid)
    parts = [g.rasterize(grid) for g in groups.values()]
    for p in parts:
        assert p.shape == (grid.n_pixels, grid.n_pixels)
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert 0.0 < p.sum() < full.sum(), "each exposure prints a proper subset"
    assert np.allclose(sum(parts), full, atol=1e-9), (
        "colour rasters must recompose to the full layout"
    )


def test_split_layouts_survive_flow_serialisation():
    """Flow.from_dict rebuilds layouts via Layout.from_dict — the colour
    layouts produced by a split must round-trip that path."""
    lay = Layout(dense_lines(), name="dense")
    dec = decompose(lay.shapes, min_spacing=50e-9, n_colors=2)
    for key, sub in split_by_color(lay, dec).items():
        back = Layout.from_dict(sub.to_dict())
        assert back.shapes == sub.shapes
        assert back.name == sub.name == f"dense:{key}"


# ---------------------------------------------------------------------------
# Consumer chain: litho_sim.patterning.lele
# ---------------------------------------------------------------------------


@pytest.fixture
def optics() -> OpticsConfig:
    return OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=11)


@pytest.fixture
def resist() -> ResistConfig:
    return ResistConfig(
        n_resist=1.7, dill_A=0.8, dill_B=0.05, dill_C=0.04,
        dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9,
    )


def test_lele_flow_wires_one_exposure_per_color(grid, optics, resist):
    lay = Layout(dense_lines(), name="dense")
    flow = lele(lay, grid, optics, resist, min_spacing=50e-9)

    assert set(flow.layouts) == {"colorA", "colorB"}
    assert sum(len(g) for g in flow.layouts.values()) == len(lay)
    exposed = [s.layout for s in flow.steps if isinstance(s, Expose)]
    assert exposed == ["colorA", "colorB"], (
        "LELE must expose each colour exactly once, in order"
    )


def test_lele_refuses_odd_conflict_cycle(grid, optics, resist):
    """A triangle of mutual conflicts is a design-rule violation, and the
    consumer must surface it as an error naming the pairs."""
    tri = Layout([
        Rect("m", 0.0, 0.0, 20e-9, 20e-9),
        Rect("m", 30e-9, 0.0, 20e-9, 20e-9),
        Rect("m", 15e-9, 26e-9, 20e-9, 20e-9),
    ], name="triangle")
    with pytest.raises(ValueError, match="not 2-colourable"):
        lele(tri, grid, optics, resist, min_spacing=25e-9)


# ---------------------------------------------------------------------------
# Edge and error cases
# ---------------------------------------------------------------------------


def test_empty_layout_rasterizes_blank(grid):
    lay = Layout(name="empty")
    assert len(lay) == 0 and lay.layers() == []
    clear = lay.rasterize(grid, tone="clear")
    assert clear.shape == (grid.n_pixels, grid.n_pixels)
    assert not clear.any()
    assert np.allclose(lay.rasterize(grid, tone="dark"), 1.0)


def test_empty_layout_decomposes_cleanly():
    dec = decompose([], min_spacing=50e-9, n_colors=2)
    assert dec.ok and dec.colors == []
    groups = split_by_color(Layout(name="empty"), dec)
    assert set(groups) == {"colorA", "colorB"}
    assert all(len(g) == 0 for g in groups.values())


def test_degenerate_builder_sizes(grid):
    solo = line_array(1, pitch=100e-9, cd=30e-9, length=200e-9)
    assert len(solo) == 1
    assert solo[0].cx == pytest.approx(0.0), "a single line sits at the centre"

    assert contact_grid(0, 3, 80e-9, 80e-9, 40e-9) == []
    assert not Layout(contact_grid(0, 3, 80e-9, 80e-9, 40e-9)).rasterize(grid).any()


def test_bad_contact_shape_fails_at_rasterize(grid):
    """contact_grid builds contacts lazily; an unknown profile must still be
    rejected when the layout is rendered."""
    lay = Layout(contact_grid(2, 2, 80e-9, 80e-9, 40e-9, shape="hexagon"))
    with pytest.raises(ValueError):
        lay.rasterize(grid)
