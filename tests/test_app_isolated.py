"""
The isolated designs: a frame with bars and a contact cluster, drawn in units
of the CD with empty field around them — the case inverse lithography answers
with curves and assist rings.

Four contracts:

* **They are geometry, and the geometry is sound.** One polygon for the frame
  (four abutting rectangles would hand the edge loop shared edges to correct
  twice), three bars, five contacts; nothing overlaps; the raster has the area
  the drawing says.
* **They need room, and the app says so.** On the default 512 nm field both
  correction pages refuse with a sentence naming the grid to set; on a field
  that fits they run. Print never refuses — it draws what fits.
* **Inverse lithography corrects them, curvilinearly.** On a 2048 nm field the
  solve cuts the pattern error, leaves grey along the edges, and puts mask
  energy in the empty field where nothing was drawn.
* **Edge-based OPC runs on the same drawing.** The fragment loop corrects
  every shape without an exception, so the two corrections can be compared.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.app.compute import build_mask, compute_imaging
from litho_sim.app.ilt import ILTRequest, compute_ilt, ilt_availability
from litho_sim.app.opc import (
    ISOLATED,
    app_layout,
    compute_opc,
    design_extent,
    dose_to_size,
    field_fit,
    mask_from_layout,
    opc_availability,
)
from litho_sim.app.params import PATTERNS, ParameterModel
from litho_sim.mask import Contact, Polygon, Rect

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _big(pattern: str, n: int = 128, px: float = 16.0) -> ParameterModel:
    """A 2048 nm field, coarse enough to solve in a test."""
    p = ParameterModel()
    p.set("pattern", pattern)
    p.set("n_pixels", n)
    p.set("pixel_size", px)
    p.set("source_grid", 11)
    return p


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_the_isolated_designs_are_in_the_menu():
    assert set(ISOLATED) <= set(PATTERNS)


def test_frame_and_bars_is_one_polygon_and_three_bars():
    p = _big("frame and bars", n=256, px=8.0)
    layout = app_layout(p)
    kinds = [type(s) for s in layout.shapes]
    assert kinds == [Polygon, Rect, Rect, Rect]
    assert all(s.layer == "main" for s in layout.shapes)
    # The raster carries the drawing's area: 40 CD² of frame, 12 CD² of bars.
    cd_px = p.si("cd") / p.grid().pixel_size
    cov = mask_from_layout(layout, p.grid(), "binary")
    assert cov.sum() == pytest.approx(52.0 * cd_px ** 2, rel=0.01)
    # Nothing abuts or overlaps — the edge loop's precondition.
    parts = [mask_from_layout(type(layout)([s]), p.grid(), "binary") for s in layout.shapes]
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            assert (parts[i] * parts[j]).sum() == 0.0


def test_isolated_contacts_is_a_group_and_a_loner():
    p = _big("isolated contacts", n=256, px=8.0)
    layout = app_layout(p)
    assert all(isinstance(s, Contact) for s in layout.shapes)
    assert len(layout.shapes) == 5
    xs = sorted(s.cx for s in layout.shapes)
    pitch = p.si("pitch")
    assert xs[-1] - xs[-2] == pytest.approx(2.0 * pitch)          # the loner stands off
    w, h = design_extent(p)
    assert w == pytest.approx(3.0 * pitch + p.si("cd"))
    assert h == pytest.approx(pitch + p.si("cd"))


# ---------------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ISOLATED)
def test_the_default_field_is_refused_with_the_grid_to_set(pattern):
    p = ParameterModel()
    p.set("pattern", pattern)
    reason = field_fit(p)
    assert reason is not None
    assert "Grid" in reason and "Pixel size" in reason
    # Both correction pages carry the same sentence.
    assert opc_availability(p) == reason
    assert ilt_availability(p) == reason


@pytest.mark.parametrize("pattern", ISOLATED)
def test_a_field_that_fits_is_offered(pattern):
    p = _big(pattern, n=256, px=8.0)
    assert field_fit(p) is None
    assert opc_availability(p) is None
    assert ilt_availability(p) is None


def test_periodic_patterns_never_have_a_fit_problem():
    for pattern in PATTERNS:
        if pattern in ISOLATED:
            continue
        p = ParameterModel()
        p.set("pattern", pattern)
        p.set("n_pixels", 32)
        assert design_extent(p) is None and field_fit(p) is None


@pytest.mark.parametrize("pattern", ISOLATED)
def test_print_draws_what_fits_and_never_refuses(pattern):
    p = ParameterModel()
    p.set("pattern", pattern)
    m = build_mask(p)
    assert m.shape == (128, 128) and 0.0 < m.sum() < m.size
    r = compute_imaging(p)
    assert np.isfinite(r.aerial).all()


# ---------------------------------------------------------------------------
# The corrections
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame_solve():
    p = _big("frame and bars")
    return p, compute_ilt(ILTRequest(params=p, max_iter=40))


def test_ilt_corrects_the_frame(frame_solve):
    _p, b = frame_solve
    assert b.pattern_error_after < 0.5 * b.pattern_error_before


def test_the_frame_solve_is_curvilinear_and_grows_assist_features(frame_solve):
    _p, b = frame_solve
    m = b.result.mask
    grey = np.count_nonzero((m > 0.05) & (m < 0.95))
    assert grey > 0 and m.min() < 0.05 and m.max() > 0.95
    # Mask energy off the drawn design, in the empty field.
    assert m[b.target < 0.5].max() > 0.25


def test_the_bars_inside_the_frame_are_not_mistaken_for_abutting_it(caplog):
    """The loop's abut-or-overlap check starts from bounding boxes, and the
    frame's box encloses its bars. The exact outline distance says otherwise,
    and the warning that would send a user off to merge polygons stays quiet."""
    import logging

    p = _big("frame and bars")
    p.set("opc_iterations", 1)
    p.set("normalisation", "clear")
    with caplog.at_level(logging.WARNING, logger="litho_sim.opc.correct"):
        compute_opc(p)
    assert not [r for r in caplog.records if "abut or overlap" in r.getMessage()]


@pytest.mark.parametrize("pattern", ISOLATED)
def test_edge_opc_corrects_the_same_drawing(pattern):
    """At the dense-array dose the isolated contacts do not print at all —
    every edge is missing before the loop starts, which is the iso-dense bias
    the correction exists for. Afterwards every edge is found and sits within
    a fraction of a pixel of where it was drawn."""
    p = _big(pattern)
    p.set("opc_iterations", 4)
    p.set("normalisation", "clear")
    p.set("dose", dose_to_size(p))
    r = compute_opc(p)
    assert len(r.corrected.on_layer("main")) == len(app_layout(p).shapes)
    before, after = r.epe_before.stats(), r.epe_after.stats()
    assert after["n_failed"] == 0
    assert after["rms_epe_nm"] < 0.5 * p.si("pixel_size") * 1e9
    if np.isfinite(before["rms_epe_nm"]):
        assert after["rms_epe_nm"] <= before["rms_epe_nm"]
