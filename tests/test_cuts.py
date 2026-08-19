"""
Tests for line-end formation: cut, block, and keep.

The semantics worth defending (see docs/patterning/Line-End Formation):

* **Cut** — drawn shapes are removed from existing lines. Litho route:
  ``tone="clear"`` opens the resist exactly at the drawn shapes, and the
  following etch severs whatever line runs through them. Ideal route: the
  drawn shapes gate the etch directly (``Etch(open_layout=...)``).
* **Block** — drawn shapes are protected during the next etch and nothing
  else happens; the block composes with whatever etch follows.
* **Keep** — only drawn regions survive: block litho + blanket etch + strip.

The old implementation was a keep mask labelled as a cut (tone="dark"), and
inside SADP/SAQP it ran while the spacer still capped the lines, so the etch
found no exposed target and silently did nothing. Both are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask.layout import Layout, cut_bar, line_array
from litho_sim.patterning import (
    Deposit,
    Develop,
    Etch,
    Flow,
    Measure,
    Strip,
    add_cut_mask,
    block_resist,
    cut_block,
    keep_block,
    sadp,
)
from litho_sim.patterning.stackspec import RESIST, TARGET
from litho_sim.wafer import Stack, get_material

POLY = get_material("poly-Si").id


def _poly_presence(stack: Stack) -> np.ndarray:
    """(ny, nx) boolean map: does the column contain any target material?"""
    return (stack.mat == POLY).any(axis=0)


def _runs(column: np.ndarray) -> int:
    """Number of connected True-runs in a 1-D boolean array."""
    padded = np.concatenate([[False], column, [False]])
    return int((np.diff(padded.astype(int)) == 1).sum())


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=128, pixel_size=4e-9, dz=4e-9, n_z_slices=7)


@pytest.fixture(scope="module")
def optics() -> OpticsConfig:
    return OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=11)


@pytest.fixture(scope="module")
def resist() -> ResistConfig:
    return ResistConfig(
        n_resist=1.7, dill_A=0.8, dill_B=0.05, dill_C=0.04,
        dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9,
    )


# ---------------------------------------------------------------------------
# Etch open_layout — the JSON-safe ideal-cut gate
# ---------------------------------------------------------------------------


def test_etch_open_layout_json_roundtrip(grid, optics, resist):
    """A flow holding Etch(open_layout=...) must survive serialisation."""
    lines = Layout(line_array(2, pitch=160e-9, cd=40e-9, length=400e-9), name="lines")
    cut = Layout([cut_bar(0.0, 0.0, 120e-9, 40e-9)], name="cut")
    flow = Flow(
        name="ideal-cut", grid=grid, optics=optics, resist=resist,
        layouts={"lines": lines, "cut": cut},
    )
    flow.add(Etch(targets="poly-Si", depth=40e-9, open_layout="cut", open_tone="clear"))

    restored = Flow.from_dict(flow.to_dict())
    step = restored.steps[0]
    assert isinstance(step, Etch)
    assert step.open_layout == "cut"
    assert step.open_tone == "clear"
    assert "through cut" in step.describe()


def test_etch_legacy_dict_without_open_fields_loads():
    """Pre-slice serialised flows carry no open_layout/open_tone keys."""
    legacy = {
        "kind": "etch",
        "targets": "SOC",
        "depth": 6e-08,
        "anisotropy": 1.0,
        "selectivity": {},
    }
    from litho_sim.patterning.steps import ProcessStep

    step = ProcessStep.from_dict(legacy)
    assert isinstance(step, Etch)
    assert step.open_layout is None
    assert step.open_tone == "clear"


def test_etch_unknown_open_layout_is_a_clear_error(grid, optics, resist):
    flow = Flow(
        name="bad", grid=grid, optics=optics, resist=resist,
        layouts={},
    )
    flow.add(Etch(targets="poly-Si", depth=40e-9, open_layout="nonexistent"))
    with pytest.raises(RuntimeError, match="nonexistent"):
        flow.run(snapshot=False)


def test_etch_open_mask_gates_the_emergent_mask(grid):
    """An ideal cut etches only inside the drawn window, a block only outside."""
    stack = Stack.blank(grid, dz=grid.dz, headroom=120e-9)
    stack.deposit_blanket("poly-Si", 40e-9)

    cut = Layout([cut_bar(0.0, 0.0, 100e-9, 40e-9)], name="cut")
    window = cut.rasterize(grid) > 0.5

    cut_stack = stack.copy()
    cut_stack.etch("poly-Si", depth=60e-9, open_mask=window)
    poly = get_material("poly-Si").id
    remaining = (cut_stack.mat == poly).any(axis=0)
    assert not remaining[window].any(), "cut window should be cleared"
    assert remaining[~window].all(), "outside the window must be untouched"

    block_stack = stack.copy()
    block_stack.etch("poly-Si", depth=60e-9, open_mask=~window)
    remaining = (block_stack.mat == poly).any(axis=0)
    assert remaining[window].all(), "blocked region must survive"
    assert not remaining[~window].any(), "unblocked field should be cleared"


# ---------------------------------------------------------------------------
# Litho-based cut / block / keep semantics
# ---------------------------------------------------------------------------


def test_cut_tone_opens_resist_at_drawn_shapes(grid, optics, resist):
    """tone="clear" must open the resist AT the drawn bar, not around it.

    This is the tone bug pinned: the old _cut_block used tone="dark", which
    left resist standing on the bar — a keep mask claiming to be a cut.
    """
    bar = Layout([cut_bar(0.0, 0.0, 160e-9, 80e-9)], name="cut")
    flow = Flow(
        name="cut-print", grid=grid, optics=optics, resist=resist,
        layouts={"cut": bar},
    )
    flow.add(Deposit(material=TARGET, thickness=40e-9, conformal=False))
    flow.add(*cut_block("cut"))
    result = flow.run()

    develop_idx = next(
        i for i, s in enumerate(flow.steps) if isinstance(s, Develop)
    )
    after_develop = result.snapshots[develop_idx]
    resist_map = (after_develop.mat == get_material(RESIST).id).any(axis=0)

    ny, nx = resist_map.shape
    assert not resist_map[ny // 2, nx // 2], "resist must be OPEN on the drawn bar"
    assert resist_map[8, 8], "resist must SURVIVE in the field"


def test_keep_leaves_only_drawn_regions(grid, optics, resist):
    """keep_block: only the drawn region of the target survives."""
    keep = Layout([cut_bar(0.0, 0.0, 200e-9, 120e-9)], name="keep")
    flow = Flow(
        name="keep", grid=grid, optics=optics, resist=resist,
        layouts={"keep": keep},
    )
    flow.add(Deposit(material=TARGET, thickness=40e-9, conformal=False))
    flow.add(*keep_block("keep"))
    result = flow.run(snapshot=False)

    poly = _poly_presence(result.stack)
    ny, nx = poly.shape
    assert poly[ny // 2, nx // 2], "target must survive at the kept centre"
    assert not poly[8, 8] and not poly[-8, -8], "field must be cleared"
    drawn = keep.rasterize(grid) > 0.5
    assert poly[drawn].all(), "everything drawn must survive"
    # An isolated dark feature prints fat at this dose calibration (tuned for
    # dense 40 nm trenches), so the bound is on topology, not CD fidelity.
    assert poly.sum() < 4 * drawn.sum()


def test_block_protects_during_following_etch(grid, optics, resist):
    """block_resist composes with the caller's etch; caller strips after."""
    blk = Layout([cut_bar(0.0, 0.0, 200e-9, 120e-9)], name="blk")
    flow = Flow(
        name="block", grid=grid, optics=optics, resist=resist,
        layouts={"blk": blk},
    )
    flow.add(Deposit(material=TARGET, thickness=40e-9, conformal=False))
    flow.add(*block_resist("blk"))
    flow.add(
        Etch(targets=TARGET, depth=60e-9, selectivity={RESIST: 0.1}),
        Strip(material=RESIST),
    )
    result = flow.run(snapshot=False)

    poly = _poly_presence(result.stack)
    ny, nx = poly.shape
    assert poly[ny // 2, nx // 2], "blocked region must survive the etch"
    assert not poly[8, 8], "unblocked field must be etched away"


def test_cut_severs_a_line(grid, optics, resist):
    """A cut bar across a line leaves two disjoint segments."""
    line = Layout([cut_bar(0.0, 0.0, 40e-9, 440e-9)], name="line")
    bar = Layout([cut_bar(0.0, 0.0, 160e-9, 80e-9)], name="cut")

    def build(with_cut: bool) -> Stack:
        layouts = {"line": line, "cut": bar}
        flow = Flow(name="line", grid=grid, optics=optics, resist=resist,
                    layouts=layouts)
        flow.add(Deposit(material=TARGET, thickness=40e-9, conformal=False))
        flow.add(*keep_block("line"))
        if with_cut:
            flow.add(*cut_block("cut"))
        return flow.run(snapshot=False).stack

    nx = grid.n_pixels
    uncut_col = _poly_presence(build(False))[:, nx // 2]
    cut_col = _poly_presence(build(True))[:, nx // 2]
    assert _runs(uncut_col) == 1, "sanity: one continuous line before the cut"
    assert _runs(cut_col) == 2, "the cut must leave exactly two segments"


# ---------------------------------------------------------------------------
# Cuts inside SADP — the insertion-point regression
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mandrel() -> Layout:
    return Layout(line_array(3, pitch=150e-9, cd=70e-9, length=460e-9),
                  name="mandrel")


@pytest.fixture(scope="module")
def sadp_cut_bar() -> Layout:
    # Off-centre in y so the centre-row Measure is untouched, but well inside
    # the PRINTED line extent: the litho'd mandrel suffers ~80 nm of line-end
    # pullback per end, so drawn 460 nm lines really span only ≈ ±150 nm.
    # Wide enough in x to cross the two spacer lines flanking the central
    # mandrel.
    return Layout([cut_bar(0.0, 60e-9, 160e-9, 80e-9)], name="cut")


@pytest.fixture(scope="module")
def sadp_nocut(grid, optics, resist, mandrel):
    return sadp(mandrel, grid, optics, resist).run(snapshot=False)


@pytest.fixture(scope="module")
def sadp_litho_cut(grid, optics, resist, mandrel, sadp_cut_bar):
    flow = sadp(mandrel, grid, optics, resist, cut_layout=sadp_cut_bar)
    return flow.run(snapshot=False)


@pytest.fixture(scope="module")
def sadp_ideal_cut(grid, optics, resist, mandrel, sadp_cut_bar):
    flow = sadp(mandrel, grid, optics, resist, cut_layout=sadp_cut_bar,
                cut_style="ideal")
    return flow.run(snapshot=False)


def test_sadp_cut_removes_material(sadp_nocut, sadp_litho_cut):
    v_nocut = _poly_presence(sadp_nocut.stack).sum()
    v_cut = _poly_presence(sadp_litho_cut.stack).sum()
    assert v_cut < v_nocut, "the cut must remove target material"


def test_sadp_cut_is_not_a_noop(sadp_litho_cut):
    """Regression: the cut used to run while spacer capped the lines, so the
    etch found no exposed target and silently did nothing."""
    history = "\n".join(sadp_litho_cut.stack.history)
    assert "no-op" not in history, f"silent no-op in flow history:\n{history}"


def test_sadp_cut_preserves_pitch(sadp_nocut, sadp_litho_cut):
    """Away from the (off-centre) cut, the line count must be unchanged."""
    final_nocut = sadp_nocut.measurements.iloc[-1]
    final_cut = sadp_litho_cut.measurements.iloc[-1]
    assert final_cut["n_lines"] == final_nocut["n_lines"]


def test_ideal_cut_matches_litho_cut_topology(
    grid, sadp_litho_cut, sadp_ideal_cut, sadp_cut_bar
):
    """Both cut styles must sever every line crossing the bar.

    Assertions run on the line *thickness* map, not raw presence: SADP's
    target etch under-etches by rate (40 nm budget × poly rate 0.8 = 32 nm
    of a 40 nm film), leaving an ~8 nm poly slab everywhere — a known
    deferred defect (roadmap phase 3). Lines are where the film is thick.
    The litho cut's printed slot is also much thinner than drawn at this
    dose calibration (an isolated slot, dosed for dense trenches), which is
    fine: a thin cut still cuts.
    """
    window = sadp_cut_bar.rasterize(grid) > 0.5
    bar_rows = np.where(window.any(axis=1))[0]
    bar_cols = window.any(axis=0)

    for name, result in (("litho", sadp_litho_cut), ("ideal", sadp_ideal_cut)):
        lines = np.asarray(result.stack.thickness_of("poly-Si")) > 20e-9
        # Line columns that actually cross the bar: inside its x-extent and
        # thick on both sides of its rows. (Lines of the neighbouring
        # mandrels pass OUTSIDE the bar and must stay uncut.)
        crossing = [
            c for c in range(grid.n_pixels)
            if bar_cols[c] and lines[bar_rows[0] - 4, c] and lines[bar_rows[-1] + 4, c]
        ]
        assert crossing, f"{name}: expected spacer lines crossing the bar"
        for c in crossing:
            assert _runs(lines[:, c]) == 2, (
                f"{name}: line at column {c} was not severed"
            )

    # The ideal cut is geometry-gated, so its window is cleared exactly.
    ideal_lines = np.asarray(sadp_ideal_cut.stack.thickness_of("poly-Si")) > 20e-9
    assert not ideal_lines[window].any()


def test_add_cut_mask_inserts_before_measure(grid, optics, resist, mandrel,
                                             sadp_cut_bar):
    flow = sadp(mandrel, grid, optics, resist)
    add_cut_mask(flow, sadp_cut_bar, style="ideal")
    assert isinstance(flow.steps[-1], Measure)
    assert "through cut" in flow.steps[-2].describe()
    result = flow.run(snapshot=False)
    assert len(result.measurements) > 0
