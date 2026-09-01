"""Interface-level tests: cut masks and the stack specification.

Cuts are exercised through the public ``litho_sim.patterning`` namespace on
the fast litho-free route (``ideal_cut``): print target lines with the ideal
``Pattern`` step, sever them with a geometry-gated etch, and assert the
severing happened exactly inside the drawn window and nowhere else.  The
litho-route cut behaviour (tone table, block/keep semantics, litho-vs-ideal
topology) lives in tests/test_cuts.py and is not repeated.

The stackspec is a module of shared material names; the test pins the
contract that matters — every name resolves in the wafer material registry,
and the cut builders default to the same TARGET the recipes pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig
from litho_sim.mask.layout import Layout, cut_bar
from litho_sim.patterning import (
    Etch,
    Flow,
    Measure,
    Pattern,
    ProcessStep,
    SpinCoat,
    Strip,
    add_cut_mask,
    block_resist,
    cut_block,
    ideal_cut,
    keep_block,
)
from litho_sim.patterning.stackspec import (
    HARDMASK,
    MANDREL,
    RESIST,
    SPACER1,
    SPACER2,
    TARGET,
)
from litho_sim.wafer import get_material


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=64, pixel_size=4e-9, dz=4e-9, n_z_slices=5)


@pytest.fixture(scope="module")
def cut_layout() -> Layout:
    # A bar across the whole field at y = 0: severs every vertical line.
    return Layout([cut_bar(0.0, 0.0, 600e-9, 48e-9)], name="cut")


# ---------------------------------------------------------------------------
# Happy path: an ideal cut severs printed lines, exactly where drawn
# ---------------------------------------------------------------------------


def test_ideal_cut_severs_lines_only_inside_the_drawn_window(grid, cut_layout):
    flow = Flow(name="cut-demo", grid=grid, layouts={"cut": cut_layout})
    flow.add(
        Pattern(material=TARGET, thickness=40e-9,
                shape="lines", pitch=128e-9, cd=64e-9),
        *ideal_cut("cut", target=TARGET, depth=60e-9),
    )
    result = flow.run(snapshot=True)

    before = result.snapshots[0].thickness_of(TARGET)  # after Pattern
    after = result.stack.thickness_of(TARGET)          # after the cut
    window = cut_layout.rasterize(grid, tone="clear") > 0.5

    # The bar really crossed material...
    assert (before[window] > 0).any()
    # ...and inside the window every target column is now cleared,
    assert after[window].max() == 0.0
    # ...while outside the window nothing was touched.
    assert np.array_equal(after[~window], before[~window])
    # The lines survive away from the cut: still material above and below it.
    assert (after > 0).any()


def test_cut_builders_return_composable_process_steps():
    """The builders hand back plain step lists a Flow can splice anywhere."""
    for steps in (
        cut_block("cut"),
        block_resist("cut"),
        keep_block("cut"),
        ideal_cut("cut"),
    ):
        assert isinstance(steps, list) and steps
        assert all(isinstance(s, ProcessStep) for s in steps)

    # The litho cut cleans up after itself; a block leaves resist standing
    # for the caller's etch by design.
    assert isinstance(cut_block("cut")[-1], Strip)
    assert not any(isinstance(s, (Etch, Strip)) for s in block_resist("cut"))
    # The ideal route is a single geometry-gated etch.
    (only,) = ideal_cut("cut")
    assert isinstance(only, Etch) and only.open_layout == "cut"


# ---------------------------------------------------------------------------
# Stackspec: the shared names actually exist on the wafer side
# ---------------------------------------------------------------------------


def test_stackspec_names_resolve_in_the_material_registry():
    names = {HARDMASK, RESIST, TARGET, MANDREL, SPACER1, SPACER2}
    assert len(names) == 6, "stackspec roles must be distinct materials"
    for name in names:
        assert isinstance(name, str) and name
        mat = get_material(name)  # raises if the registry lacks it
        assert mat.id >= 0


def test_cut_builders_default_to_the_stackspec_target(grid):
    """cuts and stackspec agree on what is being patterned."""
    etch = next(s for s in cut_block("cut") if isinstance(s, Etch))
    assert etch.targets == TARGET
    resist_coat = next(s for s in cut_block("cut") if isinstance(s, SpinCoat))
    assert resist_coat.material == RESIST


# ---------------------------------------------------------------------------
# Edge case: invalid cut parameters fail loudly
# ---------------------------------------------------------------------------


def test_add_cut_mask_rejects_an_unknown_style(grid, cut_layout):
    flow = Flow(name="victim", grid=grid).add(Measure(name="final"))
    with pytest.raises(ValueError, match="style"):
        add_cut_mask(flow, cut_layout, style="sideways")
