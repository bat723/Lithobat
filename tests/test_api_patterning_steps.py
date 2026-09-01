"""
Interface-level tests for the process-step vocabulary.

Everything here goes through ``litho_sim.patterning``'s public exports and
exercises contracts the step-behaviour suites (test_process, test_cuts,
test_stepspecs) do not:

* A **hand-built print-and-transfer cycle** — coat, expose, PEB, develop,
  measure, transfer etch, strip, re-measure — asserting the wafer actually
  changes at each stage and the pattern survives into the substrate.
* The **ProcessContext.state contract**: Expose leaves a latent image, PEB
  smooths it in place (and records a diffusion override), Develop consumes
  it and writes resist into the stack.
* **CMP applied as a step** in each of its addressing modes. The behaviour
  lives in Stack.planarize (tested in test_stack); what is tested here is
  that the *step* forwards its fields — historically it was only ever
  round-tripped, never run.
* The **register_step extension point**: a step defined outside the package
  participates in the registry, from_dict, and a Flow run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask.layout import Layout, line_array
from litho_sim.patterning import (
    CMP,
    STEP_REGISTRY,
    Deposit,
    Develop,
    Etch,
    Expose,
    Flow,
    Measure,
    Pattern,
    PostExposureBake,
    ProcessContext,
    ProcessStep,
    SpacerEtchback,
    SpinCoat,
    Strip,
    register_step,
)
from litho_sim.wafer import Stack


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    """256 nm field — room for two interior 64 nm lines at 128 nm pitch."""
    return GridConfig(n_pixels=64, pixel_size=4e-9, dz=4e-9, n_z_slices=7)


@pytest.fixture(scope="module")
def small_grid() -> GridConfig:
    return GridConfig(n_pixels=48, pixel_size=4e-9, dz=4e-9, n_z_slices=7)


@pytest.fixture(scope="module")
def optics() -> OpticsConfig:
    return OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=11)


@pytest.fixture(scope="module")
def resist() -> ResistConfig:
    # Same calibration as test_process: printed CD tracks drawn CD at dose 1.
    return ResistConfig(
        n_resist=1.7, dill_A=0.8, dill_B=0.05, dill_C=0.04,
        dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9,
    )


@pytest.fixture(scope="module")
def gates() -> Layout:
    """Two lines at 128 nm pitch / 64 nm CD — comfortably printable at 193i."""
    return Layout(line_array(2, pitch=128e-9, cd=64e-9, length=600e-9),
                  name="gates")


# ---------------------------------------------------------------------------
# End-to-end: print a level and transfer it into the substrate
# ---------------------------------------------------------------------------


def test_print_and_transfer_cycle(grid, optics, resist, gates):
    """The nine-step vocabulary, composed by hand, prints and transfers.

    Dark-tone exposure leaves resist lines where the layout draws them; the
    transfer etch then carves the substrate only between those lines (the
    emergent mask), and after the strip the pattern lives on in silicon.
    """
    drawn_nm = 64.0
    flow = Flow(
        steps=[
            SpinCoat(material="photoresist", thickness=90e-9),
            # 0.9 is dose-to-size for this two-line dark-field level: the
            # threshold model's response is steep here (1.0 clears the film,
            # 0.8 leaves it unbroken), but it is deterministic.
            Expose(layout="gates", dose=0.9, tone="dark"),
            PostExposureBake(),
            Develop(),
            Measure(name="resist_cd", material="photoresist"),
            # The base of the line, not mid-height: absorption leaves the
            # bottom less exposed, so a positive-tone line stands on a wider
            # foot — and the *footprint* is what masks the transfer etch.
            Measure(name="resist_base", material="photoresist", z_frac=0.05),
            Etch(targets="Si", depth=30e-9, selectivity={"Si": 1.0}),
            Strip(material="photoresist"),
            Measure(name="si_cd", material="Si"),
        ],
        name="print-and-transfer",
        layouts={"gates": gates},
        grid=grid, optics=optics, resist=resist,
    )

    # The recipe reads back as one numbered line per step.
    lines = flow.describe()
    assert len(lines) == len(flow.steps) == 9
    assert lines[0].startswith(" 1.") and "photoresist" in lines[0]

    result = flow.run(snapshot=True)

    # Flow bookkeeping: one snapshot and one log line per step.
    assert len(result.snapshots) == 9
    assert len(result.log) == 9

    # Every Measure step landed as a row.
    assert list(result.measurements["name"]) == [
        "resist_cd", "resist_base", "si_cd"]

    # The printed resist: two interior lines near the drawn CD.
    resist_row = result.measurements.iloc[0]
    base_row = result.measurements.iloc[1]
    assert resist_row["n_lines"] == base_row["n_lines"] == 2
    assert abs(resist_row["mean_line_nm"] - drawn_nm) <= 20.0
    # Absorption footing: the base is at least as wide as mid-height.
    assert base_row["mean_line_nm"] >= resist_row["mean_line_nm"]

    # The strip really removed the resist...
    stack = result.stack
    assert stack.thickness_of("photoresist").max() == 0.0

    # ...and the etch really carved the substrate between the lines: the
    # 40 nm slab keeps its full height under the (former) resist and lost
    # 30 nm where it was open.  One voxel of slack for dz quantisation.
    t_si = stack.thickness_of("Si")
    carved = float(t_si.max() - t_si.min())
    assert abs(carved - 30e-9) <= 4.5e-9
    assert t_si.max() == pytest.approx(40e-9, abs=4.5e-9)

    # The transferred lines match the resist *footprint* — the emergent
    # mask is whatever the column wears, i.e. the base of the line.
    si_row = result.measurements.iloc[2]
    assert si_row["n_lines"] == 2
    assert abs(si_row["mean_line_nm"] - base_row["mean_line_nm"]) <= 8.5

    # A vertical anisotropic etch introduces no alternation.
    assert si_row["pitch_walk_nm"] <= 8.0


# ---------------------------------------------------------------------------
# The ProcessContext.state contract between Expose, PEB, and Develop
# ---------------------------------------------------------------------------


def test_context_carries_the_latent_image(small_grid, optics, resist, gates):
    """Expose deposits state, PEB smooths it in place, Develop consumes it."""
    ctx = ProcessContext(grid=small_grid, optics=optics, resist=resist,
                         layouts={"gates": gates})
    stack = Stack.blank(small_grid, dz=small_grid.dz)

    SpinCoat(material="photoresist", thickness=90e-9).apply(stack, ctx)
    assert stack.thickness_of("photoresist").max() == pytest.approx(90e-9, abs=4.5e-9)

    Expose(layout="gates", dose=1.0, tone="dark").apply(stack, ctx)
    assert "pac" in ctx.state and "resist_cfg" in ctx.state
    pac_before = ctx.state["pac"].copy()
    assert pac_before.ndim == 3
    assert pac_before.std() > 0.0  # the image actually modulates

    PostExposureBake(diffusion_sigma=25e-9).apply(stack, ctx)
    pac_after = ctx.state["pac"]
    # Diffusion smooths: contrast can only go down, and measurably so.
    assert pac_after.std() < pac_before.std()
    # The override is recorded so Develop sees the same chemistry.
    assert ctx.state["resist_cfg"].diffusion_sigma == pytest.approx(25e-9)

    Develop().apply(stack, ctx)
    assert "pac" not in ctx.state          # consumed
    assert "remaining" in ctx.state
    t = stack.thickness_of("photoresist")
    assert t.max() > 0.0                   # a line survived
    assert t.min() == 0.0                  # and the spaces cleared


# ---------------------------------------------------------------------------
# CMP as a step
# ---------------------------------------------------------------------------


def test_cmp_step_forwards_each_addressing_mode(small_grid):
    ctx = ProcessContext(grid=small_grid)
    stack = Stack.blank(small_grid, dz=small_grid.dz)

    # Topography without litho: 60 nm of patterned carbon on the substrate.
    Pattern(material="SOC", thickness=60e-9, shape="lines",
            pitch=96e-9, cd=48e-9).apply(stack, ctx)
    top = stack.top_height()
    assert top.max() > top.min()  # there is something to planarise

    # No-argument CMP: down to the lowest peak (the open substrate), flat.
    CMP().apply(stack, ctx)
    top = stack.top_height()
    assert top.max() == top.min()
    assert top.max() == pytest.approx(40e-9, abs=4.5e-9)

    # depth mode: take a known amount off the new surface.
    SpinCoat(material="SOC", thickness=20e-9).apply(stack, ctx)
    before = float(stack.top_height().max())
    CMP(depth=8e-9).apply(stack, ctx)
    assert before - float(stack.top_height().max()) == pytest.approx(8e-9, abs=2.1e-9)

    # Conflicting modes are rejected with the reason.
    with pytest.raises(ValueError, match="at most one"):
        CMP(depth=10e-9, stop_on="Si").apply(stack, ctx)


# ---------------------------------------------------------------------------
# Errors and edges
# ---------------------------------------------------------------------------


def test_pattern_rejects_an_unknown_shape(small_grid):
    ctx = ProcessContext(grid=small_grid)
    stack = Stack.blank(small_grid, dz=small_grid.dz)
    with pytest.raises(ValueError, match="lines.*contacts"):
        Pattern(shape="dots").apply(stack, ctx)


def test_every_exported_step_describes_itself():
    """describe() is the recipe panel's one line — never empty, never raising."""
    for cls in (SpinCoat, Deposit, Pattern, Expose, PostExposureBake, Develop,
                Etch, SpacerEtchback, Strip, CMP, Measure):
        text = cls().describe()
        assert isinstance(text, str) and text.strip()


# ---------------------------------------------------------------------------
# The extension point
# ---------------------------------------------------------------------------


def test_register_step_admits_an_outside_step(small_grid):
    """A step defined outside the package joins the registry, the dict
    round-trip, and a Flow run — the contract tech/flows.py's ``step()``
    helper and the app's palette both build on."""
    kind = "api-test-annotate"
    try:
        @register_step(kind)
        @dataclass
        class Annotate(ProcessStep):
            note: str = "hello"
            kind: str = "api-test-annotate"

            def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
                stack.history.append(f"annotate: {self.note}")
                return stack

        assert STEP_REGISTRY[kind] is Annotate

        # from_dict finds it by kind, exactly as a saved recipe would.
        revived = ProcessStep.from_dict({"kind": kind, "note": "via-dict"})
        assert isinstance(revived, Annotate)
        assert revived.note == "via-dict"
        assert ProcessStep.from_dict(revived.to_dict()) == revived

        # And it runs inside a Flow like any built-in.
        result = Flow(steps=[revived], grid=small_grid).run(snapshot=False)
        assert any("via-dict" in line for line in result.stack.history)
        assert len(result.log) == 1
    finally:
        STEP_REGISTRY.pop(kind, None)
