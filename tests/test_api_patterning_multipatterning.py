"""Interface-level tests: multi-patterning recipes through ``litho_sim.patterning``.

A minimal SADP run on a small (96 px) field verifies that pitch division
actually happened geometrically, using only the public measurements
interface.  The heavyweight physics assertions (spacer CD equals deposited
thickness, SAQP runs, LELE pitch walking) live in tests/test_process.py on
a 160 px field and are not repeated; recipe *structure* is checked here
build-only, which costs nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask.layout import Layout, cut_bar, line_array
from litho_sim.patterning import (
    Flow,
    SpacerEtchback,
    lele,
    sadp,
    saqp,
    single_exposure,
)

N_DRAWN = 3          # drawn mandrel lines
DRAWN_PITCH = 128e-9  # the pitch SADP is supposed to divide


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    # 384 nm field: the smallest that holds three printable mandrel lines.
    return GridConfig(n_pixels=96, pixel_size=4e-9, dz=4e-9, n_z_slices=5)


@pytest.fixture(scope="module")
def optics() -> OpticsConfig:
    return OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=7)


@pytest.fixture(scope="module")
def resist() -> ResistConfig:
    return ResistConfig(
        n_resist=1.7, dill_A=0.8, dill_B=0.05, dill_C=0.04,
        dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9,
    )


@pytest.fixture(scope="module")
def mandrel() -> Layout:
    return Layout(
        line_array(N_DRAWN, pitch=DRAWN_PITCH, cd=64e-9, length=900e-9),
        name="mandrel",
    )


@pytest.fixture(scope="module")
def sadp_result(grid, optics, resist, mandrel):
    """One SADP run shared by the geometric assertions below."""
    flow = sadp(mandrel, grid, optics, resist, spacer_thickness=16e-9)
    return flow.run(snapshot=False)


# ---------------------------------------------------------------------------
# Happy path: minimal SADP, pitch division verified geometrically
# ---------------------------------------------------------------------------


def test_sadp_builds_a_flow_with_the_mandrel_layout(grid, optics, resist, mandrel):
    flow = sadp(mandrel, grid, optics, resist)
    assert isinstance(flow, Flow)
    assert "mandrel" in flow.layouts
    assert len(flow) > 0


def test_sadp_run_records_every_checkpoint(sadp_result):
    names = list(sadp_result.measurements["name"])
    assert names == ["mandrel", "spacers", "final"]


def test_sadp_divides_the_pitch_geometrically(sadp_result):
    by_name = {r["name"]: r for _, r in sadp_result.measurements.iterrows()}
    spacers = by_name["spacers"]
    final = by_name["final"]

    # More printed lines than drawn mandrels — the count multiplied.
    assert spacers["n_lines"] > N_DRAWN
    assert final["n_lines"] > N_DRAWN

    # And the resulting pitch (line + space) is well under the drawn pitch:
    # deposition, not lithography, now sets the period.
    final_pitch_nm = final["mean_line_nm"] + final["mean_space_nm"]
    assert final_pitch_nm < 0.75 * DRAWN_PITCH * 1e9, (
        f"final pitch {final_pitch_nm:.0f} nm did not divide the "
        f"{DRAWN_PITCH*1e9:.0f} nm drawn pitch"
    )


# ---------------------------------------------------------------------------
# Recipe structure (build-only — no physics run)
# ---------------------------------------------------------------------------


def test_single_exposure_structure(grid, optics, resist, mandrel):
    flow = single_exposure(mandrel, grid, optics, resist)
    assert isinstance(flow, Flow)
    assert set(flow.layouts) == {"main"}
    text = "\n".join(flow.describe())
    assert "Expose main" in text
    assert "Measure" in text


def test_lele_structure_splits_the_layout_in_two(grid, optics, resist):
    dense = Layout(
        line_array(4, pitch=80e-9, cd=40e-9, length=600e-9), name="dense"
    )
    flow = lele(dense, grid, optics, resist, min_spacing=90e-9)
    # Two colours, each with its own expose step.
    assert len(flow.layouts) == 2
    text = "\n".join(flow.describe())
    for key in flow.layouts:
        assert f"Expose {key}" in text


def test_saqp_is_the_sadp_block_twice(grid, optics, resist, mandrel):
    flow = saqp(mandrel, grid, optics, resist)
    etchbacks = [s for s in flow.steps if isinstance(s, SpacerEtchback)]
    assert len(etchbacks) == 2, "SAQP must run two spacer generations"
    assert {e.material for e in etchbacks} == {"spacer-oxide", "spacer-nitride"}


# ---------------------------------------------------------------------------
# Edge case: invalid recipe parameters fail at build time
# ---------------------------------------------------------------------------


def test_sadp_rejects_an_unknown_cut_style(grid, optics, resist, mandrel):
    cut = Layout([cut_bar(0.0, 0.0, 60e-9, 60e-9)], name="cut")
    with pytest.raises(ValueError, match="cut_style"):
        sadp(mandrel, grid, optics, resist,
             cut_layout=cut, cut_style="bogus")
