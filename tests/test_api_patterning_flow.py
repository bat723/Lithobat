"""Interface-level tests: Flow orchestration through ``litho_sim.patterning``.

Everything here goes through the public package namespace — build a small
recipe from exported steps, run it, and inspect the RunResult contract
(final stack, per-step snapshots, log, measurements).  Unit-level flow
behaviour (snapshot independence, callbacks, JSON round-trips, failure
attribution for a custom step) lives in tests/test_process.py and is not
repeated here.

Kept fast on purpose: a 64 px grid and the litho-free ``Pattern`` step, so
no optics are simulated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from litho_sim.core.config import GridConfig
from litho_sim.patterning import (
    Deposit,
    Etch,
    Flow,
    Measure,
    Pattern,
    RunResult,
    Strip,
)
from litho_sim.wafer import Stack


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=64, pixel_size=4e-9, dz=4e-9, n_z_slices=5)


def _small_recipe(grid: GridConfig) -> Flow:
    """Deposit a target film, print an ideal mask, transfer, strip, measure."""
    flow = Flow(name="api-happy", grid=grid)
    flow.add(
        Deposit(material="poly-Si", thickness=40e-9, conformal=False),
        Pattern(material="photoresist", thickness=80e-9,
                shape="lines", pitch=64e-9, cd=32e-9),
        Etch(targets="poly-Si", depth=60e-9,
             selectivity={"photoresist": 0.1}),
        Strip(material="photoresist"),
        Measure(name="final", material="poly-Si"),
    )
    return flow


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_flow_builds_and_describes(grid):
    flow = _small_recipe(grid)
    assert len(flow) == 5
    lines = flow.describe()
    assert len(lines) == 5
    # One numbered, human-readable line per step, in recipe order.
    assert lines[0].startswith(" 1.")
    assert "Deposit poly-Si" in lines[0]
    assert "Measure" in lines[-1]


def test_flow_run_returns_a_complete_runresult(grid):
    flow = _small_recipe(grid)
    result = flow.run(snapshot=True)

    assert isinstance(result, RunResult)
    assert isinstance(result.stack, Stack)
    # One snapshot and one log line per step.
    assert len(result.snapshots) == len(flow)
    assert len(result.log) == len(flow)
    assert result.log[0].startswith(" 1.")

    # The recipe actually happened: resist is gone, patterned poly remains.
    assert result.stack.thickness_of("photoresist").max() == 0.0
    assert result.stack.thickness_of("poly-Si").max() > 0.0

    # The Measure step landed in the measurements DataFrame.
    assert isinstance(result.measurements, pd.DataFrame)
    assert len(result.measurements) == 1
    row = result.measurements.iloc[0]
    assert row["name"] == "final"
    # 32 nm lines at 64 nm pitch on a 4 nm grid: 4 interior lines.
    assert row["n_lines"] == 4
    assert row["mean_line_nm"] == pytest.approx(32.0, abs=8.0)


def test_flow_run_without_snapshots_and_with_a_given_stack(grid):
    flow = _small_recipe(grid)
    start = Stack.blank(grid, dz=grid.dz)
    result = flow.run(stack=start, snapshot=False)
    assert result.snapshots == []
    # The provided stack was consumed as the starting wafer.
    assert result.stack.thickness_of("poly-Si").max() > 0.0
    assert len(result.log) == len(flow)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_running_an_empty_flow_yields_a_bare_wafer(grid):
    result = Flow(name="empty", grid=grid).run()
    assert isinstance(result, RunResult)
    assert result.log == []
    assert result.snapshots == []
    assert len(result.measurements) == 0
    # Nothing but the substrate.
    assert isinstance(result.stack, Stack)
    assert result.stack.thickness_of("Si").max() > 0.0
    assert result.stack.thickness_of("photoresist").max() == 0.0


def test_invalid_step_parameters_surface_through_run(grid):
    """A bad parameter on a real step fails loudly, naming the step."""
    flow = Flow(name="bad-shape", grid=grid).add(
        Pattern(shape="widgets")  # only "lines" | "contacts" are valid
    )
    with pytest.raises(RuntimeError, match=r"Step 1 .*widgets"):
        flow.run(snapshot=False)
