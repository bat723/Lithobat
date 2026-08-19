"""
Process flows — an ordered list of steps, run against a wafer stack.

A :class:`Flow` is deliberately just a list. Everything interesting about
LELE, LE³, SADP, and SAQP lives in the *order* of the steps and the overlay
values, not in special-case code. See :mod:`litho_sim.patterning.multipatterning`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.core.utils import JsonFileMixin
from litho_sim.mask.layout import Layout
from litho_sim.patterning.steps import ProcessContext, ProcessStep
from litho_sim.wafer import Stack

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Everything a flow produced.

    Attributes
    ----------
    stack : Stack
        Final wafer state.
    snapshots : list of Stack
        A copy after each step, for scrubbing back through the flow.
        Empty when ``snapshot=False``.
    measurements : DataFrame
        Rows recorded by :class:`~litho_sim.patterning.steps.Measure`.
    log : list of str
        One line per step.
    """

    stack: Stack
    snapshots: list[Stack] = field(default_factory=list)
    measurements: pd.DataFrame = field(default_factory=pd.DataFrame)
    log: list[str] = field(default_factory=list)


@dataclass
class Flow(JsonFileMixin):
    """An ordered process recipe.

    Parameters
    ----------
    steps : list of ProcessStep
        The recipe.
    name : str
        Identifier.
    layouts : dict
        ``{name: Layout}`` that :class:`~litho_sim.patterning.steps.Expose`
        steps refer to.
    grid, optics, resist : config objects
        Defaults applied to every step.
    """

    steps: list[ProcessStep] = field(default_factory=list)
    name: str = "flow"
    layouts: dict[str, Layout] = field(default_factory=dict)
    grid: GridConfig = field(default_factory=GridConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    resist: ResistConfig = field(default_factory=ResistConfig)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add(self, *steps: ProcessStep) -> Flow:
        """Append steps; returns self so calls chain."""
        self.steps.extend(steps)
        return self

    def describe(self) -> list[str]:
        """One human-readable line per step, for a UI listing."""
        return [f"{i + 1:2d}. {s.describe()}" for i, s in enumerate(self.steps)]

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run(
        self,
        stack: Stack | None = None,
        snapshot: bool = True,
        on_step: Callable[[int, ProcessStep, Stack], None] | None = None,
    ) -> RunResult:
        """Execute the recipe.

        Parameters
        ----------
        stack : Stack, optional
            Starting wafer.  Defaults to a bare substrate.
        snapshot : bool
            Keep a copy of the stack after every step.  Costs memory but is
            what lets a UI scrub through the flow.
        on_step : callable, optional
            Called as ``on_step(index, step, stack)`` after each step —
            useful for a progress bar.

        Returns
        -------
        RunResult
        """
        if stack is None:
            stack = Stack.blank(self.grid, dz=self.grid.dz)

        ctx = ProcessContext(
            grid=self.grid, optics=self.optics, resist=self.resist,
            layouts=dict(self.layouts),
        )
        snaps: list[Stack] = []
        log: list[str] = []

        for i, step in enumerate(self.steps):
            t0 = time.perf_counter()
            try:
                stack = step.apply(stack, ctx)
            except Exception as exc:
                raise RuntimeError(
                    f"Step {i + 1} ({step.describe()}) failed: {exc}"
                ) from exc
            dt = time.perf_counter() - t0
            line = f"{i + 1:2d}. {step.describe()}  [{dt:.2f}s]"
            log.append(line)
            logger.info("%s: %s", self.name, line)
            if snapshot:
                snaps.append(stack.copy())
            if on_step is not None:
                on_step(i, step, stack)

        return RunResult(
            stack=stack,
            snapshots=snaps,
            measurements=pd.DataFrame(ctx.measurements),
            log=log,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "layouts": {k: v.to_dict() for k, v in self.layouts.items()},
            "grid": asdict(self.grid),
            "optics": asdict(self.optics),
            "resist": asdict(self.resist),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Flow:
        optics_data = dict(data.get("optics", {}))
        # JSON turns integer keys into strings; Zernike indices must come back
        # as ints or the pupil silently gets no aberrations.
        if "zernike_coeffs" in optics_data:
            optics_data["zernike_coeffs"] = {
                int(k): float(v) for k, v in optics_data["zernike_coeffs"].items()
            }
        return cls(
            steps=[ProcessStep.from_dict(d) for d in data.get("steps", [])],
            name=data.get("name", "flow"),
            layouts={
                k: Layout.from_dict(v) for k, v in data.get("layouts", {}).items()
            },
            grid=GridConfig(**data.get("grid", {})),
            optics=OpticsConfig(**optics_data),
            resist=ResistConfig(**data.get("resist", {})),
        )

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Flow({self.name!r}, {len(self.steps)} steps)"
