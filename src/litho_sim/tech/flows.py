"""Shared machinery for printed-device process flows.

The device flows themselves (:mod:`litho_sim.tech.gaa`,
:mod:`litho_sim.tech.nfet`) are runnable, documented narratives. What they
share — the litho level applied as real process steps, the narration and
recording hooks, the dose-calibration sweep, and the morphology their
structural checks lean on — lives here, so each flow reads as *its recipe*
rather than as a copy of the other's plumbing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from litho_sim.patterning import STEP_REGISTRY
from litho_sim.patterning.steps import ProcessContext, ProcessStep
from litho_sim.wafer import Stack, get_material

logger = logging.getLogger(__name__)


def step(kind: str, **kw) -> ProcessStep:
    """Build one process step from the registry, by its recipe name."""
    return STEP_REGISTRY[kind](**kw)


class FlowRun:
    """One device build: the wafer, its context, and the flow's hooks.

    Every process operation goes through :meth:`do` rather than a direct
    :class:`~litho_sim.wafer.Stack` call. That is not ceremony: a step carries
    its parameters as data, so the flow can be *shown* — the app's recipe
    panel lists what actually ran, with the selectivity and exposure mode a
    history string throws away.

    Parameters
    ----------
    stack, ctx : Stack, ProcessContext
        The wafer being built and the optics/resist/layouts it is built with.
    verbose : bool
        Route :meth:`say` narration to stdout.
    on_step : callable, optional
        Called ``(step, stack)`` after each step, in order, and once as
        ``(None, stack)`` on construction for the bare wafer — the one state
        no step produced. The device cache uses it to collect the recipe and
        a snapshot per step.
    on_context : callable, optional
        Called once with *ctx*. Without it a recorded recipe cannot be
        *re-run*: an ``Expose`` naming a layout has nowhere to look it up,
        and a re-exposure would fall back to library-default optics.
    """

    def __init__(
        self,
        stack: Stack,
        ctx: ProcessContext,
        verbose: bool = True,
        on_step: Callable | None = None,
        on_context: Callable | None = None,
    ):
        self.stack = stack
        self.ctx = ctx
        self.metrics: dict[str, float] = {}
        self._say = print if verbose else (lambda *a, **k: None)
        self._on_step = on_step
        if on_context is not None:
            on_context(ctx)
        if on_step is not None:
            # `None` announces the bare wafer, before any step: the state a
            # scrubber's first position shows, and the only one no step
            # produced.
            on_step(None, stack)

    def say(self, message: str) -> None:
        """Narrate one line of the flow, if the run is verbose."""
        self._say(message)

    def do(self, kind: str, **kw) -> ProcessStep:
        """Apply one step to the wafer, and tell the recorder what ran."""
        s = step(kind, **kw)
        s.apply(self.stack, self.ctx)
        if self._on_step is not None:
            self._on_step(s, self.stack)
        return s

    def print_level(
        self, name: str, layout: str, dose: float, measure_along: str = "x"
    ) -> float:
        """Real lithography, on the device wafer: coat, expose, develop.

        The resist thickness comes from the context's :class:`ResistConfig`,
        so the coat and the develop model always agree on the film.

        Records and returns the printed resist width [nm] perpendicular to
        the feature (*measure_along* is that perpendicular axis), so the flow
        can report drawn-vs-printed for each level. The metric lands in
        :attr:`metrics` as ``"<name>_printed_nm"``.
        """
        self.do("spincoat", material="photoresist",
                thickness=self.ctx.resist.thickness)
        self.do("expose", layout=layout, dose=dose, tone="dark")
        self.do("develop")
        resist = (self.stack.mat == get_material("photoresist").id).any(axis=0)
        n = resist.shape[0]
        line = resist[:, n // 2] if measure_along == "y" else resist[n // 2, :]
        printed = float(line.sum()) * self.ctx.grid.pixel_size * 1e9
        self.metrics[f"{name}_printed_nm"] = printed
        return printed


# ---------------------------------------------------------------------------
# Structural-check morphology
# ---------------------------------------------------------------------------


def adjacent_to(occupied: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Voxels face-adjacent to *occupied* (6-neighbourhood, no diagonals).

    The building block of the gate-shorts check: gate metal is shorted
    exactly where it lands adjacent to channel silicon.
    """
    nb = np.zeros_like(occupied)
    for ax in (0, 1, 2):
        for sh in (1, -1):
            nb |= np.roll(occupied, sh, axis=ax)
    return nb


def contact_count(stack: Stack, a: str, b: str) -> int:
    """How many voxels of material *a* touch material *b* face-on."""
    return int(((stack.mat == get_material(a).id)
                & adjacent_to(stack.mat == get_material(b).id)).sum())


# ---------------------------------------------------------------------------
# Dose calibration
# ---------------------------------------------------------------------------


def run_calibration(
    builder: Callable,
    levels: Sequence[tuple[str, float]],
    doses: Iterable[float],
    dose_keys: Sequence[str],
    **build_kw,
) -> None:
    """Sweep dose for each litho level and print the printed CD at each.

    *levels* pairs a ``stop_after`` name with the drawn CD it targets, e.g.
    ``("fin_litho", 96.0)``. Every entry in *dose_keys* is set to the swept
    dose — matching how dose-to-size is actually read off, one level at a
    time with the flow stopped right after it prints.
    """
    for level, drawn in levels:
        key = level.split("_")[0]
        print(f"{key}: drawn {drawn:.0f} nm")
        for dose in doses:
            kw = dict(build_kw)
            kw.update({k: dose for k in dose_keys})
            _, m = builder(verbose=False, stop_after=level, **kw)
            print(f"   dose {dose:.2f} -> {m[f'{key}_printed_nm']:6.1f} nm")
