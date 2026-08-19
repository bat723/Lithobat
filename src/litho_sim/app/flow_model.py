"""
Editing a process flow while it runs.

:meth:`~litho_sim.patterning.flow.Flow.run` is all-or-nothing: it builds a
fresh :class:`~litho_sim.patterning.steps.ProcessContext`, walks every step, and
hands back a finished stack. That is right for a recipe you already have, and
wrong for one you are still writing, where appending a step should cost only
that step and editing step 4 of 20 should not re-run steps 1 to 3.

This module owns the mutable middle: a step list, the stack, the context, and a
snapshot per completed step. Qt-free, like ``params`` / ``compute`` /
``pipeline`` — a test asserts that boundary holds.

The rule that shapes it
-----------------------
``ctx.state`` carries the **in-flight latent image** between ``Expose``,
``PostExposureBake`` and ``Develop``. ``Expose`` writes ``state["pac"]``, the
bake mutates it, and ``Develop`` consumes and pops it. So the context is not
incidental bookkeeping that can be rebuilt on demand — resuming from a snapshot
is only sound where ``state`` is *empty*, which is to say at a step boundary
outside a litho triplet.

Editing a step therefore replays from the nearest **safe boundary** at or before
it, not from the step itself. Resuming inside a triplet would either raise
(``Develop`` with no latent: *"no Expose step has run since the last
Develop"*) or, worse, develop a latent left over from a previous exposure.

Snapshots are the caller's memory budget, not the engine's: steps mutate the
:class:`~litho_sim.wafer.stack.Stack` in place and return it, so anything worth
keeping has to be copied.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.patterning.steps import ProcessContext, ProcessStep
from litho_sim.wafer import Stack

logger = logging.getLogger(__name__)

__all__ = ["FlowSession", "STATEFUL_KINDS"]

#: Steps that leave something in ``ctx.state`` for a later step to consume, or
#: that consume it. A boundary *inside* this run of steps cannot be resumed
#: from, because the latent image lives in the context rather than the stack.
STATEFUL_KINDS = frozenset({"expose", "peb", "develop"})

#: Steps whose whole effect is on the latent image in ``ctx.state``, never on
#: the wafer. They *always* leave ``mat`` untouched, so the no-change marker
#: must not read them as idle: an exposure that changed the stack would be the
#: surprising one. ``develop`` is absent deliberately — it writes the resist
#: profile into the stack, so a develop that changes nothing is worth flagging.
LATENT_ONLY_KINDS = frozenset({"expose", "peb"})

#: How :func:`~litho_sim.viz.viz3d.crop` marks the lines it adds. Cropping is a
#: viewing decision, so its note is not part of what the wafer went through.
DISPLAY_TAG = "(display)"


def _process_history(stack: Stack) -> list[str]:
    """A stack's history with the display bookkeeping taken out."""
    return [h for h in stack.history if DISPLAY_TAG not in h]


class FlowSession:
    """A process flow you can edit between runs.

    Holds the steps, the stack they build, and one snapshot per completed step.
    Everything is recomputed lazily: :meth:`run_to` brings the session up to a
    given step, doing only the work that is actually missing.

    Parameters
    ----------
    grid, optics, resist : config objects
        Defaults handed to every step through the :class:`ProcessContext`.
    layouts : dict, optional
        ``{name: Layout}`` that ``Expose`` and ``Etch`` steps refer to by name.
    base : Stack, optional
        Starting wafer. Defaults to a bare substrate at ``grid``'s dimensions.
    """

    def __init__(
        self,
        grid: GridConfig | None = None,
        optics: OpticsConfig | None = None,
        resist: ResistConfig | None = None,
        layouts: dict[str, object] | None = None,
        base: Stack | None = None,
    ) -> None:
        self.grid = grid or GridConfig()
        self.optics = optics or OpticsConfig()
        self.resist = resist or ResistConfig()
        self.layouts = dict(layouts or {})
        self.steps: list[ProcessStep] = []

        self._base = base.copy() if base is not None else Stack.blank(
            self.grid, dz=self.grid.dz
        )
        #: ``_snaps[i]`` is the stack **after** step ``i``. Shorter than
        #: ``steps`` whenever the tail has been invalidated.
        self._snaps: list[Stack] = []
        #: Counts genuine step applications, for tests and telemetry.
        self.applied = 0
        #: Parallel to ``_snaps``: did step *i* actually change the wafer?
        #: A step is allowed to do nothing — etching a material that is not
        #: exposed is a no-op, not an error — but it must not do nothing
        #: *silently*, which is how a working recipe comes to look broken.
        self._changed: list[bool] = []
        #: Steps below this index cannot be edited, because this session has no
        #: context to replay them with. See :meth:`adopt`.
        self.locked_upto = 0
        #: Steps below this index arrived already run, from a device preset.
        #: Provenance, not permission — it stays true after you edit one, and
        #: it is what the list mutes to separate "what the device did" from
        #: "what you added".
        self.as_built_upto = 0

    # -- the step list -------------------------------------------------
    def _check_editable(self, *indices: int) -> None:
        """Refuse to touch a step this session could not replay.

        Not because the step is old, but because there is nothing to re-run it
        *with*: a flow adopted without its layouts and optics cannot honestly
        reproduce an exposure. Editing one would leave the list claiming
        something the wafer underneath it does not say.
        """
        if any(i < self.locked_upto for i in indices):
            raise ValueError(
                f"steps 1-{self.locked_upto} arrived without the layouts and "
                f"optics they ran with, so this session cannot replay them. "
                f"Add steps after them instead."
            )

    def append(self, step: ProcessStep) -> None:
        self.steps.append(step)

    def insert(self, index: int, step: ProcessStep) -> None:
        self._check_editable(index)
        self.steps.insert(index, step)
        self.invalidate_from(index)

    def remove(self, index: int) -> ProcessStep:
        self._check_editable(index)
        step = self.steps.pop(index)
        self.invalidate_from(index)
        return step

    def replace(self, index: int, step: ProcessStep) -> None:
        self._check_editable(index)
        self.steps[index] = step
        self.invalidate_from(index)

    def move(self, index: int, to: int) -> None:
        self._check_editable(index, to)
        step = self.steps.pop(index)
        self.steps.insert(to, step)
        self.invalidate_from(min(index, to))

    def __len__(self) -> int:
        return len(self.steps)

    def describe(self) -> list[str]:
        """Numbered step lines, ready for a list widget."""
        return [f"{i + 1:2d}. {s.describe()}" for i, s in enumerate(self.steps)]

    # -- validity ------------------------------------------------------
    def invalidate_from(self, index: int) -> None:
        """Discard snapshots from *index* onward; they are no longer reachable."""
        boundary = self._safe_boundary(index)
        if boundary < len(self._snaps):
            dropped = len(self._snaps) - boundary
            del self._snaps[boundary:]
            del self._changed[boundary:]
            logger.debug("invalidated %d snapshot(s) from step %d", dropped, boundary)

    def _safe_boundary(self, index: int) -> int:
        """The earliest step at or before *index* that can be resumed from.

        Walks back over any litho triplet the index lands inside. Resuming at
        ``develop`` with no latent raises; resuming at ``peb`` would bake
        whatever the previous exposure happened to leave behind. Neither is a
        thing to discover at runtime, so the boundary moves instead.
        """
        i = min(index, len(self.steps))
        while i > 0 and self.steps[i - 1].kind in STATEFUL_KINDS:
            i -= 1
        return i

    @property
    def valid_upto(self) -> int:
        """How many steps have a live snapshot."""
        return len(self._snaps)

    def did_nothing(self) -> list[int]:
        """Indices of completed steps that left the wafer exactly as it was.

        Usually a mistake rather than an intent — an etch naming a material
        that is not exposed, a CMP above the current surface, a strip of
        something already gone. Each is legal, and each looks identical to a
        broken application until someone says so.

        :data:`LATENT_ONLY_KINDS` are exempt: an ``expose`` leaves the wafer
        alone every single time, so flagging it is not a warning, it is noise
        on a correct recipe — and a marker that fires on healthy flows is one
        nobody reads on a broken one.
        """
        return [
            i for i, changed in enumerate(self._changed)
            if not changed
            and not (i < len(self.steps)
                     and self.steps[i].kind in LATENT_ONLY_KINDS)
        ]

    def notes_for(self, index: int) -> list[str]:
        """The history lines step *index* contributed, and nothing else.

        Steps log to ``Stack.history``, snapshots carry it, and it only ever
        grows — so the lines one step added are the delta between its snapshot
        and its predecessor's. :meth:`did_nothing` says *that* a step was idle;
        this says *why*, in the sentence the engine wrote at the moment it
        decided. Two channels because they answer different questions and are
        good at different things: one is structured and drives a marker, the
        other is prose and is displayed verbatim, so the engine can reword its
        explanations without the GUI having to parse anything.
        """
        if not (0 <= index < len(self._snaps)):
            return []
        previous = self._base if index == 0 else self._snaps[index - 1]
        # Display crops are not process steps and must not be counted as
        # history, or the delta stops being a suffix and every note shifts by
        # one — reporting "crop z[12:82] … (display)" where the etch should be.
        # `viz3d.process_flow_panel` drops the same lines for the same reason.
        mine = _process_history(self._snaps[index])
        theirs = _process_history(previous)
        return mine[len(theirs):]

    def snapshots(self) -> list[Stack]:
        """One stack per completed step. ``snapshots()[i]`` is after step ``i``.

        The list is fresh; the stacks in it are not. Copying per call would
        cost a few MB every time a scrubber moved, and every caller so far only
        reads them — so they are handed out by reference, and a reader that
        mutates one corrupts the session.
        """
        return list(self._snaps)

    def stack_at(self, index: int) -> Stack:
        """The stack as of step *index*, where ``-1`` (or 0) is the bare wafer.

        Indices run over *steps*, so the caller's scrubber wants
        ``range(-1, len(session))`` — the extra slot at the bottom being the
        wafer as loaded, which ``snapshots()`` has no entry for.
        """
        if index < 0:
            return self._base
        return self._snaps[index]

    # -- running -------------------------------------------------------
    def run_to(
        self,
        index: int | None = None,
        on_step: Callable[[int, ProcessStep, Stack], None] | None = None,
    ) -> Stack:
        """Bring the session up to (and including) step *index*.

        Replays only what is missing. With every snapshot already live this is
        a no-op that returns the cached stack.

        Raises
        ------
        RuntimeError
            If a step fails, naming which one — matching ``Flow.run``'s
            message so the two are diagnosable the same way.
        """
        last = len(self.steps) - 1 if index is None else index
        if last < 0:
            return self._base
        if last >= len(self.steps):
            raise IndexError(f"step {last} of {len(self.steps)}")

        start = self._safe_boundary(min(self.valid_upto, last + 1))
        del self._snaps[start:]
        del self._changed[start:]

        stack = (self._base if start == 0 else self._snaps[start - 1]).copy()
        ctx = ProcessContext(
            grid=self.grid, optics=self.optics, resist=self.resist,
            layouts=dict(self.layouts),
        )

        for i in range(start, last + 1):
            step = self.steps[i]
            try:
                stack = step.apply(stack, ctx)
            except Exception as exc:                       # noqa: BLE001
                # Leave the session consistent: whatever completed stays
                # valid, so the user can fix the offending step and re-run
                # without losing the work in front of it.
                raise RuntimeError(
                    f"Step {i + 1} ({step.describe()}) failed: {exc}"
                ) from exc
            self.applied += 1
            previous = self._snaps[-1] if self._snaps else self._base
            self._changed.append(not np.array_equal(previous.mat, stack.mat))
            self._snaps.append(stack.copy())
            if on_step is not None:
                on_step(i, step, stack)

        self.measurements = list(ctx.measurements)
        return stack

    def adopt(
        self,
        steps: list[ProcessStep],
        snapshots: list[Stack],
        base: Stack,
        changed: list[bool] | None = None,
        layouts: dict[str, object] | None = None,
        optics: OpticsConfig | None = None,
        resist: ResistConfig | None = None,
    ) -> None:
        """Take on a flow that has **already run**, results and all.

        :meth:`run_to` earns its snapshots by applying steps. This is the other
        way in: a device preset arrives with the recipe it was built from and
        one wafer per step, so the session can show the whole flow — and scrub
        through it — without recomputing a thing.

        Whether those steps can then be **edited** is not a policy choice, it
        follows from what came with them. A step names its layout, and looks it
        up in the context along with the optics and the resist; hand over that
        context and the flow is a flow like any other, editable and replayable.
        Hand over only the steps and it is a *record*: replaying an ``Expose``
        would raise on the missing layout, or — worse, if some other layout
        happened to share the name — print the GAA at the library's default
        193 nm on a flow that is EUV throughout. So a context-less adoption
        sets :attr:`locked_upto` and refuses edits, and one with a context does
        not.

        :attr:`as_built_upto` is set either way. It is provenance, not
        permission: these steps came from a device rather than from you, which
        stays true after you have edited one.

        Raises
        ------
        ValueError
            If the counts disagree, or if the flow ends inside a litho triplet.
            A trailing ``expose`` or ``peb`` leaves a latent image in the
            context, which snapshots do not carry, so the next step would
            resume from a boundary :meth:`_safe_boundary` has to walk back past
            — into the adopted block. Ending on a completed step keeps that
            boundary reachable.
        """
        if len(steps) != len(snapshots):
            raise ValueError(
                f"{len(steps)} steps but {len(snapshots)} snapshots — one "
                f"snapshot per step, taken after it ran."
            )
        if changed is not None and len(changed) != len(steps):
            raise ValueError(
                f"{len(steps)} steps but {len(changed)} change flags."
            )
        if steps and steps[-1].kind in STATEFUL_KINDS:
            raise ValueError(
                f"flow ends on '{steps[-1].kind}', mid-litho: the latent image "
                f"lives in the process context, which an adopted flow does not "
                f"carry. Adopt a flow that ends on a completed step."
            )

        self.steps = list(steps)
        self._snaps = list(snapshots)
        self._changed = (
            list(changed) if changed is not None else [True] * len(steps)
        )
        self._base = base
        self.as_built_upto = len(steps)

        replayable = layouts is not None or optics is not None
        if layouts is not None:
            self.layouts = dict(layouts)
        if optics is not None:
            self.optics = optics
        if resist is not None:
            self.resist = resist
        self.locked_upto = 0 if replayable else len(steps)

        logger.info("adopted %d as-built step(s), %s", len(steps),
                    "editable" if replayable else "read-only (no context)")

    def reset(self, base: Stack | None = None) -> None:
        """Throw away every result, optionally starting from a new wafer.

        Also voids any as-built claim: without its snapshots an adopted flow
        is no longer a record of what a wafer did, so it stops being locked.
        """
        if base is not None:
            self._base = base.copy()
        self._snaps.clear()
        self._changed.clear()
        self.locked_upto = 0
        self.as_built_upto = 0
