"""The engine's thread-side: runs every computation off the GUI thread."""

from __future__ import annotations

import logging

from litho_sim.app.compute import (
    compute_imaging,
    compute_profile_3d,
)
from litho_sim.app.fem import FemRequest, ProcessWindowResult, compute_fem
from litho_sim.app.params import (
    ParameterModel,
)
from litho_sim.app.pipeline import Pipeline
from litho_sim.app.qt import QtCore

logger = logging.getLogger(__name__)


class Worker(QtCore.QObject):
    """Runs :func:`compute_imaging` off the GUI thread.

    Owns its own :class:`Pipeline`, so the cache lives entirely on the worker
    thread and is never touched from the GUI — no lock, no shared mutable
    state, and the staged reuse still applies because every request goes
    through this one object.
    """

    done = QtCore.Signal(object)
    done3d = QtCore.Signal(object)
    done_flow = QtCore.Signal(object)   # None, or an error string
    done_device = QtCore.Signal(object)  # (name, stack, label)
    done_fem = QtCore.Signal(object)     # ProcessWindowResult, or None on failure
    progress_fem = QtCore.Signal(int, int)
    done_stoch = QtCore.Signal(object)   # StochResult, or None on failure
    progress_stoch = QtCore.Signal(int, int)
    done_opc = QtCore.Signal(object)     # OPCResult
    done_dose = QtCore.Signal(float)     # dose-to-size, NaN when nothing sizes
    failed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = Pipeline()

    @QtCore.Slot(object)
    def run(self, params: ParameterModel) -> None:
        try:
            self.done.emit(compute_imaging(params, self.pipeline))
        except Exception as exc:                      # noqa: BLE001
            # A bad parameter combination must not take the app down; the
            # message goes to the status bar and the user carries on.
            logger.exception("compute failed")
            self.failed.emit(str(exc))

    @QtCore.Slot(object)
    def run_opc(self, params: ParameterModel) -> None:
        """The correction on its own: the mask the next Print would image.

        Through the pipeline, so a Print that follows with the switch on
        finds it already done — and a Print that ran first has left it
        here for this.
        """
        try:
            self.done_opc.emit(self.pipeline.opc(params))
        except Exception as exc:                      # noqa: BLE001
            logger.exception("OPC failed")
            self.failed.emit(str(exc))

    @QtCore.Slot(object)
    def run_dose_to_size(self, params: ParameterModel) -> None:
        """The dose that prints the dense array to size — one image and a
        bisection of rescales, so tens of milliseconds. NaN when no dose does."""
        from litho_sim.app.opc import dose_to_size

        try:
            self.done_dose.emit(dose_to_size(params))
        except Exception as exc:                      # noqa: BLE001
            logger.exception("dose-to-size failed")
            self.failed.emit(str(exc))

    @QtCore.Slot(object)
    def run_flow(self, session) -> None:
        """Advance a :class:`FlowSession` to the end of its recipe.

        The session is handed over by reference rather than deep-copied, unlike
        the imaging requests: it owns megabytes of snapshots, and copying it
        per run would defeat the incremental replay it exists to provide. That
        is safe only because the GUI does not touch it while a run is in
        flight — the Run button is disabled for exactly that window.
        """
        try:
            session.run_to()
            self.done_flow.emit(None)
        except Exception as exc:                      # noqa: BLE001
            logger.exception("flow failed")
            self.done_flow.emit(str(exc))

    @QtCore.Slot(str)
    def run_device(self, name: str) -> None:
        """Build (or load from cache) a named device preset.

        Off the GUI thread because a cold build is 2–7 s of Abbe sums and etch
        passes — long enough that doing it inline would look like a hang, which
        is the failure mode this whole class exists to avoid.
        """
        try:
            from litho_sim.tech.devices import build, flow_for

            stack, label = build(name)
            # After `build`, which writes the flow cache the first time round.
            # `None` if this preset predates recipes being recorded — the
            # device still loads, the recipe panel is what goes missing.
            self.done_device.emit((name, stack, label, flow_for(name)))
        except Exception as exc:                      # noqa: BLE001
            logger.exception("device build failed")
            self.failed.emit(f"could not load {name}: {exc}")

    @QtCore.Slot(object)
    def run_3d(self, params: ParameterModel) -> None:
        """The expensive branch, entered only when the user asks for it."""
        try:
            self.done3d.emit(compute_profile_3d(params, self.pipeline))
        except Exception as exc:                      # noqa: BLE001
            logger.exception("3-D compute failed")
            self.failed.emit(str(exc))

    @QtCore.Slot(object)
    def run_fem(self, req: FemRequest) -> None:
        """A focus-exposure matrix — seconds of Abbe sums, one per focus step.

        Deliberately not through ``self.pipeline``: the sweep reuses its raw
        sums internally, and pushing a dozen foreign defocus signatures
        through the shared cache would evict the ``aerial`` entry the live
        view's next slider drag depends on.
        """
        result: ProcessWindowResult | None = None
        try:
            result = compute_fem(
                req, progress=lambda i, n: self.progress_fem.emit(i, n)
            )
        except Exception as exc:                      # noqa: BLE001
            logger.exception("FEM sweep failed")
            self.failed.emit(str(exc))
        # Always emitted, even on failure — the tab's controls are disabled
        # for exactly the window this signal closes.
        self.done_fem.emit(result)

    @QtCore.Slot(object)
    def run_stoch(self, req) -> None:
        """A Monte-Carlo printing batch — one car develop per trial.

        Not through ``self.pipeline`` for the FEM's reason: the batch reuses
        nothing the live view caches, and would evict what it does cache.
        """
        from litho_sim.app.stochastics import StochResult, compute_stochastics

        result: StochResult | None = None
        try:
            result = compute_stochastics(
                req, progress=lambda i, n: self.progress_stoch.emit(i, n)
            )
        except Exception as exc:                      # noqa: BLE001
            logger.exception("stochastic batch failed")
            self.failed.emit(str(exc))
        # Always emitted, even on failure — same contract as done_fem.
        self.done_stoch.emit(result)


