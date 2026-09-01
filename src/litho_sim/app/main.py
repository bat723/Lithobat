"""
The Qt front end — the only module that needs a Qt binding.

Everything that decides *what* to simulate and *when* lives in ``params``,
``compute`` and ``scheduler``, none of which need a display. What is left here
is genuinely just presentation: build controls from the specs, run the work on
a thread, draw the result.

Two rules that keep it responsive and correct:

* **The engine never runs on the GUI thread.** A :class:`Worker` lives on a
  ``QThread`` and communicates only by signal. NumPy's FFT releases the GIL,
  so this is real concurrency, not just tidiness — the interface keeps
  repainting while an expensive vector image is being computed.
* **Figures are built with ``Figure()``, never ``pyplot``.** Pyplot keeps a
  global registry of figures that is not thread-safe and would leak one entry
  per redraw. ``FigureCanvasQTAgg`` over a bare ``Figure`` touches none of it.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

from litho_sim.app.compute import (
    ImagingResult,
    estimate_cost_ms,
)
from litho_sim.app.controls import ControlPanel
from litho_sim.app.params import (
    DEVELOP3D_ONLY,
    MODE_3D_GROUPS,
    SPECS_BY_KEY,
    ParameterModel,
)
from litho_sim.app.process_window_tab import ProcessWindowTab
from litho_sim.app.qt import QtCore, QtGui, QtWidgets
from litho_sim.app.scheduler import SETTLE_MS, Request, Scheduler
from litho_sim.app.stack_tab import StackTab
from litho_sim.app.stochastics_tab import StochasticsTab
from litho_sim.app.views import (
    DRAFT_STRIDE,
    DevelopView,
    ExposeView,
    MaskView,
    Profile3DView,
)
from litho_sim.app.worker import Worker

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    request = QtCore.Signal(object)
    request3d = QtCore.Signal(object)
    request_flow = QtCore.Signal(object)
    request_device = QtCore.Signal(str)
    request_fem = QtCore.Signal(object)
    request_stoch = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LithoPy")
        self.resize(1280, 820)

        self.model = ParameterModel()
        self.scheduler = Scheduler()

        # Tabs run left to right in the order the physics does: the wafer
        # stack is the substrate everything else acts on, then one tab per
        # pipeline stage, mirroring the package's own coat -> expose -> bake
        # -> develop taxonomy. Note the tabs are for organisation, not
        # economy: the pipeline is sequential and the Abbe sum is ~98% of it,
        # so showing only the Develop tab still computes everything upstream.
        # The saving comes from the staged cache.
        self.tabs = QtWidgets.QTabWidget()
        self.mask_view = MaskView()
        self.expose_view = ExposeView()
        self.develop_view = DevelopView()
        self.profile_view = Profile3DView()

        # The resist profile is always present; 2-D or 3-D is a mode, not a
        # separate tab.
        self.develop_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(self.develop_tab)
        dv.setContentsMargins(4, 4, 4, 4)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Profile:"))
        self.mode_2d = QtWidgets.QRadioButton("2-D")
        self.mode_3d = QtWidgets.QRadioButton("3-D")
        self.mode_2d.setChecked(True)
        mode_row.addWidget(self.mode_2d)
        mode_row.addWidget(self.mode_3d)
        mode_row.addStretch(1)
        dv.addLayout(mode_row)
        self.develop_stack = QtWidgets.QStackedWidget()
        self.develop_stack.addWidget(self.develop_view)
        self.develop_stack.addWidget(self.profile_view)
        dv.addWidget(self.develop_stack)

        self.stack_tab = StackTab(self.model)
        self.pw_tab = ProcessWindowTab(self.model)
        self.stoch_tab = StochasticsTab(self.model)
        self.tabs.addTab(self.stack_tab, "Wafer Stack")
        self.tabs.addTab(self.mask_view, "Mask")
        self.tabs.addTab(self.expose_view, "Expose")
        self.tabs.addTab(self.develop_tab, "Develop")
        self.tabs.addTab(self.pw_tab, "Process Window")
        self.tabs.addTab(self.stoch_tab, "Stochastics")
        self.tabs.setCurrentWidget(self.stack_tab)
        self.setCentralWidget(self.tabs)
        self._last_result: ImagingResult | None = None
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.controls = ControlPanel(self.model)
        # The default is 193 nm, where 'multilayer' cannot run — so the
        # panel has to reflect that before the user touches anything.
        self.controls.refresh_mask_models()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        dock = QtWidgets.QDockWidget("Parameters", self)
        dock.setWidget(scroll)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        self.status = self.statusBar()
        self.status.showMessage("starting…")

        # -- worker thread ---------------------------------------------
        self.thread = QtCore.QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.request.connect(self.worker.run)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.thread.start()

        # -- scheduling ------------------------------------------------
        # One timer decides when a deferred request is allowed to run. Cheap
        # changes go immediately; expensive ones wait for the drag to stop.
        self._settle = QtCore.QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(int(SETTLE_MS))
        self._settle.timeout.connect(lambda: self._pump(settled=True))

        # The 3-D mesh is the one thing here that cannot be drawn at full
        # detail inside a drag: 129 ms of paint at stride 2 against 67 at
        # stride 4. So a live refresh draws coarse and this timer redraws it
        # properly once the control stops moving — the same cheap-now,
        # accurate-when-settled bargain the scheduler already makes for the
        # 2-D path.
        self._draft_3d = False
        self._settle3d = QtCore.QTimer(self)
        self._settle3d.setSingleShot(True)
        self._settle3d.setInterval(int(SETTLE_MS))
        self._settle3d.timeout.connect(self._refine_3d)

        self.controls.changed.connect(self._on_changed)
        self.mode_2d.toggled.connect(self._on_mode_changed)
        self.controls.compute_3d.clicked.connect(self._request_3d)
        self.request3d.connect(self.worker.run_3d)
        self.request_flow.connect(self.worker.run_flow)
        self.worker.done_flow.connect(self._on_flow_done)
        self.stack_tab.run_requested.connect(self._request_flow)
        self.stack_tab.import_requested.connect(self._import_profile)
        self.worker.done3d.connect(self._on_done_3d)
        self.worker.done_device.connect(self._on_device_ready)
        self.request_device.connect(self.worker.run_device)
        # Signal-to-signal: the tab already built the FemRequest (with its
        # own deep copy of the model), so the window just relays it onto the
        # worker thread.
        self.pw_tab.run_requested.connect(self.request_fem)
        self.request_fem.connect(self.worker.run_fem)
        self.worker.done_fem.connect(self.pw_tab.on_finished)
        self.worker.progress_fem.connect(self.pw_tab.on_progress)
        # Same relay pattern for the Monte-Carlo batch.
        self.stoch_tab.run_requested.connect(self.request_stoch)
        self.request_stoch.connect(self.worker.run_stoch)
        self.worker.done_stoch.connect(self.stoch_tab.on_finished)
        self.worker.progress_stoch.connect(self.stoch_tab.on_progress)
        self._build_menu()

        self._profile: object | None = None
        # Live 3-D refresh latch. A develop-only change costs ~5 ms of physics
        # now that the latent is cached, so it need not wait for the button —
        # but it still costs a mesh draw, so at most one refresh is queued and
        # it always ends on the newest parameters.
        self._busy_3d = False
        self._pending_3d = False
        self._on_mode_changed()          # hide the Profile group in 2-D
        self.profile_view.show_placeholder(
            "Switch to 3-D and press Compute 3-D."
        )

        # Queue *and* release the first image. Submitting alone only fills the
        # scheduler; without a pump the window opens empty and stays that way
        # until the user happens to move a control.
        self._submit()
        self._pump(settled=True)

    # -- plumbing ------------------------------------------------------
    def _submit(self) -> None:
        self.scheduler.submit(
            self.model.signature(),
            payload=copy.deepcopy(self.model),
            cost_ms=estimate_cost_ms(self.model),
        )

    def _on_changed(self, key: str, _value) -> None:
        spec = SPECS_BY_KEY[key]

        if spec.stage == "view":
            # z-exaggeration and render detail change the picture, not the
            # physics. Redraw from the volume already in hand.
            if self._profile is not None and self._is_3d:
                self._draw_profile()
            return

        if key in DEVELOP3D_ONLY:
            # These change only the final develop step, which the latent cache
            # has reduced to ~5 ms out of ~730 — cheap enough to be live, so
            # dragging Threshold moves the 3-D solid the way it moves the 2-D
            # cross-section. The rest of the Profile group still costs a full
            # set of Abbe sums and stays behind the button.
            self._refresh_3d()

        if spec.stage == "profile3d":
            # 3-D-only settings do not touch the 2-D image, and the 3-D run is
            # explicit — so nothing recomputes until Compute 3-D is pressed.
            return

        self._submit()
        self._settle.start()
        self._pump(settled=False)

    def _pump(self, settled: bool) -> None:
        req = self.scheduler.take(settled=settled)
        if req is None:
            return
        self.status.showMessage("computing…")
        self.request.emit(req.payload)

    @QtCore.Slot(object)
    def _on_done(self, result: ImagingResult) -> None:
        self._last_result = result
        self._draw(result)
        self.status.showMessage(result.summary)
        # The result carries the signature it was computed from, which is all
        # the scheduler compares — so the running request can be closed out
        # without holding a reference to it.
        self.scheduler.finish(Request(result.signature, None, 0.0))
        self._pump(settled=not self._settle.isActive())

    def _visible_2d_view(self):
        """The 2-D view the user is actually looking at, or None.

        The Develop tab is a container — a mode radio over a stacked widget —
        so the tab itself has no show_result. Returning None while the 3-D
        mode is showing is what stops a 2-D result being painted over a 3-D
        picture.

        Keyed on widget *identity*, not tab index. Indices were fine while
        there were exactly three tabs and the last case could be an else, but
        that shape has a trapdoor: any fourth tab falls into the Develop branch
        and gets a 2-D imaging result painted into a canvas that is not even
        showing. Identity has no default case to get wrong.
        """
        current = self.tabs.currentWidget()
        if current is self.mask_view:
            return self.mask_view
        if current is self.expose_view:
            return self.expose_view
        if current is self.develop_tab:
            return None if self._is_3d else self.develop_view
        return None

    def _draw(self, result: ImagingResult) -> None:
        """Draw into the visible view only.

        Hidden tabs keep stale artists until they are shown again, which costs
        nothing and saves a canvas draw per update — the one place tab
        splitting genuinely does buy something on the 2-D side.
        """
        view = self._visible_2d_view()
        if view is not None:
            view.show_result(result)

    def _on_tab_changed(self, _index: int) -> None:
        """Bring a newly visible tab up to date with the latest result."""
        if self._last_result is not None:
            self._draw(self._last_result)

    # -- 3-D ------------------------------------------------------------
    @property
    def _is_3d(self) -> bool:
        return self.mode_3d.isChecked()

    def _on_mode_changed(self, _checked: bool = False) -> None:
        """Swap the Develop panel, and show the 3-D-only controls with it."""
        self.develop_stack.setCurrentIndex(1 if self._is_3d else 0)
        for group in MODE_3D_GROUPS:
            self.controls.set_group_visible(group, self._is_3d)
        # The readout belongs to whichever mode is showing. Leaving the 3-D
        # numbers up over a 2-D picture is the same lie as a stale plot.
        if self._is_3d:
            if self._profile is not None:
                self.status.showMessage(self._profile.summary)
        else:
            if self._last_result is not None:
                self.develop_view.show_result(self._last_result)
                self.status.showMessage(self._last_result.summary)

    def _request_3d(self) -> None:
        self.status.showMessage("developing in 3-D…")
        self._emit_3d()

    def _refresh_3d(self) -> None:
        """Re-develop the existing profile, if there is one to re-develop."""
        if self._profile is None or not self._is_3d:
            return
        self._draft_3d = True
        self._settle3d.start()
        self._emit_3d()

    def _refine_3d(self) -> None:
        """The drag has stopped — redraw the parked profile at full detail."""
        self._draft_3d = False
        if self._profile is not None and self._is_3d and not self._busy_3d:
            self._draw_profile()

    def _draw_profile(self, draft: bool = False) -> None:
        """Render the profile already in hand. No physics happens here."""
        self.profile_view.show_profile(
            self._profile,
            self.model.grid(),
            z_exaggeration=float(self.model["z_exaggeration"]),
            downsample=max(int(self.model["downsample"]), DRAFT_STRIDE) if draft
                       else int(self.model["downsample"]),
            render_mode=str(self.model["render_mode"]),
        )

    def _emit_3d(self) -> None:
        if self._busy_3d:
            # Coalesce: one queued refresh, fired with whatever the parameters
            # are by the time the running one lands.
            self._pending_3d = True
            return
        self._busy_3d = True
        self.controls.compute_3d.setEnabled(False)
        self.request3d.emit(copy.deepcopy(self.model))

    @QtCore.Slot(object)
    def _on_done_3d(self, profile) -> None:
        self._profile = profile
        self._busy_3d = False
        self.controls.compute_3d.setEnabled(True)
        # There is now something to adopt, so the button stops being greyed out.
        self.stack_tab.import_btn.setEnabled(True)
        # Drawing the mesh costs more than computing it, so say so rather
        # than appearing to hang.
        self.status.showMessage(profile.summary + "  — rendering…")
        self._draw_profile(draft=self._draft_3d)
        self.status.showMessage(profile.summary)

        if self._pending_3d:
            self._pending_3d = False
            self._emit_3d()

    # -- File menu -----------------------------------------------------
    def _build_menu(self) -> None:
        """File ▸ open, save, and the named device presets.

        The Wafer Stack tab's own button adopts the *in-memory* develop result
        and nothing else; anything arriving from disk comes through here. The
        two were conflated once — a button called "Import resist profile" that
        opened no dialog — and the menu is the half that was missing.
        """
        from litho_sim.tech.devices import DEVICES

        # Held as attributes, not locals. A QMenu returned by addMenu is owned
        # by its parent in C++, but keeping the Python handles removes any
        # question about lifetime and gives tests something to find.
        self.file_menu = file_menu = self.menuBar().addMenu("&File")

        act_open = file_menu.addAction("&Open wafer…")
        act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_wafer)

        act_save = file_menu.addAction("&Save wafer as…")
        act_save.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        act_save.triggered.connect(self._save_wafer)

        file_menu.addSeparator()
        self.devices_menu = devices = file_menu.addMenu("&Load device")
        for key, spec in DEVICES.items():
            act = devices.addAction(spec.title)
            act.setStatusTip(spec.blurb)
            act.triggered.connect(lambda _=False, k=key: self._load_device(k))
        # The tab's own button drops the same menu, anchored under itself.
        btn = self.stack_tab.device_btn
        btn.setMenu(devices)
        devices.addSeparator()
        self.act_section = devices.addAction("Section when loading")
        self.act_section.setCheckable(True)
        self.act_section.setChecked(True)
        self.act_section.setStatusTip(
            "Cut the device open at the channel. Uncut it is a solid block — "
            "every interesting feature is interior by the last step."
        )

        file_menu.addSeparator()
        act_quit = file_menu.addAction("&Quit")
        act_quit.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)

    def _load_device(self, name: str) -> None:
        from litho_sim.tech.devices import DEVICES, cache_path

        cached = cache_path(name).is_file()
        self.status.showMessage(
            f"loading {DEVICES[name].title}…" if cached
            else f"building {DEVICES[name].title} — first time only, "
                 f"then it is cached…"
        )
        self.request_device.emit(name)

    def _on_device_ready(self, payload) -> None:
        from litho_sim.tech.devices import sectioned

        name, stack, label, flow = payload
        # Handed over as a function, not applied to the wafers: the 3-D view
        # needs the device cut open, and the cross-section needs it whole.
        section = (
            (lambda s, _n=name: sectioned(_n, s))
            if self.act_section.isChecked() else None
        )
        self.stack_tab.load_stack(stack, label=label, downsample=1, flow=flow,
                                  section=section)
        self.tabs.setCurrentWidget(self.stack_tab)
        steps = f"{len(flow)} steps  —  " if flow is not None else ""
        self.status.showMessage(
            f"{label}  —  {steps}drag inside the view to rotate"
        )

    def _open_wafer(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open wafer", "", "Wafer stack (*.npz);;All files (*)")
        if not path:
            return
        try:
            from litho_sim.wafer import Stack

            stack = Stack.load(path)
        except Exception as exc:                      # noqa: BLE001
            logger.exception("wafer load failed")
            QtWidgets.QMessageBox.warning(
                self, "Could not open wafer", f"{Path(path).name}:\n{exc}")
            return
        self.stack_tab.load_stack(stack, label=Path(path).name, downsample=1)
        self.tabs.setCurrentWidget(self.stack_tab)
        self.status.showMessage(f"opened {Path(path).name}")

    def _save_wafer(self) -> None:
        """Save whatever the scrubber is showing — what you see is what you get."""
        session = self.stack_tab.session
        index = self.stack_tab.scrubber.value() - 1
        try:
            stack = session.stack_at(index)
        except Exception:                             # noqa: BLE001
            stack = None
        if stack is None or not stack.mat.any():
            self.status.showMessage("nothing to save — the wafer is empty")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save wafer as", "wafer.npz", "Wafer stack (*.npz)")
        if not path:
            return
        try:
            stack.save(path)
        except Exception as exc:                      # noqa: BLE001
            logger.exception("wafer save failed")
            QtWidgets.QMessageBox.warning(
                self, "Could not save wafer", str(exc))
            return
        self.status.showMessage(f"saved {Path(path).name}")

    def _import_profile(self) -> None:
        if self._profile is None:
            self.status.showMessage(
                "No developed profile yet — switch Develop to 3-D and press "
                "Compute 3-D first."
            )
            return
        try:
            self.stack_tab.import_profile(self._profile.remaining, self.model.grid())
        except Exception as exc:                      # noqa: BLE001
            logger.exception("profile import failed")
            self.status.showMessage(f"import failed: {exc}")
            return
        self.status.showMessage("imported the developed profile as the starting wafer")
        self.tabs.setCurrentWidget(self.stack_tab)

    def _request_flow(self, session) -> None:
        self.status.showMessage(f"running {len(session)} step(s)…")
        self.request_flow.emit(session)

    @QtCore.Slot(object)
    def _on_flow_done(self, error) -> None:
        self.stack_tab.on_finished(error)
        # A courtesy only. The authoritative readout is the tab's own label:
        # this status bar belongs to the imaging pipeline too, which overwrites
        # it on every recompute, and auto-run makes that collision routine.
        if error is None:
            n = self.stack_tab.session.valid_upto
            self.status.showMessage(f"flow complete — {n} step(s)")
        else:
            self.status.showMessage(f"flow error: {error}")

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        self._busy_3d = False
        self._pending_3d = False
        self.controls.compute_3d.setEnabled(True)
        self.scheduler.abandon()
        self.status.showMessage(f"error: {message}")
        # A failure must not wedge the queue: whatever the user did next
        # still deserves to run.
        self._pump(settled=True)

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        self.thread.quit()
        self.thread.wait(2000)
        super().closeEvent(event)


def main(stack=None, label: str = "printed device",
         downsample: int = 1, flow=None, section=None) -> int:
    """Open the app.

    Parameters
    ----------
    stack : Stack, optional
        A wafer to adopt on startup — the app opens on the Wafer Stack tab in
        3-D with this already drawn, instead of an empty recipe. See
        ``scripts/show_device.py`` for the device flows that produce one.
    label : str
        Shown against the scrubber's loaded position.
    downsample : int
        Mesh stride for the preloaded wafer.
    flow : DeviceFlow, optional
        The recipe that built *stack*, so the Wafer Stack panel lists its steps
        and the scrubber can walk them.
    section : callable, optional
        ``Stack -> Stack``, cutting the device open for the 3-D view.
    """
    logging.getLogger("litho_sim").setLevel(logging.WARNING)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    if stack is not None:
        window.stack_tab.load_stack(stack, label=label, downsample=downsample,
                                    flow=flow, section=section)
        window.tabs.setCurrentWidget(window.stack_tab)
    window.show()
    # PyQt5 spells it exec_(); Qt6 bindings and PyQt5>=5.15 have exec().
    return (getattr(app, "exec", None) or app.exec_)()


if __name__ == "__main__":
    raise SystemExit(main())
