"""
The Qt front end — the only module that needs a Qt binding.

Everything that decides *what* to simulate lives in ``params``, ``compute``
and ``pipeline``, none of which need a display. What is left here is
genuinely just presentation: build the tabs, run the work on a thread, draw
the result.

The shape of the window
-----------------------
One tab per processing step, left to right in the order the physics runs —
**Mask, Source, Resist, Expose, Bake, Develop** — then **Wafer Stack** for
what happens to the printed resist, **Simulate** to run things, and **SEM** to
look at what was run the way a fab would. Each step tab holds that step's
controls beside its picture; there is no separate parameter panel.

Nothing physical computes while a control moves. Exposure and development
run when the Simulate tab is asked, on a snapshot of the settings taken at
that moment, and the step tabs show the result — with a banner the moment a
setting moves on from what the picture was made with. Two pictures are
exempt because they are drawings of the settings rather than results: the
mask as drawn and the illumination as sampled, on the Mask and Source tabs.

Two rules that keep it responsive and correct:

* **The engine never runs on the GUI thread.** A :class:`Worker` lives on a
  ``QThread`` and communicates only by signal.
* **Figures are built with ``Figure()``, never ``pyplot``.** Pyplot keeps a
  global registry of figures that is not thread-safe and would leak one entry
  per redraw.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from litho_sim.app.compute import (
    FILM_PREVIEW_KEYS,
    ImagingResult,
    Profile3DResult,
    build_mask,
    film_preview,
    mask_preview,
    source_preview,
)
from litho_sim.app.ilt_tab import ILTTab
from litho_sim.app.opc import opc_summary
from litho_sim.app.params import (
    SPECS_BY_KEY,
    TAB_GROUPS,
    TABS,
    ParameterModel,
)
from litho_sim.app.qt import QtCore, QtGui, QtWidgets
from litho_sim.app.sem_tab import SemTab
from litho_sim.app.simulate_tab import SimulateTab
from litho_sim.app.solid_view import SolidView, install_surface_format
from litho_sim.app.stack_tab import StackTab
from litho_sim.app.step_tab import StepTab
from litho_sim.app.views import (
    BakeView,
    DevelopView,
    ExposeView,
    MaskView,
    Profile3DView,
    ResistView,
    SourceView,
)
from litho_sim.app.worker import Worker

logger = logging.getLogger(__name__)

#: Keys that decide which thick-mask models can run. Their combo lives on
#: the Mask tab; two of the three keys do not.
_MASK_MODEL_GATES = ("wavelength", "pattern", "mask_type")

#: Which stage each result-showing step tab's picture depends on. A change
#: at or before that stage makes the picture stale; a change after it does
#: not — moving the develop threshold does not date the aerial image.
_PRINT_STAGE_OF_TAB = {"Expose": "scale", "Bake": "resist", "Develop": "resist"}

STALE_PRINT = (
    "Settings changed since this was printed — run Print on the Simulate "
    "tab to bring it up to date."
)
STALE_PROFILE = (
    "Settings changed since this profile was developed — run 3-D resist "
    "profile on the Simulate tab."
)
PLACEHOLDER_PRINT = "Nothing printed yet — run Print on the Simulate tab."
PLACEHOLDER_PROFILE = (
    "No 3-D profile yet — run 3-D resist profile on the Simulate tab."
)
STALE_OPC = (
    "Settings changed since this mask was corrected — run OPC or Print on "
    "the Simulate tab to correct it for them."
)
NO_OPC_YET = (
    "Shown as drawn: the correction has not run yet. Run OPC or Print on "
    "the Simulate tab, and the corrected mask appears here."
)


class MainWindow(QtWidgets.QMainWindow):
    request_print = QtCore.Signal(object)
    request_opc = QtCore.Signal(object)
    request_dose = QtCore.Signal(object)
    request3d = QtCore.Signal(object)
    request_flow = QtCore.Signal(object)
    request_device = QtCore.Signal(str)
    request_fem = QtCore.Signal(object)
    request_stoch = QtCore.Signal(object)
    request_ilt = QtCore.Signal(object)   # ILTRequest

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LithoPy")
        self.resize(1320, 840)

        self.model = ParameterModel()

        # -- the views ---------------------------------------------------
        self.mask_view = MaskView()
        self.source_view = SourceView()
        self.resist_view = ResistView()
        self.expose_view = ExposeView()
        self.bake_view = BakeView()
        self.develop_view = DevelopView()
        self.profile_view = Profile3DView()

        # The resist profile is always present; 2-D or 3-D is which result
        # to look at, not a separate place to go.
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Profile:"))
        self.mode_2d = QtWidgets.QRadioButton("2-D")
        self.mode_3d = QtWidgets.QRadioButton("3-D")
        self.mode_2d.setChecked(True)
        mode_row.addWidget(self.mode_2d)
        mode_row.addWidget(self.mode_3d)
        mode_row.addStretch(1)
        self.develop_stack = QtWidgets.QStackedWidget()
        self.develop_stack.addWidget(self.develop_view)
        self.develop_stack.addWidget(self.profile_view)

        views = {
            "Mask": self.mask_view,
            "Source": self.source_view,
            "Resist": self.resist_view,
            "Expose": self.expose_view,
            "Bake": self.bake_view,
            "Develop": self.develop_stack,
        }

        # -- the tabs ----------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        self.step_tabs: dict[str, StepTab] = {}
        for name in TABS:
            tab = StepTab(
                self.model, TAB_GROUPS[name], views[name],
                header=mode_row if name == "Develop" else None,
            )
            tab.changed.connect(self._on_changed)
            self.tabs.addTab(tab, name)
            self.step_tabs[name] = tab
        self.develop_tab = self.step_tabs["Develop"]

        self.stack_tab = StackTab(self.model)
        self.simulate_tab = SimulateTab(self.model)
        self.sem_tab = SemTab()
        self.ilt_tab = ILTTab(self.model)
        self.tabs.addTab(self.stack_tab, "Wafer Stack")
        self.tabs.addTab(self.simulate_tab, "Simulate")
        self.tabs.addTab(self.ilt_tab, "ILT")
        self.tabs.addTab(self.sem_tab, "SEM")
        # The batch pages, by the names the rest of the app knows them by.
        self.pw_tab = self.simulate_tab.pw_tab
        self.stoch_tab = self.simulate_tab.stoch_tab

        self.tabs.setCurrentWidget(self.step_tabs["Mask"])
        self.setCentralWidget(self.tabs)

        self.status = self.statusBar()

        # -- results and the settings they were made from -----------------
        # A result is stale when the model has moved on from the snapshot it
        # was computed with, stage by stage; the snapshot is kept for that.
        self._last_result: ImagingResult | None = None
        self._print_params: ParameterModel | None = None
        self._inflight_print: ParameterModel | None = None
        self._busy_print = False
        self._profile: Profile3DResult | None = None
        self._profile_params: ParameterModel | None = None
        self._inflight_profile: ParameterModel | None = None
        self._busy_3d = False
        # The correction in hand — from its own run or inside a Print — the
        # settings it was made for, and the corrected mask rasterised on
        # that grid, which is what the Mask tab shows while it is on.
        self._opc = None
        self._opc_params: ParameterModel | None = None
        self._opc_mask = None
        self._drawn_opc = None      # the correction the Mask tab is showing
        self._inflight_opc: ParameterModel | None = None
        self._busy_opc = False

        # -- worker thread ---------------------------------------------
        self.thread = QtCore.QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.request_print.connect(self.worker.run)
        self.worker.done.connect(self._on_print_done)
        self.worker.failed.connect(self._on_failed)
        self.request_opc.connect(self.worker.run_opc)
        self.worker.done_opc.connect(self._on_opc_done)
        self.request_dose.connect(self.worker.run_dose_to_size)
        self.worker.done_dose.connect(self._on_dose_done)
        self.request3d.connect(self.worker.run_3d)
        self.worker.done3d.connect(self._on_done_3d)
        self.request_flow.connect(self.worker.run_flow)
        self.worker.done_flow.connect(self._on_flow_done)
        self.request_device.connect(self.worker.run_device)
        self.worker.done_device.connect(self._on_device_ready)
        # Signal-to-signal relays: the pages build their own requests (with
        # their own deep copies of the model); the window only carries them
        # onto the worker thread and the answers back.
        self.simulate_tab.run_fem.connect(self.request_fem)
        self.request_fem.connect(self.worker.run_fem)
        self.worker.done_fem.connect(self.pw_tab.on_finished)
        self.worker.progress_fem.connect(self.pw_tab.on_progress)
        self.simulate_tab.run_stoch.connect(self.request_stoch)
        self.request_stoch.connect(self.worker.run_stoch)
        self.worker.done_stoch.connect(self.stoch_tab.on_finished)
        self.worker.progress_stoch.connect(self.stoch_tab.on_progress)
        # ILT streams its iterations rather than a percentage: the frames are
        # the result, so progress_ilt carries whole states and the tab draws
        # each one. Queued across the thread boundary like every other.
        self.ilt_tab.run_requested.connect(self.request_ilt)
        self.request_ilt.connect(self.worker.run_ilt)
        self.worker.progress_ilt.connect(self.ilt_tab.on_frame)
        self.worker.done_ilt.connect(self.ilt_tab.on_done)
        self.thread.start()

        self.simulate_tab.run_print.connect(self._request_print)
        self.simulate_tab.run_opc.connect(self._request_opc)
        self.simulate_tab.run_dose.connect(self._request_dose)
        self.simulate_tab.run_profile.connect(self._request_3d)
        # One microscope: the SEM tab's instrument also images the Develop
        # tab's 3-D profile, so its knobs re-form that picture too.
        self.sem_tab.form.changed.connect(
            lambda _k: self.profile_view.set_instrument(self.sem_tab.config()))
        self.stack_tab.run_requested.connect(self._request_flow)
        self.stack_tab.import_requested.connect(self._import_profile)
        # The SEM images whatever step the Wafer Stack tab is showing — a
        # loaded device, or a flow as it is built — without a run of its own.
        self.stack_tab.shown.connect(self.sem_tab.set_stack)
        self.mode_2d.toggled.connect(self._on_mode_changed)
        self._build_menu()

        # The default is 193 nm, where 'multilayer' cannot run — so the Mask
        # tab has to reflect that before the user touches anything.
        self.step_tabs["Mask"].panel.refresh_mask_models()
        self._draw_mask()
        self._draw_source()
        self._draw_film()
        for view in (self.expose_view, self.bake_view, self.develop_view):
            view.show_placeholder(PLACEHOLDER_PRINT)
        self.profile_view.show_placeholder(PLACEHOLDER_PROFILE)
        self._on_mode_changed()
        self.status.showMessage(
            "Set up each step on its tab, then run Print on the Simulate tab."
        )

    # -- settings changed -------------------------------------------------
    def _on_changed(self, key: str, _value) -> None:
        """A control moved. Redraw what is a drawing; date what is a result."""
        spec = SPECS_BY_KEY[key]
        if key in _MASK_MODEL_GATES:
            self.step_tabs["Mask"].panel.refresh_mask_models()

        if spec.stage == "view":
            # The stage moved: the picture changes, the physics does not.
            # The profile in hand is re-imaged from the new angle.
            self.profile_view.set_stage(float(self.model["tilt"]),
                                        float(self.model["azimuth"]))
            return

        if spec.stage == "mask":
            self._draw_mask()
        if spec.target == "optics" or key in ("pattern", "pitch", "cd"):
            self._draw_source()
        if key in FILM_PREVIEW_KEYS:
            self._draw_film()

        self._refresh_stale()
        self.simulate_tab.refresh_costs()
        # The ILT tab's Run button and its reason follow the dock too: a
        # pattern that needs a bigger field says so there before a click.
        self.ilt_tab.refresh_availability()

    def _draw_mask(self) -> None:
        """The mask as it will go on the reticle: drawn, or corrected.

        With the correction on and one in hand, the corrected mask is shown
        on the grid it was corrected for — the banner says if the settings
        have moved past it. Otherwise the drawing of the Pattern settings,
        live as ever.
        """
        if self.model.opc_enabled and self._opc is not None and self._opc_params is not None:
            if self._drawn_opc is self._opc:
                # A Pattern tick cannot change a correction already in hand
                # — the banner says it is behind — and redrawing forty
                # outlines and a legend per tick is what a lagging slider
                # is made of.
                return
            self._drawn_opc = self._opc
            self.mask_view.show_corrected(
                self._opc_mask, float(self._opc_params["pixel_size"]), self._opc)
            return
        self._drawn_opc = None
        self.mask_view.show_mask(
            mask_preview(self.model), pixel_nm=float(self.model["pixel_size"])
        )

    def _draw_source(self) -> None:
        self.source_view.show_source(source_preview(self.model))

    def _draw_film(self) -> None:
        self.resist_view.show_film(film_preview(self.model))

    def _stale_print(self, stage: str) -> bool:
        p = self._print_params
        return p is not None and (
            self.model.stage_signature(stage) != p.stage_signature(stage)
        )

    @property
    def _stale_profile(self) -> bool:
        q = self._profile_params
        return q is not None and (
            self.model.stage_signature("profile3d")
            != q.stage_signature("profile3d")
        )

    @property
    def _stale_opc(self) -> bool:
        """Whether the correction in hand is for settings that have moved on."""
        q = self._opc_params
        return q is not None and self.model.opc_signature() != q.opc_signature()

    def _refresh_stale(self) -> None:
        """Put the banner on every picture the settings have moved past."""
        for name, stage in _PRINT_STAGE_OF_TAB.items():
            stale = self._stale_print(stage)
            if name == "Develop" and self._is_3d:
                stale = self._stale_profile
                text = STALE_PROFILE
            else:
                text = STALE_PRINT
            self.step_tabs[name].set_stale(text if stale else None)
        # The Mask tab is a drawing until the correction is on; then it is a
        # result like the others, and says so when it is behind or absent.
        mask_note = None
        if self.model.opc_enabled:
            if self._opc is None:
                mask_note = NO_OPC_YET
            elif self._stale_opc:
                mask_note = STALE_OPC
        self.step_tabs["Mask"].set_stale(mask_note)
        print_stale = self._stale_print("resist")
        self.simulate_tab.set_stale(print_stale, self._stale_profile, self._stale_opc)
        self.sem_tab.set_stale(print_stale, self._stale_profile)

    # -- print -------------------------------------------------------------
    def _request_print(self, params: ParameterModel) -> None:
        if self._busy_print:
            return
        self._busy_print = True
        self._inflight_print = params
        self.status.showMessage("printing…")
        self.request_print.emit(params)

    @QtCore.Slot(object)
    def _on_print_done(self, result: ImagingResult) -> None:
        self._busy_print = False
        self._last_result = result
        self._print_params, self._inflight_print = self._inflight_print, None
        # Drawn into every step view, shown or not: this happens once per
        # run rather than once per slider tick, and a hidden canvas defers
        # its paint until it is shown anyway.
        self.expose_view.show_result(result)
        self.bake_view.show_result(result)
        self.develop_view.show_result(result)
        self.simulate_tab.on_print_done(result)
        self.sem_tab.set_print(result)
        if result.opc is not None and self._print_params is not None:
            # The print corrected its mask on the way: that correction is
            # the one in hand now, on the Mask tab and the OPC page both.
            self._adopt_opc(result.opc, self._print_params)
        self.status.showMessage(result.summary)
        self._refresh_stale()

    # -- OPC ---------------------------------------------------------------
    def _request_opc(self, params: ParameterModel) -> None:
        if self._busy_opc:
            return
        self._busy_opc = True
        self._inflight_opc = params
        self.status.showMessage("correcting the mask…")
        self.request_opc.emit(params)

    def _adopt_opc(self, result, params: ParameterModel) -> None:
        """Take a correction as the one in hand and show it where it lands."""
        self._opc, self._opc_params = result, params
        # Rasterised once, on the grid it was corrected for: the Mask tab
        # redraws on every control tick and must not rasterise polygons then.
        mask = build_mask(params, result.corrected)
        self._opc_mask = mask.real if hasattr(mask, "real") and mask.dtype.kind == "c" else mask
        self.simulate_tab.on_opc_done(result, params.grid())
        self._draw_mask()

    @QtCore.Slot(object)
    def _on_opc_done(self, result) -> None:
        self._busy_opc = False
        params, self._inflight_opc = self._inflight_opc, None
        if params is None:
            return
        self._adopt_opc(result, params)
        self.status.showMessage(opc_summary(result))
        self._refresh_stale()

    def _request_dose(self, params: ParameterModel) -> None:
        self.status.showMessage("sizing the dose on the dense array…")
        self.request_dose.emit(params)

    @QtCore.Slot(float)
    def _on_dose_done(self, dose: float) -> None:
        """Move the Dose knob to the anchor dose — through its own slider,
        so the change is visible, dates what it dates, and is undone the
        same way any setting is."""
        import math

        cd = float(self.model["cd"])
        if math.isfinite(dose):
            self.step_tabs["Expose"].panel.set_value("dose", dose)
            self.status.showMessage(
                f"dose set to {self.model['dose']:.3f} — the dense array prints "
                f"{cd:.0f} nm")
        else:
            self.status.showMessage("no dose sizes the dense array at these settings")
        self.simulate_tab.on_dose_done(dose, cd)

    # -- 3-D ---------------------------------------------------------------
    @property
    def _is_3d(self) -> bool:
        return self.mode_3d.isChecked()

    def _on_mode_changed(self, _checked: bool = False) -> None:
        """Swap the Develop picture between the 2-D result and the solid."""
        self.develop_stack.setCurrentIndex(1 if self._is_3d else 0)
        # The readout belongs to whichever mode is showing. Leaving the 3-D
        # numbers up over a 2-D picture is the same lie as a stale plot.
        if self._is_3d and self._profile is not None:
            self.status.showMessage(self._profile.summary)
        elif not self._is_3d and self._last_result is not None:
            self.status.showMessage(self._last_result.summary)
        self._refresh_stale()

    def _request_3d(self, params: ParameterModel) -> None:
        if self._busy_3d:
            return
        self._busy_3d = True
        self._inflight_profile = params
        self.status.showMessage("developing in 3-D…")
        self.request3d.emit(params)

    @property
    def _profile_grid(self):
        """The grid the profile was developed on — not necessarily today's."""
        return (self._profile_params or self.model).grid()

    def _draw_profile(self) -> None:
        """Render the profile already in hand. No physics happens here."""
        self.profile_view.show_profile(
            self._profile,
            self._profile_grid,
            self.sem_tab.config(),
            tilt=float(self.model["tilt"]),
            azimuth=float(self.model["azimuth"]),
        )

    @QtCore.Slot(object)
    def _on_done_3d(self, profile) -> None:
        self._busy_3d = False
        self._profile = profile
        self._profile_params, self._inflight_profile = self._inflight_profile, None
        # There is now something to adopt, so the button stops being greyed out.
        self.stack_tab.import_btn.setEnabled(True)
        self._draw_profile()
        # Asking for the 3-D profile is asking to see it. Flip the Develop
        # tab to 3-D so the run lands somewhere visible.
        self.mode_3d.setChecked(True)
        self.simulate_tab.on_profile_done(profile, self._profile_grid)
        self.sem_tab.set_profile(profile, self._profile_grid)
        self.status.showMessage(profile.summary)
        self._refresh_stale()

    # -- File and Simulate menus ------------------------------------------
    def _build_menu(self) -> None:
        """File ▸ open, save, and the named device presets; Simulate ▸ the runs.

        The Wafer Stack tab's own button adopts the *in-memory* develop result
        and nothing else; anything arriving from disk comes through here.
        """
        from litho_sim.app.simulate_tab import KINDS
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

        # The same runs the Simulate tab lists, reachable from any tab. Each
        # jumps to the Simulate tab, selects the run, and presses its button.
        self.simulate_menu = sim = self.menuBar().addMenu("&Simulate")
        shortcuts = {"Print": "Ctrl+R", "3-D resist profile": "Ctrl+Shift+R"}
        self.simulate_actions: dict[str, QtGui.QAction] = {}
        for kind in KINDS:
            act = sim.addAction(kind)
            if kind in shortcuts:
                act.setShortcut(QtGui.QKeySequence(shortcuts[kind]))
            act.triggered.connect(lambda _=False, k=kind: self._run_from_menu(k))
            self.simulate_actions[kind] = act

    def _run_from_menu(self, kind: str) -> None:
        self.simulate_tab.select(kind)
        self.tabs.setCurrentWidget(self.simulate_tab)
        self.simulate_tab.run_current()

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
        self.stack_tab.load_stack(stack, label=label, flow=flow, section=section)
        self.tabs.setCurrentWidget(self.stack_tab)
        steps = f"{len(flow)} steps  —  " if flow is not None else ""
        self.status.showMessage(
            f"{label}  —  {steps}drag to rotate, scroll to zoom, r reframes"
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
        self.stack_tab.load_stack(stack, label=Path(path).name)
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
                "No developed profile yet — run 3-D resist profile on the "
                "Simulate tab first."
            )
            return
        try:
            self.stack_tab.import_profile(self._profile.remaining, self._profile_grid)
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
        # A courtesy only. The authoritative readout is the tab's own label.
        if error is None:
            n = self.stack_tab.session.valid_upto
            self.status.showMessage(f"flow complete — {n} step(s)")
        else:
            self.status.showMessage(f"flow error: {error}")

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        # Whichever run was waiting is not coming back; release everything
        # so the next press still runs.
        self._busy_print = False
        self._busy_3d = False
        self._busy_opc = False
        self._inflight_print = None
        self._inflight_profile = None
        self._inflight_opc = None
        self.simulate_tab.on_failed(message)
        self.status.showMessage(f"error: {message}")

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        self.thread.quit()
        self.thread.wait(2000)
        # The render windows go before Qt tears down their GL contexts —
        # the other order is the classic VTK exit-time crash.
        self.profile_view.shutdown()
        self.stack_tab.view.shutdown()
        super().closeEvent(event)


def main(stack=None, label: str = "printed device",
         flow=None, section=None) -> int:
    """Open the app.

    Parameters
    ----------
    stack : Stack, optional
        A wafer to adopt on startup — the app opens on the Wafer Stack tab in
        3-D with this already drawn. See ``scripts/show_device.py`` for the
        device flows that produce one.
    label : str
        Shown against the scrubber's loaded position.
    flow : DeviceFlow, optional
        The recipe that built *stack*, so the Wafer Stack panel lists its steps
        and the scrubber can walk them.
    section : callable, optional
        ``Stack -> Stack``, cutting the device open for the 3-D view.
    """
    logging.getLogger("litho_sim").setLevel(logging.WARNING)
    # Before the QApplication: the GL surface format is read when the first
    # window is created and cannot be changed afterwards.
    install_surface_format()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    if not SolidView.available:
        logger.warning("3-D views are off: %s", "pyvista/pyvistaqt not installed")
    if stack is not None:
        window.stack_tab.load_stack(stack, label=label, flow=flow, section=section)
        window.tabs.setCurrentWidget(window.stack_tab)
    window.show()
    # PyQt5 spells it exec_(); Qt6 bindings and PyQt5>=5.15 have exec().
    return (getattr(app, "exec", None) or app.exec_)()


if __name__ == "__main__":
    raise SystemExit(main())
