"""The printed-stack tab: device flows, the recipe editor, and the 3-D view."""

from __future__ import annotations

import logging

from litho_sim.app.controls import SpecForm, StepReadout
from litho_sim.app.params import (
    ParameterModel,
)
from litho_sim.app.qt import QtCore, QtGui, QtWidgets
from litho_sim.app.solid_view import INSTALL_HINT, SolidView
from litho_sim.app.views import StackView
from litho_sim.patterning.steps import ProcessStep

logger = logging.getLogger(__name__)

#: How long an edit has to sit before the flow re-runs, in milliseconds. The
#: recipe editor is the one place the app still recomputes on its own — a
#: step append replays one step and the picture should follow the edit — so
#: this is a trailing-edge debounce over slider drags, and nothing else.
SETTLE_MS = 180.0


class StackTab(QtWidgets.QWidget):
    """Build a process flow step by step and watch the wafer change.

    Deliberately self-contained: the palette, the recipe, the step form and the
    view controls all live here rather than in the global parameter dock. The
    dock belongs to the imaging pipeline, and a step's parameters are not
    global knobs — mixing them would also fold every stack edit into the
    imaging cache keys, which are computed by exclusion and would happily
    invalidate a 730 ms 3-D latent because someone changed a deposit thickness.
    """

    run_requested = QtCore.Signal(object)      # FlowSession
    import_requested = QtCore.Signal()
    #: ``(Stack, label)`` — the wafer the view is drawing, whole. Fires on
    #: every scrub, load and finished run, so anything else that images the
    #: wafer (the SEM tab) follows the step on screen.
    shown = QtCore.Signal(object, str)

    def __init__(self, model: ParameterModel, parent=None):
        super().__init__(parent)
        from litho_sim.app.flow_model import FlowSession
        from litho_sim.app.stepspecs import PALETTE, STEP_SPECS, defaults_for

        self.model = model
        self.session = FlowSession(grid=model.grid())
        self._specs_for = STEP_SPECS
        self._defaults_for = defaults_for

        # The worker takes the session by reference, so the GUI thread must not
        # touch it while a run is in flight. `_busy_flow` is that window;
        # `_deferred` holds edits made during it, keyed by row, and is drained
        # when the run lands. `_pending_flow` is the one queued re-run.
        self._busy_flow = False
        self._pending_flow = False
        self._deferred: dict[int, ProcessStep] = {}
        #: Mesh stride for the 3-D view. 4 is right for a flow being built —
        #: the wafer is redrawn on every edit — and wrong for a finished device
        #: loaded to be looked at, which `load_stack` drops to 1.
        self._loaded_label = "as loaded"
        #: Has a wafer been handed to this tab — a device, a saved ``.npz``, a
        #: developed profile? If so its grid is the authority, and adding the
        #: first step must not quietly replace it with the dock's.
        self._inherited = False
        #: How to cut a device open for the 3-D view, or ``None``. Applied at
        #: draw time rather than stored, because it is a *viewing* decision and
        #: only the solid view needs it — see `_on_scrub`.
        self._section = None
        self._edit_settle = QtCore.QTimer(self)
        self._edit_settle.setSingleShot(True)
        self._edit_settle.setInterval(int(SETTLE_MS))
        self._edit_settle.timeout.connect(self._run)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # -- left: the recipe ------------------------------------------
        left = QtWidgets.QVBoxLayout()
        palette = QtWidgets.QHBoxLayout()
        palette.addWidget(QtWidgets.QLabel("Add:"))
        self.palette_box = QtWidgets.QComboBox()
        self.palette_box.addItems(list(PALETTE))
        palette.addWidget(self.palette_box, 1)
        self.add_btn = QtWidgets.QPushButton("+")
        self.add_btn.setFixedWidth(30)
        palette.addWidget(self.add_btn)
        left.addLayout(palette)

        self.recipe = QtWidgets.QListWidget()
        self.recipe.setMinimumWidth(230)
        left.addWidget(self.recipe, 2)

        row = QtWidgets.QHBoxLayout()
        self.up_btn = QtWidgets.QPushButton("▲")
        self.down_btn = QtWidgets.QPushButton("▼")
        self.del_btn = QtWidgets.QPushButton("Remove")
        for b in (self.up_btn, self.down_btn, self.del_btn):
            row.addWidget(b)
        left.addLayout(row)

        self.form_box = QtWidgets.QGroupBox("Step settings")
        self.form = SpecForm()
        # The as-built twin. Both live here and exactly one is ever visible:
        # which one depends on whether the selected step is editable, and
        # swapping the widget is clearer than a form that silently stops
        # accepting input.
        self.readout = StepReadout()
        self.readout.setVisible(False)

        holder = QtWidgets.QWidget()
        holder_layout = QtWidgets.QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.addWidget(self.form)
        holder_layout.addWidget(self.readout)
        holder_layout.addStretch(1)

        # Long recipes have long readouts — a sixteen-field etch does not fit
        # the panel, and a settings box you cannot scroll is one that hides
        # settings.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(holder)

        box_layout = QtWidgets.QVBoxLayout(self.form_box)
        box_layout.setContentsMargins(4, 4, 4, 4)
        box_layout.addWidget(scroll)
        left.addWidget(self.form_box, 1)

        # Named for what it does, not for what it sounds like. "Import resist
        # profile" reads as a file dialog, and it is not one — it adopts the
        # volume the Develop tab is holding. Reported as "the button has little
        # to no functionality" by someone reasonably looking for a file picker;
        # File ▸ Open wafer is the one that reads from disk.
        self.import_btn = QtWidgets.QPushButton("Use developed profile as wafer")
        # Enabled only once there is a profile to use. An always-enabled button
        # whose whole response is one line in the status bar is indistinguishable
        # from a dead control — which is exactly how it was read.
        self.import_btn.setEnabled(False)
        self.import_btn.setToolTip(
            "Take the developed 3-D profile (run 3-D resist profile on the "
            "Simulate tab) and make it this flow's starting wafer: substrate "
            "plus patterned resist, ready to etch through.\n\n"
            "Not a file dialog — use File ▸ Open wafer to load a saved .npz, "
            "or File ▸ Load device for the GAA and nFET presets."
        )
        left.addWidget(self.import_btn)

        # Next to the profile button, not only in the File menu. Someone with
        # an empty Wafer Stack tab looks for a button *on that tab* to put
        # something in it — which is exactly how the profile button came to be
        # read as a broken file importer. The menu is still there; this is the
        # same actions, where the eye already is.
        self.device_btn = QtWidgets.QPushButton("Load device ▾")
        self.device_btn.setToolTip(
            "Load a printed device — the GAA nanosheet or the planar nFET — "
            "as this flow's starting wafer, drawn in 3-D.\n\n"
            "First load builds it (2–7 s); after that it is cached."
        )
        left.addWidget(self.device_btn)

        self.run_btn = QtWidgets.QPushButton("Run flow")
        left.addWidget(self.run_btn)

        # The tab's own readout. `MainWindow.status` is shared with the imaging
        # pipeline, which overwrites it on every recompute — a flow's "step 2
        # left the wafer unchanged" was reliably replaced by "computing…"
        # before anyone could read it, and auto-run makes that fight constant.
        self.notes = QtWidgets.QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setMinimumHeight(48)
        self.notes.setStyleSheet("color: #666;")
        left.addWidget(self.notes)
        outer.addLayout(left)

        # -- right: the wafer ------------------------------------------
        right = QtWidgets.QVBoxLayout()
        modes = QtWidgets.QHBoxLayout()
        modes.addWidget(QtWidgets.QLabel("View:"))
        self.mode_section = QtWidgets.QRadioButton("cross-section")
        self.mode_solid = QtWidgets.QRadioButton("3-D")
        self.mode_section.setChecked(True)
        if not SolidView.available:
            # Greyed, not hidden: the control exists, and the tooltip says
            # what it would take to make it act.
            self.mode_solid.setEnabled(False)
            self.mode_solid.setToolTip(INSTALL_HINT)
        modes.addWidget(self.mode_section)
        modes.addWidget(self.mode_solid)
        modes.addStretch(1)

        # At the far end, because it answers a different question: the radios
        # pick *which* view, this says what is shown beside it.
        self.show_flow = QtWidgets.QCheckBox("process flow")
        self.show_flow.setChecked(True)
        self.show_flow.setToolTip(
            "The step strip beside the cross-section, coloured by process "
            "family — deposition, litho, etch, CMP.\n\n"
            "It overlaps the recipe list on the left, which names the same "
            "steps in the vocabulary you edit them in. Turn it off to give "
            "the wafer the full width; the list still has the flow.\n\n"
            "The 3-D view has no strip."
        )
        modes.addWidget(self.show_flow)
        right.addLayout(modes)

        self.view = StackView()
        right.addWidget(self.view, 1)

        scrub = QtWidgets.QHBoxLayout()
        scrub.addWidget(QtWidgets.QLabel("Step:"))
        self.scrubber = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.scrubber.setMinimum(0)
        self.scrubber.setMaximum(0)
        scrub.addWidget(self.scrubber, 1)
        self.scrub_label = QtWidgets.QLabel("as loaded")
        self.scrub_label.setMinimumWidth(200)
        scrub.addWidget(self.scrub_label)
        right.addLayout(scrub)
        outer.addLayout(right, 1)

        self.add_btn.clicked.connect(self._add)
        self.del_btn.clicked.connect(self._remove)
        self.up_btn.clicked.connect(lambda: self._move(-1))
        self.down_btn.clicked.connect(lambda: self._move(+1))
        self.run_btn.clicked.connect(self._run)
        self.import_btn.clicked.connect(self.import_requested)
        self.recipe.currentRowChanged.connect(self._on_selected)
        self.form.changed.connect(self._on_form_changed)
        self.scrubber.valueChanged.connect(self._on_scrub)
        self.mode_section.toggled.connect(self._on_view_mode)
        self.show_flow.toggled.connect(self._on_show_flow)

        self.view.show_placeholder("Add a step to start building a wafer")

    # -- the view ------------------------------------------------------
    def _on_view_mode(self, section: bool) -> None:
        """Cross-section or solid. `_on_scrub` does the drawing, once."""
        self.view.set_mode(not section, redraw=False)
        # Greyed rather than left live in 3-D: the solid view has no strip, so
        # a checkbox that still ticked there would be a control with nothing
        # to act on — the failure this app has been bitten by twice.
        self.show_flow.setEnabled(section)
        self._on_scrub(self.scrubber.value())

    def _on_show_flow(self, show: bool) -> None:
        self.view.set_panel(show)

    # -- recipe editing ------------------------------------------------
    def _add(self) -> None:
        from litho_sim.app.stepspecs import build_step

        kind = self.palette_box.currentText()
        # An empty recipe adopts the dock's grid. Once steps exist it must not
        # change underneath them — and `Pattern` derives its line count from
        # `ctx.grid`, so a stale grid stopped being invisible.
        #
        # "Empty" is not the test, though: an inherited wafer brought its own
        # grid, and File ▸ Open wafer leaves the recipe empty, so the first
        # `+` used to swap that grid for the dock's and leave the steps
        # running at a pixel size the wafer underneath does not have.
        if not len(self.session) and not self._inherited:
            self.session.grid = self.model.grid()
            self.session.reset()

        # Defaults chosen for the wafer as it stands, not in the abstract: a
        # step whose target is not on the surface runs and does nothing, and
        # a no-op is not an error, so nothing would say why.
        #
        # "As it stands" has to mean *after the steps already queued*, not
        # after the last simulated one. Building a recipe before pressing Run
        # left every step reading the bare substrate, so a coat followed by an
        # etch proposed etching the substrate through 90 nm of resist. `run_to`
        # is incremental and returns immediately when nothing is missing, so
        # this normally costs nothing; auto-run keeps it that way.
        self._catch_up()
        wafer = self.session.stack_at(self.session.valid_upto - 1)
        self.session.append(build_step(kind, self._defaults_for(kind, wafer)))
        self._refresh_list()
        self.recipe.setCurrentRow(len(self.session) - 1)
        self._touch()

    def _catch_up(self) -> None:
        """Bring the session up to date, quietly, if it has fallen behind."""
        if self._busy_flow or self.session.valid_upto >= len(self.session):
            return
        try:
            self.session.run_to()
        except Exception:                                 # noqa: BLE001
            # A broken step is the run's problem to report, not the add's.
            logger.debug("catch-up run failed; defaults fall back to the "
                         "last good wafer", exc_info=True)

    def _remove(self) -> None:
        row = self.recipe.currentRow()
        if self.session.locked_upto <= row < len(self.session):
            self.session.remove(row)
            self._refresh_list()
            self._touch()

    def _move(self, delta: int) -> None:
        row = self.recipe.currentRow()
        to = row + delta
        low = self.session.locked_upto
        if low <= row < len(self.session) and low <= to < len(self.session):
            self.session.move(row, to)
            self._refresh_list()
            self.recipe.setCurrentRow(to)
            self._touch()

    def _refresh_list(self) -> None:
        row = self.recipe.currentRow()
        idle = set(self.session.did_nothing())
        self.recipe.blockSignals(True)
        self.recipe.clear()
        locked = self.session.locked_upto
        as_built = self.session.as_built_upto
        # Only an adopted flow shows staleness: a flow being built re-runs
        # itself on a settle timer, so its tail is never out of date for long
        # enough to be worth saying.
        valid = self.session.valid_upto if as_built else len(self.session)
        for i, line in enumerate(self.session.describe()):
            suffix = "   — no change" if i in idle else ""
            item = QtWidgets.QListWidgetItem(line + suffix)
            tip = []
            if i < as_built:
                # Muted rather than marked with a glyph: this is the normal
                # state for every row of a loaded device, and 36 badges would
                # be decoration. The colour separates "what the device did"
                # from "what you have added on top", which is the distinction
                # that actually matters when both are in one list.
                item.setForeground(QtGui.QColor("#7a8ba0"))
                tip.append(
                    "As built — this step ran when the device was made. Edit "
                    "it and the flow replays from here when you press Run flow."
                )
            if i < locked:
                tip.append(
                    "Read-only: this recipe arrived without the layouts and "
                    "optics it ran with, so it cannot be replayed."
                )
            if i in idle:
                item.setForeground(QtGui.QColor("#b06000"))
                tip.append(
                    "This step ran and left the wafer exactly as it was. Usually "
                    "the target material is not exposed at the surface."
                )
            if i >= valid:
                # Beyond the last live snapshot: this step's result no longer
                # reflects the settings above it. Last, so it wins the colour —
                # of everything a row can be saying, "out of date" is the one
                # that is actionable.
                item.setText(item.text() + "   · stale")
                item.setForeground(QtGui.QColor("#c05000"))
                tip.append(
                    "Out of date — an edit above this invalidated it. Press "
                    "Run flow to replay from the edit."
                )
            # Whatever the engine had to say about this step, verbatim. A step
            # can also be *partly* idle — an etch that removed material but
            # silently dropped its profile — which no change flag can catch.
            tip.extend(self.session.notes_for(i))
            if tip:
                item.setToolTip("\n\n".join(tip))
            self.recipe.addItem(item)
        self.recipe.setCurrentRow(min(row, len(self.session) - 1))
        self.recipe.blockSignals(False)
        self._sync_scrubber()

    def _on_selected(self, row: int) -> None:
        from litho_sim.app.stepspecs import specs_for_step, uneditable_keys, values_for

        self._sync_edit_buttons(row)
        if not (0 <= row < len(self.session)):
            self.form.set_specs((), {})
            self._show_readout(None)
            self._show_notes()
            return
        self._sync_scrubber_to_row(row)
        # Through the deferred buffer: during a run the session still holds the
        # pre-edit step, and repopulating from it would visibly revert whatever
        # the user just typed.
        step = self._deferred.get(row)
        if step is None:
            step = self.session.steps[row]

        if row < self.session.locked_upto:
            # Nothing here can replay this step, so nothing here may change it.
            self.form.set_specs((), {})
            self._show_readout(step)
        else:
            self._show_readout(None)
            specs = specs_for_step(step, self.session.layouts)
            self.form.set_specs(
                specs,
                values_for(step) if specs else {},
                readonly=uneditable_keys(step),
            )
        self._show_notes()

    def _show_readout(self, step) -> None:
        """Swap the settings box between the editor and the as-built readout."""
        if step is None:
            self.readout.clear()
            self.readout.setVisible(False)
            self.form.setVisible(True)
            self.form_box.setTitle("Step settings")
            return
        self.readout.show_step(step)
        self.readout.setVisible(True)
        self.form.setVisible(False)
        self.form_box.setTitle("Step settings — as built")

    def _sync_edit_buttons(self, row: int) -> None:
        """Grey the editing controls for a step that cannot be edited.

        A locked row refuses removal and reordering anyway; disabling the
        buttons says so before the click rather than after it.
        """
        if self._busy_flow:
            return                       # `_set_editing_enabled` owns them
        editable = self.session.locked_upto <= row < len(self.session)
        for b in (self.del_btn, self.up_btn, self.down_btn):
            b.setEnabled(editable)

    def _on_form_changed(self, key: str) -> None:
        from litho_sim.app.stepspecs import apply_edit

        row = self.recipe.currentRow()
        if not (self.session.locked_upto <= row < len(self.session)):
            # A locked row has no editor showing, so this should not fire — but
            # `replace` would raise, and a stray signal must not take the
            # window down.
            return
        # One field onto the step that is already there, rather than a step
        # rebuilt from the form. Everything the form does not show — the rest
        # of a selectivity table, a second etch target — is therefore never
        # written, and so cannot be lost.
        current = self._deferred.get(row) or self.session.steps[row]
        step = apply_edit(current, key, self.form.values())
        if self._busy_flow:
            # The worker owns the session for this window. Show the edit, hold
            # it. Keyed by row rather than a single slot so editing step 5 does
            # not discard a pending edit to step 3; the keys stay valid because
            # add/remove/move are disabled for exactly this window.
            self._deferred[row] = step
            self._pending_flow = True
        else:
            self.session.replace(row, step)
            self._sync_scrubber()
        item = self.recipe.item(row)
        if item is not None:
            item.setText(f"{row + 1:2d}. {step.describe()}")
        self._touch()

    # -- running -------------------------------------------------------
    def _touch(self) -> None:
        """An edit landed — re-run once the user stops changing things.

        Restarting a running single-shot timer *is* trailing-edge debounce, so
        a slider drag costs one run at the end rather than one per tick.

        Not for an adopted device, where the cost is the other way round.
        Appending to a flow you are writing replays one step and the picture
        should follow you; editing step 8 of a 36-step GAA replays 29 and takes
        about six seconds, which is worse on every slider release than a
        button pressed once. There the edit only marks the tail stale.
        """
        if self.session.as_built_upto:
            self._mark_stale()
            return
        self._edit_settle.start()

    def _mark_stale(self) -> None:
        """Say what is now out of date, and wait to be told to recompute."""
        self._refresh_list()
        self._sync_scrubber()
        self.run_btn.setEnabled(True)
        self._show_notes()

    def _set_editing_enabled(self, on: bool) -> None:
        # Discrete click targets only. Greying the sliders would fight a drag
        # in progress; these are safe to disable for the run window, and they
        # are exactly the ones that would reorder the rows `_deferred` keys.
        for w in (self.add_btn, self.del_btn, self.up_btn, self.down_btn,
                  self.import_btn, self.palette_box, self.run_btn):
            w.setEnabled(on)

    def _run(self) -> None:
        self._edit_settle.stop()
        if self._busy_flow:
            self._pending_flow = True
            return
        if not len(self.session):
            return
        self._busy_flow = True
        self._set_editing_enabled(False)
        self.run_requested.emit(self.session)

    def on_finished(self, error: str | None = None) -> None:
        self._busy_flow = False
        # Before the refresh, so the markers describe the recipe the user is
        # actually looking at rather than the one the run started from.
        drained = bool(self._deferred)
        for row, step in self._deferred.items():
            if 0 <= row < len(self.session):
                self.session.replace(row, step)
        self._deferred.clear()

        self._set_editing_enabled(True)
        self._refresh_list()
        if error is None:
            self.scrubber.setValue(self.scrubber.maximum())
            self._on_scrub(self.scrubber.value())
        self._show_notes(error)

        if drained or self._pending_flow:
            self._pending_flow = False
            # Through the settle timer, not straight back into `_run`. The
            # worker connection is queued, so a direct call is safe in the app
            # — but it makes finishing a run able to start one inside itself,
            # which is a stack overflow waiting for the first caller that
            # connects `run_requested` synchronously. It also means a burst of
            # edits made during a run coalesces into one follow-up.
            self._touch()

    def load_stack(self, stack, label: str = "printed device",
                   solid: bool = True, flow=None, section=None) -> None:
        """Adopt an already-built wafer as this flow's starting state.

        The sibling of :meth:`import_profile`, one level further along: that
        one takes a developed resist volume and coats it onto a fresh
        substrate, this one takes a finished :class:`~litho_sim.wafer.stack.Stack`
        — a device out of ``demo_gaa`` or ``demo_nfet``, or anything
        ``Stack.load`` can read — and makes it the wafer you inherit. Whatever
        you add runs *on top of* it rather than rebuilding it.

        With a *flow*, the recipe that built the wafer is adopted too, so the
        list shows the steps that made the device instead of standing empty.
        Without one — a bare ``.npz`` off disk, which carries no recipe — the
        behaviour is what it always was.

        Parameters
        ----------
        stack : Stack
            The wafer to adopt.
        label : str
            Shown against the scrubber's "as loaded" position.
        solid : bool
            Open in the 3-D view rather than the cross-section.
        flow : DeviceFlow, optional
            The as-built recipe and one wafer per step of it. See
            :meth:`~litho_sim.app.flow_model.FlowSession.adopt`.
        section : callable, optional
            ``Stack -> Stack``, cutting the wafer open for the 3-D view. Kept
            as a function rather than applied to the wafers because the
            cross-section must not be cut: see :meth:`_on_scrub`.
        """
        self._section = section
        # A recipe's own grid, where there is one. `Stack.save` keeps only the
        # lateral grid, so the grid read back off a wafer carries the default
        # 21 resist slices rather than the 5 the device exposed through —
        # enough to make a replayed exposure disagree with the one on screen.
        self.session.grid = getattr(flow, "grid", None) or stack.grid
        if flow is not None:
            # The context travels with the recipe, which is what makes the
            # steps editable rather than a record: without the layouts and
            # optics, replaying an exposure would raise or print at the wrong
            # wavelength, and `adopt` locks the flow instead.
            self.session.adopt(
                flow.steps, flow.snapshots, flow.base, flow.changed,
                layouts=getattr(flow, "layouts", None),
                optics=getattr(flow, "optics", None),
                resist=getattr(flow, "resist", None),
            )
        else:
            self.session.reset(stack)
        # Set before `_refresh_list`, which selects a row and so reaches
        # `_on_selected` → `_add`'s grid guard.
        self._inherited = True
        # What the scrubber's first position *is*. Without a flow that slot
        # holds the finished device, so it takes the device's name; with one it
        # holds the bare wafer the flow started from, and calling that "GAA
        # nanosheet" would label a blank substrate as the transistor.
        self._loaded_label = "bare wafer" if flow is not None else label
        self._refresh_list()
        if solid and self.mode_solid.isEnabled():
            # The toggle is wired to `mode_section`, so setting the 3-D radio
            # is what actually swaps the page.
            self.mode_solid.setChecked(True)
        # Land on the finished wafer, not the bare substrate: loading a device
        # is a request to look at the device. With a flow that is the last
        # step; without one the only position there is.
        self.scrubber.setValue(self.scrubber.maximum())
        if flow is not None and len(self.session):
            self.recipe.setCurrentRow(len(self.session) - 1)
        self._on_scrub(self.scrubber.value())
        # A different wafer deserves a fresh framing; scrubbing and re-running
        # keep the camera, loading does not.
        self.view.reframe()
        # Only if something is actually missing. An adopted flow arrives with
        # every snapshot already taken, and scheduling a run for it marked the
        # tab busy for a round trip that had nothing to compute — which, while
        # it lasted, froze the scrubber and greyed the controls.
        if self.session.valid_upto < len(self.session):
            self._touch()

    def import_profile(self, remaining, grid, material: str = "photoresist") -> None:
        """Make a developed litho profile this flow's starting wafer.

        Not a recipe step, and deliberately so: the profile is a volume, and a
        step has to survive ``to_dict``. It arrives the way a wafer arrives in
        a fab — as the state you inherit — so the recipe stays serialisable
        data describing what you do *next*.

        Coats a resist film thick enough to hold the profile and carves it
        in through ``write_resist_to_stack``, which aligns by the **top**
        surface. Aligning tops rather than bottoms is what makes it work over
        existing topography, where the film is thicker in trenches than over
        lines.
        """
        from litho_sim.develop.resist3d import write_resist_to_stack
        from litho_sim.wafer import Stack

        thickness = remaining.shape[0] * grid.dz
        base = Stack.blank(grid, dz=grid.dz, headroom=thickness + 200e-9)
        base.deposit_blanket(material, thickness, planarize=True)
        write_resist_to_stack(base, remaining, material=material)

        self.session.grid = grid
        self.session.reset(base)
        self._inherited = True
        self._refresh_list()
        self.scrubber.setValue(0)
        self._on_scrub(0)
        self._touch()

    def _show_notes(self, error: str | None = None) -> None:
        """What the last run had to say, and what the selected step had to say."""
        if error is not None:
            self.notes.setText(f"Flow error — {error}")
            return
        lines = []
        pending = len(self.session) - self.session.valid_upto
        if self.session.as_built_upto and pending > 0:
            lines.append(
                f"{pending} step{'s' if pending > 1 else ''} out of date — "
                f"press Run flow to replay from your edit."
            )
        idle = self.session.did_nothing()
        if idle:
            which = ", ".join(str(i + 1) for i in idle)
            lines.append(
                f"Step{'s' if len(idle) > 1 else ''} {which} ran and left the "
                f"wafer unchanged."
            )
        row = self.recipe.currentRow()
        # Verbatim, never parsed: the engine owns the wording of these.
        lines.extend(self.session.notes_for(row))
        self.notes.setText("\n".join(lines))

    def _sync_scrubber(self) -> None:
        # +1 for "as loaded", which snapshots() has no entry for.
        self.scrubber.setMaximum(max(self.session.valid_upto, 0))

    def _sync_scrubber_to_row(self, row: int) -> None:
        """Selecting a step shows the wafer as that step left it.

        The list and the scrubber are two views of one sequence — a step and
        the wafer it produced — so browsing the recipe walks the build. Without
        this the settings of step 10 sit beside the picture of step 36, which
        reads as the panel describing something other than what is on screen.
        """
        if self._busy_flow or not (0 <= row < self.session.valid_upto):
            return
        if self.scrubber.value() != row + 1:
            self.scrubber.setValue(row + 1)     # fires `_on_scrub`

    def _on_scrub(self, value: int) -> None:
        # The worker is extending `_snaps` and `_changed` right now, and they
        # can be caught at different lengths. The picture is about to be
        # replaced anyway.
        if self._busy_flow:
            return
        index = value - 1
        # Both bounds: a snapshot without a step behind it has no label, and
        # the two lists are only equal in length between runs.
        if index >= min(self.session.valid_upto, len(self.session)):
            return
        stack = self.session.stack_at(index)
        label = (
            self._loaded_label if index < 0
            else self.session.steps[index].describe()
        )
        # Whole, before the section: the SEM cleaves where it is told to,
        # and a cut that had already moved the fin out of the middle would
        # image the wrong place for the same reason the cross-section below
        # is not cut.
        self.shown.emit(stack, label)
        # The section is a 3-D decision, applied here rather than baked into
        # the wafers, and only in the solid view.
        #
        # A cross-section takes the middle slice of whatever it is given, so
        # cropping first *moves* that slice. The GAA cut keeps y from 45%, and
        # its midpoint lands just outside the fin — which reads fine on the
        # finished device, where the STI and ILD have filled that region back
        # in, and shows a blank plot at the fin etch, where everything off-fin
        # has just been etched to the substrate. Measured: 0.0% of that slice
        # is material. Whole, the same step shows resist over the superlattice.
        if self._section is not None and not self.mode_section.isChecked():
            stack = self._section(stack)
        self.scrub_label.setText(label)
        self.view.show_stack(stack, label=label)


