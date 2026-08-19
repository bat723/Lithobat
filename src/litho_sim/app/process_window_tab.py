"""The process-window tab: focus-exposure matrices, computed off-thread."""

from __future__ import annotations

import copy
import logging

from litho_sim.app.compute import (
    estimate_cost_ms,
)
from litho_sim.app.fem import FemRequest, ProcessWindowResult
from litho_sim.app.params import (
    ParameterModel,
)
from litho_sim.app.qt import Figure, FigureCanvasQTAgg, QtCore, QtWidgets

logger = logging.getLogger(__name__)


class ProcessWindowTab(QtWidgets.QWidget):
    """Focus-exposure matrix over the dock's optics: Bossung curves, CD map,
    the process window, EL-vs-DOF, NILS through focus, MEEF.

    Self-contained like :class:`StackTab`, and for the same reason its
    docstring gives: the sweep ranges parameterise this tab alone, and folding
    them into the dock would write them into every imaging cache signature.
    The optics and resist themselves *do* come from the dock — the sweep
    describes the configuration the live tabs are showing, snapshotted by
    deep copy at Run-press time so the worker never races the sliders.

    A sweep is seconds of Abbe sums, far past the live budget, so everything
    here is behind an explicit Run button — the same bargain as Compute 3-D.
    """

    run_requested = QtCore.Signal(object)      # FemRequest

    def __init__(self, model: ParameterModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._busy = False
        self._result: ProcessWindowResult | None = None

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # -- left: the sweep settings ----------------------------------
        box = QtWidgets.QGroupBox("Sweep")
        form = QtWidgets.QFormLayout(box)

        self.focus_range = QtWidgets.QDoubleSpinBox()
        self.focus_range.setRange(50.0, 2000.0)
        self.focus_range.setSingleStep(50.0)
        self.focus_range.setValue(300.0)
        self.focus_range.setDecimals(0)
        self.focus_range.setSuffix(" nm")
        self.focus_range.setToolTip(
            "Half-range: the sweep spans −range … +range around the focal "
            "plane. Sweeps OpticsConfig.defocus — a Zernike Z4 term, if set, "
            "adds its own defocus on top of every point."
        )
        form.addRow("Focus ±", self.focus_range)

        self.n_focus = QtWidgets.QSpinBox()
        self.n_focus.setRange(3, 41)
        self.n_focus.setValue(11)
        form.addRow("Focus steps", self.n_focus)

        self.dose_range = QtWidgets.QDoubleSpinBox()
        self.dose_range.setRange(0.05, 0.90)
        self.dose_range.setSingleStep(0.05)
        self.dose_range.setValue(0.30)
        self.dose_range.setDecimals(2)
        self.dose_range.setPrefix("±")
        self.dose_range.setToolTip(
            "Fractional dose variation around the centre dose. Doses are "
            "clear-field normalised for the sweep, so a dose means the same "
            "energy at every focus — unlike the dock's peak-normalised Dose "
            "slider."
        )
        form.addRow("Dose range", self.dose_range)

        self.n_dose = QtWidgets.QSpinBox()
        self.n_dose.setRange(3, 25)
        self.n_dose.setValue(7)
        self.n_dose.setToolTip(
            "Dose steps are nearly free: dose rescales the cached image, "
            "only focus steps pay for an Abbe sum."
        )
        form.addRow("Dose steps", self.n_dose)

        self.target_cd = QtWidgets.QDoubleSpinBox()
        self.target_cd.setRange(10.0, 500.0)
        self.target_cd.setSingleStep(5.0)
        self.target_cd.setValue(float(model["cd"]))
        self.target_cd.setDecimals(0)
        self.target_cd.setSuffix(" nm")
        form.addRow("Target CD", self.target_cd)

        self.tolerance = QtWidgets.QDoubleSpinBox()
        self.tolerance.setRange(1.0, 30.0)
        self.tolerance.setSingleStep(1.0)
        self.tolerance.setValue(10.0)
        self.tolerance.setDecimals(0)
        self.tolerance.setSuffix(" %")
        form.addRow("CD tolerance", self.tolerance)

        self.auto_centre = QtWidgets.QCheckBox("Centre dose on target CD")
        self.auto_centre.setChecked(True)
        self.auto_centre.setToolTip(
            "Calibrate dose-to-size first (one extra image) and centre the "
            "dose axis on the dose that prints the target CD at focus. "
            "Without it the axis centres on clear-normalised dose 1.0, "
            "which may not print at all."
        )
        form.addRow(self.auto_centre)

        self.meef_check = QtWidgets.QCheckBox("Measure MEEF")
        self.meef_check.setChecked(True)
        self.meef_check.setToolTip(
            "Mask Error Enhancement Factor at best focus and best dose, by "
            "central difference over a ±1 px mask bias (two extra images)."
        )
        form.addRow(self.meef_check)

        self.fourth = QtWidgets.QComboBox()
        self.fourth.addItems(["EL vs DOF", "NILS through focus"])
        form.addRow("4th panel", self.fourth)

        self.cost_label = QtWidgets.QLabel("")
        self.cost_label.setStyleSheet("color: #666666;")
        form.addRow(self.cost_label)

        self.run_btn = QtWidgets.QPushButton("Run sweep")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        # Tab-local notes rather than the shared status bar, which the live
        # recompute path overwrites within a slider drag (StackTab precedent).
        self.notes = QtWidgets.QLabel("")
        self.notes.setWordWrap(True)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(box)
        left.addWidget(self.run_btn)
        left.addWidget(self.progress)
        left.addWidget(self.notes)
        left.addStretch(1)
        holder = QtWidgets.QWidget()
        holder.setLayout(left)
        holder.setMaximumWidth(280)
        outer.addWidget(holder)

        # -- right: the figure ------------------------------------------
        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        outer.addWidget(self.canvas, 1)
        self._placeholder("Set the sweep ranges and press Run sweep.")

        self.run_btn.clicked.connect(self._run)
        self.fourth.currentIndexChanged.connect(self._redraw)
        for w in (self.n_focus, self.auto_centre, self.meef_check):
            (w.valueChanged if isinstance(w, QtWidgets.QSpinBox)
             else w.toggled).connect(self._refresh_cost)
        # showEvent keeps these current, but only once the tab has actually
        # been shown — the initial state must not depend on that.
        self._refresh_cost()
        self._refresh_meef_availability()

    # -- presentation ----------------------------------------------------

    def _placeholder(self, text: str) -> None:
        self.figure.clf()
        self.figure.text(0.5, 0.5, text, ha="center", va="center",
                         color="#888888")
        self.canvas.draw_idle()

    def showEvent(self, event):        # noqa: N802 - Qt's spelling
        super().showEvent(event)
        # The dock can change while this tab is hidden; both of these read
        # the model, so bring them up to date at the moment they are seen.
        self._refresh_cost()
        self._refresh_meef_availability()

    def _refresh_meef_availability(self) -> None:
        fdtd = self.model["mask_model"] == "fdtd"
        self.meef_check.setEnabled(not fdtd and not self._busy)
        if fdtd:
            self.meef_check.setToolTip(
                "Unavailable with the FDTD mask model: each biased mask is a "
                "different mask and would trigger a fresh multi-second "
                "near-field solve rather than a cache hit."
            )

    def _refresh_cost(self, *_args) -> None:
        per_image = estimate_cost_ms(self.model)
        units = self.n_focus.value()
        if self.auto_centre.isChecked():
            units += 1
        if self.meef_check.isChecked() and self.meef_check.isEnabled():
            units += 2
        self.cost_label.setText(
            f"≈ {per_image * units / 1000.0:.1f} s  ({units} images)"
        )

    def _set_editing_enabled(self, on: bool) -> None:
        for w in (self.focus_range, self.n_focus, self.dose_range,
                  self.n_dose, self.target_cd, self.tolerance,
                  self.auto_centre, self.meef_check, self.run_btn):
            w.setEnabled(on)
        if on:
            self._refresh_meef_availability()

    # -- running ----------------------------------------------------------

    def _run(self) -> None:
        if self._busy:
            return
        # Read the controls *before* disabling them: a disabled checkbox
        # still answers isChecked(), but isEnabled() is False by then — an
        # ordering that silently turned MEEF off on every run.
        req = FemRequest(
            params=copy.deepcopy(self.model),
            focus_half_range_nm=float(self.focus_range.value()),
            n_focus=int(self.n_focus.value()),
            dose_half_range=float(self.dose_range.value()),
            n_dose=int(self.n_dose.value()),
            target_cd_nm=float(self.target_cd.value()),
            tolerance_pct=float(self.tolerance.value()),
            auto_centre_dose=self.auto_centre.isChecked(),
            with_meef=self.meef_check.isChecked() and self.meef_check.isEnabled(),
        )
        self._busy = True
        self._set_editing_enabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.notes.setText("sweeping…")
        self.run_requested.emit(req)

    @QtCore.Slot(int, int)
    def on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    @QtCore.Slot(object)
    def on_finished(self, result: ProcessWindowResult | None) -> None:
        self._busy = False
        self._set_editing_enabled(True)
        if result is None:
            # The worker already routed the exception text to the status bar.
            self.notes.setText("Sweep failed — see the status bar.")
            return
        self._result = result
        self.progress.setValue(self.progress.maximum())
        self.notes.setText(result.diagnosis or result.summary)
        self._redraw()

    def _redraw(self, *_args) -> None:
        """Rebuild all four panels from the held result.

        Remove-and-rebuild rather than artist mutation: this figure redraws
        once per button press, not once per slider tick, so the hundreds of
        milliseconds that pattern saves elsewhere are not being spent here.
        """
        r = self._result
        if r is None:
            return
        from litho_sim.viz.plots import (
            plot_bossung_curves,
            plot_cd_heatmap,
            plot_el_dof_curve,
            plot_nils_through_focus,
            plot_process_window,
        )

        fig = self.figure
        fig.clf()
        gs = fig.add_gridspec(2, 2)
        ax_bossung = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_pw = fig.add_subplot(gs[1, 0])
        ax_fourth = fig.add_subplot(gs[1, 1])

        prints = r.pw.get("prints", False)
        plot_bossung_curves(
            r.bossung_df, r.target_cd_nm, r.tolerance_pct, ax=ax_bossung,
            best_focus_nm=r.pw["best_focus_nm"] if prints else None,
        )
        plot_cd_heatmap(r.bossung_df, r.target_cd_nm, r.tolerance_pct, ax=ax_heat)

        if prints:
            plot_process_window(r.pw, r.target_cd_nm, ax=ax_pw)
            if self.fourth.currentIndex() == 0:
                plot_el_dof_curve(r.el_dof, ax=ax_fourth)
            else:
                plot_nils_through_focus(
                    r.bossung_df, dose=r.nominal_dose, ax=ax_fourth
                )
        else:
            for ax in (ax_pw, ax_fourth):
                ax.set_axis_off()
            ax_pw.text(0.5, 0.5, r.diagnosis, ha="center", va="center",
                       wrap=True, color="#888888", fontsize=9)

        fig.suptitle(r.label, fontsize=9, color="#555555")
        self.canvas.draw_idle()

