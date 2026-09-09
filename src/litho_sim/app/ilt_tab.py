"""The ILT tab: a correction you watch rather than wait for.

Every other computation in this app reports a percentage while it runs and a
picture when it finishes. This one reports pictures the whole way, because the
intermediate states *are* the explanation: the drawn mask goes soft, its edges
find positions no polygon vertex could have reached, and assist features
condense out of empty field where the designer drew nothing at all. A progress
bar would hide precisely the part worth seeing.

Three panels, redrawn on every frame the solver sends:

* **Mask** — the current continuous mask, with its 0.5 level drawn over it:
  the curvilinear outline a writer would cut. Grey wherever the solve is
  still deciding. The *Curvature* setting is what makes that outline a curve
  rather than a staircase: the smallest radius it may turn on.
* **Printed** — the signed resist field, with its zero contour (what prints)
  over the drawn target (what was wanted). The gap between the two lines is
  the correction's remaining work, and it is the same "what printed"
  definition the process window and the OPC tab use.
* **Convergence** — cost and pattern error against iteration.

The redraw is the frame budget, not the solve. A 128-pixel solve produces a
frame every ~66 ms, and matplotlib cannot repaint three axes that fast, so the
canvas is throttled: frames that arrive while a repaint is pending replace the
pending one instead of queueing behind it. Dropping frames keeps the animation
in step with the solve; queueing them would leave it running long after the
answer was in.
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from litho_sim.app.ilt import (
    ILTRequest,
    ILTResultBundle,
    ilt_availability,
    ilt_print_model,
    normalisation_note,
    target_for,
)
from litho_sim.app.params import ParameterModel
from litho_sim.app.qt import Figure, FigureCanvasQTAgg, QtCore, QtWidgets
from litho_sim.opc.ilt import ILTFrame
from litho_sim.viz.theme import CMAP, MUTED, SERIES

logger = logging.getLogger(__name__)

#: Shortest gap between canvas repaints [ms]. The solve outruns matplotlib on
#: any grid worth solving, so this is what the animation actually runs at.
_REPAINT_MS = 60


class ILTTab(QtWidgets.QWidget):
    """Inverse lithography over the dock's configuration, drawn live.

    Self-contained like :class:`ProcessWindowTab` and
    :class:`StochasticsTab`, and for the same reason: the solver settings
    parameterise this tab alone, and the process itself comes from the dock —
    snapshotted by deep copy at Run-press time so the worker never races the
    sliders.
    """

    run_requested = QtCore.Signal(object)      # ILTRequest

    def __init__(self, model: ParameterModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._busy = False
        self._pending: ILTFrame | None = None
        self._result: ILTResultBundle | None = None
        self._target: np.ndarray | None = None
        self._history: list[tuple[int, float, int]] = []

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # -- left: the solver settings ---------------------------------
        box = QtWidgets.QGroupBox("Solve")
        form = QtWidgets.QFormLayout(box)

        self.max_iter = QtWidgets.QSpinBox()
        self.max_iter.setRange(5, 1000)
        self.max_iter.setValue(120)
        self.max_iter.setToolTip(
            "Iteration cap. Each iteration costs about three aerial images — "
            "the forward print and its adjoint."
        )
        form.addRow("Iterations", self.max_iter)

        self.step = QtWidgets.QDoubleSpinBox()
        self.step.setRange(0.001, 1.0)
        self.step.setDecimals(3)
        self.step.setSingleStep(0.005)
        self.step.setValue(0.03)
        self.step.setToolTip(
            "Adam's learning rate. Too large is not a slow solve but a wrong "
            "one: the first step drives the whole mask dark, which costs less "
            "than the initial state and looks exactly like convergence."
        )
        form.addRow("Step", self.step)

        self.beta_final = QtWidgets.QDoubleSpinBox()
        self.beta_final.setRange(1.0, 200.0)
        self.beta_final.setValue(16.0)
        self.beta_final.setToolTip(
            "Mask sharpening at the last iteration, annealed up from 4. Low "
            "early lets grey mask move and find where the assist features "
            "belong; high late drives it toward the two levels a writer makes."
        )
        form.addRow("Final β", self.beta_final)

        self.weight_tv = QtWidgets.QDoubleSpinBox()
        self.weight_tv.setRange(0.0, 1.0)
        self.weight_tv.setDecimals(4)
        self.weight_tv.setSingleStep(0.0005)
        self.weight_tv.setValue(0.001)
        self.weight_tv.setToolTip(
            "Mask-complexity penalty. Zero leaves the solve free to answer "
            "with single-pixel speckle that images as nothing."
        )
        form.addRow("Complexity", self.weight_tv)

        self.smooth = QtWidgets.QDoubleSpinBox()
        self.smooth.setRange(0.0, 100.0)
        self.smooth.setDecimals(0)
        self.smooth.setSingleStep(4.0)
        self.smooth.setSuffix(" nm")
        self.smooth.setValue(24.0)
        self.smooth.setToolTip(
            "The smallest radius the mask outline may turn on. The mask is "
            "drawn through a Gaussian this wide, so nothing in it is finer "
            "than this. Zero leaves every pixel free, and the sharpened "
            "result is a staircase with bumps on it rather than a curve."
        )
        form.addRow("Curvature", self.smooth)

        self.size_dose = QtWidgets.QCheckBox("Size dose to target area")
        self.size_dose.setChecked(True)
        self.size_dose.setToolTip(
            "Size the dose so the drawn mask prints its own area before "
            "solving. Without it a contact array at nominal dose prints "
            "nothing, and a solve against a blank print has no gradient."
        )
        form.addRow(self.size_dose)

        self.run = QtWidgets.QPushButton("Run")
        self.run.clicked.connect(self._on_run)
        form.addRow(self.run)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {MUTED};")
        form.addRow(self.status)

        panel = QtWidgets.QVBoxLayout()
        panel.addWidget(box)
        panel.addStretch(1)
        outer.addLayout(panel, 0)

        # -- right: the three views ------------------------------------
        self.figure = Figure(figsize=(9.5, 3.6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax_mask = self.figure.add_subplot(1, 3, 1)
        self.ax_print = self.figure.add_subplot(1, 3, 2)
        self.ax_conv = self.figure.add_subplot(1, 3, 3)
        outer.addWidget(self.canvas, 1)

        # Repaints are rate-limited to _REPAINT_MS; frames arriving inside a
        # tick replace the pending one rather than queueing behind it.
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(_REPAINT_MS)
        self._timer.timeout.connect(self._repaint)

        self._reset_axes()
        self.refresh_availability()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def refresh_availability(self, keep_status: bool = False) -> None:
        """Enable Run only when the configuration can be solved, and say why not.

        *keep_status* leaves the label alone — the end of a solve has just
        written the result into it, and re-deriving the idle note there would
        wipe the numbers the run was for.
        """
        try:
            reason = ilt_availability(self.model)
        except Exception as exc:                      # noqa: BLE001
            reason = str(exc)
        self.run.setEnabled(reason is None and not self._busy)
        if keep_status:
            return
        if reason is not None:
            self.status.setText(reason)
        elif not self._busy:
            grid = self.model.grid()
            n, field_nm = grid.n_pixels, grid.n_pixels * grid.pixel_size * 1e9
            note = normalisation_note(self.model)
            # No kernel count here: it would take a decomposition to know, and
            # this runs on every control that moves. The rate is measured.
            self.status.setText(
                f"{n}×{n} field of {field_nm:.0f} nm — an iteration is about "
                "three aerial images; the rate is measured, and reported after."
                + ("" if note is None else "\n" + note)
            )

    def _on_run(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.run.setEnabled(False)
        self.status.setText("solving…")
        self._history.clear()
        self._pending = None
        self._target = None
        self._reset_axes()
        try:
            self._target = target_for(self.model, ilt_print_model(self.model))
        except Exception:                             # noqa: BLE001
            self._target = None                       # the solve will say why
        self._timer.start()
        # Deep copy: the solve reads this on the worker thread while the dock
        # stays live under the user's hands.
        self.run_requested.emit(ILTRequest(
            params=copy.deepcopy(self.model),
            max_iter=int(self.max_iter.value()),
            step=float(self.step.value()),
            beta_final=float(self.beta_final.value()),
            weight_tv=float(self.weight_tv.value()),
            smooth_nm=float(self.smooth.value()),
            size_dose=self.size_dose.isChecked(),
        ))

    @QtCore.Slot(object)
    def on_frame(self, frame: ILTFrame) -> None:
        """One solver iteration. Stored, not drawn — the timer draws."""
        self._pending = frame
        self._history.append((frame.iteration, frame.cost, frame.pattern_error))

    @QtCore.Slot(object)
    def on_done(self, bundle: ILTResultBundle | None) -> None:
        self._busy = False
        self._timer.stop()
        self._repaint()                     # whatever the last frame was
        self._result = bundle
        if bundle is None:
            self.status.setText("the solve failed — see the log")
        else:
            self.status.setText(
                f"{bundle.result.iterations} iterations in {bundle.seconds:.1f} s "
                f"({1e3 * bundle.seconds / max(bundle.result.iterations, 1):.0f} ms each)\n"
                f"pattern error {bundle.pattern_error_before} → "
                f"{bundle.pattern_error_after} "
                f"({bundle.pattern_error_binary} thresholded, at dose "
                f"{bundle.dose_binary:.3f})"
            )
        # Either branch above wrote something worth keeping — the result, or
        # the reason there isn't one.
        self.refresh_availability(keep_status=True)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _reset_axes(self) -> None:
        for ax, title in (
            (self.ax_mask, "Mask"),
            (self.ax_print, "Printed"),
            (self.ax_conv, "Convergence"),
        ):
            ax.clear()
            ax.set_title(title, fontsize=10)
        for ax in (self.ax_mask, self.ax_print):
            ax.set_xticks([])
            ax.set_yticks([])
        self.canvas.draw_idle()

    def set_target(self, target: np.ndarray) -> None:
        """The design the printed contour is drawn against."""
        self._target = np.asarray(target, dtype=np.float64)

    def _repaint(self) -> None:
        frame = self._pending
        if frame is None:
            return
        self._pending = None

        self.ax_mask.clear()
        self.ax_mask.imshow(frame.mask, origin="lower", cmap=CMAP.intensity, vmin=0.0, vmax=1.0)
        # The shape a writer would cut: the mask's 0.5 level. This is the
        # curvilinear outline, and what the thresholded pattern error is for.
        self.ax_mask.contour(frame.mask, levels=[0.5], colors="w", linewidths=0.7)
        self.ax_mask.set_title(f"Mask — iteration {frame.iteration}", fontsize=10)

        self.ax_print.clear()
        lim = float(np.max(np.abs(frame.signed))) or 1.0
        self.ax_print.imshow(
            frame.signed, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim
        )
        # The printed edge, by the engine's one definition of what printed.
        self.ax_print.contour(frame.signed, levels=[0.0], colors="k", linewidths=1.3)
        if self._target is not None:
            self.ax_print.contour(
                self._target, levels=[0.5], colors=SERIES.warn,
                linewidths=1.0, linestyles="--",
            )
        self.ax_print.set_title(
            f"Printed — {frame.pattern_error} px wrong", fontsize=10
        )

        for ax in (self.ax_mask, self.ax_print):
            ax.set_xticks([])
            ax.set_yticks([])

        if self._history:
            it = [h[0] for h in self._history]
            cost = [h[1] for h in self._history]
            perr = [h[2] for h in self._history]
            self.ax_conv.clear()
            self.ax_conv.plot(it, cost, color=SERIES.aerial, lw=1.4)
            self.ax_conv.set_xlabel("iteration", fontsize=8)
            self.ax_conv.set_ylabel("cost", color=SERIES.aerial, fontsize=8)
            self.ax_conv.tick_params(labelsize=7)
            twin = getattr(self, "_twin", None)
            if twin is None:
                twin = self._twin = self.ax_conv.twinx()
            twin.clear()
            twin.plot(it, perr, color=SERIES.latent, lw=1.2, ls=":")
            twin.yaxis.set_label_position("right")
            twin.set_ylabel("pattern error [px]", color=SERIES.latent, fontsize=8)
            twin.tick_params(labelsize=7)
            self.ax_conv.set_title("Convergence", fontsize=10)

        self.canvas.draw_idle()


__all__ = ["ILTTab"]
