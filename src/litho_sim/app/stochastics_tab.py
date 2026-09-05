"""The stochastics tab: Monte-Carlo printing trials, computed off-thread."""

from __future__ import annotations

import copy
import logging

from litho_sim.app.params import ParameterModel
from litho_sim.app.qt import Figure, FigureCanvasQTAgg, QtCore, QtWidgets
from litho_sim.app.stochastics import StochRequest, StochResult
from litho_sim.viz import theme
from litho_sim.viz.theme import CMAP, MUTED, SERIES

logger = logging.getLogger(__name__)


class StochasticsTab(QtWidgets.QWidget):
    """Monte-Carlo printing over the dock's configuration.

    Runs the chemically amplified chain with the molecule counts sampled —
    Poisson photons, Poisson PAG and quencher, binomial acid conversion —
    once per trial, and reduces the batch to LER/LWR, LCDU, and the
    bridge/break failure rate against the deterministic print. Nothing here
    injects a noise parameter: the wavelength on the dock decides the photon
    count, which is why the same panel is quiet at 193 nm and loud at 13.5.

    Self-contained like :class:`ProcessWindowTab`, and for the same reason:
    trials and seed parameterise this tab alone, and the chemistry itself
    comes from the dock — snapshotted by deep copy at Run-press time so the
    worker never races the sliders. A batch is many reaction–diffusion
    bakes, far past the live budget, so everything sits behind an explicit
    Run button.
    """

    run_requested = QtCore.Signal(object)      # StochRequest

    def __init__(self, model: ParameterModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._busy = False
        self._result: StochResult | None = None

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # -- left: the batch settings ----------------------------------
        box = QtWidgets.QGroupBox("Trials")
        form = QtWidgets.QFormLayout(box)

        self.n_trials = QtWidgets.QSpinBox()
        self.n_trials.setRange(2, 64)
        self.n_trials.setValue(16)
        self.n_trials.setToolTip(
            "Independent realisations. Each pays for a full "
            "reaction–diffusion bake; 16 resolves LCDU to ~20 % and failure "
            "rates to ~1-in-16."
        )
        form.addRow("Trials", self.n_trials)

        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 99_999)
        self.seed.setValue(0)
        self.seed.setToolTip(
            "Base RNG seed. The same seed reproduces the whole batch "
            "bit-for-bit; change it to draw a fresh one."
        )
        form.addRow("Seed", self.seed)

        self.cost_label = QtWidgets.QLabel(
            "≈ (trials + 1) car develops — chemistry from the dock"
        )
        self.cost_label.setStyleSheet("color: #666666;")
        self.cost_label.setWordWrap(True)
        form.addRow(self.cost_label)

        self.run_btn = QtWidgets.QPushButton("Run trials")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.notes = QtWidgets.QLabel(
            "Chemistry, dose and wavelength come from the dock — the "
            "wavelength sets the photon count, which is the whole "
            "difference between ArF and EUV here."
        )
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
        self._placeholder("Set trials and press Run trials.")

        self.run_btn.clicked.connect(self._run)

    # -- presentation ----------------------------------------------------

    def _placeholder(self, text: str) -> None:
        self.figure.clf()
        self.figure.text(0.5, 0.5, text, ha="center", va="center",
                         color=theme.INK2)
        self.canvas.draw_idle()

    def _set_editing_enabled(self, on: bool) -> None:
        for w in (self.n_trials, self.seed, self.run_btn):
            w.setEnabled(on)

    # -- running ----------------------------------------------------------

    def _run(self) -> None:
        if self._busy:
            return
        req = StochRequest(
            params=copy.deepcopy(self.model),
            trials=int(self.n_trials.value()),
            seed=int(self.seed.value()),
        )
        self._busy = True
        self._set_editing_enabled(False)
        self.progress.setRange(0, req.trials + 1)
        self.progress.setValue(0)
        self.notes.setText("printing…")
        self.run_requested.emit(req)

    @QtCore.Slot(int, int)
    def on_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    @QtCore.Slot(object)
    def on_finished(self, result: StochResult | None) -> None:
        self._busy = False
        self._set_editing_enabled(True)
        if result is None:
            # The worker already routed the exception text to the status bar.
            self.notes.setText("Trials failed — see the status bar.")
            return
        self._result = result
        self.progress.setValue(self.progress.maximum())
        self.notes.setText(result.diagnosis or result.summary)
        self._redraw()

    def _redraw(self) -> None:
        """Rebuild all four panels from the held result.

        Remove-and-rebuild, the ProcessWindowTab precedent: one redraw per
        button press, not per slider tick.
        """
        import numpy as np

        r = self._result
        if r is None:
            return

        fig = self.figure
        fig.clf()
        gs = fig.add_gridspec(2, 2)
        ax_ref = fig.add_subplot(gs[0, 0])
        ax_one = fig.add_subplot(gs[0, 1])
        ax_prob = fig.add_subplot(gs[1, 0])
        ax_hist = fig.add_subplot(gs[1, 1])

        extent = (0.0, float(r.x_nm[-1]), 0.0, float(r.x_nm[-1]))
        cmap, norm = theme.binary_cmap(theme.material_colour("photoresist"))
        theme.physical_image(ax_ref, r.reference, extent, cmap, norm=norm)
        theme.title(ax_ref, "Deterministic print", caption_text="design intent")
        theme.physical_image(ax_one, r.example, extent, cmap, norm=norm)
        theme.title(ax_one, "One trial")
        theme.physical_image(ax_prob, r.prob_map, extent, CMAP.probability,
                             vmin=0.0, vmax=1.0, cbar_label="print probability")
        n_trials = int(np.size(r.lcdu.cds))
        theme.title(ax_prob, "Print probability",
                    caption_text=f"over {n_trials} trials")

        cds_nm = r.lcdu.cds[r.lcdu.cds > 0] * 1e9
        if cds_nm.size:
            ax_hist.hist(cds_nm, bins=min(12, max(4, cds_nm.size // 2)),
                         color=SERIES.aerial, edgecolor=theme.SURFACE, linewidth=1.5)
            if np.isfinite(r.lcdu.cd_mean):
                theme.rule(ax_hist, x=r.lcdu.cd_mean * 1e9, color=MUTED, lw=1.0,
                           text=f"mean {r.lcdu.cd_mean * 1e9:.1f} nm")
            ax_hist.set_xlabel("CD [nm]")
            ax_hist.set_ylabel("trials")
            theme.grid(ax_hist)
            lcdu = ("—" if not np.isfinite(r.lcdu.lcdu)
                    else f"{r.lcdu.lcdu * 1e9:.2f} nm")
            theme.title(ax_hist, "CD across trials", caption_text=f"LCDU (3σ) {lcdu}")
        else:
            ax_hist.set_axis_off()
            theme.placeholder(ax_hist, "no trial printed a measurable CD")

        theme.caption(fig, r.label)
        self.canvas.draw_idle()
