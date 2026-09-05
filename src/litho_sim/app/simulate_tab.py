"""The Simulate tab: every computation the engine can run, in one list."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from litho_sim.app.compute import (
    ImagingResult,
    Profile3DResult,
    estimate_cost_ms,
)
from litho_sim.app.params import ParameterModel
from litho_sim.app.process_window_tab import ProcessWindowTab
from litho_sim.app.qt import Figure, FigureCanvasQTAgg, QtCore, QtWidgets
from litho_sim.app.stochastics_tab import StochasticsTab
from litho_sim.viz import theme
from litho_sim.viz.theme import CMAP, SERIES

logger = logging.getLogger(__name__)

#: The runs, in the order they are offered. Each is a page below.
KINDS: tuple[str, ...] = (
    "Print",
    "3-D resist profile",
    "Focus-exposure matrix",
    "Stochastic printing",
)

BLURBS: dict[str, str] = {
    "Print": (
        "Mask → aerial image → bake → develop, in 2-D. The result lands on "
        "the Expose, Bake and Develop tabs; the overview is here."
    ),
    "3-D resist profile": (
        "The same print resolved through the film: one Abbe sum per optical "
        "plane, absorption and standing waves, then a depth develop. The solid "
        "lands on the Develop tab in 3-D."
    ),
    "Focus-exposure matrix": (
        "Sweep dose and focus over the current settings: Bossung curves, the "
        "CD map, the process window, EL vs DOF or NILS through focus, MEEF."
    ),
    "Stochastic printing": (
        "Print the current settings many times with sampled photon and "
        "molecule counts: print probability, LER/LWR, LCDU and failure rate."
    ),
}

#: What a stale page says. The settings moved after this ran; the picture is
#: of the old ones.
STALE_NOTE = "Settings have changed since this ran — press Run again."


class _RunPage(QtWidgets.QWidget):
    """A settings column with a Run button, beside a figure.

    The two batch tabs (process window, stochastics) already have this shape
    and are reused as pages unchanged; this is the same shape for the two
    runs that carry no settings of their own beyond the step tabs.
    """

    run_requested = QtCore.Signal()

    def __init__(self, run_label: str, placeholder: str, parent=None):
        super().__init__(parent)
        self._busy = False
        self._stale = False

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        box = QtWidgets.QGroupBox("Run")
        form = QtWidgets.QVBoxLayout(box)
        self.cost_label = QtWidgets.QLabel("")
        self.cost_label.setStyleSheet("color: #666666;")
        self.cost_label.setWordWrap(True)
        form.addWidget(self.cost_label)
        self.run_btn = QtWidgets.QPushButton(run_label)
        form.addWidget(self.run_btn)
        self.notes = QtWidgets.QLabel("")
        self.notes.setWordWrap(True)
        form.addWidget(self.notes)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(box)
        left.addStretch(1)
        holder = QtWidgets.QWidget()
        holder.setLayout(left)
        holder.setMaximumWidth(280)
        holder.setMinimumWidth(220)
        outer.addWidget(holder)

        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        outer.addWidget(self.canvas, 1)
        self._placeholder(placeholder)

        self.run_btn.clicked.connect(self._on_run)

    def _placeholder(self, text: str) -> None:
        self.figure.clf()
        self.figure.text(0.5, 0.5, text, ha="center", va="center",
                         color=theme.INK2, wrap=True)
        self.canvas.draw_idle()

    @property
    def busy(self) -> bool:
        return self._busy

    def _on_run(self) -> None:
        if self._busy:
            return
        self.set_busy(True)
        self.run_requested.emit()

    def set_busy(self, on: bool) -> None:
        self._busy = on
        self.run_btn.setEnabled(not on)
        if on:
            self.notes.setText("running…")

    #: The last result's one-line summary, restored when a stale note clears.
    _summary: str = ""

    def set_stale(self, stale: bool) -> None:
        """Note that the settings moved since this page's result was made."""
        self._stale = stale
        if self._busy:
            return
        self.notes.setText(STALE_NOTE if stale else self._summary)

    @property
    def stale(self) -> bool:
        return self._stale


class PrintPage(_RunPage):
    """The 2-D print: mask, aerial image and developed resist side by side,
    with the CD, NILS and contrast under them."""

    def __init__(self, parent=None):
        super().__init__(
            "Run Print",
            "Set up the steps on their tabs, then press Run Print.",
            parent,
        )

    def refresh_cost(self, model: ParameterModel) -> None:
        ms = estimate_cost_ms(model)
        text = f"≈ {ms / 1000.0:.2f} s" if ms >= 1000 else f"≈ {ms:.0f} ms"
        self.cost_label.setText(
            f"{text} for one image at these settings. Less when only a "
            f"develop or bake setting changed — the aerial image is cached."
        )

    def show_result(self, r: ImagingResult) -> None:
        fig = self.figure
        fig.clf()
        gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1.0])
        px = float(r.x_nm[1] - r.x_nm[0]) if len(r.x_nm) > 1 else 1.0
        h, w = r.aerial.shape[:2]
        extent = (0.0, w * px, 0.0, h * px)
        resist = theme.material_colour("photoresist")
        resist_cmap, resist_norm = theme.binary_cmap(resist)
        panels: tuple[tuple[Any, ...], ...] = (
            (r.mask, CMAP.micrograph, "Mask", None,
             {"vmin": 0.0, "vmax": 1.0}),
            (r.aerial, CMAP.intensity, "Aerial image", f"NILS {r.nils:.2f}",
             {"vmin": 0.0, "cbar_label": "intensity [a.u.]"}),
            (r.resist, resist_cmap, "Developed resist", f"CD {r.cd_text}",
             {"norm": resist_norm}),
        )
        for col, (data, cmap, title, cap, kw) in enumerate(panels):
            ax = fig.add_subplot(gs[0, col])
            theme.physical_image(ax, data, extent, cmap, **kw)
            theme.title(ax, title, caption_text=cap)
        # The cut through all three, which is where the CD actually comes
        # from: the latent crossing the threshold is the printed edge.
        ax = fig.add_subplot(gs[1, :])
        top = max(float(r.cut_latent.max()), float(r.cut_aerial.max()), 1.0) * 1.2
        ax.fill_between(r.x_nm, 0.0, np.asarray(r.cut_resist, dtype=float) * top,
                        color=resist, alpha=0.18, lw=0, label="resist remains")
        ax.plot(r.x_nm, r.cut_aerial, color=SERIES.aerial, lw=1.2, alpha=0.6,
                label="aerial")
        ax.plot(r.x_nm, r.cut_latent, color=SERIES.latent,
                label=f"latent ({r.latent_kind})")
        theme.rule(ax, y=r.threshold, color=SERIES.threshold, ls="--", lw=1.0,
                   label="threshold")
        ax.set_xlim(float(r.x_nm[0]), float(r.x_nm[-1]))
        ax.set_ylim(0.0, top)
        ax.set_xlabel("x [nm]")
        theme.grid(ax)
        theme.title(ax, "Cut through the centre row")
        theme.legend(ax, where="top")
        theme.caption(fig, r.summary)
        self.canvas.draw_idle()
        self._summary = r.summary
        self.notes.setText(r.summary)


class ProfilePage(_RunPage):
    """The depth-resolved print: the profile through the film and the latent
    image it developed from. The rotatable solid is on the Develop tab."""

    def __init__(self, parent=None):
        super().__init__(
            "Run 3-D profile",
            "Press Run 3-D profile. The solid appears on the Develop tab "
            "in 3-D; the cross-section appears here.",
            parent,
        )

    def refresh_cost(self, model: ParameterModel) -> None:
        planes = int(model["n_z_slices"])
        ms = estimate_cost_ms(model) * planes
        text = f"≈ {ms / 1000.0:.1f} s" if ms >= 1000 else f"≈ {ms:.0f} ms"
        self.cost_label.setText(
            f"{text}: one Abbe sum per optical plane, {planes} planes. "
            f"Milliseconds when only a develop setting changed — the latent "
            f"image is cached."
        )

    def show_profile(self, r: Profile3DResult, grid) -> None:
        fig = self.figure
        fig.clf()
        ax_prof, ax_lat = fig.subplots(2, 1)
        width = r.cut_remaining.shape[1] * grid.pixel_size * 1e9
        extent = (0.0, width, 0.0, float(r.height_nm))
        cmap, norm = theme.binary_cmap(theme.material_colour("photoresist"))
        theme.physical_image(
            ax_prof, np.asarray(r.cut_remaining, dtype=np.uint8), extent, cmap,
            norm=norm, aspect="auto", ylabel="z [nm]",
        )
        theme.title(ax_prof, "Profile through the film",
                    caption_text=r.summary.replace("    ", " · "))
        if r.diagnosis:
            ax_prof.text(0.5, 0.5, r.diagnosis, transform=ax_prof.transAxes,
                         ha="center", va="center", fontsize=8.5,
                         color=SERIES.warn, wrap=True)
        theme.physical_image(
            ax_lat, r.cut_latent, extent, CMAP.latent, aspect="auto",
            ylabel="z [nm]", cbar_label="PAC after bake",
        )
        theme.title(ax_lat, "Latent image after bake")
        theme.caption(fig, r.label)
        self.canvas.draw_idle()
        self._summary = r.summary
        self.notes.setText(r.summary)


class SimulateTab(QtWidgets.QWidget):
    """One place to run things.

    Nothing on the step tabs computes on its own — a control there changes a
    setting, and this tab is where the setting is turned into a result. The
    list at the top is every run the engine offers; each page below carries
    that run's own settings (sweep ranges, trial counts) and its Run button,
    and reads everything else from the step tabs at the moment Run is
    pressed, by deep copy, so a run describes the settings that were showing
    when it started.
    """

    run_print = QtCore.Signal(object)      # ParameterModel, a snapshot
    run_profile = QtCore.Signal(object)    # ParameterModel, a snapshot
    run_fem = QtCore.Signal(object)        # FemRequest
    run_stoch = QtCore.Signal(object)      # StochRequest

    def __init__(self, model: ParameterModel, parent=None):
        super().__init__(parent)
        self.model = model

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Simulation:"))
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(list(KINDS))
        self.kind.setMinimumWidth(220)
        top.addWidget(self.kind)
        self.blurb = QtWidgets.QLabel("")
        self.blurb.setWordWrap(True)
        self.blurb.setStyleSheet("color: #555555;")
        top.addWidget(self.blurb, 1)
        outer.addLayout(top)

        self.pages = QtWidgets.QStackedWidget()
        self.print_page = PrintPage()
        self.profile_page = ProfilePage()
        self.pw_tab = ProcessWindowTab(model)
        self.stoch_tab = StochasticsTab(model)
        for page in (self.print_page, self.profile_page, self.pw_tab, self.stoch_tab):
            self.pages.addWidget(page)
        outer.addWidget(self.pages, 1)

        self.kind.currentIndexChanged.connect(self._on_kind)
        self.print_page.run_requested.connect(self._run_print)
        self.profile_page.run_requested.connect(self._run_profile)
        self.pw_tab.run_requested.connect(self.run_fem)
        self.stoch_tab.run_requested.connect(self.run_stoch)
        self._on_kind(0)
        self.refresh_costs()

    # -- the list ---------------------------------------------------------
    def _on_kind(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.blurb.setText(BLURBS[KINDS[index]])

    def select(self, kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown run {kind!r}; expected one of {KINDS}")
        self.kind.setCurrentIndex(KINDS.index(kind))

    @property
    def current_kind(self) -> str:
        return KINDS[self.kind.currentIndex()]

    def run_current(self) -> None:
        """Press the Run button of whichever page is showing."""
        page = self.pages.currentWidget()
        page.run_btn.click()

    def showEvent(self, event):        # noqa: N802 - Qt's spelling
        super().showEvent(event)
        self.refresh_costs()

    def refresh_costs(self) -> None:
        self.print_page.refresh_cost(self.model)
        self.profile_page.refresh_cost(self.model)

    # -- print ------------------------------------------------------------
    def _run_print(self) -> None:
        self.run_print.emit(copy.deepcopy(self.model))

    def on_print_done(self, result: ImagingResult) -> None:
        self.print_page.set_busy(False)
        self.print_page.show_result(result)

    # -- 3-D --------------------------------------------------------------
    def _run_profile(self) -> None:
        self.run_profile.emit(copy.deepcopy(self.model))

    def on_profile_done(self, profile: Profile3DResult, grid) -> None:
        self.profile_page.set_busy(False)
        self.profile_page.show_profile(profile, grid)

    # -- shared -----------------------------------------------------------
    def on_failed(self, message: str) -> None:
        """A run failed. Release whichever page was waiting on it."""
        for page in (self.print_page, self.profile_page):
            if page.busy:
                page.set_busy(False)
                page.notes.setText(f"Failed — {message}")

    def set_stale(self, print_stale: bool, profile_stale: bool) -> None:
        self.print_page.set_stale(print_stale)
        self.profile_page.set_stale(profile_stale)
