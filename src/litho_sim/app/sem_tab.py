"""The SEM tab: what an inspection tool would see of the printed resist."""

from __future__ import annotations

import logging

import numpy as np

from litho_sim.app.compute import ImagingResult, Profile3DResult
from litho_sim.app.controls import SpecForm
from litho_sim.app.params import ParamSpec
from litho_sim.app.qt import Figure, FigureCanvasQTAgg, QtCore, QtWidgets
from litho_sim.develop import measure_cd_2d
from litho_sim.metrology import (
    SEMConfig,
    SEMImage,
    SEMMeasurement,
    measure_cd_sem,
    stack_topdown_sem,
    stack_xsection_sem,
    top_surface,
    topdown_sem,
    xsection_sem,
)

logger = logging.getLogger(__name__)

#: The instrument's knobs. Tab-local, like the sweep ranges on the process
#: window: they describe the microscope, not the process, and must not enter
#: the imaging cache keys. ``target``/``stage`` are placeholders — nothing
#: assembles these into an engine config.
SEM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("beam_fwhm", "Beam FWHM", "float", 3.0, 0.5, 12.0, 0.5, "nm", 1e-9,
              group="SEM", target="view", stage="view",
              help="Probe diameter. Blurs the whole image — material contrast "
                   "and edge bloom alike."),
    ParamSpec("escape_length", "Edge escape", "float", 2.0, 0.0, 10.0, 0.5, "nm",
              1e-9, group="SEM", target="view", stage="view",
              help="How far secondary electrons travel out of a sidewall. "
                   "Widens only the bright edges, not the material contrast."),
    ParamSpec("edge_yield", "Edge yield", "float", 3.0, 0.0, 8.0, 0.25,
              group="SEM", target="view", stage="view",
              help="Secondary electrons from a full-height sidewall, relative "
                   "to the flat resist top. How bright edges are."),
    ParamSpec("substrate_yield", "Substrate yield", "float", 0.6, 0.0, 1.5, 0.05,
              group="SEM", target="view", stage="view",
              help="Yield of the exposed substrate relative to the resist. "
                   "Below 1 the developed floor reads dark.\n\n"
                   "On a wafer stack this is silicon's yield; every other "
                   "material keeps the library's."),
    ParamSpec("electrons_per_pixel", "Electrons / px", "int", 100, 5, 2000, 5,
              group="SEM", target="view", stage="view",
              help="Primary electrons per pixel per frame. Shot noise goes as "
                   "one over the square root."),
    ParamSpec("frames", "Frames", "int", 4, 1, 64, 1,
              group="SEM", target="view", stage="view",
              help="Frames averaged. A CD-SEM trades frames against resist "
                   "shrinkage; here it only trades against noise."),
    ParamSpec("seed", "Seed", "int", 0, 0, 999, 1,
              group="SEM", target="view", stage="view",
              help="Noise seed. The same seed gives the same grain."),
)

#: Rows the cross-section of a 2-D print is rasterised onto. The 2-D path
#: keeps a thickness per column rather than a volume; this is the volume it
#: implies, at a resolution fine enough that the sidewall slope survives.
_XSECTION_ROWS = 48

NO_PRINT = "Run Print on the Simulate tab, then come back to image it."
NO_PROFILE = "Run 3-D resist profile on the Simulate tab to image the volume."
NO_STACK = (
    "Load a device or build a flow on the Wafer Stack tab, then come back "
    "to image it."
)

#: The source combo's rows, in order.
SOURCES = ("2-D print", "3-D profile", "Wafer stack")

#: The cut combo's rows: ``(label, Stack.cross_section axis, axis across)``.
#: ``axis="y"`` cuts at constant y and shows the x–z plane.
CUTS = (("x–z plane", "y", "x"), ("y–z plane", "x", "y"))


class SemTab(QtWidgets.QWidget):
    """A CD-SEM pointed at the last print — or at the wafer being built.

    Top-down or cross-section, from the 2-D print, the 3-D profile, or the
    wafer stack. Live — the instrument settings redraw the image as they
    move, because forming an SEM image is a few milliseconds of numpy over a
    result already in hand and the physics that produced the result is
    untouched. The readout measures the CD off the image the way a tool
    does, from the edge peaks, beside the CD the simulation reported: the
    difference is the measurement bias the edge bloom introduces, and
    watching it move with the beam width is the point of the tab.

    The wafer stack is the third source, and the one with a cut to choose:
    a device is interesting in more than one plane — along a nanosheet's
    channel and across it — so the cleave has a plane and a position, where
    the prints have only the cut the profile already carries.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._print: ImagingResult | None = None
        self._profile: tuple[Profile3DResult, object] | None = None
        self._stack: tuple[object, str] | None = None
        self._print_stale = False
        self._profile_stale = False
        self.last: SEMImage | None = None
        self.measurement: SEMMeasurement | None = None

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # -- left: the instrument and the view ---------------------------
        left = QtWidgets.QVBoxLayout()
        view_box = QtWidgets.QGroupBox("Image")
        view_form = QtWidgets.QFormLayout(view_box)
        self.source = QtWidgets.QComboBox()
        self.source.addItems(list(SOURCES))
        self.source.setToolTip(
            "Which result to image. The 2-D print carries a thickness per "
            "column; the 3-D profile carries the developed volume, so its "
            "cross-section can show an undercut the 2-D one cannot.\n\n"
            "The wafer stack is whatever the Wafer Stack tab is showing — a "
            "loaded device at the step you scrubbed to, or the flow you are "
            "building — imaged with each material's own yield."
        )
        view_form.addRow("Result", self.source)
        modes = QtWidgets.QHBoxLayout()
        self.mode_top = QtWidgets.QRadioButton("top-down")
        self.mode_xs = QtWidgets.QRadioButton("cross-section")
        self.mode_top.setChecked(True)
        modes.addWidget(self.mode_top)
        modes.addWidget(self.mode_xs)
        modes.addStretch(1)
        view_form.addRow("View", modes)

        # The cleave. Only the wafer stack has a choice to make here — a
        # print's cross-section is the one cut its profile carries — so the
        # row greys out for the other two sources rather than hiding, which
        # would make the form jump.
        cut = QtWidgets.QHBoxLayout()
        self.cut_axis = QtWidgets.QComboBox()
        self.cut_axis.addItems([c[0] for c in CUTS])
        self.cut_axis.setToolTip(
            "Which plane the cleave exposes. x–z cuts at constant y; y–z at "
            "constant x.\n\n"
            "On the GAA nanosheet the fin runs along x, so x–z at the centre "
            "looks along the channel — source, spacers, gate, drain — and "
            "y–z at the centre looks across it, through the gate: the "
            "sheets stacked with the metal wrapped round them."
        )
        cut.addWidget(self.cut_axis)
        self.cut_pos = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.cut_pos.setRange(0, 100)
        self.cut_pos.setValue(50)
        self.cut_pos.setToolTip(
            "Where across the field the cleave lands, as a fraction of the "
            "field. The centre is where a device preset puts its feature."
        )
        cut.addWidget(self.cut_pos, 1)
        self.cut_label = QtWidgets.QLabel("50 %")
        self.cut_label.setMinimumWidth(40)
        cut.addWidget(self.cut_label)
        self.cut_row = QtWidgets.QWidget()
        self.cut_row.setLayout(cut)
        cut.setContentsMargins(0, 0, 0, 0)
        view_form.addRow("Cut", self.cut_row)

        self.feature = QtWidgets.QComboBox()
        self.feature.addItems(["line", "space"])
        self.feature.setToolTip(
            "What to measure between the edge peaks: a line (bright resist "
            "interior) or a space (dark substrate interior)."
        )
        view_form.addRow("Measure", self.feature)
        left.addWidget(view_box)

        inst_box = QtWidgets.QGroupBox("Instrument")
        inst_layout = QtWidgets.QVBoxLayout(inst_box)
        inst_layout.setContentsMargins(4, 4, 4, 4)
        self.form = SpecForm()
        self.form.set_specs(SEM_SPECS, {s.key: s.default for s in SEM_SPECS})
        inst_layout.addWidget(self.form)
        left.addWidget(inst_box)

        self.notes = QtWidgets.QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color: #666666;")
        left.addWidget(self.notes)
        left.addStretch(1)
        holder = QtWidgets.QWidget()
        holder.setLayout(left)
        holder.setMinimumWidth(300)
        holder.setMaximumWidth(340)
        outer.addWidget(holder)

        # -- right: the micrograph and its profile ------------------------
        right = QtWidgets.QVBoxLayout()
        self.banner = QtWidgets.QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        self.banner.setStyleSheet(
            "background: #fff3d6; color: #7a4a00; border: 1px solid #e0b860; "
            "border-radius: 3px; padding: 4px 8px;"
        )
        right.addWidget(self.banner)
        self.figure = Figure(figsize=(8, 7), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        right.addWidget(self.canvas, 1)
        self.readout = QtWidgets.QLabel("")
        self.readout.setStyleSheet("font-family: monospace;")
        right.addWidget(self.readout)
        outer.addLayout(right, 1)

        self.form.changed.connect(lambda _k: self.render())
        self.source.currentIndexChanged.connect(self._on_source_changed)
        self.feature.currentIndexChanged.connect(lambda _i: self.render())
        self.mode_top.toggled.connect(lambda _c: self.render())
        self.cut_axis.currentIndexChanged.connect(lambda _i: self.render())
        self.cut_pos.valueChanged.connect(self._on_cut_moved)
        self._sync_cut_controls()
        self._placeholder(NO_PRINT)

    # -- inputs -------------------------------------------------------------
    def set_print(self, result: ImagingResult) -> None:
        self._print = result
        self._print_stale = False
        self.render()

    def set_profile(self, profile: Profile3DResult, grid) -> None:
        self._profile = (profile, grid)
        self._profile_stale = False
        self.render()

    def set_stack(self, stack, label: str = "") -> None:
        """The wafer the Wafer Stack tab is showing, whole.

        Arrives on every scrub of that tab's step slider, so only the stack
        source redraws for it — the prints are not what moved.
        """
        self._stack = (stack, str(label))
        if self.from_stack:
            self.render()

    def set_stale(self, print_stale: bool, profile_stale: bool) -> None:
        self._print_stale = print_stale
        self._profile_stale = profile_stale
        self._refresh_banner()

    def config(self) -> SEMConfig:
        v = self.form.values()
        return SEMConfig(
            beam_fwhm=float(v["beam_fwhm"]) * 1e-9,
            escape_length=float(v["escape_length"]) * 1e-9,
            edge_yield=float(v["edge_yield"]),
            substrate_yield=float(v["substrate_yield"]),
            electrons_per_pixel=float(v["electrons_per_pixel"]),
            frames=int(v["frames"]),
            seed=int(v["seed"]),
        )

    @property
    def mode(self) -> str:
        return "topdown" if self.mode_top.isChecked() else "xsection"

    @property
    def from_profile(self) -> bool:
        return self.source.currentIndex() == 1

    @property
    def from_stack(self) -> bool:
        return self.source.currentIndex() == 2

    @property
    def cut(self) -> tuple[str, int]:
        """``(axis, index)`` for :meth:`Stack.cross_section` on the stack in hand."""
        _label, axis, _across = CUTS[self.cut_axis.currentIndex()]
        stack = self._stack[0] if self._stack is not None else None
        if stack is None:
            return axis, 0
        ny, nx = stack.shape_xy
        n = ny if axis == "y" else nx
        return axis, int(round(self.cut_pos.value() / 100.0 * (n - 1)))

    # -- presentation -------------------------------------------------------
    def _on_source_changed(self, _index: int) -> None:
        self._sync_cut_controls()
        self.render()

    def _on_cut_moved(self, value: int) -> None:
        self.cut_label.setText(f"{int(value)} %")
        self.render()

    def _sync_cut_controls(self) -> None:
        self.cut_row.setEnabled(self.from_stack)

    def _missing(self) -> str:
        if self.from_stack:
            return NO_STACK
        return NO_PROFILE if self.from_profile else NO_PRINT

    def _placeholder(self, text: str) -> None:
        self.figure.clf()
        self.figure.text(0.5, 0.5, text, ha="center", va="center",
                         color="#888888", wrap=True)
        self.canvas.draw_idle()
        self.readout.setText("")
        self.last = None
        self.measurement = None

    def _refresh_banner(self) -> None:
        # The stack is never stale: it is whatever is on the Wafer Stack tab
        # right now, and that tab dates itself.
        if self.from_stack:
            text = ""
        else:
            stale = self._profile_stale if self.from_profile else self._print_stale
            has = self._profile is not None if self.from_profile else self._print is not None
            text = (
                "Imaging a result that is out of date — the settings moved after "
                "it was computed. Run it again on the Simulate tab."
                if has and stale else ""
            )
        self.banner.setText(text)
        self.banner.setVisible(bool(text))

    def _topdown_input(self):
        """``(height, pixel_size, film)`` for the chosen resist result."""
        if self.from_profile:
            if self._profile is None:
                return None
            r, grid = self._profile
            height = top_surface(r.remaining, grid.dz)
            return height, float(grid.pixel_size), float(r.height_nm) * 1e-9
        if self._print is None:
            return None
        r = self._print
        px = float(r.x_nm[1] - r.x_nm[0]) * 1e-9 if r.x_nm.size > 1 else 1e-9
        return r.thickness_nm * 1e-9, px, float(r.film_nm) * 1e-9

    @staticmethod
    def _half_height_cd(height, px: float, film: float, feature: str) -> float:
        """The CD of the height map itself, at half the film height.

        The number the SEM CD is compared against. It has to be measured on
        the *same* surface the beam scans: the 2-D print also reports a CD
        from the threshold crossing of the latent image, which for a sloped
        profile is a different width from any height on the slope, and
        comparing the two would show a "bias" that is really two
        definitions. Half height is the convention a CD-SEM's own
        calibration uses.
        """
        cd = measure_cd_2d(
            height, px, threshold=0.5 * film,
            feature="above" if feature == "line" else "below",
        )
        return float(cd) * 1e9

    def _xsection_input(self):
        """``(slice, pixel_size, dz)`` for the chosen resist result."""
        if self.from_profile:
            if self._profile is None:
                return None
            r, grid = self._profile
            return r.cut_remaining, float(grid.pixel_size), float(grid.dz)
        if self._print is None:
            return None
        r = self._print
        px = float(r.x_nm[1] - r.x_nm[0]) * 1e-9 if r.x_nm.size > 1 else 1e-9
        film = float(r.film_nm) * 1e-9
        dz = film / _XSECTION_ROWS
        z = (np.arange(_XSECTION_ROWS) + 0.5) * dz
        # A column is resist up to its remaining thickness — the volume the
        # 2-D profile panel already draws, made explicit.
        slab = z[:, None] < (r.cut_thickness * 1e-9)[None, :]
        return slab, px, dz

    def render(self) -> None:
        """Form the image from the chosen result and the instrument settings."""
        cfg = self.config()
        self._refresh_banner()
        if self.from_stack:
            self._render_stack(cfg)
            return
        src = "3-D profile" if self.from_profile else "2-D print"
        if self.mode == "topdown":
            inp = self._topdown_input()
            if inp is None:
                self._placeholder(self._missing())
                return
            height, px, film = inp
            feature = self.feature.currentText()
            sem = topdown_sem(height, px, cfg, film_thickness=film)
            meas = measure_cd_sem(sem.image, px, feature=feature)
            self._draw_topdown(
                sem, meas, self._half_height_cd(height, px, film, feature), src,
            )
        else:
            inp = self._xsection_input()
            if inp is None:
                self._placeholder(self._missing())
                return
            slab, px, dz = inp
            sem = xsection_sem(slab, px, dz, cfg)
            meas = None
            self._draw_xsection(sem, src, across="x", floor="film bottom")
        self.last = sem
        self.measurement = meas

    def _render_stack(self, cfg: SEMConfig) -> None:
        """The wafer stack: material contrast from the library, silicon
        from the knob, the cleave from the Cut row."""
        if self._stack is None:
            self._placeholder(self._missing())
            return
        stack, label = self._stack
        src = f"wafer stack — {label}" if label else "wafer stack"
        # The one knob that names a material maps onto the one material a
        # wafer always has; the rest read their yield off the library.
        overrides = {"Si": cfg.substrate_yield}
        if self.mode == "topdown":
            feature = self.feature.currentText()
            sem = stack_topdown_sem(stack, cfg, overrides=overrides)
            meas = measure_cd_sem(sem.image, sem.pixel_size, feature=feature)
            self._draw_topdown(sem, meas, None, src)
            self.notes.setText(
                "Looking down on the wafer as it stands: each column shows "
                "the yield of whatever is on top, and the walls between "
                "columns bloom. A planarised wafer has no walls and images "
                "as material contrast alone."
            )
        else:
            axis, index = self.cut
            plane, _axis, across = CUTS[self.cut_axis.currentIndex()]
            pos_nm = index * float(stack.grid.pixel_size) * 1e9
            sem = stack_xsection_sem(stack, axis, index, cfg, overrides=overrides)
            meas = None
            self._draw_xsection(
                sem, f"{src}, {plane} at {axis} = {pos_nm:.0f} nm",
                across=across, floor="substrate bottom",
            )
            self.notes.setText(
                "The cleaved face of the wafer stack. Each material sits at "
                "its own grey — oxides and nitrides bright, silicon dark, "
                "the metal between — and the outline of the solid blooms. "
                "A buried interface is a step in grey and nothing more: on "
                "a flat face it has no sidewall to emit from."
            )
        self.last = sem
        self.measurement = meas

    def _draw_topdown(self, sem: SEMImage, meas: SEMMeasurement,
                      sim_cd_nm: float | None, src: str) -> None:
        fig = self.figure
        fig.clf()
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2])
        ax = fig.add_subplot(gs[0])
        ax_p = fig.add_subplot(gs[1])

        x0, x1, y0, y1 = sem.extent_nm
        vmax = float(np.percentile(sem.image, 99.5))
        ax.imshow(sem.image, origin="lower", cmap="gray", extent=(x0, x1, y0, y1),
                  vmin=0.0, vmax=max(vmax, 1e-6), interpolation="nearest")
        ax.set_xlabel("x [nm]")
        ax.set_ylabel("y [nm]")
        ax.set_title(
            f"top-down SEM of the {src} — {sem.electrons:.0f} e⁻/px, "
            f"SNR {sem.snr:.1f}",
            fontsize=10,
        )

        xs = meas.x * 1e9
        ax_p.plot(xs, meas.profile, color="#c8913a", lw=1.4, label="row-averaged signal")
        if meas.edges.size:
            ax_p.plot(meas.edges * 1e9, np.interp(meas.edges, meas.x, meas.profile),
                      "v", color="#7f9fd9", ms=6, label="edge peaks")
        if meas.found:
            ax_p.axvspan(meas.left * 1e9, meas.right * 1e9, color="#4fd97f",
                         alpha=0.25, label=f"measured {meas.feature}")
        ax_p.set_xlim(x0, x1)
        ax_p.set_xlabel("x [nm]")
        ax_p.set_ylabel("SE / primary")
        ax_p.grid(alpha=0.25)
        ax_p.legend(fontsize=8, loc="upper right", ncol=3)
        self.canvas.draw_idle()

        if meas.found:
            text = f"SEM CD ({meas.feature}, peak-to-peak)  {meas.cd * 1e9:6.1f} nm"
            if sim_cd_nm is not None and np.isfinite(sim_cd_nm) and sim_cd_nm > 0:
                text += (f"    profile at half height {sim_cd_nm:6.1f} nm"
                         f"    Δ {meas.cd * 1e9 - sim_cd_nm:+.1f} nm")
        else:
            text = f"no {meas.feature} found between edge peaks"
        self.readout.setText(text)
        if sim_cd_nm is not None:
            self.notes.setText(
                "The edge bloom is what the tool measures from. Widen the beam or "
                "the escape length and watch the measured CD drift from the "
                "simulated one — that is the metrology bias."
            )

    def _draw_xsection(self, sem: SEMImage, src: str, across: str,
                       floor: str) -> None:
        fig = self.figure
        fig.clf()
        ax = fig.add_subplot(111)
        x0, x1, y0, y1 = sem.extent_nm
        vmax = float(np.percentile(sem.image, 99.5))
        ax.imshow(sem.image, origin="lower", cmap="gray", extent=(x0, x1, y0, y1),
                  vmin=0.0, vmax=max(vmax, 1e-6), aspect="auto",
                  interpolation="nearest")
        ax.set_xlabel(f"{across} [nm]")
        ax.set_ylabel(f"z [nm]  ({floor} at 0)")
        ax.set_title(
            f"cross-section SEM of the {src} — {sem.electrons:.0f} e⁻/px, "
            f"SNR {sem.snr:.1f}",
            fontsize=10,
        )
        self.canvas.draw_idle()
        self.readout.setText("")
        if not self.from_stack:
            self.notes.setText(
                "The cleaved face: material contrast in the bulk, bloom along "
                "every boundary the cleave cuts. From the 3-D profile an undercut "
                "or a T-top shows here; the 2-D print has no way to make one."
            )
