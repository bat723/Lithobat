"""Figure views: the matplotlib canvases the app draws results on."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from litho_sim.app.compute import (
    FilmPreview,
    ImagingResult,
    SourcePreview,
)
from litho_sim.app.qt import Figure, FigureCanvasQTAgg, QtWidgets

logger = logging.getLogger(__name__)


class _CanvasView(QtWidgets.QWidget):
    """A widget that is one matplotlib canvas, and knows how to reuse artists.

    Every tab is a variation on this. Artists are built once on the first
    result and mutated afterwards: rebuilding axes per update costs hundreds
    of milliseconds, which is the difference between a slider that tracks the
    mouse and one that lurches.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self._artists: dict[str, Any] = {}
        self._relayout = True

    def _paint(self) -> None:
        """Repaint, running the constrained-layout solver only when needed.

        The solver calls ``get_tightbbox`` on every axes, and on a 3-D axes
        that is expensive — measured at 37 ms of every redraw, spent
        re-deriving a layout that changes only when the window does. Axes
        positions are in figure coordinates, so a frozen layout still scales
        with the canvas; only the padding would go stale, and a resize
        re-solves it.
        """
        self.figure.set_layout_engine("constrained" if self._relayout else "none")
        self._relayout = False
        self.canvas.draw_idle()

    def resizeEvent(self, event):        # noqa: N802 - Qt's spelling
        super().resizeEvent(event)
        self._relayout = True

    def _image(self, key: str, ax, data, cmap: str, **kw):
        """Draw or update one imshow, rescaling to the data."""
        im = self._artists.get(key)
        if im is None:
            im = self._artists[key] = ax.imshow(
                data, origin="lower", cmap=cmap, **kw
            )
        else:
            im.set_data(data)
            im.set_clim(float(np.min(data)), float(max(np.max(data), 1e-12)))
        return im

    @staticmethod
    def _bare(ax, title: str):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def show_placeholder(self, message: str) -> None:
        """Say why there is no picture yet, in the space the picture will use.

        Every step view starts here: nothing is computed until the Simulate
        tab is asked, so an empty axes would read as a broken one.
        """
        for ax in self.figure.axes:
            ax.clear()
            ax.set_axis_off()
        self._artists.clear()
        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#555555")
        self._relayout = True
        self._paint()

    def _restore_axes(self) -> None:
        """Undo :meth:`show_placeholder` before the first real draw.

        The layout has to be re-solved too: with the axes switched off the
        solver gave them the whole canvas, and titles and labels drawn into
        that layout land outside the figure — which is what a picture with
        no axis labels looked like the first time this was run.
        """
        for ax in self.figure.axes:
            for t in list(ax.texts):
                t.remove()
            ax.set_axis_on()
        self._relayout = True


class MaskView(_CanvasView):
    """What was drawn — the pattern before any optics touch it.

    Live: the mask is a drawing of the settings, not a simulation, and it
    costs microseconds, so it follows the Pattern controls as they move.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax = self.figure.add_subplot(111)
        self._bare(self.ax, "mask")

    def show_mask(self, mask: np.ndarray, pixel_nm: float | None = None) -> None:
        self._image("mask", self.ax, mask, "gray")
        px = f" · {pixel_nm:g} nm px" if pixel_nm else ""
        self.ax.set_title(
            f"mask — transmittance (1 = clear)   "
            f"{mask.shape[1]}×{mask.shape[0]} px{px}",
            fontsize=10,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._paint()

    def show_result(self, r: ImagingResult) -> None:
        self.show_mask(r.mask)


class SourceView(_CanvasView):
    """The illumination as the Abbe sum samples it, and where the drawn
    pitch's ±1 orders land against it.

    Live, like the mask: it is a picture of the settings. The dashed circles
    are the pupil displaced by one diffraction order — a source point inside
    the overlap gives a two-beam image of the pattern; one outside it only
    adds background. Watching the overlap grow as σ or NA moves, or as the
    pitch tightens until the circles part, is the Source tab's whole lesson.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")
        self._bare(self.ax, "illumination")

    def show_source(self, p: SourcePreview) -> None:
        ax = self.ax
        ax.clear()
        n = p.source.shape[0]
        # One pixel per source point, on the ±1 σ square the grid spans.
        ax.imshow(
            p.source, origin="lower", cmap="magma", extent=(-1, 1, -1, 1),
            interpolation="nearest",
            vmin=0.0, vmax=max(float(p.source.max()), 1e-12),
        )
        t = np.linspace(0.0, 2.0 * np.pi, 361)
        ax.plot(np.cos(t), np.sin(t), color="#ffffff", lw=1.0, alpha=0.9,
                label="pupil (σ = 1)")
        if p.order_shift is not None:
            for sign in (-1.0, 1.0):
                ax.plot(sign * p.order_shift + np.cos(t), np.sin(t),
                        color="#7fd9c8", lw=1.0, ls="--", alpha=0.9,
                        label="±1 order of the pitch" if sign > 0 else None)
            if p.order_shift > 2.0:
                ax.text(0.5, 0.04, "pitch below the resolution limit — "
                        "no source point sees a first order",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8, color="#d97f9b")
        lim = max(1.15, (p.order_shift or 0.0) + 1.05)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("ξ  (pupil units)")
        ax.set_ylabel("η")
        ax.set_title(f"illumination — {p.label}   ({n}×{n} grid)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_facecolor("#000000")
        self._relayout = True
        self._paint()


class ResistView(_CanvasView):
    """The coated film, sketched: the stack with its optical planes and voxel
    rows, and the Dill exposure curve the formulation implies.

    Live and cheap — nothing here is a simulation, it is the coat step's
    settings drawn so that "11 planes on 50 voxel rows" and "C = 0.04 cm²/mJ"
    are pictures rather than numbers.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        gs = self.figure.add_gridspec(1, 2, width_ratios=[1.0, 1.4])
        self.ax_film = self.figure.add_subplot(gs[0])
        self.ax_dill = self.figure.add_subplot(gs[1])

    def show_film(self, p: FilmPreview) -> None:
        ax = self.ax_film
        ax.clear()
        top = p.thickness_nm
        # Substrate slab, then the film; widths are arbitrary, heights real.
        ax.fill_between([0, 1], -0.25 * top, 0.0, color="#8a8a8a", lw=0,
                        label="substrate")
        ax.fill_between([0, 1], 0.0, top, color="#4fd97f", alpha=0.85, lw=0,
                        label=f"resist {top:.0f} nm")
        # Voxel rows on the left edge, optical planes across the film.
        for z in np.arange(0.0, top + 1e-9, p.dz_nm):
            ax.plot([0.0, 0.06], [z, z], color="#1c5a30", lw=0.6)
        for i, z in enumerate(p.plane_z_nm):
            ax.plot([0.1, 1.0], [z, z], color="#ffffff", lw=0.9, ls="--",
                    alpha=0.9, label="optical planes" if i == 0 else None)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.25 * top, top * 1.12)
        ax.set_xticks([])
        ax.set_ylabel("z [nm]")
        ax.set_title(
            f"{p.n_planes} planes · {p.n_voxels} voxel rows of {p.dz_nm:g} nm",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper right")

        ax = self.ax_dill
        ax.clear()
        ax.plot(p.dose_axis, p.pac, color="#c8913a", lw=2.0,
                label="PAC left,  m = exp(−C·E)")
        m0 = float(np.exp(-p.dill_C * p.dose_to_clear))
        ax.axvline(p.dose_to_clear, color="#7f9fd9", ls="--", lw=1.2,
                   label=f"dose 1.0 = {p.dose_to_clear:.0f} mJ/cm²  (m = {m0:.2f})")
        ax.set_xlim(0.0, float(p.dose_axis[-1]))
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("exposure dose E [mJ/cm²]")
        ax.set_ylabel("PAC remaining")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
        note = (" — the threshold model does not read the chemistry"
                if p.resist_model == "threshold" else "")
        ax.set_title(f"Dill exposure, C = {p.dill_C:g} cm²/mJ{note}", fontsize=9)
        self.figure.suptitle(p.label, fontsize=9, color="#555555")
        self._relayout = True
        self._paint()


class ExposeView(_CanvasView):
    """The aerial image, and the intensity cut through it. Where ~98 % of the
    compute goes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1.4, 1.0])
        self.ax_img = self.figure.add_subplot(gs[0])
        self.ax_cut = self.figure.add_subplot(gs[1])
        self._bare(self.ax_img, "aerial image")

    def show_result(self, r: ImagingResult) -> None:
        self._restore_axes()
        self._image("aerial", self.ax_img, r.aerial, "inferno")
        self.ax_img.set_title(
            f"aerial image — contrast {r.contrast:.3f}, NILS {r.nils:.2f}",
            fontsize=10,
        )
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        line = self._artists.get("cut")
        if line is None:
            (self._artists["cut"],) = self.ax_cut.plot(
                r.x_nm, r.cut_aerial, color="#c8913a", lw=2.0, label="aerial")
            self.ax_cut.set(xlabel="x [nm]", ylabel="intensity")
            self.ax_cut.legend(fontsize=8, loc="upper right")
            self.ax_cut.grid(alpha=0.25)
        else:
            line.set_data(r.x_nm, r.cut_aerial)
        self.ax_cut.set_xlim(float(r.x_nm[0]), float(r.x_nm[-1]))
        self.ax_cut.set_ylim(0.0, max(float(r.cut_aerial.max()), 1.0) * 1.2)
        self._paint()


class BakeView(_CanvasView):
    """The latent image after the post-exposure bake — what the developer
    will actually see — against the aerial image that went in.

    In its own units: intensity for the threshold model, PAC for mack, the
    protected fraction for car. The result says which, and the axes repeat
    it, because a curve drawn in the wrong units is the lie this app has
    told before.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1.4, 1.0])
        self.ax_img = self.figure.add_subplot(gs[0])
        self.ax_cut = self.figure.add_subplot(gs[1])
        self._bare(self.ax_img, "latent image")

    def show_result(self, r: ImagingResult) -> None:
        self._restore_axes()
        self._image("latent", self.ax_img, r.latent, "magma")
        self.ax_img.set_title(f"latent image — {r.latent_kind}", fontsize=10)
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        line = self._artists.get("latent_cut")
        if line is None:
            (self._artists["aerial_cut"],) = self.ax_cut.plot(
                r.x_nm, r.cut_aerial, color="#c8913a", lw=1.2, ls=":",
                label="aerial (intensity)")
            (self._artists["latent_cut"],) = self.ax_cut.plot(
                r.x_nm, r.cut_latent, color="#7f9fd9", lw=2.0,
                label="after bake")
            self.ax_cut.set(xlabel="x [nm]")
            self.ax_cut.legend(fontsize=8, ncol=2, loc="upper right")
            self.ax_cut.grid(alpha=0.25)
        else:
            self._artists["aerial_cut"].set_data(r.x_nm, r.cut_aerial)
            line.set_data(r.x_nm, r.cut_latent)
        self.ax_cut.set_ylabel(r.latent_kind)
        self.ax_cut.set_xlim(float(r.x_nm[0]), float(r.x_nm[-1]))
        top = max(float(r.cut_latent.max()), float(r.cut_aerial.max()), 1.0)
        self.ax_cut.set_ylim(0.0, top * 1.2)
        self._paint()


class DevelopView(_CanvasView):
    """The resist profile — footprint and cross-section in 2-D mode.

    The 3-D mode shares this widget; the toggle lives above it in the tab.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 1.0])
        self.ax_img = self.figure.add_subplot(gs[0])
        self.ax_cut = self.figure.add_subplot(gs[1])
        self.ax_prof = self.figure.add_subplot(gs[2])
        self._bare(self.ax_img, "resist")

    def show_result(self, r: ImagingResult) -> None:
        self._restore_axes()
        self._image("resist", self.ax_img, r.resist, "RdYlGn")
        self.ax_img.set_title(f"developed resist — CD {r.cd_text}", fontsize=10)
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        line = self._artists.get("cut")
        if line is None:
            (self._artists["cut"],) = self.ax_cut.plot(
                r.x_nm, r.cut_latent, color="#c8913a", lw=2.0,
                label="latent (post-PEB)")
            # Distinct key from the imshow above: one namespace holds both,
            # and reusing "resist" made the second redraw call set_data on a
            # Line2D with image-shaped input.
            (self._artists["resist_cut"],) = self.ax_cut.plot(
                r.x_nm, r.cut_resist, color="#4fd97f", lw=1.8, label="resist")
            self._artists["threshold"] = self.ax_cut.axhline(
                r.threshold, color="#d97f9b", ls="--", lw=1.2, label="threshold")
            self.ax_cut.set(xlabel="x [nm]", ylabel="intensity")
            self.ax_cut.legend(fontsize=8, ncol=3, loc="upper right")
            self.ax_cut.grid(alpha=0.25)
        else:
            line.set_data(r.x_nm, r.cut_latent)
            self._artists["resist_cut"].set_data(r.x_nm, r.cut_resist)
            self._artists["threshold"].set_ydata([r.threshold, r.threshold])
            self.ax_cut.set_xlim(float(r.x_nm[0]), float(r.x_nm[-1]))
        self.ax_cut.set_ylim(0.0, max(float(r.cut_latent.max()), 1.0) * 1.2)
        # The profile, as a filled cross-section. Same Mack rate law the 3-D
        # path uses, integrated straight down each column — so it is a height
        # field and cannot show an undercut, but it needs no volume and no
        # Compute button, and it turns a one-bit footprint into a shape.
        prof = self._artists.get("profile")
        if prof is None:
            self._artists["profile"] = self.ax_prof.fill_between(
                r.x_nm, 0.0, r.cut_thickness, color="#4fd97f", lw=0)
            (self._artists["film"],) = self.ax_prof.plot(
                r.x_nm, np.full_like(r.x_nm, r.film_nm),
                color="#888888", lw=1.0, ls=":", label="as coated")
            self.ax_prof.set(xlabel="x [nm]", ylabel="resist left [nm]")
            self.ax_prof.legend(fontsize=8, loc="upper right")
            self.ax_prof.grid(alpha=0.25)
        else:
            # fill_between has no set_data; the polygon is rebuilt.
            prof.remove()
            self._artists["profile"] = self.ax_prof.fill_between(
                r.x_nm, 0.0, r.cut_thickness, color="#4fd97f", lw=0)
            self._artists["film"].set_data(r.x_nm, np.full_like(r.x_nm, r.film_nm))
        self.ax_prof.set_xlim(float(r.x_nm[0]), float(r.x_nm[-1]))
        self.ax_prof.set_ylim(0.0, r.film_nm * 1.15)
        self.ax_prof.set_title(
            f"resist profile — top loss {r.film_nm - float(r.cut_thickness.max()):.1f} nm",
            fontsize=10)

        self._paint()


class Rotatable3D:
    """Keep a heavy 3-D axes usable under the mouse.

    matplotlib has no depth buffer: every frame it projects the polygons and
    sorts them by depth in Python, then repaints. That is fine for a few
    thousand triangles and hopeless for a real stack — a three-material wafer
    at 128 px is **268 k triangles**, which measures at 5.4 s a frame, and even
    the halved mesh the views ask for is 1.3 s. Dragging that does not feel
    slow, it feels broken: the events queue and the picture lurches.

    Nothing about the *data* changes while the camera moves, so the fix is to
    draw fewer polygons until the button comes up, and to cache whatever else
    is on the canvas rather than repainting it.

    A mixin rather than a base class because its users differ in shape — a
    3-D axes beside two 2-D panels, or a 3-D axes alone — so they keep their
    own re-render and only share the interaction.

    Subclasses call :meth:`_enable_rotation` once and implement
    ``_rerender_3d(detail_floor)``.
    """

    #: Mesh coarseness while the camera moves. A class attribute so a view
    #: with a heavier scene can ask for less — but not much less: at stride 16
    #: a three-material stack renders at 31 fps and **drops the 20 nm nitride
    #: layer altogether**, and a preview that omits a layer is worse than a
    #: slow one. 8 keeps every film and costs ~96 ms a frame.
    ROTATE_DETAIL = 8

    def _enable_rotation(self, ax3d) -> None:
        self._ax3d = ax3d
        self._rotating = False
        self._bg = None
        self._saved_draw_idle = None
        self.canvas.mpl_connect("button_press_event", self._rot_press)
        self.canvas.mpl_connect("button_release_event", self._rot_release)

    def _rerender_3d(self, detail_floor: int) -> None:
        raise NotImplementedError

    def _rot_press(self, event) -> None:
        if event.inaxes is not self._ax3d or self._rotating:
            return
        self._rotating = True
        self._rerender_3d(self.ROTATE_DETAIL)
        self._start_blit()

    def _rot_release(self, event) -> None:
        if not self._rotating:
            return
        self._stop_blit()
        self._rotating = False
        self._rerender_3d(1)

    # -- blitting ------------------------------------------------------
    # The static panels cost 25 ms of every rotation frame and cannot change
    # while the camera moves. So they are captured once as a bitmap and the
    # mesh is drawn over it — the difference between 21 fps and 45.
    def _start_blit(self) -> None:
        """Cache everything that is not the 3-D axes and draw over it.

        Worth a lot when there are static 2-D panels beside the mesh and
        almost nothing when the 3-D axes is the whole figure — harmless
        either way, so it is not conditional.
        """
        try:
            self._ax3d.set_visible(False)
            self.canvas.draw()
            self._bg = self.canvas.copy_from_bbox(self.figure.bbox)
            self._ax3d.set_visible(True)
            self._saved_draw_idle = self.canvas.draw_idle
            self.canvas.draw_idle = self._blit
        except Exception:                             # noqa: BLE001
            # Not every canvas can blit. A slow rotation beats a broken one.
            logger.debug("blitting unavailable; rotating with full redraws",
                         exc_info=True)
            self._stop_blit()

    def _stop_blit(self) -> None:
        if self._saved_draw_idle is not None:
            self.canvas.draw_idle = self._saved_draw_idle
            self._saved_draw_idle = None
        self._bg = None
        self._ax3d.set_visible(True)

    def _blit(self) -> None:
        """One rotation frame: restore the panels, draw only the mesh."""
        if self._bg is None:
            return
        self.canvas.restore_region(self._bg)
        self._ax3d.draw(self.canvas.get_renderer())
        self.canvas.blit(self.figure.bbox)


class Profile3DView(Rotatable3D, _CanvasView):
    """The developed solid, its cross-section, and the latent image.

    Deliberately does *not* call ``viz3d.resist_profile_3d_figure``: that
    builds its own figure through pyplot, which would leak one per redraw and
    breaks this module's rule about the global figure registry. The pieces it
    composes — ``stack_figure_mpl`` and ``profile_figure`` — both accept an
    ``ax``, so the same picture is assembled into axes we own.

    Rotation (coarsen on press, blit the static panels, restore on release)
    is :class:`Rotatable3D`'s; this class only supplies the redraw.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        gs = self.figure.add_gridspec(2, 2, width_ratios=[1.35, 1.0])
        self.ax3d = self.figure.add_subplot(gs[:, 0], projection="3d")
        self.ax_prof = self.figure.add_subplot(gs[0, 1])
        self.ax_lat = self.figure.add_subplot(gs[1, 1])

        # Enough to redraw the same picture at a different detail level, so
        # rotating can coarsen the mesh without going back for the physics.
        self._last: tuple | None = None
        self._drawn_once = False
        self._enable_rotation(self.ax3d)

    # -- rotation ------------------------------------------------------
    def _rot_press(self, event) -> None:
        if self._last is None:      # nothing drawn yet, nothing to coarsen
            return
        super()._rot_press(event)

    def _rerender_3d(self, detail_floor: int) -> None:
        if self._last is not None:
            self._render(*self._last, detail_floor=detail_floor, only_3d=True)

    def show_profile(self, r, grid, z_exaggeration: float, downsample: int,
                     render_mode: str = "surface") -> None:
        self._last = (r, grid, z_exaggeration, downsample, render_mode)
        self._render(r, grid, z_exaggeration, downsample, render_mode,
                     detail_floor=self.ROTATE_DETAIL if self._rotating else 1)

    def _render(self, r, grid, z_exaggeration: float, downsample: int,
                render_mode: str = "surface", detail_floor: int = 1,
                only_3d: bool = False) -> None:
        from litho_sim.viz.viz3d import (
            profile_figure,
            resist_stack_for_display,
            resist_surface_mpl,
            stack_figure_mpl,
        )

        detail = max(int(downsample), int(detail_floor), 1)

        # Rebuilt rather than mutated: neither a Poly3DCollection nor a
        # surface has a set_data, and the geometry changes shape with every
        # detail change or new volume anyway. But `clear()` resets the camera,
        # which would throw away the angle the user just rotated to — so the
        # view is carried across the rebuild once there is one worth keeping.
        camera = None if self._last is None or not self._drawn_once else (
            self.ax3d.elev, self.ax3d.azim, self.ax3d.roll
        )
        if not self._drawn_once:
            # First real picture after the placeholder: the layout was
            # solved for bare axes and has to be solved again for real ones.
            self._relayout = True
        self.ax3d.clear()
        if render_mode == "solid":
            # Honest voxel faces — tens of thousands of triangles for
            # matplotlib to depth-sort in Python, and roughly 7x the draw
            # cost. Worth it only when the profile is not single-valued.
            stack_figure_mpl(
                resist_stack_for_display(r.remaining, grid),
                z_exaggeration=z_exaggeration,
                downsample=detail,
                ax=self.ax3d,
            )
        else:
            resist_surface_mpl(
                r.remaining, grid,
                ax=self.ax3d,
                z_exaggeration=z_exaggeration,
                stride=detail,
            )
        self.ax3d.set_title(f"developed resist — {r.label}", fontsize=9)
        if camera is not None:
            self.ax3d.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        self._drawn_once = True

        if only_3d:
            # Rotating changes the camera, not the data — the two 2-D panels
            # are already correct and rebuilding them is pure waste.
            self._paint()
            return

        self.ax_prof.clear()
        profile_figure(r.remaining, grid, row=r.row, ax=self.ax_prof)
        self.ax_prof.set_title("profile through the film", fontsize=9)
        self.ax_prof.text(
            0.02, 0.96, r.summary.replace("    ", "\n"),
            transform=self.ax_prof.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
        )
        if r.diagnosis:
            # A degenerate outcome deserves an explanation, not a blank axis.
            self.ax_prof.text(
                0.5, 0.45, r.diagnosis, transform=self.ax_prof.transAxes,
                ha="center", va="center", fontsize=8, color="#B00020", wrap=True,
            )

        self.ax_lat.clear()
        self.ax_lat.imshow(
            r.cut_latent, origin="lower", aspect="auto", cmap="magma",
            extent=[0, float(r.x_nm[-1]), 0, r.height_nm],
        )
        self.ax_lat.set(xlabel="x [nm]", ylabel="z [nm]")
        self.ax_lat.set_title("latent image (PAC after bake)", fontsize=9)

        self._paint()

    def show_placeholder(self, message: str) -> None:
        for ax in (self.ax3d, self.ax_prof, self.ax_lat):
            ax.clear()
        self.ax_prof.text(
            0.5, 0.5, message, transform=self.ax_prof.transAxes,
            ha="center", va="center", fontsize=10, color="#555555",
        )
        self._paint()


class StackView(Rotatable3D, _CanvasView):
    """The wafer stack — a material cross-section, or the solid in 3-D.

    Cross-section is the default because it is nearly free and, as
    ``cross_section_figure``'s own docstring puts it, usually the most
    informative single view: it shows every layer at once, in colour, with the
    legend that names them.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_3d = False
        #: Show the process-flow strip beside the cross-section. It says the
        #: same thing as the recipe list, in the engine's families rather than
        #: the step vocabulary — worth having and worth being able to put away,
        #: because it costs a quarter of the figure's width whether the flow is
        #: four steps or forty. Set before `_build_axes`, which reads it.
        self.show_panel = True
        self._build_axes(False)
        self._last: tuple | None = None
        self._drawn_once = False
        #: Extra coarsening applied on top of the caller's downsample, raised
        #: while the camera is moving.
        self._detail_floor = 1
        self._bare(self.ax, "wafer stack")
        self._enable_rotation(self.ax)

    def _build_axes(self, three_d: bool) -> None:
        """Lay the figure out for one mode.

        Used by ``__init__`` as well as :meth:`set_mode`, because the opening
        state is a mode too. Building the panel only in ``set_mode`` left the
        initial cross-section without one — and ``set_mode`` returns early when
        the mode has not changed, so it never got built at all.
        """
        if three_d or not self.show_panel:
            self.ax_panel = None
            self.ax = self.figure.add_subplot(
                111, projection="3d" if three_d else None
            )
        else:
            # Cross-section gets a companion: the film stack at one column,
            # sharing the z axis. The section shows *where* the materials are;
            # the panel shows *what the stack is*, in order, with thicknesses.
            gs = self.figure.add_gridspec(1, 2, width_ratios=[1.15, 3],
                                          wspace=0.06)
            self.ax_panel = self.figure.add_subplot(gs[0])
            self.ax = self.figure.add_subplot(gs[1])

    def set_mode(self, three_d: bool, redraw: bool = True) -> None:
        """Swap between the cross-section and the solid.

        A 3-D axes cannot be turned into a 2-D one, so the axes is replaced
        rather than reconfigured — and the layout solver has to run once more
        afterwards, since the geometry genuinely did change.
        """
        if three_d == self._is_3d:
            return
        self._is_3d = three_d
        self.figure.clear()
        self._build_axes(three_d)
        self._drawn_once = False
        self._relayout = True
        # figure.clear() destroyed the axes the handlers were bound to.
        self._ax3d = self.ax
        if not redraw:
            # The caller is about to draw something better. Repainting the
            # cached wafer first is not just wasted: which wafer belongs on
            # screen now depends on the mode we have only just changed to —
            # the solid view wants the device cut open, the cross-section
            # wants it whole — so `_last` is the *previous* mode's answer.
            return
        if self._last is not None:
            self.show_stack(*self._last)
        else:
            self._bare(self.ax, "wafer stack")
            self._paint()

    def set_panel(self, show: bool) -> None:
        """Show or hide the process-flow strip beside the cross-section.

        The strip is one column of a gridspec, so removing it is a relayout
        rather than a visibility flag — hiding the axes would leave its width
        reserved, which is the whole thing being reclaimed.
        """
        if show == self.show_panel:
            return
        self.show_panel = show
        if self._is_3d:
            # Nothing on screen to change; the solid view has no strip. Taking
            # the preference now means switching back honours it, rather than
            # coming up with the panel the user just put away.
            return
        self.figure.clear()
        self._build_axes(self._is_3d)
        self._drawn_once = False
        self._relayout = True
        self._ax3d = self.ax
        if self._last is not None:
            self.show_stack(*self._last)
        else:
            self._bare(self.ax, "wafer stack")
            self._paint()

    def show_stack(self, stack, label: str = "", z_exaggeration: float = 1.0,
                   downsample: int = 4) -> None:
        from litho_sim.viz.viz3d import cross_section_figure, device_figure_mpl

        self._last = (stack, label, z_exaggeration, downsample)

        if self._is_3d:
            # `device_figure_mpl`, not `stack_figure_mpl`: a wafer carries
            # several interleaved films, and one Poly3DCollection per material
            # lets Matplotlib sort them independently — a buried layer paints
            # over the layer burying it, and the library's sub-1 opacities veil
            # what is behind them. It also calls view_init unconditionally, so
            # the camera the user set is lost on every redraw unless carried.
            camera = (
                (self.ax.elev, self.ax.azim, self.ax.roll)
                if self._drawn_once else None
            )
            self.ax.clear()
            device_figure_mpl(
                stack, z_exaggeration=z_exaggeration,
                downsample=max(int(downsample), self._detail_floor, 1),
                ax=self.ax,
                # `ax.clear()` puts the 3-D decoration back, so the chrome has
                # to be re-stated on every draw rather than set once.
                chrome="none", scale_bar=True, projection="ortho",
                # Matplotlib leaves wide margins around a 3-D artist; the panel
                # is not so large that they can be spared. Kept modest because
                # the panel's aspect changes with the window and an overzoomed
                # wafer would clip.
                zoom=1.18,
            )
            if camera is not None:
                self.ax.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        else:
            from litho_sim.viz.viz3d import process_flow_panel

            self.ax.clear()
            cross_section_figure(stack, axis="y", ax=self.ax, title="")
            # Only the solid is worth showing: a process stack carries whatever
            # headroom its deposits needed, and scaling z to that squashes the
            # films into the bottom fifth of the axes.
            top = max(float(stack.top_height().max()) * 1e9, 1.0) * 1.05
            self.ax.set_ylim(0.0, top)
            if self.ax_panel is not None:
                # The recipe that built this wafer, beside the wafer it built.
                self.ax_panel.clear()
                process_flow_panel(stack, ax=self.ax_panel)

        # Set after the renderer, which writes its own title.
        self.ax.set_title(label or "wafer stack", fontsize=10)
        self._drawn_once = True
        self._paint()

    def _rerender_3d(self, detail_floor: int) -> None:
        """Redraw the parked stack at a different mesh detail. No new physics."""
        self._detail_floor = max(int(detail_floor), 1)
        if self._is_3d and self._last is not None:
            self.show_stack(*self._last)

    def show_placeholder(self, message: str) -> None:
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, transform=self.ax.transAxes,
                     ha="center", va="center", fontsize=10, color="#555555")
        self.ax.set_axis_off()
        self._paint()


