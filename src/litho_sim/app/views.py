"""Figure views: the canvases the app draws results on.

Every 2-D picture is a matplotlib canvas built on :mod:`litho_sim.viz.theme`;
the wafer stack's solid is :class:`~litho_sim.app.solid_view.SolidView`, and
the developed resist is imaged rather than drawn — :class:`TiltSemView` is
a tilt-stage micrograph of it. The rules
the theme states — a title names, a caption counts, every image is in
nanometres, nothing dashed, no legend on the data — are applied here once
per view, so a tab's picture and the CLI's figure of the same quantity look
like the same instrument.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

from litho_sim.app.compute import (
    FilmPreview,
    ImagingResult,
    Profile3DResult,
    SourcePreview,
)
from litho_sim.app.qt import Figure, FigureCanvasQTAgg, QtCore, QtWidgets
from litho_sim.app.solid_view import SolidView
from litho_sim.metrology.sem import SEMConfig
from litho_sim.viz import render, theme
from litho_sim.viz.theme import CMAP, INK2, MUTED, SERIES

logger = logging.getLogger(__name__)


def _extent_nm(x_nm: np.ndarray, shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    """The image's frame in nm, from the cut's x axis and the array shape."""
    px = float(x_nm[1] - x_nm[0]) if len(x_nm) > 1 else 1.0
    h, w = shape[:2]
    return (0.0, w * px, 0.0, h * px)


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
        self._placeholder_art = None

    def _paint(self) -> None:
        """Repaint, running the constrained-layout solver only when needed.

        The solver calls ``get_tightbbox`` on every axes and every artist
        outside them — colourbars, captions, legends on the title row — and
        re-derives a layout that changes only when the window does. Axes
        positions are in figure coordinates, so a frozen layout still scales
        with the canvas; a resize re-solves it.
        """
        self.figure.set_layout_engine("constrained" if self._relayout else "none")
        self._relayout = False
        self.canvas.draw_idle()

    def resizeEvent(self, event):        # noqa: N802 - Qt's spelling
        super().resizeEvent(event)
        self._relayout = True

    def _image(self, key: str, ax, data, cmap, *, extent=None, cbar_label=None,
               norm=None, vmin=None, vmax=None, **kw):
        """An image in physical units, created once and mutated afterwards."""
        im = self._artists.get(key)
        if im is None:
            if extent is None:
                h, w = data.shape[:2]
                extent = (0.0, float(w), 0.0, float(h))
            im = self._artists[key] = theme.physical_image(
                ax, data, extent, cmap, cbar_label=cbar_label, norm=norm,
                vmin=vmin, vmax=vmax, **kw,
            )
            self._relayout = True
        else:
            im.set_data(data)
            if extent is not None and tuple(im.get_extent()) != tuple(extent):
                im.set_extent(extent)
            if norm is None:
                lo = float(np.min(data)) if vmin is None else float(vmin)
                hi = float(np.max(data)) if vmax is None else float(vmax)
                im.set_clim(lo, max(hi, lo + 1e-12))
        return im

    def show_placeholder(self, message: str) -> None:
        """Say why there is no picture yet, in the space the picture will use.

        Every step view starts here: nothing is computed until the Simulate
        tab is asked, so an empty axes would read as a broken one.
        """
        for art in self._artists.values():
            cb = getattr(art, "colorbar", None)
            if cb is not None:
                cb.remove()
        self._artists.clear()
        for ax in self.figure.axes:
            ax.clear()
            ax.set_axis_off()
        ax = self.figure.axes[0] if self.figure.axes else self.figure.add_subplot(111)
        self._placeholder_art = theme.placeholder(ax, message)
        self._relayout = True
        self._paint()

    def _restore_axes(self) -> None:
        """Undo :meth:`show_placeholder` before the first real draw.

        Only the placeholder's own text goes — the captions and rule labels
        the views draw are theirs to keep across redraws. The layout has to
        be re-solved too: with the axes switched off the solver gave them the
        whole canvas, and titles and labels drawn into that layout land
        outside the figure — which is what a picture with no axis labels
        looked like the first time this was run.
        """
        art = self._placeholder_art
        if art is not None:
            if art.axes is not None:
                art.remove()
            self._placeholder_art = None
            for ax in self.figure.axes:
                ax.set_axis_on()
            self._relayout = True

    @property
    def readout(self) -> str:
        """The caption row of the main axes — the numbers the title does not carry."""
        ax = getattr(self, "ax", None) or getattr(self, "ax_img", None)
        if ax is None and self.figure.axes:
            ax = self.figure.axes[0]
        art = getattr(ax, "_litho_caption", None)
        return art.get_text() if art is not None and art.axes is ax else ""


class MaskView(_CanvasView):
    """What was drawn — the pattern before any optics touch it.

    Live: the mask is a drawing of the settings, not a simulation, and it
    costs microseconds, so it follows the Pattern controls as they move.

    With the correction on, what goes on the mask is the loop's output
    rather than a drawing, and :meth:`show_corrected` shows that: the
    corrected raster, with the design's outline over it so the jogs and
    serifs the loop grew are visible against what was asked for.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax = self.figure.add_subplot(111)
        theme.title(self.ax, "Mask")
        self._outlines: list[Any] = []

    def _clear_outlines(self) -> None:
        for art in self._outlines:
            art.remove()
        self._outlines = []
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()

    def _draw_raster(self, mask: np.ndarray, pixel_nm: float | None):
        h, w = mask.shape
        if pixel_nm:
            extent = (0.0, w * pixel_nm, 0.0, h * pixel_nm)
            labels = {"xlabel": "x [nm]", "ylabel": "y [nm]"}
        else:
            extent = (0.0, float(w), 0.0, float(h))
            labels = {"xlabel": "x [px]", "ylabel": "y [px]"}
        self._image("mask", self.ax, mask, CMAP.micrograph, extent=extent,
                    vmin=0.0, vmax=1.0, **labels)
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        return extent

    def show_mask(self, mask: np.ndarray, pixel_nm: float | None = None) -> None:
        self._clear_outlines()
        self._draw_raster(mask, pixel_nm)
        h, w = mask.shape
        px = f" · {pixel_nm:g} nm px" if pixel_nm else ""
        theme.title(self.ax, "Mask",
                    caption_text=f"transmittance, 1 = clear · {w}×{h} px{px}")
        self._paint()

    def show_corrected(self, mask: np.ndarray, pixel_nm: float, result) -> None:
        """The corrected mask as it will be imaged, with the design over it.

        *mask* is the corrected layout rasterised on the grid it was
        corrected for; *result* the :class:`~litho_sim.opc.OPCResult` it
        came from, whose geometry is drawn in metres from the field centre
        and shifted here onto the raster's frame.
        """
        self._clear_outlines()
        self._draw_raster(mask, pixel_nm)
        h, w = mask.shape
        # Pixel (i, j) is centred at ((i − n//2)·px, (j − n//2)·px) in the
        # geometry's frame and spans [i, i+1]·px in the image's.
        shift = np.array([(w // 2 + 0.5) * pixel_nm, (h // 2 + 0.5) * pixel_nm])
        layer = result.settings.get("layer")
        self._outline(result.design, SERIES.reference, 0.9, "design", layer, shift)
        if "sraf" in result.corrected.layers():
            self._outline(result.corrected, MUTED, 0.9, "assist features", "sraf", shift)
        self._outline(result.corrected, SERIES.warn, 1.0, "corrected", layer, shift)
        offs = np.abs(result.offsets) * 1e9
        sb, sa = result.epe_before.stats(), result.epe_after.stats()
        theme.title(
            self.ax, "Mask",
            caption_text=(f"corrected · {len(offs)} fragments · largest move "
                          f"{offs.max() if offs.size else 0.0:.1f} nm · max |EPE| "
                          f"{sb['max_abs_epe_nm']:.1f} to {sa['max_abs_epe_nm']:.1f} nm · "
                          f"{w}×{h} px · {pixel_nm:g} nm px"),
        )
        theme.legend(self.ax, where="top", ncol=3)
        self._relayout = True
        self._paint()

    def _outline(self, layout, color, lw, label, layer, shift) -> None:
        shapes = layout.shapes if layer is None else layout.on_layer(layer)
        first = True
        for sh in shapes:
            p = np.asarray(sh.polygon(), dtype=np.float64) * 1e9 + shift
            p = np.vstack([p, p[:1]])
            (line,) = self.ax.plot(p[:, 0], p[:, 1], color=color, lw=lw,
                                   label=label if first else None,
                                   solid_joinstyle="miter")
            self._outlines.append(line)
            first = False

    def show_result(self, r: ImagingResult) -> None:
        px = float(r.x_nm[1] - r.x_nm[0]) if len(r.x_nm) > 1 else None
        self.show_mask(r.mask, pixel_nm=px)


class SourceView(_CanvasView):
    """The illumination as the Abbe sum samples it, and where the drawn
    pitch's ±1 orders land against it.

    Live, like the mask: it is a picture of the settings. The two outer
    circles are the pupil displaced by one diffraction order — a source point
    inside the overlap gives a two-beam image of the pattern; one outside it
    only adds background. Watching the overlap grow as σ or NA moves, or as
    the pitch tightens until the circles part, is the Source tab's whole
    lesson.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")
        theme.title(self.ax, "Illumination")

    def show_source(self, p: SourcePreview) -> None:
        ax = self.ax
        ax.clear()
        n = p.source.shape[0]
        # One pixel per source point, on the ±1 σ square the grid spans.
        theme.physical_image(
            ax, p.source, (-1.0, 1.0, -1.0, 1.0), CMAP.source,
            vmin=0.0, vmax=max(float(p.source.max()), 1e-12),
            xlabel="ξ  [pupil units]", ylabel="η  [pupil units]",
        )
        t = np.linspace(0.0, 2.0 * np.pi, 361)
        ax.plot(np.cos(t), np.sin(t), color=INK2, lw=1.0, label="pupil, σ = 1")
        if p.order_shift is not None:
            for sign in (-1.0, 1.0):
                ax.plot(sign * p.order_shift + np.cos(t), np.sin(t),
                        color=SERIES.aerial, lw=1.0,
                        label="±1 order of the pitch" if sign > 0 else None)
            if p.order_shift > 2.0:
                ax.text(0.5, 0.03, "pitch below the resolution limit — "
                        "no source point sees a first order",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8, color=SERIES.warn)
        lim = max(1.15, (p.order_shift or 0.0) + 1.05)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        theme.title(ax, "Illumination", caption_text=f"{p.label} · {n}×{n} grid")
        theme.legend(ax, where="top")
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
        resist = theme.material_colour("photoresist")
        substrate = theme.material_colour("Si")
        # Substrate slab, then the film; widths are arbitrary, heights real.
        ax.fill_between([0, 1], -0.25 * top, 0.0, color=substrate, lw=0,
                        label="substrate")
        ax.fill_between([0, 1], 0.0, top, color=resist, lw=0,
                        label=f"resist, {top:.0f} nm")
        # Voxel rows on the left edge, optical planes across the film.
        for z in np.arange(0.0, top + 1e-9, p.dz_nm):
            ax.plot([0.0, 0.06], [z, z], color=theme.SURFACE, lw=0.6, alpha=0.8)
        for i, z in enumerate(p.plane_z_nm):
            ax.plot([0.1, 1.0], [z, z], color=INK2, lw=0.8, alpha=0.9,
                    label="optical planes" if i == 0 else None)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.25 * top, top * 1.12)
        ax.set_xticks([])
        ax.set_ylabel("z [nm]")
        theme.title(
            ax, "Coated film",
            caption_text=f"{p.n_planes} planes · {p.n_voxels} voxel rows of {p.dz_nm:g} nm",
        )
        theme.legend(ax, where="below", bbox_to_anchor=(0.5, -0.02))

        ax = self.ax_dill
        ax.clear()
        ax.plot(p.dose_axis, p.pac, color=SERIES.aerial)
        m0 = float(np.exp(-p.dill_C * p.dose_to_clear))
        theme.rule(ax, x=p.dose_to_clear, color=MUTED, lw=1.0,
                   text=f"dose 1.0 = {p.dose_to_clear:.0f} mJ/cm², m = {m0:.2f}")
        ax.set_xlim(0.0, float(p.dose_axis[-1]))
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("exposure dose E [mJ/cm²]")
        ax.set_ylabel("PAC remaining, m = exp(−C·E)")
        theme.grid(ax)
        note = (" — the threshold model does not read the chemistry"
                if p.resist_model == "threshold" else "")
        theme.title(ax, "Dill exposure", caption_text=f"C = {p.dill_C:g} cm²/mJ{note}")
        theme.caption(self.figure, p.label)
        self._relayout = True
        self._paint()


class ExposeView(_CanvasView):
    """The aerial image, and the intensity cut through it. Where ~98 % of the
    compute goes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1.5, 1.0])
        self.ax_img = self.figure.add_subplot(gs[0])
        self.ax_cut = self.figure.add_subplot(gs[1])
        theme.title(self.ax_img, "Aerial image")

    def show_result(self, r: ImagingResult) -> None:
        self._restore_axes()
        self._image("aerial", self.ax_img, r.aerial, CMAP.intensity,
                    extent=_extent_nm(r.x_nm, r.aerial.shape),
                    vmin=0.0, vmax=max(float(r.aerial.max()), 1e-12),
                    cbar_label="intensity [a.u.]")
        theme.title(self.ax_img, "Aerial image",
                    caption_text=f"contrast {r.contrast:.3f} · NILS {r.nils:.2f}")

        line = self._artists.get("cut")
        if line is None:
            (self._artists["cut"],) = self.ax_cut.plot(
                r.x_nm, r.cut_aerial, color=SERIES.aerial)
            self.ax_cut.set(xlabel="x [nm]", ylabel="intensity")
            theme.grid(self.ax_cut)
            theme.title(self.ax_cut, "Cut through the centre row")
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
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1.5, 1.0])
        self.ax_img = self.figure.add_subplot(gs[0])
        self.ax_cut = self.figure.add_subplot(gs[1])
        theme.title(self.ax_img, "Latent image after bake")

    def show_result(self, r: ImagingResult) -> None:
        self._restore_axes()
        im = self._image("latent", self.ax_img, r.latent, CMAP.latent,
                         extent=_extent_nm(r.x_nm, r.latent.shape),
                         cbar_label=r.latent_kind)
        if im.colorbar is not None:
            im.colorbar.set_label(r.latent_kind, color=INK2, fontsize=8.5)
        theme.title(self.ax_img, "Latent image after bake",
                    caption_text=f"shown as {r.latent_kind}")

        line = self._artists.get("latent_cut")
        if line is None:
            (self._artists["aerial_cut"],) = self.ax_cut.plot(
                r.x_nm, r.cut_aerial, color=SERIES.aerial, lw=1.2, alpha=0.6,
                label="aerial (intensity)")
            (self._artists["latent_cut"],) = self.ax_cut.plot(
                r.x_nm, r.cut_latent, color=SERIES.latent, label="after bake")
            self.ax_cut.set(xlabel="x [nm]")
            theme.grid(self.ax_cut)
            theme.title(self.ax_cut, "Cut through the centre row")
            theme.legend(self.ax_cut, where="top")
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

    The 3-D mode is :class:`Profile3DView`; the toggle lives above both in
    the tab.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1.5, 1.0, 1.0])
        self.ax_img = self.figure.add_subplot(gs[0])
        self.ax_cut = self.figure.add_subplot(gs[1])
        self.ax_prof = self.figure.add_subplot(gs[2])
        theme.title(self.ax_img, "Developed resist")

    def show_result(self, r: ImagingResult) -> None:
        self._restore_axes()
        resist = theme.material_colour("photoresist")
        cmap, norm = theme.binary_cmap(resist)
        self._image("resist", self.ax_img, r.resist, cmap, norm=norm,
                    extent=_extent_nm(r.x_nm, r.resist.shape))
        theme.title(self.ax_img, "Developed resist", caption_text=f"CD {r.cd_text}")

        ax = self.ax_cut
        top = max(float(r.cut_latent.max()), 1.0) * 1.2
        line = self._artists.get("cut")
        if line is None:
            (self._artists["cut"],) = ax.plot(
                r.x_nm, r.cut_latent, color=SERIES.latent, label="latent (post-PEB)")
            self._artists["threshold"] = theme.rule(
                ax, y=r.threshold, color=SERIES.threshold, ls="--", lw=1.0,
                label="threshold")
            ax.set(xlabel="x [nm]", ylabel="intensity")
            theme.grid(ax)
            theme.title(ax, "Cut through the centre row")
        else:
            line.set_data(r.x_nm, r.cut_latent)
            self._artists["threshold"].set_ydata([r.threshold, r.threshold])
        # The footprint as a wash under the curve: where resist remains, the
        # cut is shaded in the resist's own colour. fill_between has no
        # set_data, so the polygon is rebuilt — it is a few dozen vertices.
        wash = self._artists.get("resist_wash")
        if wash is not None:
            wash.remove()
        self._artists["resist_wash"] = ax.fill_between(
            r.x_nm, 0.0, np.asarray(r.cut_resist, dtype=float) * top,
            color=resist, alpha=0.18, lw=0, label="resist remains")
        ax.set_xlim(float(r.x_nm[0]), float(r.x_nm[-1]))
        ax.set_ylim(0.0, top)
        theme.legend(ax, where="top")

        # The profile, as a filled cross-section. Same Mack rate law the 3-D
        # path uses, integrated straight down each column — so it is a height
        # field and cannot show an undercut, but it needs no volume and no
        # Compute button, and it turns a one-bit footprint into a shape.
        ax = self.ax_prof
        prof = self._artists.get("profile")
        if prof is not None:
            prof.remove()
        self._artists["profile"] = ax.fill_between(
            r.x_nm, 0.0, r.cut_thickness, color=resist, lw=0)
        film = self._artists.get("film")
        if film is None:
            self._artists["film"] = theme.rule(
                ax, y=r.film_nm, color=MUTED, lw=1.0, text="as coated")
            ax.set(xlabel="x [nm]", ylabel="resist left [nm]")
            theme.grid(ax)
        else:
            film.set_ydata([r.film_nm, r.film_nm])
            film._litho_text.xy = (1.0, r.film_nm)
        ax.set_xlim(float(r.x_nm[0]), float(r.x_nm[-1]))
        ax.set_ylim(0.0, r.film_nm * 1.15)
        theme.title(
            ax, "Resist profile",
            caption_text=(f"top loss {r.film_nm - float(r.cut_thickness.max()):.1f} nm"
                          " · height field, no undercut"),
        )
        self._paint()


class ProfilePanels(_CanvasView):
    """The two 2-D readings of a 3-D profile: the section through the film
    and the latent image it developed from. Sits beside the solid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        gs = self.figure.add_gridspec(2, 1)
        self.ax_prof = self.figure.add_subplot(gs[0])
        self.ax_lat = self.figure.add_subplot(gs[1])
        self.ax = self.ax_prof
        self._cbar = None

    def _clear(self) -> None:
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        for ax in (self.ax_prof, self.ax_lat):
            ax.clear()
            ax.set_axis_on()

    def show(self, r: Profile3DResult, grid) -> None:
        self._clear()
        px = grid.pixel_size * 1e9
        width = r.cut_remaining.shape[1] * px
        extent = (0.0, width, 0.0, float(r.height_nm))
        cmap, norm = theme.binary_cmap(theme.material_colour("photoresist"))
        theme.physical_image(
            self.ax_prof, np.asarray(r.cut_remaining, dtype=np.uint8), extent,
            cmap, norm=norm, aspect="auto", ylabel="z [nm]",
        )
        theme.title(self.ax_prof, "Profile through the film",
                    caption_text=r.summary.replace("    ", " · "))
        if r.diagnosis:
            # A degenerate outcome deserves an explanation, not a blank axis.
            self.ax_prof.text(
                0.5, 0.5, r.diagnosis, transform=self.ax_prof.transAxes,
                ha="center", va="center", fontsize=8.5, color=SERIES.warn, wrap=True,
            )
        im = theme.physical_image(
            self.ax_lat, r.cut_latent, extent, CMAP.latent, aspect="auto",
            ylabel="z [nm]", cbar_label="PAC after bake",
        )
        self._cbar = im.colorbar
        theme.title(self.ax_lat, "Latent image after bake")
        self._relayout = True
        self._paint()

    def show_placeholder(self, message: str) -> None:
        self._clear()
        for ax in (self.ax_prof, self.ax_lat):
            ax.set_axis_off()
        theme.placeholder(self.ax_prof, message)
        self._relayout = True
        self._paint()


NO_RENDERER = (
    "The tilt-SEM view needs VTK — install the viz3d extra:\n"
    "    pip install -e '.[viz3d]'\n"
    "(pyvista)"
)


class TiltSemView(_CanvasView):
    """The developed resist as a tilt-stage SEM micrograph.

    Not a drawing of the solid: an image of it, formed the way the SEM tab
    forms its top-down and cross-section frames, from the geometry a
    depth-buffer render measures (:mod:`litho_sim.viz.render`). The stage
    (tilt, rotation) is the Develop tab's; the instrument is the SEM tab's.

    Two costs, kept apart. Moving the stage re-renders the geometry —
    a few hundred milliseconds for a 128-pixel field. Turning an
    instrument knob only re-forms the signal from the geometry in hand,
    which is numpy over an image and quick. The geometry is cached against
    the profile and the stage, so a knob never pays for a render.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure.set_layout_engine("none")
        self.ax = self.figure.add_subplot(111)
        self._profile: Profile3DResult | None = None
        self._grid = None
        self._cfg = SEMConfig()
        self._stage = render.Stage()
        self._buffers = None
        self._buffers_key: tuple | None = None
        self.last = None            # the SEMImage on screen

    @property
    def available(self) -> bool:
        return render.available()

    def show(self, r: Profile3DResult, grid, cfg: SEMConfig | None = None,
             *, tilt: float | None = None, azimuth: float | None = None) -> None:
        self._profile, self._grid = r, grid
        if cfg is not None:
            self._cfg = cfg
        if tilt is not None or azimuth is not None:
            self._stage = dataclasses.replace(
                self._stage,
                tilt=float(self._stage.tilt if tilt is None else tilt),
                azimuth=float(self._stage.azimuth if azimuth is None else azimuth),
            )
        self._render()

    def set_stage(self, tilt: float, azimuth: float) -> None:
        """Move the stage. Re-images the profile in hand, if there is one."""
        self._stage = dataclasses.replace(self._stage, tilt=float(tilt), azimuth=float(azimuth))
        if self._profile is not None:
            self._render()

    def set_instrument(self, cfg: SEMConfig) -> None:
        """The SEM tab's knobs moved. Re-forms the image from the cached geometry."""
        self._cfg = cfg
        if self._profile is not None:
            self._render()

    def _geometry(self):
        r, grid, st = self._profile, self._grid, self._stage
        key = (id(r), r.signature, st.tilt, st.azimuth, st.pixel_nm, st.frame_px)
        if self._buffers is None or self._buffers_key != key:
            px, dz = float(grid.pixel_size) * 1e9, float(grid.dz) * 1e9
            if r.field is not None:
                surface = render.profile_surface(
                    field=r.field, level=r.level, feature=r.feature, spacing_nm=(dz, px, px))
            else:
                surface = render.profile_surface(r.remaining, spacing_nm=(dz, px, px))
            nz, ny, nx = r.remaining.shape
            self._buffers = render.render_buffers(
                surface, (nx * px, ny * px), float(r.height_nm), st, voxel_nm=(px, px))
            self._buffers_key = key
        return self._buffers

    def _render(self) -> None:
        from litho_sim.metrology.sem import tilt_sem
        from litho_sim.viz.plots import plot_tilt_sem

        if not self.available:
            self.show_placeholder(NO_RENDERER)
            return
        r = self._profile
        assert r is not None
        buffers = self._geometry()
        sem = tilt_sem(buffers, self._cfg)
        self.last = sem
        self._placeholder_art = None
        self._artists.clear()
        plot_tilt_sem(
            sem, buffers, fig=self.figure,
            title=f"Developed resist  ·  {r.label}",
            caption=(f"stage tilt {self._stage.tilt:g}°, rotation {self._stage.azimuth:g}°"
                     f"  ·  {r.summary.replace('    ', ' · ')}"),
        )
        self.ax = self.figure.axes[0]
        self.canvas.draw_idle()

    def show_placeholder(self, message: str) -> None:
        self._profile = None
        self.last = None
        self.figure.clf()
        self.figure.set_facecolor(theme.SURFACE)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self._placeholder_art = theme.placeholder(
            self.ax, message if self.available else NO_RENDERER)
        self.canvas.draw_idle()


class Profile3DView(QtWidgets.QWidget):
    """The developed resist, imaged, beside its section and latent image.

    The micrograph is a :class:`TiltSemView`; the two 2-D panels a
    :class:`ProfilePanels`. Two canvases, one splitter.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sem = TiltSemView()
        self.panels = ProfilePanels()
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.sem)
        self.splitter.addWidget(self.panels)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

    @property
    def figure(self):
        return self.panels.figure

    @property
    def canvas(self):
        return self.panels.canvas

    def show_profile(self, r: Profile3DResult, grid, cfg: SEMConfig | None = None,
                     *, tilt: float | None = None, azimuth: float | None = None) -> None:
        self.sem.show(r, grid, cfg, tilt=tilt, azimuth=azimuth)
        self.panels.show(r, grid)

    def set_stage(self, tilt: float, azimuth: float) -> None:
        self.sem.set_stage(tilt, azimuth)

    def set_instrument(self, cfg: SEMConfig) -> None:
        self.sem.set_instrument(cfg)

    def show_placeholder(self, message: str) -> None:
        self.sem.show_placeholder(message)
        self.panels.show_placeholder(message)

    def shutdown(self) -> None:
        """Nothing to release: the micrograph is a matplotlib canvas."""


class SectionView(_CanvasView):
    """The wafer's material cross-section, with the recipe that built it
    as a strip beside it.

    Usually the most informative single view of a stack: every layer at
    once, in colour, with the swatches that name them.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        #: Show the process-flow strip beside the cross-section. It says the
        #: same thing as the recipe list, in the engine's families rather than
        #: the step vocabulary — worth having and worth being able to put
        #: away, because it costs a quarter of the figure's width whether the
        #: flow is four steps or forty.
        self.show_panel = True
        self.ax_panel = None
        self.ax = None
        self._build_axes()
        self._last: tuple | None = None
        theme.title(self.ax, "Wafer stack")

    def _build_axes(self) -> None:
        if self.show_panel:
            gs = self.figure.add_gridspec(1, 2, width_ratios=[1.15, 3], wspace=0.06)
            self.ax_panel = self.figure.add_subplot(gs[0])
            self.ax = self.figure.add_subplot(gs[1])
        else:
            self.ax_panel = None
            self.ax = self.figure.add_subplot(111)

    def set_panel(self, show: bool) -> None:
        """Show or hide the process-flow strip.

        The strip is one column of a gridspec, so removing it is a relayout
        rather than a visibility flag — hiding the axes would leave its width
        reserved, which is the whole thing being reclaimed.
        """
        if show == self.show_panel:
            return
        self.show_panel = show
        self.figure.clear()
        self._build_axes()
        self._relayout = True
        if self._last is not None:
            self.show_stack(*self._last)
        else:
            theme.title(self.ax, "Wafer stack")
            self._paint()

    def show_stack(self, stack, label: str = "") -> None:
        from litho_sim.viz.viz3d import cross_section_figure, process_flow_panel

        self._last = (stack, label)
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
        theme.title(self.ax, label or "Wafer stack")
        self._relayout = True
        self._paint()

    def show_placeholder(self, message: str) -> None:
        self.ax.clear()
        theme.placeholder(self.ax, message)
        self.ax.set_axis_off()
        if self.ax_panel is not None:
            self.ax_panel.clear()
            self.ax_panel.set_axis_off()
        self._relayout = True
        self._paint()


class StackView(QtWidgets.QWidget):
    """The wafer stack — a material cross-section, or the solid in 3-D.

    Two pages: a :class:`SectionView` and a :class:`SolidView`. Switching is
    a page flip, not a rebuild, so each keeps whatever it was showing — and
    the solid keeps its camera.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section = SectionView()
        self.solid = SolidView()
        self._pages = QtWidgets.QStackedWidget()
        self._pages.addWidget(self.section)
        self._pages.addWidget(self.solid)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._pages)
        self._is_3d = False
        self._last: tuple | None = None

    # The cross-section's figure, for the tab and the tests that read it.
    @property
    def ax(self):
        return self.section.ax

    @property
    def ax_panel(self):
        return self.section.ax_panel

    @property
    def canvas(self):
        return self.section.canvas

    @property
    def figure(self):
        return self.section.figure

    @property
    def show_panel(self) -> bool:
        return self.section.show_panel

    def set_mode(self, three_d: bool, redraw: bool = True) -> None:
        """Swap between the cross-section and the solid."""
        if three_d == self._is_3d:
            return
        self._is_3d = three_d
        self._pages.setCurrentIndex(1 if three_d else 0)
        if redraw and self._last is not None:
            self.show_stack(*self._last)

    def set_panel(self, show: bool) -> None:
        self.section.set_panel(show)

    def show_stack(self, stack, label: str = "", z_exaggeration: float = 1.0,
                   *, reset_camera: bool | None = None) -> None:
        self._last = (stack, label, z_exaggeration)
        if self._is_3d:
            self.solid.show_stack(stack, label or "Wafer stack", z_exaggeration,
                                  reset_camera=reset_camera)
        else:
            self.section.show_stack(stack, label)

    def show_placeholder(self, message: str) -> None:
        self.section.show_placeholder(message)
        self.solid.show_placeholder(message)

    def reframe(self) -> None:
        """Frame the solid afresh — for a newly loaded wafer, not a scrub."""
        self.solid.reset_camera()

    def shutdown(self) -> None:
        self.solid.shutdown()
