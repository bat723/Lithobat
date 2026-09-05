"""The solid view: a wafer stack rendered by VTK, inside the Qt window.

Why VTK
-------
The app's first 3-D views were matplotlib's ``mplot3d``. Matplotlib has no
depth buffer: every frame it projects every polygon and sorts them in
Python, so cost tracks triangle count and a sectioned transistor at full
voxel detail was six seconds a frame. Merging coplanar faces bought a
settled frame of ~100 ms at the price of sort artefacts, and a hundred lines
of coarsen-while-dragging and blitting machinery bought the rest. The vault
note *Six Seconds a Frame* ends with the honest conclusion: past here needs
a real depth buffer. This is it. The same 300 k-triangle mesh draws in 8 ms.

What the widget promises
------------------------
* **One actor per material**, updated in place. :meth:`show_stack` swaps
  each actor's mesh, adds actors for materials that appeared and removes
  those that vanished; the scene is never rebuilt.
* **The camera is yours.** It is framed once, on the first draw (or when
  asked with ``reset_camera=True``), and never touched again — scrubbing a
  flow or re-running a develop keeps the angle you set. This is the property
  the Plotly-in-Streamlit detour could not deliver and the reason the app
  moved to Qt in the first place.
* **Z exaggeration is a scale on the actors**, not a remesh, so the slider
  is live.
* **Exact geometry.** The mesh is :func:`~litho_sim.viz.viz3d.voxel_mesh` at
  stride 1 — every exposed voxel face. No greedy merge, no ``MAX_SPAN``, no
  detail floor: those existed to keep a software painter's algorithm
  tolerable, and a z-buffer does not care.

Without VTK
-----------
``pyvista`` and ``pyvistaqt`` are the ``viz3d`` extra. When they are not
importable the widget is a placeholder that says so, every scene method is a
no-op, and :attr:`SolidView.available` is ``False`` so the tabs can grey
their 3-D controls. Nothing else in the app has to know.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import numpy as np

from litho_sim.app.qt import QtCore, QtGui, QtWidgets
from litho_sim.viz import theme

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "The 3-D view needs VTK — install the viz3d extra:\n"
    "    pip install -e '.[viz3d]'\n"
    "(pyvista and pyvistaqt)"
)

#: Camera framing that reads like a technical drawing: a little above the
#: wafer, looking across the front-right corner. Degrees.
_ELEV = 22.0
_AZIM = -62.0
#: How much of matplotlib's margin to reclaim after framing.
_ZOOM = 1.18

_FRAME_COLOUR = "#b8bcc4"
_BAR_COLOUR = "#33373d"


def available() -> bool:
    """Whether the GPU renderer can be imported at all."""
    return all(importlib.util.find_spec(m) is not None
               for m in ("pyvista", "pyvistaqt", "vtkmodules"))


def install_surface_format() -> None:
    """Ask Qt for a core-profile OpenGL context before the application exists.

    Mandatory on macOS: a ``QOpenGLWidget`` (which is what pyvistaqt's render
    widget is) gets the *default* surface format, and the default has to be
    set before the first top-level window is created. Harmless elsewhere.
    A no-op when VTK is not installed — a plain Qt app has no use for it.
    """
    if not available():
        return
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 2)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(0)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


def _polydata(V: np.ndarray, F: np.ndarray):
    """A triangle mesh from ``voxel_mesh``'s ``(V, F)``."""
    import pyvista as pv

    V = np.ascontiguousarray(V, dtype=np.float32)
    F = np.ascontiguousarray(F, dtype=np.int64)
    if hasattr(pv.PolyData, "from_regular_faces"):
        return pv.PolyData.from_regular_faces(V, F)
    padded = np.column_stack([np.full(len(F), 3, dtype=np.int64), F]).ravel()
    return pv.PolyData(V, faces=padded)


def meshes_for(stack) -> list[tuple[Any, Any]]:
    """``[(Material, PolyData)]`` for every material present in *stack*.

    A pure function of the stack, kept apart from the widget so it could
    move onto the worker thread if extraction ever became the cost — today
    it is ~25 ms for a sectioned GAA and the draw is free, so it runs where
    the stack arrives.
    """
    from litho_sim.viz.viz3d import voxel_mesh

    out = []
    for m in stack.present_materials():
        mesh = voxel_mesh(stack, m, 1)
        if mesh is None:
            continue
        V, F = mesh
        out.append((m, _polydata(V, F)))
    return out


class SolidView(QtWidgets.QWidget):
    """A wafer stack as a lit solid, one actor per material.

    Parameters
    ----------
    off_screen : bool, optional
        Render without an interactor. For tests, which never show the widget.
    multi_samples : int
        MSAA samples on VTK's framebuffer.
    """

    #: Whether the renderer imported. The tabs read this to grey their 3-D
    #: controls; the widget itself degrades to a placeholder either way.
    available: bool = available()

    def __init__(self, parent=None, *, off_screen: bool | None = None,
                 multi_samples: int = 4) -> None:
        super().__init__(parent)
        self._plotter = None
        self._actors: dict[int, Any] = {}
        self._frame: Any = None
        self._bar: Any = None
        self._bar_label: Any = None
        self._extent: tuple[float, float] | None = None
        self._zex = 1.0
        self._drawn = False
        self._closed = False
        #: A reframe asked for while the widget had no real size yet. VTK
        #: fits the camera to the viewport's aspect, and a hidden tab's
        #: viewport is a few pixels; the fit is redone on first show.
        self._reframe_pending = False

        self._pages = QtWidgets.QStackedLayout(self)
        self._pages.setContentsMargins(0, 0, 0, 0)

        # Both pages are paper, like the figures beside them: the app's own
        # palette may be dark, and ink set for paper vanishes on it.
        self._note = QtWidgets.QLabel(INSTALL_HINT if not self.available else "")
        self._note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._note.setWordWrap(True)
        self._note.setAutoFillBackground(True)
        self._note.setStyleSheet(
            f"background: {theme.SURFACE}; color: {theme.INK2}; padding: 12px;")
        self._pages.addWidget(self._note)

        scene = QtWidgets.QWidget()
        scene.setObjectName("solidScene")
        scene.setAutoFillBackground(True)
        scene.setStyleSheet(f"#solidScene {{ background: {theme.SURFACE}; }}")
        col = QtWidgets.QVBoxLayout(scene)
        col.setContentsMargins(4, 2, 4, 2)
        col.setSpacing(2)
        self._title = QtWidgets.QLabel("")
        self._title.setStyleSheet(
            f"color: {theme.INK}; font-weight: 500; padding: 0 2px;")
        self._caption = QtWidgets.QLabel("")
        self._caption.setStyleSheet(
            f"color: {theme.INK2}; font-size: 11px; padding: 0 2px;")
        self._caption.setVisible(False)
        col.addWidget(self._title)
        col.addWidget(self._caption)
        self._legend = QtWidgets.QHBoxLayout()
        self._legend.setContentsMargins(2, 2, 2, 2)
        self._legend.setSpacing(10)
        self._legend.addStretch(1)

        if self.available:
            from pyvistaqt import QtInteractor

            self._plotter = QtInteractor(
                scene, off_screen=off_screen, multi_samples=multi_samples,
                auto_update=False, lighting="light kit",
            )
            self._plotter.set_background(theme.SURFACE)  # type: ignore[arg-type]
            self._plotter.enable_parallel_projection()  # type: ignore[call-arg]
            if self._plotter.iren is not None:
                try:
                    # pyvista binds 'q' to closing the plotter, which in a
                    # docked widget means a blank panel with no way back.
                    self._plotter.clear_events_for_key("q")  # type: ignore[arg-type]
                except Exception:                     # noqa: BLE001
                    logger.debug("could not unbind 'q'", exc_info=True)
            self._plotter.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            self._plotter.setMinimumSize(240, 200)
            col.addWidget(self._plotter, 1)
        col.addLayout(self._legend)
        self._pages.addWidget(scene)
        self._pages.setCurrentIndex(0)

    # -- introspection -------------------------------------------------
    @property
    def plotter(self):
        return self._plotter

    @property
    def actors(self) -> dict[int, Any]:
        """Material id → actor, for what is on screen now."""
        return dict(self._actors)

    @property
    def page(self) -> str:
        return "scene" if self._pages.currentIndex() == 1 else "placeholder"

    def legend_names(self) -> list[str]:
        names = []
        for i in range(self._legend.count()):
            w = self._legend.itemAt(i).widget()
            if isinstance(w, QtWidgets.QLabel) and w.property("material"):
                names.append(w.text())
        return names

    @property
    def title(self) -> str:
        return self._title.text()

    @property
    def caption(self) -> str:
        return self._caption.text()

    # -- camera ---------------------------------------------------------
    @property
    def camera(self):
        """A copy of the camera, or ``None`` without a renderer."""
        if self._plotter is None:
            return None
        return self._plotter.camera.copy()

    @camera.setter
    def camera(self, cam) -> None:
        if self._plotter is None or cam is None:
            return
        self._plotter.camera = cam.copy()
        self._render()

    def reset_camera(self) -> None:
        """Frame whatever is on screen from the standard angle."""
        p = self._plotter
        if p is None:
            return
        el, az = np.radians(_ELEV), np.radians(_AZIM)
        direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                              np.sin(el)])
        b = p.renderer.bounds
        centre = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1.0)
        p.camera_position = [tuple(centre + 3.0 * span * direction), tuple(centre),
                             (0.0, 0.0, 1.0)]
        p.reset_camera(render=False)  # type: ignore[call-arg]
        p.camera.zoom(_ZOOM)
        # Fitting used the viewport's aspect. Hidden, the viewport is not
        # the size it will be shown at, so the fit is repeated on show.
        self._reframe_pending = not self._has_real_size()
        self._render()

    def _has_real_size(self) -> bool:
        p = self._plotter
        return (p is not None and p.isVisible() and p.width() > 60
                and p.height() > 60)

    def _reframe_if_pending(self) -> None:
        if self._reframe_pending and self._has_real_size():
            self._reframe_pending = False
            self.reset_camera()

    def showEvent(self, event):         # noqa: N802 - Qt's spelling
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._reframe_if_pending)

    def resizeEvent(self, event):       # noqa: N802 - Qt's spelling
        super().resizeEvent(event)
        self._reframe_if_pending()

    # -- scene ----------------------------------------------------------
    def show_placeholder(self, text: str) -> None:
        """Say why there is nothing to look at, in the space the solid will use."""
        self._note.setText(text if self.available else INSTALL_HINT)
        self._pages.setCurrentIndex(0)
        if self._plotter is not None:
            for actor in list(self._actors.values()):
                self._plotter.remove_actor(actor, reset_camera=False, render=False)
            self._actors.clear()
            self._drop_chrome()
            self._drawn = False
            self._render()

    def show_stack(self, stack, label: str = "", z_exaggeration: float = 1.0,
                   *, caption: str = "", reset_camera: bool | None = None) -> None:
        """Draw *stack*, updating the scene in place.

        Parameters
        ----------
        stack : Stack
        label : str
            The title over the picture.
        z_exaggeration : float
            Vertical scale applied to the actors — the mesh is untouched.
        caption : str
            The numbers row under the title.
        reset_camera : bool, optional
            Frame the scene afresh. ``None`` frames only the first draw and
            leaves the camera alone afterwards.
        """
        self._title.setText(label)
        self._caption.setText(caption)
        self._caption.setVisible(bool(caption))
        self._pages.setCurrentIndex(1)
        p = self._plotter
        if p is None:
            return

        present: list = []
        for m, mesh in meshes_for(stack):
            present.append(m)
            actor = self._actors.get(m.id)
            if actor is None:
                actor = p.add_mesh(
                    mesh, color=m.color, opacity=1.0, smooth_shading=False,
                    name=f"mat{m.id}", reset_camera=False, render=False,
                    culling=False,
                )
                self._actors[m.id] = actor
            else:
                actor.mapper.dataset = mesh
                actor.mapper.scalar_visibility = False
                actor.prop.color = m.color
        keep = {m.id for m in present}
        for mid in [k for k in self._actors if k not in keep]:
            p.remove_actor(self._actors.pop(mid), reset_camera=False, render=False)

        ny, nx = stack.shape_xy
        extent = (nx * stack.pixel_size * 1e9, ny * stack.pixel_size * 1e9)
        if extent != self._extent:
            self._extent = extent
            self._draw_chrome(*extent)
        self._set_scale(float(z_exaggeration))
        self._set_legend(present)

        if reset_camera or (reset_camera is None and not self._drawn):
            self._drawn = True
            self.reset_camera()
        else:
            p.renderer.ResetCameraClippingRange()
            self._render()

    def set_z_exaggeration(self, z: float) -> None:
        """Stretch the picture vertically. No remesh; the slider is live."""
        if self._plotter is None or not self._actors:
            self._zex = float(z)
            return
        self._set_scale(float(z))
        self._plotter.renderer.ResetCameraClippingRange()
        self._render()

    # -- helpers --------------------------------------------------------
    def _set_scale(self, z: float) -> None:
        self._zex = z
        for actor in self._actors.values():
            actor.scale = (1.0, 1.0, z)

    def _render(self) -> None:
        if self._plotter is not None and not self._closed:
            self._plotter.render()

    def _drop_chrome(self) -> None:
        p = self._plotter
        for name in ("_frame", "_bar", "_bar_label"):
            actor = getattr(self, name)
            if actor is not None and p is not None:
                p.remove_actor(actor, reset_camera=False, render=False)
            setattr(self, name, None)
        self._extent = None

    def _draw_chrome(self, x1: float, y1: float) -> None:
        """A hairline floor outline and one round scale bar in world units.

        Under an orthographic camera a world length is honest at any zoom,
        and at z = 0 the bar is untouched by the actors' z scale. This is
        the whole framing: no panes, no ticks, no triad — the solid shows
        its own extent.
        """
        import pyvista as pv

        from litho_sim.viz.viz3d import nice_length

        p = self._plotter
        self._drop_chrome()
        if p is None:
            return
        self._extent = (x1, y1)
        floor = pv.PolyData(
            np.array([[0, 0, 0], [x1, 0, 0], [x1, y1, 0], [0, y1, 0]], dtype=float),
            lines=np.array([5, 0, 1, 2, 3, 0]),
        )
        self._frame = p.add_mesh(floor, color=_FRAME_COLOUR, line_width=1.0,
                                 name="frame", reset_camera=False, render=False,
                                 lighting=False)
        length = nice_length(x1)
        # Well in front of the wafer, so the front face never sits over it.
        y_bar = -0.12 * y1
        bar = pv.Line((0.0, y_bar, 0.0), (length, y_bar, 0.0))
        self._bar = p.add_mesh(bar, color=_BAR_COLOUR, line_width=3.0,
                               name="scale_bar", reset_camera=False,
                               render=False, lighting=False)
        self._bar_label = p.add_point_labels(
            np.array([[length / 2.0, y_bar - 0.05 * y1, 0.0]]), [f"{length:g} nm"],
            show_points=False, shape=None, always_visible=True,
            text_color=_BAR_COLOUR, font_size=11, name="scale_label",
            reset_camera=False, render=False,
        )
        self._bar_length = length

    @property
    def scale_bar_nm(self) -> float | None:
        return getattr(self, "_bar_length", None) if self._bar is not None else None

    def _set_legend(self, materials) -> None:
        while self._legend.count() > 1:
            item = self._legend.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for i, m in enumerate(materials):
            swatch = QtWidgets.QLabel("")
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background: {m.color}; border-radius: 2px;")
            name = QtWidgets.QLabel(m.name)
            name.setProperty("material", True)
            name.setStyleSheet(f"color: {theme.INK2}; font-size: 11px;")
            self._legend.insertWidget(2 * i, swatch)
            self._legend.insertWidget(2 * i + 1, name)

    # -- lifecycle ------------------------------------------------------
    def shutdown(self) -> None:
        """Release the render window. Idempotent; safe before the app exits.

        VTK tearing down its GL objects *after* Qt has destroyed the context
        is the classic exit-time crash; the main window calls this first.
        """
        if self._closed:
            return
        self._closed = True
        p, self._plotter = self._plotter, None
        self._actors.clear()
        if p is not None:
            try:
                p.close()
            except Exception:                         # noqa: BLE001
                logger.debug("plotter close failed", exc_info=True)

    def closeEvent(self, event):        # noqa: N802 - Qt's spelling
        self.shutdown()
        super().closeEvent(event)


__all__ = ["INSTALL_HINT", "SolidView", "available", "install_surface_format",
           "meshes_for"]
