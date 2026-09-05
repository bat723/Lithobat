"""The solid view keeps its promises: one actor per material, updated in
place; a camera that is framed once and then left alone; z exaggeration as
a scale, not a remesh.

Everything is asserted through VTK's object model and never through pixels:
under the offscreen platform there is no OpenGL context, and the render
window is a no-op until a real one exists — which is exactly why these tests
can run on a machine with no display.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

needs_qt = pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None
    and importlib.util.find_spec("PyQt6") is None
    and importlib.util.find_spec("PySide2") is None
    and importlib.util.find_spec("PyQt5") is None,
    reason="needs a Qt binding",
)
needs_vtk = pytest.mark.skipif(
    importlib.util.find_spec("pyvistaqt") is None
    or importlib.util.find_spec("vtkmodules") is None,
    reason="needs pyvista + pyvistaqt: pip install -e '.[viz3d]'",
)


@pytest.fixture
def qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from litho_sim.app.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def wafer():
    from litho_sim.core.config import GridConfig
    from litho_sim.wafer import Stack

    grid = GridConfig(n_pixels=24, pixel_size=4e-9)
    stack = Stack.blank(grid, dz=4e-9, substrate_thickness=24e-9, headroom=120e-9)
    stack.deposit_blanket("SiO2", 24e-9)
    stack.deposit_blanket("photoresist", 40e-9)
    return stack


def _view(off_screen=True):
    import pyvista as pv

    pv.OFF_SCREEN = True
    from litho_sim.app.solid_view import SolidView

    return SolidView(off_screen=off_screen)


@needs_qt
@needs_vtk
def test_one_actor_per_material_in_the_material_colour(qapp, wafer):
    from litho_sim.viz.viz3d import voxel_mesh

    view = _view()
    try:
        view.show_stack(wafer, "test wafer")
        present = {m.id: m for m in wafer.present_materials()}
        assert set(view.actors) == set(present)
        for mid, actor in view.actors.items():
            mesh = voxel_mesh(wafer, present[mid], 1)
            assert mesh is not None
            assert actor.mapper.dataset.n_cells == len(mesh[1])
            want = np.array(pv_rgb(present[mid].color))
            assert np.allclose(np.array(actor.prop.color.float_rgb), want, atol=1e-2)
        assert view.page == "scene"
        assert view.title == "test wafer"
        assert view.legend_names() == [m.name for m in wafer.present_materials()]
    finally:
        view.shutdown()


def pv_rgb(hex_colour: str) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    return to_rgb(hex_colour)


@needs_qt
@needs_vtk
def test_a_second_draw_leaves_the_camera_alone(qapp, wafer):
    view = _view()
    try:
        def pose(cam):
            return (tuple(cam.position), tuple(cam.focal_point), tuple(cam.up),
                    float(cam.parallel_scale))

        view.show_stack(wafer, "first")
        before = pose(view.camera)
        taller = wafer.copy()
        taller.deposit_blanket("SiN", 40e-9)
        view.show_stack(taller, "second")
        # The pose is untouched; only the clipping range follows the bounds.
        assert pose(view.camera) == before, "scrubbing must not move the camera"
        view.show_stack(taller, "third", reset_camera=True)
        assert pose(view.camera) != before, "asking for a reframe reframes"
    finally:
        view.shutdown()


@needs_qt
@needs_vtk
def test_z_exaggeration_scales_the_actor_and_not_the_mesh(qapp, wafer):
    view = _view()
    try:
        view.show_stack(wafer, "w")
        actor = next(iter(view.actors.values()))
        mesh = actor.mapper.dataset
        z_top = float(mesh.points[:, 2].max())
        view.set_z_exaggeration(3.0)
        assert tuple(actor.scale) == pytest.approx((1.0, 1.0, 3.0))
        assert actor.mapper.dataset is mesh, "no remesh"
        assert float(mesh.points[:, 2].max()) == z_top
        view.show_stack(wafer, "w", z_exaggeration=1.5)
        assert tuple(actor.scale) == pytest.approx((1.0, 1.0, 1.5))
    finally:
        view.shutdown()


@needs_qt
@needs_vtk
def test_a_material_that_vanishes_loses_its_actor(qapp, wafer):
    from litho_sim.wafer import get_material

    view = _view()
    try:
        view.show_stack(wafer, "with resist")
        resist = get_material("photoresist").id
        assert resist in view.actors
        wafer.strip("photoresist")
        view.show_stack(wafer, "stripped")
        assert resist not in view.actors
        plotter = view.plotter
        assert plotter is not None
        assert f"mat{resist}" not in plotter.renderer.actors
        assert "photoresist" not in view.legend_names()
    finally:
        view.shutdown()


@needs_qt
@needs_vtk
def test_placeholder_hides_the_scene_and_the_scene_comes_back(qapp, wafer):
    view = _view()
    try:
        view.show_placeholder("nothing yet")
        assert view.page == "placeholder"
        assert view.actors == {}
        view.show_stack(wafer, "w")
        assert view.page == "scene"
        view.show_placeholder("gone again")
        assert view.page == "placeholder" and view.actors == {}
    finally:
        view.shutdown()


@needs_qt
@needs_vtk
def test_the_scale_bar_is_a_round_number_of_the_field(qapp, wafer):
    from litho_sim.viz.viz3d import nice_length

    view = _view()
    try:
        view.show_stack(wafer, "w")
        ny, nx = wafer.shape_xy
        assert view.scale_bar_nm == nice_length(nx * wafer.pixel_size * 1e9)
    finally:
        view.shutdown()


@needs_qt
@needs_vtk
def test_shutdown_twice_does_not_raise(qapp, wafer):
    view = _view()
    view.show_stack(wafer, "w")
    view.shutdown()
    view.shutdown()
    assert view.plotter is None
    assert view.camera is None


@needs_qt
def test_without_vtk_the_widget_is_a_placeholder_and_the_radio_is_greyed(qapp, wafer, monkeypatch):
    from litho_sim.app import solid_view

    monkeypatch.setattr(solid_view.SolidView, "available", False)
    view = solid_view.SolidView()
    try:
        assert view.plotter is None
        assert view.camera is None
        assert solid_view.INSTALL_HINT.splitlines()[0] in view._note.text()
        view.show_stack(wafer, "w")           # a no-op, not an error
        assert view.actors == {}
        view.set_z_exaggeration(2.0)
        view.reset_camera()
    finally:
        view.shutdown()

    from litho_sim.app.params import ParameterModel
    from litho_sim.app.stack_tab import StackTab

    tab = StackTab(ParameterModel())
    try:
        assert not tab.mode_solid.isEnabled()
        assert "viz3d" in tab.mode_solid.toolTip()
        tab.load_stack(wafer, label="w", solid=True)
        assert tab.mode_section.isChecked(), "no 3-D to switch to"
    finally:
        tab.view.shutdown()
