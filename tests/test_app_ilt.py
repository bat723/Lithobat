"""
The app's inverse-lithography page: the compute core, and the tab that
animates it.

Four contracts:

* **The solve is the engine's.** The tab's model images the same layout the
  OPC page corrects, so both are aiming at the same drawn edges.
* **Frames stream.** ``compute_ilt`` hands out a frame per iteration, starting
  with the uncorrected print, and each carries enough to draw the state.
* **The tab survives a whole solve.** Frames in, result out, the summary left
  on screen afterwards rather than overwritten by the idle note.
* **Normalisation is forced, and said.** The app defaults to peak
  normalisation, which a gradient cannot follow; the page switches to clear
  and tells the user rather than silently printing something Print would not.

Headless throughout — the Qt tests use the offscreen platform, as the rest of
the app's tests do.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.app.ilt import (
    ILTRequest,
    compute_ilt,
    ilt_availability,
    ilt_print_model,
    normalisation_note,
    target_for,
)
from litho_sim.app.params import ParameterModel
from litho_sim.opc.ilt import ILTFrame

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def params():
    return ParameterModel()


# ---------------------------------------------------------------------------
# The compute core
# ---------------------------------------------------------------------------


def test_the_default_configuration_can_be_solved(params):
    assert ilt_availability(params) is None


def test_normalisation_is_forced_to_clear_and_announced(params):
    """The app defaults to peak; a gradient needs a mask-independent scale."""
    assert params.optics().normalisation.lower() == "peak"
    assert ilt_print_model(params).optics.normalisation.lower() == "clear"
    note = normalisation_note(params)
    assert note is not None
    assert "clear" in note.lower()


def test_the_target_is_the_drawn_layout_on_the_imaging_grid(params):
    model = ilt_print_model(params)
    target = target_for(params, model)
    n = model.grid.n_pixels
    assert target.shape == (n, n)
    assert set(np.unique(target)) <= {0.0, 1.0}
    assert 0 < target.sum() < target.size


def test_compute_ilt_streams_a_frame_per_iteration(params):
    frames: list[ILTFrame] = []
    bundle = compute_ilt(
        ILTRequest(params=params, max_iter=8), progress=frames.append
    )
    # One before the first step, one after each, and the answer re-emitted.
    assert len(frames) >= bundle.result.iterations + 1
    assert frames[0].iteration == 0
    n = ilt_print_model(params).grid.n_pixels
    for f in frames:
        assert f.mask.shape == (n, n)
        assert f.signed.shape == (n, n)
        assert 0.0 <= f.mask.min() <= f.mask.max() <= 1.0
    assert np.allclose(frames[-1].mask, bundle.result.mask)


def test_compute_ilt_reduces_pattern_error(params):
    bundle = compute_ilt(ILTRequest(params=params, max_iter=60))
    assert bundle.pattern_error_after < bundle.pattern_error_before
    assert bundle.seconds > 0


def test_frames_are_thinned_by_min_frame_interval(params):
    """A GUI that cannot keep up drops frames rather than growing a backlog."""
    every: list[ILTFrame] = []
    compute_ilt(ILTRequest(params=params, max_iter=12), progress=every.append)
    thinned: list[ILTFrame] = []
    compute_ilt(
        ILTRequest(params=params, max_iter=12, min_frame_interval=60.0),
        progress=thinned.append,
    )
    assert len(thinned) < len(every)
    # The first frame and the answer always arrive.
    assert thinned[0].iteration == 0
    assert len(thinned) >= 2


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------


@pytest.fixture
def qt_app():
    from litho_sim.app.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_tab_animates_a_solve_and_keeps_its_summary(qt_app, params):
    from litho_sim.app.ilt_tab import ILTTab

    tab = ILTTab(params)
    assert tab.run.isEnabled()

    frames: list[ILTFrame] = []

    def relay(f):
        frames.append(f)
        tab.on_frame(f)

    tab.set_target(target_for(params, ilt_print_model(params)))
    bundle = compute_ilt(ILTRequest(params=params, max_iter=10), progress=relay)

    assert len(tab._history) == len(frames)
    tab._repaint()                      # draws without raising
    tab.on_done(bundle)

    # The finished summary survives the availability refresh that follows it.
    text = tab.status.text()
    assert "pattern error" in text
    assert str(bundle.pattern_error_before) in text
    assert tab.run.isEnabled()


def test_the_tab_reports_a_failed_solve(qt_app, params):
    from litho_sim.app.ilt_tab import ILTTab

    tab = ILTTab(params)
    tab.on_done(None)
    assert "failed" in tab.status.text().lower()
    assert tab.run.isEnabled()


def test_the_window_carries_an_ilt_page(qt_app):
    from litho_sim.app.main import MainWindow

    window = MainWindow()
    try:
        names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert "ILT" in names
        assert window.ilt_tab is not None
    finally:
        window.close()


def test_the_ilt_tab_follows_the_dock(qt_app):
    """A pattern that needs a bigger field says so on the ILT tab the moment it
    is chosen, and Run comes back once the grid is set."""
    from litho_sim.app.main import MainWindow
    from litho_sim.app.params import SPECS_BY_KEY, tab_of

    window = MainWindow()
    try:
        def setv(key, value):
            window.step_tabs[tab_of(SPECS_BY_KEY[key].group)].panel.set_value(key, value)

        assert window.ilt_tab.run.isEnabled()
        setv("pattern", "frame and bars")
        assert not window.ilt_tab.run.isEnabled()
        assert "Grid" in window.ilt_tab.status.text()
        setv("n_pixels", 256)
        setv("pixel_size", 8.0)
        assert window.ilt_tab.run.isEnabled()
        assert "2048 nm" in window.ilt_tab.status.text()
    finally:
        window.close()


def test_the_tab_hands_its_curvature_limit_to_the_request(qt_app, params):
    from litho_sim.app.ilt_tab import ILTTab

    tab = ILTTab(params)
    assert tab.smooth.value() == ILTRequest(params=params).smooth_nm
    tab.smooth.setValue(40.0)
    got: list[ILTRequest] = []
    tab.run_requested.connect(got.append)
    tab._on_run()
    try:
        assert len(got) == 1 and got[0].smooth_nm == 40.0
    finally:
        tab.on_done(None)


def test_compute_ilt_solves_through_the_curvature_limit(params):
    frames: list[ILTFrame] = []
    bundle = compute_ilt(
        ILTRequest(params=params, max_iter=6, smooth_nm=16.0), progress=frames.append
    )
    assert bundle.pattern_error_after <= bundle.pattern_error_before
    for f in frames:
        assert 0.0 <= f.mask.min() <= f.mask.max() <= 1.0
