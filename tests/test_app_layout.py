"""
The app's layout after the step-tab reorganisation.

One tab per processing step carrying that step's controls beside its picture;
nothing physical computes while a control moves; the Simulate tab runs
things and the results land on the step tabs with a banner the moment the
settings move on; an SEM tab images what was run. Headless where it can be,
offscreen Qt where it cannot.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.app.compute import film_preview, mask_preview, source_preview
from litho_sim.app.params import (
    GROUPS,
    GROUPS_3D,
    SPECS,
    SPECS_BY_KEY,
    TAB_GROUPS,
    TABS,
    ParameterModel,
    tab_of,
)


def _qt_binding_available() -> bool:
    import importlib.util

    return any(
        importlib.util.find_spec(m) is not None
        for m in ("PySide6", "PyQt6", "PySide2", "PyQt5")
    )


needs_qt = pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")

#: What the tab bar must read, left to right.
EXPECTED_TABS = [
    "Mask", "Source", "Resist", "Expose", "Bake", "Develop",
    "Wafer Stack", "Simulate", "SEM",
]


def _fast(win) -> None:
    """Settings a Print finishes in milliseconds at."""
    win.model.set("n_pixels", 64)
    win.model.set("source_grid", 5)
    win.model.set("n_z_slices", 3)


def _wait(app, until, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert until(), "timed out waiting for the worker"


# ---------------------------------------------------------------------------
# The declaration: every knob on exactly one step tab
# ---------------------------------------------------------------------------


def test_every_section_is_on_exactly_one_tab():
    seen: dict[str, str] = {}
    for tab, groups in TAB_GROUPS.items():
        for g in groups:
            assert g not in seen, f"section {g!r} is on both {seen[g]} and {tab}"
            seen[g] = tab
    assert tuple(seen) == GROUPS, "GROUPS must be the flat view of TAB_GROUPS"
    assert set(GROUPS_3D) <= set(GROUPS)
    for g in GROUPS:
        assert tab_of(g) == seen[g]
    with pytest.raises(KeyError):
        tab_of("Profile")


def test_every_knob_is_in_a_section_that_is_on_a_tab():
    for s in SPECS:
        assert s.group in GROUPS, f"{s.key} is in section {s.group!r}, which is on no tab"


def test_the_steps_read_in_the_order_the_physics_runs():
    assert TABS == ("Mask", "Source", "Resist", "Expose", "Bake", "Develop")


def test_the_source_tab_holds_all_of_the_optics_and_the_vector_model():
    keys = {s.key for s in SPECS if tab_of(s.group) == "Source"}
    for k in ("NA", "wavelength", "sigma_outer", "source_type", "source_grid",
              "defocus", "imaging_model", "polarisation", "n_image",
              "exact_defocus", "normalisation"):
        assert k in keys, f"{k} belongs on the Source tab"


def test_the_mask_tab_holds_the_mask_and_only_the_mask():
    keys = {s.key for s in SPECS if tab_of(s.group) == "Mask"}
    assert {"pattern", "pitch", "cd", "mask_type", "mask_model", "n_pixels"} <= keys
    assert not any(SPECS_BY_KEY[k].target == "resist" for k in keys)


# ---------------------------------------------------------------------------
# The live previews — pictures of the settings, not results
# ---------------------------------------------------------------------------


def test_mask_preview_is_real_and_tracks_the_grid():
    m = ParameterModel()
    m.set("n_pixels", 64)
    m.set("mask_type", "att-psm")
    mask = mask_preview(m)
    assert mask.shape == (64, 64)
    assert not np.iscomplexobj(mask)


def test_source_preview_places_the_first_orders():
    m = ParameterModel()          # 193 nm, NA 0.93, pitch 200 nm, σ 0.8, grid 21
    p = source_preview(m)
    assert p.source.shape == (21, 21)
    assert p.n_points > 0
    assert p.order_shift == pytest.approx(193.0 / (200.0 * 0.93), rel=1e-6)
    assert p.k1 == pytest.approx(100.0 * 0.93 / 193.0, rel=1e-6)
    assert "197 source points" in p.label or "source points" in p.label

    m.set("pattern", "isolated line")
    assert source_preview(m).order_shift is None, "an isolated line has no pitch"
    m.set("pattern", "lines and spaces")
    m.set("source_grid", 5)
    assert source_preview(m).source.shape == (5, 5), "the grid is the real sampling"


def test_film_preview_reads_the_coat_step():
    m = ParameterModel()
    m.set("thickness", 120.0)
    m.set("dz", 4.0)
    m.set("n_z_slices", 7)
    m.set("dill_C", 0.05)
    m.set("dose_nominal", 40.0)
    f = film_preview(m)
    assert f.n_voxels == 30
    assert f.n_planes == 7 and f.plane_z_nm.size == 7
    assert f.pac[0] == pytest.approx(1.0)
    e = f.dose_axis
    assert np.allclose(f.pac, np.exp(-0.05 * e))
    assert f.dose_to_clear == 40.0


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


@needs_qt
def test_the_window_builds_with_one_tab_per_step(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.compute import compute_imaging
    from litho_sim.app.main import MainWindow

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == EXPECTED_TABS
        assert win.tabs.currentWidget() is win.step_tabs["Mask"], "opens on the first step"

        # Every spec has exactly one widget, on exactly one tab — and there
        # is no parameter dock any more.
        seen: dict[str, str] = {}
        for name, tab in win.step_tabs.items():
            for key in tab.panel.keys():
                assert key not in seen, f"{key} is on both {seen[key]} and {name}"
                seen[key] = name
        assert set(seen) == set(SPECS_BY_KEY)
        assert not win.findChildren(QtWidgets.QDockWidget)

        # Drive a slider and a combo the way a user would, on their tab.
        win.step_tabs["Source"].panel._emit("NA", 1.20)
        assert win.model["NA"] == pytest.approx(1.20)
        win.step_tabs["Source"].panel._emit("polarisation", "te")
        assert win.model["polarisation"] == "te"

        # Draw a result into every step view twice: artist creation, then reuse.
        _fast(win)
        result = compute_imaging(win.model)
        for view in (win.expose_view, win.bake_view, win.develop_view):
            view.show_result(result)
            view.show_result(result)
            assert view.canvas is not None
        win.simulate_tab.print_page.show_result(result)
    finally:
        win.thread.quit()
        win.thread.wait(2000)
        win.close()


@needs_qt
def test_moving_a_control_computes_nothing(monkeypatch):
    """The whole point of the reorganisation: settings are settings.

    A slider on a step tab changes the model and the drawings of the settings
    — the mask, the source, the film — and asks the worker for nothing.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app import worker as worker_mod
    from litho_sim.app.main import MainWindow

    runs: list[str] = []
    for name in ("run", "run_3d", "run_fem", "run_stoch"):
        monkeypatch.setattr(
            worker_mod.Worker, name,
            lambda self, *_a, _n=name, **_k: runs.append(_n),
        )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        before = win.mask_view.ax.get_title()
        win.step_tabs["Mask"].panel._emit("n_pixels", 64)
        win.step_tabs["Mask"].panel._emit("pitch", 240.0)
        win.step_tabs["Source"].panel._emit("NA", 1.10)
        win.step_tabs["Develop"].panel._emit("threshold", 0.5)
        win.step_tabs["Resist"].panel._emit("thickness", 150.0)
        for _ in range(20):
            app.processEvents()
        assert runs == [], f"moving a control asked the worker to {runs}"
        assert win.model["n_pixels"] == 64 and win.model["NA"] == pytest.approx(1.10)
        # The drawings of the settings did follow.
        assert "64×64" in win.mask_view.ax.get_title()
        assert win.mask_view.ax.get_title() != before
        assert "source points" in win.source_view.ax.get_title()
        assert "150 nm" in win.resist_view.figure._suptitle.get_text()
        # And the result views still say nothing has been printed.
        assert not win.step_tabs["Expose"].stale
        assert win.expose_view._artists == {}
    finally:
        win.thread.quit()
        win.thread.wait(2000)
        win.close()


@needs_qt
def test_print_lands_on_every_step_tab_and_dates_when_the_settings_move(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        _fast(win)
        page = win.simulate_tab.print_page
        assert "ms" in page.cost_label.text() or "s" in page.cost_label.text()
        page.run_btn.click()
        assert not page.run_btn.isEnabled(), "Run must latch while busy"
        _wait(app, lambda: win._last_result is not None and not win._busy_print)
        assert page.run_btn.isEnabled()

        # Drawn into all three step views, whichever tab is showing.
        assert win.tabs.currentWidget() is win.step_tabs["Mask"]
        for view in (win.expose_view, win.bake_view, win.develop_view):
            assert view._artists, "a step view was left blank after Print"
        assert "CD" in win.status.currentMessage()
        for name in ("Expose", "Bake", "Develop"):
            assert not win.step_tabs[name].stale

        # A develop knob dates the develop and bake pictures, not the aerial image.
        win.step_tabs["Develop"].panel._emit("threshold", 0.55)
        assert win.step_tabs["Develop"].stale
        assert win.step_tabs["Bake"].stale
        assert not win.step_tabs["Expose"].stale
        assert "Run again" in page.notes.text() or "changed" in page.notes.text()
        # An optics knob dates everything.
        win.step_tabs["Source"].panel._emit("NA", 1.05)
        assert win.step_tabs["Expose"].stale
        # Printing again clears every banner.
        page.run_btn.click()
        _wait(app, lambda: not win._busy_print)
        for name in ("Expose", "Bake", "Develop"):
            assert not win.step_tabs[name].stale
        assert "CD" in page.notes.text()
        assert win.sem_tab.last is not None, "the SEM tab imaged the print"
    finally:
        win.thread.quit()
        win.thread.wait(2000)
        win.close()


@needs_qt
def test_the_3d_profile_run_lands_on_develop_in_3d_and_feeds_the_sem(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        _fast(win)
        win.model.set("n_pixels", 48)
        assert not win.stack_tab.import_btn.isEnabled()
        win.simulate_tab.select("3-D resist profile")
        assert win.simulate_tab.current_kind == "3-D resist profile"
        win.simulate_tab.run_current()
        _wait(app, lambda: win._profile is not None and not win._busy_3d, 60.0)

        assert win.mode_3d.isChecked(), "asking for the profile is asking to see it"
        assert win.stack_tab.import_btn.isEnabled()
        assert "remaining" in win.status.currentMessage()
        assert not win.step_tabs["Develop"].stale

        # The SEM tab can image the volume, top-down and in cross-section.
        win.sem_tab.source.setCurrentIndex(1)
        assert win.sem_tab.last is not None and win.sem_tab.last.mode == "topdown"
        win.sem_tab.mode_xs.setChecked(True)
        assert win.sem_tab.last.mode == "xsection"
        assert win.sem_tab.last.row_origin < 0.0

        # A 3-D-only knob dates the profile but not the 2-D pictures.
        win.step_tabs["Develop"].panel._emit("develop_model", "mack")
        assert win.step_tabs["Develop"].stale
        assert not win.step_tabs["Expose"].stale
        assert not win.sem_tab.banner.isHidden()
    finally:
        win.thread.quit()
        win.thread.wait(2000)
        win.close()


@needs_qt
def test_the_simulate_menu_runs_from_any_tab(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        _fast(win)
        labels = [a.text() for a in win.simulate_menu.actions()]
        assert labels == ["Print", "3-D resist profile", "Focus-exposure matrix",
                          "Stochastic printing"]
        win.tabs.setCurrentWidget(win.step_tabs["Bake"])
        win.simulate_actions["Print"].trigger()
        assert win.tabs.currentWidget() is win.simulate_tab
        assert win.simulate_tab.current_kind == "Print"
        _wait(app, lambda: win._last_result is not None and not win._busy_print)
    finally:
        win.thread.quit()
        win.thread.wait(2000)
        win.close()


@needs_qt
def test_a_failed_run_releases_the_button(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app import worker as worker_mod
    from litho_sim.app.main import MainWindow

    def boom(*_a, **_k):
        raise RuntimeError("deliberate")

    monkeypatch.setattr(worker_mod, "compute_imaging", boom)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        _fast(win)
        page = win.simulate_tab.print_page
        page.run_btn.click()
        _wait(app, lambda: not win._busy_print)
        assert page.run_btn.isEnabled(), "a failure must not wedge the button"
        assert "deliberate" in page.notes.text()
        assert "error" in win.status.currentMessage()
        assert win._last_result is None
    finally:
        win.thread.quit()
        win.thread.wait(2000)
        win.close()


@needs_qt
def test_the_sem_tab_measures_the_print_and_follows_its_instrument(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        assert win.sem_tab.last is None, "nothing to image before a print"
        _fast(win)
        win.model.set("n_pixels", 96)
        win.model.set("pitch", 256.0)
        win.model.set("cd", 128.0)
        win.simulate_tab.print_page.run_btn.click()
        _wait(app, lambda: win._last_result is not None and not win._busy_print)

        tab = win.sem_tab
        assert tab.last is not None and tab.measurement is not None
        assert tab.measurement.found
        assert "SEM CD" in tab.readout.text()
        assert "half height" in tab.readout.text()
        px = win.model.grid().pixel_size
        # Within two pixels of the profile's own half-height width — the
        # readout shows both, and the difference is the bloom bias.
        import re

        nums = [float(v) for v in re.findall(r"(\d+\.\d) nm", tab.readout.text())]
        assert len(nums) >= 2
        assert abs(nums[0] - nums[1]) <= 2 * px * 1e9

        before = tab.last.image.copy()
        tab.form._set("beam_fwhm", 9.0)
        assert not np.array_equal(tab.last.image, before), "the instrument is live"
        assert tab.config().beam_fwhm == pytest.approx(9e-9)
        tab.form._set("seed", 5)
        a = tab.last.image.copy()
        tab.form._set("seed", 5)
        assert np.array_equal(tab.last.image, a), "same seed, same grain"

        tab.mode_xs.setChecked(True)
        assert tab.last.mode == "xsection"
        assert tab.readout.text() == ""
    finally:
        win.thread.quit()
        win.thread.wait(2000)
        win.close()


@needs_qt
def test_scrolling_a_step_tab_does_not_touch_any_parameter(monkeypatch):
    """The wheel guard, on every panel it now lives on.

    Qt hands the wheel to sliders and combo boxes by default, so scrolling a
    panel past a control would edit it. Measured before the guard existed:
    one scroll down the old dock altered 20 of 25 parameters. Both halves
    are asserted — the wheel reaches the scroll area, and no value moves.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        before = dict(win.model.values)
        emitted: list[str] = []
        for tab in win.step_tabs.values():
            tab.changed.connect(lambda k, v: emitted.append(k))

        # Short window, so the panels genuinely overflow and can scroll.
        win.resize(1400, 400)
        win.show()
        app.processEvents()

        def wheel(widget, clicks):
            return QtGui.QWheelEvent(
                QtCore.QPointF(widget.rect().center()), QtCore.QPointF(0, 0),
                QtCore.QPoint(0, 0), QtCore.QPoint(0, clicks * 120),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.ScrollPhase.NoScrollPhase, False,
            )

        tested = 0
        for name, tab in win.step_tabs.items():
            win.tabs.setCurrentWidget(tab)
            app.processEvents()
            bar = tab.scroll.verticalScrollBar()
            if bar.maximum() == 0:
                continue                  # this panel fits; nothing to scroll
            controls = (tab.panel.findChildren(QtWidgets.QSlider)
                        + tab.panel.findChildren(QtWidgets.QComboBox))
            scrolled = 0
            for widget in controls:
                bar.setValue(0)
                app.processEvents()
                app.sendEvent(widget, wheel(widget, -3))
                app.processEvents()
                if bar.value() != 0:
                    scrolled += 1
            assert scrolled == len(controls), (
                f"{name}: only {scrolled}/{len(controls)} controls passed the "
                f"wheel through to the scroll area"
            )
            tested += 1
        assert tested >= 1, "no step tab overflowed, so nothing was tested"

        changed = {k for k in before if before[k] != win.model[k]}
        assert not changed, f"scrolling silently edited {sorted(changed)}"
        assert not emitted, f"scrolling fired {len(emitted)} changes"
    finally:
        win.close()


@needs_qt
def test_the_wheel_still_adjusts_a_focused_control(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        win.tabs.setCurrentWidget(win.step_tabs["Source"])
        win.show()
        app.processEvents()
        slider = win.step_tabs["Source"].panel.findChildren(QtWidgets.QSlider)[1]
        slider.setFocus()
        app.processEvents()
        assert slider.hasFocus(), "could not focus the slider to test against"
        before = slider.value()
        app.sendEvent(slider, QtGui.QWheelEvent(
            QtCore.QPointF(slider.rect().center()), QtCore.QPointF(0, 0),
            QtCore.QPoint(0, 0), QtCore.QPoint(0, -3 * 120),
            QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase, False,
        ))
        app.processEvents()
        assert slider.value() != before
    finally:
        win.close()
