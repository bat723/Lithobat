"""Open the app on the ILT tab with an isolated design loaded, and press Run.

The configuration inverse lithography is usually shown on: the *frame and
bars* pattern alone on a 2048 nm field (Grid 256, Pixel size 8 nm) under
annular illumination, with the mask sharpening annealed to a final β of 32 so
the answer cuts cleanly to two levels, and a 24 nm curvature limit so the
outline is a curve rather than a staircase. Everything is set through the dock's
own controls, so the window is the ordinary app afterwards — change anything
and press Run again.

    python scripts/ilt_live_demo.py [--pattern "isolated contacts"] [--grid 128 --pixel 16]

Numpy is held to one thread unless ``OMP_NUM_THREADS`` is already set; the
solve is FFT-bound and runs faster that way.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

from litho_sim.app import main as appmain  # noqa: E402
from litho_sim.app.params import SPECS_BY_KEY, tab_of  # noqa: E402
from litho_sim.app.qt import QtCore, QtWidgets  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--pattern", default="frame and bars",
                    choices=SPECS_BY_KEY["pattern"].choices)
    ap.add_argument("--grid", type=int, default=256, help="field side [px]")
    ap.add_argument("--pixel", type=float, default=8.0, help="pixel size [nm]")
    ap.add_argument("--beta-final", type=float, default=32.0)
    ap.add_argument("--smooth-nm", type=float, default=24.0,
                    help="curvature limit: the smallest radius the mask outline may turn on")
    ap.add_argument("--iterations", type=int, default=120)
    ap.add_argument("--conventional", action="store_true",
                    help="keep the dock's conventional illumination instead of annular")
    args = ap.parse_args()

    logging.getLogger("litho_sim").setLevel(logging.INFO)
    appmain.install_surface_format()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = appmain.MainWindow()
    window.setWindowTitle("Lithobat — live ILT demo")

    def setv(key: str, value) -> None:
        window.step_tabs[tab_of(SPECS_BY_KEY[key].group)].panel.set_value(key, value)

    setv("pattern", args.pattern)
    setv("n_pixels", args.grid)
    setv("pixel_size", args.pixel)
    if not args.conventional:
        setv("source_type", "annular")
        setv("sigma_inner", 0.5)
        setv("sigma_outer", 0.85)
    window.ilt_tab.max_iter.setValue(args.iterations)
    window.ilt_tab.beta_final.setValue(args.beta_final)
    window.ilt_tab.smooth.setValue(args.smooth_nm)
    window.ilt_tab.refresh_availability()
    window.tabs.setCurrentWidget(window.ilt_tab)

    # The widest screen: the three-panel figure wants a landscape monitor.
    screen = max(app.screens(), key=lambda s: s.geometry().width())
    g = screen.geometry()
    w, h = min(1800, g.width() - 80), min(1000, g.height() - 80)
    window.setGeometry(g.x() + (g.width() - w) // 2, g.y() + (g.height() - h) // 2, w, h)
    window.show()
    window.raise_()
    window.activateWindow()
    if window.ilt_tab.run.isEnabled():
        QtCore.QTimer.singleShot(2000, window.ilt_tab.run.click)
    else:
        print(window.ilt_tab.status.text(), file=sys.stderr)
    return (getattr(app, "exec", None) or app.exec_)()


if __name__ == "__main__":
    raise SystemExit(main())
