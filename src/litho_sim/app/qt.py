"""The one place the Qt binding is chosen.

Every GUI module imports Qt from here, so the app and the matplotlib canvas
are guaranteed to agree on one binding — loading two Qt bindings into one
process crashes.
"""

from __future__ import annotations

import os
import warnings

import matplotlib

try:
    # matplotlib's own Qt shim rather than importing a binding directly: it
    # finds whichever of PySide6/PyQt6/PySide2/PyQt5 is present and papers
    # over the differences that matter here (PyQt spells these pyqtSignal and
    # pyqtSlot). Since the canvas comes from matplotlib anyway, using its
    # choice also guarantees the app and the canvas agree on one binding.
    #
    # This has to come *before* matplotlib.use("QtAgg"), because selecting
    # the backend imports the binding too — and would raise matplotlib's own
    # unhelpful error first, having already mutated the global backend on a
    # machine that cannot use it.
    from matplotlib.backends.qt_compat import QT_API, QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "The desktop app needs a Qt binding, and none is installed.\n"
        "    pip install PySide6\n"
        "or, for the whole optional group:\n"
        "    pip install -e '.[app]'\n"
        "(PyQt6, PySide2 and PyQt5 also work if you already have one.)"
    ) from exc

# The 3-D widget reaches Qt through QtPy, which would honour an already
# imported binding anyway; saying so explicitly makes it deterministic even
# when a test imports the widget before this module.
os.environ.setdefault("QT_API", QT_API.lower())

# Must precede any pyplot/litho_sim.viz import: matplotlib picks a backend on
# first use and will not change it afterwards.
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from litho_sim.viz import theme  # noqa: E402

# A hidden canvas can be a few pixels tall while its tab is not showing, and
# constrained layout warns that the axes collapsed. It re-solves at the real
# size the moment the tab is shown; the warning is noise here.
warnings.filterwarnings(
    "ignore", message="constrained_layout not applied", category=UserWarning,
)

# The one place the figure style is installed for the app. Every canvas is
# built after this, so the first picture and the last look the same — the
# theme used to arrive as a side effect of the first 3-D or sweep view being
# opened, and the app changed its look mid-session.
theme.apply()

__all__ = ["Figure", "FigureCanvasQTAgg", "QtCore", "QtGui", "QtWidgets"]
