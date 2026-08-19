"""The one place the Qt binding is chosen.

Every GUI module imports Qt from here, so the app and the matplotlib canvas
are guaranteed to agree on one binding — loading two Qt bindings into one
process crashes.
"""

from __future__ import annotations

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
    from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "The desktop app needs a Qt binding, and none is installed.\n"
        "    pip install PySide6\n"
        "or, for the whole optional group:\n"
        "    pip install -e '.[app]'\n"
        "(PyQt6, PySide2 and PyQt5 also work if you already have one.)"
    ) from exc

# Must precede any pyplot/litho_sim.viz import: matplotlib picks a backend on
# first use and will not change it afterwards.
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

__all__ = ["Figure", "FigureCanvasQTAgg", "QtCore", "QtGui", "QtWidgets"]
