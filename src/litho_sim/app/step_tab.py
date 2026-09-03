"""One processing step as a tab: its controls beside its picture."""

from __future__ import annotations

from litho_sim.app.controls import ControlPanel
from litho_sim.app.params import ParameterModel
from litho_sim.app.qt import QtCore, QtWidgets

#: Width of the control column. Wide enough for a slider with its live
#: value in the label; narrow enough that the picture keeps most of the tab.
PANEL_WIDTH = 320


class StepTab(QtWidgets.QWidget):
    """A step's settings on the left, what it produces on the right.

    The panel is a :class:`ControlPanel` over only the sections that belong to
    this step, writing into the one shared :class:`ParameterModel` — so the
    tabs are views of a single configuration, not copies of it. Above the
    picture sits a banner that says when the picture no longer matches the
    settings, because nothing recomputes on its own: exposure and development
    run from the Simulate tab, and a result drawn from settings that have
    since moved is a stale plot pretending to be a live one.

    Parameters
    ----------
    model : ParameterModel
    groups : sequence of str
        Sections from :data:`~litho_sim.app.params.TAB_GROUPS` this tab holds.
    view : QWidget
        Whatever draws the step's result.
    header : QLayout, optional
        A row above the view — the Develop tab's 2-D/3-D radio.
    """

    #: ``(key, value)`` — relayed from the panel, so the window wires one
    #: signal per tab rather than reaching into it.
    changed = QtCore.Signal(str, object)

    def __init__(self, model: ParameterModel, groups, view: QtWidgets.QWidget,
                 header: QtWidgets.QLayout | None = None, parent=None):
        super().__init__(parent)
        self.groups = tuple(groups)
        self.view = view

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        self.panel = ControlPanel(model, groups=self.groups)
        self.panel.changed.connect(self.changed)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(PANEL_WIDTH)
        scroll.setMaximumWidth(PANEL_WIDTH + 60)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll = scroll
        outer.addWidget(scroll)

        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        if header is not None:
            right.addLayout(header)
        self.banner = QtWidgets.QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        self.banner.setStyleSheet(
            "background: #fff3d6; color: #7a4a00; border: 1px solid #e0b860; "
            "border-radius: 3px; padding: 4px 8px;"
        )
        right.addWidget(self.banner)
        right.addWidget(view, 1)
        outer.addLayout(right, 1)

    def set_stale(self, text: str | None) -> None:
        """Show *text* over the picture, or clear the banner with ``None``."""
        self.banner.setText(text or "")
        self.banner.setVisible(bool(text))

    @property
    def stale(self) -> bool:
        # `isHidden`, not `isVisible`: the latter is false for every widget
        # in a window that has not been shown, which is every test.
        return not self.banner.isHidden() and bool(self.banner.text())
