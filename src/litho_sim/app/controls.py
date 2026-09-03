"""Parameter controls: forms built from the specs, and the control panel."""

from __future__ import annotations

import logging
from typing import Any

from litho_sim.app.params import (
    GROUPS,
    GROUPS_3D,
    SPECS_BY_KEY,
    ParameterModel,
    ParamSpec,
    mask_model_availability,
)
from litho_sim.app.qt import QtCore, QtWidgets

logger = logging.getLogger(__name__)


class _WheelGuard(QtCore.QObject):
    """Stop a scroll gesture from being read as a parameter change.

    Qt gives sliders and combo boxes the mouse wheel by default, which is
    reasonable on a form and wrong inside a scrolling dock: the wheel lands on
    whatever control the pointer happens to be over, so scrolling past the Mask
    section drags *Drawn CD* with it. Measured before this existed — five wheel
    clicks took CD from 100 nm to 20 nm, and each click queued a full
    recompute, which is what the scrolling felt like.

    The wheel still works once a control is focused, so it remains available
    for fine adjustment; it just cannot fire at a control you were only
    scrolling past.
    """

    def eventFilter(self, obj, event):        # noqa: N802 - Qt's spelling
        if event.type() != QtCore.QEvent.Type.Wheel or obj.hasFocus():
            return False

        # Returning True stops the event dead — Qt does *not* then offer it to
        # the parent, which would leave the dock unscrollable everywhere a
        # control sits, and controls are most of it. `event.ignore()` only
        # propagates when a widget's own wheelEvent declines it, which is not
        # what an event filter is doing. So hand it on explicitly.
        area = obj.parent()
        while area is not None and not isinstance(area, QtWidgets.QScrollArea):
            area = area.parent()
        if area is not None:
            QtWidgets.QApplication.sendEvent(area.viewport(), event)
        return True


class SpecForm(QtWidgets.QWidget):
    """A form over an arbitrary list of :class:`ParamSpec`, holding its own values.

    ``ControlPanel`` does the same job for the global parameter model, but is
    welded to it — one `ParameterModel`, one set of `SPECS`, values read back
    through `model[key]`. The step editor needs the same widgets over a
    different, changing list, so the widget-building rules live here and both
    use them.
    """

    #: Carries the key that changed, so an edit can be applied as a
    #: replacement of *that field* rather than a rebuild of the whole step —
    #: which is what stops the fields the form cannot show from being lost.
    changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QFormLayout(self)
        self._layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._specs: tuple = ()
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._labels: dict[str, QtWidgets.QLabel] = {}
        self._values: dict[str, Any] = {}
        self._readonly: tuple = ()
        self._wheel_guard = _WheelGuard(self)
        self._loading = False

    def set_specs(self, specs, values: dict[str, Any], readonly=()) -> None:
        """Rebuild the form for a new set of fields.

        *readonly* names keys whose value this form's widgets cannot represent
        — a two-material etch target, a five-row selectivity table. They are
        shown as text rather than dropped, so the panel never hides a setting,
        and never offers a control that would narrow one.
        """
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                # setParent(None) *before* deleteLater, which is deferred to
                # the next event-loop turn — until then the widget still has
                # this form as its parent and keeps painting, so the outgoing
                # step's controls appear overlapped on the incoming ones.
                w.setParent(None)
                w.deleteLater()
        self._specs = tuple(specs)
        self._widgets.clear()
        self._labels.clear()
        self._values = dict(values)
        self._readonly = tuple(readonly)

        self._loading = True
        for spec in self._specs:
            self._layout.addRow(*self._build_row(spec))
        self._loading = False

    def values(self) -> dict[str, Any]:
        return dict(self._values)

    def _build_row(self, spec: ParamSpec):
        value = self._values.get(spec.key, spec.default)

        if spec.key in self._readonly:
            from litho_sim.app.stepspecs import format_field

            w = QtWidgets.QLabel(format_field(spec.key, value))
            w.setWordWrap(True)
            w.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            w.setStyleSheet("color: #808080;")
            label = QtWidgets.QLabel(spec.label)
            label.setStyleSheet("color: #808080;")
            tip = (
                f"{spec.help}\n\nShown read-only: this step's value is one the "
                f"control cannot hold without narrowing it."
            )
            w.setToolTip(tip)
            label.setToolTip(tip)
            return label, w

        label, w, is_slider = build_spec_row(
            spec, value, self._set, self._wheel_guard, slider_width=140
        )
        if is_slider:
            self._labels[spec.key] = label
        self._widgets[spec.key] = w
        return label, w

    def _set(self, key: str, value) -> None:
        self._values[key] = value
        label = self._labels.get(key)
        if label is not None:
            # Always this form's own spec. `thickness` and `material` both
            # exist in the global SPECS too, with different units and ranges.
            label.setText(spec_label(self._spec(key), value))
        if not self._loading:
            self.changed.emit(key)

    def _spec(self, key: str):
        return next(s for s in self._specs if s.key == key)


def spec_label(spec: ParamSpec, value) -> str:
    """The label a slider row wears: name, live value, unit."""
    fmt = "{:.0f}" if spec.kind == "int" else "{:.2f}"
    unit = f" {spec.unit}" if spec.unit else ""
    return f"{spec.label}  {fmt.format(value)}{unit}"


def slider_steps(spec: ParamSpec) -> int:
    return max(int(round((spec.hi - spec.lo) / spec.step)), 1)


def to_slider(spec: ParamSpec, value) -> int:
    return int(round((float(value) - spec.lo) / spec.step))


def from_slider(spec: ParamSpec, index: int):
    v = spec.lo + index * spec.step
    return int(round(v)) if spec.kind == "int" else v


def build_spec_row(spec: ParamSpec, value, on_change, wheel_guard,
                   slider_width: int = 150):
    """One ``(label, widget)`` row for a spec: checkbox, combo, or slider.

    The one set of widget-building rules both :class:`SpecForm` and
    :class:`ControlPanel` draw from. *on_change* receives ``(key, value)``
    with the value already in spec units.

    Returns ``(label, widget, is_slider)`` — a slider's label carries the
    live value and needs updating on change, which the caller owns.
    """
    if spec.kind == "bool":
        w = QtWidgets.QCheckBox()
        w.setChecked(bool(value))
        w.toggled.connect(lambda v, k=spec.key: on_change(k, bool(v)))
        label = QtWidgets.QLabel(spec.label)
    elif spec.kind == "choice":
        w = QtWidgets.QComboBox()
        w.addItems([str(c) for c in spec.choices])
        w.setCurrentText(str(value))
        w.currentTextChanged.connect(lambda v, k=spec.key: on_change(k, v))
        label = QtWidgets.QLabel(spec.label)
    else:
        w = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        w.setMinimum(0)
        w.setMaximum(slider_steps(spec))
        w.setValue(to_slider(spec, value))
        w.setMinimumWidth(slider_width)
        w.valueChanged.connect(
            lambda i, s=spec: on_change(s.key, from_slider(s, i))
        )
        label = QtWidgets.QLabel(spec_label(spec, value))

    # Focus on click only: with Qt's default the wheel itself can give a
    # slider focus, which would defeat the guard on the very next click.
    w.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    w.installEventFilter(wheel_guard)

    w.setToolTip(spec.help)
    label.setToolTip(spec.help)
    return label, w, spec.kind not in ("bool", "choice")


class StepReadout(QtWidgets.QWidget):
    """What a step ran with, shown rather than offered for editing.

    The sibling of :class:`SpecForm`, for steps that arrived already run — a
    device preset's as-built recipe. Two reasons it is not just `SpecForm` with
    the widgets disabled:

    * `SpecForm` can only show what ``STEP_SPECS`` describes, and the whole
      point here is the settings the editor has no widget for — a two-material
      etch, five rows of selectivity, an ``expose`` with no editor at all.
    * A disabled slider still rounds its value to the slider's grid, so a
      62 nm etch shown on a 1 nm grid is luck rather than accuracy.

    So this reads the step's own fields, via
    :func:`~litho_sim.app.stepspecs.readout_for`, and paints them as text.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QFormLayout(self)
        self._layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.rows: list[tuple[str, str, bool]] = []

    def show_step(self, step) -> None:
        from litho_sim.app.stepspecs import readout_for

        self._clear()
        self.rows = readout_for(step)
        for label, value, is_default in self.rows:
            name = QtWidgets.QLabel(label)
            field = QtWidgets.QLabel(value)
            # Selectable so a value can be copied out — the settings are the
            # point of the panel, and a number you cannot copy is half useful.
            field.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            field.setWordWrap(True)
            if is_default:
                # Still shown: the step *ran* with it. Muted so the values
                # someone chose stand out from the ones nobody touched.
                for w in (name, field):
                    w.setStyleSheet("color: #808080;")
            else:
                field.setStyleSheet("font-weight: bold;")
            self._layout.addRow(name, field)

    def clear(self) -> None:
        self._clear()
        self.rows = []

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                # Same ordering trap as `SpecForm.set_specs`: unparent before
                # the deferred delete, or the outgoing rows paint over these.
                w.setParent(None)
                w.deleteLater()


class ControlPanel(QtWidgets.QWidget):
    """One widget per :class:`ParamSpec`, grouped into sections.

    Built over a *subset* of the sections — each step tab owns the panel for
    the sections that belong to that step — or over all of them when *groups*
    is left out. Every panel writes into the one shared :class:`ParameterModel`,
    so the tabs are views of a single configuration rather than copies of it.
    """

    changed = QtCore.Signal(str, object)   # key, value

    def __init__(self, model: ParameterModel, groups=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.groups = tuple(GROUPS if groups is None else groups)
        self._widgets: dict[str, QtWidgets.QWidget] = {}
        self._wheel_guard = _WheelGuard(self)
        self._labels: dict[str, QtWidgets.QLabel] = {}
        self._boxes: dict[str, QtWidgets.QGroupBox] = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        for group in self.groups:
            specs = model.in_group(group)
            if not specs:
                continue
            # Say which settings only the depth-resolved run reads. They sit
            # beside the 2-D controls on the same tab — a 3-D setting is a
            # setting, not a mode — but a user dragging one and pressing
            # Print would otherwise be left wondering why nothing moved.
            title = f"{group}  (3-D profile)" if group in GROUPS_3D else group
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            for spec in specs:
                form.addRow(*self._build_row(spec))
            self._boxes[group] = box
            outer.addWidget(box)

        outer.addStretch(1)

    def keys(self) -> tuple[str, ...]:
        """The parameters this panel has a widget for."""
        return tuple(self._widgets)

    # -- construction --------------------------------------------------
    def _build_row(self, spec: ParamSpec):
        label, w, is_slider = build_spec_row(
            spec, self.model[spec.key], self._emit, self._wheel_guard
        )
        if is_slider:
            self._labels[spec.key] = label
        self._widgets[spec.key] = w
        return label, w

    # -- events --------------------------------------------------------
    def _emit(self, key: str, value) -> None:
        stored = self.model.set(key, value)
        if key in self._labels:
            self._labels[key].setText(spec_label(SPECS_BY_KEY[key], stored))
        if key in ("wavelength", "pattern", "mask_type"):
            # All three decide which thick-mask models can run, and none of
            # them shares a section with the control they gate. A no-op on a
            # panel that does not hold the mask-model combo — the window
            # forwards the change to the panel that does.
            self.refresh_mask_models()
        self.changed.emit(key, stored)

    def refresh_mask_models(self) -> None:
        """Grey out mask models the current wavelength and pattern rule out.

        ``multilayer`` needs an EUV mirror and ``fdtd`` needs a pattern with a
        cross-section, and neither condition is visible from the Mask 3-D
        section itself — the wavelength is on another tab entirely. Offering
        the choice and then raising is the wrong
        trade — a user changing one combo box should not have to know that a
        slider three sections up made it impossible. The reason goes in the
        item's tooltip, so it is still discoverable.
        """
        combo = self._widgets.get("mask_model")
        if combo is None:
            return
        reasons = mask_model_availability(
            self.model.si("wavelength"), self.model["pattern"],
            self.model["mask_type"],
        )
        for i in range(combo.count()):
            reason = reasons.get(combo.itemText(i))
            item = combo.model().item(i)
            if item is not None:
                item.setEnabled(reason is None)
            combo.setItemData(
                i, reason or "", QtCore.Qt.ItemDataRole.ToolTipRole
            )

        # If the setting in force has just become impossible, fall back to the
        # one model that always works rather than leaving an unrunnable state.
        current = self.model["mask_model"]
        if reasons.get(current) is not None:
            blocked = QtCore.QSignalBlocker(combo)
            combo.setCurrentText("thin")
            del blocked
            self.model.set("mask_model", "thin")


