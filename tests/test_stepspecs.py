"""
The step editor's field specs.

The failure this file exists to prevent is the one the project keeps hitting: a
control that does not do what it says. Here that would be a spec naming a field
the step does not have, a nanometre slider that forgets its scale, or a
build/read pair that disagree — so that merely selecting a step in the list
edits it.
"""

import dataclasses
import math

import pytest

from litho_sim.app.stepspecs import (
    _SYNTHETIC,
    NO_STOP,
    PALETTE,
    STEP_SPECS,
    build_step,
    defaults_for,
    material_choices,
    unspecced_kinds,
    values_for,
)
from litho_sim.patterning import STEP_REGISTRY, ProcessStep
from litho_sim.wafer import MATERIAL_LIBRARY, get_material


@pytest.mark.parametrize("kind", sorted(STEP_SPECS))
def test_every_spec_names_a_real_field(kind):
    """A spec for a field the dataclass lacks is a control wired to nothing."""
    cls = STEP_REGISTRY[kind]
    fields = {f.name for f in dataclasses.fields(cls)}
    # Read the real list rather than restating it, so the two cannot drift.
    # It is kind-aware on purpose: `stop_on` is synthetic for an etch (folded
    # into `selectivity`) and a genuine field on a CMP.
    for spec in STEP_SPECS[kind]:
        if spec.key in _SYNTHETIC.get(kind, ()):
            continue
        assert spec.key in fields, f"{kind}.{spec.key} is not a field of {cls.__name__}"


@pytest.mark.parametrize("kind", sorted(STEP_SPECS))
def test_every_spec_kind_is_registered(kind):
    assert kind in STEP_REGISTRY


def test_the_palette_only_offers_things_with_editors():
    missing = set(PALETTE) - set(STEP_SPECS)
    assert not missing, f"palette offers steps with no editor: {missing}"


@pytest.mark.parametrize("kind", sorted(STEP_SPECS))
def test_defaults_build_a_valid_step(kind):
    step = build_step(kind, defaults_for(kind))
    assert step.kind == kind
    assert isinstance(step, STEP_REGISTRY[kind])


@pytest.mark.parametrize("kind", sorted(STEP_SPECS))
def test_build_and_read_round_trip(kind):
    """Selecting a step must not change it.

    The form is repopulated from the selected step every time the selection
    moves. If `values_for` and `build_step` disagreed, clicking through the
    recipe would quietly rewrite it.
    """
    first = build_step(kind, defaults_for(kind))
    again = build_step(kind, values_for(first))
    assert again == first


@pytest.mark.parametrize("kind", sorted(STEP_SPECS))
def test_a_built_step_survives_serialisation(kind):
    step = build_step(kind, defaults_for(kind))
    assert ProcessStep.from_dict(step.to_dict()) == step


def test_nanometre_fields_reach_the_engine_in_metres():
    """The scale is the whole reason this table exists rather than
    `dataclasses.fields`."""
    step = build_step("deposit", {**defaults_for("deposit"), "thickness": 20.0})
    assert step.thickness == pytest.approx(20e-9)

    step = build_step("etch", {**defaults_for("etch"), "depth": 60.0})
    assert step.depth == pytest.approx(60e-9)


def test_material_choices_exclude_vacuum():
    choices = material_choices()
    assert "vacuum" not in choices
    assert set(choices) == set(MATERIAL_LIBRARY) - {"vacuum"}
    for name in choices:
        assert get_material(name).name == name


@pytest.mark.parametrize(
    "kind,spec",
    [(k, s) for k, ss in sorted(STEP_SPECS.items()) for s in ss
     if s.kind in ("float", "int")],
    ids=lambda v: v.key if hasattr(v, "key") else v,
)
def test_every_slider_can_reach_its_own_default(kind, spec):
    """A slider reports ``lo + i*step``, so the default must sit on that grid.

    Otherwise touching the control moves the value without the user asking:
    ``depth`` opened at 60 nm and jumped to 61 nm on the first drag. Worse for
    ``stop_rate``, whose default was 0.01 on a grid of 0.001, 0.006, 0.011 —
    a number the control could never return to, straddling the threshold that
    decides whether a stop layer is protected at all.
    """
    steps = (spec.default - spec.lo) / spec.step
    assert steps == pytest.approx(round(steps), abs=1e-9), (
        f"{kind}.{spec.key}: default {spec.default} is {steps:.2f} steps above "
        f"lo={spec.lo}; the slider can only reach whole steps"
    )


def test_the_default_stop_rate_is_a_stop_even_on_a_shaped_etch():
    """The editor's own default has to be on the protected side of the line.

    The shaped path has no cost integral, so a single threshold comparison is
    all that arrests it. The default sat exactly on that threshold against a
    strict ``<``, which meant turning on a wall angle silently broke an etch
    stop that worked perfectly without one.
    """
    from litho_sim.core.config import GridConfig
    from litho_sim.wafer import Stack
    from litho_sim.wafer.etch_profile import EtchProfile
    from litho_sim.wafer.stack import _STOP_RATE

    rate = defaults_for("etch")["stop_rate"]
    assert rate <= _STOP_RATE, "the default is not selective enough to stop"

    grid = GridConfig(n_pixels=64, pixel_size=4e-9, dz=4e-9)
    wafer = Stack.blank(grid, dz=4e-9, headroom=400e-9)
    wafer.deposit_blanket("poly-Si", 60e-9)
    wafer.deposit_blanket("SiO2", 80e-9)
    open_col = _open_half(wafer)
    wafer.etch("SiO2", depth=300e-9, open_mask=open_col,
               selectivity={"poly-Si": rate},
               profile=EtchProfile(bias=12e-9, sidewall_deg=85.0))
    assert wafer.thickness_of("poly-Si").min() >= 55e-9


def _open_half(wafer):
    import numpy as np

    open_col = np.zeros(wafer.shape_xy, bool)
    open_col[:, wafer.shape_xy[1] // 2:] = True
    return open_col


def test_stop_on_becomes_a_selectivity_dict():
    values = {**defaults_for("etch"), "stop_on": "poly-Si", "stop_rate": 0.002}
    step = build_step("etch", values)
    assert step.selectivity == {"poly-Si": 0.002}


def test_no_stop_means_no_selectivity():
    step = build_step("etch", {**defaults_for("etch"), "stop_on": NO_STOP})
    assert step.selectivity == {}


def test_the_stop_pair_round_trips():
    values = {**defaults_for("etch"), "stop_on": "SiN", "stop_rate": 0.05}
    read_back = values_for(build_step("etch", values))
    assert read_back["stop_on"] == "SiN"
    assert read_back["stop_rate"] == pytest.approx(0.05)


def test_an_etch_that_stops_actually_stops():
    """End to end: the spec pair really does arrest an etch on the wafer.

    Not just that the dict is shaped right — that the step built from it
    behaves. This is the operation the whole tab exists to make easy.
    """
    from litho_sim.app.flow_model import FlowSession
    from litho_sim.core.config import GridConfig

    grid = GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)
    session = FlowSession(grid=grid)
    session.append(build_step("deposit", {
        "material": "poly-Si", "thickness": 60.0, "conformal": False}))
    session.append(build_step("deposit", {
        "material": "a-C", "thickness": 40.0, "conformal": False}))
    session.append(build_step("etch", {
        "targets": "a-C", "depth": 200.0, "anisotropy": 0.5,
        "stop_on": "poly-Si", "stop_rate": 0.001}))
    stack = session.run_to()

    assert stack.thickness_of("a-C").max() == 0.0, "the mandrel should clear"
    assert stack.thickness_of("poly-Si").min() >= 55e-9, (
        "the etch went through the layer it was told to stop on"
    )


def test_unknown_kind_is_rejected_clearly():
    with pytest.raises(KeyError, match="No editor for step"):
        build_step("not-a-step", {})


def test_registered_steps_without_editors_are_reported_not_hidden():
    """The palette is a subset on purpose; this makes the gap legible.

    The litho triplet used to be in this gap, which was fine while nothing in
    the app contained one — and stopped being fine when a printed device's
    recipe arrived with two exposures whose dose could not be touched. Only
    metrology is left, and no device flow uses it.
    """
    gap = unspecced_kinds()
    assert set(gap).isdisjoint(STEP_SPECS)
    assert set(gap) == {"measure"}
    # Editable, but still not offered by the palette: `pattern` remains how
    # you add lithography to a flow you are writing.
    assert {"expose", "develop", "peb"}.isdisjoint(PALETTE)


# ---------------------------------------------------------------------------
# Defaults that suit the wafer in front of you
# ---------------------------------------------------------------------------


def _wafer(*layers):
    from litho_sim.core.config import GridConfig
    from litho_sim.wafer import Stack

    s = Stack.blank(GridConfig(n_pixels=16, pixel_size=4e-9), dz=4e-9)
    for mat, nm in layers:
        s.deposit_blanket(mat, nm * 1e-9)
    return s


def test_etch_defaults_to_what_is_actually_exposed():
    """The bug this exists to prevent.

    The palette's etch defaulted to SOC, which is not on a bare wafer or on
    most stacks. Adding an etch and pressing Run therefore did nothing — and a
    no-op etch is not an error, so nothing said why.
    """
    assert defaults_for("etch")["targets"] == "SOC"          # abstract default
    assert defaults_for("etch", _wafer())["targets"] == "Si"
    assert defaults_for("etch", _wafer(("poly-Si", 40)))["targets"] == "poly-Si"
    assert defaults_for(
        "etch", _wafer(("poly-Si", 40), ("photoresist", 20))
    )["targets"] == "photoresist"


def test_strip_and_etchback_also_follow_the_surface():
    wafer = _wafer(("poly-Si", 40), ("spacer-oxide", 20))
    assert defaults_for("strip", wafer)["material"] == "spacer-oxide"
    assert defaults_for("etchback", wafer)["material"] == "spacer-oxide"


def test_an_etch_aims_through_the_opening_not_at_the_mask():
    """A strip names the film on top; an etch names what a hole leads down to.

    "Whatever covers most of the surface" is right for a strip and wrong for an
    etch: after a pattern the mask is usually the larger share of the surface,
    so that rule proposed etching the mask away instead of etching through it.
    """
    from litho_sim.core.config import GridConfig
    from litho_sim.patterning import Pattern, ProcessContext
    from litho_sim.wafer import Stack

    grid = GridConfig(n_pixels=64, pixel_size=4e-9, dz=4e-9)
    wafer = Stack.blank(grid, dz=4e-9, headroom=400e-9)
    wafer.deposit_blanket("poly-Si", 60e-9)
    # A mask covering more of the surface than it opens, so the two rules
    # genuinely disagree.
    Pattern(material="photoresist", pitch=200e-9, cd=60e-9).apply(
        wafer, ProcessContext(grid=grid))

    assert defaults_for("strip", wafer)["material"] == "photoresist"
    assert defaults_for("etch", wafer)["targets"] == "poly-Si"


def test_an_unpatterned_wafer_gives_both_rules_the_same_answer():
    wafer = _wafer(("poly-Si", 40), ("photoresist", 20))
    assert defaults_for("etch", wafer)["targets"] == "photoresist"
    assert defaults_for("strip", wafer)["material"] == "photoresist"


def test_depositing_ignores_the_surface():
    """A deposit brings its own material; the wafer has no say."""
    plain = defaults_for("deposit")["material"]
    assert defaults_for("deposit", _wafer(("SiN", 30)))["material"] == plain


def test_a_bare_stack_falls_back_to_the_static_default():
    from litho_sim.core.config import GridConfig
    from litho_sim.wafer import Stack

    empty = Stack.blank(GridConfig(n_pixels=8, pixel_size=4e-9), dz=4e-9)
    empty.strip("Si")
    assert defaults_for("etch", empty)["targets"] == "SOC"


# ---------------------------------------------------------------------------
# Lateral exposure
# ---------------------------------------------------------------------------


def test_the_etch_form_offers_the_lateral_mode():
    keys = {s.key for s in STEP_SPECS["etch"]}
    assert "exposure" in keys, "the mode exists in the engine but not the form"
    spec = next(s for s in STEP_SPECS["etch"] if s.key == "exposure")
    assert spec.choices == ("surface", "any")
    assert spec.default == "surface", "the default must not move any recipe"


def test_exposure_round_trips_through_the_form():
    """`values_for` is the inverse of `build_step`; if it disagreed, merely
    selecting the step would silently edit it."""
    for mode in ("surface", "any"):
        step = build_step("etch", {**defaults_for("etch"), "exposure": mode})
        assert step.exposure == mode
        assert values_for(step)["exposure"] == mode


def test_a_lateral_etch_says_so_in_the_recipe_line():
    step = build_step("etch", {**defaults_for("etch"),
                               "targets": "SiGe", "exposure": "any"})
    assert "lateral" in step.describe(), step.describe()
    plain = build_step("etch", {**defaults_for("etch"), "targets": "SiGe"})
    assert "lateral" not in plain.describe()


# ---------------------------------------------------------------------------
# The as-built readout
# ---------------------------------------------------------------------------
#
# `values_for` is bounded by what the sliders can express. `readout_for` is
# bounded by nothing, because the settings a form cannot show are exactly the
# ones worth showing when the question is "what built this device".


def test_the_readout_covers_every_field_of_every_registered_step():
    """No field may go missing, including on kinds with no editor.

    A recipe panel that quietly omitted `exposure` or `selectivity` would be
    worse than one that showed nothing: it would look complete.
    """
    from dataclasses import fields as dataclass_fields

    from litho_sim.app.stepspecs import readout_for
    from litho_sim.patterning import STEP_REGISTRY

    for kind, cls in STEP_REGISTRY.items():
        step = cls()
        expected = {f.name for f in dataclass_fields(step)} - {"kind"}
        assert len(readout_for(step)) == len(expected), (
            f"{kind}: readout has {len(readout_for(step))} rows for "
            f"{len(expected)} fields"
        )


def test_the_readout_reaches_kinds_the_editor_has_none_of():
    from litho_sim.app.stepspecs import readout_for, unspecced_kinds
    from litho_sim.patterning import STEP_REGISTRY

    assert "measure" in unspecced_kinds(), "premise: measure has no editor"
    rows = dict((label, value) for label, value, _ in
                readout_for(STEP_REGISTRY["measure"](name="fin CD")))
    assert rows["Name"] == "fin CD"


def test_the_readout_speaks_nanometres_and_keeps_the_whole_table():
    from litho_sim.app.stepspecs import readout_for
    from litho_sim.patterning import Etch

    step = Etch(targets=["Si", "SiGe"], depth=62e-9,
                selectivity={"Si": 1.0, "SiGe": 0.0})
    rows = dict((label, value) for label, value, _ in readout_for(step))

    assert rows["Depth"] == "62 nm", "engine metres, display nanometres"
    assert rows["Etch"] == "Si, SiGe", "both targets, not just the first"
    assert rows["Selectivity"] == "Si 1, SiGe 0", (
        "the whole selectivity table — `values_for` keeps only one row of it, "
        "which is the limit this exists to get past"
    )


def test_the_readout_marks_which_values_were_left_at_their_default():
    from litho_sim.app.stepspecs import readout_for
    from litho_sim.patterning import Deposit

    flags = {label: is_default for label, _, is_default in
             readout_for(Deposit(material="Si", thickness=6e-9,
                                 conformal=True, on="Si"))}
    assert flags["Nucleates on"] is False, "explicitly chosen"
    assert flags["Planarising"] is True, "never touched — shown, but muted"


def test_metre_fields_agree_with_the_specs_that_declare_a_nanometre_scale():
    """Two lists of what is a length; a test so they cannot drift apart.

    `_METRE_FIELDS` exists for the fields no spec covers. Where a spec *does*
    cover one, the spec's scale is the authority and the two must agree.
    """
    from litho_sim.app.stepspecs import _METRE_FIELDS, _SYNTHETIC, STEP_SPECS

    for kind, specs in STEP_SPECS.items():
        for spec in specs:
            if spec.key in _SYNTHETIC.get(kind, ()):
                continue
            if spec.scale == 1e-9:
                assert spec.key in _METRE_FIELDS, (
                    f"{kind}.{spec.key} is a nanometre slider but the readout "
                    f"would print it as a bare number"
                )


def test_the_readout_never_raises_on_a_real_device_recipe():
    """The panel shows whatever the device did; it may not choke on any of it."""
    from litho_sim.app.stepspecs import readout_for
    from litho_sim.tech.devices import flow_for

    flow = flow_for("gaa")
    if flow is None:
        pytest.skip("no cached gaa flow to read")
    for step in flow.steps:
        rows = readout_for(step)
        assert rows, f"{step.kind} produced an empty readout"
        assert all(value != "" for _, value, _ in rows)


# ---------------------------------------------------------------------------
# Editing a step that already exists
# ---------------------------------------------------------------------------
#
# `build_step` rebuilds a step from a form's whole value dict. Applied to a
# recipe that came from somewhere else, that resets every field the widgets
# cannot say — which, measured against the two device presets, damaged 10 of
# 57 steps. `apply_edit` writes one field and leaves the rest alone.


def _same(a, b) -> bool:
    """Equality that tolerates the float noise of a nm ↔ m round trip.

    A slider in nanometres times 1e-9 does not always land on the metre value
    it came from — 1.74e-07 goes out and 1.7399999999999997e-07 comes back.
    That is representation, not damage, and pinning it exactly would fail on
    every scalar for reasons that have nothing to do with what is under test.
    """
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-18)
    return a == b


def _device_flows():
    from litho_sim.tech.devices import flow_for

    flows = [(n, flow_for(n)) for n in ("gaa", "nfet")]
    if any(f is None for _, f in flows):
        pytest.skip("no cached device flows to edit")
    return flows


def test_editing_a_field_leaves_every_other_field_alone():
    """The whole point. Re-apply each field's own value; nothing may move.

    This is the assertion the read-only lock was standing in for. It covers
    all three ways the rebuild used to damage a step: a field the spec omits
    reset to its default, a container truncated to what one widget holds, and
    a value clamped to a slider that could not reach it.
    """
    from litho_sim.app.stepspecs import (
        STEP_SPECS,
        apply_edit,
        specs_for_step,
        uneditable_keys,
        values_for,
    )

    for name, flow in _device_flows():
        for i, step in enumerate(flow.steps):
            if step.kind not in STEP_SPECS:
                continue
            values, frozen = values_for(step), uneditable_keys(step)
            for spec in specs_for_step(step, flow.layouts):
                if spec.key in frozen:
                    continue
                back = apply_edit(step, spec.key, values)
                moved = [
                    k for k, v in step.to_dict().items()
                    if not _same(getattr(back, k, None), getattr(step, k, None))
                ]
                assert not moved, (
                    f"{name} step {i+1} ({step.kind}): re-applying "
                    f"'{spec.key}' changed {moved}"
                )


def test_an_edit_actually_changes_the_field_it_names():
    """Both halves: nothing else moves, *and* the named field does."""
    from litho_sim.app.stepspecs import apply_edit, values_for
    from litho_sim.patterning import Etch

    step = Etch(targets=["Si", "SiGe"], depth=62e-9,
                selectivity={"Si": 1.0, "SiGe": 1.0}, sidewall_deg=90.0)
    edited = apply_edit(step, "sidewall_deg",
                        {**values_for(step), "sidewall_deg": 78.0})
    assert edited.sidewall_deg == 78.0
    assert edited.targets == ["Si", "SiGe"], "the second target survived"
    assert edited.selectivity == {"Si": 1.0, "SiGe": 1.0}, "the table survived"


def test_a_multi_row_selectivity_is_read_only_rather_than_truncated():
    from litho_sim.app.stepspecs import uneditable_keys
    from litho_sim.patterning import Etch

    one = Etch(targets="SiO2", depth=50e-9, selectivity={"Si": 0.0})
    many = Etch(targets=["Si", "SiGe"], depth=50e-9,
                selectivity={"SiO2": 1.0, "Si": 0.0, "SiN": 0.0})

    assert uneditable_keys(one) == (), "one row fits the widgets; leave it live"
    assert set(uneditable_keys(many)) == {"targets", "stop_on", "stop_rate"}


def test_a_read_only_field_refuses_to_be_edited():
    """Not a silent no-op: the caller asked for something unrepresentable."""
    from litho_sim.app.stepspecs import apply_edit, values_for
    from litho_sim.patterning import Etch

    step = Etch(targets=["Si", "SiGe"], depth=50e-9,
                selectivity={"SiO2": 1.0, "Si": 0.0})
    with pytest.raises(ValueError, match="read-only"):
        apply_edit(step, "targets", {**values_for(step), "targets": "Si"})


def test_bounds_widen_to_admit_the_value_the_step_already_holds():
    """A slider that cannot reach what it displays is a trap, not a control."""
    from litho_sim.app.stepspecs import STEP_SPECS, apply_edit, specs_for_step, values_for
    from litho_sim.patterning import Deposit

    ceiling = next(s.hi for s in STEP_SPECS["deposit"] if s.key == "thickness")
    step = Deposit(material="SiO2", thickness=254e-9, conformal=False,
                   planarize=True)
    assert 254.0 > ceiling, "premise: the ILD is past the default ceiling"

    spec = next(s for s in specs_for_step(step) if s.key == "thickness")
    assert spec.hi >= 254.0
    assert apply_edit(step, "thickness", values_for(step)).thickness == \
        pytest.approx(254e-9), "clicking the step must not thin the film"


def test_every_step_a_device_uses_has_an_editor():
    """Dose was the one anyone would reach for, and had no widget at all."""
    from litho_sim.app.stepspecs import STEP_SPECS

    for name, flow in _device_flows():
        missing = sorted({s.kind for s in flow.steps} - set(STEP_SPECS))
        assert not missing, f"{name} uses {missing}, which cannot be edited"


def test_the_layout_choice_comes_from_the_recipe():
    """Offering 'main' on a GAA would name a layout that raises on the run."""
    from litho_sim.app.stepspecs import specs_for_step
    from litho_sim.patterning.steps import Expose

    step = Expose(layout="fin", dose=1.8)
    spec = next(s for s in specs_for_step(step, {"fin": None, "gate": None})
                if s.key == "layout")
    assert spec.choices == ("fin", "gate")
