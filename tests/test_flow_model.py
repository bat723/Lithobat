"""
The editable process-flow session.

Two things need pinning: that editing a step re-runs only what it must, and
that resuming never lands inside a litho triplet — where the latent image lives
in the context rather than the stack, so a resume would either raise or develop
someone else's exposure.
"""

import numpy as np
import pytest

from litho_sim.app.flow_model import STATEFUL_KINDS, FlowSession
from litho_sim.core.config import GridConfig
from litho_sim.patterning import CMP, Deposit, ProcessStep, Strip
from litho_sim.wafer import Stack


@pytest.fixture
def grid():
    return GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)


@pytest.fixture
def session(grid):
    base = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9, headroom=200e-9)
    return FlowSession(grid=grid, base=base)


def _geometry(_grid=None):
    """Three steps that touch only the stack — no litho, so no context state."""
    return [
        Deposit(material="poly-Si", thickness=40e-9, conformal=False),
        Deposit(material="SiO2", thickness=20e-9, conformal=False),
        CMP(height=60e-9),
    ]


# ---------------------------------------------------------------------------
# Doing only the work that is missing
# ---------------------------------------------------------------------------


def test_running_applies_each_step_once(session, grid):
    for s in _geometry(grid):
        session.append(s)
    session.run_to()
    assert session.applied == 3
    assert session.valid_upto == 3


def test_running_again_does_nothing(session, grid):
    for s in _geometry(grid):
        session.append(s)
    session.run_to()
    session.run_to()
    assert session.applied == 3, "a second run re-applied steps that were already done"


def test_appending_applies_only_the_new_step(session, grid):
    for s in _geometry(grid)[:2]:
        session.append(s)
    session.run_to()
    assert session.applied == 2

    session.append(Strip(material="SiO2"))
    session.run_to()
    assert session.applied == 3, "appending re-ran the steps in front of it"


def test_editing_a_step_invalidates_from_there(session, grid):
    for s in _geometry(grid):
        session.append(s)
    session.run_to()
    assert session.valid_upto == 3

    session.replace(1, Deposit(material="SiO2", thickness=50e-9, conformal=False))
    assert session.valid_upto == 1, "steps before the edit should still be valid"

    session.run_to()
    assert session.applied == 3 + 2, "should have replayed exactly steps 1 and 2"


def test_removing_a_step_invalidates_from_there(session, grid):
    for s in _geometry(grid):
        session.append(s)
    session.run_to()
    session.remove(0)
    assert session.valid_upto == 0


def test_moving_a_step_invalidates_from_the_earlier_position(session, grid):
    for s in _geometry(grid):
        session.append(s)
    session.run_to()
    session.move(2, 0)
    assert session.valid_upto == 0


# ---------------------------------------------------------------------------
# The litho triplet
# ---------------------------------------------------------------------------


def _triplet():
    """expose → peb → develop, built by keyword through from_dict.

    `kind` is field 0 on every step, so positional construction silently makes
    `kind` the first argument. from_dict is keyword-only by construction.
    """
    return [
        ProcessStep.from_dict({"kind": "spincoat", "material": "photoresist"}),
        ProcessStep.from_dict({"kind": "expose", "layout": "main"}),
        ProcessStep.from_dict({"kind": "peb"}),
        ProcessStep.from_dict({"kind": "develop"}),
    ]


def test_resume_never_lands_inside_a_triplet(session):
    """The rule the whole module exists for.

    `ctx.state` carries the latent between expose, bake and develop. Resuming
    at `develop` would raise (no latent); resuming at `peb` would bake whatever
    a previous exposure left behind. So the boundary walks back to before the
    exposure.
    """
    for s in _triplet():
        session.append(s)

    # Editing the bake (index 2) must replay from the exposure (index 1),
    # because index 2 and 3 both depend on context state index 1 created.
    assert session._safe_boundary(2) == 1
    assert session._safe_boundary(3) == 1
    # And editing the develop is the same story.
    assert session._safe_boundary(4) == 1
    # The coat is not stateful, so a boundary at it is fine.
    assert session._safe_boundary(1) == 1
    assert session._safe_boundary(0) == 0


def test_every_stateful_kind_is_a_real_step_kind():
    from litho_sim.patterning import STEP_REGISTRY

    unknown = STATEFUL_KINDS - set(STEP_REGISTRY)
    assert not unknown, f"STATEFUL_KINDS names steps that do not exist: {unknown}"


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def test_snapshots_are_independent_of_each_other(session, grid):
    """Each snapshot froze its own moment.

    Steps mutate the stack in place, so without a copy per step every snapshot
    would be the same object and the scrubber would show the final state at
    every position. Note `snapshots()` hands out the stored objects, not fresh
    copies — a reader must not mutate them.
    """
    for s in _geometry(grid):
        session.append(s)
    session.run_to()
    snaps = session.snapshots()
    assert len(snaps) == 3
    assert len({id(s) for s in snaps}) == 3, "snapshots alias one another"

    # The oxide arrives at step 1; step 0 must not know about it.
    assert snaps[0].thickness_of("SiO2").max() == 0.0
    assert snaps[1].thickness_of("SiO2").max() > 0.0


def test_later_steps_do_not_reach_back_into_earlier_snapshots(session, grid):
    for s in _geometry(grid)[:1]:
        session.append(s)
    session.run_to()
    before = session.snapshots()[0].mat.copy()

    session.append(Strip(material="poly-Si"))
    session.run_to()
    assert np.array_equal(session.snapshots()[0].mat, before), (
        "stripping in a later step altered the snapshot taken before it"
    )


def test_snapshots_show_the_flow_progressing(session, grid):
    for s in _geometry(grid):
        session.append(s)
    session.run_to()
    snaps = session.snapshots()
    assert snaps[0].thickness_of("poly-Si").max() == pytest.approx(40e-9, abs=grid.dz)
    assert snaps[1].thickness_of("SiO2").max() == pytest.approx(20e-9, abs=grid.dz)
    assert snaps[2].top_height().std() == pytest.approx(0.0, abs=1e-12)


def test_stack_at_minus_one_is_the_bare_wafer(session, grid):
    """The scrubber needs a slot for 'as loaded' — snapshots() has no entry."""
    session.append(_geometry(grid)[0])
    session.run_to()
    assert session.stack_at(-1).thickness_of("poly-Si").max() == 0.0
    assert session.stack_at(0).thickness_of("poly-Si").max() > 0.0


def test_the_base_wafer_is_never_mutated(session, grid):
    session.append(_geometry(grid)[0])
    session.run_to()
    assert session.stack_at(-1).thickness_of("poly-Si").max() == 0.0, (
        "a step wrote through to the base wafer"
    )


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


def test_a_failing_step_says_which_one(session, grid):
    session.append(_geometry(grid)[0])
    session.append(Deposit(material="not-a-material", thickness=10e-9))
    with pytest.raises(RuntimeError, match="Step 2"):
        session.run_to()


def test_a_failure_leaves_earlier_work_intact(session, grid):
    session.append(_geometry(grid)[0])
    session.append(Deposit(material="not-a-material", thickness=10e-9))
    with pytest.raises(RuntimeError):
        session.run_to()
    assert session.valid_upto == 1, (
        "the step that succeeded should still be usable after a later failure"
    )


def test_on_step_fires_per_applied_step(session, grid):
    for s in _geometry(grid):
        session.append(s)
    seen = []
    session.run_to(on_step=lambda i, step, stack: seen.append(i))
    assert seen == [0, 1, 2]


def test_running_past_the_end_raises(session, grid):
    session.append(_geometry(grid)[0])
    with pytest.raises(IndexError):
        session.run_to(5)


# ---------------------------------------------------------------------------
# The boundary this module keeps
# ---------------------------------------------------------------------------


def test_flow_model_stays_qt_free():
    """Same rule params/compute/pipeline follow, for the same reason."""
    import pathlib

    src = pathlib.Path(
        "src/litho_sim/app/flow_model.py"
    ).read_text()
    for banned in ("PySide6", "QtWidgets", "QtCore"):
        assert banned not in src, f"{banned} leaked into the session model"


# ---------------------------------------------------------------------------
# The acceptance test: SADP, by hand, through the tab's own vocabulary
# ---------------------------------------------------------------------------


def test_pitch_division_falls_out_of_three_ordinary_steps():
    """Two mandrels must become four freestanding walls.

    Nothing counts anything. A conformal film has two sidewalls per mandrel,
    the etch-back clears horizontal film but not vertical, and the strip
    removes what they were standing against. If the session and the step specs
    can express that and get four, the vocabulary and the ordering are right —
    which is the whole claim the tab rests on.
    """
    from litho_sim.app.stepspecs import build_step

    # Its own grid: the module fixture is 32 px, and at 4 nm pixels that is a
    # 128 nm field — exactly one mandrel pitch, so there is no pattern to
    # double and the count would be right for the wrong reason.
    grid = GridConfig(n_pixels=64, pixel_size=4e-9, dz=4e-9)
    nx, px = grid.n_pixels, grid.pixel_size
    base = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9,
                       headroom=300e-9)
    base.deposit_blanket("poly-Si", 40e-9)
    base.deposit_blanket("a-C", 80e-9)

    x = np.arange(nx) * px
    pitch, cd, n_mandrels = 128e-9, 64e-9, 2
    start = (nx * px - n_mandrels * pitch) / 2 + pitch / 2
    keep = np.zeros((nx, nx), dtype=bool)
    for i in range(n_mandrels):
        keep |= (np.abs(x - (start + i * pitch)) < cd / 2)[None, :]
    base.etch("a-C", depth=100e-9, open_mask=~keep,
              selectivity={"poly-Si": 0.001})

    session = FlowSession(grid=grid, base=base)
    session.append(build_step("deposit", {"material": "spacer-oxide",
                                          "thickness": 16.0, "conformal": True}))
    session.append(build_step("etchback", {"material": "spacer-oxide",
                                           "overetch": 0.2}))
    session.append(build_step("strip", {"material": "a-C"}))
    final = session.run_to()

    row = final.mat.shape[1] // 2
    present = final.thickness_of("spacer-oxide")[row] > 0
    walls = int((np.diff(present.astype(int)) == 1).sum() + present[0])
    assert walls == 2 * n_mandrels, (
        f"{n_mandrels} mandrels gave {walls} walls, not {2 * n_mandrels}"
    )

    # And spacer CD is the deposited thickness — also not computed anywhere.
    widths, i = [], 0
    while i < len(present):
        if present[i]:
            j = i
            while j < len(present) and present[j]:
                j += 1
            widths.append((j - i) * px)
            i = j
        else:
            i += 1
    assert all(abs(w - 16e-9) <= px for w in widths), f"widths {widths}"


# ---------------------------------------------------------------------------
# Steps that do nothing
# ---------------------------------------------------------------------------


def test_a_step_that_changes_nothing_is_reported():
    """A no-op is legal and must not be silent.

    Etching a material that is not exposed is not an error — the engine logs
    it at debug and carries on. But from the outside it is indistinguishable
    from the application being broken, which is exactly how it was reported.
    """
    from litho_sim.app.stepspecs import build_step, defaults_for

    grid = GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)
    session = FlowSession(grid=grid)
    session.append(build_step("deposit", {**defaults_for("deposit"),
                                          "material": "spacer-oxide",
                                          "thickness": 20.0}))
    session.append(build_step("etch", {**defaults_for("etch"),
                                       "targets": "SOC"}))       # not present
    session.append(build_step("cmp", {**defaults_for("cmp"),
                                      "cmp_mode": "height",
                                      "height": 500.0}))         # above it all
    session.append(build_step("strip", {**defaults_for("strip"),
                                        "material": "a-C"}))     # not present
    session.run_to()

    assert session.did_nothing() == [1, 2, 3]


def test_a_working_recipe_reports_no_no_ops():
    from litho_sim.app.stepspecs import build_step, defaults_for

    grid = GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)
    session = FlowSession(grid=grid)
    session.append(build_step("deposit", {**defaults_for("deposit"),
                                          "material": "poly-Si",
                                          "thickness": 40.0,
                                          "conformal": False}))
    session.append(build_step("etch", {**defaults_for("etch"),
                                       "targets": "poly-Si", "depth": 20.0}))
    session.run_to()
    assert session.did_nothing() == []


def test_the_no_op_record_follows_invalidation():
    from litho_sim.app.stepspecs import build_step, defaults_for

    grid = GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)
    session = FlowSession(grid=grid)
    session.append(build_step("deposit", {**defaults_for("deposit"),
                                          "material": "poly-Si",
                                          "thickness": 40.0,
                                          "conformal": False}))
    session.append(build_step("etch", {**defaults_for("etch"),
                                       "targets": "SOC"}))
    session.run_to()
    assert session.did_nothing() == [1]

    # Fix the offending step; the record must be recomputed, not remembered.
    session.replace(1, build_step("etch", {**defaults_for("etch"),
                                           "targets": "poly-Si",
                                           "depth": 20.0}))
    session.run_to()
    assert session.did_nothing() == []


def test_a_step_that_did_nothing_says_why():
    """`did_nothing` reports *that*; the notes report *why*, in the engine's
    own words. A marker alone still leaves the user guessing.

    And *why* has to be the specific reason. One sentence used to cover four
    situations — absent, buried, masked out, and reachable only from the side —
    which is no better than a marker when the fixes are nothing alike.
    """
    from litho_sim.app.stepspecs import build_step, defaults_for

    grid = GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)
    session = FlowSession(grid=grid)
    session.append(build_step("spincoat", {**defaults_for("spincoat")}))
    session.append(build_step("etch", {**defaults_for("etch"),
                                       "targets": "Si"}))   # buried under resist
    session.run_to()

    assert session.did_nothing() == [1]
    note = " ".join(session.notes_for(1))
    assert "buried under photoresist" in note, (
        f"the idle etch explained nothing: {session.notes_for(1)}"
    )


def test_the_reported_recipe_says_the_target_is_not_on_the_wafer_yet():
    """The screenshot, verbatim: an etch placed above the step that deposits
    its target.

    The engine was right and said so, but "no exposed target" reads the same
    whether the material is absent, buried, masked out, or exposed only on a
    sidewall — and only one of those is fixed by reordering the recipe.
    """
    from litho_sim.app.stepspecs import build_step, defaults_for

    grid = GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)
    session = FlowSession(grid=grid)
    session.append(build_step("spincoat", {**defaults_for("spincoat"),
                                           "material": "SiO2",
                                           "thickness": 25.0}))
    session.append(build_step("etch", {**defaults_for("etch"),
                                       "targets": "SiARC", "depth": 60.0}))
    session.append(build_step("deposit", {**defaults_for("deposit"),
                                          "material": "SiARC",
                                          "thickness": 20.0,
                                          "conformal": True}))
    session.run_to()

    assert session.did_nothing() == [1]
    note = " ".join(session.notes_for(1))
    assert "no SiARC anywhere" in note, note
    assert "SiO2" in note, "it should name what *is* on the wafer to aim at"
    assert "step order" in note, "the fix here is to move the step, so say so"


def test_a_step_can_report_something_while_still_changing_the_wafer():
    """The case no change flag can catch.

    An etch that removed material but silently dropped its profile is not
    idle, so `did_nothing` says nothing about it — and six shape controls
    doing nothing is exactly the failure that was reported.
    """
    from litho_sim.app.stepspecs import build_step, defaults_for

    grid = GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)
    session = FlowSession(grid=grid)
    session.append(build_step("deposit", {**defaults_for("deposit"),
                                          "material": "poly-Si",
                                          "thickness": 60.0,
                                          "conformal": False}))
    session.append(build_step("etch", {**defaults_for("etch"),
                                       "targets": "poly-Si", "depth": 20.0,
                                       "sidewall_deg": 70.0}))
    session.run_to()

    assert session.did_nothing() == [], "the etch did remove material"
    assert any("no mask edge" in line for line in session.notes_for(1)), (
        f"the ignored wall angle went unreported: {session.notes_for(1)}"
    )


def test_notes_are_per_step_not_cumulative():
    """The delta arithmetic is the whole mechanism, so pin it directly."""
    from litho_sim.app.stepspecs import build_step, defaults_for

    grid = GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)
    session = FlowSession(grid=grid)
    for material in ("poly-Si", "SiO2", "SiN"):
        session.append(build_step("deposit", {**defaults_for("deposit"),
                                              "material": material,
                                              "thickness": 20.0,
                                              "conformal": False}))
    base_lines = len(session.stack_at(-1).history)
    final = session.run_to()

    per_step = [session.notes_for(i) for i in range(len(session))]
    assert all(notes for notes in per_step), "every deposit logs something"
    flattened = [line for notes in per_step for line in notes]
    assert flattened == list(final.history[base_lines:]), (
        "the per-step notes must partition everything the run added, and "
        "claim nothing the wafer arrived with"
    )


def test_notes_for_an_unrun_step_are_empty():
    session = FlowSession(grid=GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9))
    session.append(Deposit(material="poly-Si", thickness=40e-9, conformal=False))
    assert session.notes_for(0) == []
    assert session.notes_for(99) == []


# ---------------------------------------------------------------------------
# Adopting a flow that has already run
# ---------------------------------------------------------------------------
#
# A device preset arrives finished: the recipe it was built from, and one wafer
# per step of it. The session has to be able to show that without recomputing
# it, and has to refuse to pretend it could re-run it.


def _ran(grid, steps):
    """Apply *steps* to a bare wafer, returning (base, snapshots, changed)."""
    base = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9,
                       headroom=400e-9)
    session = FlowSession(grid=grid, base=base)
    for step in steps:
        session.append(step)
    session.run_to()
    return base, session.snapshots(), [
        i not in session.did_nothing() for i in range(len(steps))
    ]


def test_adopting_a_finished_flow_needs_no_run(grid):
    steps = [Deposit(material="poly-Si", thickness=40e-9, conformal=False),
             Deposit(material="SiO2", thickness=20e-9, conformal=False)]
    base, snaps, changed = _ran(grid, steps)

    session = FlowSession(grid=grid)
    session.adopt(steps, snaps, base, changed)

    assert len(session) == 2
    assert session.valid_upto == 2, "every step arrives with its snapshot"
    assert session.applied == 0, "adopting must not re-apply anything"
    assert session.locked_upto == 2
    assert session.stack_at(-1) is base, "position -1 is the wafer it began on"
    assert session.describe()[0].endswith("(blanket)")


def test_adopted_steps_refuse_to_be_edited(grid):
    """Adopted *without a context*, that is — see the editable case below."""
    steps = [Deposit(material="poly-Si", thickness=40e-9, conformal=False),
             Deposit(material="SiO2", thickness=20e-9, conformal=False)]
    base, snaps, changed = _ran(grid, steps)
    session = FlowSession(grid=grid)
    session.adopt(steps, snaps, base, changed)

    # Not a silent no-op: the caller asked for something that cannot be done
    # honestly, and doing nothing would leave the list disagreeing with itself.
    for attempt in (
        lambda: session.remove(0),
        lambda: session.replace(1, Strip(material="SiO2")),
        lambda: session.move(0, 1),
        lambda: session.insert(0, Strip(material="SiO2")),
    ):
        with pytest.raises(ValueError, match="cannot replay"):
            attempt()

    assert len(session) == 2, "and nothing changed on the way out"
    assert session.valid_upto == 2


def test_a_step_added_after_an_adopted_flow_runs_only_itself(grid):
    steps = [Deposit(material="poly-Si", thickness=40e-9, conformal=False),
             Deposit(material="SiO2", thickness=20e-9, conformal=False)]
    base, snaps, changed = _ran(grid, steps)
    session = FlowSession(grid=grid)
    session.adopt(steps, snaps, base, changed)

    session.append(Strip(material="SiO2"))
    session.run_to()

    assert session.applied == 1, (
        "the adopted steps are already done; only the new one may run"
    )
    assert session.valid_upto == 3
    present = [m.name for m in session.stack_at(2).present_materials()]
    assert "SiO2" not in present, "the appended strip actually ran"


def test_adopting_a_flow_that_ends_mid_litho_is_refused(grid):
    """The latent image lives in the context, and an adopted flow has none.

    Ending on `expose` would leave the first appended step resuming from a
    boundary `_safe_boundary` has to walk back past — into steps this session
    cannot replay.
    """
    from litho_sim.patterning.steps import Expose

    base = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9)
    steps = [Deposit(material="poly-Si", thickness=40e-9, conformal=False),
             Expose(layout="main", dose=1.0)]
    with pytest.raises(ValueError, match="mid-litho"):
        FlowSession(grid=grid).adopt(steps, [base, base], base, [True, True])

    assert set(STATEFUL_KINDS) >= {"expose"}, "the guard tracks this set"


def test_adopt_checks_it_was_given_one_snapshot_per_step(grid):
    base = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9)
    steps = [Deposit(material="poly-Si", thickness=40e-9, conformal=False)]
    session = FlowSession(grid=grid)

    with pytest.raises(ValueError, match="snapshot"):
        session.adopt(steps, [], base, [True])
    with pytest.raises(ValueError, match="change flag"):
        session.adopt(steps, [base], base, [True, False])


def test_resetting_releases_the_lock(grid):
    """Without its snapshots an adopted flow is no longer a record of anything."""
    steps = [Deposit(material="poly-Si", thickness=40e-9, conformal=False)]
    base, snaps, changed = _ran(grid, steps)
    session = FlowSession(grid=grid)
    session.adopt(steps, snaps, base, changed)
    session.reset()

    assert session.locked_upto == 0
    session.remove(0)          # no longer refused


def test_an_exposure_is_not_reported_as_having_done_nothing(grid):
    """`expose` never touches the wafer — that is the design, not a fault.

    The no-change marker exists to catch an etch that found nothing to etch. An
    exposure writes the latent image into the process context, so flagging it
    fires on every correct flow, and a warning that is always on is a warning
    nobody reads.
    """
    from litho_sim.patterning.steps import Expose

    base = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9)
    steps = [Expose(layout="main", dose=1.0),
             Deposit(material="SiO2", thickness=20e-9, conformal=False)]
    session = FlowSession(grid=grid)
    # `expose` mid-flow, so the mid-litho guard does not fire.
    session.adopt(steps, [base, base], base, changed=[False, False])

    assert session.did_nothing() == [1], (
        "the deposit that changed nothing is worth flagging; the exposure "
        "is not"
    )


def test_a_display_crop_is_not_reported_as_a_process_step(grid):
    """Cropping for the screen must not show up as something the wafer did.

    The notes are the delta between consecutive snapshot histories, which works
    only while each history extends the last. A crop appended to *every*
    snapshot breaks that: the delta stops being a suffix and slides by one, so
    the fin etch's note read "crop z[12:82] y[50:112] x[17:95] (display)".
    """
    from litho_sim.viz.viz3d import crop

    base = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9,
                       headroom=200e-9)
    session = FlowSession(grid=grid, base=base)
    session.append(Deposit(material="poly-Si", thickness=40e-9, conformal=False))
    session.append(Deposit(material="SiO2", thickness=20e-9, conformal=False))
    session.run_to()
    clean = [session.notes_for(i) for i in range(2)]

    # Same flow, but every wafer cropped for display, as a sectioned device is.
    cropped = FlowSession(grid=grid)
    cropped.adopt(
        session.steps,
        [crop(s, y=(0.45, 1.0)) for s in session.snapshots()],
        crop(base, y=(0.45, 1.0)),
        [True, True],
    )
    assert [cropped.notes_for(i) for i in range(2)] == clean
    assert all("(display)" not in line
               for notes in clean for line in notes), clean
    assert "poly-Si" in clean[0][0]


def test_a_flow_adopted_with_its_context_is_editable(grid):
    """The lock was never about age — it was about being able to replay.

    Hand over the layouts and optics an exposure needs and the flow is a flow
    like any other; hand over only the steps and it stays a record.
    """
    from litho_sim.core.config import OpticsConfig

    steps = [Deposit(material="poly-Si", thickness=40e-9, conformal=False),
             Deposit(material="SiO2", thickness=20e-9, conformal=False)]
    base, snaps, changed = _ran(grid, steps)

    session = FlowSession(grid=grid)
    session.adopt(steps, snaps, base, changed,
                  layouts={"fin": object()}, optics=OpticsConfig(wavelength=13.5e-9))

    assert session.locked_upto == 0, "with a context, nothing is locked"
    assert session.as_built_upto == 2, "but provenance is still recorded"
    assert session.optics.wavelength == 13.5e-9, "the device's optics, not the default"
    assert "fin" in session.layouts

    session.replace(0, Deposit(material="poly-Si", thickness=60e-9,
                               conformal=False))
    assert session.valid_upto == 0, "the edit invalidated from there"


def test_a_flow_adopted_without_its_context_is_still_read_only(grid):
    steps = [Deposit(material="poly-Si", thickness=40e-9, conformal=False)]
    base, snaps, changed = _ran(grid, steps)
    session = FlowSession(grid=grid)
    session.adopt(steps, snaps, base, changed)

    assert session.locked_upto == 1
    with pytest.raises(ValueError, match="cannot replay"):
        session.replace(0, Strip(material="poly-Si"))


def test_editing_an_adopted_step_replays_only_from_there(grid):
    """The cost argument for a manual Run flow: the tail, not the flow."""
    from litho_sim.core.config import OpticsConfig

    steps = [Deposit(material="poly-Si", thickness=40e-9, conformal=False),
             Deposit(material="SiO2", thickness=20e-9, conformal=False),
             CMP(height=60e-9)]
    base, snaps, changed = _ran(grid, steps)
    session = FlowSession(grid=grid)
    session.adopt(steps, snaps, base, changed, optics=OpticsConfig())

    session.replace(2, CMP(height=50e-9))       # the last step only
    assert session.valid_upto == 2
    session.run_to()
    assert session.applied == 1, "steps 1 and 2 must not have re-run"
