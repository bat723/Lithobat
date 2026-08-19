"""The GAA nanosheet flow: two printed masks, everything else self-aligned.

`tests/test_devices.py` proves the *primitives* — lateral release, conformal
wrap, selective epi — each in isolation with hand-placed masks. This file
proves the claim those primitives add up to: a gate-all-around transistor can
be built where the only pattern information on the wafer arrives through
lithography. Resist is spun on the device stack, exposed through a drawn
layout, developed in place; every etch is gated either by that resist (the
emergent-mask rule) or by nothing at all, because the step is self-aligned.

The structural assertion is the important one. A flow that quietly passed one
`open_mask` array would still print pretty pictures and pass every geometry
check — it would just no longer be a manufacturability statement.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from litho_sim.wafer import Stack  # noqa: E402


@pytest.mark.slow
def test_two_printed_masks_build_a_wrapped_transistor(monkeypatch):
    """The whole flow, with an etch spy checking no mask array ever enters.

    One build serves every assertion: geometry checks would pass with a
    smuggled mask, and the structural check would pass on a broken device, so
    only together do they say "this design can be printed and processed into
    a GAA transistor".
    """
    import demo_gaa

    etch_masks = []
    real_etch = Stack.etch

    def spy(self, targets, *a, **kw):
        etch_masks.append(kw.get("open_mask"))
        return real_etch(self, targets, *a, **kw)

    monkeypatch.setattr(Stack, "etch", spy)
    s, metrics = demo_gaa.build_gaa(verbose=False)

    # Structural: pattern reaches the wafer through resist alone.
    assert etch_masks, "the flow ran no etches at all?"
    assert all(m is None for m in etch_masks), (
        "an etch received a hand-drawn open_mask; the two litho levels plus "
        "self-alignment are supposed to carry all the pattern information"
    )
    exposures = [h for h in s.history if h.startswith("expose ")]
    assert len(exposures) == 2, exposures

    # The printed levels resolved near the drawn CD — dose-to-size, not
    # wishful thinking. One imaging pixel (2 nm per edge) of bias is accepted.
    assert metrics["fin_printed_nm"] == pytest.approx(96.0, abs=5.0)
    assert metrics["gate_printed_nm"] == pytest.approx(64.0, abs=5.0)

    # Geometry: the five claims that make it a GAA transistor.
    c = demo_gaa.checks(s, s.shape_xy[1])
    assert c["sheets"] == 3
    assert c["wrapped"] == 3, "every sheet needs dielectric above AND below"
    assert c["sacrificial_left"] == 0, "release left SiGe behind"
    assert c["gate_shorts"] == 0, (
        "gate metal touches silicon somewhere — at 2 nm lateral cells the "
        "3 nm gate oxide must survive on vertical faces"
    )
    assert c["sd_tied"], "a sheet is not connected to the source/drain epi"

    # The inner spacer is real, not decorative: nitride survives at the sheet
    # ends between channel and epi. The first tuning cleared the caps by
    # pulling back so far the pockets emptied too, and every check above still
    # passed — which is exactly why this one exists.
    mid = s.shape_xy[1] // 2
    for lo, hi in demo_gaa.sheet_runs(s, mid, mid)[1:]:
        z = (lo + hi) // 2
        row = s.mat[z, mid, :]
        assert (row == 3).sum() >= 2, (
            f"no SiN inner-spacer plug at sheet level z={z}"
        )


@pytest.mark.slow
def test_the_flow_scales_with_sheet_count():
    """Same recipe at 2 and 4 sheets: the flow is parametric, not hand-fitted.

    Fixed dummy-gate and epi heights passed at 3 sheets and silently failed at
    4 — the superlattice outgrew them and the symptom was a disconnected
    channel, not an error. Both now scale with the stack, and this is the test
    that keeps them scaling.
    """
    import demo_gaa

    for sheets in (2, 4):
        s, _ = demo_gaa.build_gaa(sheets=sheets, verbose=False)
        c = demo_gaa.checks(s, s.shape_xy[1])
        assert c["sheets"] == sheets, c
        assert c["wrapped"] == sheets, c
        assert c["sacrificial_left"] == 0 and c["gate_shorts"] == 0, c
        assert c["sd_tied"], c


@pytest.mark.slow
def test_the_recorded_recipe_is_the_recipe_that_built_the_device():
    """What the app lists must be what the wafer actually went through.

    The panel shows `ProcessStep` objects, not the engine's history prose,
    because the prose drops the selectivity table and the exposure mode. That
    only helps if the objects are the ones that ran — a recipe assembled
    *alongside* the build could drift from it silently, and would look right.

    So: replay the recorded steps onto a bare wafer through the ordinary
    engine, and require the same wafer out. Byte-identical, not approximately.
    """
    import demo_gaa
    import numpy as np

    from litho_sim.core.config import OpticsConfig, ResistConfig
    from litho_sim.mask.geometry import Rect
    from litho_sim.mask.layout import Layout
    from litho_sim.patterning.flow import Flow

    recorded = []
    base = []
    built, _ = demo_gaa.build_gaa(
        verbose=False,
        on_step=lambda step, stack: (
            base.append(stack.copy()) if step is None else recorded.append(step)
        ),
    )
    assert recorded, "the flow recorded nothing"
    assert base, "the bare wafer was never announced"

    # The same grid, optics and layouts the flow used — replaying needs the
    # context, which is exactly why an adopted flow is not re-runnable.
    n, px, W = 112, 2.0e-9, 112 * 2.0e-9
    grid = demo_gaa.GridConfig(n_pixels=n, pixel_size=px, dz=2e-9, n_z_slices=5)
    replayed = Flow(
        steps=list(recorded),
        grid=grid,
        optics=OpticsConfig(**demo_gaa.EUV),
        resist=ResistConfig(thickness=demo_gaa.RESIST_NM * 1e-9, threshold=0.32),
        layouts={
            "fin": Layout([Rect(cx=0, cy=0, w=3 * W, h=96.0e-9)], name="fin"),
            "gate": Layout([Rect(cx=0, cy=0, w=64.0e-9, h=3 * W)], name="gate"),
        },
    ).run(stack=base[0].copy(), snapshot=False).stack

    assert np.array_equal(replayed.mat, built.mat), (
        "replaying the recorded recipe did not reproduce the device — the "
        "steps on screen are not the steps that ran"
    )
    assert replayed.history == built.history


@pytest.mark.slow
def test_editing_the_recipe_changes_the_printed_device():
    """The point of the whole editable panel: a control has to move silicon.

    Dose first, because it is the knob anyone reaches for when a printed CD
    misses the drawn one — and because it exercises the part that made this
    possible, replaying an exposure with the layouts and EUV optics the device
    was built with rather than the library defaults.
    """
    from litho_sim.app.flow_model import FlowSession
    from litho_sim.tech.devices import flow_for

    flow = flow_for("gaa")
    if flow is None:
        pytest.skip("no cached gaa flow")

    session = FlowSession(grid=flow.grid)
    session.adopt(flow.steps, flow.snapshots, flow.base, flow.changed,
                  layouts=flow.layouts, optics=flow.optics, resist=flow.resist)
    assert session.locked_upto == 0, "with its context, the flow is editable"

    fin = next(i for i, s in enumerate(session.steps) if s.kind == "expose")
    develop = fin + 1
    while session.steps[develop].kind != "develop":
        develop += 1

    def printed_nm(index):
        s = session.stack_at(index)
        resist = (s.mat == 8).any(axis=0)
        return float(resist[:, resist.shape[0] // 2].sum()) * s.pixel_size * 1e9

    at_dose_to_size = printed_nm(develop)

    session.replace(fin, dataclasses.replace(session.steps[fin], dose=1.20))
    assert session.valid_upto == fin, "only the tail was invalidated"
    session.run_to()

    softer = printed_nm(develop)
    assert softer > at_dose_to_size, (
        f"a lower dose must widen a dark-tone line: {at_dose_to_size:.0f} nm "
        f"at 1.80 became {softer:.0f} nm at 1.20"
    )
    # And the exposure really did use the device's own optics: at 193 nm on
    # this 64 nm-pitch-class geometry the aerial image would not resolve.
    assert session.optics.wavelength == pytest.approx(13.5e-9)


@pytest.mark.slow
def test_an_etch_profile_edit_reaches_the_wafer():
    """Footing and rounding are adjustable, and adjusting them does something."""
    import numpy as np

    from litho_sim.app.flow_model import FlowSession
    from litho_sim.tech.devices import flow_for

    flow = flow_for("gaa")
    if flow is None:
        pytest.skip("no cached gaa flow")

    session = FlowSession(grid=flow.grid)
    session.adopt(flow.steps, flow.snapshots, flow.base, flow.changed,
                  layouts=flow.layouts, optics=flow.optics, resist=flow.resist)

    etch = next(i for i, s in enumerate(session.steps)
                if s.kind == "etch" and not isinstance(s.targets, str))
    before = session.stack_at(etch).mat.copy()

    session.replace(etch, dataclasses.replace(
        session.steps[etch], sidewall_deg=75.0, footer_height=12e-9,
        footer_extent=8e-9))
    session.run_to(etch)

    after = session.stack_at(etch).mat
    assert not np.array_equal(before, after), (
        "a tapered wall and a foot must change the etched profile"
    )
    # The fields the editor cannot show came through the edit intact.
    assert session.steps[etch].targets == ["Si", "SiGe"]
    assert session.steps[etch].selectivity == {"Si": 1.0, "SiGe": 1.0}
