"""The planar nFET flow: two printed masks, everything else self-aligned.

The companion to `tests/test_gaa_flow.py`, and the same standard. The point is
not that a transistor-shaped pile of films can be assembled — `Stack` could
always do that — but that every pattern on the wafer arrives through
lithography: resist spun on the device stack, exposed through a drawn layout,
developed in place, with every etch gated either by that resist (the emergent
-mask rule) or by nothing at all, because the step is self-aligned.

A flow that quietly passed one `open_mask` array would still render a
convincing device and pass every geometry check. It would just no longer be a
manufacturability statement, which is the only thing these flows are for.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.wafer import Stack

SI, OX, NIT, POLY = 1, 2, 3, 4


@pytest.mark.slow
def test_two_printed_masks_build_a_planar_nfet(monkeypatch):
    """The whole flow, with an etch spy checking no mask array ever enters."""
    from litho_sim.tech import nfet as demo_nfet

    etch_masks = []
    real_etch = Stack.etch

    def spy(self, targets, *a, **kw):
        etch_masks.append(kw.get("open_mask"))
        return real_etch(self, targets, *a, **kw)

    monkeypatch.setattr(Stack, "etch", spy)
    s, metrics = demo_nfet.build_nfet(verbose=False)

    # Structural: pattern reaches the wafer through resist alone.
    assert etch_masks, "the flow ran no etches at all?"
    assert all(m is None for m in etch_masks), (
        "an etch received a hand-drawn open_mask; the two litho levels plus "
        "self-alignment are supposed to carry all the pattern information"
    )
    exposures = [h for h in s.history if h.startswith("expose ")]
    assert len(exposures) == 2, exposures

    # The printed levels resolved near the drawn CD — dose-to-size, not
    # wishful thinking. One imaging pixel (3 nm per edge) of bias is accepted.
    assert metrics["active_printed_nm"] == pytest.approx(210.0, abs=7.0)
    assert metrics["gate_printed_nm"] == pytest.approx(96.0, abs=7.0)

    # Geometry: the claims that make it an nFET.
    c = demo_nfet.checks(s, s.shape_xy[1])
    assert c["gate_on_oxide"], "the gate does not sit on oxide over silicon"
    assert c["gate_shorts"] == 0, (
        "poly touches silicon somewhere — the gate oxide must survive under "
        "the whole gate footprint"
    )
    assert c["spacer_collars"] == 2, "a spacer collar is missing"
    assert c["sd_epi_sides"] == 2, "raised source/drain did not grow both sides"
    assert c["sti_isolated"], "no field oxide below the active surface"


@pytest.mark.slow
def test_the_spacer_separates_the_gate_from_the_source_drain():
    """The spacer is load-bearing, not decorative.

    Without it the raised epi nucleates against the gate's own sidewall and
    the device is a short with a nice cross-section. The check that catches
    that is nitride standing between poly and epi at the gate's own height —
    `gate_shorts` alone would not, because the gate oxide underneath is
    intact either way.
    """
    from litho_sim.tech import nfet as demo_nfet

    s, _ = demo_nfet.build_nfet(verbose=False)
    n = s.shape_xy[1]
    mid = n // 2

    poly_x = [x for x in range(n) if (s.mat[:, mid, x] == POLY).any()]
    assert poly_x, "no gate on the channel row"

    # At a height inside the gate, walking out from either gate edge must meet
    # nitride before it meets silicon.
    z = int(max(np.flatnonzero(s.mat[:, mid, poly_x[0]] == POLY).min() + 2, 0))
    for x0, step in ((poly_x[0], -1), (poly_x[-1], +1)):
        x = x0 + step
        seen = []
        while 0 <= x < n and len(seen) < 12:
            v = int(s.mat[z, mid, x])
            if v in (NIT, SI):
                seen.append(v)
                break
            x += step
        assert seen and seen[0] == NIT, (
            f"walking {'left' if step < 0 else 'right'} from the gate at "
            f"z={z} met {seen} before nitride — the spacer is not covering "
            f"the gate sidewall"
        )


@pytest.mark.slow
def test_the_flow_scales_with_gate_length():
    """Same recipe at two gate lengths: parametric, not hand-fitted."""
    from litho_sim.tech import nfet as demo_nfet

    for gate_nm in (72.0, 132.0):
        s, m = demo_nfet.build_nfet(gate_nm=gate_nm, verbose=False)
        c = demo_nfet.checks(s, s.shape_xy[1])
        assert c["gate_on_oxide"], f"gate {gate_nm:.0f} nm: gate not on oxide"
        assert c["gate_shorts"] == 0, f"gate {gate_nm:.0f} nm: gate shorted"
        assert c["spacer_collars"] == 2, f"gate {gate_nm:.0f} nm: spacer lost"
        # Printed CD tracks the drawn CD rather than sticking at one value.
        assert m["gate_printed_nm"] == pytest.approx(gate_nm, abs=12.0)
