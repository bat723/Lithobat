"""Device geometry the patterning primitives are supposed to be able to build.

These are acceptance tests, not unit tests: each one builds a recognisable
piece of a transistor out of nothing but registered process steps, and asserts
the thing that makes it that device rather than a pile of films.

The gate-all-around flow is here because it is the case that showed the engine
had a hole in it. Every shape a nanosheet device needs — a suspended channel,
a gate wrapped underneath it — the voxel stack could already hold. What it
could not do was *reach* the sacrificial layer: `Stack.etch` opened a column
only where the target was the material on top, and a superlattice's sacrificial
layers are capped by channel and open only on the wall of the source/drain
trench. The etch correctly found nothing and said "no exposed target", which
reads exactly like a user error and was in fact a missing capability.
"""

import numpy as np
import pytest

from litho_sim.core.config import GridConfig
from litho_sim.wafer import Stack

SHEETS = 3
SAC_NM = 10.0          # SiGe between the sheets
CHANNEL_NM = 8.0       # Si nanosheet
FIN_LO, FIN_HI = 20, 44        # the fin survives between these columns


def _superlattice_with_sd_trench() -> Stack:
    """Si/SiGe superlattice with the source/drain recessed on both sides.

    After this the sacrificial SiGe is capped by Si everywhere and touches
    open space *only* on the two trench walls.
    """
    grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    s = Stack.blank(grid, dz=2e-9, headroom=250e-9)
    for _ in range(SHEETS):
        s.deposit_blanket("SiGe", SAC_NM * 1e-9)
        s.deposit_blanket("Si", CHANNEL_NM * 1e-9)

    open_sd = np.zeros(s.shape_xy, dtype=bool)
    open_sd[:, :FIN_LO] = True
    open_sd[:, FIN_HI:] = True
    # Generous budget: Si is slow (rate 0.30), so clearing three sheets costs
    # far more than the geometric height of the stack.
    s.etch(["SiGe", "Si"], depth=200e-9, blanket=True, open_mask=open_sd)
    return s


def _sheet_rows(s: Stack, x: int, y: int) -> list:
    """Contiguous runs of channel silicon in one column, bottom-up."""
    si = s.mat[:, y, x] == 1
    runs, iz = [], 0
    while iz < s.nz:
        if si[iz]:
            start = iz
            while iz < s.nz and si[iz]:
                iz += 1
            runs.append((start, iz))
        else:
            iz += 1
    return runs


def test_the_sacrificial_layer_is_unreachable_without_a_lateral_etch():
    """The gap, stated as a test: the natural way to write a release finds
    nothing, and now says something a user can act on."""
    s = _superlattice_with_sd_trench()
    before = s.mat.copy()
    s.etch("SiGe", depth=40e-9)                 # exposure="surface", the default

    assert np.array_equal(before, s.mat), "a top-down etch cannot turn a corner"
    note = s.history[-1]
    assert "only on a sidewall" in note, note
    assert 'exposure="any"' in note, "the note has to name the mode that works"


def test_a_channel_release_suspends_every_sheet_and_costs_no_channel():
    """The release: all of the sacrificial layer gone, none of the channel.

    Selectivity here is not a special case — it is the rate contrast between
    SiGe and Si driving the same front solve.
    """
    s = _superlattice_with_sd_trench()
    si_before = int((s.mat == 1).sum())
    assert int((s.mat == 11).sum()) > 0, "no SiGe to release"

    s.etch("SiGe", depth=40e-9, exposure="any")

    assert int((s.mat == 11).sum()) == 0, "sacrificial layer not fully released"
    assert int((s.mat == 1).sum()) == si_before, (
        "the release ate into the channel; a nanosheet flow cannot afford that"
    )
    assert "lateral front" in s.history[-1]

    # Every sheet is now standing in vacuum, at the middle of the fin as well
    # as at its edge — the blanket-etch workaround only ever cleared the edge.
    for x in (FIN_LO + 1, (FIN_LO + FIN_HI) // 2):
        runs = _sheet_rows(s, x=x, y=32)
        # The substrate is silicon too, so it shows up as the bottom-most run.
        assert len(runs) == SHEETS + 1, f"x={x}: expected {SHEETS} sheets, got {runs}"
        for lo, _hi in runs[1:]:
            assert s.mat[lo - 1, 32, x] == 0, f"x={x}: sheet at {lo} is not undercut"


def test_a_short_release_recesses_the_sacrificial_layer_without_freeing_it():
    """The inner-spacer step, which the engine could not express at all.

    A release is not all-or-nothing: the same etch run short leaves the sheets
    still anchored and opens a lateral pocket at each end, which is exactly the
    cavity an inner spacer is deposited into.
    """
    s = _superlattice_with_sd_trench()
    total = int((s.mat == 11).sum())
    s.etch("SiGe", depth=8e-9, exposure="any")
    left = int((s.mat == 11).sum())

    assert 0 < left < total, f"expected a partial recess, got {left} of {total}"


@pytest.mark.parametrize("budget_nm, want_nm", [(4.0, 8.0), (8.0, 16.0), (16.0, 32.0)])
def test_the_lateral_front_advances_at_the_material_rate(budget_nm, want_nm):
    """Not just "something happened": the front travels rate x budget.

    SiGe's rate is 2.0, so a budget of N nm reaches 2N nm in from each wall.
    A model that merely dilated by a fixed radius would pass the release test
    above and fail this one.
    """
    s = _superlattice_with_sd_trench()
    mid = s.mat[:, 32, :]
    layers = [z for z in range(s.nz) if (mid[z] == 11).any()]
    z = layers[len(layers) // 2]

    before = int((s.mat[z, 32, FIN_LO:FIN_HI] == 11).sum())
    s.etch("SiGe", depth=budget_nm * 1e-9, exposure="any")
    after = int((s.mat[z, 32, FIN_LO:FIN_HI] == 11).sum())

    # Cleared from both walls, so half the columns per side.
    per_side_nm = (before - after) * 4.0 / 2
    assert abs(per_side_nm - want_nm) <= 4.0, (
        f"budget {budget_nm} nm reached {per_side_nm} nm, wanted {want_nm} nm"
    )


def test_the_gate_wraps_all_the_way_around_every_sheet():
    """What makes it gate-*all-around* rather than a gate sitting on top.

    Conformal growth into the released cavities is what has to work here: the
    dielectric must land on the underside of each sheet, not just its top.
    """
    s = _superlattice_with_sd_trench()
    s.etch("SiGe", depth=40e-9, exposure="any")
    s.deposit_conformal("SiO2", 4e-9)      # gate dielectric
    s.deposit_conformal("TiN", 8e-9)       # gate metal

    x, y = (FIN_LO + FIN_HI) // 2, 32
    runs = _sheet_rows(s, x=x, y=y)
    assert len(runs) == SHEETS + 1

    for lo, hi in runs[1:]:                 # skip the substrate
        assert s.mat[hi, y, x] == 2, "no dielectric above the sheet"
        assert s.mat[lo - 1, y, x] == 2, "no dielectric *under* the sheet"

    # And metal between the sheets, or it is a spacer rather than a gate.
    between = s.mat[runs[1][1]:runs[2][0], y, x]
    assert (between == 12).any(), f"no gate metal in the cavity: {between}"
    assert not (between == 0).any(), f"the gate cavity did not fill: {between}"


def test_source_drain_epi_grows_on_silicon_and_not_on_the_oxide_beside_it():
    """Selective epitaxy: the other front-end step the engine was missing.

    An unselective deposit puts the film down everywhere, which makes a raised
    source/drain indistinguishable from a blanket film.
    """
    grid = GridConfig(n_pixels=32, pixel_size=4e-9)
    s = Stack.blank(grid, dz=4e-9, headroom=200e-9)
    s.deposit_blanket("SiO2", 20e-9)
    window = np.zeros(s.shape_xy, dtype=bool)
    window[:, 10:22] = True
    s.etch("SiO2", depth=40e-9, open_mask=window)      # open down to silicon

    assert (s.top_material() == 1).any() and (s.top_material() == 2).any(), (
        "the fixture needs both silicon and oxide exposed"
    )
    s.deposit_blanket("SiGe", 12e-9, on="Si")

    assert int((s.mat[:, :, :10] == 11).sum()) == 0, "epi nucleated on the oxide"
    assert int((s.mat[:, :, 10:22] == 11).sum()) > 0, "no epi in the silicon window"
    assert "on Si" in s.history[-1]


# ---------------------------------------------------------------------------
# The recipe cache
# ---------------------------------------------------------------------------
#
# A device is cached as a finished wafer plus, alongside it, the recipe that
# built it and one snapshot per step. The sibling file is what lets the app's
# recipe panel show a flow it never ran.


def test_the_flow_cache_round_trips(tmp_path):
    from litho_sim.tech.devices import build, flow_for, flow_path

    stack, label = build("nfet", use_cache=False, preset_dir=tmp_path)
    assert flow_path("nfet", tmp_path).is_file(), "no sibling recipe written"

    flow = flow_for("nfet", tmp_path)
    assert flow is not None
    assert len(flow) == len(flow.snapshots) == len(flow.changed)
    assert flow.label == label, (
        "the label lives in the flow cache so a cached load reports the same "
        "printed CDs a fresh build does"
    )
    assert np.array_equal(flow.snapshots[-1].mat, stack.mat), (
        "the last snapshot is the finished device"
    )
    # The bare wafer is a step of nobody's, and the only state the scrubber's
    # first position can show.
    assert flow.base.history == ["blank: Si 90 nm"]
    assert flow.base.grid.n_pixels == stack.grid.n_pixels

    # Every step survives as a step, parameters and all.
    etches = [s for s in flow.steps if s.kind == "etch"]
    assert etches and all(e.selectivity for e in etches)


def test_a_cached_load_reports_the_printed_label(tmp_path):
    """The two paths used to disagree: a cold build said what it printed and
    a cache hit fell back to the bare title."""
    from litho_sim.tech.devices import build

    _, fresh = build("nfet", use_cache=False, preset_dir=tmp_path)
    _, cached = build("nfet", use_cache=True, preset_dir=tmp_path)
    assert cached == fresh
    assert "printed" in cached


def test_a_missing_recipe_is_not_an_error(tmp_path):
    """A preset written before recipes existed must still open and draw.

    The recipe panel is the thing that goes missing, not the device.
    """
    from litho_sim.tech.devices import build, flow_for, flow_path

    build("nfet", use_cache=False, preset_dir=tmp_path)
    flow_path("nfet", tmp_path).unlink()
    assert flow_for("nfet", tmp_path) is None


def test_an_unreadable_recipe_is_not_an_error(tmp_path):
    from litho_sim.tech.devices import build, flow_for, flow_path

    build("nfet", use_cache=False, preset_dir=tmp_path)
    flow_path("nfet", tmp_path).write_bytes(b"not an npz")
    assert flow_for("nfet", tmp_path) is None


def test_the_flow_cache_carries_the_context_the_recipe_needs(tmp_path):
    """Steps alone can be shown; only steps plus context can be re-run.

    An `Expose` names its layout and looks it up in a `ProcessContext` that
    also holds the optics. Persist the steps without it and a replayed GAA
    exposure raises on the missing layout, or prints at the library's default
    193 nm on a flow that is 13.5 nm EUV throughout.
    """
    from litho_sim.tech.devices import build, flow_for

    build("gaa", use_cache=False, preset_dir=tmp_path)
    flow = flow_for("gaa", tmp_path)
    assert flow is not None

    assert sorted(flow.layouts) == ["fin", "gate"], "the drawn levels"
    assert flow.layouts["fin"].shapes, "the geometry, not just the key"
    assert flow.optics.wavelength == pytest.approx(13.5e-9), "EUV, not the default"
    assert flow.resist.thickness == pytest.approx(45e-9)
    # `Stack.save` keeps only the lateral grid, so the resist slice count has
    # to come from the recipe or a replayed exposure resolves differently.
    assert flow.grid.n_z_slices == 5

    layout_names = {s.layout for s in flow.steps if s.kind == "expose"}
    assert layout_names <= set(flow.layouts), (
        "every exposure must name a layout the context can resolve"
    )


def test_a_recipe_without_its_context_is_a_cache_miss(tmp_path):
    """Version 1 stored steps only. Loading one would look editable and lie."""
    import json

    import numpy as np

    from litho_sim.tech.devices import build, flow_for, flow_path

    build("nfet", use_cache=False, preset_dir=tmp_path)
    path = flow_path("nfet", tmp_path)

    data = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(data["meta"]))
    meta["version"] = 1
    data["meta"] = json.dumps(meta)
    np.savez_compressed(path, **data)

    assert flow_for("nfet", tmp_path) is None, "an old cache must not load"
    # And the device rebuilds rather than opening without a usable recipe.
    _, label = build("nfet", use_cache=True, preset_dir=tmp_path)
    assert "printed" in label
    assert flow_for("nfet", tmp_path) is not None
