"""
Unit tests for the voxel wafer stack.

The load-bearing cases here are the self-aligned patterning primitives:
conformal deposition, spacer etch-back, and mandrel pull. Those three
together must turn N mandrels into 2N freestanding walls, which is the
entire basis of SADP/SAQP.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.core.config import GridConfig
from litho_sim.wafer import MATERIAL_LIBRARY, VACUUM, Stack, get_material


@pytest.fixture
def grid() -> GridConfig:
    return GridConfig(n_pixels=64, pixel_size=4e-9)


@pytest.fixture
def stack(grid) -> Stack:
    return Stack.blank(grid, dz=4e-9, substrate_thickness=20e-9, headroom=200e-9)


def _mandrel_open_mask(st: Stack, pitch: float, cd: float, n_lines: int) -> np.ndarray:
    """Columns that are NOT protected by a mandrel line.

    Lines are centred on ``pitch/2 + i*pitch`` so none of them touches the
    field edge — ``measure_features`` drops edge-truncated runs, since their
    width is set by the simulation window rather than by the pattern.
    """
    ny, nx = st.shape_xy
    x = np.arange(nx) * st.pixel_size
    keep = np.ones((ny, nx), bool)
    for i in range(n_lines):
        keep[:, np.abs(x - (pitch / 2 + pitch * i)) <= cd / 2] = False
    return keep


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def test_material_lookup_by_name_id_and_object():
    m = get_material("Si")
    assert get_material(m.id) is m
    assert get_material(m) is m


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        get_material("unobtainium")


def test_material_ids_are_unique():
    ids = [m.id for m in MATERIAL_LIBRARY.values()]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Construction and queries
# ---------------------------------------------------------------------------


def test_blank_has_substrate_at_the_bottom(stack):
    si = get_material("Si").id
    assert (stack.mat[0] == si).all()
    assert (stack.mat[-1] == VACUUM).all()


def test_top_height_matches_deposits(stack):
    stack.deposit_blanket("SiO2", 40e-9)
    assert stack.top_height() == pytest.approx(60e-9)  # 20 substrate + 40


def test_thickness_of(stack):
    stack.deposit_blanket("SiO2", 40e-9)
    assert stack.thickness_of("SiO2") == pytest.approx(40e-9)


def test_top_material_sees_the_last_deposit(stack):
    stack.deposit_blanket("SiO2", 20e-9)
    stack.deposit_blanket("a-C", 20e-9)
    assert (stack.top_material() == get_material("a-C").id).all()


def test_surface_index_is_minus_one_for_empty_columns(grid):
    st = Stack.blank(grid, dz=4e-9, substrate_thickness=8e-9, headroom=40e-9)
    st.strip("Si")
    assert (st.surface_index() == -1).all()
    assert (st.top_height() == 0.0).all()


def test_copy_is_independent(stack):
    snap = stack.copy()
    stack.deposit_blanket("SiO2", 20e-9)
    assert snap.thickness_of("SiO2").max() == 0.0


# ---------------------------------------------------------------------------
# Deposition
# ---------------------------------------------------------------------------


def test_blanket_deposit_follows_topography(stack):
    """A non-planarizing deposit adds the same thickness to every column."""
    stack.deposit_blanket("a-C", 40e-9)
    stack.etch("a-C", depth=40e-9, open_mask=_mandrel_open_mask(stack, 128e-9, 64e-9, 2))
    before = stack.top_height()
    stack.deposit_blanket("SiO2", 20e-9)
    after = stack.top_height()
    assert np.allclose(after - before, 20e-9)


def test_planarizing_deposit_flattens(stack):
    stack.deposit_blanket("a-C", 40e-9)
    stack.etch("a-C", depth=40e-9, open_mask=_mandrel_open_mask(stack, 128e-9, 64e-9, 2))
    stack.deposit_blanket("SOC", 40e-9, planarize=True)
    assert stack.top_height().std() == pytest.approx(0.0, abs=1e-12)


def test_conformal_deposit_grows_sideways(stack):
    """The defining property: a conformal film thickens a vertical sidewall.

    A blanket deposit only adds height, so the patterned line gets no wider.
    A conformal deposit coats the sidewalls too, so it does.
    """
    stack.deposit_blanket("a-C", 60e-9)
    stack.etch("a-C", depth=60e-9, open_mask=_mandrel_open_mask(stack, 128e-9, 64e-9, 2))
    width_before = sum(stack.measure_features("a-C")["lines"])

    stack.deposit_conformal("spacer-oxide", 12e-9)
    prof = stack.line_profile("spacer-oxide", z_frac=0.5)
    assert prof.any(), "conformal film did not appear at mid-height"

    combined = stack.solid()[stack.nz // 4].sum()
    assert combined > 0
    assert width_before > 0


# ---------------------------------------------------------------------------
# Etch
# ---------------------------------------------------------------------------


def test_etch_only_opens_exposed_target(stack):
    """A column capped by a different material must not etch.

    The etch mask is emergent — whatever sits on top masks the column. This
    is what lets one primitive serve LELE, SADP, and cut masks alike.
    """
    stack.deposit_blanket("poly-Si", 40e-9)
    before = stack.thickness_of("poly-Si").copy()
    stack.etch("a-C", depth=40e-9)  # a-C is not present at all
    assert np.array_equal(stack.thickness_of("poly-Si"), before)


def test_etch_removes_requested_depth(stack):
    stack.deposit_blanket("SiO2", 60e-9)
    stack.etch("SiO2", depth=20e-9 * get_material("SiO2").etch_rate)
    assert stack.thickness_of("SiO2").max() == pytest.approx(40e-9, abs=8e-9)


def test_etch_stops_on_a_selective_underlayer(stack):
    """Near-zero selectivity must arrest the etch at the underlayer."""
    stack.deposit_blanket("poly-Si", 60e-9)
    stack.deposit_blanket("a-C", 40e-9)
    stack.etch("a-C", depth=200e-9, selectivity={"poly-Si": 0.001})
    assert stack.thickness_of("a-C").max() == 0.0, "mandrel should be gone"
    assert stack.thickness_of("poly-Si").min() >= 55e-9, "etch broke through the stop layer"


def test_etch_does_not_dig_below_the_surface_without_budget(stack):
    """A zero-depth etch is a no-op, not a wafer-clearing event."""
    stack.deposit_blanket("SiO2", 40e-9)
    before = stack.mat.copy()
    stack.etch("SiO2", depth=0.0)
    assert np.array_equal(stack.mat, before)


def test_anisotropy_produces_undercut(stack):
    stack.deposit_blanket("SiO2", 60e-9)
    mask = _mandrel_open_mask(stack, 128e-9, 64e-9, 2)
    aniso = stack.copy()
    aniso.etch("SiO2", depth=60e-9, anisotropy=1.0, open_mask=mask)
    iso = stack.copy()
    iso.etch("SiO2", depth=60e-9, anisotropy=0.5, open_mask=mask)
    assert iso.volume_fraction("SiO2") < aniso.volume_fraction("SiO2"), (
        "reducing anisotropy should remove extra material laterally"
    )


def test_an_isotropic_etch_still_respects_the_etch_stop(stack):
    """Selectivity must arrest the lateral spread, not only the vertical one.

    The undercut pass dilates the removed set by a distance transform, and
    used to remove anything non-vacuum within reach — regardless of material.
    So an etch told to stop on an underlayer stopped on it going down and then
    ate it sideways. Measured before the fix: at anisotropy 0.7 the whole 60 nm
    stop layer was removed, not merely thinned.
    """
    stack.deposit_blanket("poly-Si", 60e-9)
    stack.deposit_blanket("a-C", 40e-9)
    stack.etch("a-C", depth=200e-9, anisotropy=0.5, selectivity={"poly-Si": 0.001})

    assert stack.thickness_of("a-C").max() == 0.0, "mandrel should still clear"
    assert stack.thickness_of("poly-Si").min() >= 55e-9, (
        "the isotropic pass undercut through the etch stop"
    )


@pytest.mark.parametrize("anisotropy", [0.2, 0.5, 0.7, 1.0])
def test_the_etch_stop_holds_at_every_anisotropy(stack, anisotropy):
    """Sweeping rather than spot-checking, because the old failure was
    invisible at the default (1.0) and total everywhere below it."""
    stack.deposit_blanket("poly-Si", 60e-9)
    stack.deposit_blanket("a-C", 40e-9)
    stack.etch("a-C", depth=200e-9, anisotropy=anisotropy,
               selectivity={"poly-Si": 0.001})
    assert stack.thickness_of("poly-Si").min() >= 55e-9


def test_planarize_cuts_to_the_lowest_peak_by_default(stack):
    """CMP with no height argument stops at the lowest point on the wafer.

    Untested until now, and about to become the first thing a stack-builder
    reaches for after a fill.
    """
    stack.deposit_blanket("SiO2", 60e-9)
    mask = _mandrel_open_mask(stack, 128e-9, 64e-9, 2)
    stack.etch("SiO2", depth=30e-9, open_mask=mask)
    assert stack.top_height().std() > 0, "fixture is flat; nothing to planarise"

    lowest = stack.top_height().min()
    stack.planarize()

    assert stack.top_height().std() == pytest.approx(0.0, abs=1e-12)
    assert stack.top_height().max() == pytest.approx(lowest, abs=stack.dz)


def test_planarize_to_an_explicit_height(stack):
    stack.deposit_blanket("SiO2", 80e-9)
    stack.planarize(height=50e-9)
    assert stack.top_height().max() == pytest.approx(50e-9, abs=stack.dz)
    assert stack.top_height().std() == pytest.approx(0.0, abs=1e-12)


def test_planarize_is_indifferent_to_material(stack):
    """A guillotine, not a selective process — it takes whatever is up there.

    Worth pinning as a decision rather than leaving implicit: real CMP dishes
    and erodes by material, and this model deliberately does neither.
    """
    # 20 nm substrate + 40 nm poly-Si puts the oxide's base at 60 nm, so
    # cutting to 80 nm should take half the oxide and none of the poly.
    stack.deposit_blanket("poly-Si", 40e-9)
    stack.deposit_blanket("SiO2", 40e-9)
    stack.planarize(height=80e-9)
    assert stack.thickness_of("SiO2").max() == pytest.approx(20e-9, abs=stack.dz)
    assert stack.thickness_of("poly-Si").max() == pytest.approx(40e-9, abs=stack.dz)


def test_planarize_above_the_stack_is_a_no_op(stack):
    stack.deposit_blanket("SiO2", 40e-9)
    before = stack.mat.copy()
    stack.planarize(height=500e-9)
    assert np.array_equal(stack.mat, before)


def test_strip_removes_everything_of_one_material(stack):
    stack.deposit_blanket("SiO2", 20e-9)
    stack.strip("SiO2")
    assert stack.volume_fraction("SiO2") == 0.0
    assert stack.volume_fraction("Si") > 0.0


# ---------------------------------------------------------------------------
# The SADP sequence — the reason this module is voxel-based
# ---------------------------------------------------------------------------


def test_sadp_doubles_the_line_count(grid):
    """N mandrels must become 2N freestanding spacer walls.

    A height-field representation cannot even express the end state here:
    after the mandrel is pulled, a column through the middle of where it used
    to be reads vacuum → spacer → vacuum → spacer.
    """
    st = Stack.blank(grid, dz=4e-9, substrate_thickness=20e-9, headroom=240e-9)
    st.deposit_blanket("poly-Si", 40e-9)
    st.deposit_blanket("a-C", 80e-9)

    n_mandrels = 2
    pitch, cd, spacer_t = 128e-9, 64e-9, 16e-9
    st.etch(
        "a-C", depth=80e-9, anisotropy=1.0,
        open_mask=_mandrel_open_mask(st, pitch, cd, n_mandrels),
        selectivity={"poly-Si": 0.01},
    )
    assert len(st.measure_features("a-C")["lines"]) == n_mandrels

    st.deposit_conformal("spacer-oxide", spacer_t)
    st.etch_back("spacer-oxide", thickness=spacer_t, overetch=0.25)
    st.strip("a-C")

    walls = st.measure_features("spacer-oxide")["lines"]
    assert len(walls) == 2 * n_mandrels, (
        f"expected {2*n_mandrels} spacer walls from {n_mandrels} mandrels, got {len(walls)}"
    )
    for w in walls:
        assert w == pytest.approx(spacer_t, abs=2 * grid.pixel_size), (
            f"spacer wall {w*1e9:.0f} nm should match the {spacer_t*1e9:.0f} nm deposit"
        )


def test_sadp_halves_the_pitch(grid):
    st = Stack.blank(grid, dz=4e-9, substrate_thickness=20e-9, headroom=240e-9)
    st.deposit_blanket("a-C", 80e-9)
    pitch = 128e-9
    st.etch("a-C", depth=80e-9, open_mask=_mandrel_open_mask(st, pitch, 64e-9, 2))
    st.deposit_conformal("spacer-oxide", 16e-9)
    st.etch_back("spacer-oxide", thickness=16e-9)
    st.strip("a-C")

    prof = st.line_profile("spacer-oxide", z_frac=0.5)
    centres = []
    run = None
    for i, v in enumerate(prof):
        if v and run is None:
            run = i
        elif not v and run is not None:
            centres.append((run + i - 1) / 2 * st.pixel_size)
            run = None
    assert len(centres) >= 3
    spacings = np.diff(centres)
    assert spacings.min() == pytest.approx(pitch / 2, abs=3 * st.pixel_size), (
        f"spacer spacing {spacings.min()*1e9:.0f} nm should be half the "
        f"{pitch*1e9:.0f} nm mandrel pitch"
    )


def test_etch_back_defaults_to_flat_field_thickness(grid):
    """etch_back must not size itself off the sidewall column.

    Over a mandrel sidewall the spacer column is as tall as the mandrel; using
    that as the etch depth clears the entire wafer.
    """
    st = Stack.blank(grid, dz=4e-9, substrate_thickness=20e-9, headroom=200e-9)
    st.deposit_blanket("a-C", 60e-9)
    st.etch("a-C", depth=60e-9, open_mask=_mandrel_open_mask(st, 128e-9, 64e-9, 2))
    st.deposit_conformal("spacer-oxide", 16e-9)
    st.etch_back("spacer-oxide")  # no explicit thickness
    st.strip("a-C")
    assert st.volume_fraction("spacer-oxide") > 0.0, "etch-back cleared the whole film"
    assert st.volume_fraction("Si") > 0.0, "etch-back destroyed the substrate"


# ---------------------------------------------------------------------------
# Metrology and I/O
# ---------------------------------------------------------------------------


def test_measure_features_drops_edge_truncated_runs(stack):
    stack.deposit_blanket("SiO2", 20e-9)  # covers the whole field
    feats = stack.measure_features("SiO2")
    assert feats["lines"] == [] and feats["spaces"] == []


def test_height_fields_bracket_the_material(stack):
    stack.deposit_blanket("SiO2", 40e-9)
    z_bot, z_top, present = stack.height_fields("SiO2")
    assert present.all()
    assert np.allclose(z_top - z_bot, 40e-9)


# ---------------------------------------------------------------------------
# Auxiliary voxel fields — the dopant volume's future home
# ---------------------------------------------------------------------------


def test_a_new_field_matches_the_material_volume(stack):
    dopant = stack.add_field("dopant")
    assert dopant.shape == stack.mat.shape
    assert dopant.dtype == np.float32
    assert stack.fields["dopant"] is dopant


def test_add_field_is_idempotent(stack):
    first = stack.add_field("dopant")
    first[0, 0, 0] = 7.0
    again = stack.add_field("dopant")
    assert again is first, "re-adding a field must not wipe it"
    assert again[0, 0, 0] == 7.0


def test_fields_grow_with_the_stack(stack):
    """The one that would break silently.

    `ensure_headroom` grows `mat` by concatenation. A field that did not grow
    with it would keep its old `nz`, stay the same object, and quietly index
    against the wrong depth from then on — no exception, just wrong answers.
    """
    stack.add_field("dopant")[:] = 3.0
    before = stack.nz
    stack.ensure_headroom(400e-9)

    assert stack.nz > before, "fixture had enough headroom; nothing was tested"
    assert stack.fields["dopant"].shape == stack.mat.shape
    # Existing values survive; the new space starts empty.
    assert stack.fields["dopant"][:before].min() == 3.0
    assert stack.fields["dopant"][before:].max() == 0.0


def test_fields_are_deep_copied(stack):
    stack.add_field("dopant")[:] = 1.0
    snap = stack.copy()
    stack.fields["dopant"][:] = 9.0
    assert snap.fields["dopant"].max() == 1.0


def test_fields_survive_a_save_load_round_trip(stack, tmp_path):
    stack.add_field("dopant")[3, 2, 1] = 2.5
    path = tmp_path / "with_fields.npz"
    stack.save(path)
    back = Stack.load(path)
    assert set(back.fields) == {"dopant"}
    assert back.fields["dopant"].shape == back.mat.shape
    assert back.fields["dopant"][3, 2, 1] == pytest.approx(2.5)


def test_a_stack_saved_without_fields_still_loads(stack, tmp_path):
    """Forward compatibility: files written before fields existed."""
    path = tmp_path / "no_fields.npz"
    stack.save(path)
    assert Stack.load(path).fields == {}


def test_save_load_round_trip(stack, tmp_path):
    stack.deposit_blanket("SiO2", 20e-9)
    stack.deposit_conformal("spacer-oxide", 8e-9)
    path = tmp_path / "stack.npz"
    stack.save(path)
    back = Stack.load(path)
    assert np.array_equal(back.mat, stack.mat)
    assert back.dz == stack.dz
    assert back.grid.pixel_size == stack.grid.pixel_size
    assert back.history == stack.history


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Stack.load(tmp_path / "nope.npz")


# ---------------------------------------------------------------------------
# CMP modes
# ---------------------------------------------------------------------------


def _two_films(stack):
    """20 nm substrate, 40 nm SiN stop, 60 nm oxide on top."""
    stack.deposit_blanket("SiN", 40e-9)
    stack.deposit_blanket("SiO2", 60e-9)
    return stack


def test_cmp_depth_measures_down_from_the_high_point(stack):
    _two_films(stack)
    before = stack.top_height().max()
    stack.planarize(depth=30e-9)
    assert stack.top_height().max() == pytest.approx(before - 30e-9, abs=stack.dz)
    assert stack.top_height().std() == pytest.approx(0.0, abs=1e-12)


def test_cmp_depth_on_topography_starts_at_the_peak(stack):
    """The point of a depth-mode CMP: you know how much to take off, not the
    absolute level the incoming topography happens to sit at."""
    stack.deposit_blanket("SiO2", 60e-9)
    mask = _mandrel_open_mask(stack, 128e-9, 64e-9, 2)
    stack.etch("SiO2", depth=30e-9, open_mask=mask)
    peak = stack.top_height().max()
    stack.planarize(depth=10e-9)
    assert stack.top_height().max() == pytest.approx(peak - 10e-9, abs=stack.dz)


def test_cmp_stops_on_a_layer(stack):
    _two_films(stack)
    stack.planarize(stop_on="SiN")
    assert stack.thickness_of("SiO2").max() == 0.0, "oxide should be gone"
    assert stack.thickness_of("SiN").max() == pytest.approx(40e-9, abs=stack.dz)


def test_cmp_stop_lands_at_the_layers_highest_point(stack):
    """A flat pad meets the stop layer where it is tallest, so a low stop layer
    leaves material behind rather than being polished into."""
    stack.deposit_blanket("SiN", 20e-9)
    mask = _mandrel_open_mask(stack, 128e-9, 64e-9, 2)
    stack.fill(
        (np.arange(stack.nz)[:, None, None] >= stack._n_voxels(40e-9))
        & (np.arange(stack.nz)[:, None, None] < stack._n_voxels(60e-9))
        & ~mask[None],
        "SiN",
    )
    stack.deposit_blanket("SiO2", 80e-9)
    stack.planarize(stop_on="SiN")
    assert stack.top_height().max() == pytest.approx(60e-9, abs=stack.dz)
    assert stack.thickness_of("SiO2").max() > 0.0, (
        "polishing to the highest stop point must leave oxide in the low regions"
    )


def test_cmp_takes_at_most_one_mode(stack):
    _two_films(stack)
    with pytest.raises(ValueError, match="at most one"):
        stack.planarize(depth=10e-9, stop_on="SiN")


def test_cmp_cannot_stop_on_an_absent_material(stack):
    _two_films(stack)
    with pytest.raises(ValueError, match="contains none of it"):
        stack.planarize(stop_on="poly-Si")


# ---------------------------------------------------------------------------
# Etch: blanket, access, profile
# ---------------------------------------------------------------------------


def test_blanket_recesses_whatever_is_on_top(stack):
    """The one thing the emergent mask cannot express.

    A normal etch opens only columns whose top material is a target; a blanket
    recess takes the surface down everywhere, whatever is exposed.
    """
    stack.deposit_blanket("SiO2", 40e-9)
    mask = _mandrel_open_mask(stack, 128e-9, 64e-9, 2)
    stack.fill(
        (np.arange(stack.nz)[:, None, None] >= stack._n_voxels(60e-9))
        & (np.arange(stack.nz)[:, None, None] < stack._n_voxels(80e-9))
        & ~mask[None],
        "a-C",
    )
    before = stack.top_height().copy()

    # Naming SiO2 alone would skip every column capped with a-C.
    stack.etch("SiO2", depth=12e-9, blanket=True)
    dropped = before - stack.top_height()
    assert dropped.min() > 0, "some columns were not recessed at all"
    assert stack.thickness_of("a-C").max() < 20e-9, "the cap should be thinned too"


def test_line_of_sight_suppresses_a_re_entrant_undercut(stack):
    """Ions travel in straight lines, so they cannot cut back under an overhang.

    A re-entrant wall is wider at depth than at the top, which is exactly an
    overhang — turning line of sight on must leave more material behind. Note a
    uniform *bias* is not this: it widens the opening at every depth including
    the top, so nothing is shadowed and LOS correctly changes nothing.
    """
    from litho_sim.wafer.etch_profile import EtchProfile

    stack.deposit_blanket("poly-Si", 80e-9)
    mask = _mandrel_open_mask(stack, 160e-9, 80e-9, 1)
    profile = EtchProfile(sidewall_deg=100.0)

    plain = stack.copy()
    plain.etch("poly-Si", depth=80e-9, open_mask=mask, profile=profile)
    shadowed = stack.copy()
    shadowed.etch("poly-Si", depth=80e-9, open_mask=mask, profile=profile,
                  line_of_sight=True)
    assert shadowed.volume_fraction("poly-Si") > plain.volume_fraction("poly-Si"), (
        "line of sight did not suppress the overhang"
    )


def test_line_of_sight_does_not_shadow_an_unmasked_etch(stack):
    """The regression that made it useless.

    Shadowing was measured from the top of the *array*, which is empty
    headroom — never in the removed set, so it counted as an obstruction and
    every column was shadowed. Vacuum has to be transparent.
    """
    stack.deposit_blanket("SiN", 20e-9)
    stack.deposit_blanket("poly-Si", 60e-9)
    stack.etch("poly-Si", depth=100e-9, stop_on="SiN", line_of_sight=True)
    assert stack.thickness_of("poly-Si").max() == 0.0, (
        "an unmasked etch shadowed itself against empty space above the wafer"
    )


def test_a_profile_needs_a_mask_edge(stack):
    """With no mask there is no sidewall, and the transform has no boundary.

    `distance_transform_edt` of an all-True array measures the distance to the
    array edge, so an unmasked etch with a profile would carve a bowl out of
    the field boundary. Skipped rather than believed.
    """
    from litho_sim.wafer.etch_profile import EtchProfile

    stack.deposit_blanket("SiN", 20e-9)
    stack.deposit_blanket("poly-Si", 60e-9)
    shaped = stack.copy()
    shaped.etch("poly-Si", depth=100e-9, stop_on="SiN",
                profile=EtchProfile(sidewall_deg=82.0))
    plain = stack.copy()
    plain.etch("poly-Si", depth=100e-9, stop_on="SiN")
    assert np.array_equal(shaped.mat, plain.mat)


def test_contact_leaves_a_sealed_void_alone(stack):
    """Chemical access: the etchant has to be able to get there."""
    stack.deposit_blanket("SiO2", 40e-9)
    buried = (
        (np.arange(stack.nz)[:, None, None] >= stack._n_voxels(28e-9))
        & (np.arange(stack.nz)[:, None, None] < stack._n_voxels(36e-9))
    )
    inner = np.zeros(stack.shape_xy, dtype=bool)
    ny, nx = stack.shape_xy
    inner[ny // 2 - 4:ny // 2 + 4, nx // 2 - 4:nx // 2 + 4] = True
    stack.fill(buried & inner[None], "a-C")
    sealed = stack.copy()

    # a-C is fully enclosed by oxide, so nothing can reach it.
    sealed.etch("a-C", depth=40e-9, contact=True)
    assert sealed.thickness_of("a-C").max() == pytest.approx(
        stack.thickness_of("a-C").max()
    ), "a sealed void was etched by something that could not reach it"


def test_a_tapered_profile_narrows_with_depth(stack):
    from litho_sim.wafer.etch_profile import EtchProfile

    stack.deposit_blanket("poly-Si", 100e-9)
    mask = _mandrel_open_mask(stack, 200e-9, 96e-9, 1)
    stack.etch("poly-Si", depth=100e-9, open_mask=mask,
               profile=EtchProfile(sidewall_deg=75.0))

    row = stack.shape_xy[0] // 2
    poly = stack.mat == get_material("poly-Si").id
    top_i = stack._n_voxels(95e-9) - 1
    bot_i = stack._n_voxels(30e-9)
    open_top = int((~poly[top_i, row]).sum())
    open_bot = int((~poly[bot_i, row]).sum())
    assert open_bot < open_top, (
        f"a 75 degree wall should narrow with depth: {open_top} -> {open_bot} px"
    )


def test_a_profile_still_respects_the_etch_stop(stack):
    """A shaped front has no cost integral to arrest it, so the stop layer is
    protected explicitly — the same failure the isotropic pass once had."""
    from litho_sim.wafer.etch_profile import EtchProfile

    stack.deposit_blanket("poly-Si", 40e-9)
    stack.deposit_blanket("SiO2", 80e-9)
    mask = _mandrel_open_mask(stack, 200e-9, 96e-9, 1)
    stack.etch("SiO2", depth=200e-9, open_mask=mask,
               selectivity={"poly-Si": 0.001},
               profile=EtchProfile(bias=12e-9, sidewall_deg=85.0))
    assert stack.thickness_of("poly-Si").min() >= 36e-9, (
        "the shaped front ate into the stop layer"
    )


def test_a_shaped_etch_does_not_carve_its_own_mask(stack):
    """A re-entrant wall must flare the hole, not the thing defining it.

    ``depth_below`` is measured from each column's own top — right for a
    recessed open column, wrong for a masked one, where the reference becomes
    the top of the mask. An outward offset then reached sideways into the mask
    at mask-relative depths and carved it into the shape of the opening. Only
    visible once the wafer could carry a mask at all.
    """
    from litho_sim.wafer.etch_profile import EtchProfile

    stack.deposit_blanket("poly-Si", 120e-9)
    zz = np.arange(stack.nz)[:, None, None]
    surf = stack.surface_index()
    cap = (zz > surf[None]) & (zz <= surf[None] + 20)      # 80 nm of resist
    open_col = _mandrel_open_mask(stack, 200e-9, 96e-9, 1)
    cap[:, open_col] = False
    stack.fill(cap, "photoresist")
    mask_before = stack.thickness_of("photoresist").copy()

    stack.etch("poly-Si", depth=120e-9,
               profile=EtchProfile(sidewall_deg=110.0))

    assert np.array_equal(stack.thickness_of("photoresist"), mask_before), (
        "the etch ate the mask that was masking it"
    )
    assert stack.thickness_of("poly-Si").min() < 120e-9, (
        "and it must still have etched the film — protecting the mask is not "
        "the same as doing nothing"
    )


def test_an_identity_profile_changes_nothing(stack):
    """The default has to be a no-op, or every existing recipe moves."""
    from litho_sim.wafer.etch_profile import EtchProfile

    stack.deposit_blanket("SiO2", 60e-9)
    mask = _mandrel_open_mask(stack, 128e-9, 64e-9, 2)
    plain = stack.copy()
    plain.etch("SiO2", depth=40e-9, open_mask=mask)
    shaped = stack.copy()
    shaped.etch("SiO2", depth=40e-9, open_mask=mask, profile=EtchProfile())
    assert np.array_equal(plain.mat, shaped.mat)


@pytest.mark.parametrize("stop_rate", [0.0, 0.0099, 0.01])
def test_a_stop_rate_at_the_threshold_still_stops(stack, stop_rate):
    """A layer told to etch a hundred times slower is a stop layer.

    The shaped path throws away the cost integral, so a single ``_STOP_RATE``
    comparison is the only thing arresting it — and that comparison used to be
    strict, so the rate the editor offered as its own default landed on the
    wrong side of it and the stop layer was quietly eaten. Whether the rate is
    spelled 0.0099 or 0.01 is not a physical distinction.
    """
    from litho_sim.wafer.etch_profile import EtchProfile

    stack.deposit_blanket("poly-Si", 60e-9)
    stack.deposit_blanket("SiO2", 80e-9)
    mask = _mandrel_open_mask(stack, 200e-9, 96e-9, 1)
    stack.etch("SiO2", depth=200e-9, open_mask=mask,
               selectivity={"poly-Si": stop_rate},
               profile=EtchProfile(bias=12e-9, sidewall_deg=85.0))
    assert stack.thickness_of("poly-Si").min() >= 55e-9, (
        f"the shaped front broke through at stop_rate={stop_rate}"
    )


def test_a_blanket_field_says_the_profile_was_ignored(stack):
    """No mask edge, no sidewall — and the wafer must say so.

    Skipping the profile is the *correct* physics: a distance transform of an
    all-open field measures to the array edge, and shaping against that would
    carve a bowl out of the field boundary. What was wrong was doing it in
    silence, which is how six working controls came to look broken.
    """
    stack.deposit_blanket("SiO2", 60e-9)
    from litho_sim.wafer.etch_profile import EtchProfile

    plain = stack.copy()
    plain.etch("SiO2", depth=40e-9)
    shaped = stack.copy()
    shaped.etch("SiO2", depth=40e-9, profile=EtchProfile(sidewall_deg=70.0))

    assert np.array_equal(plain.mat, shaped.mat), "a blanket etch has no sidewall"
    assert any("no mask edge" in line for line in shaped.history), (
        f"nothing explained the ignored profile: {shaped.history}"
    )


def test_stop_on_is_shorthand_for_selectivity(stack):
    stack.deposit_blanket("poly-Si", 60e-9)
    stack.deposit_blanket("a-C", 40e-9)
    stack.etch("a-C", depth=200e-9, stop_on="poly-Si")
    assert stack.thickness_of("poly-Si").min() >= 55e-9


# ---------------------------------------------------------------------------
# The profile geometry itself
# ---------------------------------------------------------------------------


def test_profile_terms_match_hand_calculation():
    """Each term has a closed form; check them rather than eyeballing shapes."""
    from litho_sim.wafer.etch_profile import EtchProfile

    total = 100e-9

    def at(p, d):
        return float(p.offsets(np.array([d]), total)[0])

    assert EtchProfile().is_identity
    assert at(EtchProfile(), 50e-9) == pytest.approx(0.0, abs=1e-15)

    # A wall at 80 degrees runs 1/tan(80) laterally per unit of depth.
    run = -at(EtchProfile(sidewall_deg=80.0), total)
    assert run == pytest.approx(total / np.tan(np.radians(80.0)), rel=1e-9)

    # Above 90 degrees is re-entrant — the hole widens with depth.
    assert at(EtchProfile(sidewall_deg=100.0), total) > 0

    # Corner arcs are tangent at both ends: full radius at the corner, zero a
    # radius away.
    r = 20e-9
    assert at(EtchProfile(top_radius=r), 0.0) == pytest.approx(r)
    assert at(EtchProfile(top_radius=r), r) == pytest.approx(0.0, abs=1e-15)
    assert at(EtchProfile(bottom_radius=r), total) == pytest.approx(-r)
    assert at(EtchProfile(bottom_radius=r), total - r) == pytest.approx(0.0, abs=1e-15)

    # The foot tapers linearly from its extent at the floor to nothing.
    foot = EtchProfile(footer_height=20e-9, footer_extent=15e-9)
    assert at(foot, total) == pytest.approx(-15e-9)
    assert at(foot, total - 10e-9) == pytest.approx(-7.5e-9)
    assert at(foot, total - 20e-9) == pytest.approx(0.0, abs=1e-15)


def test_profile_terms_add():
    from litho_sim.wafer.etch_profile import EtchProfile

    d, total = np.array([40e-9]), 100e-9
    bias = EtchProfile(bias=10e-9).offsets(d, total)
    wall = EtchProfile(sidewall_deg=80.0).offsets(d, total)
    both = EtchProfile(bias=10e-9, sidewall_deg=80.0).offsets(d, total)
    assert both[0] == pytest.approx(bias[0] + wall[0])


def test_describe_is_empty_for_an_identity_profile():
    from litho_sim.wafer.etch_profile import EtchProfile

    assert EtchProfile().describe() == ""
    assert "82" in EtchProfile(sidewall_deg=82.0).describe()


# ---------------------------------------------------------------------------
# Lateral exposure: etching a surface that is not the top of a column
# ---------------------------------------------------------------------------


def _capped_layer(grid) -> Stack:
    """A nitride layer capped by oxide, open only on the wall of a trench."""
    s = Stack.blank(grid, dz=4e-9, headroom=200e-9)
    s.deposit_blanket("SiN", 20e-9)
    s.deposit_blanket("SiO2", 20e-9)
    trench = np.zeros(s.shape_xy, dtype=bool)
    trench[:, :4] = True
    s.etch(["SiO2", "SiN"], depth=200e-9, blanket=True, open_mask=trench)
    return s


def test_exposure_defaults_to_the_column_rule(grid):
    """The new mode must not change a single existing recipe."""
    s = _capped_layer(grid)
    before = s.mat.copy()
    s.etch("SiN", depth=20e-9)
    assert np.array_equal(before, s.mat)
    s.etch("SiN", depth=20e-9, exposure="surface")
    assert np.array_equal(before, s.mat)


def test_a_lateral_etch_reaches_a_layer_open_only_on_its_side(grid):
    s = _capped_layer(grid)
    capped = int((s.mat == 3).sum())
    s.etch("SiN", depth=40e-9, exposure="any")
    assert int((s.mat == 3).sum()) < capped, "the front never entered the wall"


def test_a_lateral_etch_leaves_the_cap_alone(grid):
    """Selectivity as rate contrast: everything unnamed is a wall."""
    s = _capped_layer(grid)
    oxide = int((s.mat == 2).sum())
    s.etch("SiN", depth=40e-9, exposure="any")
    assert int((s.mat == 2).sum()) == oxide, "the etch chewed through the cap"


def test_a_lateral_etch_will_not_open_a_sealed_void(grid):
    """The etchant has to come from outside. A void left by a pinched-off
    conformal film is not a source, or every buried seam would etch itself
    open from the inside."""
    s = Stack.blank(grid, dz=4e-9, headroom=200e-9)
    s.deposit_blanket("SiN", 40e-9)
    s.deposit_blanket("SiO2", 20e-9)          # cap it, so the top is not a way in
    # Substrate is 40 nm at dz=4 nm, so the nitride occupies iz 10..19.
    void = np.zeros(s.mat.shape, dtype=bool)
    void[13:16, 4:8, 4:8] = True
    s.clear(void)
    assert (s.mat[13:16, 4:8, 4:8] == VACUUM).all()
    assert (s.mat[12, 4:8, 4:8] == get_material("SiN").id).all(), (
        "the void has to be inside the nitride, or it is not sealed"
    )

    sealed = s.mat.copy()
    s.etch("SiN", depth=40e-9, exposure="any")
    assert np.array_equal(sealed, s.mat), "a sealed void acted as an etch source"


def test_a_lateral_etch_refuses_the_controls_it_cannot_honour(grid):
    """Refuse rather than ignore. A shape control that silently does nothing is
    how a working engine comes to look broken."""
    from litho_sim.wafer.etch_profile import EtchProfile

    s = _capped_layer(grid)
    with pytest.raises(ValueError, match="no single edge"):
        s.etch("SiN", depth=20e-9, exposure="any",
               profile=EtchProfile(sidewall_deg=80.0))
    with pytest.raises(ValueError, match="contradict"):
        s.etch("SiN", depth=20e-9, exposure="any", line_of_sight=True)
    with pytest.raises(ValueError, match="exposure must be"):
        s.etch("SiN", depth=20e-9, exposure="sideways")


def test_an_idle_etch_names_which_of_the_four_reasons_it_was(grid):
    """One sentence used to cover four situations with four different fixes."""
    def note(build, **kw):
        s = build()
        s.etch(**kw)
        return s.history[-1]

    def bare():
        return Stack.blank(grid, dz=4e-9, headroom=200e-9)

    def coated():
        s = bare()
        s.deposit_blanket("SiO2", 20e-9)
        return s

    assert "no SiARC anywhere" in note(coated, targets="SiARC", depth=20e-9)
    assert "step order" in note(coated, targets="SiARC", depth=20e-9)

    def buried():
        s = bare()
        s.deposit_blanket("poly-Si", 20e-9)
        s.deposit_blanket("SiO2", 20e-9)
        return s

    assert "buried under SiO2" in note(buried, targets="poly-Si", depth=20e-9)

    assert "only on a sidewall" in note(
        lambda: _capped_layer(grid), targets="SiN", depth=20e-9)

    closed = np.zeros((grid.n_pixels, grid.n_pixels), dtype=bool)
    assert "open mask closed" in note(
        coated, targets="SiO2", depth=20e-9, open_mask=closed)


# ---------------------------------------------------------------------------
# Selective deposition
# ---------------------------------------------------------------------------


def test_selective_deposition_reduces_to_the_old_behaviour(grid):
    """`on=` naming everything present must be byte-identical to omitting it,
    or every existing recipe is at risk."""
    for kw in ({}, {"planarize": True}):
        a = Stack.blank(grid, dz=4e-9, headroom=100e-9)
        b = Stack.blank(grid, dz=4e-9, headroom=100e-9)
        a.deposit_blanket("SiGe", 12e-9, **kw)
        b.deposit_blanket("SiGe", 12e-9, on="Si", **kw)
        assert np.array_equal(a.mat, b.mat)

    c = Stack.blank(grid, dz=4e-9, headroom=100e-9)
    d = Stack.blank(grid, dz=4e-9, headroom=100e-9)
    c.deposit_conformal("SiO2", 8e-9)
    d.deposit_conformal("SiO2", 8e-9, on="Si")
    assert np.array_equal(c.mat, d.mat)


def test_selective_deposition_with_nothing_to_nucleate_on_says_so(grid):
    s = Stack.blank(grid, dz=4e-9, headroom=100e-9)
    s.deposit_blanket("SiGe", 10e-9, on="TiN")
    assert "nucleate" in s.history[-1] and "no-op" in s.history[-1]

    t = Stack.blank(grid, dz=4e-9, headroom=100e-9)
    t.deposit_conformal("SiO2", 8e-9, on="TiN")
    assert "nucleate" in t.history[-1] and "no-op" in t.history[-1]


def test_a_lateral_etch_is_independent_of_the_empty_headroom(grid):
    """The solve is cropped for speed; the answer must not know that.

    Padding the crop by "how far the front could travel" is the intuitive
    choice and 6x more expensive than needed, because every open voxel is
    already a front source and the front never travels *through* vacuum. This
    pins the equivalence: a taller stack is more empty space and nothing else,
    so it must give byte-identical material.
    """
    out = []
    for headroom in (120e-9, 400e-9):
        s = Stack.blank(grid, dz=4e-9, headroom=headroom)
        s.deposit_blanket("SiN", 20e-9)
        s.deposit_blanket("SiO2", 20e-9)
        trench = np.zeros(s.shape_xy, dtype=bool)
        trench[:, :4] = True
        s.etch(["SiO2", "SiN"], depth=200e-9, blanket=True, open_mask=trench)
        s.etch("SiN", depth=40e-9, exposure="any")
        out.append(s.mat[:30].copy())

    assert np.array_equal(out[0], out[1]), "the crop changed the answer"
