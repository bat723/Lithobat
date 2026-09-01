"""
Tests for the process flow engine and the multi-patterning recipes.

The claims worth defending here are the emergent ones:

* **Pitch walking is not modelled, it emerges.** Give one exposure in a LELE
  flow an overlay offset and the printed lines must alternate by roughly
  twice that offset — nothing in the code computes that number.
* **Pitch division is not modelled either.** Conformal deposit + directional
  etch-back + strip must turn N mandrels into 2N lines whose width equals
  the deposited thickness, with no step that knows about "doubling".
* **The etch mask is emergent.** Etch takes no mask argument; whatever sits
  on top of a column protects it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask.layout import Layout, line_array
from litho_sim.patterning import (
    CMP,
    STEP_REGISTRY,
    Deposit,
    Develop,
    Etch,
    Expose,
    Flow,
    Measure,
    Pattern,
    PostExposureBake,
    ProcessStep,
    SpacerEtchback,
    SpinCoat,
    Strip,
    lele,
    sadp,
    saqp,
    single_exposure,
)
from litho_sim.wafer import Stack, get_material


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=128, pixel_size=4e-9, dz=4e-9, n_z_slices=7)


@pytest.fixture(scope="module")
def optics() -> OpticsConfig:
    return OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=11)


@pytest.fixture(scope="module")
def resist() -> ResistConfig:
    # dose_nominal is calibrated so the printed trench matches the drawn CD.
    return ResistConfig(
        n_resist=1.7, dill_A=0.8, dill_B=0.05, dill_C=0.04,
        dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9,
    )


@pytest.fixture(scope="module")
def dense_layout() -> Layout:
    """Six lines at 80 nm pitch — below single-exposure contrast."""
    return Layout(line_array(6, pitch=80e-9, cd=40e-9, length=600e-9), name="gates")


def _centres(widths_profile: np.ndarray, pixel_size: float) -> list:
    out, run = [], None
    for i, v in enumerate(widths_profile):
        if v and run is None:
            run = i
        elif not v and run is not None:
            out.append((run + i - 1) / 2 * pixel_size)
            run = None
    return out


# ---------------------------------------------------------------------------
# Step plumbing
# ---------------------------------------------------------------------------


def test_all_steps_are_registered():
    for kind in ("spincoat", "deposit", "pattern", "expose", "peb", "develop",
                 "etch", "etchback", "strip", "cmp", "measure"):
        assert kind in STEP_REGISTRY


@pytest.mark.parametrize(
    "step",
    [
        SpinCoat(material="SOC", thickness=50e-9),
        Expose(layout="a", dose=1.2, overlay_dx=3e-9),
        PostExposureBake(diffusion_sigma=15e-9),
        Develop(model="mack"),
        Etch(targets="SOC", depth=40e-9, selectivity={"Si": 0.1}),
        Strip(material="photoresist"),
        Deposit(material="spacer-oxide", thickness=18e-9),
        Measure(name="cd"),
        # The two that were missing. CMP in particular was never round-tripped
        # and never used by any flow, and its underlying Stack.planarize()
        # turned out to raise IndexError on every call.
        CMP(height=60e-9),
        CMP(),
        SpacerEtchback(material="spacer-nitride", thickness=14e-9),
        Pattern(pitch=100e-9, cd=50e-9, shape="contacts"),
    ],
)
def test_step_dict_round_trip(step):
    assert ProcessStep.from_dict(step.to_dict()) == step


def test_unknown_step_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown process step"):
        ProcessStep.from_dict({"kind": "levitate"})


def test_expose_without_resist_is_a_clear_error(grid, optics, resist):
    flow = Flow(steps=[Expose(layout="a")], grid=grid, optics=optics,
                resist=resist, layouts={"a": Layout()})
    with pytest.raises(RuntimeError, match="no photoresist"):
        flow.run(snapshot=False)


def test_expose_with_unknown_layout_is_a_clear_error(grid, optics, resist):
    flow = Flow(
        steps=[SpinCoat(material="photoresist", thickness=40e-9), Expose(layout="nope")],
        grid=grid, optics=optics, resist=resist, layouts={"a": Layout()},
    )
    with pytest.raises(RuntimeError, match="not in the recipe"):
        flow.run(snapshot=False)


def test_develop_without_expose_is_a_clear_error(grid, optics, resist):
    flow = Flow(
        steps=[SpinCoat(material="photoresist", thickness=40e-9), Develop()],
        grid=grid, optics=optics, resist=resist,
    )
    with pytest.raises(RuntimeError, match="no Expose step has run"):
        flow.run(snapshot=False)


# ---------------------------------------------------------------------------
# Flow mechanics
# ---------------------------------------------------------------------------


def test_flow_snapshots_are_independent(grid, optics, resist):
    flow = Flow(
        steps=[SpinCoat(material="SOC", thickness=40e-9),
               SpinCoat(material="a-C", thickness=40e-9)],
        grid=grid, optics=optics, resist=resist,
    )
    res = flow.run(snapshot=True)
    assert len(res.snapshots) == 2
    assert res.snapshots[0].volume_fraction("a-C") == 0.0
    assert res.snapshots[1].volume_fraction("a-C") > 0.0


def test_flow_on_step_callback_fires(grid, optics, resist):
    seen = []
    flow = Flow(steps=[SpinCoat(material="SOC", thickness=20e-9)] * 3,
                grid=grid, optics=optics, resist=resist)
    flow.run(snapshot=False, on_step=lambda i, s, st: seen.append(i))
    assert seen == [0, 1, 2]


def test_flow_reports_which_step_failed(grid, optics, resist):
    flow = Flow(steps=[SpinCoat(material="SOC", thickness=20e-9), Develop()],
                grid=grid, optics=optics, resist=resist)
    with pytest.raises(RuntimeError, match="Step 2"):
        flow.run(snapshot=False)


def test_flow_json_round_trip(tmp_path, grid, optics, resist, dense_layout):
    flow = lele(dense_layout, grid, optics, resist, min_spacing=70e-9,
                overlay=(4e-9, 0.0))
    p = tmp_path / "flow.json"
    flow.to_json(p)
    back = Flow.from_json(p)
    assert back.name == flow.name
    assert len(back) == len(flow)
    assert [s.to_dict() for s in back.steps] == [s.to_dict() for s in flow.steps]
    assert set(back.layouts) == set(flow.layouts)


def test_flow_json_restores_integer_zernike_keys(tmp_path, grid, resist):
    """JSON stringifies dict keys; Zernike indices must come back as ints.

    If they do not, the pupil silently applies no aberration at all.
    """
    optics = OpticsConfig(zernike_coeffs={7: 0.05, 11: -0.02})
    flow = Flow(steps=[], grid=grid, optics=optics, resist=resist)
    p = tmp_path / "f.json"
    flow.to_json(p)
    back = Flow.from_json(p)
    assert back.optics.zernike_coeffs == {7: 0.05, 11: -0.02}


def test_flow_from_json_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Flow.from_json(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Emergent etch masking
# ---------------------------------------------------------------------------


def test_etch_is_masked_by_whatever_is_on_top(grid, optics, resist):
    """Etch takes no mask argument — the top material is the mask.

    This is the property that lets one primitive serve LELE, SADP, and cut
    masks without any of them being special-cased.
    """
    st = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9, headroom=200e-9)
    st.deposit_blanket("SOC", 40e-9)
    ny, nx = st.shape_xy
    # Cap half the field with resist.
    zz = np.arange(st.nz)[:, None, None]
    surf = st.surface_index()
    cap = (zz > surf[None]) & (zz <= surf[None] + 10)
    cap[:, :, nx // 2:] = False
    st.fill(cap, "photoresist")

    st.etch("SOC", depth=40e-9)
    left = st.thickness_of("SOC")[:, : nx // 2].mean()
    right = st.thickness_of("SOC")[:, nx // 2:].mean()
    assert left > right, "resist-capped half should have been protected"
    assert right < 8e-9, "uncapped half should have cleared"


# ---------------------------------------------------------------------------
# Pattern — the step that gives an etch a mask edge to work on
# ---------------------------------------------------------------------------


def _half_field(st) -> np.ndarray:
    """Open the right-hand half of the field, so an etch leaves a step in it."""
    open_col = np.zeros(st.shape_xy, bool)
    open_col[:, st.shape_xy[1] // 2:] = True
    return open_col


def _patterned(grid, **kw):
    """A film with an ideal resist mask printed on it."""
    from litho_sim.patterning import ProcessContext

    st = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9, headroom=400e-9)
    st.deposit_blanket("poly-Si", 100e-9)
    Pattern(**kw).apply(st, ProcessContext(grid=grid))
    return st


def test_a_pattern_leaves_a_mask_edge_for_the_etch_to_shape(grid):
    """The whole reason the step exists.

    Every laterally-varying etch control — bias, wall angle, corner radii,
    line-of-sight, undercut — needs a mask edge to act on. On a blanket film
    there is none, and the tab could build nothing else, so eight of the
    etch's fourteen controls were byte-for-byte inert.
    """
    from litho_sim.wafer.etch_profile import EtchProfile

    st = _patterned(grid, pitch=200e-9, cd=96e-9)
    vertical = st.copy()
    vertical.etch("poly-Si", depth=100e-9)
    tapered = st.copy()
    tapered.etch("poly-Si", depth=100e-9, profile=EtchProfile(sidewall_deg=75.0))

    assert not np.array_equal(vertical.mat, tapered.mat), (
        "a wall angle must change the wafer once there is a wall to angle"
    )

    # And it must narrow downward, not merely differ.
    row = tapered.shape_xy[0] // 2
    poly = tapered.mat == get_material("poly-Si").id
    open_top = int((~poly[tapered._n_voxels(95e-9) - 1, row]).sum())
    open_bot = int((~poly[tapered._n_voxels(30e-9), row]).sum())
    assert open_bot < open_top, f"75° wall did not taper: {open_top} -> {open_bot} px"


def test_a_pattern_carves_only_the_film_it_just_printed(grid):
    """The mask is identified by "what was vacuum before", and it has to be.

    A planarising coat is thicker in a trench than over a line, so no fixed
    voxel count finds it; and the wafer may already carry a buried layer of
    the same material, which carving by material alone would destroy.
    """
    from litho_sim.patterning import ProcessContext

    st = Stack.blank(grid, dz=grid.dz, substrate_thickness=20e-9, headroom=500e-9)
    st.deposit_blanket("photoresist", 40e-9)      # a buried layer of the same stuff
    st.deposit_blanket("poly-Si", 60e-9)
    st.etch("poly-Si", depth=30e-9,               # real topography to planarise over
            open_mask=_half_field(st))
    buried = st.thickness_of("photoresist").copy()

    Pattern(material="photoresist", thickness=90e-9,
            pitch=200e-9, cd=96e-9).apply(st, ProcessContext(grid=grid))

    resist = st.thickness_of("photoresist")
    assert (resist >= buried).all(), "the buried layer of the same material was eaten"
    masked = resist[resist > buried + 1e-12]
    assert masked.size, "no mask was printed at all"
    assert masked.max() > masked.min(), (
        "a planarising coat over topography cannot be one uniform thickness — "
        "so identifying it by a fixed voxel count would be wrong"
    )


@pytest.mark.parametrize("pitch", [96e-9, 128e-9, 200e-9, 300e-9])
def test_the_array_tiles_the_field_at_any_pitch(grid, pitch):
    """Commensurate or not, no corner of the field goes unpatterned.

    Asserting the tiling, never the line count: the count is derived from the
    grid on purpose, so pinning it would pin the wrong thing.
    """
    st = _patterned(grid, pitch=pitch, cd=pitch / 2)
    top = st.top_material()
    mask = top == get_material("photoresist").id

    assert mask.any() and not mask.all(), "both a mask and an opening must exist"
    # Every column band of one pitch has to contain some of each.
    span = max(int(round(pitch / grid.pixel_size)), 1)
    for x0 in range(0, st.shape_xy[1] - span + 1, span):
        band = mask[:, x0:x0 + span]
        assert band.any(), f"columns {x0}:{x0 + span} have no mask at all"


def test_a_pattern_that_cannot_break_the_film_says_so(grid):
    """CD at or above the pitch leaves no mask edge — and must not do it quietly."""
    st = _patterned(grid, pitch=100e-9, cd=100e-9)
    assert any("no mask edge" in line for line in st.history), (
        f"an unbroken film went unreported: {st.history}"
    )


# ---------------------------------------------------------------------------
# LELE — pitch walking must emerge from overlay
# ---------------------------------------------------------------------------


def test_lele_decomposes_and_runs(grid, optics, resist, dense_layout):
    flow = lele(dense_layout, grid, optics, resist, min_spacing=70e-9)
    assert set(flow.layouts) == {"colorA", "colorB"}
    assert sum(len(layer) for layer in flow.layouts.values()) == len(dense_layout)
    res = flow.run(snapshot=False)
    m = res.measurements.iloc[-1]
    assert m["n_lines"] >= 4, "LELE should print the dense pattern"


def test_lele_rejects_an_uncolourable_layout(grid, optics, resist, dense_layout):
    """An odd conflict cycle is a design-rule violation, and must say so."""
    tri = Layout([
        line_array(1, 1e-9, 20e-9, 100e-9)[0],
    ])
    from litho_sim.mask.geometry import Rect

    tri = Layout([
        Rect("m", 0, 0, 20e-9, 20e-9),
        Rect("m", 30e-9, 0, 20e-9, 20e-9),
        Rect("m", 15e-9, 26e-9, 20e-9, 20e-9),
    ])
    with pytest.raises(ValueError, match="not 2-colourable"):
        lele(tri, grid, optics, resist, min_spacing=25e-9)


def test_lele_zero_overlay_prints_uniformly(grid, optics, resist, dense_layout):
    flow = lele(dense_layout, grid, optics, resist, min_spacing=70e-9,
                overlay=(0.0, 0.0))
    m = flow.run(snapshot=False).measurements.iloc[-1]
    # Only grid quantisation should remain.
    assert m["pitch_walk_nm"] <= grid.pixel_size * 1e9, (
        f"perfect overlay still produced {m['pitch_walk_nm']:.1f} nm of walk"
    )


@pytest.mark.parametrize("overlay_nm", [6.0, 12.0])
def test_lele_overlay_produces_pitch_walking(
    grid, optics, resist, dense_layout, overlay_nm
):
    """Displacing one exposure must alternate the printed features.

    Nothing in the engine computes pitch walk — the second exposure is
    simply imaged from geometry shifted by the overlay, and the alternation
    falls out. It should land near twice the overlay.
    """
    flow = lele(dense_layout, grid, optics, resist, min_spacing=70e-9,
                overlay=(overlay_nm * 1e-9, 0.0))
    m = flow.run(snapshot=False).measurements.iloc[-1]
    walk = m["pitch_walk_nm"]
    expected = 2.0 * overlay_nm
    assert walk == pytest.approx(expected, abs=1.5 * grid.pixel_size * 1e9), (
        f"overlay {overlay_nm} nm gave {walk:.1f} nm walk, expected ~{expected:.1f} nm"
    )


def test_lele_pitch_walk_is_monotonic_in_overlay(grid, optics, resist, dense_layout):
    walks = []
    for ov in (0.0, 4.0, 8.0, 12.0):
        flow = lele(dense_layout, grid, optics, resist, min_spacing=70e-9,
                    overlay=(ov * 1e-9, 0.0))
        walks.append(flow.run(snapshot=False).measurements.iloc[-1]["pitch_walk_nm"])
    assert walks == sorted(walks), f"pitch walk should grow with overlay: {walks}"


def test_le3_uses_three_masks(grid, optics, resist, dense_layout):
    flow = lele(dense_layout, grid, optics, resist, min_spacing=70e-9, n_colors=3)
    assert set(flow.layouts) == {"colorA", "colorB", "colorC"}


# ---------------------------------------------------------------------------
# SADP / SAQP — pitch division must emerge from deposition
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sadp_grid() -> GridConfig:
    """640 nm field — a 4-slit mandrel array at 160 nm pitch needs 640 nm."""
    return GridConfig(n_pixels=160, pixel_size=4e-9, dz=4e-9, n_z_slices=7)


@pytest.fixture(scope="module")
def mandrel_layout() -> Layout:
    return Layout(line_array(4, pitch=160e-9, cd=80e-9, length=700e-9), name="mandrel")


def test_sadp_doubles_the_feature_count(sadp_grid, optics, resist, mandrel_layout):
    """N mandrels must become 2N spacer walls.

    No step in the flow knows it is doubling anything — the count follows
    from a conformal film having two sidewalls per mandrel.
    """
    flow = sadp(mandrel_layout, sadp_grid, optics, resist, spacer_thickness=20e-9)
    res = flow.run(snapshot=False)
    by_name = {r["name"]: r for _, r in res.measurements.iterrows()}
    n_mandrel = by_name["mandrel"]["n_lines"]
    n_spacer = by_name["spacers"]["n_lines"]
    assert n_mandrel >= 2
    assert n_spacer >= 2 * n_mandrel, (
        f"{n_mandrel} mandrels produced only {n_spacer} spacers"
    )


def test_sadp_spacer_width_equals_deposited_thickness(
    sadp_grid, optics, resist, mandrel_layout
):
    """The final CD is set by deposition, not by the optics.

    That is the whole reason SADP exists — deposition thickness is far
    better controlled than a printed edge.

    The sweep stops where the geometry does: the printed mandrels leave a
    ~44 nm gap, so two opposing spacers fit only while 2t stays under it.
    The old ceiling of 24 nm passed only because ``deposit_conformal``
    under-deposited films at exact voxel multiples; with that fixed, 24 nm
    spacers genuinely merge — asserted below as its own case.
    """
    for t in (12e-9, 16e-9, 20e-9):
        flow = sadp(mandrel_layout, sadp_grid, optics, resist, spacer_thickness=t)
        res = flow.run(snapshot=False)
        by_name = {r["name"]: r for _, r in res.measurements.iterrows()}
        widths = by_name["spacers"]["lines_nm"]
        assert widths, "no spacers printed"
        assert np.mean(widths) == pytest.approx(
            t * 1e9, abs=1.5 * sadp_grid.pixel_size * 1e9
        ), f"deposited {t*1e9:.0f} nm, measured {np.mean(widths):.0f} nm"

    # Past the gap limit the two sidewall films meet: the measured feature is
    # the whole filled gap, decisively wider than the deposited thickness.
    flow = sadp(mandrel_layout, sadp_grid, optics, resist, spacer_thickness=24e-9)
    res = flow.run(snapshot=False)
    by_name = {r["name"]: r for _, r in res.measurements.iterrows()}
    widths = by_name["spacers"]["lines_nm"]
    assert widths, "no spacers printed"
    assert np.mean(widths) > 24.0 + 1.5 * sadp_grid.pixel_size * 1e9, (
        f"24 nm spacers in a ~44 nm gap should merge, measured "
        f"{np.mean(widths):.0f} nm"
    )


def test_sadp_halves_the_pitch(sadp_grid, optics, resist, mandrel_layout):
    flow = sadp(mandrel_layout, sadp_grid, optics, resist, spacer_thickness=20e-9)
    res = flow.run(snapshot=False)
    st = res.stack
    # The spacer is stripped at the end of the flow, so measure the pattern it
    # transferred into the target film.
    prof = st.line_profile("poly-Si", z_frac=0.9)
    centres = _centres(prof, st.pixel_size)
    assert len(centres) >= 4, f"only {len(centres)} transferred features"
    spacing = float(np.min(np.diff(centres)))
    assert spacing < 160e-9, (
        f"feature spacing {spacing*1e9:.0f} nm should be under the "
        "160 nm mandrel pitch — SADP must divide the pitch"
    )


def test_saqp_divides_pitch_twice(sadp_grid, optics, resist, mandrel_layout):
    """SAQP is literally the SADP block run twice.

    Spacer 1 becomes the mandrel for spacer 2. That this needs no new step
    types is the evidence the abstraction is right.
    """
    flow = saqp(mandrel_layout, sadp_grid, optics, resist,
                spacer1_thickness=24e-9, spacer2_thickness=16e-9)
    res = flow.run(snapshot=False)
    by_name = {r["name"]: r for _, r in res.measurements.iterrows()}
    n1 = by_name["spacer1"]["n_lines"]
    n2 = by_name["spacer2"]["n_lines"]
    assert n1 > by_name["mandrel"]["n_lines"]
    assert n2 > n1, f"second division did not increase the count: {n1} -> {n2}"


def test_single_exposure_runs(grid, optics, resist):
    lay = Layout(line_array(3, pitch=200e-9, cd=100e-9, length=600e-9))
    flow = single_exposure(lay, grid, optics, resist)
    res = flow.run(snapshot=False)
    assert res.measurements.iloc[-1]["n_lines"] >= 1
    assert res.stack.volume_fraction("poly-Si") > 0.0


def test_registered_kind_matches_the_field_default():
    """`register_step("etch")` and `kind: str = "etch"` are written separately.

    Nothing checks they agree. If they ever diverged, the decorator would set
    the class attribute while `__init__` kept the old default — so `to_dict()`
    would emit one name and `from_dict()` would reconstruct under another, and
    a saved recipe would quietly load as a different step. Cheap to pin, and a
    GUI that builds steps from `STEP_REGISTRY` keys depends on it.
    """
    for kind, cls in STEP_REGISTRY.items():
        assert cls.kind == kind, f"{cls.__name__} registered as '{kind}'"
        assert cls().kind == kind, (
            f"{cls.__name__}() defaults to kind='{cls().kind}', not '{kind}'"
        )
