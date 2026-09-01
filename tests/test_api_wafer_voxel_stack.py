"""
Interface-level tests for the voxel wafer stack.

Everything here goes through the public package surface — ``litho_sim.wafer``
— the way real consumers do: patterning steps drive the mutators, the device
presets snapshot and reload stacks, the 3-D viewer reads occupancy and
present-material queries, and the app asks what is on top. Unit-level etch
and CMP mechanics live in tests/test_stack.py; these tests pin the
*contract* those consumers rely on, quantitatively where the unit tests are
qualitative:

* conformal growth is the same physical thickness on a horizontal top, a
  horizontal bottom, and a vertical sidewall of one step;
* selective deposition actually partitions the field by exposed seed, not
  merely reduces to the old behaviour;
* occupancy agrees with volume_fraction and smoothing keeps the mass;
* column_runs keeps two same-material intervals in one column apart;
* deposits grow the array when headroom runs out, invisibly to the caller;
* a save/load round trip preserves the derived queries, not just the bytes;
* degenerate dimensions (sub-voxel thicknesses, a one-pixel grid) build and
  answer rather than crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig
from litho_sim.wafer import VACUUM, Stack, get_material

PX = 4e-9
DZ = 4e-9


def _eq(a, b) -> bool:
    """Strict geometric equality.

    ``np.allclose`` defaults to ``atol=1e-8`` — 10 nm, larger than every
    feature in these tests, which makes the default comparison vacuously
    true. All geometry here is voxel-exact, so compare to picometres.
    """
    return np.allclose(a, b, rtol=0.0, atol=1e-12)


@pytest.fixture
def grid() -> GridConfig:
    return GridConfig(n_pixels=32, pixel_size=PX)


def _right_half_open(st: Stack) -> np.ndarray:
    """Open-mask that etches the right half of the field, edge at nx // 2."""
    ny, nx = st.shape_xy
    m = np.zeros((ny, nx), dtype=bool)
    m[:, nx // 2:] = True
    return m


def _mesa(grid: GridConfig) -> Stack:
    """A 48 nm poly-Si mesa over the left half, bare Si over the right.

    Built entirely through the public mutators, the way a patterning flow
    would: blanket deposit, then a masked etch down to the substrate.
    Substrate top is at 20 nm, mesa top at 68 nm, step edge at x = 16.
    """
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=100e-9)
    st.deposit_blanket("poly-Si", 48e-9)
    st.etch("poly-Si", depth=200e-9, open_mask=_right_half_open(st),
            stop_on="Si")
    return st


def _half_capped(grid: GridConfig) -> Stack:
    """Left half capped by 20 nm SiO2, right half exposing bare Si."""
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=100e-9)
    st.deposit_blanket("SiO2", 20e-9)
    st.etch("SiO2", depth=100e-9, open_mask=_right_half_open(st),
            stop_on="Si")
    return st


# ---------------------------------------------------------------------------
# End-to-end: conformal coating of a real step
# ---------------------------------------------------------------------------


def test_end_to_end_conformal_coating_of_a_step(grid):
    """One flow, checked in every direction the film is supposed to grow.

    The unit tests establish that a conformal film appears on a sidewall at
    all; a consumer building a spacer needs more — that one nominal
    thickness quantizes to the *same* number of voxels on the mesa top, the
    low field, and sideways off the vertical wall.

    The nominal thickness is deliberately not a voxel multiple: the growth
    shell is ``dist <= t``, and at an exact multiple the last shell sits on
    a floating-point knife edge (3 * 4e-9 > 1.2e-8 in float64), so a
    "12 nm" film would land at 8 nm. 14 nm quantizes unambiguously to
    12 nm everywhere.
    """
    st = _mesa(grid)
    t_nom = 14e-9
    t_vox = 12e-9  # floor(t_nom / voxel) * voxel, in every direction
    st.deposit_conformal("SiN", t_nom)

    ny, nx = st.shape_xy
    edge = nx // 2  # first etched column
    row = ny // 2

    # The masked etch stopped on Si but consumed one voxel of it — the
    # documented cost of stop_on (cf. the >= 55 nm tolerances on a 60 nm
    # stop layer in the unit tests). The low field is therefore at 16 nm.
    field_base = 20e-9 - DZ

    # Horizontal surfaces, away from the step: the same quantized thickness
    # on both levels, measured per column.
    thick = st.thickness_of("SiN")
    mesa_interior = thick[:, : edge - 4]
    field_interior = thick[:, edge + 6:]
    assert _eq(mesa_interior, t_vox), (
        f"mesa-top coverage {np.unique(mesa_interior)} != {t_vox}"
    )
    assert _eq(field_interior, t_vox), (
        f"low-field coverage {np.unique(field_interior)} != {t_vox}"
    )

    # Top heights: each level rises by exactly the film thickness.
    top = st.top_height()
    assert _eq(top[:, : edge - 4], 68e-9 + t_vox)
    assert _eq(top[:, edge + 6:], field_base + t_vox)

    # The sidewall: at a height above the low-field film but below the mesa
    # top, the only SiN in the centre row is the wall coating, and it
    # extends the same quantized thickness sideways from the step edge.
    iz = int(48e-9 / DZ)  # 48 nm: inside the mesa, above the field film
    sin_id = get_material("SiN").id
    wall = np.nonzero(st.mat[iz, row] == sin_id)[0]
    assert wall.size == int(t_vox / PX), (
        f"sidewall is {wall.size} px, expected {int(t_vox / PX)}"
    )
    assert wall.min() == edge, "the wall must start at the mesa edge"

    # And the film is a *coating*: the mesa underneath is untouched.
    assert _eq(st.thickness_of("poly-Si")[:, : edge - 4], 48e-9)

    # The step announced itself to the history log the app displays.
    assert any("conformal SiN" in line for line in st.history)


def test_a_voxel_multiple_conformal_thickness_keeps_its_last_shell(grid):
    """A 12 nm film on a 4 nm grid must be 12 nm, not 8.

    The growth shell is ``dist <= t`` on distances that come out of a square
    root, so at 3 voxels the computed distance was one ULP above 1.2e-8 and
    the last shell vanished — a full voxel short — while 16 nm (exact in
    float) deposited in full. Pinned here in both directions, since spacer
    CDs are built directly from this thickness.
    """
    st = _mesa(grid)
    t = 12e-9  # an exact voxel multiple, the once-broken case
    st.deposit_conformal("SiN", t)

    nx = st.shape_xy[1]
    edge = nx // 2
    assert _eq(st.thickness_of("SiN")[:, edge + 6:], t), (
        "the film lost its last shell to floating-point rounding"
    )
    row = st.shape_xy[0] // 2
    wall = np.nonzero(
        st.mat[int(48e-9 / DZ), row] == get_material("SiN").id
    )[0]
    assert wall.size == int(t / PX), "the sidewall lost its last shell"


# ---------------------------------------------------------------------------
# Selective deposition partitions the field
# ---------------------------------------------------------------------------


def test_selective_blanket_grows_only_where_the_seed_is_exposed(grid):
    """`on="Si"` must split a half-capped field, not just no-op or blanket.

    The unit tests check the two ends (reduces to blanket when everything is
    a seed; no-ops when nothing is). This is the case selective epi exists
    for: the same call, one field, two different answers by column.
    """
    st = _half_capped(grid)
    nx = st.shape_xy[1]
    edge = nx // 2
    st.deposit_blanket("SiGe", 16e-9, on="Si")

    thick = st.thickness_of("SiGe")
    assert _eq(thick[:, edge:], 16e-9), "epi missing over exposed Si"
    assert (thick[:, :edge] == 0.0).all(), "epi nucleated on the oxide cap"

    # What the app's "what is on top" query now sees, per region.
    top = st.top_material()
    assert (top[:, edge:] == get_material("SiGe").id).all()
    assert (top[:, :edge] == get_material("SiO2").id).all()
    assert any("on Si" in line for line in st.history)


def test_selective_conformal_stays_off_the_capped_region(grid):
    """Conformal selectivity: the film coats exposed Si and not the cap.

    The growth distance is straight-line, so a thin bleed past the boundary
    is documented behaviour — the assertion stays a bleed-width away from
    the edge on the capped side.
    """
    st = _half_capped(grid)
    nx = st.shape_xy[1]
    edge = nx // 2
    t_nom = 10e-9  # off the voxel grid on purpose; quantizes to 8 nm
    t_vox = 8e-9
    st.deposit_conformal("SiGe", t_nom, on="Si")

    thick = st.thickness_of("SiGe")
    assert _eq(thick[:, edge + 2:], t_vox), (
        "conformal film missing over the exposed seed"
    )
    margin = int(round(t_nom / PX)) + 1
    assert (thick[:, : edge - margin] == 0.0).all(), (
        "conformal film grew over the capped region, beyond the "
        "documented straight-line bleed"
    )


# ---------------------------------------------------------------------------
# Occupancy — the renderer's contract
# ---------------------------------------------------------------------------


def test_occupancy_agrees_with_volume_fraction(grid):
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=100e-9)
    st.deposit_blanket("SiO2", 24e-9)

    occ = st.occupancy("SiO2")
    assert occ.dtype == np.float32
    assert set(np.unique(occ)) <= {0.0, 1.0}, "raw occupancy must be binary"
    assert occ.shape == st.mat.shape
    assert occ.mean() == pytest.approx(st.volume_fraction("SiO2"))


def test_smoothed_occupancy_keeps_the_mass_and_softens_the_edge(grid):
    """Smoothing exists so marching cubes can find a sub-voxel surface.

    That only works if the smoothed field actually has intermediate values
    at the interface — and only makes sense if it still integrates to the
    same amount of material.
    """
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=100e-9)
    st.deposit_blanket("SiO2", 24e-9)

    raw = st.occupancy("SiO2")
    smooth = st.occupancy("SiO2", smooth=1.0)
    assert smooth.sum() == pytest.approx(raw.sum(), rel=0.02), (
        "smoothing changed the amount of material"
    )
    assert ((smooth > 0.05) & (smooth < 0.95)).any(), (
        "no intermediate values — nothing for marching cubes to interpolate"
    )
    assert float(smooth.min()) >= 0.0 and float(smooth.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# column_runs — the metrology query that justifies voxels
# ---------------------------------------------------------------------------


def test_column_runs_separates_stacked_liners(grid):
    """Two SiN films with oxide between them are two runs, not one.

    This is the stacked-nanosheet / liner-under-fill shape: the whole reason
    the wafer state is a voxel volume and not a height field per layer.
    """
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=100e-9)
    st.deposit_blanket("SiN", 12e-9)   # iz 5..7
    st.deposit_blanket("SiO2", 12e-9)  # iz 8..10
    st.deposit_blanket("SiN", 12e-9)   # iz 11..13

    runs = st.column_runs("SiN", x=3, y=4)
    assert runs == [(5, 8), (11, 14)]

    sin_id = get_material("SiN").id
    for z0, z1 in runs:
        assert (st.mat[z0:z1, 4, 3] == sin_id).all(), (
            "the half-open run indices must slice exactly the material"
        )
    # And the per-column total agrees with the two runs together.
    assert st.thickness_of("SiN")[4, 3] == pytest.approx(24e-9)


# ---------------------------------------------------------------------------
# Headroom grows invisibly
# ---------------------------------------------------------------------------


def test_deposit_grows_the_stack_when_headroom_runs_out(grid):
    """A caller deposits by thickness and never thinks about the array.

    ensure_headroom is unit-tested directly; this pins that the *public*
    mutator invokes it, so a flow that overfills its initial headroom keeps
    working instead of silently truncating the film.
    """
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=20e-9)
    nz_before = st.nz
    st.deposit_blanket("SiO2", 100e-9)

    assert st.nz > nz_before, "the array did not grow"
    assert _eq(st.thickness_of("SiO2"), 100e-9), (
        "the deposit was truncated by the original headroom"
    )
    assert _eq(st.top_height(), 120e-9)
    assert (st.mat[-1] == VACUUM).all(), "growth must add vacuum, not film"


# ---------------------------------------------------------------------------
# present_materials — the viewer legend's contract
# ---------------------------------------------------------------------------


def test_present_materials_orders_bottom_up(grid):
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=100e-9)
    st.deposit_blanket("SiN", 12e-9)
    st.deposit_blanket("SiO2", 12e-9)
    names = [m.name for m in st.present_materials()]
    assert names == ["Si", "SiN", "SiO2"]


# ---------------------------------------------------------------------------
# Save / load preserves the answers, not just the bytes
# ---------------------------------------------------------------------------


def test_save_load_preserves_queries_on_a_topographic_stack(grid, tmp_path):
    """The device presets reload saved stacks and keep measuring them.

    The unit round-trip checks byte equality of a flat film; this one checks
    that a stack with real topography and a conformal film still answers the
    consumer-facing queries identically after a disk round trip.
    """
    st = _mesa(grid)
    st.deposit_conformal("SiN", 14e-9)

    path = tmp_path / "mesa.npz"
    st.save(path)
    back = Stack.load(path)

    assert _eq(back.top_height(), st.top_height())
    for name in ("Si", "poly-Si", "SiN"):
        assert np.array_equal(back.thickness_of(name), st.thickness_of(name))
    assert [m.name for m in back.present_materials()] == \
           [m.name for m in st.present_materials()]
    assert back.history == st.history
    assert back.dz == st.dz
    assert back.pixel_size == st.pixel_size

    # A reloaded stack is a working stack: keep processing it.
    back.strip("SiN")
    assert back.volume_fraction("SiN") == 0.0


# ---------------------------------------------------------------------------
# Cross-sections
# ---------------------------------------------------------------------------


def test_cross_section_shapes_and_bad_axis(grid):
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=40e-9)
    ny, nx = st.shape_xy
    assert st.cross_section("y").shape == (st.nz, nx)
    assert st.cross_section("x").shape == (st.nz, ny)
    with pytest.raises(ValueError, match="axis must be"):
        st.cross_section("z")


def test_ascii_section_renders_every_plane(grid):
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=40e-9)
    st.deposit_blanket("SiO2", 12e-9)
    art = st.ascii_section()
    lines = art.split("\n")
    assert len(lines) == st.nz
    assert all(len(line) == st.shape_xy[1] for line in lines)
    assert lines[0] == "." * st.shape_xy[1], "top line must be vacuum"
    assert "." not in lines[-1], "bottom line must be substrate"


# ---------------------------------------------------------------------------
# Degenerate dimensions
# ---------------------------------------------------------------------------


def test_sub_voxel_thicknesses_round_up_to_one_voxel(grid):
    """Thicknesses far below dz must build a real (one-voxel) layer.

    max(round(t/dz), 1) is the documented floor; a zero-voxel substrate
    would make surface_index -1 everywhere and every later step a no-op.
    """
    st = Stack.blank(grid, dz=DZ, substrate_thickness=1e-10, headroom=1e-10)
    assert st.nz == 2
    assert _eq(st.thickness_of("Si"), DZ)

    st.deposit_blanket("SiO2", 1e-10)
    assert _eq(st.thickness_of("SiO2"), DZ)
    assert _eq(st.top_height(), 2 * DZ)


def test_a_one_pixel_grid_still_works():
    """The smallest lateral field: one column, still a valid wafer."""
    grid = GridConfig(n_pixels=1, pixel_size=PX)
    st = Stack.blank(grid, dz=DZ, substrate_thickness=20e-9, headroom=60e-9)
    assert st.shape_xy == (1, 1)

    st.deposit_blanket("SiO2", 20e-9)
    assert st.thickness_of("SiO2")[0, 0] == pytest.approx(20e-9)
    assert st.top_height()[0, 0] == pytest.approx(40e-9)

    st.etch("SiO2", depth=100e-9, stop_on="Si")
    assert st.thickness_of("SiO2")[0, 0] == 0.0
    # stop_on arrests the etch at the cost of one voxel of the stop layer.
    assert st.thickness_of("Si")[0, 0] >= 20e-9 - DZ - 1e-12
