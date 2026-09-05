"""Interface tests for the wafer material library.

Everything here goes through the public package surface —
``litho_sim.wafer`` — because that is what the rest of the engine imports.
The load-bearing contract is ID stability: a voxel volume is uint8, saved
stacks store raw IDs, and the viewer, the patterning steps and the device
presets all name materials by the strings pinned below. If an ID or a name
moves, every saved stack on disk silently changes meaning.

Complementary to ``tests/test_stack.py``, which checks lookup mechanics on
one material; here the whole library is held to the contract at once, along
with resolution of materials from saved records.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import litho_sim.wafer as wafer
from litho_sim.core.config import GridConfig
from litho_sim.wafer import MATERIAL_LIBRARY, VACUUM, Material, Stack, get_material

# ---------------------------------------------------------------------------
# The public surface
# ---------------------------------------------------------------------------


def test_the_package_exports_the_material_api():
    """Consumers import from ``litho_sim.wafer``, never the private module."""
    assert {"MATERIAL_LIBRARY", "VACUUM", "Material", "get_material"} <= set(
        wafer.__all__
    )
    for name in ("MATERIAL_LIBRARY", "VACUUM", "Material", "get_material"):
        assert hasattr(wafer, name)


# ---------------------------------------------------------------------------
# Stable IDs — the saved-stack contract
# ---------------------------------------------------------------------------

#: The built-in table, pinned. A saved stack is a uint8 volume of these IDs
#: with no names attached, so renumbering is a silent corruption of every
#: stack on disk. Adding new materials is fine; moving these is not.
PINNED_IDS = {
    "vacuum": 0,
    "Si": 1,
    "SiO2": 2,
    "SiN": 3,
    "poly-Si": 4,
    "a-C": 5,
    "SOC": 6,
    "SiARC": 7,
    "photoresist": 8,
    "spacer-oxide": 9,
    "spacer-nitride": 10,
    "SiGe": 11,
    "TiN": 12,
}


def test_builtin_ids_are_pinned():
    assert {m.name: m.id for m in MATERIAL_LIBRARY.values()} == PINNED_IDS


def test_builtin_ids_leave_room_for_user_registration():
    """The documented plan reserves IDs >= 16 for ``register_material()``."""
    for m in MATERIAL_LIBRARY.values():
        assert 0 <= m.id < 16, f"{m.name} collides with the user-ID range"


def test_the_vacuum_sentinel_is_the_vacuum_material():
    """``VACUUM`` is compared against raw voxel values all over the engine."""
    assert VACUUM == 0
    assert get_material(VACUUM).name == "vacuum"
    assert get_material("vacuum").id == VACUUM


def test_library_keys_are_the_material_names():
    """The name is the key — app material pickers iterate the dict keys."""
    for name, m in MATERIAL_LIBRARY.items():
        assert m.name == name


def test_every_material_resolves_to_the_same_object_by_name_and_id():
    """Name and ID are two spellings of one identity, across the whole
    library — not just for the one material ``test_stack`` spot-checks."""
    for m in MATERIAL_LIBRARY.values():
        assert get_material(m.name) is m
        assert get_material(m.id) is m


def test_lookup_accepts_numpy_integers_from_a_voxel_volume():
    """The volume is uint8, so IDs arrive as numpy scalars, not ints.

    ``np.unique(stack.mat)`` hands the viewer ``np.uint8`` values and those
    go straight into ``get_material`` — a Python-int-only check would break
    every legend."""
    ids = np.array([PINNED_IDS["poly-Si"]], dtype=np.uint8)
    assert get_material(ids[0]).name == "poly-Si"
    assert get_material(np.int64(PINNED_IDS["Si"])).name == "Si"


# ---------------------------------------------------------------------------
# Properties the consumers actually read
# ---------------------------------------------------------------------------


def test_properties_are_sane_for_rendering_and_physics():
    """The viewer feeds ``color``/``opacity`` to matplotlib, the resist model
    reads ``n_index`` — a bad value fails deep inside a draw call."""
    hex_color = re.compile(r"^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$")
    for m in MATERIAL_LIBRARY.values():
        assert hex_color.match(m.color), f"{m.name}: unparseable color {m.color!r}"
        assert 0.0 <= m.opacity <= 1.0, f"{m.name}: opacity {m.opacity}"
        assert m.etch_rate >= 0.0, f"{m.name}: negative etch rate"
        n = complex(m.n_index)
        assert n.real > 0.0, f"{m.name}: non-physical refractive index {n}"
        assert n.imag >= 0.0, f"{m.name}: gain medium {n} (k must be >= 0)"
        assert isinstance(m.role, str) and m.role, f"{m.name}: empty role"
    # Everything real must etch at *some* rate, or unit-rate depth budgets
    # (depth = time x rate) can never remove it even when it is the target.
    for name, m in MATERIAL_LIBRARY.items():
        if name != "vacuum":
            assert m.etch_rate > 0.0, f"{name} can never be etched"


def test_roles_consumers_dispatch_on():
    """The 3-D viewer hides ``role == "substrate"``; process steps find
    resist by role rather than hard-coding a name."""
    assert get_material("Si").role == "substrate"
    assert get_material("photoresist").role == "resist"
    assert get_material("a-C").role == "mandrel"
    assert get_material("spacer-oxide").role == "spacer"
    assert get_material("SOC").role == "hardmask"


def test_materials_are_immutable():
    """Library entries are shared singletons — mutating one would silently
    repaint or re-rate every stack that references it."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_material("Si").etch_rate = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", [13, 99, 255, -1])
def test_an_unknown_id_raises_and_names_the_known_ones(bad_id):
    """A stray voxel value must fail loudly, and the message must say what
    *is* known — this error surfaces to someone loading a foreign stack."""
    with pytest.raises(KeyError, match="Known:"):
        get_material(bad_id)


def test_an_unknown_name_error_lists_the_library():
    with pytest.raises(KeyError, match="Known:.*photoresist"):
        get_material("unobtanium-nitride")


# ---------------------------------------------------------------------------
# End to end: resolution from records
# ---------------------------------------------------------------------------


@pytest.fixture
def grid() -> GridConfig:
    return GridConfig(n_pixels=32, pixel_size=4e-9)


def test_saved_stack_records_resolve_to_equal_materials(grid, tmp_path):
    """The happy path the whole design rests on.

    A stack is built with the default substrate and the two films the
    patterning steps deposit by default, saved, and loaded back. The save
    path flattens each ``Material`` to a JSON record (complex ``n_index``
    becomes a ``[re, im]`` pair); the load path must resolve those records
    to materials equal to the originals, and the voxel volume must still
    mean the same films.
    """
    st = Stack.blank(grid, dz=4e-9, substrate_thickness=16e-9, headroom=120e-9)
    st.deposit_blanket("photoresist", 24e-9)  # patterning's default coat
    st.deposit_conformal("spacer-oxide", 8e-9)  # patterning's default spacer

    path = tmp_path / "records.npz"
    st.save(path)
    back = Stack.load(path)

    # Every record resolves to a Material equal to the one that was saved —
    # dataclass equality covers the complex n_index round trip through JSON.
    assert back.materials == st.materials
    assert back.materials[PINNED_IDS["Si"]] == get_material("Si")

    # The volume still means the same films, resolved through the library.
    present = {m.name for m in back.present_materials()}
    assert present == {"Si", "photoresist", "spacer-oxide"}


def test_a_stack_saved_before_materials_had_a_yield_takes_the_library_value(grid, tmp_path):
    """Every cached device preset predates ``se_yield``.

    Loading one must not hand every material the dataclass default of 1 —
    a GAA that images as one flat grey — but the library's yield for the
    same ID, which is what the record would have carried had the field
    existed when it was written.
    """
    import json

    st = Stack.blank(grid, dz=4e-9, substrate_thickness=16e-9, headroom=60e-9)
    st.deposit_blanket("SiO2", 12e-9)
    path = tmp_path / "old.npz"
    st.save(path)

    # Rewrite the file the way an older version wrote it: no se_yield key.
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    for rec in meta["materials"].values():
        assert rec.pop("se_yield") is not None, "premise: the new file carries it"
    extra = {k: data[k] for k in data.files if k not in ("mat", "meta")}
    np.savez_compressed(path, mat=data["mat"], meta=json.dumps(meta), **extra)

    back = Stack.load(path)
    assert back.materials == st.materials
    assert back.materials[get_material("SiO2").id].se_yield == get_material("SiO2").se_yield
    assert back.materials[get_material("Si").id].se_yield == get_material("Si").se_yield
    assert get_material("SiO2").se_yield != 1.0, "premise: the library value is not the default"
    for i in np.unique(back.mat):
        if int(i) != VACUUM:
            assert get_material(i) == back.materials[int(i)]


def test_a_voxel_value_without_a_record_resolves_to_a_placeholder(grid):
    """A stack written by a *newer* library may carry IDs this one has no
    record for. The viewer asks ``present_materials`` for a legend, and the
    answer must be a placeholder — not a KeyError in a draw call."""
    st = Stack.blank(grid, dz=4e-9, substrate_thickness=16e-9, headroom=40e-9)
    st.mat[2, 0, 0] = 99  # no such material in this library
    names = {m.name for m in st.present_materials()}
    assert "mat99" in names, f"unknown ID did not fall back cleanly: {names}"
