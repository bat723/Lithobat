"""The material library: what a voxel value means.

Split out of ``stack.py`` so the library can grow independently of the wafer
state — ``register_material()`` (user-defined metals, low-k dielectrics,
extra spacers for SAnP) lands here per the roadmap. Built-in IDs are stable
so saved stacks stay readable; user-registered IDs will start at 16.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VACUUM = 0


@dataclass(frozen=True)
class Material:
    """A film material.

    Attributes
    ----------
    id : int
        Voxel value used for this material.  ``0`` is reserved for vacuum.
    name : str
        Human-readable name, also the key into :data:`MATERIAL_LIBRARY`.
    color : str
        Hex colour used by the 3-D viewer.
    opacity : float
        Render opacity in [0, 1].
    etch_rate : float
        Etch rate *relative* to a nominal rate of 1.0.  A near-zero value
        makes the material an etch stop; this is how selectivity is
        expressed.
    role : str
        ``"substrate"``, ``"hardmask"``, ``"mandrel"``, ``"spacer"``,
        ``"resist"``, ``"film"``, ``"metal"``, or ``"vacuum"``.  Process
        steps use this to find their target without hard-coding material
        names.
    n_index : complex
        Refractive index at the exposure wavelength.  Used by the 3-D resist
        model for absorption and substrate reflection.
    se_yield : float
        Secondary-electron yield under a CD-SEM's beam, *relative to the flat
        top of photoresist* (which emits 1 by the SEM model's convention).
        This is what makes one material read brighter than another on a
        micrograph — see :mod:`litho_sim.metrology.sem`. The library's
        numbers are contrast ratios at a low-kV landing energy, ordered the
        way the materials image and about the right size, not calibrated
        yields: oxides and nitrides high (insulators emit strongly), silicon
        and germanium below the resist, carbon lowest.
    """

    id: int
    name: str
    color: str = "#8899aa"
    opacity: float = 1.0
    etch_rate: float = 1.0
    role: str = "film"
    n_index: complex = 1.5 + 0j
    se_yield: float = 1.0


def _lib(*materials: Material) -> dict[str, Material]:
    return {m.name: m for m in materials}


#: Built-in materials.  IDs are stable so saved stacks stay readable.
#:
#: The last column is the SE yield relative to the resist top. Silicon's 0.6
#: is the same number as :attr:`SEMConfig.substrate_yield`'s default, so a
#: resist line imaged from a print and the same line imaged as a wafer stack
#: put the floor at the same grey. Insulators sit above the resist — a low-kV
#: beam charges them and they emit strongly — germanium a little above
#: silicon, carbon films below it, and the metal between.
MATERIAL_LIBRARY: dict[str, Material] = _lib(
    Material(0, "vacuum", "#00000000", 0.0, 0.0, "vacuum", 1.0 + 0j, 0.0),
    Material(1, "Si", "#5a6270", 1.0, 0.30, "substrate", 0.883 + 2.778j, 0.60),
    Material(2, "SiO2", "#7fb2d9", 0.9, 1.00, "film", 1.563 + 0j, 1.10),
    Material(3, "SiN", "#d9a441", 0.95, 0.60, "film", 2.010 + 0j, 0.90),
    Material(4, "poly-Si", "#9b7fd9", 1.0, 0.80, "film", 1.900 + 1.100j, 0.60),
    Material(5, "a-C", "#3f4550", 1.0, 1.20, "mandrel", 1.700 + 0.400j, 0.45),
    Material(6, "SOC", "#4a3f35", 1.0, 1.20, "hardmask", 1.500 + 0.300j, 0.45),
    Material(7, "SiARC", "#c8913a", 0.95, 0.90, "hardmask", 1.700 + 0.200j, 0.80),
    Material(8, "photoresist", "#4fd97f", 0.85, 1.50, "resist", 1.700 + 0.010j, 1.00),
    Material(9, "spacer-oxide", "#7fd9d2", 0.9, 0.15, "spacer", 1.563 + 0j, 1.10),
    Material(10, "spacer-nitride", "#d97f9b", 0.9, 0.20, "spacer", 2.010 + 0j, 0.90),
    # Front-end materials. SiGe is the sacrificial half of a gate-all-around
    # superlattice: grown alternating with Si, then removed laterally to leave
    # the channel sheets suspended (`Stack.etch(exposure="any")`). Its rate is
    # set well above Si's so the release is selective by default — a real flow
    # runs 100:1 or better, and `selectivity=` still overrides this.
    #
    # SiGe images only a little brighter than Si: the two are hard to tell
    # apart on a real SE micrograph too, which is why a fab reaches for the
    # backscatter detector or a TEM to count sheets. The gap here is enough
    # to see the superlattice, not so much that it looks like a different
    # class of material.
    Material(11, "SiGe", "#6f8f6a", 1.0, 2.00, "film", 1.100 + 2.900j, 0.70),
    Material(12, "TiN", "#c9c04f", 1.0, 0.25, "metal", 1.300 + 1.600j, 0.85),
)

_BY_ID: dict[int, Material] = {m.id: m for m in MATERIAL_LIBRARY.values()}


def get_material(ref: str | int | Material) -> Material:
    """Resolve a material by name, ID, or pass-through.

    Raises
    ------
    KeyError
        If *ref* names no known material.
    """
    if isinstance(ref, Material):
        return ref
    if isinstance(ref, (int, np.integer)):
        if int(ref) not in _BY_ID:
            raise KeyError(f"No material with id {ref}. Known: {sorted(_BY_ID)}")
        return _BY_ID[int(ref)]
    if ref not in MATERIAL_LIBRARY:
        raise KeyError(
            f"Unknown material '{ref}'. Known: {sorted(MATERIAL_LIBRARY)}"
        )
    return MATERIAL_LIBRARY[ref]
