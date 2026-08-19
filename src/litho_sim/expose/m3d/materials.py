"""Optical constants for mask materials, at the wavelengths masks are used at.

:mod:`litho_sim.wafer.materials` already carries a refractive index per
material, but it is a *wafer* library: single-wavelength, 193 nm, and it knows
about etch rates and voxel colours. None of the materials a mask is built from
are in it — no chrome, no MoSi, no TaBN, no ruthenium, no molybdenum — and at
13.5 nm its numbers are meaningless, because every material there has
``n ≈ 1`` and the interesting digits are in the fourth decimal place.

So mask constants live here, keyed by wavelength.

The sign convention, which is a trap
------------------------------------
This package stores ``n + ik`` with **positive k meaning absorbing**, matching
:data:`litho_sim.wafer.materials.MATERIAL_LIBRARY`, and
:func:`litho_sim.coat.films._propagation_index` conjugates it for propagation.

EUV constants are published everywhere as ``n = 1 − δ − iβ``. Entering that
form directly gives layers with a *negative* stored k, which the conjugation
then turns into gain: multilayers that reflect more than 100 %. Store
``complex(1 - delta, +beta)``.

Where the numbers come from
---------------------------
EUV values are the standard CXRO/Henke atomic scattering factors at 13.5 nm.
The check that they are entered correctly is not a citation but a physical
one: a 40-bilayer Mo/Si stack built from them reflects **74.7 % at 6°**, which
is the textbook ideal. Real blanks measure 65–68 %; the gap is Mo/Si
interdiffusion, which a sharp-interface model does not have and which is worth
knowing before the difference is mistaken for a bug.
"""

from __future__ import annotations

#: Mask-material refractive indices, ``{wavelength_m: {name: n + ik}}``.
#:
#: Positive imaginary part = absorbing.  See the module docstring: EUV values
#: are ``1 - delta + i*beta``, *not* the ``1 - delta - i*beta`` of the tables
#: they come from.
MASK_MATERIALS: dict[float, dict[str, complex]] = {
    13.5e-9: {
        "vacuum": 1.0 + 0.0j,
        # Multilayer constituents.  The Mo/Si pair is what it is because their
        # optical contrast at 13.5 nm is large while Si's absorption is small —
        # no other pair of practical materials does better.
        "Si": 0.99931 + 0.00182j,
        "Mo": 0.92108 + 0.00644j,
        "Ru": 0.88651 + 0.01710j,       # capping layer, protects against oxidation
        # Absorbers.  TaBN is the workhorse; the boron and nitrogen are there
        # for amorphousness and etch behaviour, not optics.
        "TaBN": 0.94900 + 0.03120j,
        "TaBO": 0.97000 + 0.02500j,     # anti-reflective topcoat on the absorber
        "Ta": 0.94270 + 0.04170j,
        "SiO2": 0.97800 + 0.01070j,     # low-thermal-expansion substrate
    },
    193e-9: {
        "vacuum": 1.0 + 0.0j,
        "air": 1.0 + 0.0j,
        "quartz": 1.5603 + 0.0j,        # fused silica, the transmissive substrate
        "Cr": 0.84 + 1.65j,             # binary absorber; 70 nm is OD > 3
        "CrON": 1.95 + 1.05j,           # anti-reflective coating over chrome
        "MoSi": 2.343 + 0.586j,         # attenuated PSM; ~72 nm gives 6 % at 180 deg
    },
}


def mask_material(name: str, wavelength: float, tol: float = 1e-11) -> complex:
    """Refractive index of a mask material at an exposure wavelength.

    Parameters
    ----------
    name : str
        Material name, e.g. ``"TaBN"``.
    wavelength : float
        Exposure wavelength [m].  Must match a tabulated wavelength — these are
        not dispersion curves, and interpolating between 13.5 nm and 193 nm
        would be meaningless.
    tol : float
        Absolute tolerance [m] when matching the wavelength.

    Returns
    -------
    complex
        ``n + ik``, positive ``k`` absorbing.

    Raises
    ------
    KeyError
        For an unknown material, or a wavelength with no table.
    """
    for lam, table in MASK_MATERIALS.items():
        if abs(lam - wavelength) <= tol:
            if name not in table:
                raise KeyError(
                    f"Unknown mask material '{name}' at {lam * 1e9:.1f} nm. "
                    f"Known: {sorted(table)}"
                )
            return table[name]
    known = ", ".join(f"{w * 1e9:.1f} nm" for w in sorted(MASK_MATERIALS))
    raise KeyError(
        f"No mask-material table at {wavelength * 1e9:.3f} nm. Tabulated: {known}"
    )
