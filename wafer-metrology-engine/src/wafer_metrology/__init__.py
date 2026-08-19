"""
wafer-metrology-engine
======================
Wafer surface-metrology and warpage characterisation for silicon-photonics
packaging.

The package is a metrology-side companion to the LithoPy forward imaging
engine: where LithoPy answers *what prints*, this package answers *where the
surface actually is* -- the bow, warp, thickness variation and site flatness
that set the focus budget of a scanner and the fibre-to-chip coupling loss of a
packaged photonic die.

Pipeline
--------
``synthesize`` -> ``zernike`` -> ``interferometry`` -> ``flatness`` ->
``defects`` -> ``doe``

Units
-----
Every public function takes and returns **metres** unless the name says
otherwise (``*_nm``, ``*_um``, ``*_mm``, ``*_pct``).  Wafer maps are square
arrays with ``NaN`` outside the circular aperture.

Quickstart
----------
>>> from wafer_metrology.synthesize import synthesize_wafer_pair
>>> from wafer_metrology.flatness import compute_flatness
>>>
>>> pair = synthesize_wafer_pair(n_pixels=256, seed=0)
>>> m = compute_flatness(pair)
>>> round(m.warp * 1e6, 1)                                # doctest: +SKIP
41.2
"""

from __future__ import annotations

__version__ = "1.1.0"
__author__ = "LithoPy Contributors"
__all__ = [
    "synthesize",
    "zernike",
    "flatness",
    "interferometry",
    "deflectometry",
    "defects",
    "coupling",
    "doe",
    "plotting",
    "pipeline",
]

# Physical constants shared across the package.
WAFER_DIAMETER_300MM: float = 300e-3
"""Nominal diameter of a 300 mm wafer [m]."""

WAFER_THICKNESS_300MM: float = 775e-6
"""SEMI M1 nominal thickness of a 300 mm wafer [m]."""

WAFER_DIAMETER_150MM: float = 150e-3
"""Nominal diameter of a 150 mm wafer [m] (the hardware build's substrate)."""

WAFER_THICKNESS_150MM: float = 675e-6
"""SEMI nominal thickness of a 150 mm wafer [m]."""

HENE_WAVELENGTH: float = 632.8e-9
"""Helium-neon laser vacuum wavelength [m]."""
