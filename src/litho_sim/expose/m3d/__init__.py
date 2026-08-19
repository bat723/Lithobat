"""Mask 3-D effects: what a mask with real thickness does to the image.

The rest of the engine treats the mask as a Kirchhoff screen — a 2-D complex
transmittance, infinitely thin, with amplitude and phase dropping abruptly to
zero at the absorber edge. That step is not a solution of Maxwell's equations,
and real absorbers are not thin: 60 nm of TaBN is 4.4 wavelengths at EUV, and
68 nm of MoSi is a third of a wavelength at 193 nm.

What the thin mask throws away is everything that depends on the *angle* light
arrives at, and that is where the artefacts come from: contrast loss, a
pitch-dependent best-focus shift, image placement error, telecentricity error,
and — at EUV, where the chief ray comes in at 6° — the absorber casting a
shadow into its own trench.

Three models, in increasing cost
--------------------------------
``"thin"``
    The Kirchhoff screen. One FFT, shared by every source point.

``"multilayer"``
    Thin-mask spectrum, but each diffraction order reflects off the EUV
    multilayer at *its own* angle. Nearly free, because
    :mod:`litho_sim.coat.films` already solves a planar stack exactly. It is
    not a small correction — see :mod:`~litho_sim.expose.m3d.multilayer`.

``"fdtd"``
    Maxwell's equations solved around the absorber; the near field on the mask
    exit plane becomes the mask transmission function. The only model that
    captures shadowing, and the only one that needs a cached angle library.

The seam
--------
All three present the same interface — :func:`make_spectrum_provider` returns
something that, given a source point, yields a mask spectrum. That is the whole
coupling to :mod:`litho_sim.expose.aerial_image`, and it is why the imaging
path did not have to be restructured to gain a rigorous mask model.
"""

from __future__ import annotations

from litho_sim.expose.m3d.materials import MASK_MATERIALS, mask_material
from litho_sim.expose.m3d.multilayer import Multilayer
from litho_sim.expose.m3d.nearfield import (
    MaskGeometry,
    NearFieldLibrary,
    build_library,
    build_topography,
    spectrum_on_imaging_grid,
)
from litho_sim.expose.m3d.provider import (
    MASK_MODELS,
    FDTDSpectra,
    MultilayerSpectra,
    SpectrumProvider,
    ThinMaskSpectra,
    clear_library_cache,
    make_spectrum_provider,
    mask_side_sin_theta,
)
from litho_sim.expose.m3d.stack import MASK_STACK_PRESETS, REGIMES, MaskStack
from litho_sim.expose.m3d.yee import FDTDResult, solve_near_field

__all__ = [
    "FDTDResult",
    "MASK_MATERIALS",
    "MASK_MODELS",
    "MASK_STACK_PRESETS",
    "REGIMES",
    "MaskGeometry",
    "MaskStack",
    "Multilayer",
    "NearFieldLibrary",
    "FDTDSpectra",
    "MultilayerSpectra",
    "SpectrumProvider",
    "ThinMaskSpectra",
    "build_library",
    "build_topography",
    "clear_library_cache",
    "make_spectrum_provider",
    "mask_material",
    "mask_side_sin_theta",
    "solve_near_field",
    "spectrum_on_imaging_grid",
]
