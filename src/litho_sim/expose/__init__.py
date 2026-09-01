"""Exposure: illumination, sources, and partially coherent imaging.

The Abbe loop (one FFT per source point) lives in ``aerial_image``; source
shapes come from ``illumination`` (legacy named builders) and ``source``
(the parametric ``Source``/``Pole`` objects); ``pupil`` holds the Zernike
machinery; ``photochem`` converts the aerial image into chemistry — photon
absorption and acid generation, deterministic or with the molecular counts
sampled. Mask 3-D effects live in the ``m3d`` subpackage.
"""

from litho_sim.expose.aerial_image import (
    compute_aerial_image,
    extract_cross_section,
    normalisation_scale,
)
from litho_sim.expose.illumination import SOURCE_TYPES, build_source
from litho_sim.expose.photochem import (
    SpeciesSample,
    absorbed_fraction,
    generate_acid,
    mean_absorbed_photons,
    photon_energy,
    sample_species,
)
from litho_sim.expose.pupil import pupil_grid, zernike_noll
from litho_sim.expose.source import SOURCE_PRESETS, Pole, Source

__all__ = [
    "compute_aerial_image",
    "extract_cross_section",
    "normalisation_scale",
    "SOURCE_TYPES",
    "build_source",
    "zernike_noll",
    "pupil_grid",
    "SOURCE_PRESETS",
    "Pole",
    "Source",
    "SpeciesSample",
    "absorbed_fraction",
    "generate_acid",
    "mean_absorbed_photons",
    "photon_energy",
    "sample_species",
]
