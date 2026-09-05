"""Exposure: illumination, sources, and partially coherent imaging.

The Abbe loop (one FFT per source point, batched) lives in ``aerial_image``,
with ``compute_aerial_planes`` for several defocus planes in one pass;
``hopkins`` is the same imaging as sum-of-coherent-systems kernels, for loops
that hold the optics fixed and change the mask; source shapes come from
``illumination`` (legacy named builders) and ``source`` (the parametric
``Source``/``Pole`` objects); ``pupil`` holds the Zernike machinery;
``photochem`` converts the aerial image into chemistry — photon absorption
and acid generation, deterministic or with the molecular counts sampled.
Mask 3-D effects live in the ``m3d`` subpackage.
"""

from litho_sim.expose.aerial_image import (
    SourcePoints,
    compute_aerial_image,
    compute_aerial_planes,
    extract_cross_section,
    normalisation_scale,
    source_points,
)
from litho_sim.expose.hopkins import SOCSKernels
from litho_sim.expose.illumination import SOURCE_TYPES, build_source
from litho_sim.expose.photochem import (
    SpeciesSample,
    absorbed_fraction,
    generate_acid,
    generate_acid_3d,
    mean_absorbed_photons,
    mean_absorbed_photons_3d,
    photon_energy,
    sample_species,
    sample_species_3d,
)
from litho_sim.expose.pupil import pupil_grid, zernike_noll
from litho_sim.expose.source import SOURCE_PRESETS, Pole, Source

__all__ = [
    "compute_aerial_image",
    "compute_aerial_planes",
    "extract_cross_section",
    "normalisation_scale",
    "SourcePoints",
    "source_points",
    "SOCSKernels",
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
    "generate_acid_3d",
    "mean_absorbed_photons_3d",
    "sample_species_3d",
]
