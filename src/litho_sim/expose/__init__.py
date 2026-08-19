"""Exposure: illumination, sources, and partially coherent imaging.

The Abbe loop (one FFT per source point) lives in ``aerial_image``; source
shapes come from ``illumination`` (legacy named builders) and ``source``
(the parametric ``Source``/``Pole`` objects); ``pupil`` holds the Zernike
machinery. Vector imaging and mask 3-D effects land here later.
"""

from litho_sim.expose.aerial_image import compute_aerial_image, extract_cross_section
from litho_sim.expose.illumination import SOURCE_TYPES, build_source
from litho_sim.expose.pupil import pupil_grid, zernike_noll
from litho_sim.expose.source import SOURCE_PRESETS, Pole, Source

__all__ = [
    "compute_aerial_image",
    "extract_cross_section",
    "SOURCE_TYPES",
    "build_source",
    "zernike_noll",
    "pupil_grid",
    "SOURCE_PRESETS",
    "Pole",
    "Source",
]
