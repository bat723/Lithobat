"""
LithoPy
=======
A photolithography simulation engine organised by processing step: coat →
expose → bake → develop, patterning flows on a voxel wafer, and the analysis
on top. Covers the full aerial-image → resist → CD pipeline, 3-D resist
profiles, multi-patterning (LELE/SADP/SAQP), process windows, and Bossung
curves.

Quickstart
----------
>>> from litho_sim.core import SimulationConfig
>>> from litho_sim.mask import lines_and_spaces
>>> from litho_sim.expose import compute_aerial_image
>>>
>>> cfg = SimulationConfig.from_tech_node("ArF")
>>> mask = lines_and_spaces(cfg.grid.n_pixels, cfg.grid.pixel_size,
...                          pitch=200e-9, cd=100e-9)
>>> aerial = compute_aerial_image(mask, cfg.optics, cfg.grid)

Each subpackage's ``__init__`` is its public API — import from the package
(``litho_sim.wafer``), not the module (``litho_sim.wafer.stack``), and later
internal splits will never touch your code. Subpackages import lazily; ``ml``
additionally needs the optional torch extra for everything except ``defects``.
"""

__version__ = "2.0.0"
__author__ = "LithoPy Contributors"
__all__ = [
    "core",        # configs, grids, logging
    "mask",        # geometry, layouts, pixel patterns
    "expose",      # pupil, illumination, sources, aerial image
    "coat",        # film-stack optics (TMM, BARC, swing curves)
    "bake",        # PEB diffusion (arrives with the bake/ split)
    "develop",     # resist models, 2-D and 3-D
    "wafer",       # voxel Stack + materials
    "patterning",  # steps, flows, recipe families
    "analysis",    # process windows, Bossung, NILS
    "viz",         # 2-D plots + 3-D renderers
    "tech",        # reserved: DRAM/NAND pipelines, device ladder
    "ml",          # defect study (torch optional)
]
