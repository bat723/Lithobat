"""The wafer state: a voxel material volume and the material library.

``stack`` holds the (nz, ny, nx) uint8 volume that process steps mutate and
the 3-D viewer draws; ``materials`` is the library a voxel value indexes
into. Reserved here for the device ladder: a second per-voxel dopant volume
(``wafer/dopants.py``) — nothing in this package may assume the wafer state
is a single array.
"""

from litho_sim.wafer.materials import (
    MATERIAL_LIBRARY,
    VACUUM,
    Material,
    get_material,
)
from litho_sim.wafer.stack import Stack

__all__ = ["MATERIAL_LIBRARY", "VACUUM", "Material", "Stack", "get_material"]
