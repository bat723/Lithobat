"""What gets printed: vector geometry, layouts, and pixel mask patterns.

``geometry``/``layout`` are the vector path (shapes → anti-aliased raster,
decomposition, overlay); ``patterns`` is the legacy pixel path (analytic
L/S, contacts, PSM conversion). Both produce the (n, n) transmittance array
the imaging engine consumes.
"""

from litho_sim.mask.geometry import (
    Contact,
    PathShape,
    Polygon,
    Rect,
    Shape,
    polygon_distance,
    rasterize_shapes,
    shape_from_dict,
    shapes_overlap_distance,
)
from litho_sim.mask.layout import (
    Layout,
    contact_grid,
    cut_bar,
    decompose,
    line_array,
    split_by_color,
)
from litho_sim.mask.patterns import (
    apply_bias,
    checkerboard,
    contact_array,
    isolated_line,
    lines_and_spaces,
    to_attenuated_psm,
)

__all__ = [
    "Contact", "PathShape", "Polygon", "Rect", "Shape", "polygon_distance",
    "rasterize_shapes", "shape_from_dict", "shapes_overlap_distance",
    "Layout", "contact_grid", "cut_bar", "decompose", "line_array", "split_by_color",
    "apply_bias", "checkerboard", "contact_array", "isolated_line",
    "lines_and_spaces", "to_attenuated_psm",
]
