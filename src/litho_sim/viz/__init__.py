"""Visualisation: 2-D matplotlib figures and the 3-D stack renderers.

``plots`` draws aerial images, resist footprints, Bossung curves, and process
windows; ``viz3d`` renders the voxel Stack (Plotly when available, matplotlib
otherwise) and the 3-D resist profile.
"""

from litho_sim.viz.plots import (
    apply_style,
    plot_aerial_image,
    plot_bossung_curves,
    plot_cd_heatmap,
    plot_process_window,
    plot_resist_profile,
    save_figure,
)
from litho_sim.viz.viz3d import (
    cross_section_figure,
    payload_size,
    profile_figure,
    resist_profile_3d_figure,
    resist_stack_for_display,
    resist_surface_mpl,
    stack_figure,
    stack_figure_mpl,
    surface_payload,
    voxel_mesh,
)

__all__ = [
    "apply_style", "plot_aerial_image", "plot_bossung_curves", "plot_cd_heatmap",
    "plot_process_window", "plot_resist_profile", "save_figure",
    "cross_section_figure", "payload_size",
    "profile_figure", "resist_profile_3d_figure", "resist_surface_mpl",
    "resist_stack_for_display",
    "stack_figure", "stack_figure_mpl", "surface_payload", "voxel_mesh",
]
