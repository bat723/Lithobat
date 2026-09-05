"""Visualisation: 2-D matplotlib figures and the 3-D stack renderers.

``theme`` is the one visual system every figure follows; ``plots`` draws
aerial images, resist footprints, Bossung curves, and process windows on it;
``viz3d`` extracts voxel meshes and renders the stack as stills (the desktop
app's live solid view is ``litho_sim.app.solid_view``).
"""

from litho_sim.viz import theme
from litho_sim.viz.plots import (
    apply_style,
    plot_aerial_image,
    plot_bossung_curves,
    plot_cd_heatmap,
    plot_opc,
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
    "theme", "apply_style", "plot_aerial_image", "plot_bossung_curves", "plot_cd_heatmap",
    "plot_opc", "plot_process_window", "plot_resist_profile", "save_figure",
    "cross_section_figure", "payload_size",
    "profile_figure", "resist_profile_3d_figure", "resist_surface_mpl",
    "resist_stack_for_display",
    "stack_figure", "stack_figure_mpl", "surface_payload", "voxel_mesh",
]
