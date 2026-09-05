"""
3-D visualisation of a printed wafer stack.

Rendering strategy
------------------
The stack is stored as voxels, but voxels are the *wrong thing to send to a
browser*.  A 128×128×40 volume is 655,360 scalar points — roughly 8–12 MB of
JSON per render, and hundreds of megabytes of client memory.  ``go.Volume``
and ``go.Isosurface`` are therefore not used.

Instead each material is reduced to the two height fields that bound it and
drawn as a pair of :class:`plotly.graph_objects.Surface` meshes plus a side
skirt.  A six-material stack is then ~98k points — about 6.7× lighter than
the equivalent volume, and it renders instantly.

That reduction is exact for ordinary films, which occupy one contiguous
z-interval per column.  It is an approximation only for the freestanding
spacer case, where a column can hold two disjoint intervals; the voxel
renderers (:func:`voxel_mesh`, :func:`greedy_mesh`) draw that case exactly.

Plotly is an optional dependency.  Import this module freely — it only
raises if you actually ask for a Plotly figure, and
:func:`stack_figure_mpl` provides a Matplotlib fallback that always works.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig
from litho_sim.viz import theme as _theme
from litho_sim.wafer import VACUUM, Material, Stack, get_material

logger = logging.getLogger(__name__)

try:  # pragma: no cover - trivially environment-dependent
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    go = None
    HAS_PLOTLY = False


_PLOTLY_HINT = (
    "Plotly is required for interactive 3-D figures. Install it with:\n"
    "    pip install plotly\n"
    "or use stack_figure_mpl() / cross_section_figure() for a static "
    "Matplotlib rendering that needs no extra dependencies."
)


def _require_plotly() -> None:
    if not HAS_PLOTLY:
        raise ImportError(_PLOTLY_HINT)


# ---------------------------------------------------------------------------
# Geometry extraction (pure numpy — no plotting dependency)
# ---------------------------------------------------------------------------


def surface_payload(
    stack: Stack, ref: str | int | Material
) -> dict[str, NDArray] | None:
    """Reduce one material to the arrays a surface renderer needs.

    Parameters
    ----------
    stack : Stack
        The wafer stack.
    ref : material ref
        Which material to extract.

    Returns
    -------
    dict or None
        ``x``, ``y`` (1-D axis coordinates in nm), ``z_bottom``, ``z_top``
        (2-D height fields in nm, NaN where the material is absent), and
        ``n_points``.  Returns None if the material is not present.
    """
    z_bot, z_top, present = stack.height_fields(ref)
    if not present.any():
        return None

    ny, nx = stack.shape_xy
    x = np.arange(nx) * stack.pixel_size * 1e9
    y = np.arange(ny) * stack.pixel_size * 1e9
    return {
        "x": x,
        "y": y,
        "z_bottom": z_bot * 1e9,
        "z_top": z_top * 1e9,
        "present": present,
        "n_points": int(2 * present.sum()),
    }


def payload_size(stack: Stack) -> dict[str, int]:
    """Point counts for a surface rendering vs. a naive volume rendering.

    Useful for showing *why* the height-field route is the right default.
    """
    n_surface = 0
    for m in stack.present_materials():
        p = surface_payload(stack, m)
        if p:
            n_surface += p["n_points"]
    return {
        "surface_points": n_surface,
        "volume_points": int(stack.mat.size),
        "ratio": int(stack.mat.size / max(n_surface, 1)),
    }


def _occupancy_and_scale(
    stack: Stack, ref: str | int | Material, downsample: int
):
    """Occupancy mask for one material plus the (dz, px) nm scale, or ``None``.

    The shared front half of both mesh extractors: which voxels are the
    material, decimated, and what one index step is worth in nm.
    """
    occ = stack.mat == get_material(ref).id
    if downsample > 1:
        occ = occ[::downsample, ::downsample, ::downsample]
    if not occ.any():
        return None
    dz = stack.dz * downsample * 1e9
    px = stack.pixel_size * downsample * 1e9
    return occ, dz, px


def voxel_mesh(
    stack: Stack,
    ref: str | int | Material,
    downsample: int = 1,
) -> tuple[NDArray, NDArray] | None:
    """Extract the exact surface of one material as a triangle mesh.

    Emits a quad for every voxel face that touches vacuum (or a different
    material), split into two triangles. The result is the true silhouette of
    the solid — sidewalls, undercut, freestanding spacers and all.

    This exists because :func:`surface_payload` draws only a top and bottom
    height field, which renders as two flat sheets with holes in them and no
    sides. A printed resist line then looks like a floating plane rather than
    a solid with a profile, which is exactly the wrong impression: the
    topology is in the voxels, it just was not being drawn.

    Needs **no scikit-image** — it is pure numpy — so it works everywhere
    and is the default for both renderers.

    Parameters
    ----------
    stack : Stack
        The wafer stack.
    ref : material ref
        Which material to extract.
    downsample : int
        Take every n-th voxel before extraction. Cuts triangle count by
        roughly ``n²`` for large stacks; 1 keeps full detail.

    Returns
    -------
    (vertices, faces) : tuple of NDArray or None
        Vertices in nm as ``(V, 3)`` ordered ``(x, y, z)``; faces as ``(F, 3)``
        index triples. ``None`` if the material is absent.
    """
    extracted = _occupancy_and_scale(stack, ref, downsample)
    if extracted is None:
        return None
    occ, dz, px = extracted

    # A face is exposed where a solid voxel borders a non-solid one. Padding
    # with False means the outer boundary of the array counts as exposed, so
    # the solid is closed rather than open at the field edge.
    pad = np.pad(occ, 1, constant_values=False)
    verts: list[NDArray] = []
    faces: list[NDArray] = []
    n_v = 0

    # (axis, offset, the four corner offsets of the quad in (z, y, x) units)
    directions = [
        (0, +1, [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),   # +z  (top)
        (0, -1, [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),   # -z  (bottom)
        (1, +1, [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),   # +y
        (1, -1, [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),   # -y
        (2, +1, [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),   # +x
        (2, -1, [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),   # -x
    ]
    scale = np.array([dz, px, px])   # per (z, y, x) index step

    for axis, sign, corners in directions:
        neighbour = np.roll(pad, -sign, axis=axis)
        exposed = pad & ~neighbour
        # Strip the padding back off.
        exposed = exposed[1:-1, 1:-1, 1:-1]
        idx = np.argwhere(exposed)
        if idx.size == 0:
            continue
        for c in corners:
            p = (idx + np.array(c)) * scale
            # Reorder (z, y, x) -> (x, y, z) for the renderers.
            verts.append(p[:, [2, 1, 0]])
        m = len(idx)
        base = n_v + np.arange(m)
        # Two triangles per quad; corners were emitted in four blocks of m.
        faces.append(np.stack([base, base + m, base + 2 * m], axis=1))
        faces.append(np.stack([base, base + 2 * m, base + 3 * m], axis=1))
        n_v += 4 * m

    if not verts:
        return None
    V = np.concatenate(verts, axis=0)
    F = np.concatenate(faces, axis=0)
    logger.debug(
        "voxel_mesh(%s): %d triangles", get_material(ref).name, len(F)
    )
    return V, F


#: How far a merged rectangle may grow, in voxels. Not a quality knob — a
#: correctness one. Matplotlib sorts each polygon by a single averaged depth,
#: which is a fair approximation for a small quad and a poor one for a large
#: one, so an unbounded merge produces visible sort errors: far surfaces
#: painting over near ones. Measured on a sectioned GAA against the per-voxel
#: mesh: unbounded is 2,426 triangles / 51 ms / 1.43 % of pixels wrong and has
#: obvious wedge artefacts; 16 is 4,714 / 137 ms / 0.79 % and is hard to fault
#: by eye; 2 is 82,494 / 1861 ms / 0.23 %. 16 is where the curve turns.
MAX_SPAN = 16


def _cover_with_rectangles(mask: NDArray[np.bool_],
                           max_span: int = 0) -> list[tuple[int, int, int, int]]:
    """Greedy maximal-rectangle cover of a 2-D boolean mask.

    Returns ``(r0, r1, c0, c1)`` inclusive index bounds. Walk the first set
    cell, run right while the row stays set, then extend down while every row
    is set across that whole run — the standard greedy pass. It is not the
    minimal cover (that is NP-hard) but on axis-aligned voxel geometry it is
    within a few per cent of it, and a flat wall collapses to one rectangle.
    """
    m = mask.copy()
    out: list[tuple[int, int, int, int]] = []
    rows, cols = m.shape
    while True:
        idx = np.argmax(m)
        if not m.flat[idx]:
            break
        r0, c0 = divmod(int(idx), cols)
        row = m[r0]
        c1 = c0
        cap_c = c0 + max_span - 1 if max_span else cols
        while c1 + 1 < cols and c1 + 1 <= cap_c and row[c1 + 1]:
            c1 += 1
        r1 = r0
        cap_r = r0 + max_span - 1 if max_span else rows
        while r1 + 1 < rows and r1 + 1 <= cap_r and m[r1 + 1, c0:c1 + 1].all():
            r1 += 1
        m[r0:r1 + 1, c0:c1 + 1] = False
        out.append((r0, r1, c0, c1))
    return out


#: Per face direction: (axis, sign, corner order). Each corner is a callable
#: mapping the rectangle's inclusive bounds to a (z, y, x) lattice point. The
#: winding matches :func:`voxel_mesh` exactly, so shading comes out the same.
_FACE_DIRS = (
    (0, +1, lambda k, a0, a1, b0, b1: ((k + 1, a0, b0), (k + 1, a1 + 1, b0),
                                       (k + 1, a1 + 1, b1 + 1), (k + 1, a0, b1 + 1))),
    (0, -1, lambda k, a0, a1, b0, b1: ((k, a0, b0), (k, a0, b1 + 1),
                                       (k, a1 + 1, b1 + 1), (k, a1 + 1, b0))),
    (1, +1, lambda k, a0, a1, b0, b1: ((a0, k + 1, b0), (a0, k + 1, b1 + 1),
                                       (a1 + 1, k + 1, b1 + 1), (a1 + 1, k + 1, b0))),
    (1, -1, lambda k, a0, a1, b0, b1: ((a0, k, b0), (a1 + 1, k, b0),
                                       (a1 + 1, k, b1 + 1), (a0, k, b1 + 1))),
    (2, +1, lambda k, a0, a1, b0, b1: ((a0, b0, k + 1), (a1 + 1, b0, k + 1),
                                       (a1 + 1, b1 + 1, k + 1), (a0, b1 + 1, k + 1))),
    (2, -1, lambda k, a0, a1, b0, b1: ((a0, b0, k), (a0, b1 + 1, k),
                                       (a1 + 1, b1 + 1, k), (a1 + 1, b0, k))),
)


def greedy_mesh(
    stack: Stack,
    ref: str | int | Material,
    downsample: int = 1,
    max_span: int = MAX_SPAN,
) -> tuple[NDArray, NDArray] | None:
    """The surface of one material, with coplanar faces merged.

    Same contract and same silhouette as :func:`voxel_mesh` — this only stops
    describing a flat wall as ten thousand separate quads.

    Why it matters: Matplotlib has no depth buffer. Every frame it projects
    each polygon and sorts them in Python, so cost is set by *triangle count*,
    not by area or by resolution. On a sectioned GAA the per-voxel mesh is
    **292,180 triangles and 6.2 s a frame**; merged it is **2,426 triangles**,
    which is where the interactive view stops being a slideshow. Extraction
    costs more than :func:`voxel_mesh` (~90 ms against ~23 ms) and is paid once
    per stack, against a draw cost paid on every rotation frame.

    Returns
    -------
    (vertices, faces) : tuple of NDArray or None
        Vertices in nm as ``(V, 3)`` ordered ``(x, y, z)``; faces as ``(F, 3)``
        index triples. ``None`` if the material is absent.
    """
    extracted = _occupancy_and_scale(stack, ref, downsample)
    if extracted is None:
        return None
    occ, dz, px = extracted
    scale = np.array([dz, px, px])

    quads: list[tuple] = []
    pad = np.pad(occ, 1, constant_values=False)
    for axis, sign, corners in _FACE_DIRS:
        neighbour = np.roll(pad, -sign, axis=axis)
        exposed = (pad & ~neighbour)[1:-1, 1:-1, 1:-1]
        if not exposed.any():
            continue
        # Only slices that actually carry a face are worth covering.
        for k in np.flatnonzero(exposed.any(axis=tuple(a for a in (0, 1, 2)
                                                       if a != axis))):
            plane = np.take(exposed, int(k), axis=axis)
            for a0, a1, b0, b1 in _cover_with_rectangles(plane, max_span):
                quads.append(corners(int(k), a0, a1, b0, b1))

    if not quads:
        return None

    # (Q, 4, 3) in (z, y, x) lattice units -> nm, reordered to (x, y, z).
    V = (np.asarray(quads, dtype=np.float64) * scale).reshape(-1, 3)[:, [2, 1, 0]]
    base = np.arange(len(quads)) * 4
    F = np.concatenate([
        np.stack([base, base + 1, base + 2], axis=1),
        np.stack([base, base + 2, base + 3], axis=1),
    ], axis=0)
    logger.debug("greedy_mesh(%s): %d quads -> %d triangles",
                 get_material(ref).name, len(quads), len(F))
    return V, F


# ---------------------------------------------------------------------------
# Shared figure plumbing
# ---------------------------------------------------------------------------


def _selected_materials(
    stack: Stack,
    materials: Sequence[str] | None,
    show_substrate: bool = True,
) -> list:
    """The materials a figure will draw, in stack order."""
    present = stack.present_materials()
    if materials is not None:
        wanted = set(materials)
        present = [m for m in present if m.name in wanted]
    if not show_substrate:
        present = [m for m in present if m.role != "substrate"]
    return present


def _scaled_meshes(
    stack: Stack,
    present: Sequence,
    downsample: int,
    z_exaggeration: float,
    y_limit: float | None = None,
    extract=voxel_mesh,
):
    """Yield ``(material, V, F)`` for each material that yields a mesh.

    Vertices come back z-exaggerated. With a *y_limit* [nm], faces beyond the
    cutaway plane are dropped, and a material entirely beyond it is skipped.
    """
    for m in present:
        mesh = extract(stack, m, downsample=downsample)
        if mesh is None:
            continue
        V, F = mesh
        V = V.copy()
        V[:, 2] *= z_exaggeration
        if y_limit is not None:
            F = F[V[F].max(axis=1)[:, 1] <= y_limit]
            if len(F) == 0:
                continue
        yield m, V, F


def _dress_3d_axes(ax, extent_x: float, extent_y: float, height: float,
                   z_exaggeration: float, elev: float, azim: float,
                   zoom: float | None = None) -> None:
    """Extents, labels, box aspect and camera — the shared mpl epilogue."""
    ax.set(
        xlim=(0, extent_x), ylim=(0, extent_y), zlim=(0, height),
        xlabel="x [nm]", ylabel="y [nm]",
        zlabel=f"z [nm]{f' ×{z_exaggeration:g}' if z_exaggeration != 1 else ''}",
    )
    aspect = (extent_x, extent_y, height)
    try:
        if zoom is None:
            ax.set_box_aspect(aspect)
        else:
            ax.set_box_aspect(aspect, zoom=zoom)
    except TypeError:  # pragma: no cover - matplotlib without the zoom kwarg
        try:
            ax.set_box_aspect(aspect)
        except (AttributeError, ValueError):
            pass
    except (AttributeError, ValueError):  # pragma: no cover - old matplotlib
        pass
    ax.view_init(elev=elev, azim=azim)


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------


def stack_figure(
    stack: Stack,
    materials: Sequence[str] | None = None,
    z_exaggeration: float = 1.0,
    cutaway: float | None = None,
    show_substrate: bool = True,
    title: str = "Printed stack",
    downsample: int = 1,
) -> go.Figure:
    """Build an interactive 3-D view of the stack, drawn as solids.

    Parameters
    ----------
    stack : Stack
        The wafer stack to draw.
    materials : sequence of str, optional
        Restrict the view to these materials.  Defaults to everything present.
    z_exaggeration : float
        Vertical scale factor.  100 nm of resist over a 512 nm field is a
        pancake, so every real litho tool exaggerates z; 2–4 is a good range.
    cutaway : float, optional
        Fraction of the y range to keep, in (0, 1].  0.5 slices the stack in
        half so you can see inside it.
    show_substrate : bool
        Draw the substrate slab.  Turning it off declutters the view.
    title : str
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    fig = go.Figure()
    ny, nx = stack.shape_xy

    present = _selected_materials(stack, materials, show_substrate)
    y_keep = ny if cutaway is None else ny * cutaway
    y_limit = None if cutaway is None else y_keep * stack.pixel_size * 1e9

    for m, V, F in _scaled_meshes(stack, present, downsample,
                                  z_exaggeration, y_limit):
        fig.add_trace(
            go.Mesh3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2],
                i=F[:, 0], j=F[:, 1], k=F[:, 2],
                color=m.color, opacity=m.opacity, name=m.name,
                showlegend=True, flatshading=True,
                lighting=dict(ambient=0.45, diffuse=0.85, specular=0.15),
                hovertemplate=(
                    f"<b>{m.name}</b><br>x=%{{x:.0f}} nm<br>"
                    "y=%{y:.0f} nm<br>z=%{z:.1f} nm<extra></extra>"
                ),
            )
        )

    extent = nx * stack.pixel_size * 1e9
    height = stack.nz * stack.dz * 1e9 * z_exaggeration
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x [nm]",
            yaxis_title="y [nm]",
            zaxis_title=(
                f"z [nm]{f' ×{z_exaggeration:g}' if z_exaggeration != 1 else ''}"
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=(y_keep / nx), z=min(height / extent, 1.2)),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    logger.info(
        "stack_figure: %d materials, %d surface points",
        len(present), payload_size(stack)["surface_points"],
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib fallbacks (always available)
# ---------------------------------------------------------------------------


def material_colormap(ids, vacuum: str = "#00000000"):
    """Dense colour LUT for a sparse set of material IDs.

    Returns ``(lut, cmap, norm, materials)``: pass ``lut[mat_array]`` to
    ``imshow`` with *cmap* and *norm* to draw a material section; *materials*
    are the :class:`Material` objects in palette order (vacuum excluded),
    ready for a legend.

    The caller chooses the id set deliberately. A cross-section wants the
    materials present *now*; a step-by-step figure wants every material that
    appears in **any** snapshot, because the sacrificial ones are gone by the
    end — which is exactly why they must still be drawable.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    materials = [get_material(i) for i in ids]
    lut = np.zeros(256, dtype=int)
    for i, mid in enumerate([VACUUM] + list(ids)):
        lut[mid] = i
    cmap = ListedColormap([vacuum] + [m.color for m in materials])
    norm = BoundaryNorm(np.arange(len(ids) + 2) - 0.5, len(ids) + 1)
    return lut, cmap, norm, materials


def cross_section_figure(
    stack: Stack,
    axis: str = "y",
    index: int | None = None,
    ax=None,
    title: str | None = None,
):
    """Draw a vertical cross-section as a labelled Matplotlib image.

    Usually the most informative single view: sidewall angle, film
    thicknesses, spacer position, and standing-wave scalloping are all
    directly readable, and it costs nothing to render.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from litho_sim.viz import theme

    sec = stack.cross_section(axis, index)
    mats = stack.present_materials()
    lut, cmap, norm, _ = material_colormap([m.id for m in mats],
                                           vacuum=theme.SURFACE)
    img = lut[sec]

    if ax is None:
        theme.apply()
        _, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)

    width_nm = sec.shape[1] * stack.pixel_size * 1e9
    height_nm = sec.shape[0] * stack.dz * 1e9
    theme.physical_image(
        ax, img, (0, width_nm, 0, height_nm), cmap, norm=norm, aspect="auto",
        xlabel="x [nm]" if axis == "y" else "y [nm]", ylabel="z [nm]",
    )
    if title is None:
        title = f"Cross-section, {axis}-cut"
    if title:
        theme.title(ax, title)
    # The materials, as swatches beside the picture — never on it.
    ax.legend(handles=theme.material_swatches(mats), loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    return ax


#: Process families, in the vocabulary a process engineer uses rather than the
#: engine's method names. Seven, deliberately: sub-types (blanket vs planarized
#: fill, anisotropic vs lateral etch) are carried in the row text, not by a
#: colour, because past about eight classes adjacent hues stop being telling
#: apart and the panel becomes decoration.
PROCESS_FAMILIES: dict[str, str] = dict(_theme.FAMILY)

#: (matcher, family, how to say it). Order matters — the first match wins, so
#: the specific forms sit above the general ones.
_STEP_RULES: tuple[tuple[str, str, str], ...] = (
    ("blank:",            "substrate", "Substrate"),
    ("deposit conformal", "ald",       "ALD / conformal"),
    ("spincoat",          "litho",     "Resist coat"),
    ("deposit photoresist", "litho",   "Resist coat"),
    ("expose",            "litho",     "Expose"),
    ("develop",           "litho",     "Develop"),
    ("etch_back",         "etch",      "Spacer etch-back"),
    ("lateral front",     "etch",      "Lateral etch"),
    ("etch",              "etch",      "Etch"),
    # "CMP" only. A lowercase "planarize" rule looks like belt-and-braces and
    # is a trap: "planarize" is a substring of "(planarized)", which every
    # planarised *deposition* carries — so a flowable fill got filed as a
    # polish. `Stack.planarize` always writes "CMP ...", so this covers it.
    ("CMP",               "cmp",       "CMP"),
    ("strip",             "strip",     "Strip"),
    ("deposit",           "deposit",   "Deposition"),
)


def classify_step(entry: str) -> tuple[str, str, str]:
    """Sort one history line into a process family.

    Returns ``(family, verb, detail)`` — the family key for colour, the
    process name, and whatever the engine recorded after it (materials,
    thicknesses, doses), kept verbatim because the engine owns that wording.
    """
    for needle, family, verb in _STEP_RULES:
        if needle in entry:
            detail = entry
            # Trim the engine's own leading verb; the row already names it.
            for lead in ("deposit conformal ", "deposit ", "etch_back ",
                         "etch ", "strip ", "blank: ", "expose ", "develop ",
                         "spincoat "):
                if detail.startswith(lead):
                    detail = detail[len(lead):]
                    break
            return family, verb, detail
    return "deposit", "Step", entry


def process_flow_panel(
    stack: Stack,
    ax=None,
    max_rows: int = 46,
    fontsize: float | None = None,
):
    """Draw the wafer's process history as a flow, first step at the top.

    This is the *recipe* that built the wafer, not the films it ended up with:
    every deposition, conformal/ALD step, etch, CMP, strip and litho level, in
    order, as :attr:`Stack.history` recorded them at the time. A cross-section
    shows what the wafer is; this shows what was done to it.

    Every entry is engine-written — no step is inferred from the final
    geometry, so a flow that did nothing still appears, which is the point.

    Parameters
    ----------
    stack : Stack
        The wafer whose history to draw.
    ax : Axes, optional
        Drawn into if given.
    max_rows : int
        Above this, consecutive identical operations collapse to one row with
        a ``x N`` multiplier — a superlattice is six deposits that say the same
        two things three times.
    fontsize : float, optional
        Defaults to a size chosen from the row count.

    Returns
    -------
    (ax, n_rows) : the axes and how many rows were drawn.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 7.0))

    # Display-only bookkeeping is not process. `crop` records that a viewer
    # sliced the array to frame a picture; putting it in a travellers list
    # beside real depositions and etches would be a lie about what was run.
    steps = [h for h in stack.history if "(display)" not in h]

    rows: list[tuple[str, str, str, int]] = []
    for entry in steps:
        family, verb, detail = classify_step(entry)
        if rows and rows[-1][0] == family and rows[-1][2] == detail:
            f, v, d, n = rows[-1]
            rows[-1] = (f, v, d, n + 1)
        else:
            rows.append((family, verb, detail, 1))

    if len(rows) > max_rows:
        # Collapse runs of the same *family* once the list outgrows the panel,
        # rather than truncating: a flow that silently loses its tail is worse
        # than one that says "8 depositions here".
        packed: list[tuple[str, str, str, int]] = []
        for r in rows:
            if packed and packed[-1][0] == r[0]:
                f, v, d, n = packed[-1]
                packed[-1] = (f, v, f"{n + r[3]} steps", n + r[3])
            else:
                packed.append(r)
        rows = packed

    n = len(rows)
    fs = fontsize if fontsize else float(np.clip(190.0 / max(n, 1), 5.0, 8.5))

    ax.set(xlim=(0, 1), ylim=(n, 0), xticks=[], yticks=[])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    for i, (family, verb, detail, count) in enumerate(rows):
        colour = PROCESS_FAMILIES[family]
        ax.add_patch(Rectangle((0.02, i + 0.12), 0.055, 0.76,
                               facecolor=colour, edgecolor="none"))
        times = f"  ×{count}" if count > 1 else ""
        # Two rows inside the band: the process name, then what it acted on.
        # One string with a newline centres the pair and lets long details
        # push the name off its own row.
        ax.text(0.10, i + 0.34, f"{verb}{times}", fontsize=fs, va="center",
                ha="left", color=_theme.INK, weight="bold")
        ax.text(0.10, i + 0.72, detail, fontsize=fs * 0.88, va="center",
                ha="left", color=_theme.INK2)
    _theme.title(ax, "Process flow", caption_text=f"{len(steps)} steps")
    return ax, n


def stack_figure_mpl(
    stack: Stack,
    materials: Sequence[str] | None = None,
    z_exaggeration: float = 1.0,
    downsample: int = 1,
    cutaway: float | None = None,
    ax=None,
    elev: float = 26.0,
    azim: float = -58.0,
):
    """Static 3-D rendering of the stack as **solids**.

    Draws the real voxel surface via :func:`voxel_mesh`, so a printed line
    appears as a solid with sidewalls and a visible profile. The earlier
    version plotted only the top height field, which rendered as flat sheets
    with holes — all of the topology was in the data and none of it on screen.

    Parameters
    ----------
    stack : Stack
        The wafer stack.
    materials : sequence of str, optional
        Restrict to these materials.
    z_exaggeration : float
        Vertical scale factor. 100 nm of resist over a 500 nm field is a
        pancake; 2–4 makes the profile legible.
    downsample : int
        Voxel decimation before meshing. Raise it if rendering is slow.
    cutaway : float, optional
        Keep this fraction of the y range, slicing the stack open.
    ax : Axes3D, optional
        Draw into an existing 3-D axis.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(9, 6.5))
        ax = fig.add_subplot(111, projection="3d")

    present = _selected_materials(stack, materials)

    ny, nx = stack.shape_xy
    y_limit = None if cutaway is None else ny * stack.pixel_size * 1e9 * cutaway
    n_tri = 0

    for m, V, F in _scaled_meshes(stack, present, downsample,
                                  z_exaggeration, y_limit):
        n_tri += len(F)
        # `facecolors` (plural) is required for shade=True — the singular form
        # silently disables shading, which flattens the solid into a
        # silhouette. And edgecolors must be left unset: passing "none" makes
        # it an empty array that fails to broadcast inside the shader.
        poly = Poly3DCollection(V[F], facecolors=m.color, shade=True)
        poly.set_alpha(m.opacity)
        poly.set_linewidth(0)
        ax.add_collection3d(poly)

    extent = nx * stack.pixel_size * 1e9
    height = stack.nz * stack.dz * 1e9 * z_exaggeration
    _dress_3d_axes(ax, extent, y_limit or ny * stack.pixel_size * 1e9, height,
                   z_exaggeration, elev, azim)
    ax.set_title(f"Printed stack — {n_tri:,} triangles")
    logger.info("stack_figure_mpl: %d triangles", n_tri)
    return ax


def crop(
    stack: Stack,
    y: tuple[float, float] | None = None,
    x: tuple[float, float] | None = None,
    z_substrate: float | None = None,
    z_headroom: float = 4e-9,
) -> Stack:
    """Cut the stack down to the part worth drawing, as a real array crop.

    Two jobs, both of which have to happen *before* meshing rather than after.

    **Sectioning.** ``stack_figure_mpl(cutaway=…)`` discards triangles behind a
    plane, which opens the solid: the cut face itself was never a boundary
    between two materials, so no face was ever emitted there and the result
    renders as a hollow shell. Cropping the voxel array instead makes the cut
    plane the edge of the array, and :func:`voxel_mesh` pads with ``False`` —
    so the cut face comes out capped and you get a true cross-section.

    **Framing.** A process stack carries whatever headroom the deposits needed
    plus a substrate slab. Left in, the renderer scales z to the full voxel
    height and the device becomes a sliver at the bottom of the frame.

    Parameters
    ----------
    stack : Stack
        The stack to crop.
    y, x : (float, float), optional
        Keep this fraction of the axis, as ``(lo, hi)`` in [0, 1].
    z_substrate : float, optional
        Keep this much substrate [m] below the lowest voxel that is neither
        vacuum nor the substrate material — i.e. below where the wafer stops
        being bulk. ``None`` keeps the stack's full depth.
    z_headroom : float
        Vacuum kept above the highest solid voxel [m].

    Returns
    -------
    Stack
        A cropped copy. The original is untouched.
    """
    mat = stack.mat
    z0, z1 = 0, mat.shape[0]

    if z_substrate is not None:
        substrate_id = get_material("Si").id
        solid = np.flatnonzero((mat != VACUUM).any(axis=(1, 2)))
        built = np.flatnonzero(
            ((mat != VACUUM) & (mat != substrate_id)).any(axis=(1, 2))
        )
        if solid.size:
            n_sub = max(int(round(z_substrate / stack.dz)), 1)
            n_head = max(int(round(z_headroom / stack.dz)), 1)
            z0 = max(0, int(built.min()) - n_sub) if built.size else 0
            z1 = min(mat.shape[0], int(solid.max()) + 1 + n_head)

    def _span(frac, n):
        if frac is None:
            return 0, n
        lo, hi = frac
        a, b = int(round(lo * n)), int(round(hi * n))
        return max(0, min(a, n - 1)), max(1, min(b, n))

    y0, y1 = _span(y, mat.shape[1])
    x0, x1 = _span(x, mat.shape[2])

    out = stack.copy()
    out.mat = np.ascontiguousarray(mat[z0:z1, y0:y1, x0:x1])
    out.history.append(
        f"crop z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}] (display)"
    )
    return out


#: Scale-bar lengths worth printing, in nm. A bar reads as a measurement, so it
#: gets a round number; the picker takes the largest that fits the target span.
_NICE_NM = (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)


def nice_length(span_nm: float, target: float = 0.38) -> float:
    """Largest round length that is at most *target* of the span."""
    want = span_nm * target
    fits = [v for v in _NICE_NM if v <= want]
    return float(fits[-1]) if fits else float(_NICE_NM[0])


_nice_length = nice_length


def _draw_frame(ax, x1: float, y1: float, z1: float, style: str,
                colour: str = "#b8bcc4", lw: float = 0.7) -> None:
    """Hairline framing in data coordinates, drawn by us rather than by mpl.

    Matplotlib's own 3-D chrome is filled panes plus a grid plus ticks on three
    axes, which for a solid object competes with the object. These are the two
    restrained alternatives: the full twelve-edge box (Mathematica's
    ``Boxed -> True, Axes -> False``) and a corner frame that keeps only the
    floor and the two back verticals as a depth cue.
    """
    if style == "none":
        return
    seg = []
    floor = [((0, 0, 0), (x1, 0, 0)), ((x1, 0, 0), (x1, y1, 0)),
             ((x1, y1, 0), (0, y1, 0)), ((0, y1, 0), (0, 0, 0))]
    if style == "corner":
        # Floor, plus the two verticals on the far corner only.
        seg = floor + [((0, y1, 0), (0, y1, z1)), ((x1, y1, 0), (x1, y1, z1))]
    elif style == "box":
        top = [((0, 0, z1), (x1, 0, z1)), ((x1, 0, z1), (x1, y1, z1)),
               ((x1, y1, z1), (0, y1, z1)), ((0, y1, z1), (0, 0, z1))]
        posts = [((0, 0, 0), (0, 0, z1)), ((x1, 0, 0), (x1, 0, z1)),
                 ((x1, y1, 0), (x1, y1, z1)), ((0, y1, 0), (0, y1, z1))]
        seg = floor + top + posts
    for (xa, ya, za), (xb, yb, zb) in seg:
        ax.plot([xa, xb], [ya, yb], [za, zb], color=colour, linewidth=lw,
                zorder=0)


def _draw_scale_bar(ax, length_nm: float, x1: float, y1: float,
                    colour: str = "#33373d") -> None:
    """A bar along x at the near-bottom edge, with its length written under it.

    Replaces the tick sprawl: one measurement, stated once, instead of numbers
    on three axes. It is only honest under an orthographic projection, where a
    length on screen does not depend on depth — which is why this function is
    paired with ``projection="ortho"``.
    """
    pad = 0.04 * x1
    x0 = pad
    y = -0.06 * y1          # just outside the near face
    z = 0.0
    ax.plot([x0, x0 + length_nm], [y, y], [z, z], color=colour, linewidth=2.4,
            solid_capstyle="butt", zorder=6)
    ax.text(x0 + length_nm / 2, y, -0.055 * max(x1, y1),
            f"{length_nm:g} nm", color=colour, fontsize=9.5,
            ha="center", va="top", zorder=6)


def device_figure_mpl(
    stack: Stack,
    materials: Sequence[str] | None = None,
    z_exaggeration: float = 1.0,
    downsample: int = 1,
    ax=None,
    elev: float = 22.0,
    azim: float = -62.0,
    opaque: bool = True,
    shade: bool = True,
    merge: bool = True,
    chrome: str = "axes",
    scale_bar: bool = False,
    scale_bar_nm: float | None = None,
    projection: str = "persp",
    zoom: float = 1.0,
):
    """Static 3-D rendering of a **multi-material** stack as one solid.

    :func:`stack_figure_mpl` adds one
    :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection` per material, and
    Matplotlib depth-sorts each collection independently. With one or two films
    that is invisible; with the five or six a finished device carries it is not.
    Collections get drawn in whole-collection order, so a gate buried under a
    spacer can paint *over* the spacer, and a film with ``opacity < 1`` — the
    oxides, in the built-in library — veils everything behind it. The result
    reads as coloured slabs rather than as a device.

    This draws every material into a **single** collection with per-face
    colours, so all faces sort against each other, and forces opacity so the
    painter's algorithm has a well-defined answer. Pair it with :func:`crop`
    to cap a cross-section.

    Parameters
    ----------
    stack : Stack
        The wafer stack.
    materials : sequence of str, optional
        Restrict to these materials. Defaults to everything present.
    z_exaggeration : float
        Vertical scale factor.
    downsample : int
        Voxel decimation before meshing.
    ax : Axes3D, optional
        Draw into an existing 3-D axis.
    elev, azim : float
        Camera angles.
    opaque : bool
        Force alpha to 1. Turning it off restores each material's library
        opacity and reintroduces the veiling described above.
    shade : bool
        Light the faces from their normals.
    merge : bool
        Merge coplanar faces via :func:`greedy_mesh` instead of emitting one
        quad per voxel face. Two orders of magnitude fewer triangles for the
        same silhouette, and Matplotlib's cost is per triangle. Turn it off
        only to compare against the unmerged mesh.
    chrome : {"axes", "box", "corner", "none"}
        How much framing to draw. ``"axes"`` keeps Matplotlib's own 3-D
        decoration — filled panes, grid, ticks and labels on all three axes.
        The other three switch it off entirely and draw a hairline frame
        instead: the full twelve-edge ``"box"``, a ``"corner"`` frame (floor
        plus the two far verticals), or ``"none"``. For a solid object the
        default decoration competes with the geometry, which is the whole
        reason the alternatives exist.
    scale_bar : bool
        Draw a labelled bar along x instead of relying on tick numbers. Pair it
        with ``projection="ortho"``; under perspective a screen length depends
        on depth and the bar would be lying.
    scale_bar_nm : float, optional
        Bar length [nm]. Defaults to the largest round value under ~30 % of the
        x extent.
    projection : {"persp", "ortho"}
        Camera model. ``"ortho"`` removes perspective convergence, which reads
        as a technical drawing and makes the scale bar valid everywhere in the
        frame.
    zoom : float
        Enlarge the object within the axes box. Matplotlib leaves generous
        margins around a 3-D artist; 1.1–1.3 reclaims them.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(10, 7.5))
        ax = fig.add_subplot(111, projection="3d")

    present = _selected_materials(stack, materials)

    tris: list[NDArray] = []
    colours: list[NDArray] = []
    extract = greedy_mesh if merge else voxel_mesh
    for m, V, F in _scaled_meshes(stack, present, downsample,
                                  z_exaggeration, extract=extract):
        tris.append(V[F])
        rgba = to_rgba(m.color, 1.0 if opaque else m.opacity)
        colours.append(np.tile(np.asarray(rgba), (len(F), 1)))

    n_tri = 0
    if tris:
        T = np.concatenate(tris, axis=0)
        C = np.concatenate(colours, axis=0)
        n_tri = len(T)
        poly = Poly3DCollection(T, facecolors=C, shade=shade)
        poly.set_linewidth(0)
        # Two triangles meet along every voxel-face diagonal, exactly coplanar,
        # and antialiasing blends each of those seams a little way toward the
        # background — across a wall of thousands it reads as a woven hatch
        # rather than as a surface. Switching AA off closes them.
        #
        # Do NOT try to close them by setting `edgecolor` from `get_facecolor()`
        # instead: Poly3DCollection re-sorts its polygons by depth on every
        # draw, and the edge-colour array is not permuted to match, so each
        # triangle ends up outlined in some other material's colour and the
        # whole solid comes back as a rainbow wireframe.
        poly.set_antialiased(False)
        ax.add_collection3d(poly)

    ny, nx = stack.shape_xy
    extent_x = nx * stack.pixel_size * 1e9
    extent_y = ny * stack.pixel_size * 1e9
    height = stack.nz * stack.dz * 1e9 * z_exaggeration
    _dress_3d_axes(ax, extent_x, extent_y, height, z_exaggeration,
                   elev, azim, zoom=zoom)

    if projection in ("ortho", "persp"):
        ax.set_proj_type(projection)

    if chrome != "axes":
        # Off wholesale, then framed by hand. Turning the pieces off one at a
        # time (panes, grid, ticks, spines, labels) leaves stragglers that
        # differ between Matplotlib versions; `set_axis_off` is the one switch
        # that is stable, and it leaves other artists — legend, our frame, the
        # scale bar — alone.
        ax.set_axis_off()
        _draw_frame(ax, extent_x, extent_y, height, chrome)

    if scale_bar:
        length = scale_bar_nm or nice_length(extent_x)
        _draw_scale_bar(ax, length, extent_x, extent_y)

    logger.info("device_figure_mpl: %d materials, %d triangles, chrome=%s",
                len(present), n_tri, chrome)
    return ax


def profile_figure(
    remaining: NDArray[np.bool_],
    grid: GridConfig,
    row: int | None = None,
    ax=None,
):
    """Plot a developed 3-D resist profile as an x–z cross-section.

    Works directly on the boolean volume from
    :func:`~litho_sim.develop.resist3d.print_resist_3d`, before it is written into a
    :class:`~litho_sim.wafer.stack.Stack`.
    """
    import matplotlib.pyplot as plt

    nz, ny, nx = remaining.shape
    r = ny // 2 if row is None else row
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    ax.imshow(
        remaining[:, r, :], origin="lower", cmap="Greens", aspect="auto",
        extent=(0, nx * grid.pixel_size * 1e9, 0, nz * grid.dz * 1e9),
        interpolation="nearest",
    )
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("z [nm]")
    ax.set_title("Developed resist profile")
    return ax


def _darken(colour: str, factor: float) -> tuple:
    """Same hue, less light — for the shaded faces of an unlit slab."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(colour)
    return (r * factor, g * factor, b * factor)


def resist_surface_mpl(
    remaining: NDArray[np.bool_],
    grid: GridConfig,
    ax=None,
    z_exaggeration: float = 1.0,
    stride: int = 2,
    cmap: str | None = None,
    substrate: bool = True,
    elev: float = 26.0,
    azim: float = -58.0,
):
    """Draw a developed resist profile as a height field.

    The fast path, and for this engine an **exact** one. ``develop_3d`` has no
    lateral component — a column is cleared from the top down — so the
    remaining resist in each column is a single contiguous run and one height
    per column describes it completely. Verified across both develop models
    with and without standing waves: no column ever has two runs.

    That matters because the honest alternative, :func:`stack_figure_mpl`,
    emits one quad per exposed voxel face and hands matplotlib tens of
    thousands of triangles to depth-sort in Python on every single draw.
    Measured at 128 px: 1046 ms per draw against **110 ms** here — an order of
    magnitude, for the same picture.

    The voxel path stays worth having for wafer stacks, where freestanding
    spacers genuinely need two disjoint intervals in one column (see the
    vault's Engine note), and would be needed here too if lateral development
    ever lands.

    Parameters
    ----------
    remaining : NDArray[bool]
        ``(nz, ny, nx)`` developed volume, ``iz = 0`` at the bottom.
    grid : GridConfig
        Supplies ``pixel_size`` and ``dz``.
    ax : Axes3D, optional
        Drawn into if given; otherwise one is created.
    z_exaggeration : float
        Stretch the depth axis for legibility.
    stride : int
        Sample every *stride*-th point. 2 is a good default; 1 is exact and
        about 3× dearer to draw.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from litho_sim.wafer import get_material

    remaining = np.asarray(remaining)
    nz, ny, nx = remaining.shape
    s = max(int(stride), 1)

    # Highest occupied voxel per column, as a height. argmax over the reversed
    # axis finds the first solid from the top; columns with no resist get 0.
    any_solid = remaining.any(axis=0)
    top_from_end = np.argmax(remaining[::-1], axis=0)
    top_index = nz - 1 - top_from_end
    height_nm = np.where(any_solid, (top_index + 1) * grid.dz * 1e9, 0.0)[::s, ::s]

    x_nm = (np.arange(nx) * grid.pixel_size * 1e9)[::s]
    y_nm = (np.arange(ny) * grid.pixel_size * 1e9)[::s]

    # Pad with a ring at z = 0 laid on *duplicated* edge coordinates. The quad
    # between the rim and its pad therefore has zero width and full height —
    # a vertical wall — so the block's sides fall out of the same mesh. Doing
    # it this way rather than as a second collection matters: matplotlib
    # depth-sorts each collection independently, so a separate skirt would
    # punch through the surface it belongs to from some angles.
    xp = np.concatenate(([x_nm[0]], x_nm, [x_nm[-1]]))
    yp = np.concatenate(([y_nm[0]], y_nm, [y_nm[-1]]))
    hp = np.pad(height_nm, 1)
    X, Y = np.meshgrid(xp, yp)

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    top = max(float(hp.max()), 1e-9)
    sub_h = max(0.12 * top, 4.0) if substrate else 0.0

    if substrate:
        # A slab to stand on. Without it the film reads as a sheet hanging in
        # space rather than a layer on a wafer — and the eye has nothing to
        # judge the profile's depth against.
        c_sub = get_material("Si").color
        gx, gy = np.array([x_nm[0], x_nm[-1]]), np.array([y_nm[0], y_nm[-1]])
        GX, GY = np.meshgrid(gx, gy)
        ax.plot_surface(GX, GY, np.zeros_like(GX), color=c_sub,
                        shade=False, linewidth=0)
        ax.plot_surface(GX, GY, np.full_like(GX, -sub_h), color=_darken(c_sub, 0.7),
                        shade=False, linewidth=0)
        wall = np.array([[0.0, 0.0], [-sub_h, -sub_h]])
        for ex, ey in (((x_nm[0], x_nm[0]), (y_nm[0], y_nm[-1])),
                       ((x_nm[-1], x_nm[-1]), (y_nm[0], y_nm[-1])),
                       ((x_nm[0], x_nm[-1]), (y_nm[0], y_nm[0])),
                       ((x_nm[0], x_nm[-1]), (y_nm[-1], y_nm[-1]))):
            WX, WY = np.meshgrid(np.array(ex), np.array(ey))
            ax.plot_surface(WX, WY, wall, color=_darken(c_sub, 0.85),
                            shade=False, linewidth=0)

    kw = {}
    if cmap is not None:
        kw["cmap"] = cmap
    else:
        c_resist = get_material("photoresist").color
        # One colour per quad rather than one per surface, so a trench the
        # developer took all the way down reads as exposed substrate instead
        # of as a resist-coloured floor. Costs about 1 ms.
        quad = np.maximum.reduce([hp[:-1, :-1], hp[1:, :-1], hp[:-1, 1:], hp[1:, 1:]])
        face = np.empty(quad.shape + (4,))
        face[...] = to_rgba(c_resist)
        if substrate:
            face[quad <= 0.0] = to_rgba(get_material("Si").color)
        kw["facecolors"] = face

    ax.plot_surface(
        X, Y, hp, rstride=1, cstride=1,
        linewidth=0, antialiased=True, shade=True, **kw,
    )
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    ax.set_zlabel("z [nm]" + (f" ×{z_exaggeration:g}" if z_exaggeration != 1 else ""))
    ax.set_zlim(-sub_h, top)
    ax.set_box_aspect((1.0, 1.0, 0.55 * z_exaggeration))
    ax.view_init(elev=elev, azim=azim)
    logger.debug("resist_surface_mpl: %d x %d height field, stride %d", ny, nx, s)
    return ax


def resist_stack_for_display(
    remaining: NDArray[np.bool_],
    grid: GridConfig,
    substrate: str = "Si",
    substrate_thickness: float = 30e-9,
) -> Stack:
    """Wrap a developed resist volume in a minimal stack for 3-D rendering.

    :func:`stack_figure_mpl` and friends draw :class:`~litho_sim.wafer.stack.Stack`
    objects, but a profile straight out of
    :func:`~litho_sim.develop.resist3d.print_resist_3d` is just a boolean volume.
    This puts that volume on a thin substrate slab so the mesh extractors can
    render it as a solid, without running any process flow.

    Parameters
    ----------
    remaining : NDArray[np.bool_]
        ``(nz, ny, nx)`` developed volume, True where resist remains.
    grid : GridConfig
        Supplies the lateral grid and ``dz``.
    substrate : str
        Material drawn under the resist, purely for visual grounding.
    substrate_thickness : float
        Substrate slab thickness [m].

    Returns
    -------
    Stack
        A stack holding only the substrate and the developed resist.
    """
    from litho_sim.wafer import get_material

    nz = int(remaining.shape[0])
    stack = Stack.blank(
        grid, dz=grid.dz, substrate=substrate,
        substrate_thickness=substrate_thickness,
        headroom=(nz + 4) * grid.dz,
    )
    n_sub = max(int(round(substrate_thickness / grid.dz)), 1)
    rid = get_material("photoresist").id
    stack.mat[n_sub:n_sub + nz][remaining] = rid
    stack.materials.setdefault(rid, get_material("photoresist"))
    stack.history.append("resist profile (display)")
    return stack


def resist_profile_3d_figure(
    result: dict,
    grid: GridConfig,
    row: int | None = None,
    z_exaggeration: float = 2.0,
    cutaway: float = 0.55,
    title: str = "3-D resist profile",
):
    """Render everything :func:`~litho_sim.develop.resist3d.print_resist_3d` produced.

    One figure, three panels: the developed resist as a rotatable solid on a
    substrate slab, the x-z cross-section of the profile, and the latent
    image the develop step consumed, with the profile outline over it.  The
    profile metrics that matter — sidewall angle, film retention, resist top
    loss — are annotated on the figure, and the two degenerate outcomes
    (film sealed by standing-wave nodes, film fully cleared) are called out
    with the likely remedy instead of presenting an empty axis.

    Parameters
    ----------
    result : dict
        Output of :func:`~litho_sim.develop.resist3d.print_resist_3d` (needs
        ``remaining`` and ``latent``).
    grid : GridConfig
        Simulation grid.
    row : int, optional
        y-row for the cross-sections; defaults to the centre.
    z_exaggeration : float
        Vertical scale for the 3-D panel.
    cutaway : float
        Fraction of y kept in the 3-D panel, slicing the solid open so the
        cross-section faces the camera.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from litho_sim.develop.profile import (
        film_remaining,
        sidewall_angle,
        top_loss,
    )

    remaining = result["remaining"]
    latent = result.get("latent")
    nz, ny, nx = remaining.shape
    r = ny // 2 if row is None else row

    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.42, wspace=0.18)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_prof = fig.add_subplot(gs[0, 1])
    ax_lat = fig.add_subplot(gs[1, 1])

    # --- 3-D solid ---------------------------------------------------------
    display = resist_stack_for_display(remaining, grid)
    stack_figure_mpl(
        display, z_exaggeration=z_exaggeration, cutaway=cutaway, ax=ax3d,
    )
    ax3d.set_title("Developed resist (solid)")

    # --- profile cross-section --------------------------------------------
    profile_figure(remaining, grid, row=r, ax=ax_prof)

    frac = film_remaining(remaining)
    angle = sidewall_angle(remaining, grid, row=r)
    top_loss_nm = top_loss(remaining, grid)
    lines = [f"remaining {100 * frac:5.1f}%"]
    if np.isfinite(angle):
        lines.append(f"sidewall  {angle:5.1f}°")
    if np.isfinite(top_loss_nm):
        lines.append(f"top loss  {top_loss_nm:5.1f} nm")
    ax_prof.text(
        0.02, 0.96, "\n".join(lines), transform=ax_prof.transAxes,
        va="top", ha="left", fontsize=8.5, family="monospace",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )

    # Degenerate outcomes deserve a diagnosis, not an empty picture.
    if frac > 0.995:
        ax_prof.text(
            0.5, 0.5,
            "Film did not develop.\nStanding-wave nodes can seal the top under\n"
            "the threshold model — raise PEB diffusion, add a\n"
            "BARC (substrate_reflectance=0), or use the finite-\n"
            "rate develop_model='mack'.",
            transform=ax_prof.transAxes, ha="center", va="center",
            fontsize=9, color="#B00020",
        )
    elif frac < 0.005:
        ax_prof.text(
            0.5, 0.5,
            "Film fully cleared.\nDose too high for this contrast, or the Mack\n"
            "front outran develop_time — reduce dose or\n"
            "develop_time, or check image contrast at this pitch.",
            transform=ax_prof.transAxes, ha="center", va="center",
            fontsize=9, color="#B00020",
        )

    # --- latent image ------------------------------------------------------
    if latent is not None:
        extent = (0, nx * grid.pixel_size * 1e9, 0, nz * grid.dz * 1e9)
        im = ax_lat.imshow(
            latent[:, r, :], origin="lower", cmap="viridis", aspect="auto",
            extent=extent, interpolation="nearest",
        )
        fig.colorbar(im, ax=ax_lat, label="PAC", shrink=0.9)
        # The developed outline over the latent image shows exactly where the
        # develop criterion cut.
        ax_lat.contour(
            remaining[:, r, :].astype(float), levels=[0.5], colors="white",
            linewidths=1.2, extent=extent, origin="lower",
        )
        ax_lat.set_xlabel("x [nm]")
        ax_lat.set_ylabel("z [nm]")
        ax_lat.set_title("Latent image (PAC) with developed outline")
    else:  # pragma: no cover - latent is always present from print_resist_3d
        ax_lat.set_axis_off()

    fig.suptitle(title, fontsize=13)
    return fig
