"""
The geometry pass behind the tilt-SEM view: a surface seen from an angle.

A tilt-stage micrograph is a picture of a *surface* — the developed resist
on its substrate, cleaved and looked at from 35 or 45 degrees — and the SE
physics that turns it into a micrograph (:func:`litho_sim.metrology.sem.tilt_signal`)
needs three things per pixel: which way the surface faces, how far along
the beam it is, and what it is made of. Those are exactly what a renderer
with a depth buffer produces, so this module asks VTK for them and hands
them back as arrays (:class:`~litho_sim.metrology.sem.GeometryBuffers`).
No lighting, no shading, no colours: the render is a *measurement* of the
geometry, and every grey in the final picture comes from the physics.

Two passes, one scene. The first writes the surface normal into the
colour buffer and reads the depth buffer; the second flat-fills each
actor with its material. Both are lighting-off, so what comes back is the
data, not a picture of it.

The projection is parallel. A SEM's scan angles at these magnifications
are a fraction of a degree, so its images have no perspective — a length
on the sample is the same number of pixels wherever it sits, which is
what makes a scale bar honest and is why the earlier perspective renders
read as photographs rather than micrographs.

The surface itself
------------------
:func:`profile_surface` builds a closed iso-surface of the resist. From a
develop *arrival-time field* the surface is the level set at the develop
time, with sub-voxel accuracy — the surface the stochastic trials carry.
From a boolean volume it is the 0.5 level of a lightly smoothed
occupancy, which rounds the voxel staircase without moving the walls by
more than a fraction of a voxel. Either way the field is padded with a
layer of "developed" so the surface closes on the field boundary and the
substrate, and the cleaved face at the front of the field is a face, not
a hole.

``pyvista`` is the ``viz3d`` extra; :func:`available` says whether it
imported, and every caller falls back to the matplotlib figure when it
did not.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter

from litho_sim.metrology.sem import GeometryBuffers, SEMConfig, SEMImage, tilt_sem

logger = logging.getLogger(__name__)

__all__ = [
    "Stage", "available", "profile_surface", "render_buffers",
    "tilt_sem_of_profile", "camera_basis",
]

#: How far the clip of an arrival-time field reaches above the develop
#: time, in develop times. Only the crossing matters to the surface; the
#: clip keeps the interior from dominating the interpolation.
_FIELD_CLIP = 50.0

#: Renders larger than this on a side are capped, and ``pixel_nm`` grows to
#: fit. 4 k on a side is 16 MB of float normals, which is plenty.
_MAX_SIDE = 4096


def available() -> bool:
    """Whether the geometry pass can run: pyvista (and VTK) import."""
    return all(importlib.util.find_spec(m) is not None
               for m in ("pyvista", "vtkmodules"))


@dataclass(frozen=True)
class Stage:
    """Where the sample sits under the column.

    Parameters
    ----------
    tilt : float
        Stage tilt in degrees — the angle between the beam and the wafer
        normal. 0 looks straight down (a top-down view with the cleaved
        face invisible), 90 looks along the wafer. Cross-section work is
        done between 30 and 60.
    azimuth : float
        Rotation of the stage about the wafer normal, degrees. 0 looks
        straight at the cleaved face along the lines; a few degrees either
        way shows the near sidewall of every line and is how such images
        are usually taken.
    pixel_nm : float, optional
        The scan pitch, nanometres per pixel. Sets the render size for a
        given field; the beam's blur is then a physical width, not a pixel
        count. ``None`` picks the pitch that makes the frame ``frame_px``
        across — the magnification a tool would choose for the field —
        but never finer than 0.2 nm.
    frame_px : int
        The frame's longer side when ``pixel_nm`` is ``None``.
    substrate_nm : float
        How much substrate to show under the film on the cleaved face.
    margin_px : int
        Vacuum around the sample in the frame.
    """

    tilt: float = 40.0
    azimuth: float = 20.0
    pixel_nm: float | None = None
    frame_px: int = 1000
    substrate_nm: float = 70.0
    margin_px: int = 24

    def __post_init__(self) -> None:
        if not 0.0 <= self.tilt < 90.0:
            raise ValueError("tilt must be in [0, 90) degrees")
        if self.pixel_nm is not None and self.pixel_nm <= 0:
            raise ValueError("pixel_nm must be > 0")
        if self.frame_px < 64:
            raise ValueError("frame_px must be >= 64")
        if self.substrate_nm < 0:
            raise ValueError("substrate_nm must be >= 0")


def camera_basis(tilt: float, azimuth: float) -> tuple[NDArray, NDArray, NDArray]:
    """``(view, up, right)`` unit vectors, world axes, for a stage setting.

    The wafer is the x–y plane with z up; the cleaved face is the plane
    ``y = 0`` and the lines run along +y. The column looks down at the
    stage from in front of the cleave, tilted by *tilt* from the normal
    and swung by *azimuth* about it. ``up`` is world-z projected into the
    image plane, so "up the image" is always "away from the viewer along
    the stage"; ``right`` completes a right-handed image frame.
    """
    t, a = np.radians(tilt), np.radians(azimuth)
    towards_camera = np.array([np.sin(t) * np.sin(a), -np.sin(t) * np.cos(a), np.cos(t)])
    view = -towards_camera / np.linalg.norm(towards_camera)
    z = np.array([0.0, 0.0, 1.0])
    up = z - (z @ view) * view
    if np.linalg.norm(up) < 1e-9:              # looking straight down
        up = np.array([0.0, 1.0, 0.0])
    up /= np.linalg.norm(up)
    right = np.cross(view, up)
    right /= np.linalg.norm(right)
    return view, up, right


def _image_data(field: NDArray[np.float64], spacing_nm: tuple[float, float, float],
                outside: float):
    """A pyvista ImageData whose point data is *field* (nz, ny, nx), padded
    with one layer of *outside* — a value on the developed side of the
    level — so the iso-surface closes."""
    import pyvista as pv

    dz, py, px = spacing_nm
    padded = np.pad(np.asarray(field, dtype=np.float64), 1, mode="constant",
                    constant_values=float(outside))
    nz, ny, nx = padded.shape
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(px, py, dz),
                        origin=(-0.5 * px, -0.5 * py, -0.5 * dz))
    grid.point_data["f"] = padded.transpose(2, 1, 0).ravel(order="F")
    return grid


def profile_surface(
    remaining: NDArray[np.bool_] | None = None,
    *,
    field: NDArray[np.float64] | None = None,
    level: float | None = None,
    feature: str = "above",
    spacing_nm: tuple[float, float, float],
    smooth_iterations: int = 30,
    smooth_sigma_vox: float = 0.75,
):
    """A closed iso-surface of the resist, as a pyvista PolyData.

    Parameters
    ----------
    remaining : (nz, ny, nx) bool, optional
        The developed solid. Its occupancy is smoothed by
        ``smooth_sigma_vox`` voxels and contoured at 0.5. A wall that is
        nearly vertical steps sideways one voxel every so often in a
        boolean volume, and the smoothing cannot hide a 4 nm terrace; use
        the field whenever there is one.
    field, level, feature : optional
        Instead of the boolean: the continuous field the solid is a level
        set of — a develop arrival time against the develop time, or the
        latent image against its threshold — with resist on the *feature*
        side (``"above"`` or ``"below"``) of *level*, as
        :func:`~litho_sim.develop.resist3d.develop_surface` reports them.
        Sub-voxel, and what every profile out of the engine now carries.
    spacing_nm : (dz, dy, dx)
        Voxel size, nanometres.
    smooth_iterations : int
        Taubin passes on the mesh — a low-pass that does not shrink it.

    Returns
    -------
    pyvista.PolyData
        With point normals, outward.
    """
    if (remaining is None) == (field is None):
        raise ValueError("pass exactly one of remaining or field")
    if field is not None:
        if level is None:
            raise ValueError("a field needs its level")
        if feature not in ("above", "below"):
            raise ValueError(f"feature must be 'above' or 'below', got {feature!r}")
        iso = float(level)
        f = np.asarray(field, dtype=np.float64)
        if feature == "below":
            f = 2.0 * iso - f                      # reflect: resist is now above
        # Only the crossing matters; clip the far interior (an arrival time
        # can be infinite) so it cannot dominate the interpolation.
        span = _FIELD_CLIP * max(abs(iso), 1e-12)
        f = np.clip(np.nan_to_num(f, nan=iso - span, posinf=iso + span, neginf=iso - span),
                    iso - span, iso + span)
        outside = iso - span
    else:
        occ = np.asarray(remaining, dtype=np.float64)
        f = gaussian_filter(occ, smooth_sigma_vox, mode="nearest") if smooth_sigma_vox > 0 else occ
        iso = 0.5
        outside = 0.0
    if f.ndim != 3:
        raise ValueError(f"expected a (nz, ny, nx) volume, got shape {f.shape}")
    grid = _image_data(f, spacing_nm, outside)
    surf = grid.contour([iso], scalars="f")
    if surf.n_points == 0:
        return surf
    if smooth_iterations > 0:
        surf = surf.smooth_taubin(n_iter=int(smooth_iterations), pass_band=0.08)
    surf.compute_normals(inplace=True, auto_orient_normals=True, point_normals=True,
                         cell_normals=False)
    return surf


def _substrate(extent_nm: tuple[float, float], depth_nm: float, pad_nm: tuple[float, float]):
    """The slab under the film: the field's footprint, *depth_nm* deep,
    reaching a voxel past the field so the closed resist sits on it."""
    import pyvista as pv

    x1, y1 = extent_nm
    px, py = pad_nm
    box = pv.Box(bounds=(-px, x1 + px, -py, y1 + py, -depth_nm, 0.0)).triangulate()
    # A box's eight corners are shared between three faces each, and a
    # point normal averaged over them would smear across every face once
    # interpolated. Split the corners so each face keeps its own.
    return box.compute_normals(point_normals=True, cell_normals=False,
                               auto_orient_normals=True, split_vertices=True,
                               feature_angle=30.0)


def _frame(bounds_pts: NDArray, view, up, right,
           stage: Stage) -> tuple[int, int, NDArray, float]:
    """Image size and focal point that frame *bounds_pts* at ``stage.pixel_nm``."""
    r = bounds_pts @ right
    u = bounds_pts @ up
    span_r, span_u = float(r.max() - r.min()), float(u.max() - u.min())
    if stage.pixel_nm is None:
        content = max(stage.frame_px - 2 * stage.margin_px, 32)
        px = max(max(span_r, span_u) / content, 0.2)
    else:
        px = float(stage.pixel_nm)
    cols = int(np.ceil(span_r / px)) + 2 * stage.margin_px
    rows = int(np.ceil(span_u / px)) + 2 * stage.margin_px
    side = max(cols, rows)
    if side > _MAX_SIDE:
        scale = side / _MAX_SIDE
        px *= scale
        cols = int(np.ceil(span_r / px)) + 2 * stage.margin_px
        rows = int(np.ceil(span_u / px)) + 2 * stage.margin_px
        logger.info("render capped at %d px; pixel pitch is %.3f nm", _MAX_SIDE, px)
    centre = bounds_pts.mean(axis=0)
    mid_r, mid_u = 0.5 * (r.max() + r.min()), 0.5 * (u.max() + u.min())
    focal = centre + (mid_r - centre @ right) * right + (mid_u - centre @ up) * up
    return cols, rows, focal, px


def render_buffers(
    surface,
    extent_nm: tuple[float, float],
    film_nm: float,
    stage: Stage | None = None,
    *,
    voxel_nm: tuple[float, float] = (0.0, 0.0),
) -> GeometryBuffers:
    """The geometry the beam sees: normals, depth and material per pixel.

    Parameters
    ----------
    surface : pyvista.PolyData
        The resist, from :func:`profile_surface`. May be empty.
    extent_nm : (x, y)
        The field's footprint, nanometres.
    film_nm : float
        The as-coated film thickness, for the bloom's reference height.
    stage : Stage
    voxel_nm : (dx, dy)
        The lateral voxel size. The substrate reaches one voxel past the
        field so the closed surface, which overhangs it by up to half a
        voxel, always stands on it.
    """
    import pyvista as pv

    stage = stage or Stage()
    view, up, right = camera_basis(stage.tilt, stage.azimuth)
    x1, y1 = float(extent_nm[0]), float(extent_nm[1])
    pad = (float(voxel_nm[0]), float(voxel_nm[1]))
    sub = _substrate((x1, y1), stage.substrate_nm, pad)
    corners = np.array([[x, y, z] for x in (-pad[0], x1 + pad[0])
                        for y in (-pad[1], y1 + pad[1])
                        for z in (-stage.substrate_nm, float(film_nm))])
    cols, rows, focal, px = _frame(corners, view, up, right, stage)
    diag = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))
    cam_pos = focal - view * (3.0 * diag + 100.0)

    pl = pv.Plotter(off_screen=True, window_size=[cols, rows])
    pl.set_background("black")  # type: ignore[arg-type]
    # No multisampling and no anti-aliasing: a blended edge pixel would be
    # a normal that points nowhere and a material that is neither. The
    # probe's blur, applied later to the *signal*, is the only softening.
    pl.render_window.SetMultiSamples(0)  # type: ignore[union-attr]
    pl.disable_anti_aliasing()
    actors: list[tuple[Any, int]] = []
    meshes = [(sub, 1)]
    if surface is not None and surface.n_points > 0:
        meshes.append((surface, 2))
    for mesh, mat in meshes:
        rgb = 0.5 * (np.asarray(mesh.point_data["Normals"], dtype=np.float64) + 1.0)
        mesh.point_data["_n_rgb"] = np.clip(rgb, 0.0, 1.0)
        actor = pl.add_mesh(mesh, scalars="_n_rgb", rgb=True, lighting=False,
                            show_scalar_bar=False, interpolate_before_map=True,
                            culling=False)
        actors.append((actor, mat))
    pl.enable_parallel_projection()  # type: ignore[call-arg]
    pl.camera_position = [tuple(cam_pos), tuple(focal), tuple(up)]
    pl.camera.parallel_scale = 0.5 * rows * px
    pl.renderer.ResetCameraClippingRange()

    shot = np.asarray(pl.screenshot(return_img=True), dtype=np.float64) / 255.0
    zbuf = np.asarray(pl.get_image_depth(fill_value=np.nan), dtype=np.float64)
    # Pass two: each actor a flat material id, scalars off, lighting off.
    for actor, mat in actors:
        actor.mapper.scalar_visibility = False
        actor.prop.color = (mat / 4.0, 0.0, 0.0)
    pl.render()                      # screenshot() alone would hand back the last frame
    ids = np.asarray(pl.screenshot(return_img=True))
    pl.close()
    for mesh, _ in meshes:
        del mesh.point_data["_n_rgb"]

    material = np.rint(ids[..., 0].astype(np.float64) / 255.0 * 4.0).astype(np.uint8)
    material = np.clip(material, 0, 2)
    seen = material > 0
    # VTK hands back view-space z: negative, more negative further away.
    # Distance along the beam from the camera is what the physics wants,
    # and only differences matter, so re-zero at the nearest point.
    depth = np.abs(np.asarray(zbuf, dtype=np.float64))
    depth = np.where(seen & np.isfinite(depth), depth, np.nan)
    # Along a silhouette the colour pass paints a pixel the z-buffer left
    # as background (the two are resolved differently at an edge). Those
    # pixels are surface, and take the depth of the nearest one that has it.
    holes = seen & ~np.isfinite(depth)
    if holes.any() and np.isfinite(depth).any():
        _, (ri, ci) = distance_transform_edt(~np.isfinite(depth), return_indices=True)
        depth = np.where(holes, depth[ri, ci], depth)
    if np.isfinite(depth).any():
        depth -= float(np.nanmin(depth))
    normals = 2.0 * shot - 1.0
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(seen[..., None], normals / np.maximum(norm, 1e-9), np.nan)
    return GeometryBuffers(
        normals=normals, depth=depth, material=material,
        view=view, up=up, right=right, focal=focal,
        pixel_nm=float(px), film_nm=float(film_nm),
    )


def tilt_sem_of_profile(
    remaining: NDArray[np.bool_] | None,
    grid,
    *,
    field: NDArray[np.float64] | None = None,
    level: float | None = None,
    feature: str = "above",
    film_nm: float | None = None,
    stage: Stage | None = None,
    cfg: SEMConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SEMImage, GeometryBuffers]:
    """A tilt-SEM frame of a developed resist volume, start to finish.

    Parameters
    ----------
    remaining : (nz, ny, nx) bool, optional
        The developed solid; or pass *field*, *level* and *feature* for
        the sub-voxel surface — the ``field``/``level``/``feature`` a
        :func:`~litho_sim.develop.resist3d.print_resist_3d` result carries.
    grid : GridConfig
        Supplies ``pixel_size`` and ``dz``.
    film_nm : float, optional
        As-coated thickness. Defaults to the volume's height.
    stage, cfg
        Where the sample sits and what the instrument is.

    Returns
    -------
    (SEMImage, GeometryBuffers)
        The frame, and the geometry it was formed from — the buffers carry
        the projection, so a caller can place a scale bar on the sample.
    """
    vol = remaining if remaining is not None else field
    if vol is None:
        raise ValueError("pass remaining, or field and level")
    nz, ny, nx = np.asarray(vol).shape
    px, dz = float(grid.pixel_size) * 1e9, float(grid.dz) * 1e9
    surface = profile_surface(remaining, field=field, level=level, feature=feature,
                              spacing_nm=(dz, px, px))
    thickness = float(film_nm) if film_nm is not None else nz * dz
    buffers = render_buffers(surface, (nx * px, ny * px), thickness, stage, voxel_nm=(px, px))
    return tilt_sem(buffers, cfg, rng=rng), buffers
