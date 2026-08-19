"""
SEMI-style wafer flatness metrology.

Implements the SEMI M1 family of shape and flatness parameters:

===========  ===============================================================
Parameter    Definition
===========  ===============================================================
``bow``      Signed deflection of the **median surface** at the wafer centre,
             relative to a least-squares reference plane.  Sign only; it says
             nothing about how non-flat the wafer is.
``warp``     Peak-to-valley of the median surface about that same reference
             plane.  Always positive, and always >= ``|bow|``.
``TTV``      Total thickness variation: range of ``z_front - z_back``.
``sori``     Range of the **front** surface about its own least-squares
             reference plane.
``SFQR``     Site Front least-sQuares Range: per exposure site, the range of
             the front surface about a plane fitted to that site alone.
===========  ===============================================================

Why the median surface
----------------------
Bow and warp describe *shape*, and must not be contaminated by *thickness*.
The median surface ``0.5 * (z_front + z_back)`` is where a wafer of zero
thickness variation would sit, so thickness cancels exactly.  TTV, conversely,
is pure thickness and carries no shape.  Keeping the two separate is the whole
point of the SEMI definitions.

Why SFQR is the one that matters
--------------------------------
A scanner refocuses at every exposure field, so it tracks out global shape.
What it cannot track out is height variation *within* one field -- that eats
directly into depth of focus.  SFQR is therefore the flatness number that
gates lithography, and for silicon-photonics packaging it is also the number
that gates fibre-to-chip coupling, where sub-micron standoff error costs dB.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .plotting import (
    CMAP_MAGNITUDE,
    PALETTE,
    annotate_stats,
    wafer_imshow,
)
from .synthesize import WaferPair, WaferSurface, apply_mask

logger = logging.getLogger(__name__)

DEFAULT_SITE_SIZE: float = 25e-3
"""Default exposure-site edge length [m] (a 25 x 25 mm scanner field)."""

DEFAULT_EDGE_EXCLUSION: float = 3e-3
"""Default fixed quality area edge exclusion [m]."""


# ---------------------------------------------------------------------------
# Reference planes
# ---------------------------------------------------------------------------


def fit_reference_plane(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    mask: Optional[NDArray[np.bool_]] = None,
) -> Tuple[NDArray[np.float64], Tuple[float, float, float]]:
    """Fit a least-squares reference plane ``z = a x + b y + c``.

    Parameters
    ----------
    x, y : NDArray
        Coordinate grids [m].
    z : NDArray
        Height map [m]; ``NaN`` values are excluded.
    mask : NDArray of bool, optional
        Valid-pixel mask; defaults to the finite pixels of *z*.

    Returns
    -------
    (NDArray, (float, float, float))
        The evaluated plane over the whole grid [m] (``NaN`` outside *mask*)
        and the coefficients ``(a, b, c)``.

    Raises
    ------
    ValueError
        If fewer than three valid pixels are available.
    """
    valid = np.isfinite(z) if mask is None else (mask & np.isfinite(z))
    if int(valid.sum()) < 3:
        raise ValueError("need at least 3 valid pixels to fit a reference plane")

    design = np.column_stack([x[valid], y[valid], np.ones(int(valid.sum()))])
    (a, b, c), *_ = np.linalg.lstsq(design, z[valid], rcond=None)
    plane = a * x + b * y + c
    return apply_mask(plane, valid), (float(a), float(b), float(c))


def plane_deviation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.float64]:
    """Return *z* minus its least-squares reference plane.

    Parameters
    ----------
    x, y : NDArray
        Coordinate grids [m].
    z : NDArray
        Height map [m].
    mask : NDArray of bool, optional
        Valid-pixel mask.

    Returns
    -------
    NDArray
        Deviation from the reference plane [m], ``NaN`` outside the mask.
    """
    plane, _ = fit_reference_plane(x, y, z, mask)
    return z - plane  # NaN in either term propagates, so the mask is preserved


# ---------------------------------------------------------------------------
# Site flatness (SFQR)
# ---------------------------------------------------------------------------


def _grouped_plane_fit(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    z: NDArray[np.float64],
    group: NDArray[np.int64],
    n_groups: int,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Fit an independent plane per group; return residual range, count, coeffs.

    Solves all groups at once through batched 3x3 normal equations rather than
    looping, so a full wafer of sites costs a handful of ``bincount`` passes.

    Parameters
    ----------
    u, v : NDArray
        Flat arrays of site-local coordinates [m].
    z : NDArray
        Flat array of heights [m].
    group : NDArray of int
        Flat array of group (site) indices in ``[0, n_groups)``.
    n_groups : int
        Number of groups.

    Returns
    -------
    (NDArray, NDArray, NDArray)
        Per-group residual range [m] (``NaN`` where the fit is not defined),
        per-group point count, and per-group plane coefficients ``(a, b, c)``
        for ``z = a u + b v + c`` in the site-local frame -- so ``(a, b)`` is
        the site tilt and ``c`` the plane height at the site centre.
    """
    def s(weights: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        return np.bincount(group, weights=weights, minlength=n_groups)

    n = s()
    su, sv, sz = s(u), s(v), s(z)
    suu, suv, svv = s(u * u), s(u * v), s(v * v)
    suz, svz = s(u * z), s(v * z)

    # Normal equations for z = a*u + b*v + c, stacked over groups.
    mat = np.empty((n_groups, 3, 3), dtype=np.float64)
    mat[:, 0, 0], mat[:, 0, 1], mat[:, 0, 2] = suu, suv, su
    mat[:, 1, 0], mat[:, 1, 1], mat[:, 1, 2] = suv, svv, sv
    mat[:, 2, 0], mat[:, 2, 1], mat[:, 2, 2] = su, sv, n
    rhs = np.column_stack([suz, svz, sz])

    coeffs = np.full((n_groups, 3), np.nan, dtype=np.float64)
    # A determinant test is not enough: a site whose pixels happen to be nearly
    # collinear (a sliver at the wafer edge, or a coarse grid) gives a tiny but
    # non-zero determinant, and solving it either raises or returns garbage.
    # Compare the smallest singular value against the largest instead.
    singular = np.linalg.svd(mat, compute_uv=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        conditioning = np.where(singular[:, 0] > 0.0, singular[:, 2] / singular[:, 0], 0.0)
    solvable = (n >= 3) & (conditioning > 1e-10)
    if solvable.any():
        # NumPy 2 treats a 2-D rhs as a single matrix, so batch it explicitly.
        coeffs[solvable] = np.linalg.solve(mat[solvable], rhs[solvable][..., None])[..., 0]

    residual = z - (coeffs[group, 0] * u + coeffs[group, 1] * v + coeffs[group, 2])

    hi = np.full(n_groups, -np.inf, dtype=np.float64)
    lo = np.full(n_groups, np.inf, dtype=np.float64)
    np.maximum.at(hi, group, residual)
    np.minimum.at(lo, group, residual)

    span = hi - lo
    span[~solvable] = np.nan
    return span, n.astype(np.int64), coeffs


def _tile_sites(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    mask: NDArray[np.bool_],
    radius: float,
    site_size: float,
    edge_exclusion: float,
    site_offset: Tuple[float, float],
) -> Tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Tile the quality area into sites and build site-local coordinates.

    Parameters
    ----------
    x, y : NDArray
        Coordinate grids [m].
    z : NDArray
        Height map [m]; non-finite pixels are excluded.
    mask : NDArray of bool
        Wafer aperture mask.
    radius : float
        Wafer radius [m].
    site_size : float
        Site edge length [m].
    edge_exclusion : float
        Fixed quality area edge exclusion [m].
    site_offset : (float, float)
        Site grid offset [m]; ``(0, 0)`` centres one site on the wafer centre.

    Returns
    -------
    (quality, group, cx, cy, u, v)
        Quality-area mask; per-quality-pixel site index; site centre
        coordinates [m] (one entry per site); and site-local pixel coordinates
        [m] aligned with *group*.

    Raises
    ------
    ValueError
        If *site_size* is not positive or the quality area is empty.
    """
    if site_size <= 0.0:
        raise ValueError(f"site_size must be positive, got {site_size}")

    quality = mask & np.isfinite(z) & (np.hypot(x, y) <= radius - edge_exclusion)
    if not quality.any():
        raise ValueError("quality area is empty; check edge_exclusion")

    # Integer site indices, with a site centred on the wafer centre by default.
    ix = np.floor((x - site_offset[0]) / site_size + 0.5).astype(np.int64)
    iy = np.floor((y - site_offset[1]) / site_size + 0.5).astype(np.int64)

    pairs = np.stack([ix[quality], iy[quality]], axis=1)
    unique_sites, group = np.unique(pairs, axis=0, return_inverse=True)
    group = group.astype(np.int64).ravel()

    # Site-local coordinates, so each plane fit is well conditioned.
    cx = unique_sites[:, 0] * site_size + site_offset[0]
    cy = unique_sites[:, 1] * site_size + site_offset[1]
    u = x[quality] - cx[group]
    v = y[quality] - cy[group]
    return quality, group, cx.astype(np.float64), cy.astype(np.float64), u, v


def site_planes(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    mask: NDArray[np.bool_],
    radius: float,
    *,
    site_size: float = DEFAULT_SITE_SIZE,
    edge_exclusion: float = DEFAULT_EDGE_EXCLUSION,
    site_offset: Tuple[float, float] = (0.0, 0.0),
    min_points: int = 16,
) -> Tuple[pd.DataFrame, NDArray[np.int64]]:
    """Fit a plane to every site and return tilts, centre heights and ranges.

    This is the site-resolved primitive behind both SFQR and the fibre-attach
    loss budget: SFQR only needs each site's residual *range*, while the loss
    budget needs the *plane itself* -- the site tilt sets the angular and
    lateral misalignment of an attached fibre, and the plane height at the
    site centre sets its standoff error.

    Parameters
    ----------
    x, y : NDArray
        Coordinate grids [m].
    z : NDArray
        Height map [m].
    mask : NDArray of bool
        Wafer aperture mask.
    radius : float
        Wafer radius [m].
    site_size : float
        Site edge length [m] (an exposure field or a die footprint).
    edge_exclusion : float
        Fixed quality area edge exclusion [m].
    site_offset : (float, float)
        Site grid offset [m].
    min_points : int
        Sites sampled by fewer pixels are marked incomplete.

    Returns
    -------
    (DataFrame, NDArray)
        Per-site table with columns ``site_id, x_center_mm, y_center_mm,
        n_points, complete, tilt_x, tilt_y, height_center, range_m`` (tilts in
        rad, heights in m); and a wafer-shaped int map assigning each pixel its
        ``site_id`` (-1 outside the quality area) for painting per-site values
        back onto the wafer.

    Examples
    --------
    >>> from .synthesize import synthesize_wafer
    >>> s = synthesize_wafer(n_pixels=192, seed=0)
    >>> table, ids = site_planes(s.x, s.y, s.z, s.mask, s.radius)
    >>> bool(np.isfinite(table.loc[table.complete, "tilt_x"]).all())
    True
    """
    quality, group, cx, cy, u, v = _tile_sites(
        x, y, z, mask, radius, site_size, edge_exclusion, site_offset
    )
    n_sites = cx.size
    span, counts, coeffs = _grouped_plane_fit(u, v, z[quality], group, n_sites)

    half = 0.5 * site_size
    corner_radius = np.hypot(np.abs(cx) + half, np.abs(cy) + half)
    complete = (corner_radius <= radius - edge_exclusion) & (counts >= min_points)

    table = pd.DataFrame(
        {
            "site_id": np.arange(n_sites),
            "x_center_mm": cx * 1e3,
            "y_center_mm": cy * 1e3,
            "n_points": counts,
            "complete": complete,
            "tilt_x": coeffs[:, 0],
            "tilt_y": coeffs[:, 1],
            "height_center": coeffs[:, 2],
            "range_m": span,
        }
    )

    site_id_map = np.full(x.shape, -1, dtype=np.int64)
    site_id_map[quality] = group
    return table, site_id_map


def site_flatness(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z_front: NDArray[np.float64],
    mask: NDArray[np.bool_],
    radius: float,
    *,
    site_size: float = DEFAULT_SITE_SIZE,
    edge_exclusion: float = DEFAULT_EDGE_EXCLUSION,
    site_offset: Tuple[float, float] = (0.0, 0.0),
    min_points: int = 16,
) -> Tuple[pd.DataFrame, NDArray[np.float64]]:
    """Tile the wafer into exposure sites and compute SFQR for each.

    Each site gets its own least-squares reference plane fitted to the front
    surface within that site; SFQR is the range of the residual.  A site is
    marked *complete* only when its whole square lies inside the fixed quality
    area, matching the SEMI practice of reporting flatness over full sites.

    Parameters
    ----------
    x, y : NDArray
        Coordinate grids [m].
    z_front : NDArray
        Front-surface height map [m].
    mask : NDArray of bool
        Wafer aperture mask.
    radius : float
        Wafer radius [m].
    site_size : float
        Exposure site edge length [m].
    edge_exclusion : float
        Fixed quality area edge exclusion [m].
    site_offset : (float, float)
        Shift of the site grid relative to the wafer centre [m].  With the
        default of ``(0, 0)`` one site is centred on the wafer centre.
    min_points : int
        Sites sampled by fewer pixels than this are dropped as unresolved.

    Returns
    -------
    (DataFrame, NDArray)
        Per-site table with columns ``site_id, ix, iy, x_center, y_center,
        n_points, complete, sfqr``; and a wafer-shaped map of the SFQR of each
        pixel's site (``NaN`` where undefined).

    Raises
    ------
    ValueError
        If *site_size* is not positive.

    Examples
    --------
    >>> from .synthesize import synthesize_wafer
    >>> s = synthesize_wafer(n_pixels=192, seed=0)
    >>> table, smap = site_flatness(s.x, s.y, s.z, s.mask, s.radius)
    >>> bool((table.loc[table.complete, "sfqr"] > 0).all())
    True
    """
    quality, group, cx, cy, u, v = _tile_sites(
        x, y, z_front, mask, radius, site_size, edge_exclusion, site_offset
    )
    n_sites = cx.size
    sfqr, counts, _ = _grouped_plane_fit(u, v, z_front[quality], group, n_sites)

    # A site is complete when its farthest corner is still inside the quality area.
    half = 0.5 * site_size
    corner_radius = np.hypot(np.abs(cx) + half, np.abs(cy) + half)
    complete = (corner_radius <= radius - edge_exclusion) & (counts >= min_points)

    table = pd.DataFrame(
        {
            "site_id": np.arange(n_sites),
            "ix": np.round((cx - site_offset[0]) / site_size).astype(np.int64),
            "iy": np.round((cy - site_offset[1]) / site_size).astype(np.int64),
            "x_center_mm": cx * 1e3,
            "y_center_mm": cy * 1e3,
            "n_points": counts,
            "complete": complete,
            "sfqr_nm": sfqr * 1e9,
            "sfqr": sfqr,
        }
    ).sort_values("site_id", ignore_index=True)

    # Paint each pixel with its site's SFQR for the spatial map.
    sfqr_map = np.full(x.shape, np.nan, dtype=np.float64)
    painted = np.where(complete[group], sfqr[group], np.nan)
    sfqr_map[quality] = painted

    logger.info(
        "Site flatness: %d sites (%d complete), SFQR max %.1f nm, mean %.1f nm",
        n_sites, int(complete.sum()),
        np.nanmax(sfqr[complete]) * 1e9 if complete.any() else np.nan,
        np.nanmean(sfqr[complete]) * 1e9 if complete.any() else np.nan,
    )
    return table, sfqr_map


# ---------------------------------------------------------------------------
# Global metrics
# ---------------------------------------------------------------------------


@dataclass
class FlatnessMetrics:
    """SEMI-style flatness and shape metrics for one wafer, all in metres.

    Attributes
    ----------
    bow : float
        Signed centre deflection of the median surface [m].
    warp : float
        Peak-to-valley of the median surface about its reference plane [m].
    ttv : float
        Total thickness variation [m]; ``nan`` when only one surface is known.
    sori : float
        Front-surface range about its reference plane [m].
    sfqr_max : float
        Worst complete-site SFQR [m] -- the number that gates lithography.
    sfqr_mean : float
        Mean complete-site SFQR [m].
    sfqr_p99 : float
        99th percentile of complete-site SFQR [m].
    thickness_mean : float
        Mean wafer thickness [m]; ``nan`` when only one surface is known.
    n_sites : int
        Number of complete sites contributing to the SFQR statistics.
    site_size : float
        Exposure site edge length used [m].
    edge_exclusion : float
        Edge exclusion used [m].
    """

    bow: float
    warp: float
    ttv: float
    sori: float
    sfqr_max: float
    sfqr_mean: float
    sfqr_p99: float
    thickness_mean: float
    n_sites: int
    site_size: float
    edge_exclusion: float

    def to_series(self) -> pd.Series:
        """Return the metrics as a :class:`pandas.Series` in metres."""
        return pd.Series(asdict(self))

    def to_frame(self) -> pd.DataFrame:
        """Return a display table with conventional units and descriptions.

        Returns
        -------
        DataFrame
            Columns ``parameter, value, unit, description``.
        """
        rows = [
            ("Bow", self.bow * 1e6, "µm", "Signed centre deflection of median surface"),
            ("Warp", self.warp * 1e6, "µm", "PV of median surface about reference plane"),
            ("TTV", self.ttv * 1e6, "µm", "Total thickness variation (front − back)"),
            ("Sori", self.sori * 1e6, "µm", "Front-surface range about reference plane"),
            ("SFQR max", self.sfqr_max * 1e9, "nm", f"Worst of {self.n_sites} complete sites"),
            ("SFQR mean", self.sfqr_mean * 1e9, "nm", "Mean site front least-squares range"),
            ("SFQR p99", self.sfqr_p99 * 1e9, "nm", "99th percentile site flatness"),
            ("Thickness", self.thickness_mean * 1e6, "µm", "Mean wafer thickness"),
            ("Site size", self.site_size * 1e3, "mm", "Exposure field edge length"),
            ("Edge exclusion", self.edge_exclusion * 1e3, "mm", "Fixed quality area margin"),
        ]
        return pd.DataFrame(rows, columns=["parameter", "value", "unit", "description"])


@dataclass
class FlatnessResult:
    """Full flatness analysis: metrics, per-site table and the intermediate maps.

    Attributes
    ----------
    metrics : FlatnessMetrics
        Scalar summary.
    sites : DataFrame
        Per-site SFQR table from :func:`site_flatness`.
    sfqr_map : NDArray
        Spatial SFQR map [m].
    median_deviation : NDArray
        Median surface minus its reference plane [m] -- the map warp measures.
    thickness : NDArray
        Local thickness [m]; all-``NaN`` when only one surface is known.
    x, y : NDArray
        Coordinate grids [m].
    radius : float
        Wafer radius [m].
    """

    metrics: FlatnessMetrics
    sites: pd.DataFrame
    sfqr_map: NDArray[np.float64]
    median_deviation: NDArray[np.float64]
    thickness: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    radius: float

    def to_csv(self, path: str | Path) -> Path:
        """Write the summary metrics table to CSV.

        Parameters
        ----------
        path : str or Path
            Destination file.

        Returns
        -------
        Path
            The resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.metrics.to_frame().to_csv(out, index=False)
        logger.info("Wrote %s", out)
        return out

    def sites_to_csv(self, path: str | Path) -> Path:
        """Write the per-site SFQR table to CSV.

        Parameters
        ----------
        path : str or Path
            Destination file.

        Returns
        -------
        Path
            The resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.sites.to_csv(out, index=False)
        logger.info("Wrote %s", out)
        return out


def compute_flatness(
    wafer: WaferPair | WaferSurface,
    *,
    site_size: float = DEFAULT_SITE_SIZE,
    edge_exclusion: float = DEFAULT_EDGE_EXCLUSION,
    center_radius: float = 2e-3,
    site_offset: Tuple[float, float] = (0.0, 0.0),
) -> FlatnessResult:
    """Compute the full SEMI-style flatness metric set for a wafer.

    Accepts a :class:`~wafer_metrology.synthesize.WaferPair` (front and back
    known, so TTV is available) or a single
    :class:`~wafer_metrology.synthesize.WaferSurface`, in which case the surface
    is treated as both the median and the front surface and TTV is ``nan``.
    That single-surface mode is what a reflection interferometer gives you:
    it sees one face, so it can report warp-like shape and SFQR but not
    thickness.

    Parameters
    ----------
    wafer : WaferPair or WaferSurface
        Wafer to analyse.
    site_size : float
        Exposure site edge length [m].
    edge_exclusion : float
        Fixed quality area edge exclusion [m].
    center_radius : float
        Radius of the central patch averaged to evaluate bow [m].  Averaging
        over a small patch instead of a single pixel makes bow robust to
        nanotopography; pass ``0`` for the strict single-point definition.
    site_offset : (float, float)
        Site grid offset [m].

    Returns
    -------
    FlatnessResult
        Metrics, per-site table and maps.

    Examples
    --------
    >>> from .synthesize import synthesize_wafer_pair
    >>> pair = synthesize_wafer_pair(n_pixels=192, seed=0)
    >>> res = compute_flatness(pair)
    >>> bool(res.metrics.warp >= abs(res.metrics.bow))
    True
    """
    if isinstance(wafer, WaferPair):
        x, y, mask, radius = wafer.x, wafer.y, wafer.mask, wafer.radius
        median = wafer.median
        z_front = wafer.z_front
        thickness = wafer.thickness
    else:
        x, y, mask, radius = wafer.x, wafer.y, wafer.mask, wafer.radius
        median = wafer.z
        z_front = wafer.z
        thickness = np.full_like(wafer.z, np.nan)

    quality = mask & (np.hypot(x, y) <= radius - edge_exclusion)

    # --- shape: bow and warp on the median surface -------------------------
    median_plane, _ = fit_reference_plane(x, y, median, quality)
    deviation = apply_mask(median - median_plane, quality)

    if center_radius > 0.0:
        center_sel = quality & (np.hypot(x, y) <= center_radius)
        if not center_sel.any():  # grid too coarse for the requested patch
            center_sel = quality & (np.hypot(x, y) <= np.hypot(x, y)[quality].min())
    else:
        rr = np.where(quality, np.hypot(x, y), np.inf)
        center_sel = rr == rr.min()
    bow = float(np.nanmean(deviation[center_sel]))
    warp = float(np.nanmax(deviation[quality]) - np.nanmin(deviation[quality]))

    # --- front-surface range (sori) ----------------------------------------
    front_plane, _ = fit_reference_plane(x, y, z_front, quality)
    front_dev = apply_mask(z_front - front_plane, quality)
    sori = float(np.nanmax(front_dev[quality]) - np.nanmin(front_dev[quality]))

    # --- thickness ---------------------------------------------------------
    if np.isfinite(thickness[quality]).any():
        ttv = float(np.nanmax(thickness[quality]) - np.nanmin(thickness[quality]))
        thickness_mean = float(np.nanmean(thickness[quality]))
    else:
        ttv = float("nan")
        thickness_mean = float("nan")

    # --- site flatness -----------------------------------------------------
    sites, sfqr_map = site_flatness(
        x, y, z_front, mask, radius,
        site_size=site_size, edge_exclusion=edge_exclusion, site_offset=site_offset,
    )
    complete = sites.loc[sites["complete"], "sfqr"].to_numpy(dtype=np.float64)
    complete = complete[np.isfinite(complete)]
    if complete.size:
        sfqr_max = float(complete.max())
        sfqr_mean = float(complete.mean())
        sfqr_p99 = float(np.percentile(complete, 99))
    else:
        sfqr_max = sfqr_mean = sfqr_p99 = float("nan")
        logger.warning("No complete sites: site_size %.3g m may exceed the wafer", site_size)

    metrics = FlatnessMetrics(
        bow=bow, warp=warp, ttv=ttv, sori=sori,
        sfqr_max=sfqr_max, sfqr_mean=sfqr_mean, sfqr_p99=sfqr_p99,
        thickness_mean=thickness_mean, n_sites=int(complete.size),
        site_size=site_size, edge_exclusion=edge_exclusion,
    )
    logger.info(
        "Flatness: bow %+.2f µm, warp %.2f µm, TTV %.3f µm, sori %.2f µm, SFQR max %.1f nm",
        bow * 1e6, warp * 1e6, ttv * 1e6, sori * 1e6, sfqr_max * 1e9,
    )
    return FlatnessResult(
        metrics=metrics, sites=sites, sfqr_map=sfqr_map,
        median_deviation=deviation, thickness=apply_mask(thickness, quality),
        x=x, y=y, radius=radius,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_flatness(
    result: FlatnessResult,
    *,
    title: str = "SEMI flatness summary",
    figsize: Tuple[float, float] = (14.0, 8.2),
) -> Tuple[Figure, NDArray]:
    """Four-panel flatness figure: shape, thickness, site map and site histogram.

    Parameters
    ----------
    result : FlatnessResult
        Analysis to display.
    title : str
        Figure suptitle.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    import matplotlib.pyplot as plt

    metrics = result.metrics
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    # 1. Median surface deviation -- the map that defines bow and warp.
    wafer_imshow(
        axes[0], result.x, result.y, result.median_deviation,
        scale=1e6, label="Deviation [µm]", signed=True, radius=result.radius,
        title="Median surface vs. reference plane",
    )
    annotate_stats(
        axes[0],
        f"bow  {metrics.bow * 1e6:+.2f} µm\nwarp {metrics.warp * 1e6:.2f} µm",
        loc="lower left",
    )

    # 2. Thickness -- the map that defines TTV.
    if np.isfinite(result.thickness).any():
        wafer_imshow(
            axes[1], result.x, result.y, result.thickness,
            scale=1e6, label="Thickness [µm]", signed=False, cmap=CMAP_MAGNITUDE,
            radius=result.radius, title="Local thickness",
        )
        annotate_stats(
            axes[1],
            f"TTV  {metrics.ttv * 1e6:.3f} µm\nmean {metrics.thickness_mean * 1e6:.1f} µm",
            loc="lower left",
        )
    else:
        axes[1].text(
            0.5, 0.5,
            "Thickness unavailable\n(single-surface measurement)",
            ha="center", va="center", transform=axes[1].transAxes, fontsize=10,
        )
        axes[1].set_axis_off()

    # 3. Spatial SFQR map over complete sites.
    wafer_imshow(
        axes[2], result.x, result.y, result.sfqr_map,
        scale=1e9, label="SFQR [nm]", signed=False, cmap=CMAP_MAGNITUDE,
        radius=result.radius,
        title=f"Site flatness map ({metrics.site_size * 1e3:.0f} × "
              f"{metrics.site_size * 1e3:.0f} mm sites)",
    )
    annotate_stats(
        axes[2],
        f"SFQR max  {metrics.sfqr_max * 1e9:.1f} nm\n"
        f"SFQR mean {metrics.sfqr_mean * 1e9:.1f} nm\n"
        f"sites     {metrics.n_sites}",
        loc="lower left",
    )

    # 4. Site distribution.
    complete = result.sites.loc[result.sites["complete"], "sfqr_nm"].to_numpy()
    complete = complete[np.isfinite(complete)]
    ax = axes[3]
    if complete.size:
        ax.hist(complete, bins=min(20, max(5, complete.size // 3)),
                color=PALETTE[0], edgecolor="white", linewidth=0.6)
        ax.axvline(complete.mean(), color=PALETTE[1], lw=2, label=f"mean {complete.mean():.1f} nm")
        ax.axvline(complete.max(), color=PALETTE[5], lw=2, ls="--",
                   label=f"max {complete.max():.1f} nm")
        ax.legend(loc="upper right")
    ax.set_xlabel("SFQR [nm]")
    ax.set_ylabel("Site count")
    ax.set_title("Distribution of site flatness")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig, axes
