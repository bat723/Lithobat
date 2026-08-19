"""
Synthetic 300 mm wafer surface generation.

Builds a physically-plausible wafer height map from four superposed
contributions, each independently controllable:

1. **Bow** -- a global paraboloid, the dominant shape term of a warped wafer.
2. **Warp lobes** -- a handful of low-order polynomial lobes (astigmatism,
   coma, trefoil) that give the surface a non-rotationally-symmetric shape,
   as produced by non-uniform film stress or bonding.
3. **Nanotopography** -- smooth, spatially-correlated random height variation
   in the 0.2-20 mm spatial-wavelength band, from Gaussian-filtered white
   noise.
4. **Defects** -- discrete particles (local bumps) and scratches (line
   features), injected at known locations so detection can be scored.

The wafer is a circular aperture inscribed in a square grid; every returned
height map carries ``NaN`` outside that aperture.  All lengths are in metres.
Generation is fully deterministic given ``seed``.

Sign convention
---------------
``z`` increases *away from* the wafer front face (towards the metrology tool).
For a front/back pair, ``z_front > z_back`` everywhere, so the local thickness
``z_front - z_back`` is positive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from . import (
    WAFER_DIAMETER_150MM,
    WAFER_DIAMETER_300MM,
    WAFER_THICKNESS_150MM,
    WAFER_THICKNESS_300MM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefectTruth:
    """Ground-truth record for one injected defect.

    Attributes
    ----------
    kind : str
        ``"particle"`` or ``"scratch"``.
    x, y : float
        Defect centre (scratch mid-point) [m].
    height : float
        Peak signed height of the feature [m]; negative for scratches.
    size : float
        Gaussian sigma for a particle, or length for a scratch [m].
    """

    kind: str
    x: float
    y: float
    height: float
    size: float


@dataclass
class WaferSurface:
    """A single wafer height map on a circular aperture.

    Supports tuple unpacking so the documented ``x, y, z, mask`` contract holds::

        x, y, z, mask = synthesize_wafer(seed=0)

    Attributes
    ----------
    x, y : NDArray
        Coordinate grids [m], shape ``(n, n)``, origin at wafer centre.
    z : NDArray
        Height map [m] with ``NaN`` outside the aperture.
    mask : NDArray of bool
        ``True`` inside the wafer aperture.
    radius : float
        Wafer radius [m].
    pixel_size : float
        Grid pitch [m].
    defects : list of DefectTruth
        Ground truth for the injected defects.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    mask: NDArray[np.bool_]
    radius: float
    pixel_size: float
    defects: List[DefectTruth] = field(default_factory=list)

    def __iter__(self) -> Iterator[NDArray]:
        """Yield ``x, y, z, mask`` so the object unpacks as a 4-tuple."""
        yield self.x
        yield self.y
        yield self.z
        yield self.mask

    @property
    def n_pixels(self) -> int:
        """Grid size along one axis."""
        return int(self.z.shape[0])

    @property
    def values(self) -> NDArray[np.float64]:
        """1-D array of the in-aperture height values [m]."""
        return self.z[self.mask]

    def pv(self) -> float:
        """Peak-to-valley height across the aperture [m]."""
        return float(np.nanmax(self.z) - np.nanmin(self.z))

    def rms(self) -> float:
        """RMS height about the mean across the aperture [m]."""
        v = self.values
        return float(np.sqrt(np.mean((v - v.mean()) ** 2)))

    def with_z(self, z_new: NDArray[np.float64]) -> "WaferSurface":
        """Return a copy carrying a different height map (same grid/metadata)."""
        return WaferSurface(
            x=self.x, y=self.y, z=z_new, mask=self.mask,
            radius=self.radius, pixel_size=self.pixel_size, defects=list(self.defects),
        )


@dataclass
class WaferPair:
    """Front and back surfaces of one wafer, as a thickness-resolved pair.

    The *median surface* ``0.5 * (z_front + z_back)`` is the SEMI M1 reference
    for shape metrics (bow, warp) because thickness variation cancels in it;
    ``z_front - z_back`` is the local thickness that drives TTV.

    Attributes
    ----------
    x, y : NDArray
        Coordinate grids [m].
    z_front, z_back : NDArray
        Front and back surface heights [m], ``NaN`` outside the aperture.
    mask : NDArray of bool
        ``True`` inside the wafer aperture.
    radius : float
        Wafer radius [m].
    pixel_size : float
        Grid pitch [m].
    defects : list of DefectTruth
        Ground truth for defects injected on the front surface.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z_front: NDArray[np.float64]
    z_back: NDArray[np.float64]
    mask: NDArray[np.bool_]
    radius: float
    pixel_size: float
    defects: List[DefectTruth] = field(default_factory=list)

    @property
    def median(self) -> NDArray[np.float64]:
        """Median surface ``0.5 * (z_front + z_back)`` [m] -- thickness-independent."""
        return 0.5 * (self.z_front + self.z_back)

    @property
    def thickness(self) -> NDArray[np.float64]:
        """Local wafer thickness ``z_front - z_back`` [m]."""
        return self.z_front - self.z_back

    @property
    def front(self) -> WaferSurface:
        """The front surface as a standalone :class:`WaferSurface`."""
        return WaferSurface(
            x=self.x, y=self.y, z=self.z_front, mask=self.mask,
            radius=self.radius, pixel_size=self.pixel_size, defects=list(self.defects),
        )


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def make_wafer_grid(
    n_pixels: int = 512,
    diameter: float = WAFER_DIAMETER_300MM,
    edge_exclusion: float = 0.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.bool_]]:
    """Build the square coordinate grid and circular wafer aperture.

    The grid spans exactly one wafer diameter, so the aperture is inscribed in
    the array.

    Parameters
    ----------
    n_pixels : int
        Samples along one axis.
    diameter : float
        Wafer diameter [m].
    edge_exclusion : float
        Width of an annular exclusion zone at the wafer edge [m].  Pixels
        within this distance of the edge are excluded from *mask*.  SEMI
        flatness specifications are quoted over such a fixed quality area
        (commonly 1-3 mm).

    Returns
    -------
    x, y : NDArray
        Cartesian coordinate grids [m], origin at the wafer centre.
    r : NDArray
        Radial coordinate [m].
    theta : NDArray
        Azimuthal coordinate [rad] in ``(-pi, pi]``.
    mask : NDArray of bool
        ``True`` inside the (edge-excluded) aperture.

    Raises
    ------
    ValueError
        If *edge_exclusion* removes the whole wafer.
    """
    radius = 0.5 * diameter
    if edge_exclusion >= radius:
        raise ValueError(
            f"edge_exclusion {edge_exclusion:.4g} m exceeds wafer radius {radius:.4g} m"
        )
    pixel_size = diameter / n_pixels
    # Pixel centres, symmetric about zero.
    c = (np.arange(n_pixels, dtype=np.float64) - (n_pixels - 1) / 2.0) * pixel_size
    x, y = np.meshgrid(c, c)
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    mask = r <= (radius - edge_exclusion)
    return x, y, r, theta, mask


def apply_mask(z: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return *z* with ``NaN`` written outside *mask*.

    Parameters
    ----------
    z : NDArray
        Height map [m].
    mask : NDArray of bool
        ``True`` where the surface exists.

    Returns
    -------
    NDArray
        Copy of *z*, ``NaN`` outside the aperture.
    """
    out = np.array(z, dtype=np.float64, copy=True)
    out[~mask] = np.nan
    return out


def make_flat_surface(
    n_pixels: int = 512,
    diameter: float = WAFER_DIAMETER_300MM,
    edge_exclusion: float = 0.0,
) -> WaferSurface:
    """Build an ideally flat reference surface on the wafer grid.

    Stands in for the front-surface reference flat of the hardware build:
    measuring it returns (to within its own figure error, zero here) the
    instrument's fixed systematic, which is the absolute calibration map.

    Parameters
    ----------
    n_pixels : int
        Samples along one axis.
    diameter : float
        Aperture diameter [m].
    edge_exclusion : float
        Edge exclusion applied to the mask [m].

    Returns
    -------
    WaferSurface
        A zero-height surface with the standard grid metadata.
    """
    x, y, _, _, mask = make_wafer_grid(n_pixels, diameter, edge_exclusion)
    return WaferSurface(
        x=x, y=y, z=apply_mask(np.zeros_like(x), mask), mask=mask,
        radius=0.5 * diameter, pixel_size=diameter / n_pixels,
    )


def gravity_sag_horizontal(
    diameter: float = WAFER_DIAMETER_150MM,
    thickness: float = WAFER_THICKNESS_150MM,
    *,
    youngs_modulus: float = 150e9,
    poisson: float = 0.27,
    density: float = 2329.0,
) -> float:
    """Estimate the centre sag of a horizontally supported wafer under gravity.

    Uses the closed-form centre deflection of a uniformly loaded, simply
    supported circular plate (Timoshenko):

        w = (5 + nu) / (1 + nu) * q a^4 / (64 D),   D = E h^3 / 12 (1 - nu^2)

    with ``q = rho g h``.  A continuous edge support is the *stiffest* simple
    support, so this is a lower bound; a 3-point edge rest sags roughly 1.5-2x
    more.  Either way the answer for a 150 mm x 675 µm wafer (~7 µm here,
    ~10-14 µm on three points) is the same order as the bow being measured --
    which is the quantitative argument for mounting the wafer vertically.
    Silicon's Young's modulus is anisotropic (~130-188 GPa with direction);
    the default 150 GPa is a representative in-plane value.

    Parameters
    ----------
    diameter : float
        Wafer diameter [m].
    thickness : float
        Wafer thickness [m].
    youngs_modulus : float
        Young's modulus [Pa].
    poisson : float
        Poisson ratio.
    density : float
        Density [kg/m^3].

    Returns
    -------
    float
        Centre sag [m] for continuous simple edge support.

    Examples
    --------
    >>> round(gravity_sag_horizontal() * 1e6, 1)
    7.6
    """
    radius = 0.5 * diameter
    rigidity = youngs_modulus * thickness**3 / (12.0 * (1.0 - poisson**2))
    load = density * 9.80665 * thickness
    return float(
        (5.0 + poisson) / (1.0 + poisson) * load * radius**4 / (64.0 * rigidity)
    )


# ---------------------------------------------------------------------------
# Shape components
# ---------------------------------------------------------------------------


def parabolic_bow(rho: NDArray[np.float64], sag: float) -> NDArray[np.float64]:
    """Global parabolic bow term, zero-mean over the disk.

    Uses ``sag * (rho**2 - 0.5)``; the ``-0.5`` offset removes the mean of
    ``rho**2`` over a unit disk so the term contributes no piston.  The
    centre-to-edge sag is *sag*.

    Parameters
    ----------
    rho : NDArray
        Normalised radius ``r / R`` (dimensionless).
    sag : float
        Centre-to-edge sag [m].  Positive = edge higher than centre (concave-up).

    Returns
    -------
    NDArray
        Height contribution [m].
    """
    return sag * (rho**2 - 0.5)


def warp_lobes(
    rho: NDArray[np.float64],
    theta: NDArray[np.float64],
    amplitude: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Low-order non-symmetric warp lobes (astigmatism, coma, trefoil).

    These are the shape terms a least-squares plane cannot remove, so they
    dominate the residual warp of a bonded or heavily-filmed wafer.

    Parameters
    ----------
    rho : NDArray
        Normalised radius ``r / R``.
    theta : NDArray
        Azimuth [rad].
    amplitude : float
        Overall scale of the combined lobes [m]; individual weights are drawn
        randomly and normalised so the term's peak magnitude is *amplitude*.
    rng : numpy.random.Generator
        Seeded RNG.

    Returns
    -------
    NDArray
        Height contribution [m].
    """
    weights = rng.uniform(0.4, 1.0, size=3)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    lobes = (
        weights[0] * rho**2 * np.cos(2.0 * theta + phases[0])          # astigmatism
        + weights[1] * rho**3 * np.cos(theta + phases[1])              # coma
        + weights[2] * rho**3 * np.cos(3.0 * theta + phases[2])        # trefoil
    )
    peak = float(np.max(np.abs(lobes)))
    if peak == 0.0:
        return np.zeros_like(lobes)
    return amplitude * lobes / peak


def nanotopography(
    shape: Tuple[int, int],
    pixel_size: float,
    rms: float,
    correlation_length: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Smooth, spatially-correlated random height variation.

    White noise is low-pass filtered with a Gaussian of the requested
    correlation length, then rescaled to the requested RMS.  This reproduces
    the mid-spatial-frequency roughness that CMP leaves behind and that site
    flatness is sensitive to.

    Parameters
    ----------
    shape : (int, int)
        Array shape.
    pixel_size : float
        Grid pitch [m].
    rms : float
        Target RMS height [m].
    correlation_length : float
        Gaussian smoothing sigma [m]; sets the spatial wavelength band.
    rng : numpy.random.Generator
        Seeded RNG.

    Returns
    -------
    NDArray
        Zero-mean height contribution [m] with the requested RMS.
    """
    if rms <= 0.0:
        return np.zeros(shape, dtype=np.float64)
    sigma_px = max(correlation_length / pixel_size, 0.5)
    noise = rng.standard_normal(shape)
    smooth = gaussian_filter(noise, sigma=sigma_px, mode="reflect")
    smooth -= smooth.mean()
    current = float(np.sqrt(np.mean(smooth**2)))
    if current == 0.0:
        return smooth
    return smooth * (rms / current)


def _add_particles(
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    radius: float,
    count: int,
    height_range: Tuple[float, float],
    sigma_range: Tuple[float, float],
    rng: np.random.Generator,
) -> List[DefectTruth]:
    """Add Gaussian bumps to *z* in place; return their ground truth."""
    truth: List[DefectTruth] = []
    for _ in range(count):
        # Sample uniformly over the disk area, keeping clear of the extreme edge.
        rr = radius * 0.92 * np.sqrt(rng.uniform(0.0, 1.0))
        aa = rng.uniform(0.0, 2.0 * np.pi)
        cx, cy = rr * np.cos(aa), rr * np.sin(aa)
        height = rng.uniform(*height_range)
        sigma = rng.uniform(*sigma_range)
        z += height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))
        truth.append(DefectTruth("particle", float(cx), float(cy), float(height), float(sigma)))
    return truth


def _point_segment_distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    p0: Tuple[float, float],
    p1: Tuple[float, float],
) -> NDArray[np.float64]:
    """Vectorised distance from every grid point to the segment ``p0``-``p1`` [m]."""
    ax, ay = p0
    bx, by = p1
    dx, dy = bx - ax, by - ay
    seg_sq = dx * dx + dy * dy
    if seg_sq == 0.0:
        return np.hypot(x - ax, y - ay)
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / seg_sq, 0.0, 1.0)
    return np.hypot(x - (ax + t * dx), y - (ay + t * dy))


def _add_scratches(
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    radius: float,
    count: int,
    depth_range: Tuple[float, float],
    width: float,
    length_range: Tuple[float, float],
    rng: np.random.Generator,
) -> List[DefectTruth]:
    """Carve line-feature scratches into *z* in place; return their ground truth."""
    truth: List[DefectTruth] = []
    for _ in range(count):
        rr = radius * 0.75 * np.sqrt(rng.uniform(0.0, 1.0))
        aa = rng.uniform(0.0, 2.0 * np.pi)
        cx, cy = rr * np.cos(aa), rr * np.sin(aa)
        length = rng.uniform(*length_range)
        orient = rng.uniform(0.0, np.pi)
        hx, hy = 0.5 * length * np.cos(orient), 0.5 * length * np.sin(orient)
        depth = -abs(rng.uniform(*depth_range))
        dist = _point_segment_distance(x, y, (cx - hx, cy - hy), (cx + hx, cy + hy))
        z += depth * np.exp(-(dist**2) / (2.0 * width**2))
        truth.append(DefectTruth("scratch", float(cx), float(cy), float(depth), float(length)))
    return truth


# ---------------------------------------------------------------------------
# Top-level synthesis
# ---------------------------------------------------------------------------


def synthesize_wafer(
    n_pixels: int = 512,
    diameter: float = WAFER_DIAMETER_300MM,
    *,
    bow_sag: float = 35e-6,
    warp_amplitude: float = 12e-6,
    tilt: Tuple[float, float] = (4e-6, -2e-6),
    nanotopo_rms: float = 18e-9,
    nanotopo_correlation: float = 3e-3,
    n_particles: int = 6,
    n_scratches: int = 2,
    particle_height: Tuple[float, float] = (180e-9, 700e-9),
    particle_sigma: Tuple[float, float] = (0.6e-3, 1.8e-3),
    scratch_depth: Tuple[float, float] = (120e-9, 400e-9),
    scratch_width: float = 0.5e-3,
    scratch_length: Tuple[float, float] = (18e-3, 55e-3),
    edge_exclusion: float = 0.0,
    seed: Optional[int] = 0,
) -> WaferSurface:
    """Generate a synthetic 300 mm wafer height map.

    Superposes bow, warp lobes, tilt, nanotopography and discrete defects.
    The result is deterministic for a given *seed*.

    Parameters
    ----------
    n_pixels : int
        Samples along one axis of the square grid.
    diameter : float
        Wafer diameter [m].
    bow_sag : float
        Centre-to-edge sag of the parabolic bow term [m].
    warp_amplitude : float
        Peak magnitude of the combined low-order warp lobes [m].
    tilt : (float, float)
        Peak x and y tilt contributions at the wafer edge [m].  Tilt is a
        chucking artefact and is removed by every reference-plane fit; it is
        included so that removal can be demonstrated.
    nanotopo_rms : float
        RMS of the smooth random nanotopography [m].
    nanotopo_correlation : float
        Gaussian correlation length of the nanotopography [m].
    n_particles : int
        Number of particle (bump) defects.
    n_scratches : int
        Number of scratch (line) defects.
    particle_height : (float, float)
        Uniform sampling range for particle peak height [m].
    particle_sigma : (float, float)
        Uniform sampling range for particle Gaussian sigma [m].
    scratch_depth : (float, float)
        Uniform sampling range for scratch depth magnitude [m].
    scratch_width : float
        Gaussian half-width of a scratch cross-section [m].
    scratch_length : (float, float)
        Uniform sampling range for scratch length [m].
    edge_exclusion : float
        Edge exclusion zone width [m] applied to the returned mask.
    seed : int, optional
        RNG seed.  ``None`` draws from OS entropy (non-deterministic).

    Returns
    -------
    WaferSurface
        Surface container; unpacks as ``x, y, z, mask``.

    Examples
    --------
    >>> surf = synthesize_wafer(n_pixels=128, seed=1)
    >>> x, y, z, mask = surf
    >>> bool(np.isnan(z[~mask]).all())
    True
    """
    rng = np.random.default_rng(seed)
    x, y, r, theta, mask = make_wafer_grid(n_pixels, diameter, edge_exclusion)
    radius = 0.5 * diameter
    pixel_size = diameter / n_pixels
    rho = r / radius

    z = parabolic_bow(rho, bow_sag)
    z += warp_lobes(rho, theta, warp_amplitude, rng)
    z += tilt[0] * (x / radius) + tilt[1] * (y / radius)
    z += nanotopography(z.shape, pixel_size, nanotopo_rms, nanotopo_correlation, rng)

    defects = _add_particles(
        z, x, y, radius, n_particles, particle_height, particle_sigma, rng
    )
    defects += _add_scratches(
        z, x, y, radius, n_scratches, scratch_depth, scratch_width, scratch_length, rng
    )

    surface = WaferSurface(
        x=x, y=y, z=apply_mask(z, mask), mask=mask,
        radius=radius, pixel_size=pixel_size, defects=defects,
    )
    logger.info(
        "Synthesised wafer: %d px, PV %.2f µm, RMS %.2f µm, %d defects",
        n_pixels, surface.pv() * 1e6, surface.rms() * 1e6, len(defects),
    )
    return surface


def synthesize_wafer_pair(
    n_pixels: int = 512,
    diameter: float = WAFER_DIAMETER_300MM,
    *,
    nominal_thickness: float = WAFER_THICKNESS_300MM,
    ttv_amplitude: float = 1.2e-6,
    thickness_noise_rms: float = 40e-9,
    seed: Optional[int] = 0,
    **surface_kwargs,
) -> WaferPair:
    """Generate front and back wafer surfaces with a thickness variation.

    The shape (bow + warp + nanotopography + defects) is carried by the median
    surface; a separate low-order thickness field is then split symmetrically
    about it, so shape and thickness are independent -- exactly the separation
    SEMI M1 relies on when it defines bow and warp on the median surface but
    TTV on the thickness.

    Parameters
    ----------
    n_pixels : int
        Samples along one axis.
    diameter : float
        Wafer diameter [m].
    nominal_thickness : float
        Mean wafer thickness [m].
    ttv_amplitude : float
        Peak-to-valley of the deterministic (wedge + bowl) thickness
        variation [m].
    thickness_noise_rms : float
        RMS of the smooth random component of the thickness variation [m].
    seed : int, optional
        RNG seed.  The median surface uses *seed*; the thickness field uses a
        derived, distinct stream so the two are independent.
    **surface_kwargs
        Forwarded to :func:`synthesize_wafer` (``bow_sag``, ``warp_amplitude``,
        ``n_particles``, ``edge_exclusion``, ...).

    Returns
    -------
    WaferPair
        Front/back surface pair.

    Examples
    --------
    >>> pair = synthesize_wafer_pair(n_pixels=128, seed=2)
    >>> bool(np.all(pair.thickness[pair.mask] > 0))
    True
    """
    median_surface = synthesize_wafer(n_pixels, diameter, seed=seed, **surface_kwargs)
    x, y, mask = median_surface.x, median_surface.y, median_surface.mask
    radius = median_surface.radius
    rho_x, rho_y = x / radius, y / radius

    # Distinct RNG stream so thickness is uncorrelated with shape.
    t_rng = np.random.default_rng(None if seed is None else seed + 104_729)
    wedge = t_rng.uniform(-1.0, 1.0, size=2)
    thickness_shape = wedge[0] * rho_x + wedge[1] * rho_y + 0.5 * (rho_x**2 + rho_y**2)
    span = float(np.ptp(thickness_shape[mask]))
    if span > 0.0:
        thickness_shape = thickness_shape * (ttv_amplitude / span)
    thickness_shape = thickness_shape - float(np.mean(thickness_shape[mask]))
    thickness_shape += nanotopography(
        x.shape, median_surface.pixel_size, thickness_noise_rms, 8e-3, t_rng
    )

    thickness = nominal_thickness + thickness_shape
    median_z = np.where(mask, np.nan_to_num(median_surface.z, nan=0.0), 0.0)

    pair = WaferPair(
        x=x, y=y,
        z_front=apply_mask(median_z + 0.5 * thickness, mask),
        z_back=apply_mask(median_z - 0.5 * thickness, mask),
        mask=mask,
        radius=radius,
        pixel_size=median_surface.pixel_size,
        defects=list(median_surface.defects),
    )
    logger.info(
        "Synthesised wafer pair: mean thickness %.1f µm, TTV %.3f µm",
        float(np.nanmean(pair.thickness)) * 1e6,
        float(np.nanmax(pair.thickness) - np.nanmin(pair.thickness)) * 1e6,
    )
    return pair
