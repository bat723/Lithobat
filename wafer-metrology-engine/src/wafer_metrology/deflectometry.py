"""
Phase-measuring deflectometry (PMD): fringe-reflection metrology of a wafer.

This is the Tier-0 software engine for the hardware build: a camera views the
reflection of a fringe screen *in* the specular wafer, so what the fringes
encode is surface **slope**, not height.  A local slope ``s`` deflects the
reflected ray by ``2s``; over the screen distance ``d_s`` that walks the
observed screen point by ``2 d_s s``, giving a fringe-phase shift of

    delta_phi = 4 * pi * d_s * s / p

for fringe period ``p``.  Height comes back by integrating the two measured
slope maps -- which is why deflectometry needs no reference optic and works
with a laptop screen, and also why its low-order terms (bow) inherit any
geometry-calibration error.

Forward model (simplified, documented)
--------------------------------------
Camera far from the wafer, viewing near-normal; each pixel maps to a wafer
point ``(x, y)`` directly, and the screen is a parallel plane at ``d_s``.  The
screen coordinate seen at that pixel is ``u = x + 2 d_s (s_x + e_x)``, where
``e_x`` is the slope of an optional *system* height field fixed in the
instrument frame (screen bow, geometry error).  Displayed sinusoids pass
through a programmable display gamma; the camera adds Gaussian noise.
Perspective, oblique viewing and screen pose enter a real rig through the
checkerboard/pose calibration and are absorbed here into the system field.

Inverse pipeline
----------------
N-step phase extraction (the general least-squares estimator from
:mod:`.interferometry`) -> hierarchical multi-period temporal unwrap -> slope
maps ``(s_x, s_y)`` -> zonal least-squares integration on the masked aperture
-> height map.

Calibration
-----------
:func:`reversal_calibrate` implements the rotate-180-and-remeasure
decomposition -- and its docstring states the parity limit precisely: only the
rotation-**odd** part of the system error separates; the rotation-even part
(power, astigmatism, spherical -- exactly what a bowed screen produces) stays
in the wafer estimate.  :func:`absolute_calibrate` closes that hole by
subtracting a measurement of the reference flat, which captures the full
system error regardless of parity.  Both are demonstrated numerically in the
pipeline and pinned by tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import sparse
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.sparse.linalg import spsolve

from .flatness import fit_reference_plane
from .interferometry import recover_phase, wrap
from .plotting import annotate_stats, plot_wafer_panels, wafer_imshow
from .synthesize import DefectTruth, WaferSurface, apply_mask

logger = logging.getLogger(__name__)

DEFAULT_SCREEN_DISTANCE: float = 0.7
"""Screen-to-wafer distance [m] matching the planned bench (~0.5-0.8 m)."""

DEFAULT_PERIODS: Tuple[float, ...] = (0.35, 0.08, 0.015)
"""Fringe periods [m], coarse to fine.  The coarsest must exceed the full
screen-coordinate span so its phase is absolute; the finest (15 mm, the plan's
working period) sets the slope resolution."""

DEFAULT_GAMMA: float = 2.2
"""Typical uncorrected display gamma."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _nearest_fill(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Extend *values* outside *mask* by nearest-neighbour fill."""
    if mask.all():
        return np.nan_to_num(values, nan=0.0)
    _, indices = distance_transform_edt(~mask, return_indices=True)
    filled = np.nan_to_num(values, nan=0.0)
    return filled[tuple(indices)]


def surface_slopes(
    z: NDArray[np.float64],
    mask: NDArray[np.bool_],
    pixel_size: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Compute the slope fields of a masked height map.

    The map is nearest-extended before differencing so the aperture edge does
    not produce spurious gradients, and the validity mask is eroded by one
    pixel because the boundary ring's central differences still straddle the
    fill region.

    Parameters
    ----------
    z : NDArray
        Height map [m], ``NaN`` outside *mask*.
    mask : NDArray of bool
        Valid-pixel mask.
    pixel_size : float
        Grid pitch [m].

    Returns
    -------
    (sx, sy, inner)
        Slope fields ``dz/dx`` and ``dz/dy`` [rad], ``NaN`` outside *inner*,
        and the eroded validity mask.
    """
    filled = _nearest_fill(z, mask)
    sy, sx = np.gradient(filled, pixel_size)
    inner = binary_erosion(mask, iterations=1)
    return apply_mask(sx, inner), apply_mask(sy, inner), inner


def rotate_surface_180(surface: WaferSurface) -> WaferSurface:
    """Return the wafer physically rotated 180 degrees in its mount.

    The height map is rotated on the (symmetric) grid and the defect ground
    truth follows, so a rotated measurement can still be scored.

    Parameters
    ----------
    surface : WaferSurface
        Surface to rotate.

    Returns
    -------
    WaferSurface
        Rotated surface on the same grid.
    """
    rotated = surface.with_z(np.rot90(surface.z, 2))
    rotated.defects = [
        DefectTruth(d.kind, -d.x, -d.y, d.height, d.size) for d in surface.defects
    ]
    return rotated


# ---------------------------------------------------------------------------
# Display gamma
# ---------------------------------------------------------------------------


def simulate_gamma_sweep(
    gamma: float,
    n_levels: int = 16,
    noise: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate the gray-level sweep used to calibrate display gamma.

    Parameters
    ----------
    gamma : float
        True display gamma.
    n_levels : int
        Number of commanded gray levels in ``(0, 1]``.
    noise : float
        Gaussian capture noise (fraction of full scale).
    seed : int, optional
        RNG seed.

    Returns
    -------
    (commanded, captured)
        Commanded levels and the captured response ``commanded**gamma + noise``.
    """
    rng = np.random.default_rng(seed)
    commanded = np.linspace(1.0 / n_levels, 1.0, n_levels)
    captured = commanded**gamma + rng.normal(0.0, noise, size=n_levels)
    return commanded, np.clip(captured, 1e-6, None)


def fit_gamma(
    commanded: NDArray[np.float64], captured: NDArray[np.float64]
) -> float:
    """Fit a power-law display response ``captured = commanded**gamma``.

    Least squares in log-log space, constrained through the origin, with each
    level weighted by its captured signal.  The weighting matters: the darkest
    levels of a real sweep sit at the camera noise floor, and in log space
    their errors are enormous -- an unweighted fit is biased by exactly the
    points that carry the least information.

    Parameters
    ----------
    commanded : NDArray
        Commanded gray levels in ``(0, 1]``.
    captured : NDArray
        Captured (normalised) response.

    Returns
    -------
    float
        Estimated gamma.
    """
    commanded = np.asarray(commanded, dtype=np.float64)
    captured = np.asarray(captured, dtype=np.float64)
    good = (commanded > 0) & (commanded < 1) & (captured > 0)
    lc = np.log(commanded[good])
    lm = np.log(captured[good])
    weight = captured[good] ** 2
    return float(np.sum(weight * lc * lm) / np.sum(weight * lc * lc))


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


@dataclass
class DeflectometrySetup:
    """Geometry and acquisition parameters of the deflectometry rig.

    Attributes
    ----------
    screen_distance : float
        Wafer-to-screen distance ``d_s`` [m].
    periods : tuple of float
        Fringe periods [m], coarse to fine.  The coarsest must exceed the full
        screen-coordinate span so its phase is unambiguous.
    n_steps : int
        Phase steps per period (8 in the plan; suppresses gamma harmonics).
    gamma : float
        True display gamma applied by the simulated screen.
    lut_gamma : float, optional
        Calibrated gamma used to pre-distort the commanded pattern (the
        inverse LUT).  ``None`` displays the raw sinusoid through *gamma*.
    noise : float
        Camera noise standard deviation as a fraction of full scale.
    screen_half_width : float
        Half-extent of the physical screen [m]; a warning is logged when the
        reflected pattern walks off it.
    system_zmap : NDArray, optional
        Height field [m] fixed in the *instrument* frame whose slopes are
        added to every measurement -- screen bow, pose error.  This is the
        systematic that calibration must remove, injectable so the engine can
        prove what each calibration method actually removes.
    """

    screen_distance: float = DEFAULT_SCREEN_DISTANCE
    periods: Tuple[float, ...] = DEFAULT_PERIODS
    n_steps: int = 8
    gamma: float = DEFAULT_GAMMA
    lut_gamma: Optional[float] = None
    noise: float = 0.005
    screen_half_width: float = 0.165
    system_zmap: Optional[NDArray[np.float64]] = None


@dataclass
class Deflectograms:
    """A captured deflectometry sequence: both directions, every period.

    Attributes
    ----------
    frames_x, frames_y : list of NDArray
        One ``(n_steps, ny, nx)`` stack per period, fringes along x / y.
    deltas : NDArray
        Nominal phase steps [rad].
    setup : DeflectometrySetup
        Acquisition parameters used.
    x, y : NDArray
        Wafer coordinate grids [m].
    mask : NDArray of bool
        Validity mask (aperture eroded by one pixel for the slope fields).
    """

    frames_x: List[NDArray[np.float64]]
    frames_y: List[NDArray[np.float64]]
    deltas: NDArray[np.float64]
    setup: DeflectometrySetup
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    mask: NDArray[np.bool_]


def simulate_deflectograms(
    surface: WaferSurface,
    setup: Optional[DeflectometrySetup] = None,
    *,
    seed: Optional[int] = None,
) -> Deflectograms:
    """Render the full fringe-reflection sequence for a wafer.

    Parameters
    ----------
    surface : WaferSurface
        Wafer under test.
    setup : DeflectometrySetup, optional
        Rig parameters; defaults to the plan-matched rig.
    seed : int, optional
        RNG seed for camera noise.

    Returns
    -------
    Deflectograms
        Frame stacks for both fringe directions and every period.
    """
    if setup is None:
        setup = DeflectometrySetup()
    rng = np.random.default_rng(seed)

    sx, sy, inner = surface_slopes(surface.z, surface.mask, surface.pixel_size)
    if setup.system_zmap is not None:
        ex, ey, inner_sys = surface_slopes(
            setup.system_zmap, surface.mask, surface.pixel_size
        )
        inner = inner & inner_sys
        sx = sx + ex
        sy = sy + ey

    d_s = setup.screen_distance
    u = surface.x + 2.0 * d_s * sx
    v = surface.y + 2.0 * d_s * sy

    span = max(float(np.nanmax(np.abs(u))), float(np.nanmax(np.abs(v))))
    if span > setup.screen_half_width:
        logger.warning(
            "Reflected pattern spans ±%.0f mm but the screen is ±%.0f mm -- "
            "increase screen size or distance.",
            span * 1e3, setup.screen_half_width * 1e3,
        )
    if setup.periods[0] < 2.0 * span:
        logger.warning(
            "Coarsest period %.0f mm < twice the pattern span %.0f mm: the "
            "temporal unwrap loses its absolute reference.",
            setup.periods[0] * 1e3, 2.0 * span * 1e3,
        )

    deltas = 2.0 * np.pi * np.arange(setup.n_steps) / setup.n_steps
    exponent = setup.gamma / setup.lut_gamma if setup.lut_gamma else setup.gamma

    def render(coord: NDArray[np.float64]) -> List[NDArray[np.float64]]:
        stacks = []
        for period in setup.periods:
            frames = np.empty((setup.n_steps, *coord.shape), dtype=np.float64)
            for k, delta in enumerate(deltas):
                ideal = 0.5 * (1.0 + np.cos(2.0 * np.pi * coord / period + delta))
                displayed = ideal**exponent
                frames[k] = displayed + rng.normal(0.0, setup.noise, size=coord.shape)
            frames[:, ~inner] = np.nan
            stacks.append(frames)
        return stacks

    return Deflectograms(
        frames_x=render(u), frames_y=render(v), deltas=deltas,
        setup=setup, x=surface.x, y=surface.y, mask=inner,
    )


# ---------------------------------------------------------------------------
# Inverse pipeline
# ---------------------------------------------------------------------------


def unwrap_temporal(
    phases: Sequence[NDArray[np.float64]], periods: Sequence[float]
) -> NDArray[np.float64]:
    """Hierarchical multi-period (temporal) phase unwrap.

    The coarsest period's phase is treated as absolute (it must span the full
    coordinate range), and each finer period only resolves its own fringe
    order against the running estimate -- so no spatial unwrapping is needed
    and isolated bad pixels cannot propagate.

    Parameters
    ----------
    phases : sequence of NDArray
        Wrapped phases [rad], one per period, coarse to fine.
    periods : sequence of float
        Fringe periods [m], strictly decreasing, aligned with *phases*.

    Returns
    -------
    NDArray
        Absolute screen coordinate [m] from the finest period.

    Raises
    ------
    ValueError
        If lengths differ or periods are not strictly decreasing.
    """
    if len(phases) != len(periods):
        raise ValueError(f"{len(phases)} phase maps for {len(periods)} periods")
    if any(b >= a for a, b in zip(periods, periods[1:])):
        raise ValueError(f"periods must be strictly decreasing, got {tuple(periods)}")

    coordinate = periods[0] * phases[0] / (2.0 * np.pi)
    for period, phase in zip(periods[1:], phases[1:]):
        fine = period * phase / (2.0 * np.pi)
        order = np.round((coordinate - fine) / period)
        coordinate = fine + order * period
    return coordinate


def reconstruct_slopes(
    deflectograms: Deflectograms,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Recover the two slope maps from a deflectogram sequence.

    Parameters
    ----------
    deflectograms : Deflectograms
        Captured sequence.

    Returns
    -------
    (sx, sy)
        Slope maps [rad], ``NaN`` outside the mask.
    """
    setup = deflectograms.setup
    d_s = setup.screen_distance

    def one_direction(stacks: List[NDArray], base: NDArray) -> NDArray[np.float64]:
        phases = [recover_phase(s, deflectograms.deltas) for s in stacks]
        coordinate = unwrap_temporal(phases, setup.periods)
        return apply_mask((coordinate - base) / (2.0 * d_s), deflectograms.mask)

    sx = one_direction(deflectograms.frames_x, deflectograms.x)
    sy = one_direction(deflectograms.frames_y, deflectograms.y)
    return sx, sy


def integrate_slopes(
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    mask: NDArray[np.bool_],
    pixel_size: float,
) -> NDArray[np.float64]:
    """Zonal least-squares integration of slope fields on a masked aperture.

    Every adjacent in-mask pixel pair contributes one equation
    ``z_j - z_i = h * (s_i + s_j) / 2`` (trapezoid rule); the stacked system is
    solved through its normal equations, which form the masked graph Laplacian
    plus a single gauge anchor -- sparse, SPD, and exact on an arbitrary mask,
    unlike FFT integrators that assume a periodic rectangle.

    Parameters
    ----------
    sx, sy : NDArray
        Slope fields [rad]; non-finite pixels are excluded.
    mask : NDArray of bool
        Aperture mask.
    pixel_size : float
        Grid pitch [m].

    Returns
    -------
    NDArray
        Height map [m], zero-mean over the valid area, ``NaN`` elsewhere.
        The absolute piston is not observable from slopes; tilt is observable
        and preserved.

    Raises
    ------
    ValueError
        If fewer than three valid pixels remain.

    Notes
    -----
    Assumes the valid region is connected (a wafer aperture is); a
    disconnected region would leave the far component pinned only through the
    least-squares gauge and should be integrated per component.
    """
    valid = np.asarray(mask, dtype=bool) & np.isfinite(sx) & np.isfinite(sy)
    n_unknown = int(valid.sum())
    if n_unknown < 3:
        raise ValueError("need at least 3 valid pixels to integrate slopes")

    index = np.full(valid.shape, -1, dtype=np.int64)
    index[valid] = np.arange(n_unknown)

    horizontal = valid[:, :-1] & valid[:, 1:]
    i_h = index[:, :-1][horizontal]
    j_h = index[:, 1:][horizontal]
    b_h = 0.5 * (sx[:, :-1][horizontal] + sx[:, 1:][horizontal]) * pixel_size

    vertical = valid[:-1, :] & valid[1:, :]
    i_v = index[:-1, :][vertical]
    j_v = index[1:, :][vertical]
    b_v = 0.5 * (sy[:-1, :][vertical] + sy[1:, :][vertical]) * pixel_size

    i_all = np.concatenate([i_h, i_v])
    j_all = np.concatenate([j_h, j_v])
    b_all = np.concatenate([b_h, b_v])
    n_eq = i_all.size

    rows = np.concatenate([np.arange(n_eq), np.arange(n_eq), [n_eq]])
    cols = np.concatenate([j_all, i_all, [0]])
    data = np.concatenate([np.ones(n_eq), -np.ones(n_eq), [1.0]])
    system = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_eq + 1, n_unknown)
    ).tocsr()
    rhs = np.concatenate([b_all, [0.0]])

    normal = (system.T @ system).tocsc()
    solution = spsolve(normal, system.T @ rhs)

    out = np.full(valid.shape, np.nan, dtype=np.float64)
    out[valid] = solution - solution.mean()
    return out


# ---------------------------------------------------------------------------
# End-to-end measurement
# ---------------------------------------------------------------------------


@dataclass
class DeflectometryMeasurement:
    """Result of one simulated deflectometry measurement.

    Attributes
    ----------
    z_measured : NDArray
        Integrated height map [m], zero-mean (piston is unobservable).
        Includes any injected system error -- this is the *raw* measurement.
    sx, sy : NDArray
        Recovered slope maps [rad].
    mask : NDArray of bool
        Validity mask of the reconstruction.
    wrapped_fine_x : NDArray
        Wrapped phase of the finest x-direction fringe set [rad] (for plots).
    error : NDArray
        ``z_measured - z_true`` after best-fit-plane alignment [m].
    rms_error : float
        RMS of *error* [m]; includes the uncorrected system error.
    pv_error : float
        Peak-to-valley of *error* [m].
    deflectograms : Deflectograms
        The captured sequence.
    """

    z_measured: NDArray[np.float64]
    sx: NDArray[np.float64]
    sy: NDArray[np.float64]
    mask: NDArray[np.bool_]
    wrapped_fine_x: NDArray[np.float64]
    error: NDArray[np.float64]
    rms_error: float
    pv_error: float
    deflectograms: Deflectograms = field(repr=False)


def plane_aligned_difference(
    measured: NDArray[np.float64],
    reference: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Difference two maps after removing their best-fit relative plane.

    Deflectometry loses absolute piston (and a real rig's tilt datum), so maps
    are only comparable modulo a plane -- the same convention every plane-
    referenced SEMI metric already uses.

    Parameters
    ----------
    measured, reference : NDArray
        Height maps [m].
    x, y : NDArray
        Coordinate grids [m].
    mask : NDArray of bool
        Comparison domain.

    Returns
    -------
    NDArray
        ``measured - reference`` with its best-fit plane removed, ``NaN``
        outside *mask*.
    """
    difference = measured - reference
    both = mask & np.isfinite(difference)
    plane, _ = fit_reference_plane(x, y, difference, both)
    return apply_mask(difference - plane, both)


def measure_deflectometry(
    surface: WaferSurface,
    setup: Optional[DeflectometrySetup] = None,
    *,
    seed: Optional[int] = None,
) -> DeflectometryMeasurement:
    """Run a full simulated deflectometry measurement and score it.

    Parameters
    ----------
    surface : WaferSurface
        Wafer under test.
    setup : DeflectometrySetup, optional
        Rig parameters.
    seed : int, optional
        RNG seed for camera noise.

    Returns
    -------
    DeflectometryMeasurement
        Reconstruction plus error statistics against the ground truth.  When
        *setup* carries a ``system_zmap``, that systematic is part of the
        error on purpose -- calibration, not averaging, must remove it.

    Examples
    --------
    >>> from .synthesize import synthesize_wafer
    >>> s = synthesize_wafer(n_pixels=128, diameter=150e-3, seed=0)
    >>> m = measure_deflectometry(s, DeflectometrySetup(noise=0.0), seed=1)
    >>> bool(m.rms_error < 50e-9)
    True
    """
    deflectograms = simulate_deflectograms(surface, setup, seed=seed)
    sx, sy = reconstruct_slopes(deflectograms)
    z = integrate_slopes(sx, sy, deflectograms.mask, surface.pixel_size)

    error = plane_aligned_difference(
        z, surface.z, surface.x, surface.y, deflectograms.mask
    )
    rms = float(np.sqrt(np.nanmean(error[deflectograms.mask] ** 2)))
    pv = float(np.nanmax(error) - np.nanmin(error))

    wrapped = recover_phase(deflectograms.frames_x[-1], deflectograms.deltas)
    measurement = DeflectometryMeasurement(
        z_measured=z, sx=sx, sy=sy, mask=deflectograms.mask,
        wrapped_fine_x=apply_mask(wrapped, deflectograms.mask),
        error=error, rms_error=rms, pv_error=pv, deflectograms=deflectograms,
    )
    logger.info(
        "Deflectometry: d_s=%.2f m, periods %s mm, %d steps -> RMS error %.1f nm "
        "(incl. any injected system error)",
        deflectograms.setup.screen_distance,
        [round(p * 1e3) for p in deflectograms.setup.periods],
        deflectograms.setup.n_steps, rms * 1e9,
    )
    return measurement


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def reversal_calibrate(
    measured_0: NDArray[np.float64],
    measured_180: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotation-reversal decomposition of two measurements of one wafer.

    With ``M_0 = W + S`` and ``M_180 = rot180(W) + S`` (wafer *W* rotated in
    the mount, system error *S* fixed in the instrument frame):

        wafer_est  = (M_0 + rot180(M_180)) / 2 = W + (S + rot180(S)) / 2
        system_est = (M_0 - rot180(M_180)) / 2 =     (S - rot180(S)) / 2

    **Parity limit (important):** only the rotation-*odd* part of *S* lands in
    ``system_est``.  Zernike terms with even azimuthal order -- power,
    astigmatism, spherical, precisely the shape of a bowed screen -- are
    invariant under a 180-degree rotation and stay inside ``wafer_est``.
    Single-rotation reversal therefore bounds and removes half the systematic,
    not all of it; use :func:`absolute_calibrate` with a reference-flat
    measurement for the even half, and keep reversal as a cross-check on the
    odd half and on mount repeatability.

    Parameters
    ----------
    measured_0 : NDArray
        Height map with the wafer at 0 degrees [m].
    measured_180 : NDArray
        Height map with the wafer physically rotated 180 degrees [m].
    mask : NDArray of bool
        Validity mask (the centred wafer aperture is 180-degree symmetric).

    Returns
    -------
    (wafer_est, system_est)
        The wafer-shape estimate (still carrying the rotation-even system
        error) and the rotation-odd system error estimate, both ``NaN``
        outside the common mask.
    """
    rotated = np.rot90(measured_180, 2)
    common = mask & np.rot90(mask, 2) & np.isfinite(measured_0) & np.isfinite(rotated)
    wafer_est = apply_mask(0.5 * (measured_0 + rotated), common)
    system_est = apply_mask(0.5 * (measured_0 - rotated), common)
    return wafer_est, system_est


def absolute_calibrate(
    measured: NDArray[np.float64],
    flat_measured: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Subtract a reference-flat measurement as the instrument systematic.

    A front-surface lambda/10 flat measured in the wafer position returns (to
    within its own flatness) the full fixed system error *S* -- both parity
    halves, unlike :func:`reversal_calibrate`.  Subtracting it is the absolute
    calibration; on the bench it should be captured every session, before the
    wafer.

    Parameters
    ----------
    measured : NDArray
        Wafer measurement ``W + S`` [m].
    flat_measured : NDArray
        Reference-flat measurement ``~S`` [m] from the same rig state.

    Returns
    -------
    NDArray
        ``measured - flat_measured``; ``NaN`` wherever either input is.
    """
    return measured - flat_measured


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_deflectograms(
    surface: WaferSurface,
    measurement: DeflectometryMeasurement,
    *,
    title: str = "Phase-measuring deflectometry",
    figsize: Tuple[float, float] = (13.5, 4.4),
) -> Tuple[Figure, NDArray]:
    """Show a captured fringe frame, the wrapped phase, and a slope map.

    Parameters
    ----------
    surface : WaferSurface
        Wafer measured (supplies the grid).
    measurement : DeflectometryMeasurement
        Measurement to display.
    title : str
        Figure suptitle.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    import matplotlib.pyplot as plt

    setup = measurement.deflectograms.setup
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    wafer_imshow(
        axes[0], surface.x, surface.y, measurement.deflectograms.frames_x[-1][0],
        scale=1.0, label="Intensity [a.u.]", signed=False, cmap="gray",
        radius=surface.radius,
        title=f"Reflected fringes (p = {setup.periods[-1] * 1e3:.0f} mm, step 1/{setup.n_steps})",
    )
    wafer_imshow(
        axes[1], surface.x, surface.y, measurement.wrapped_fine_x,
        scale=1.0, label="Phase [rad]", signed=True, vlim=(-np.pi, np.pi),
        radius=surface.radius, title="Wrapped phase (finest period, x)",
    )
    wafer_imshow(
        axes[2], surface.x, surface.y, measurement.sx,
        scale=1e6, label="Slope [µrad]", signed=True,
        radius=surface.radius, title="Recovered slope $s_x$",
    )
    fig.suptitle(
        f"{title} — $d_s$ = {setup.screen_distance:.2f} m, "
        f"γ = {setup.gamma:.2f}"
        + (f" (LUT γ̂ = {setup.lut_gamma:.3f})" if setup.lut_gamma else " (no LUT)"),
        fontsize=12,
    )
    fig.tight_layout()
    return fig, axes


def plot_deflectometry_recovery(
    surface: WaferSurface,
    measurement: DeflectometryMeasurement,
    *,
    title: str = "Deflectometry height recovery",
) -> Tuple[Figure, NDArray]:
    """Truth, integrated height, and plane-aligned error, side by side.

    Parameters
    ----------
    surface : WaferSurface
        Ground truth.
    measurement : DeflectometryMeasurement
        Measurement to display.
    title : str
        Figure suptitle.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    truth = apply_mask(surface.z, measurement.mask)
    fig, axes = plot_wafer_panels(
        surface.x, surface.y,
        [truth - np.nanmean(truth), measurement.z_measured, measurement.error],
        ["True surface", "Integrated from slopes", "Error (plane-aligned)"],
        scale=1e6, label="Height [µm]", signed=True, radius=surface.radius,
        shared_scale=False, suptitle=title,
    )
    annotate_stats(
        axes[2],
        f"RMS {measurement.rms_error * 1e9:7.1f} nm\n"
        f"PV  {measurement.pv_error * 1e9:7.1f} nm",
        loc="lower left",
    )
    return fig, axes


def plot_reversal(
    surface: WaferSurface,
    wafer_est: NDArray[np.float64],
    system_est: NDArray[np.float64],
    system_true: NDArray[np.float64],
    absolute: NDArray[np.float64],
    *,
    title: str = "Reversal vs. absolute calibration",
) -> Tuple[Figure, NDArray]:
    """Four panels: injected system, reversal's two outputs, absolute residual.

    Parameters
    ----------
    surface : WaferSurface
        Wafer measured (supplies the grid and truth).
    wafer_est : NDArray
        Reversal wafer estimate [m].
    system_est : NDArray
        Reversal system estimate (rotation-odd part only) [m].
    system_true : NDArray
        Injected system error [m].
    absolute : NDArray
        Wafer estimate after reference-flat subtraction [m].
    title : str
        Figure suptitle.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    mask = np.isfinite(wafer_est)
    reversal_err = plane_aligned_difference(
        wafer_est, surface.z, surface.x, surface.y, mask
    )
    absolute_err = plane_aligned_difference(
        absolute, surface.z, surface.x, surface.y, np.isfinite(absolute)
    )
    fig, axes = plot_wafer_panels(
        surface.x, surface.y,
        [system_true, system_est, reversal_err, absolute_err],
        [
            "Injected system error\n(screen bow + pose)",
            "Reversal system estimate\n(rotation-odd part only)",
            "Reversal wafer error\n(even part leaks through)",
            "Absolute (flat-subtracted)\nwafer error",
        ],
        scale=1e6, label="Height [µm]", signed=True, radius=surface.radius,
        shared_scale=False, suptitle=title,
    )
    for ax, arr in zip(axes[2:], (reversal_err, absolute_err)):
        annotate_stats(
            ax, f"RMS {np.sqrt(np.nanmean(arr**2)) * 1e9:6.1f} nm", loc="lower left"
        )
    return fig, axes
