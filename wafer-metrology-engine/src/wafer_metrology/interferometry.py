"""
Phase-shifting interferometry (PSI) simulation and surface reconstruction.

Forward model
-------------
A reflection interferometer sees a round-trip optical path, so surface height
maps to phase as ``phi = 4 * pi * z / wavelength``.  The detector records a
sequence of intensity frames while a reference mirror is stepped::

    I_k = I0 * (1 + V * cos(phi + delta_k)) + noise

with nominal steps ``delta_k = 2 * pi * k / n_steps``.  Real piezo actuators
are miscalibrated, so an optional *step error* scales the actual steps -- the
dominant systematic error in PSI, and the one this module lets a DOE study
excite deliberately.

Inverse model
-------------
The N-step least-squares estimator recovers phase modulo 2 pi::

    phi_wrapped = atan2(-sum_k I_k sin(delta_k), sum_k I_k cos(delta_k))

Phase is then unwrapped and converted back to height.  Unwrapping uses
:func:`skimage.restoration.unwrap_phase` when scikit-image is installed and
falls back to a bundled DCT least-squares (Ghiglia-Romero) unwrapper otherwise,
so the package has no hard dependency on scikit-image.

Sampling limit
--------------
Unwrapping only works while the phase changes by less than ``pi`` between
adjacent pixels, i.e. while the height changes by less than
``wavelength / 4`` per pixel.  A 300 mm wafer with tens of microns of warp
violates this badly at 633 nm -- which is exactly why real tools measure such
wafers with a **synthetic wavelength** from two close laser lines
(:func:`synthetic_wavelength`), trading height resolution for unambiguous
range.  :func:`check_sampling` reports the margin, and :func:`measure_surface`
warns when it is exceeded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.fft import dctn, idctn
from scipy.ndimage import distance_transform_edt

from . import HENE_WAVELENGTH
from .plotting import CMAP_MAGNITUDE, annotate_stats, plot_wafer_panels, wafer_imshow
from .synthesize import WaferSurface, apply_mask

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whichever branch is installed
    from skimage.restoration import unwrap_phase as _skimage_unwrap

    HAVE_SKIMAGE = True
except ImportError:  # pragma: no cover
    _skimage_unwrap = None
    HAVE_SKIMAGE = False


# ---------------------------------------------------------------------------
# Height <-> phase
# ---------------------------------------------------------------------------


def height_to_phase(z: NDArray[np.float64], wavelength: float = HENE_WAVELENGTH) -> NDArray[np.float64]:
    """Convert surface height to round-trip optical phase.

    Parameters
    ----------
    z : NDArray
        Height map [m].
    wavelength : float
        Illumination (or synthetic) wavelength [m].

    Returns
    -------
    NDArray
        Phase [rad], ``4 * pi * z / wavelength``.
    """
    return 4.0 * np.pi * np.asarray(z, dtype=np.float64) / wavelength


def phase_to_height(phi: NDArray[np.float64], wavelength: float = HENE_WAVELENGTH) -> NDArray[np.float64]:
    """Convert round-trip optical phase back to surface height.

    Parameters
    ----------
    phi : NDArray
        Phase [rad].
    wavelength : float
        Illumination (or synthetic) wavelength [m].

    Returns
    -------
    NDArray
        Height [m], ``wavelength * phi / (4 * pi)``.
    """
    return wavelength * np.asarray(phi, dtype=np.float64) / (4.0 * np.pi)


def wrap(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap phase into ``(-pi, pi]``.

    Parameters
    ----------
    phi : NDArray
        Phase [rad].

    Returns
    -------
    NDArray
        Wrapped phase [rad].
    """
    return np.angle(np.exp(1j * np.asarray(phi, dtype=np.float64)))


def synthetic_wavelength(lambda1: float, lambda2: float) -> float:
    """Return the synthetic wavelength of a two-wavelength measurement.

    ``Lambda = lambda1 * lambda2 / |lambda1 - lambda2|``.  Two closely spaced
    lines behave like one much longer wavelength, extending the unambiguous
    range by ``Lambda / lambda`` at the cost of amplifying phase noise by the
    same factor.

    Parameters
    ----------
    lambda1, lambda2 : float
        Source wavelengths [m]; must differ.

    Returns
    -------
    float
        Synthetic wavelength [m].

    Raises
    ------
    ValueError
        If the two wavelengths are equal.

    Examples
    --------
    >>> round(synthetic_wavelength(632.8e-9, 640.0e-9) * 1e6, 1)
    56.3
    """
    if lambda1 == lambda2:
        raise ValueError("synthetic wavelength requires two distinct wavelengths")
    return abs(lambda1 * lambda2 / (lambda1 - lambda2))


def check_sampling(
    z: NDArray[np.float64],
    wavelength: float,
    mask: Optional[NDArray[np.bool_]] = None,
) -> Tuple[float, bool]:
    """Report the worst pixel-to-pixel phase step and whether unwrapping is safe.

    Parameters
    ----------
    z : NDArray
        Height map [m].
    wavelength : float
        Wavelength [m].
    mask : NDArray of bool, optional
        Valid-pixel mask; gradients are evaluated inside it only.

    Returns
    -------
    (float, bool)
        Maximum adjacent-pixel phase step [rad], and ``True`` when it stays
        below ``pi`` (the unwrapping limit).
    """
    phi = height_to_phase(z, wavelength)
    if mask is not None:
        phi = np.where(mask, phi, np.nan)
    with np.errstate(invalid="ignore"):
        gx = np.abs(np.diff(phi, axis=1))
        gy = np.abs(np.diff(phi, axis=0))
    worst = float(np.nanmax([np.nanmax(gx), np.nanmax(gy)]))
    return worst, worst < np.pi


# ---------------------------------------------------------------------------
# Forward model: fringe generation
# ---------------------------------------------------------------------------


@dataclass
class PSIFrames:
    """A phase-shifted interferogram sequence.

    Attributes
    ----------
    frames : NDArray
        Intensity frames, shape ``(n_steps, ny, nx)``, ``NaN`` outside the mask.
    deltas_nominal : NDArray
        Phase steps the algorithm assumes [rad].
    deltas_actual : NDArray
        Phase steps actually applied [rad]; differs from nominal when a step
        error is simulated.
    wavelength : float
        Wavelength used [m].
    mask : NDArray of bool
        Valid-pixel mask.
    """

    frames: NDArray[np.float64]
    deltas_nominal: NDArray[np.float64]
    deltas_actual: NDArray[np.float64]
    wavelength: float
    mask: NDArray[np.bool_]

    @property
    def n_steps(self) -> int:
        """Number of phase steps."""
        return int(self.frames.shape[0])


def simulate_fringes(
    z: NDArray[np.float64],
    wavelength: float = HENE_WAVELENGTH,
    *,
    n_steps: int = 4,
    visibility: float = 0.9,
    intensity: float = 1.0,
    noise: float = 0.01,
    step_error: float = 0.0,
    mask: Optional[NDArray[np.bool_]] = None,
    seed: Optional[int] = None,
) -> PSIFrames:
    """Render a phase-shifted interferogram sequence for a height map.

    Parameters
    ----------
    z : NDArray
        True surface height [m]; ``NaN`` marks invalid pixels.
    wavelength : float
        Illumination or synthetic wavelength [m].
    n_steps : int
        Number of phase steps, uniformly spanning 2 pi (4 or 5 are standard).
    visibility : float
        Fringe visibility (contrast) in ``[0, 1]``.
    intensity : float
        Mean intensity ``I0`` [a.u.].
    noise : float
        Gaussian detector noise standard deviation, as a fraction of *intensity*.
    step_error : float
        Fractional phase-step calibration error.  ``0.02`` means the actuator
        moves 2 % too far at every step -- the classic PSI systematic, which
        prints as a spurious 2-fringe ripple.
    mask : NDArray of bool, optional
        Valid-pixel mask; defaults to the finite pixels of *z*.
    seed : int, optional
        RNG seed for the detector noise.

    Returns
    -------
    PSIFrames
        The frame stack with both nominal and actual step values.

    Raises
    ------
    ValueError
        If *n_steps* is below 3 (the minimum for a three-unknown solve).
    """
    if n_steps < 3:
        raise ValueError(f"n_steps must be >= 3, got {n_steps}")

    rng = np.random.default_rng(seed)
    if mask is None:
        mask = np.isfinite(z)

    phi = height_to_phase(np.nan_to_num(z, nan=0.0), wavelength)
    deltas_nominal = 2.0 * np.pi * np.arange(n_steps) / n_steps
    deltas_actual = deltas_nominal * (1.0 + step_error)

    frames = np.empty((n_steps, *phi.shape), dtype=np.float64)
    for k, delta in enumerate(deltas_actual):
        ideal = intensity * (1.0 + visibility * np.cos(phi + delta))
        frames[k] = ideal + rng.normal(0.0, noise * intensity, size=phi.shape)
    frames[:, ~mask] = np.nan

    return PSIFrames(
        frames=frames,
        deltas_nominal=deltas_nominal,
        deltas_actual=deltas_actual,
        wavelength=wavelength,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Inverse model: phase recovery and unwrapping
# ---------------------------------------------------------------------------


def recover_phase(
    frames: NDArray[np.float64], deltas: Optional[Sequence[float]] = None
) -> NDArray[np.float64]:
    """Recover wrapped phase from a phase-shifted frame stack.

    Fits ``I_k = a + b_c cos(delta_k) + b_s sin(delta_k)`` by least squares at
    every pixel and returns ``atan2(-b_s, b_c)``.  Solving the general
    three-parameter problem -- rather than assuming uniformly spaced steps --
    makes the estimator exact for *any* full-rank step sequence, so a
    miscalibrated or deliberately non-uniform actuator can be corrected simply
    by passing the steps it really took.  For uniform steps spanning 2 pi this
    reduces to the familiar ``atan2`` of the sine and cosine sums.

    The fitted DC term ``a`` absorbs background intensity, and the least-squares
    solution averages down zero-mean detector noise.

    Parameters
    ----------
    frames : NDArray
        Intensity frames, shape ``(n_steps, ny, nx)``.
    deltas : sequence of float, optional
        Assumed phase steps [rad].  Defaults to uniform ``2 pi k / n_steps``.

    Returns
    -------
    NDArray
        Wrapped phase in ``(-pi, pi]``, shape ``(ny, nx)``.

    Raises
    ------
    ValueError
        If *deltas* does not match the frame count, or the steps are degenerate
        (fewer than three distinct phases, so the fit is rank deficient).
    """
    frames = np.asarray(frames, dtype=np.float64)
    n_steps = frames.shape[0]
    if deltas is None:
        deltas = 2.0 * np.pi * np.arange(n_steps) / n_steps
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size != n_steps:
        raise ValueError(f"got {deltas.size} steps for {n_steps} frames")

    design = np.column_stack([np.ones(n_steps), np.cos(deltas), np.sin(deltas)])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("phase steps are degenerate; need >= 3 distinct phases")

    # One pseudo-inverse for the whole image: the design depends only on steps.
    coefficients = np.tensordot(np.linalg.pinv(design), frames, axes=(1, 0))
    _, b_cos, b_sin = coefficients
    return np.arctan2(-b_sin, b_cos)


def _fill_outside(
    values: NDArray[np.float64], mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Extend *values* outside *mask* by nearest-neighbour fill.

    A nearest-neighbour extension keeps the exterior locally flat, so it adds
    no spurious phase residues that would corrupt the interior solution of the
    least-squares unwrapper.

    Parameters
    ----------
    values : NDArray
        Array with meaningful data inside *mask*.
    mask : NDArray of bool
        ``True`` where data is valid.

    Returns
    -------
    NDArray
        Array with every entry finite.
    """
    if mask.all():
        return np.nan_to_num(values, nan=0.0)
    # Indices of the nearest valid pixel for every invalid pixel.
    _, indices = distance_transform_edt(~mask, return_indices=True)
    filled = np.nan_to_num(values, nan=0.0)
    return filled[tuple(indices)]


def _dct_ls_unwrap(psi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unwrap phase by DCT least-squares (Ghiglia & Romero 1994).

    Solves the Poisson equation ``lap(phi) = lap_wrapped(psi)`` with Neumann
    boundary conditions via a discrete cosine transform, then snaps the result
    onto the lattice congruent with *psi*, which makes the unwrap exact for
    noise-free, well-sampled data.

    Parameters
    ----------
    psi : NDArray
        Wrapped phase [rad], finite everywhere, shape ``(ny, nx)``.

    Returns
    -------
    NDArray
        Unwrapped phase [rad], congruent with *psi* modulo 2 pi.
    """
    ny, nx = psi.shape

    # Wrapped first differences.
    dx = wrap(np.diff(psi, axis=1))
    dy = wrap(np.diff(psi, axis=0))

    # Discrete Laplacian of the wrapped gradient field.
    rho = np.zeros_like(psi)
    rho[:, :-1] += dx
    rho[:, 1:] -= dx
    rho[:-1, :] += dy
    rho[1:, :] -= dy

    # Poisson solve with Neumann BCs: DCT-II diagonalises the operator.
    rho_hat = dctn(rho, type=2, norm="ortho")
    denom = (
        2.0 * (np.cos(np.pi * np.arange(ny) / ny) - 1.0)[:, None]
        + 2.0 * (np.cos(np.pi * np.arange(nx) / nx) - 1.0)[None, :]
    )
    denom[0, 0] = 1.0  # the constant mode is unconstrained; pin it to zero
    phi_hat = rho_hat / denom
    phi_hat[0, 0] = 0.0
    phi = idctn(phi_hat, type=2, norm="ortho")

    # Congruence: keep the least-squares surface's fringe order, but the
    # measured wrapped value within each fringe.
    return psi + 2.0 * np.pi * np.round((phi - psi) / (2.0 * np.pi))


def unwrap_phase_2d(
    psi: NDArray[np.float64],
    mask: Optional[NDArray[np.bool_]] = None,
    prefer_skimage: bool = True,
) -> NDArray[np.float64]:
    """Unwrap a 2-D wrapped phase map.

    Dispatches to :func:`skimage.restoration.unwrap_phase` when scikit-image is
    installed, otherwise to the bundled :func:`_dct_ls_unwrap`.  Both branches
    honour *mask*; the fallback does so by nearest-neighbour extension outside
    the aperture, which is approximate within roughly one pixel of the edge.

    Parameters
    ----------
    psi : NDArray
        Wrapped phase [rad], shape ``(ny, nx)``.  ``NaN`` is permitted outside
        *mask*.
    mask : NDArray of bool, optional
        Valid-pixel mask.  Defaults to every finite pixel.
    prefer_skimage : bool
        Set ``False`` to force the bundled implementation (used by the tests to
        exercise the fallback regardless of what is installed).

    Returns
    -------
    NDArray
        Unwrapped phase [rad], ``NaN`` outside *mask*.
    """
    psi = np.asarray(psi, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(psi)

    if prefer_skimage and HAVE_SKIMAGE:
        masked = np.ma.masked_array(np.nan_to_num(psi, nan=0.0), mask=~mask)
        unwrapped = np.asarray(_skimage_unwrap(masked, seed=0))
    else:
        unwrapped = _dct_ls_unwrap(_fill_outside(psi, mask))

    return apply_mask(unwrapped, mask)


# ---------------------------------------------------------------------------
# End-to-end measurement
# ---------------------------------------------------------------------------


@dataclass
class Measurement:
    """Result of one simulated interferometric measurement.

    Attributes
    ----------
    z_measured : NDArray
        Reconstructed height [m], piston-aligned to the truth when available.
    z_true : NDArray or None
        Ground-truth height [m], when the measurement was made on a synthetic
        surface.
    error : NDArray or None
        ``z_measured - z_true`` after piston removal [m].
    rms_error : float
        RMS of *error* over the aperture [m]; ``nan`` without ground truth.
    pv_error : float
        Peak-to-valley of *error* [m]; ``nan`` without ground truth.
    wrapped : NDArray
        The recovered wrapped phase [rad].
    frames : PSIFrames
        The interferogram stack the reconstruction came from.
    sampling_ok : bool
        ``False`` when the surface exceeded the ``pi``-per-pixel unwrapping
        limit, so the reconstruction is expected to be wrong.
    """

    z_measured: NDArray[np.float64]
    z_true: Optional[NDArray[np.float64]]
    error: Optional[NDArray[np.float64]]
    rms_error: float
    pv_error: float
    wrapped: NDArray[np.float64]
    frames: PSIFrames
    sampling_ok: bool


def reconstruct_surface(
    frames: PSIFrames,
    *,
    prefer_skimage: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct height from an interferogram stack.

    Parameters
    ----------
    frames : PSIFrames
        The frame stack.  Recovery uses ``deltas_nominal`` -- the steps the
        instrument *believes* it took -- so any step error shows up as a real
        measurement error, as it would on a tool.
    prefer_skimage : bool
        Forwarded to :func:`unwrap_phase_2d`.

    Returns
    -------
    (NDArray, NDArray)
        Reconstructed height [m] and the intermediate wrapped phase [rad], both
        ``NaN`` outside the mask.
    """
    wrapped = recover_phase(frames.frames, frames.deltas_nominal)
    unwrapped = unwrap_phase_2d(wrapped, frames.mask, prefer_skimage=prefer_skimage)
    z = phase_to_height(unwrapped, frames.wavelength)
    return apply_mask(z, frames.mask), apply_mask(wrapped, frames.mask)


def measure_surface(
    surface: WaferSurface,
    wavelength: float = HENE_WAVELENGTH,
    *,
    n_steps: int = 4,
    noise: float = 0.01,
    step_error: float = 0.0,
    visibility: float = 0.9,
    seed: Optional[int] = None,
    prefer_skimage: bool = True,
) -> Measurement:
    """Run a full simulated PSI measurement of *surface* and score it.

    The reconstruction is only defined up to a whole fringe order, so the
    measured surface is piston-aligned to the truth before the error is
    computed -- the same convention a real tool uses when it reports form error
    against a nominal.

    Parameters
    ----------
    surface : WaferSurface
        Surface to measure.
    wavelength : float
        Illumination or synthetic wavelength [m].
    n_steps : int
        Number of phase steps.
    noise : float
        Detector noise as a fraction of mean intensity.
    step_error : float
        Fractional phase-step calibration error.
    visibility : float
        Fringe visibility.
    seed : int, optional
        RNG seed.
    prefer_skimage : bool
        Forwarded to :func:`unwrap_phase_2d`.

    Returns
    -------
    Measurement
        Reconstruction plus error statistics.

    Examples
    --------
    >>> from .synthesize import synthesize_wafer
    >>> from .zernike import flatten
    >>> surf = synthesize_wafer(n_pixels=128, seed=0)
    >>> flat, _ = flatten(surf, remove_power=True)
    >>> m = measure_surface(flat, wavelength=50e-6, noise=0.0, seed=1)
    >>> bool(m.rms_error < 1e-8)
    True
    """
    worst_step, sampling_ok = check_sampling(surface.z, wavelength, surface.mask)
    if not sampling_ok:
        logger.warning(
            "Surface undersampled at lambda=%.3g m: worst adjacent phase step %.2f rad > pi. "
            "Unwrapping will fail -- flatten the surface first or use a synthetic wavelength.",
            wavelength, worst_step,
        )

    frames = simulate_fringes(
        surface.z, wavelength,
        n_steps=n_steps, visibility=visibility, noise=noise,
        step_error=step_error, mask=surface.mask, seed=seed,
    )
    z_measured, wrapped = reconstruct_surface(frames, prefer_skimage=prefer_skimage)

    mask = surface.mask
    z_true = surface.z
    # Remove the arbitrary fringe-order offset before scoring.
    offset = float(np.nanmean(z_measured[mask] - z_true[mask]))
    z_measured = apply_mask(z_measured - offset, mask)
    error = apply_mask(z_measured - z_true, mask)
    rms_error = float(np.sqrt(np.nanmean(error[mask] ** 2)))
    pv_error = float(np.nanmax(error[mask]) - np.nanmin(error[mask]))

    logger.info(
        "PSI measurement: lambda=%.4g µm, %d steps, noise=%.3f, step_err=%.3f "
        "-> RMS error %.3f nm, PV %.3f nm",
        wavelength * 1e6, n_steps, noise, step_error, rms_error * 1e9, pv_error * 1e9,
    )
    return Measurement(
        z_measured=z_measured,
        z_true=z_true,
        error=error,
        rms_error=rms_error,
        pv_error=pv_error,
        wrapped=wrapped,
        frames=frames,
        sampling_ok=sampling_ok,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_interferogram(
    surface: WaferSurface,
    frames: PSIFrames,
    wrapped: NDArray[np.float64],
    *,
    title: str = "Phase-shifting interferometry",
    figsize: Tuple[float, float] = (12.0, 4.2),
) -> Tuple[Figure, NDArray]:
    """Show a fringe frame, the wrapped phase, and the unwrapped phase.

    Parameters
    ----------
    surface : WaferSurface
        Surface supplying the coordinate grid and aperture.
    frames : PSIFrames
        Interferogram stack; the first frame is displayed.
    wrapped : NDArray
        Recovered wrapped phase [rad].
    title : str
        Figure suptitle.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    import matplotlib.pyplot as plt

    unwrapped = unwrap_phase_2d(wrapped, surface.mask)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    wafer_imshow(
        axes[0], surface.x, surface.y, frames.frames[0],
        scale=1.0, label="Intensity [a.u.]", signed=False, cmap="gray",
        radius=surface.radius, title=f"Interferogram (step 1 of {frames.n_steps})",
    )
    wafer_imshow(
        axes[1], surface.x, surface.y, wrapped,
        scale=1.0, label="Phase [rad]", signed=True, vlim=(-np.pi, np.pi),
        radius=surface.radius, title="Wrapped phase",
    )
    wafer_imshow(
        axes[2], surface.x, surface.y, unwrapped,
        scale=1.0, label="Phase [rad]", signed=False, cmap=CMAP_MAGNITUDE,
        radius=surface.radius, title="Unwrapped phase",
    )
    engine = "scikit-image" if HAVE_SKIMAGE else "bundled DCT least-squares"
    fig.suptitle(f"{title} — λ = {frames.wavelength * 1e6:.3g} µm, unwrapper: {engine}", fontsize=12)
    fig.tight_layout()
    return fig, axes


def plot_reconstruction(
    surface: WaferSurface,
    measurement: Measurement,
    *,
    scale: float = 1e9,
    unit: str = "nm",
    title: str = "Interferometric reconstruction",
) -> Tuple[Figure, NDArray]:
    """Side-by-side truth, reconstruction and error map.

    Parameters
    ----------
    surface : WaferSurface
        Surface that was measured (supplies grid and truth).
    measurement : Measurement
        Result from :func:`measure_surface`.
    scale : float
        Height multiplier for display (``1e9`` -> nm).
    unit : str
        Unit label matching *scale*.
    title : str
        Figure suptitle.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    fig, axes = plot_wafer_panels(
        surface.x, surface.y,
        [measurement.z_true, measurement.z_measured, measurement.error],
        ["True surface", "Reconstructed", "Error (reconstructed − true)"],
        scale=scale,
        label=f"Height [{unit}]",
        signed=True,
        radius=surface.radius,
        shared_scale=False,
        suptitle=f"{title} — λ = {measurement.frames.wavelength * 1e6:.3g} µm",
    )
    annotate_stats(
        axes[2],
        f"RMS  {measurement.rms_error * scale:.3g} {unit}\n"
        f"PV   {measurement.pv_error * scale:.3g} {unit}\n"
        f"noise {measurement.frames.n_steps}-step",
        loc="lower left",
    )
    return fig, axes
