"""
Pupil machinery of the projection lens.

* Zernike wavefront aberrations in the Noll convention (j = 1…21)
* Normalised pupil coordinates for display-grid evaluation
* Defocus optical path difference, paraxial and exact

Everything here works on **normalised pupil coordinates** (ρ ∈ [0, 1] and the
azimuth ϕ), so it is independent of whichever grid the caller happens to be
on.  ``aerial_image`` evaluates these on the FFT frequency grid shifted per
source point; the app evaluates them on a centred display grid.  Keeping the
physics grid-agnostic is what stops the two drifting apart.

When vector imaging lands (see the roadmap), the Jones pupil belongs here too.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalised pupil coordinates
# ---------------------------------------------------------------------------


def pupil_grid(n: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return (ρ, ϕ, ξ, η) arrays on a normalised [-1, 1] pupil grid.

    This is THE grid convention: **rows = η, columns = ξ**. The Abbe loop
    derives spatial frequencies from pixel indices assuming exactly this
    layout, so every producer of pupil-plane arrays (``illumination``,
    ``source``) must render on this function's output — a second definition
    that drifted would misplace every source point.
    """
    c = np.linspace(-1.0, 1.0, n)
    xi, eta = np.meshgrid(c, c)
    rho = np.sqrt(xi ** 2 + eta ** 2)
    phi = np.arctan2(eta, xi)
    return rho, phi, xi, eta


# ---------------------------------------------------------------------------
# Zernike polynomials – Noll indexing
# ---------------------------------------------------------------------------

# Mapping: Noll index → (radial order n, azimuthal order m)
_NOLL_MAP: dict[int, tuple[int, int]] = {
    1:  (0,  0),   # piston
    2:  (1,  1),   # tilt X
    3:  (1, -1),   # tilt Y
    4:  (2,  0),   # defocus
    5:  (2, -2),   # astigmatism 45°
    6:  (2,  2),   # astigmatism 0°
    7:  (3, -1),   # coma Y
    8:  (3,  1),   # coma X
    9:  (3, -3),   # trefoil Y
    10: (3,  3),   # trefoil X
    11: (4,  0),   # primary spherical
    12: (4,  2),   # secondary astig 0°
    13: (4, -2),   # secondary astig 45°
    14: (4,  4),   # quadrafoil 0°
    15: (4, -4),   # quadrafoil 45°
    16: (5,  1),   # secondary coma X
    17: (5, -1),   # secondary coma Y
    18: (5,  3),   # secondary trefoil X
    19: (5, -3),   # secondary trefoil Y
    20: (5,  5),   # pentafoil X
    21: (5, -5),   # pentafoil Y
}


def _radial_poly(n: int, m: int, rho: NDArray) -> NDArray:
    """Evaluate the radial Zernike polynomial R_n^|m|(ρ).

    Uses the standard summation formula; input ρ is clipped to [0, 1].
    """
    m_abs = abs(m)
    rho_c = np.clip(rho, 0.0, 1.0)
    result = np.zeros_like(rho_c, dtype=np.float64)
    for s in range((n - m_abs) // 2 + 1):
        coeff = (
            (-1) ** s
            * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial((n + m_abs) // 2 - s)
                * math.factorial((n - m_abs) // 2 - s)
            )
        )
        result += coeff * rho_c ** (n - 2 * s)
    return result


def zernike_noll(
    j: int,
    rho: NDArray,
    phi: NDArray,
) -> NDArray[np.float64]:
    """Evaluate Zernike polynomial Zⱼ using the Noll convention.

    Terms j = 1…21 are implemented.  Higher-order terms return zero.
    (The interactive app exposes sliders up to Z21, so stopping at 15 left
    the six highest sliders silently doing nothing to the simulation.)

    Parameters
    ----------
    j : int
        Noll index (1-based).
    rho : NDArray
        Normalised pupil radius (0 ≤ ρ ≤ 1 inside pupil).
    phi : NDArray
        Azimuthal angle [rad], same shape as *rho*.

    Returns
    -------
    NDArray
        Zernike values, same shape as *rho*.
    """
    if j not in _NOLL_MAP:
        logger.warning("Zernike Noll index %d not implemented; returning 0.", j)
        return np.zeros_like(rho, dtype=np.float64)

    n, m = _NOLL_MAP[j]
    norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2.0 * (n + 1))
    R = _radial_poly(n, m, rho)

    if m > 0:
        return norm * R * np.cos(m * phi)
    elif m < 0:
        return norm * R * np.sin(abs(m) * phi)
    else:
        return norm * R


# build_pupil() used to live here: a display-grid pupil builder with an
# apodisation option.  It had no callers anywhere in the repo (the imaging
# path uses aerial_image._build_fft_pupil; the app has its own display
# builder), so it was removed in the process-step reorganisation.


# ---------------------------------------------------------------------------
# Defocus
# ---------------------------------------------------------------------------


def defocus_opd(
    rho: NDArray,
    wavelength: float,
    NA: float,
    defocus: float,
    n_image: float = 1.0,
    exact: bool = False,
) -> NDArray[np.float64]:
    """Wavefront phase [rad] from defocusing by *defocus* metres.

    Moving the wafer off the focal plane makes each ray travel a different
    extra distance depending on how steeply it converges.  A ray at angle θ
    in the image medium picks up an optical path difference

    .. math:: \\mathrm{OPD}(\\rho) = n \\, \\Delta z \\, (1 - \\cos\\theta),
        \\qquad \\sin\\theta = \\rho\\,\\mathrm{NA}/n

    which is the ``exact=True`` branch.  Expanding the cosine to second order
    gives the familiar quadratic ``π NA² ρ² Δz / (λ n)`` used everywhere in
    the paraxial literature — the ``exact=False`` branch, and the historical
    behaviour of this engine.

    The two agree while θ is small and part company when it is not: at the
    immersion preset (NA 1.35 in water) the quadratic form understates the
    OPD at the rim of the pupil by about a third.  That error is the reason
    ``exact`` exists, and why the vector-imaging path turns it on.

    Parameters
    ----------
    rho : NDArray
        Normalised pupil radius.  Values above 1 are outside the aperture;
        the caller is expected to mask them off, but they are clipped here so
        the square root stays real either way.
    wavelength : float
        Exposure wavelength [m].
    NA : float
        Numerical aperture.
    defocus : float
        Defocus [m].  Positive = above the focal plane.
    n_image : float
        Refractive index of the medium the image forms in.
    exact : bool
        Select the non-paraxial form.

    Returns
    -------
    NDArray[np.float64]
        Phase in radians, same shape as *rho*.  Zero everywhere when
        *defocus* is zero.
    """
    if defocus == 0.0:
        return np.zeros_like(rho, dtype=np.float64)

    if not exact:
        return (
            np.pi * NA ** 2 * np.asarray(rho, dtype=np.float64) ** 2
            * defocus / (wavelength * n_image)
        )

    # sin θ = ρ·NA/n. The clip keeps the root real for ρ > 1 (outside the
    # pupil, masked off by the caller) and for the unphysical NA > n case.
    sin_sq = np.clip(
        (np.asarray(rho, dtype=np.float64) * NA / n_image) ** 2, 0.0, 1.0
    )
    cos_theta = np.sqrt(1.0 - sin_sq)
    opd = n_image * defocus * (1.0 - cos_theta)          # [m]
    return (2.0 * np.pi / wavelength) * opd


# ---------------------------------------------------------------------------
# Polarisation and the vector pupil
# ---------------------------------------------------------------------------

#: Illumination polarisation states understood by the vector imaging path.
POLARISATIONS = ("unpolarised", "x", "y", "te", "tm")


def jones_states(polarisation: str, phi_source: float) -> list:
    """Input Jones vectors for a polarisation state, as (Jx, Jy, weight).

    One source point is one incident plane wave, and a plane wave has **one**
    polarisation — so the Jones vector returned here is a pair of scalars,
    constant across the pupil for that source point.  The pupil azimuth does
    not enter until :func:`vector_coefficients` decomposes each *diffracted*
    ray into its own s and p directions.  (Building the input state from the
    diffracted azimuth instead is a tempting and entirely wrong shortcut: it
    silently gives every order a different incident polarisation.)

    Three states are a single fully polarised field; ``"unpolarised"`` is
    genuinely two mutually **incoherent** fields, which is why this returns a
    list — the caller sums their intensities, never their amplitudes.

    ``"te"`` and ``"tm"`` depend on *where in the source* the point sits:
    azimuthal polarisation is tangential to the pupil at the source point,
    radial is along it.  Azimuthal is what hyper-NA scanners use, because it
    keeps every ray pair s-polarised with respect to its own plane of
    incidence however steeply the lens bends them.

    Parameters
    ----------
    polarisation : str
        One of :data:`POLARISATIONS`.
    phi_source : float
        Azimuth of this source point in the pupil [rad].  Only ``"te"`` and
        ``"tm"`` use it; for an on-axis point it is degenerate and the
        convention below reduces them to y- and x-polarisation.

    Returns
    -------
    list of (Jx, Jy, weight)
        Each entry is one incoherent field; weights sum to 1.

    Raises
    ------
    ValueError
        For an unrecognised state, listing the options.
    """
    pol = polarisation.lower()

    if pol == "x":
        return [(1.0, 0.0, 1.0)]
    if pol == "y":
        return [(0.0, 1.0, 1.0)]
    if pol == "te":                       # azimuthal: Ĵ = (−sin φ_s, cos φ_s)
        return [(-math.sin(phi_source), math.cos(phi_source), 1.0)]
    if pol == "tm":                       # radial:    Ĵ = ( cos φ_s, sin φ_s)
        return [(math.cos(phi_source), math.sin(phi_source), 1.0)]
    if pol in ("unpolarised", "unpolarized"):
        # Two orthogonal, mutually incoherent halves. Any orthogonal pair
        # gives the same total intensity; x/y is the cheapest to write.
        return [(1.0, 0.0, 0.5), (0.0, 1.0, 0.5)]

    raise ValueError(
        f"polarisation must be one of {list(POLARISATIONS)}, got '{polarisation}'"
    )


def vector_coefficients(
    rho: NDArray,
    phi: NDArray,
    jx: NDArray,
    jy: NDArray,
    sin_theta_max: float,
    obliquity: bool = True,
) -> tuple:
    """Map an entrance-pupil Jones vector to the three image-space components.

    A lens does not merely redirect a ray, it **rotates the field that rides
    on it**: the electric field must stay perpendicular to the propagation
    direction, so as a ray is bent towards the wafer its field tips with it.
    Decompose the incoming field into the two natural directions for that
    ray — azimuthal (s) and radial (p) with respect to its own meridional
    plane — and only the radial half tips:

    .. math::
        \\hat{s} = (-\\sin\\phi,\\ \\cos\\phi,\\ 0), \\qquad
        \\hat{p} = (\\cos\\theta\\cos\\phi,\\ \\cos\\theta\\sin\\phi,\\ -\\sin\\theta)

    The s-component keeps its direction at every angle, which is why two TE
    beams still interfere perfectly however steeply they converge.  The
    p-component acquires a longitudinal ``−sinθ`` piece and its transverse
    part shrinks by ``cosθ``; two TM beams separated by 2θ therefore
    interfere with their amplitude scaled by ``cos 2θ``, which passes through
    zero at θ = 45° and goes **negative** beyond it — contrast that does not
    merely fade but inverts.

    Parameters
    ----------
    rho, phi : NDArray
        Normalised pupil radius and azimuth of the *diffracted* rays.
    jx, jy : float
        Entrance-pupil Jones components of the incident wave, from
        :func:`jones_states`.  Constant across the pupil for one source
        point; broadcast against *phi*.
    sin_theta_max : float
        ``NA / n_image`` — the sine of the marginal ray angle in the medium
        the image forms in.  Refraction into a denser medium lowers this and
        with it the whole vector effect.
    obliquity : bool
        Apply the aplanatic radiometric factor ``sqrt(cos θ)``.  A lens
        obeying the sine condition maps a flat entrance pupil onto a
        spherical wavefront, and energy conservation over the projected area
        supplies this amplitude weighting.

    Returns
    -------
    (Vx, Vy, Vz) : tuple of NDArray
        Complex coefficient maps to multiply into the pupil before the
        inverse transform.
    """
    sin_sq = np.clip((np.asarray(rho, dtype=np.float64) * sin_theta_max) ** 2, 0.0, 1.0)
    cos_t = np.sqrt(1.0 - sin_sq)
    sin_t = np.sqrt(sin_sq)

    cos_p, sin_p = np.cos(phi), np.sin(phi)

    # Project the input field onto the ray's own s (azimuthal) and p (radial)
    # directions.
    e_s = -jx * sin_p + jy * cos_p
    e_p = jx * cos_p + jy * sin_p

    # Rebuild in image space: ŝ is unchanged, p̂ tips by θ.
    vx = -e_s * sin_p + e_p * cos_t * cos_p
    vy = e_s * cos_p + e_p * cos_t * sin_p
    vz = -e_p * sin_t

    if obliquity:
        amp = np.sqrt(cos_t)
        vx, vy, vz = vx * amp, vy * amp, vz * amp

    return (
        vx.astype(np.complex128),
        vy.astype(np.complex128),
        vz.astype(np.complex128),
    )

