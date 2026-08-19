"""
Aerial image computation module.

Implements Hopkins' partially coherent imaging via the Abbe decomposition:

    I(x) = Σ_{f_s}  S(f_s) · |F⁻¹[ M̃(f) · P(f − f_s) ]|²

where:
* S(f_s) – source intensity at source point f_s
* M̃(f)  – Fourier spectrum of the mask transmittance
* P(f)   – coherent pupil function (aperture + defocus + aberrations)
* F⁻¹    – 2-D inverse discrete Fourier transform

The pupil is constructed directly on the FFT frequency grid to ensure
correct spatial-frequency alignment.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.core.utils import make_freq_grid
from litho_sim.expose.illumination import build_source
from litho_sim.expose.pupil import (
    defocus_opd,
    jones_states,
    vector_coefficients,
    zernike_noll,
)

logger = logging.getLogger(__name__)


#: Which way a source point tilts the illumination reaching the mask.
#:
#: :func:`_pupil_geometry` evaluates the pupil at ``f - f_s`` and leaves the
#: mask spectrum alone.  Shifting the pupil by ``+f_s`` is the same as shifting
#: the *spectrum* by ``-f_s``, so a source point at pupil ``+sigma`` illuminates
#: the mask with a plane wave ``exp(-i 2 pi f_s . x)`` — the opposite handedness
#: to the usual textbook statement and to this module's own docstring.
#:
#: It cancels under the thin-mask model, because every built-in source is
#: centrosymmetric and pairs ``+f_s`` with ``-f_s``.  It stops cancelling as
#: soon as the mask spectrum is computed *at* an incidence angle, where this
#: sign decides which way a printed feature shifts.  Pinned by
#: ``test_a_source_point_tilts_the_mask_the_other_way``.
SOURCE_TILT_SIGN = -1.0


# ---------------------------------------------------------------------------
# Internal: pupil on FFT frequency grid
# ---------------------------------------------------------------------------


def _pupil_geometry(
    FX: NDArray[np.float64],
    FY: NDArray[np.float64],
    wavelength: float,
    NA: float,
    fs_x: float = 0.0,
    fs_y: float = 0.0,
    need_phi: bool = False,
) -> tuple:
    """Normalised pupil coordinates for one source point, on the FFT grid.

    The source-point offset is applied by *evaluating* the pupil on shifted
    frequency coordinates rather than by rolling a pre-built array.  That
    makes the shift exact for arbitrary non-integer offsets — a rolled array
    can only shift by whole pixels, which quantises every source point onto
    the FFT frequency lattice and, for small σ, can collapse distinct source
    points onto the same shift.

    ``(fx, fy)`` here is the transverse spatial frequency of the diffracted
    ray leaving the lens, so ρ and ϕ describe where that ray sits in the
    pupil — which is what both the aberration phase and (later) the vector
    coefficients are functions of.

    Parameters
    ----------
    FX, FY : NDArray
        Frequency grid [m⁻¹] in FFT ordering, from :func:`make_freq_grid`.
        Passed in rather than rebuilt because it does not depend on the
        source point, and the Abbe loop calls this once per source point.
    wavelength, NA : float
        Optical parameters; together they set the pupil radius ``NA/λ``.
    fs_x, fs_y : float
        Source-point spatial frequency offset [m⁻¹].
    need_phi : bool
        Compute the azimuth.  It is only wanted for aberrations and for the
        vector coefficients, and ``arctan2`` over the whole grid once per
        source point is a measurable cost on an otherwise FFT-bound loop, so
        it is skipped by default.

    Returns
    -------
    (RHO, PHI, inside) : tuple
        Normalised radius, azimuth [rad] (``None`` unless *need_phi*), and
        the in-aperture mask.
    """
    k_max = NA / wavelength  # pupil radius in freq space [m⁻¹]

    fx = FX - fs_x
    fy = FY - fs_y

    RHO = np.sqrt(fx ** 2 + fy ** 2) / k_max   # normalised radius (0…1)
    PHI = np.arctan2(fy, fx) if need_phi else None
    inside = RHO <= 1.0
    return RHO, PHI, inside


def _build_fft_pupil(
    RHO: NDArray[np.float64],
    PHI: NDArray[np.float64],
    inside: NDArray[np.bool_],
    wavelength: float,
    NA: float,
    defocus: float,
    n_image: float,
    zernike_coeffs: dict | None,
    exact_defocus: bool = False,
) -> NDArray[np.complex128]:
    """Build the pupil function ``P(f − f_s)`` in FFT (non-centred) ordering.

    The returned array is indexed identically to ``np.fft.fft2`` output:
    DC at [0, 0], positive frequencies first, then negative wrap-around.

    Parameters
    ----------
    RHO, PHI, inside : NDArray
        Pupil geometry from :func:`_pupil_geometry`.
    wavelength, NA, defocus, n_image : float
        Optical parameters.  *n_image* is the index of the medium the image
        forms in, which is what the defocus OPD is measured against.
    zernike_coeffs : dict or None
        ``{noll_j: coeff_in_waves}`` aberration map.
    exact_defocus : bool
        Use the non-paraxial defocus OPD.

    Returns
    -------
    NDArray[np.complex128]
        Pupil array, same shape as *RHO*.
    """
    W = np.where(
        inside,
        defocus_opd(RHO, wavelength, NA, defocus, n_image, exact=exact_defocus),
        0.0,
    )

    # Zernike aberrations
    if zernike_coeffs:
        for j, c in zernike_coeffs.items():
            Z = zernike_noll(j, np.where(inside, RHO, 0.0), PHI)
            W += np.where(inside, 2.0 * np.pi * c * Z, 0.0)

    pupil = np.where(inside, np.exp(1j * W), 0.0 + 0.0j)
    return pupil.astype(np.complex128)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalisation_scale(
    mode: str, peak: float, clear: float, dose: float = 1.0
) -> float:
    """The single constant that turns a raw Abbe sum into a dosed image.

    All three modes divide the same summed intensity by a different scalar,
    and dose multiplies whatever comes out — so the whole of normalisation is
    one number. Isolating it here is what lets a caller cache the raw sum and
    restyle it for a new dose or mode without re-running the transform.

    Raises
    ------
    ValueError
        For an unrecognised mode.
    """
    m = mode.lower()
    if m == "peak":
        return dose / peak if peak > 0 else dose
    if m == "clear":
        return dose / clear if clear > 0 else dose
    if m == "none":
        return dose
    raise ValueError(
        f"normalisation must be 'peak', 'clear' or 'none', got '{mode}'"
    )


# ---------------------------------------------------------------------------
# Main aerial image computation
# ---------------------------------------------------------------------------


def compute_aerial_image(
    mask: NDArray,
    optics: OpticsConfig,
    grid: GridConfig,
    dose: float = 1.0,
    return_raw: bool = False,
):
    """Compute the partially coherent aerial image (Abbe method).

    Parameters
    ----------
    mask : NDArray
        2-D mask transmittance, shape ``(n, n)``.  May be real (binary)
        or complex (PSM).
    optics : OpticsConfig
        Optical system parameters.
    grid : GridConfig
        Numerical grid definition.
    dose : float
        Normalised exposure dose applied as a multiplier on the intensity.
    return_raw : bool
        Return ``(raw_sum, peak, clear)`` instead of the finished image —
        the un-normalised, un-dosed intensity together with the two scalars
        that would have normalised it.  Intended for callers that want to
        vary dose or normalisation without repeating the Abbe sum, which is
        ~98 % of the cost.  See :func:`normalisation_scale`.

    Returns
    -------
    NDArray[np.float64]
        Aerial image intensity, shape ``(n, n)``, range ``[0, dose]``.
        With *return_raw*, a ``(NDArray, float, float)`` tuple instead.
    """
    n = grid.n_pixels
    dx = grid.pixel_size
    lam = optics.wavelength
    NA = optics.NA

    # --- Source ---
    # Sampled on its own coarse grid, independent of the mask grid. The Abbe
    # sum costs one FFT per source point, so tying the source to the mask grid
    # (as this used to) makes cost scale with n² for no accuracy gain.
    ns = max(int(optics.source_grid), 3)
    source = build_source(
        n_pixels=ns,
        source_type=optics.source_type,
        sigma_outer=optics.sigma_outer,
        sigma_inner=optics.sigma_inner,
        spec=optics.source_spec,
        **(optics.source_kwargs or {}),
    )

    # --- Mask spectrum ---
    # Under the thin-mask model this is one FFT, hoisted out of the source loop
    # and shared by every source point — and that hoist *is* the approximation,
    # because it asserts the mask diffracts the same way whatever angle light
    # arrives at. The provider is the seam where a thick mask can say otherwise.
    # Imported here rather than at module scope so the mask-3D package can
    # import this module for SOURCE_TILT_SIGN without a cycle.
    from litho_sim.expose.m3d import make_spectrum_provider

    spectra = make_spectrum_provider(mask, optics, grid)
    thin_mask = spectra.is_angle_independent
    # Hoisted only when it is legitimate to hoist. A thick mask recomputes it
    # per source point inside the loop.
    M_fft = spectra.base_spectrum() if thin_mask else None

    # --- Abbe sum ---
    aerial = np.zeros((n, n), dtype=np.float64)
    clear = 0.0            # open-frame throughput, accumulated alongside
    src_pts = np.argwhere(source > 0)
    half = (ns - 1) / 2.0  # matches expose.pupil.pupil_grid's linspace(-1, 1, ns)

    # Source-independent, so built once rather than per source point.
    FX, FY = make_freq_grid(n, dx)  # physical [m⁻¹], FFT ordering
    n_image = optics.image_index

    model = optics.imaging_model.lower()
    if model not in ("scalar", "vector"):
        raise ValueError(
            f"imaging_model must be 'scalar' or 'vector', got "
            f"'{optics.imaging_model}'"
        )
    vector = model == "vector"
    # Vector imaging needs cos θ anyway, so it always uses the exact defocus
    # OPD — mixing an exact vector pupil with a paraxial phase would be
    # inconsistent at exactly the angles the vector model exists to describe.
    exact_defocus = optics.exact_defocus or vector
    need_phi = vector or bool(optics.zernike_coeffs)

    for idx in src_pts:
        row_s, col_s = int(idx[0]), int(idx[1])
        weight = float(source[row_s, col_s])

        # Normalised source coords ξ_s, η_s ∈ [−1, +1], then the matching
        # spatial-frequency offset f_s = ξ_s · NA/λ [m⁻¹].
        fs_x = ((col_s - half) / half) * NA / lam
        fs_y = ((row_s - half) / half) * NA / lam

        # A thick mask diffracts differently depending on the angle light
        # arrives at, so its spectrum is a function of the source point rather
        # than a constant. This is the line the thin-mask approximation exists
        # to avoid paying for.
        if not thin_mask:
            M_fft = spectra.spectrum_for(fs_x, fs_y)

        # P(f − f_s), evaluated exactly at the shifted coordinates.
        RHO, PHI, inside = _pupil_geometry(
            FX, FY, lam, NA, fs_x, fs_y, need_phi=need_phi
        )
        P_shifted = _build_fft_pupil(
            RHO=RHO,
            PHI=PHI,
            inside=inside,
            wavelength=lam,
            NA=NA,
            defocus=optics.defocus,
            n_image=n_image,
            zernike_coeffs=optics.zernike_coeffs,
            exact_defocus=exact_defocus,
        )

        # What an *unpatterned* mask would return here. Exactly 1 for a thin
        # mask, by definition. A real blank is not 1 — an EUV mirror returns
        # about 75 % and bare quartz transmits 92 % — and without this factor
        # ``normalisation="clear"`` would measure against a mask that does not
        # exist. Multiplying by the literal 1.0 leaves the thin path's
        # floating-point result untouched.
        clear_scale = 1.0 if thin_mask else spectra.clear_intensity(fs_x, fs_y)

        if not vector:
            # Coherent image for this source point
            field = np.fft.ifft2(M_fft * P_shifted)
            aerial += weight * (np.abs(field) ** 2)
            # A clear mask diffracts into DC alone, and the DC order sits at
            # element [0, 0] of this very grid — so the open-frame throughput
            # comes for free, with no second Abbe sum.
            clear += weight * float(inside[0, 0]) * clear_scale
            continue

        # Vector: the field is a 3-vector, so each component gets its own
        # transform. The mask spectrum is shared — under the thin-mask
        # approximation the mask is a scalar transmittance and does not act
        # on polarisation. Intensity is |Ex|²+|Ey|²+|Ez|², a sum of squared
        # moduli, so the components cannot be combined before transforming.
        spectrum = M_fft * P_shifted
        phi_s = math.atan2((row_s - half), (col_s - half))
        for jx, jy, w_pol in jones_states(optics.polarisation, phi_s):
            Vx, Vy, Vz = vector_coefficients(
                RHO, PHI, jx, jy, optics.sin_theta_max,
                obliquity=optics.obliquity,
            )
            comps = np.stack((spectrum * Vx, spectrum * Vy, spectrum * Vz))
            fields = np.fft.ifft2(comps, axes=(-2, -1))
            aerial += (weight * w_pol) * (np.abs(fields) ** 2).sum(axis=0)
            clear += (weight * w_pol) * float(inside[0, 0]) * (
                abs(Vx[0, 0]) ** 2 + abs(Vy[0, 0]) ** 2 + abs(Vz[0, 0]) ** 2
            ) * clear_scale

    # --- Normalise and apply dose ---
    peak = float(aerial.max())
    if return_raw:
        # The un-normalised, un-dosed sum plus both scalars. Every
        # normalisation mode is this same array over a different constant, and
        # dose is a further constant on top, so a caller holding these three
        # can change either without paying for the Abbe sum again.
        return aerial.astype(np.float64), peak, float(clear)

    aerial *= normalisation_scale(optics.normalisation, peak, clear, dose)

    logger.info(
        "Aerial image: %d source pts (%dx%d grid), peak=%.4f, dose=%.2f",
        len(src_pts), ns, ns, float(aerial.max()), dose,
    )
    return aerial.astype(np.float64)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def extract_cross_section(
    image: NDArray[np.float64],
    axis: int = 1,
    index: int | None = None,
) -> NDArray[np.float64]:
    """Extract a 1-D cross-section through the centre of a 2-D image.

    Parameters
    ----------
    image : NDArray
        2-D array.
    axis : int
        0 = extract row (vary along x); 1 = extract column (vary along y).
    index : int, optional
        Row or column index.  Defaults to the centre.

    Returns
    -------
    NDArray
        1-D profile of length ``image.shape[axis]``.
    """
    if index is None:
        index = image.shape[axis] // 2
    return image[index, :] if axis == 0 else image[:, index]

