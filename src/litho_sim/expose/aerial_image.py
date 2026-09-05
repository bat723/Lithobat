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

Two things about how the sum is evaluated, both invisible in the result:

* **The transforms are batched.** One inverse FFT per source point is the
  cost of the method, and they are independent, so they go to the FFT
  library as a stack and it runs them across every core. Per call that is
  about three times faster than the same transforms one at a time.
* **Several image planes share one pass.** A depth-resolved exposure wants
  the same mask and source at twenty defocus values; only the pupil phase
  differs. :func:`compute_aerial_planes` builds the source, the mask
  spectrum and each source point's pupil geometry once and emits every
  plane's transforms from them. :func:`compute_aerial_image` is the
  one-plane case of the same loop.

For a scalar thin-mask process the same sum has a second, much cheaper
form — the sum-of-coherent-systems decomposition in
:mod:`litho_sim.expose.hopkins`, which trades the per-source-point
transforms for a handful of eigen-kernels when the optics are fixed and
only the mask changes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

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

#: How much complex spectrum (bytes) one batched transform may hold. Sets the
#: number of ``(source point, plane, field component)`` transforms that go to
#: the FFT library together: 256 of them at 128 px, 64 at 256 px, 16 at 512.
FFT_BATCH_BYTES = 64 * 2 ** 20


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


def _pupil_phase(
    rho: NDArray[np.float64],
    phi: NDArray[np.float64] | None,
    wavelength: float,
    NA: float,
    defocus: float,
    n_image: float,
    zernike_coeffs: dict | None,
    exact_defocus: bool,
) -> NDArray[np.float64]:
    """Wavefront phase [rad] at the given pupil coordinates — any shape.

    Defocus plus the Zernike sum, on whatever points are handed in. The Abbe
    loop calls it on the in-aperture points only: the aperture covers a few
    tens of frequency bins at typical grids, so evaluating the exponential
    over the whole ``n × n`` array — as the pupil used to be built — spent
    almost all of its time on zeros.
    """
    W = defocus_opd(rho, wavelength, NA, defocus, n_image, exact=exact_defocus)
    if zernike_coeffs:
        assert phi is not None, "aberrations need the azimuth"
        W = np.array(W, dtype=np.float64, copy=True)
        for j, c in zernike_coeffs.items():
            W += 2.0 * np.pi * c * zernike_noll(j, rho, phi)
    return W


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
    idx = np.nonzero(inside)
    W = _pupil_phase(
        RHO[idx], None if PHI is None else PHI[idx],
        wavelength, NA, defocus, n_image, zernike_coeffs, exact_defocus,
    )
    pupil = np.zeros(RHO.shape, dtype=np.complex128)
    pupil[idx] = np.exp(1j * W)
    return pupil


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
# The source, as points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePoints:
    """The illumination as the discrete set of points the Abbe sum runs over.

    Attributes
    ----------
    weight : (S,) array
        Source intensity at each point; the built-in sources sum to 1.
    fx, fy : (S,) arrays
        Spatial-frequency offset of each point [m⁻¹] — ``σ · NA/λ``.
    phi : (S,) array
        Azimuth of each point in the pupil [rad], for the TE/TM Jones states.
    grid : int
        Side of the sampling grid the source was rendered on.
    """

    weight: NDArray[np.float64]
    fx: NDArray[np.float64]
    fy: NDArray[np.float64]
    phi: NDArray[np.float64]
    grid: int

    def __len__(self) -> int:
        return int(self.weight.size)


def source_points(optics: OpticsConfig) -> SourcePoints:
    """Render the source and list its non-zero points.

    Sampled on its own coarse grid, independent of the mask grid. The Abbe
    sum costs one FFT per source point, so tying the source to the mask grid
    (as this used to) makes cost scale with n² for no accuracy gain.
    """
    ns = max(int(optics.source_grid), 3)
    source = build_source(
        n_pixels=ns,
        source_type=optics.source_type,
        sigma_outer=optics.sigma_outer,
        sigma_inner=optics.sigma_inner,
        spec=optics.source_spec,
        **(optics.source_kwargs or {}),
    )
    idx = np.argwhere(source > 0)
    rows = idx[:, 0].astype(np.float64)
    cols = idx[:, 1].astype(np.float64)
    half = (ns - 1) / 2.0  # matches expose.pupil.pupil_grid's linspace(-1, 1, ns)
    k_max = optics.NA / optics.wavelength
    # Normalised source coords ξ_s, η_s ∈ [−1, +1], then the matching
    # spatial-frequency offset f_s = ξ_s · NA/λ [m⁻¹].
    return SourcePoints(
        weight=source[idx[:, 0], idx[:, 1]].astype(np.float64),
        fx=((cols - half) / half) * k_max,
        fy=((rows - half) / half) * k_max,
        phi=np.arctan2(rows - half, cols - half),
        grid=ns,
    )


# ---------------------------------------------------------------------------
# The Abbe sum
# ---------------------------------------------------------------------------


class _Batch:
    """Spectra waiting to be transformed together, and where each one lands.

    Every pushed spectrum is one coherent field: a source point, an image
    plane, and (for the vector model) one Cartesian field component. They
    accumulate until the batch holds :data:`FFT_BATCH_BYTES` worth, then go
    to the FFT library as one stack, and each transform's intensity is added
    to its plane with its weight.
    """

    def __init__(self, aerial: NDArray[np.float64], n: int) -> None:
        self.aerial = aerial
        self.limit = max(1, FFT_BATCH_BYTES // (n * n * 16))
        self.specs: list[NDArray[np.complex128]] = []
        self.planes: list[int] = []
        self.weights: list[float] = []

    def push(self, spec: NDArray[np.complex128], plane: int, weight: float) -> None:
        self.specs.append(spec)
        self.planes.append(plane)
        self.weights.append(weight)
        if len(self.specs) >= self.limit:
            self.flush()

    def flush(self) -> None:
        if not self.specs:
            return
        fields = sfft.ifft2(np.stack(self.specs), axes=(-2, -1), workers=-1)
        intensity = fields.real ** 2 + fields.imag ** 2
        planes = np.asarray(self.planes)
        weights = np.asarray(self.weights, dtype=np.float64)
        for p in np.unique(planes):
            sel = planes == p
            self.aerial[p] += np.tensordot(weights[sel], intensity[sel], axes=1)
        self.specs.clear()
        self.planes.clear()
        self.weights.clear()


def _abbe_sum(
    mask: NDArray,
    optics: OpticsConfig,
    grid: GridConfig,
    planes: Sequence[tuple[float, float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The raw Abbe sum for one or more image planes.

    Parameters
    ----------
    mask : NDArray
        2-D mask transmittance, ``(n, n)``, real or complex.
    optics, grid
        The process. ``optics.defocus`` and ``optics.n_image`` are *ignored*
        in favour of *planes*.
    planes : sequence of (defocus, n_image)
        The image planes to form, each as a defocus [m] and the index of the
        medium it forms in.

    Returns
    -------
    (raw, peak, clear)
        ``raw`` is ``(P, n, n)``, the un-normalised, un-dosed intensity per
        plane; ``peak`` and ``clear`` are ``(P,)`` — the brightest pixel and
        the open-frame throughput of each plane, the two scalars a
        normalisation divides by.
    """
    n = grid.n_pixels
    lam = optics.wavelength
    NA = optics.NA

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
    zernike = optics.zernike_coeffs or None

    src = source_points(optics)

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

    # Source-independent, so built once rather than per source point.
    FX, FY = make_freq_grid(n, grid.pixel_size)  # physical [m⁻¹], FFT ordering

    P = len(planes)
    aerial = np.zeros((P, n, n), dtype=np.float64)
    clear = np.zeros(P, dtype=np.float64)   # open-frame throughput, accumulated alongside
    batch = _Batch(aerial, n)

    for s in range(len(src)):
        weight = float(src.weight[s])
        fs_x, fs_y = float(src.fx[s]), float(src.fy[s])

        # A thick mask diffracts differently depending on the angle light
        # arrives at, so its spectrum is a function of the source point rather
        # than a constant. This is the line the thin-mask approximation exists
        # to avoid paying for.
        M_s = M_fft if thin_mask else spectra.spectrum_for(fs_x, fs_y)
        assert M_s is not None

        # What an *unpatterned* mask would return here. Exactly 1 for a thin
        # mask, by definition. A real blank is not 1 — an EUV mirror returns
        # about 75 % and bare quartz transmits 92 % — and without this factor
        # ``normalisation="clear"`` would measure against a mask that does not
        # exist. Multiplying by the literal 1.0 leaves the thin path's
        # floating-point result untouched.
        clear_scale = 1.0 if thin_mask else spectra.clear_intensity(fs_x, fs_y)

        # P(f − f_s), evaluated exactly at the shifted coordinates — once per
        # source point, then reused by every plane. Only the in-aperture
        # points carry a pupil value, and there are few of them.
        RHO, PHI, inside = _pupil_geometry(FX, FY, lam, NA, fs_x, fs_y, need_phi=need_phi)
        idx = np.nonzero(inside)
        rho_in = RHO[idx]
        phi_in = None if PHI is None else PHI[idx]
        M_in = M_s[idx]
        dc_inside = float(inside[0, 0])   # a clear mask diffracts into DC alone

        if vector:
            states = jones_states(optics.polarisation, float(src.phi[s]))

        for p, (defocus, n_image) in enumerate(planes):
            W = _pupil_phase(rho_in, phi_in, lam, NA, defocus, n_image, zernike, exact_defocus)
            coherent = M_in * np.exp(1j * W)

            if not vector:
                spec = np.zeros((n, n), dtype=np.complex128)
                spec[idx] = coherent
                batch.push(spec, p, weight)
                # The DC order sits at element [0, 0] of this very grid — so
                # the open-frame throughput comes for free, with no second
                # Abbe sum.
                clear[p] += weight * dc_inside * clear_scale
                continue

            # Vector: the field is a 3-vector, so each component gets its own
            # transform. The mask spectrum is shared — under the thin-mask
            # approximation the mask is a scalar transmittance and does not act
            # on polarisation. Intensity is |Ex|²+|Ey|²+|Ez|², a sum of squared
            # moduli, so the components cannot be combined before transforming.
            sin_theta_max = NA / n_image
            assert phi_in is not None      # need_phi is True on the vector path
            for jx, jy, w_pol in states:
                V = vector_coefficients(
                    rho_in, phi_in, jx, jy, sin_theta_max, obliquity=optics.obliquity,
                )
                for comp in V:
                    spec = np.zeros((n, n), dtype=np.complex128)
                    spec[idx] = coherent * comp
                    batch.push(spec, p, weight * w_pol)
                if dc_inside:
                    at_dc = np.nonzero((idx[0] == 0) & (idx[1] == 0))[0][0]
                    clear[p] += (weight * w_pol) * sum(
                        abs(comp[at_dc]) ** 2 for comp in V
                    ) * clear_scale

    batch.flush()
    peak = aerial.reshape(P, -1).max(axis=1)
    logger.debug(
        "Abbe sum: %d source pts (%dx%d grid), %d plane(s), %s model",
        len(src), src.grid, src.grid, P, model,
    )
    return aerial, peak, clear


# ---------------------------------------------------------------------------
# Public entry points
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
    raw, peak, clear = _abbe_sum(mask, optics, grid, [(optics.defocus, optics.image_index)])
    aerial, peak_v, clear_v = raw[0], float(peak[0]), float(clear[0])
    if return_raw:
        # The un-normalised, un-dosed sum plus both scalars. Every
        # normalisation mode is this same array over a different constant, and
        # dose is a further constant on top, so a caller holding these three
        # can change either without paying for the Abbe sum again.
        return aerial, peak_v, clear_v

    aerial = aerial * normalisation_scale(optics.normalisation, peak_v, clear_v, dose)
    logger.debug(
        "Aerial image: peak=%.4f, dose=%.2f", float(aerial.max()), dose,
    )
    return aerial


def compute_aerial_planes(
    mask: NDArray,
    optics: OpticsConfig,
    grid: GridConfig,
    planes: Sequence[tuple[float, float]],
    dose: float = 1.0,
) -> NDArray[np.float64]:
    """Aerial images at several ``(defocus, n_image)`` planes in one pass.

    Each plane is normalised exactly as :func:`compute_aerial_image` would
    normalise it on its own — under ``"peak"`` that is per plane, which is
    the historical behaviour of the depth-resolved exposure. What is shared
    is the work: the source, the mask spectrum and every source point's
    pupil geometry are built once, and all planes' transforms go through the
    same batched FFT.

    Parameters
    ----------
    mask, optics, grid
        As for :func:`compute_aerial_image`; ``optics.defocus`` and
        ``optics.n_image`` are overridden per plane.
    planes : sequence of (defocus, n_image)
        Defocus [m] and image-medium index of each plane.
    dose : float
        Relative dose, applied to every plane.

    Returns
    -------
    NDArray[np.float64]
        ``(len(planes), n, n)`` — one normalised, dosed image per plane.
    """
    raw, peak, clear = _abbe_sum(mask, optics, grid, planes)
    for p in range(raw.shape[0]):
        raw[p] *= normalisation_scale(
            optics.normalisation, float(peak[p]), float(clear[p]), dose
        )
    return raw


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
