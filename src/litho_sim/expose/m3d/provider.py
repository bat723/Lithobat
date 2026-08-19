"""The seam between a mask model and the Abbe sum.

:func:`~litho_sim.expose.aerial_image.compute_aerial_image` needs one thing from
a mask: a diffraction spectrum on the FFT grid. Under the thin-mask
approximation that spectrum is a constant — one ``fft2``, hoisted out of the
source loop, shared by every source point. That hoist *is* the approximation.

A thick mask has no such constant. The near field depends on the angle the
illumination arrives at, so the spectrum becomes a function of the source point,
and it is polarisation-dependent besides. This module is the interface that
makes both cases look the same to the imaging code.

Mask-side geometry
------------------
Two conversions live here because they have no other natural home, and both are
easy to get wrong in a way that produces a plausible image.

**Scale.** The mask array is drawn at *wafer* scale everywhere in this engine.
The reticle is ``reduction`` times larger. Sampling a near field on the same
``n_pixels`` with ``dx_mask = reduction * pixel_size`` makes FFT bin *k* land at
mask-side frequency ``k/(N M dx_w)``, which the lens maps to wafer-side
``M * that = k/(N dx_w)`` — **the same bin index**. So a near field sampled that
way drops straight into the existing spectrum slot with no resampling and no
change to the frequency grid.

**Tilt.** See :data:`~litho_sim.expose.aerial_image.SOURCE_TILT_SIGN`. The Abbe
loop shifts the pupil rather than the spectrum, which is valid because the two
differ only by a linear phase that ``|.|^2`` removes — but it means the spectrum
handed back must be referenced to *normal* incidence, with the incident
plane-wave phase divided out. It also means a source point at pupil ``+sigma``
tilts the illumination the other way.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.expose.aerial_image import SOURCE_TILT_SIGN
from litho_sim.expose.m3d.stack import MaskStack

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime only
    from litho_sim.expose.m3d.nearfield import MaskGeometry

logger = logging.getLogger(__name__)

#: Mask models, cheapest first.  See the package docstring for what each buys.
MASK_MODELS: tuple[str, ...] = ("thin", "multilayer", "fdtd")

#: Angular range and sampling of the blank-reflection table, in degrees.  25°
#: covers every order a 4x reticle can emit into a 0.33 NA pupil at a 6° chief
#: ray; past the Bragg band the reflectance is near zero anyway.  0.05° steps
#: keep the interpolation error well inside the model's own accuracy.
_THETA_TABLE_MAX_DEG = 25.0
_THETA_TABLE_N = 501


# ---------------------------------------------------------------------------
# Mask-side geometry
# ---------------------------------------------------------------------------


def mask_side_sin_theta(
    fs_x: float,
    fs_y: float,
    optics: OpticsConfig,
) -> tuple[float, float]:
    """Direction sines of the illumination *at the reticle*, for one source point.

    Parameters
    ----------
    fs_x, fs_y : float
        Source-point spatial-frequency offset [m⁻¹], exactly as the Abbe loop
        computes it.
    optics : OpticsConfig
        Supplies wavelength, ``reduction`` and the chief-ray tilt.

    Returns
    -------
    (sin_x, sin_y) : tuple of float
        Mask-side direction sines.

    Notes
    -----
    Three things happen here, and each one is a sign or a factor that would
    otherwise be rediscovered from a placement error:

    * ``lam * f_s`` converts frequency to a wafer-side direction sine.
    * dividing by ``reduction`` converts it to the mask side — the reticle sits
      in a cone ``reduction`` times slower than the wafer.
    * :data:`~litho_sim.expose.aerial_image.SOURCE_TILT_SIGN` accounts for the
      engine shifting the pupil instead of the spectrum, which reverses the
      handedness of the illumination tilt.

    The chief ray is added *after* the reduction because it is already a
    mask-side angle — it describes how the illuminator is tilted to keep out of
    the reflected beam's way, which is a fact about the reticle, not the wafer.
    """
    lam = optics.wavelength
    m = optics.reduction
    sin_x = SOURCE_TILT_SIGN * lam * fs_x / m
    sin_y = SOURCE_TILT_SIGN * lam * fs_y / m

    cra = math.sin(math.radians(optics.chief_ray_deg))
    axis = optics.chief_ray_axis.lower()
    if axis == "x":
        sin_x += cra
    elif axis == "y":
        sin_y += cra
    else:
        raise ValueError(
            f"chief_ray_axis must be 'x' or 'y', got '{optics.chief_ray_axis}'"
        )
    return sin_x, sin_y


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class SpectrumProvider(Protocol):
    """What the Abbe loop needs from a mask model.

    Implementations are free to be expensive on construction — building an FDTD
    angle library, for instance — but every per-source-point call happens inside
    the loop and should be cheap.
    """

    #: True when the spectrum does not depend on the source point, which lets
    #: the caller hoist one ``fft2`` out of the loop exactly as it always has.
    is_angle_independent: bool

    def base_spectrum(self) -> NDArray[np.complex128]:
        """The source-point-independent spectrum.

        Only meaningful when :attr:`is_angle_independent`; the thin-mask model
        is the case that matters, and this is the array the loop multiplies by
        the shifted pupil.
        """
        ...


class ThinMaskSpectra:
    """The Kirchhoff screen: one spectrum, shared by every source point.

    This exists so the thin-mask path goes through the same seam as the thick
    ones rather than around it.  It computes exactly the ``fft2`` the imaging
    code used to compute inline, so switching the engine onto the seam is
    bit-for-bit invisible — which is the property that makes the seam safe to
    introduce ahead of the physics that needs it.
    """

    is_angle_independent = True

    def __init__(self, mask: NDArray) -> None:
        self._spectrum = np.fft.fft2(mask.astype(np.complex128))

    def base_spectrum(self) -> NDArray[np.complex128]:
        return self._spectrum

    def clear_intensity(self, fs_x: float, fs_y: float) -> float:
        return 1.0


class MultilayerSpectra:
    """Thin-mask spectrum, with every diffraction order reflected at its own angle.

    The mask pattern is still a Kirchhoff screen — this model says nothing about
    absorber thickness — but the *mirror* underneath it is solved exactly. For
    a reflective EUV mask that turns out to buy most of what a rigorous model
    buys, because the orders of a dense grating leave the mask several degrees
    apart and a Bragg stack is a strong function of angle over exactly that
    range.

    Cost is one complex multiply per source point on top of the Abbe sum. The
    transfer-matrix solves are done once, at construction, and interpolated.

    See :mod:`~litho_sim.expose.m3d.multilayer` for why the order phases are the
    interesting part.
    """

    is_angle_independent = False

    def __init__(
        self,
        mask: NDArray,
        optics: OpticsConfig,
        grid: GridConfig,
        stack: MaskStack,
    ) -> None:
        from litho_sim.core.utils import make_freq_grid

        self._base = np.fft.fft2(mask.astype(np.complex128))
        self._optics = optics
        self._stack = stack
        lam = optics.wavelength

        # Wafer-side frequency grid, converted once to the mask-side direction
        # sine each order is *offset* by. Adding the incident sine to this gives
        # the angle that order actually leaves the mask at.
        FX, FY = make_freq_grid(grid.n_pixels, grid.pixel_size)
        self._doff_x = lam * FX / optics.reduction
        self._doff_y = lam * FY / optics.reduction

        # One transfer-matrix sweep, interpolated thereafter. Magnitude and
        # unwrapped phase are tabulated separately: the phase turns through more
        # than 60 degrees across the band, and interpolating the complex value
        # would cut the corner off every arc.
        blank = stack.blank_stack(lam)
        theta = np.linspace(0.0, np.deg2rad(_THETA_TABLE_MAX_DEG), _THETA_TABLE_N)
        r_s = np.array(
            [blank.amplitude_reflection(lam, float(t), "s") for t in theta],
            dtype=np.complex128,
        )
        r_p = np.array(
            [blank.amplitude_reflection(lam, float(t), "p") for t in theta],
            dtype=np.complex128,
        )
        # The scalar path has no polarisation to speak of, so it gets the mean
        # of the two. Over the angles an EUV order reaches, s and p differ by
        # under a percent in amplitude; the approximation is pinned by a test
        # rather than left as an assurance.
        r = 0.5 * (r_s + r_p)
        self._theta = theta
        self._mag = np.abs(r)
        self._phase = np.unwrap(np.angle(r))

    def _reflection(self, sin_theta: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Interpolate the blank's reflection at each order's own angle."""
        # Beyond sin = 1 an order is evanescent: it never leaves the mask, so it
        # carries no energy and gets nothing. It is outside the pupil in any
        # case, but clipping keeps arcsin real.
        propagating = sin_theta <= 1.0
        s = np.clip(sin_theta, 0.0, 1.0)
        theta = np.arcsin(s)
        mag = np.interp(theta, self._theta, self._mag)
        phase = np.interp(theta, self._theta, self._phase)
        r = mag * np.exp(1j * phase)
        return np.where(propagating, r, 0.0).astype(np.complex128)

    def _sin_theta_map(self, fs_x: float, fs_y: float) -> NDArray[np.float64]:
        sin_x, sin_y = mask_side_sin_theta(fs_x, fs_y, self._optics)
        sx = sin_x + self._doff_x
        sy = sin_y + self._doff_y
        return np.sqrt(sx * sx + sy * sy)

    def base_spectrum(self) -> NDArray[np.complex128]:
        """The un-reflected thin-mask spectrum.

        Present so callers that only want the Kirchhoff object can have it;
        the imaging path uses :meth:`spectrum_for`.
        """
        return self._base

    def spectrum_for(self, fs_x: float, fs_y: float) -> NDArray[np.complex128]:
        return self._base * self._reflection(self._sin_theta_map(fs_x, fs_y))

    def clear_intensity(self, fs_x: float, fs_y: float) -> float:
        """What an unpatterned blank returns for this source point.

        The DC order leaves the mask along the specular direction, so this is
        the blank reflectance at the incidence angle — 75 % for an EUV mirror at
        6°, not 1.
        """
        sin_x, sin_y = mask_side_sin_theta(fs_x, fs_y, self._optics)
        s = math.hypot(sin_x, sin_y)
        r = self._reflection(np.array([s], dtype=np.float64))[0]
        return float(abs(r) ** 2)


class FDTDSpectra:
    """Near fields from a rigorous solve, interpolated across the source.

    The expensive part happens once, in ``__init__``: an FDTD solve on a coarse
    grid of illumination angles. Each Abbe source point then costs an
    interpolation and a lift onto the FFT grid.

    Blank normalisation is what makes the interpolation legitimate. The stored
    near fields are divided by an unpatterned solve at the same angle, which
    removes the sharpest angular structure — the mirror's Bragg response —
    leaving something the absorber varies only slowly over. The absolute
    throughput comes back separately in :meth:`clear_intensity`, from the
    transfer-matrix solution of the same blank.
    """

    is_angle_independent = False

    def __init__(
        self,
        optics: OpticsConfig,
        grid: GridConfig,
        stack: MaskStack,
        geometry: MaskGeometry,
        progress=None,
        library=None,
    ) -> None:
        from litho_sim.expose.m3d.nearfield import build_library

        self._optics = optics
        self._grid = grid
        self._stack = stack
        self._orientation = geometry.orientation
        if library is not None:
            # Restored from disk. The topography is not kept: nothing outside
            # the build reads it, and it is the one part that is cheap to
            # redraw from geometry if it is ever wanted again.
            self._library, self._topo = library, None
        else:
            self._library, self._topo = build_library(
                stack, geometry, optics, grid,
                n_angles=max(int(optics.m3d_angles), 2),
                progress=progress,
            )
        # The open-frame throughput comes from the library's own blank solves.
        # Asking coat.films for it instead would return the blank's
        # *reflection*, which is the right quantity for a mirror and the wrong
        # one for a mask you look through — an error that rescales a
        # transmissive image by more than an order of magnitude.

    @property
    def n_solves(self) -> int:
        """FDTD solves the library cost — the number worth quoting."""
        return self._library.n_solves

    def base_spectrum(self) -> NDArray[np.complex128]:
        """The on-axis near-field spectrum, for callers that want one array."""
        return self.spectrum_for(0.0, 0.0)

    def spectrum_for(self, fs_x: float, fs_y: float) -> NDArray[np.complex128]:
        from litho_sim.expose.m3d.nearfield import spectrum_on_imaging_grid

        # Normalised source coordinates are what the library is indexed by.
        scale = self._optics.NA / self._optics.wavelength
        sx = float(np.clip(fs_x / scale, -1.0, 1.0))
        sy = float(np.clip(fs_y / scale, -1.0, 1.0))
        coeff = self._library.interpolate(sx, sy)
        return spectrum_on_imaging_grid(coeff, self._grid, self._orientation)

    def clear_intensity(self, fs_x: float, fs_y: float) -> float:
        scale = self._optics.NA / self._optics.wavelength
        sx = float(np.clip(fs_x / scale, -1.0, 1.0))
        sy = float(np.clip(fs_y / scale, -1.0, 1.0))
        return float(abs(self._library.interpolate_blank(sx, sy)) ** 2)


#: Near-field libraries already built, keyed by everything that would change
#: one.  A process-window sweep calls ``compute_aerial_image`` hundreds of
#: times with the same mask and optics and a different dose or defocus, and
#: rebuilding a library for each would make the feature unusable.
_LIBRARY_CACHE: dict[Any, FDTDSpectra] = {}
_CACHE_LIMIT = 4

#: Repo root, from ``src/litho_sim/expose/m3d/provider.py``.
_ROOT = Path(__file__).resolve().parents[4]

#: Solved libraries, kept between runs.  Lives beside the device presets and is
#: gitignored for the same reason: rebuildable, machine-local, and large.
LIBRARY_CACHE_DIR = _ROOT / "presets" / "m3d"

#: Bumped when a stored library stops meaning what a current one does — a
#: change to the solver, the topography, or the normalisation.  Older files are
#: treated as absent rather than trusted, because a silently stale near field
#: is indistinguishable from a correct one until the image is wrong.
LIBRARY_CACHE_VERSION = 1


def _library_key(optics: OpticsConfig, grid: GridConfig, spec, geom) -> Any:
    def freeze(d):
        if isinstance(d, dict):
            return tuple(sorted((k, freeze(v)) for k, v in d.items()))
        return d

    return (
        optics.wavelength, optics.NA, optics.reduction,
        optics.chief_ray_deg, optics.chief_ray_axis, optics.m3d_angles,
        grid.n_pixels, grid.pixel_size,
        freeze(spec), freeze(geom),
    )


def _disk_name(key: Any) -> str:
    """Stable filename for a library key.

    ``hash()`` is unusable here: it is randomised per interpreter for strings,
    so the same configuration would miss its own cache on the next launch —
    which is exactly the run the cache exists for.
    """
    payload = json.dumps(key, sort_keys=True, default=str).encode()
    return f"m3d-v{LIBRARY_CACHE_VERSION}-{hashlib.sha256(payload).hexdigest()[:16]}.npz"


def _load_library(path: Path):
    """Read a stored library, or return ``None`` if it is missing or unusable."""
    from litho_sim.expose.m3d.nearfield import NearFieldLibrary

    if not path.is_file():
        return None
    try:
        with np.load(path) as z:
            lib = NearFieldLibrary(
                sigma=z["sigma"], fields=z["fields"], blank=z["blank"]
            )
    except Exception:
        # A truncated file from an interrupted write, or one written by an
        # incompatible numpy. Either way the answer is to solve again.
        logger.warning("ignoring unreadable near-field cache %s", path.name)
        return None
    logger.info("near-field library restored from %s", path.name)
    return lib


def _save_library(path: Path, library) -> None:
    """Store a solved library, atomically.  Failure to cache is not an error."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-write must not leave a half file that
        # the next run reads as a valid library.
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, sigma=library.sigma, fields=library.fields, blank=library.blank
        )
        tmp.replace(path)
        logger.info("near-field library cached → %s", path.name)
    except Exception:
        logger.warning("could not cache the near-field library", exc_info=True)


def library_is_cached(optics: OpticsConfig, grid: GridConfig) -> bool:
    """Whether a rigorous solve for this configuration has already been done.

    The app's cost model asks this before deciding whether a change can be
    applied live: an uncached library is minutes of work, a cached one is an
    interpolation.
    """
    spec = optics.mask_stack
    if optics.mask_model.lower() != "fdtd" or optics.mask_geometry is None:
        return True
    key = _library_key(optics, grid, spec, optics.mask_geometry)
    if key in _LIBRARY_CACHE:
        return True
    # A library on disk counts as cached: reading it is milliseconds, so the
    # scheduler should treat the change as live rather than deferring it.
    return (LIBRARY_CACHE_DIR / _disk_name(key)).is_file()


def clear_library_cache(disk: bool = False) -> None:
    """Drop cached near-field libraries.  Mainly for tests.

    Parameters
    ----------
    disk : bool
        Also delete the stored libraries. Off by default so a test that clears
        the in-memory cache to force a rebuild does not throw away minutes of
        the user's solved work as a side effect.
    """
    _LIBRARY_CACHE.clear()
    if disk and LIBRARY_CACHE_DIR.is_dir():
        for f in LIBRARY_CACHE_DIR.glob("m3d-v*.npz"):
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_spectrum_provider(
    mask: NDArray,
    optics: OpticsConfig,
    grid: GridConfig,
    mask_stack: dict[str, Any] | None = None,
) -> SpectrumProvider:
    """Build the spectrum provider named by ``optics.mask_model``.

    Parameters
    ----------
    mask : NDArray
        Wafer-scale mask transmittance, ``(n, n)``, real or complex.
    optics, grid : OpticsConfig, GridConfig
        Optical and numerical configuration.
    mask_stack : dict, optional
        Overrides ``optics.mask_stack``.

    Returns
    -------
    SpectrumProvider

    Raises
    ------
    ValueError
        For an unrecognised ``mask_model``.
    NotImplementedError
        For a model that is named but not yet built.
    """
    model = optics.mask_model.lower()
    if model not in MASK_MODELS:
        raise ValueError(
            f"mask_model must be one of {list(MASK_MODELS)}, got "
            f"'{optics.mask_model}'"
        )
    if model == "thin":
        return ThinMaskSpectra(mask)

    spec = mask_stack if mask_stack is not None else optics.mask_stack
    stack = (
        MaskStack.from_spec(spec)
        if spec is not None
        else MaskStack.for_wavelength(optics.wavelength)
    )

    if model == "multilayer":
        if not stack.is_reflective:
            raise ValueError(
                "mask_model='multilayer' describes the Bragg mirror under an EUV "
                f"mask, but the stack regime is '{stack.regime}'. Use "
                "mask_model='thin' or 'fdtd' for a transmissive mask."
            )
        return MultilayerSpectra(mask, optics, grid, stack)

    # mask_model == "fdtd"
    from litho_sim.expose.m3d.nearfield import MaskGeometry

    if optics.mask_geometry is None:
        raise ValueError(
            "mask_model='fdtd' needs OpticsConfig.mask_geometry — "
            "{'pitch': ..., 'cd': ..., 'orientation': ...} in wafer-side metres. "
            "It cannot be read back off the mask array: the solver grid is "
            "sub-nanometre on the reticle, so a 4 nm raster would quantise every "
            "absorber edge to 16 nm mask-side, and edge position is the whole "
            "point of a thick-mask model."
        )
    key = _library_key(optics, grid, spec, optics.mask_geometry)
    cached = _LIBRARY_CACHE.get(key)
    if cached is not None:
        return cached

    # Then disk. The solve is minutes and depends on nothing that changes
    # between runs, so paying it once per machine rather than once per launch
    # is the difference between a usable feature and one nobody turns on twice.
    geometry = MaskGeometry(**optics.mask_geometry)
    path = LIBRARY_CACHE_DIR / _disk_name(key)
    library = _load_library(path)
    provider = FDTDSpectra(optics, grid, stack, geometry, library=library)
    if library is None:
        _save_library(path, provider._library)

    if len(_LIBRARY_CACHE) >= _CACHE_LIMIT:
        _LIBRARY_CACHE.pop(next(iter(_LIBRARY_CACHE)))
    _LIBRARY_CACHE[key] = provider
    return provider
