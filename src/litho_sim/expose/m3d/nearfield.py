"""From mask geometry to a diffraction spectrum, by way of Maxwell.

This is the bridge between :mod:`~litho_sim.expose.m3d.yee`, which knows about
fields on a grid, and :mod:`litho_sim.expose.aerial_image`, which wants a
spectrum on an FFT grid. Three things happen here, and each is a place a
plausible-but-wrong answer could come from.

**The topography is built from geometry, not from the mask array.** The FDTD
grid is a few nanometres on the reticle — sub-nanometre once divided by the
reduction — while the imaging grid is 4 nm at the wafer, so rasterising first
and upsampling would quantise every absorber edge to 16 nm mask-side. Edge
position is the entire subject here, so the permittivity map is drawn directly
from ``(pitch, cd, sidewall)`` at solver resolution.

**The tilt comes out.** The Abbe loop shifts the pupil rather than the spectrum,
so what it wants back is the spectrum referenced to *normal* incidence. The
near field is divided by the incident plane-wave phase before being
transformed. Skipping this does not produce a subtly worse image; it produces a
displaced one.

**Everything is divided by the blank.** The near field is normalised against an
unpatterned stack solved at the same angle, which cancels the source-to-monitor
propagation, the injection amplitude and the entry Fresnel in one stroke, and
removes any question about where exactly the reference plane sits. What is left
is a dimensionless transmission that equals 1 in open areas. The absolute
throughput comes back separately, from the transfer-matrix solution of the same
blank — see :meth:`FDTDSpectra.clear_intensity`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.expose.m3d.materials import mask_material
from litho_sim.expose.m3d.stack import MaskStack
from litho_sim.expose.m3d.yee import solve_near_field

logger = logging.getLogger(__name__)

#: Worker processes for the angle sweep.  The solves are independent, and numpy
#: holds the GIL through the update kernel — measured at 1.09x on threads
#: against 4.2x on processes — so this has to be processes to be worth anything.
#: Two cores are left for the GUI and the OS.  ``LITHO_SIM_M3D_WORKERS`` sets it
#: explicitly; 1 forces the serial path.
_DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 2)

#: Cells per wavelength in the densest medium.  20 is the usual rule of thumb;
#: the solver's own convergence test measures 0.14 % against Fresnel at 48.
_CELLS_PER_WAVELENGTH = 24.0

#: Width in cells of the unpatterned reference run.  The blank is homogeneous
#: in x, so its near field is constant there and a sliver costs a fraction of
#: the patterned solve while dividing out everything the two have in common.
_BLANK_CELLS = 8


@dataclass(frozen=True)
class MaskGeometry:
    """The drawn pattern, in wafer-side dimensions.

    Carried separately from the rasterised mask array because the array has
    already thrown away the sub-pixel edge position the solver needs. ``pitch``
    and ``cd`` are what the pattern generator was called with.

    Parameters
    ----------
    pitch, cd : float
        Wafer-side pitch and linewidth [m].
    orientation : str
        ``"vertical"`` (lines run along y, pattern varies in x) or
        ``"horizontal"``.
    """

    pitch: float
    cd: float
    orientation: str = "vertical"

    def __post_init__(self) -> None:
        if self.pitch <= 0 or self.cd <= 0:
            raise ValueError(f"pitch and cd must be positive, got {self.pitch}, {self.cd}")
        if self.cd >= self.pitch:
            raise ValueError(
                f"cd {self.cd * 1e9:.1f} nm must be smaller than pitch "
                f"{self.pitch * 1e9:.1f} nm"
            )
        if self.orientation not in ("vertical", "horizontal"):
            raise ValueError(
                f"orientation must be 'vertical' or 'horizontal', got "
                f"'{self.orientation}'"
            )

    def mask_side(self, reduction: float) -> tuple[float, float]:
        """Pitch and CD on the reticle [m]."""
        return self.pitch * reduction, self.cd * reduction


# ---------------------------------------------------------------------------
# Topography
# ---------------------------------------------------------------------------


def _dz_for_multilayer(ml, dz_target: float) -> float:
    """Pick a cell height that divides the Bragg period exactly.

    A Mo/Si mirror is a resonator with a bandwidth of about 5 %, so a period
    quantised by the grid is a *detuned* mirror. Measured: at an unconstrained
    ``dz`` of 0.562 nm the 6.94 nm period rounds to 6.744 nm — a 2.8 % error —
    and the reflectance of a 12-bilayer stack falls from 46.3 % to 28.2 %. The
    solver reproduced that 28.2 % faithfully, which is the point: the error was
    never in the physics, it was in what the physics was handed.

    So ``dz`` is chosen as ``period / K``, which makes the period exact by
    construction, with *K* picked to also land the Mo/Si split on whole cells.
    At the standard 6.94 nm period and gamma 0.4, ``K = 25`` gives both exactly.
    """
    best, best_err = dz_target, np.inf
    k_min = max(int(np.ceil(ml.period / dz_target)), 4)
    for k in range(k_min, k_min + 40):
        dz = ml.period / k
        gamma_err = abs(round(ml.gamma * k) / k - ml.gamma)
        if round(ml.gamma * k) in (0, k):
            continue  # a layer would vanish
        if gamma_err < best_err - 1e-12:
            best, best_err = dz, gamma_err
        if gamma_err == 0.0:
            break
    logger.debug(
        "Multilayer grid: dz = %.4f nm (period/%d), gamma error %.3f",
        best * 1e9, round(ml.period / best), best_err,
    )
    return best


@dataclass
class Topography:
    """A permittivity map and where to inject into and read from it."""

    index: NDArray[np.complex128]
    source_row: int
    monitor_row: int
    dx: float
    dz: float


def absorber_centre(pitch: float, cd: float) -> float:
    """Where the absorber sits, in coordinates measured from the field centre.

    This has to agree with :func:`litho_sim.mask.patterns.lines_and_spaces`
    exactly, and two things about that function are easy to get backwards:

    * Its ``cd`` is the width of the **clear** line, not the absorber. The
      absorber is therefore ``pitch - cd`` wide.
    * It phases the pattern as ``(coords + pitch/2) mod pitch < cd`` with
      ``coords`` measured from the *field centre*, which puts the absorber
      centred at ``cd/2``, not at zero.

    Both errors are invisible at 50 % duty with a symmetric field — the width
    is right because ``pitch - cd == cd``, and the offset can look like a whole
    pitch. They show up immediately at any other duty cycle, and an offset
    between the topography and the mask array is a rigid image shift that would
    swamp the couple of nanometres of real placement error the whole model
    exists to compute.
    """
    return cd / 2.0


def _absorber_occupancy(
    coords: NDArray[np.float64], pitch: float, cd: float, dx: float
) -> NDArray[np.float64]:
    """Fraction of each cell covered by absorber, on the mask's own phasing.

    *coords* are measured from the field centre, matching ``lines_and_spaces``.
    The result is a fraction rather than a boolean so an edge that moves by
    less than a cell still moves the map — which is the whole reason the
    topography is built from geometry instead of from a rasterised array.
    """
    half = (pitch - cd) / 2.0
    offset = coords - absorber_centre(pitch, cd)
    d = np.abs((offset + pitch / 2.0) % pitch - pitch / 2.0)
    return np.clip((half - d) / dx + 0.5, 0.0, 1.0)


def build_topography(
    stack: MaskStack,
    geometry: MaskGeometry,
    optics: OpticsConfig,
    grid: GridConfig,
    dx: float | None = None,
    dz: float | None = None,
    n_pml: int = 16,
) -> Topography:
    """Draw the mask cross-section at solver resolution.

    The domain spans exactly the imaging field width on the reticle
    (``n_pixels * pixel_size * reduction``), so the solver's periodic cell and
    the imaging FFT's implicit periodicity are the same cell — which is what
    lets the resulting spectrum drop into the Abbe loop with no resampling.

    Returns
    -------
    Topography
    """
    lam = optics.wavelength
    m_pitch, m_cd = geometry.mask_side(optics.reduction)
    field = grid.n_pixels * grid.pixel_size * optics.reduction

    n_abs = mask_material(stack.absorber, lam)
    n_sub = mask_material(stack.substrate, lam)
    n_vac = mask_material("vacuum", lam)
    n_dense = max(abs(n_abs), abs(n_sub), 1.0)

    if dx is None:
        dx = lam / (_CELLS_PER_WAVELENGTH * n_dense)
        nx = max(int(round(field / dx)), 16)
        dx = field / nx
    else:
        nx = max(int(round(field / dx)), 16)
        dx = field / nx
    if dz is None:
        dz = lam / (_CELLS_PER_WAVELENGTH * n_dense)
        if stack.is_reflective and stack.multilayer is not None:
            dz = _dz_for_multilayer(stack.multilayer, dz)

    t_abs = stack.absorber_thickness
    n_abs_cells = max(int(round(t_abs / dz)), 1)
    pad = max(int(round(0.6 * lam / dz)), 12)  # clearance so monitors sit clear of PML

    # Measured from the field centre, matching lines_and_spaces.
    coords = (np.arange(nx) - nx // 2) * dx

    if stack.is_reflective:
        # Vacuum, then the absorber, then the mirror. The near field of
        # interest is the *reflected* one, so the monitor sits above the
        # injection plane where only scattered field is stored.
        ml = stack.multilayer
        assert ml is not None
        t_mo = ml.period * ml.gamma
        t_si = ml.period - t_mo
        cap_cells = max(int(round(ml.cap_thickness / dz)), 1)
        si_cells = max(int(round(t_si / dz)), 1)
        mo_cells = max(int(round(t_mo / dz)), 1)
        ml_cells = ml.n_bilayers * (si_cells + mo_cells)

        top = n_pml + pad          # monitor sits here, in the scattered region
        src = top + pad            # injection plane
        abs_top = src + pad
        rows = abs_top + n_abs_cells + cap_cells + ml_cells + pad + n_pml
        index = np.full((rows, nx), n_vac, dtype=complex)

        z0 = abs_top + n_abs_cells + cap_cells
        index[abs_top + n_abs_cells: z0, :] = mask_material(ml.cap_material, lam)
        n_si = mask_material("Si", lam)
        n_mo = mask_material("Mo", lam)
        z = z0
        for _ in range(ml.n_bilayers):
            index[z: z + si_cells, :] = n_si
            z += si_cells
            index[z: z + mo_cells, :] = n_mo
            z += mo_cells
        index[z:, :] = n_si
        monitor = top
        absorber_top = abs_top
    else:
        # Quartz above, absorber, vacuum below; the near field is transmitted.
        sub_cells = max(int(round(0.8 * lam / dz)), 20)
        src = n_pml + pad
        quartz_top = src + pad
        abs_top = quartz_top + sub_cells
        below = pad + max(int(round(0.4 * lam / dz)), 12)
        rows = abs_top + n_abs_cells + below + n_pml
        index = np.full((rows, nx), n_vac, dtype=complex)
        index[quartz_top:abs_top, :] = n_sub
        monitor = abs_top + n_abs_cells + max(int(round(0.15 * lam / dz)), 4)
        absorber_top = abs_top

    # The absorber itself, with its sidewall. A trapezoid: full CD at the
    # bottom, narrowing upward when the wall leans in.
    eps_vac, eps_abs = n_vac ** 2, n_abs ** 2
    slope = np.tan(np.radians(stack.sidewall_deg))
    for j in range(n_abs_cells):
        height = (n_abs_cells - j - 0.5) * dz  # above the absorber base
        # A leaning wall narrows the absorber upward. Expressed as an effective
        # clear width so the phasing stays the mask's.
        widen = 0.0 if not np.isfinite(slope) else 2.0 * height / slope
        frac = _absorber_occupancy(coords, m_pitch, min(m_cd + widen, m_pitch), dx)
        # Subpixel averaging, so a moving edge moves smoothly rather than in
        # whole-cell jumps — the same reason rasterize_shapes supersamples.
        #
        # The average is taken over *permittivity*, which is both the standard
        # choice and the safe one. Averaging the index instead walks a metal
        # edge through indices whose permittivity passes near zero, and a cell
        # with eps' ~ 0 has an enormous phase velocity that destabilises the
        # march. Chrome found this immediately; quartz never would.
        eps = eps_vac * (1.0 - frac) + eps_abs * frac
        index[absorber_top + j, :] = np.sqrt(eps)

    return Topography(index=index, source_row=src, monitor_row=monitor, dx=dx, dz=dz)


def _blank_topography(topo: Topography, stack: MaskStack, optics: OpticsConfig,
                      n_pml: int) -> Topography:
    """The same stack with the pattern taken out, a few cells wide.

    Homogeneous in x, so it needs no width; a sliver divides out the injection
    amplitude, the propagation to the monitor and the entry Fresnel for a few
    percent of the patterned solve.
    """
    lam = optics.wavelength
    n_vac = mask_material("vacuum", lam)
    idx = topo.index[:, :1].copy()
    # Wherever the patterned map differs across x, the blank takes the open value.
    varying = np.ptp(np.abs(topo.index), axis=1) > 1e-12
    idx[varying, 0] = n_vac
    return Topography(
        index=np.repeat(idx, _BLANK_CELLS, axis=1),
        source_row=topo.source_row,
        monitor_row=topo.monitor_row,
        dx=topo.dx,
        dz=topo.dz,
    )


# ---------------------------------------------------------------------------
# One angle
# ---------------------------------------------------------------------------


def near_field_coefficients(
    topo: Topography,
    blank: Topography,
    optics: OpticsConfig,
    sin_x: float,
    sin_y: float,
    *,
    n_pml: int = 16,
    steady_tol: float = 2e-3,
    max_periods: int = 300,
) -> tuple[NDArray[np.complex128], complex]:
    """Blank-normalised, tilt-removed near field for one mask-side angle.

    Returns
    -------
    (coefficients, blank) : NDArray, complex
        The near field relative to an unpatterned stack — 1 in open areas — and
        the blank's own amplitude relative to the incident wave.

        The second value is not a diagnostic. It *is* the open-frame throughput
        for this angle, and it has to come from the same solve: asking the
        transfer-matrix method instead would give the blank's *reflection*,
        which for a transmissive mask is the wrong quantity entirely and
        rescales the whole image by an order of magnitude.
    """
    lam = optics.wavelength
    k0 = 2.0 * np.pi / lam
    kx, ky = k0 * sin_x, k0 * sin_y
    pol = (0.0, 1.0, 0.0)  # E along the lines; the conical terms handle the rest

    patterned = solve_near_field(
        topo.index, topo.dx, topo.dz, lam, kx=kx, ky=ky, polarisation=pol,
        source_row=topo.source_row, monitor_row=topo.monitor_row,
        n_pml=n_pml, steady_tol=steady_tol, max_periods=max_periods,
    )
    reference = solve_near_field(
        blank.index, blank.dx, blank.dz, lam, kx=kx, ky=ky, polarisation=pol,
        source_row=blank.source_row, monitor_row=blank.monitor_row,
        n_pml=n_pml, steady_tol=steady_tol, max_periods=max_periods,
    )

    # Remove the incident tilt from both. Without this the plain average below
    # is the mean of a complex exponential over its period, which is zero.
    x_p = np.arange(patterned.ey.size) * topo.dx
    x_b = np.arange(reference.ey.size) * blank.dx
    field = patterned.ey * np.exp(-1j * kx * x_p)
    ref = complex((reference.ey * np.exp(-1j * kx * x_b)).mean())

    if abs(ref) < 1e-12:
        raise RuntimeError(
            "the unpatterned reference returned no field; the blank solve did "
            "not converge or the monitor is inside an opaque layer"
        )
    return (field / ref).astype(np.complex128), ref


# ---------------------------------------------------------------------------
# The angle library
# ---------------------------------------------------------------------------


@dataclass
class NearFieldLibrary:
    """Near fields on a coarse grid of illumination angles, for interpolation.

    An FDTD solve per Abbe source point would be ~200 solves per image. The
    near field varies smoothly with angle once the blank response is divided
    out — that division is what makes a coarse grid sufficient, because the
    sharp angular structure belongs to the mirror, not to the absorber — so a
    handful of solves and bilinear interpolation stands in.

    Attributes
    ----------
    sigma : NDArray
        Grid of normalised source coordinates the library was solved on.
    fields : NDArray
        ``(n_sigma, n_sigma, nx)`` complex, blank-normalised.
    """

    sigma: NDArray[np.float64]
    fields: NDArray[np.complex128]
    blank: NDArray[np.complex128]

    @property
    def n_solves(self) -> int:
        return int(self.fields.shape[0] * self.fields.shape[1])

    def _weights(self, sx: float, sy: float):
        s = self.sigma
        gx = float(np.interp(sx, s, np.arange(s.size)))
        gy = float(np.interp(sy, s, np.arange(s.size)))
        i0, j0 = int(np.floor(gx)), int(np.floor(gy))
        i1 = min(i0 + 1, s.size - 1)
        j1 = min(j0 + 1, s.size - 1)
        fx, fy = gx - i0, gy - j0
        return i0, i1, j0, j1, fx, fy

    def interpolate(self, sx: float, sy: float) -> NDArray[np.complex128]:
        """Bilinear interpolation in ``(sigma_x, sigma_y)``, clamped at the edges."""
        i0, i1, j0, j1, fx, fy = self._weights(sx, sy)
        f = self.fields
        return (
            f[j0, i0] * (1 - fx) * (1 - fy)
            + f[j0, i1] * fx * (1 - fy)
            + f[j1, i0] * (1 - fx) * fy
            + f[j1, i1] * fx * fy
        ).astype(np.complex128)

    def interpolate_blank(self, sx: float, sy: float) -> complex:
        """Open-frame amplitude for one source point."""
        i0, i1, j0, j1, fx, fy = self._weights(sx, sy)
        b = self.blank
        return complex(
            b[j0, i0] * (1 - fx) * (1 - fy)
            + b[j0, i1] * fx * (1 - fy)
            + b[j1, i0] * (1 - fx) * fy
            + b[j1, i1] * fx * fy
        )


def _solve_one_angle(args):
    """One library entry, as a picklable top-level call.

    Defined at module scope because a process pool has to pickle the callable
    by reference, which rules out the closure this would otherwise be.
    """
    jy, ix, topo, blank, optics, mx, my, n_pml, steady_tol = args
    field, ref = near_field_coefficients(
        topo, blank, optics, mx, my, n_pml=n_pml, steady_tol=steady_tol
    )
    return jy, ix, field, ref


def build_library(
    stack: MaskStack,
    geometry: MaskGeometry,
    optics: OpticsConfig,
    grid: GridConfig,
    n_angles: int = 3,
    *,
    n_pml: int = 16,
    steady_tol: float = 2e-3,
    progress=None,
    workers: int | None = None,
) -> tuple[NearFieldLibrary, Topography]:
    """Solve the near field across the illumination pupil.

    Parameters
    ----------
    n_angles : int
        Side length of the angle grid. Cost is its square, so 3 is the default
        and 5 is already 25 solves.
    progress : callable, optional
        Called as ``progress(done, total)``.
    workers : int, optional
        Solver processes. The angle solves share nothing, so this is the one
        place in the engine where a pool is straightforwardly correct. Defaults
        to :data:`_DEFAULT_WORKERS`; 1 runs the sweep in-process.
    """
    if n_angles < 2:
        raise ValueError(f"n_angles must be at least 2, got {n_angles}")

    topo = build_topography(stack, geometry, optics, grid, n_pml=n_pml)
    blank = _blank_topography(topo, stack, optics, n_pml)
    sigma = np.linspace(-1.0, 1.0, n_angles)

    nx = topo.index.shape[1]
    fields = np.empty((n_angles, n_angles, nx), dtype=np.complex128)
    blanks = np.empty((n_angles, n_angles), dtype=np.complex128)
    total = n_angles * n_angles

    from litho_sim.expose.m3d.provider import mask_side_sin_theta

    tasks = []
    for jy, sy in enumerate(sigma):
        for ix, sx in enumerate(sigma):
            # Normalised source coordinate -> mask-side direction sine, via the
            # same conversion the Abbe loop's frequency offset would give.
            fs_x = sx * optics.NA / optics.wavelength
            fs_y = sy * optics.NA / optics.wavelength
            mx, my = mask_side_sin_theta(fs_x, fs_y, optics)
            tasks.append((jy, ix, topo, blank, optics, mx, my, n_pml, steady_tol))

    if workers is None:
        workers = int(os.environ.get("LITHO_SIM_M3D_WORKERS", _DEFAULT_WORKERS))
    workers = max(1, min(int(workers), total))

    done = 0
    for jy, ix, field, ref in _run_solves(tasks, workers):
        fields[jy, ix] = field
        blanks[jy, ix] = ref
        done += 1
        if progress is not None:
            progress(done, total)

    logger.info(
        "Near-field library: %d solves on a %dx%d angle grid, %d cells wide, "
        "%d worker%s",
        total, n_angles, n_angles, nx, workers, "" if workers == 1 else "s",
    )
    return NearFieldLibrary(sigma=sigma, fields=fields, blank=blanks), topo


def _run_solves(tasks, workers: int):
    """Yield solved angles, in whatever order they finish.

    Results carry their own grid indices, so completion order is irrelevant to
    the caller.

    A pool that cannot start at all — a frozen build, a sandbox that forbids
    spawning — degrades to the serial path rather than failing. That fallback
    is deliberately confined to the window before the first result is handed
    out: once a solve has been yielded, re-running the sweep would deliver
    duplicates, and a solver error that surfaces mid-sweep is a real failure
    worth raising rather than papering over with a slow retry.
    """
    if workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            yield _solve_one_angle(t)
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # "spawn" rather than "fork": this is routinely called from the app's worker
    # thread, and forking a process that already holds Qt and a thread pool is
    # the classic way to get a child deadlocked on a lock nobody will release.
    # Both console entry points guard their ``main()``, so the re-import a
    # spawned child does is inert.
    ctx = mp.get_context("spawn")
    started = False
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(_solve_one_angle, t) for t in tasks]
            for fut in as_completed(futures):
                result = fut.result()
                started = True
                yield result
        return
    except Exception:  # pragma: no cover - platform-dependent
        if started:
            raise
        logger.warning(
            "could not start a solver pool; falling back to one process",
            exc_info=True,
        )

    for t in tasks:
        yield _solve_one_angle(t)


# ---------------------------------------------------------------------------
# Onto the imaging grid
# ---------------------------------------------------------------------------


def spectrum_on_imaging_grid(
    coefficients: NDArray[np.complex128],
    grid: GridConfig,
    orientation: str = "vertical",
) -> NDArray[np.complex128]:
    """Lift a 1-D near field onto the 2-D FFT grid the Abbe loop multiplies.

    Two steps, both exact rather than approximate:

    **Band-limited truncation, not subsampling.** The solver grid is far finer
    than the imaging grid, so taking every *n*-th sample would alias everything
    above the imaging Nyquist onto the orders the pupil collects. Keeping the
    low-frequency coefficients instead discards exactly what the lens discards.

    **A one-dimensional pattern has a one-dimensional spectrum.** For a mask
    invariant along y, ``fft2`` puts every non-zero coefficient in row 0 and
    scales by N. Building the array that way is not an approximation of the 2-D
    transform; it is the 2-D transform.
    """
    n = grid.n_pixels
    fine = np.fft.fft(coefficients) / coefficients.size  # normalised coefficients
    out = np.zeros((n, n), dtype=np.complex128)

    # Keep |k| <= n/2, in FFT order, from a longer array with the same physical
    # extent — so bin k means the same spatial frequency in both.
    half = n // 2
    line = np.zeros(n, dtype=np.complex128)
    line[:half + 1] = fine[:half + 1]
    if half > 0:
        line[-half + 1:] = fine[-half + 1:] if half > 1 else fine[-1:]

    line *= float(n) * float(n)  # match np.fft.fft2's unnormalised convention
    if orientation == "vertical":
        out[0, :] = line
    else:
        out[:, 0] = line
    return out
