"""A 2.5-D finite-difference time-domain solver for the field around a mask absorber.

This is the rigorous end of the mask model: Maxwell's equations discretised on a
Yee grid in a volume around the absorber, marched to steady state, and read off
on a plane just past the topography. That plane's amplitude and phase — the
**near field** — then plays the part the thin mask's transmittance used to.

Why the usual FDTD compromises do not apply here
------------------------------------------------
Textbook FDTD is broadband: a pulse goes in, a Fourier transform comes out. That
choice is what makes oblique incidence on a periodic cell awkward, because the
Bloch condition ``E(x+A) = E(x) e^{i k_x A}`` then wants a field from a
*different time*, and the usual fixes (split-field, sine-cosine) cost either a
degraded time step or a second set of state variables.

Lithography is monochromatic. Running one frequency with **complex fields**
makes the Bloch factor an instantaneous multiplication — causal, no time shift,
no change to the Courant limit — and the steady-state phasor, which is precisely
the amplitude and phase the near-field plane is defined as, drops straight out.
The cost is complex arithmetic, which numpy does anyway.

2.5-D, not TE/TM
----------------
A line/space mask is invariant along ``y``, which invites solving the two
decoupled 2-D polarisations. That is wrong for any source point off the ``x``
axis — which is most of an annular or quadrupole illuminator. With ``k_y != 0``
the problem is *conical* and all six field components couple.

So the grid is 2-D in ``(x, z)`` and ``d/dy`` is the exact multiplication
``i k_y``, with ``k_y`` fixed by the incident angle. Nothing is approximated by
this: it is a Fourier transform in the invariant direction, and it reduces to
the decoupled TE/TM systems at ``k_y = 0``. It costs roughly twice a single
polarisation and almost nothing in the time step, because ``k_y`` enters the
stability limit as an exact ``k_y^2/4`` rather than through a difference.

Grid layout
-----------
Arrays are ``(nz, nx)``, indexed ``[k, i]``, with the Yee half-cell offsets::

    Ey  (i,     k    )      Hy  (i+1/2, k+1/2)
    Ex  (i+1/2, k    )      Hx  (i,     k+1/2)
    Ez  (i,     k+1/2)      Hz  (i+1/2, k    )

Boundaries are Bloch-periodic in ``x`` (exact, and free — it is what ``np.roll``
already does, times a phase) and CPML in ``z``.

Conventions
-----------
Time dependence is ``exp(-i omega t)``. Under that convention a wave decays
going forward when the refractive index is ``n + ik`` with **positive** ``k``,
which is exactly how :data:`litho_sim.wafer.materials.MATERIAL_LIBRARY` and
:data:`~litho_sim.expose.m3d.materials.MASK_MATERIALS` store it — so indices go
in as stored, with no conjugation.

:mod:`litho_sim.coat.films` uses the opposite time convention (it conjugates to
``n - ik`` before propagating). The two agree on every *magnitude* and give
conjugate *phases*, so comparisons against it either use ``abs`` or conjugate
one side. Reflectances match; reflection phases come out with opposite sign.

Loss is a conductivity, not a complex permittivity
--------------------------------------------------
A complex ``eps`` describes absorption in the frequency domain and cannot be
dropped into a time-domain update: the resulting coefficient has modulus above
one, and the march diverges within a few hundred steps. Absorbing materials are
instead split into a real permittivity and a conductivity::

    eps_c = (n + ik)^2 = (n^2 - k^2) + i(2nk)
    eps'  = n^2 - k^2                  sigma = omega * eps0 * 2nk

and stepped with the standard semi-implicit lossy update, which is
unconditionally stable for positive sigma. This matters here because every
mask absorber worth simulating is strongly lossy: TaBN at 13.5 nm and chrome at
193 nm both have ``k`` comparable to ``n``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

C0 = 299792458.0
ETA0 = 376.730313668

#: Fraction of the Courant limit to run at.  Matches the safety factor the
#: explicit reaction-diffusion bake uses, and for the same reason: the limit is
#: exact only for a uniform lossless grid, and neither of ours is.
_SAFETY = 0.99

#: Smallest real permittivity treated as a dielectric.  Below this a cell is
#: given a Drude pole instead — see the discussion at its use.  It also floors
#: the Courant scaling, so a near-zero edge cell cannot demand dt -> 0.
_EPS_FLOOR = 0.05

#: CPML grading.  m=3 with a target reflection of 1e-6 is the standard choice
#: and measures below -60 dB in the test that checks it.
_PML_M = 3.0
_PML_R0 = 1e-6
_PML_KAPPA_MAX = 5.0
_PML_ALPHA_MAX = 0.05


@dataclass
class FDTDResult:
    """Steady-state phasors on the monitor plane, and how hard they were to get.

    Attributes
    ----------
    ex, ey, ez : NDArray
        Complex field phasors along the monitor line, shape ``(nx,)``.
    steps : int
        Time steps taken.
    residual : float
        Final relative change in the phasor between successive optical periods.
        The convergence criterion, reported so a caller can tell a converged
        run from one that hit the cap.
    converged : bool
        Whether *residual* fell below the requested tolerance.
    """

    ex: NDArray[np.complex128]
    ey: NDArray[np.complex128]
    ez: NDArray[np.complex128]
    steps: int
    residual: float
    converged: bool


# ---------------------------------------------------------------------------
# CPML
# ---------------------------------------------------------------------------


def _pml_profile(
    n: int, n_pml: int, dz: float, dt: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Convolutional-PML coefficients along one axis.

    Returns ``(b, a, kappa)`` sampled at *n* points, with absorbing regions of
    *n_pml* cells at each end and a transparent middle.

    The stretched coordinate is ``1 + sigma/(kappa(alpha - i omega eps0))``; in
    the time domain that becomes a first-order recursion per cell, which is the
    ``b`` and ``a`` returned here.  ``alpha`` tapering to zero at the outer wall
    is what makes it absorb evanescent as well as propagating fields — the
    reason a plain PML leaks near a patterned surface and this one does not.
    """
    kappa = np.ones(n)
    sigma = np.zeros(n)
    alpha = np.zeros(n)

    if n_pml > 0:
        eps0 = 1.0 / (C0 * C0 * 4e-7 * np.pi)
        length = n_pml * dz
        sigma_max = -(_PML_M + 1.0) * np.log(_PML_R0) / (2.0 * ETA0 * length)
        for j in range(n_pml):
            # Depth into the layer, 0 at the interface and 1 at the outer wall.
            frac = (n_pml - j) / n_pml
            g = frac ** _PML_M
            for idx in (j, n - 1 - j):
                sigma[idx] = sigma_max * g
                kappa[idx] = 1.0 + (_PML_KAPPA_MAX - 1.0) * g
                alpha[idx] = _PML_ALPHA_MAX * (1.0 - frac)

        b = np.exp(-((sigma / kappa) + alpha) * dt / eps0)
        denom = kappa * (sigma + kappa * alpha)
        a = np.where(np.abs(denom) > 0.0, sigma * (b - 1.0) / np.where(denom == 0, 1, denom), 0.0)
    else:
        b = np.ones(n)
        a = np.zeros(n)

    return b, a, kappa


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


def solve_near_field(
    index: NDArray[np.complex128],
    dx: float,
    dz: float,
    wavelength: float,
    kx: float = 0.0,
    ky: float = 0.0,
    *,
    polarisation: tuple[complex, complex, complex] = (0.0, 1.0, 0.0),
    source_row: int | None = None,
    monitor_row: int | None = None,
    n_pml: int = 16,
    ramp_periods: float = 8.0,
    steady_tol: float = 1e-3,
    max_periods: int = 400,
    dtype: type = np.complex64,
) -> FDTDResult:
    """March Maxwell's equations to steady state and read off the near field.

    Parameters
    ----------
    index : NDArray
        Complex refractive index ``n + ik``, shape ``(nz, nx)``, in the library
        storage convention — positive ``k`` absorbs, no conjugation needed.
        ``iz = 0`` is the **top** of the domain, where the light comes in.
    dx, dz : float
        Cell size [m].
    wavelength : float
        Vacuum wavelength [m].
    kx, ky : float
        Transverse wavenumbers of the incident wave [m⁻¹]. ``kx`` sets the
        Bloch phase across the periodic cell; ``ky`` drives the conical terms.
    polarisation : tuple of complex
        Incident ``(Ex, Ey, Ez)`` amplitudes. The default is ``Ey``, i.e. the
        field along the grating lines.
    source_row, monitor_row : int, optional
        Rows of the injection plane and the plane the near field is read on.
        Default to just inside the top PML, and just inside the bottom one.
    n_pml : int
        CPML thickness in cells, each end of ``z``.
    ramp_periods : float
        Optical periods over which the source turns on. A hard turn-on is
        broadband and takes far longer to settle than the ramp costs.
    steady_tol : float
        Relative phasor change between successive periods that counts as steady.
    max_periods : int
        Cap. Exceeding it **warns and returns** rather than raising, matching
        :func:`litho_sim.bake.reaction.bake_reaction_diffusion` — a
        not-quite-converged field is still informative, and the residual is
        reported so the caller can judge.
    dtype : type
        ``np.complex64`` is the default and is ~2x faster than ``complex128``
        with no accuracy consequence at the tolerances here.

    Returns
    -------
    FDTDResult
    """
    nz, nx = index.shape
    if nz < 4 * n_pml or nx < 4:
        raise ValueError(
            f"grid {nz}x{nx} is too small for {n_pml} PML cells a side; "
            f"needs at least {4 * n_pml} rows"
        )

    omega = 2.0 * np.pi * C0 / wavelength
    period = wavelength / C0

    # Courant limit, scaled by the fastest phase velocity in the grid.
    #
    # The usual `dt <= 1/(c*sqrt(...))` assumes nothing travels faster than c,
    # which is false here: at EUV *every* material has n slightly below 1, so
    # eps' ~ 0.9 and the phase velocity is ~1.05c. A vacuum-derived step is
    # marginally unstable across the whole domain, not just in one layer.
    eps_min = float(np.min(np.real(np.asarray(index, dtype=np.complex128) ** 2)))
    speed = 1.0 / np.sqrt(min(max(eps_min, _EPS_FLOOR), 1.0))
    dt = _SAFETY / (
        speed * C0 * np.sqrt(1.0 / dx**2 + 1.0 / dz**2 + (ky / 2.0) ** 2)
    )
    steps_per_period = int(np.ceil(period / dt))
    dt = period / steps_per_period  # an integer number of steps per period

    mu0 = 4e-7 * np.pi
    eps0 = 1.0 / (C0 * C0 * mu0)

    # Split the complex index into the pair a time-domain march can actually
    # use. See the module docstring: a complex permittivity in the update makes
    # the scheme diverge.
    eps_c = np.asarray(index, dtype=np.complex128) ** 2
    eps_r = np.real(eps_c).astype(np.float64)
    eps_i = np.imag(eps_c).astype(np.float64)
    if np.any(eps_i < 0.0):
        raise ValueError(
            "index has a negative imaginary part somewhere, which is gain. "
            "Materials are stored as n + ik with positive k absorbing."
        )

    # Metals need a pole, not a permittivity. Chrome at 193 nm has
    # eps' = -2.02: negative permittivity is a frequency-domain statement and
    # has no time-domain meaning, so those cells get a Drude term instead.
    #
    # The threshold is a small positive number rather than zero because of
    # subpixel-averaged edge cells, which interpolate between vacuum and metal
    # and therefore pass *through* eps' = 0. A cell left on the dielectric path
    # with eps' = 0.001 has a phase velocity of 30c and blows the march up
    # within a few hundred steps.
    metal = eps_r < _EPS_FLOOR
    eps_inf = np.where(metal, 1.0, eps_r)
    sigma = np.where(metal, 0.0, omega * eps0 * eps_i)

    # Drude parameters that reproduce the tabulated index at this one
    # frequency. Matching the imaginary part fixes the damping; the plasma
    # frequency then follows. Both are only claimed to hold at omega.
    gap = np.where(metal, 1.0 - eps_r, 1.0)
    gamma = np.where(metal, eps_i * omega / gap, 0.0)
    wp2 = np.where(metal, omega * omega * (gap + eps_i * eps_i / gap), 0.0)

    # Every update coefficient below is *real*: the loss went into sigma, not
    # into a complex permittivity. Storing them at the real precision matching
    # ``dtype`` rather than as complex halves the multiply work and — the part
    # that actually bites — stops numpy promoting the whole update to
    # complex128. A float64 coefficient against a complex64 field silently
    # yields complex128, which costs about 1.8x and quietly defeats the
    # complex64 default this solver documents.
    real_dtype = np.finfo(dtype).dtype
    loss = (sigma * dt / (2.0 * eps0 * eps_inf)).astype(np.float64)
    ca = ((1.0 - loss) / (1.0 + loss)).astype(real_dtype)
    cb = ((dt / (eps0 * eps_inf)) / (1.0 + loss)).astype(real_dtype)
    gd = gamma * dt / 2.0
    cj1 = ((1.0 - gd) / (1.0 + gd)).astype(real_dtype)
    cj2 = ((eps0 * wp2 * dt) / (1.0 + gd)).astype(real_dtype)
    has_metal = bool(metal.any())

    # Field arrays, all (nz, nx).
    z = lambda: np.zeros((nz, nx), dtype=dtype)  # noqa: E731
    ex, ey, ez = z(), z(), z()
    hx, hy, hz = z(), z(), z()
    jx, jy, jz = z(), z(), z()  # Drude currents; unused unless metal is present

    # CPML along z only; x is periodic and needs no absorber.
    b_z, a_z, kap_z = _pml_profile(nz, n_pml, dz, dt)
    b_z = b_z.astype(real_dtype)[:, None]
    a_z = a_z.astype(real_dtype)[:, None]
    inv_kap = (1.0 / kap_z).astype(real_dtype)[:, None]
    psi_hx, psi_hy, psi_ex, psi_ey = z(), z(), z(), z()

    bloch = np.exp(1j * kx * nx * dx).astype(dtype)

    def dx_fwd(f):
        """d/dx at +1/2, Bloch-periodic: f[i+1] - f[i]."""
        out = np.empty_like(f)
        out[:, :-1] = f[:, 1:] - f[:, :-1]
        out[:, -1] = f[:, 0] * bloch - f[:, -1]
        return out

    def dx_bwd(f):
        """d/dx at -1/2, Bloch-periodic: f[i] - f[i-1]."""
        out = np.empty_like(f)
        out[:, 1:] = f[:, 1:] - f[:, :-1]
        out[:, 0] = f[:, 0] - f[:, -1] / bloch
        return out

    def dz_fwd(f):
        # empty + one zeroed row, not zeros_like: every other row is about to be
        # overwritten anyway, and at these grid sizes the redundant fill is a
        # measurable fraction of the update.
        out = np.empty_like(f)
        out[:-1, :] = f[1:, :] - f[:-1, :]
        out[-1, :] = 0
        return out

    def dz_bwd(f):
        out = np.empty_like(f)
        out[1:, :] = f[1:, :] - f[:-1, :]
        out[0, :] = 0
        return out

    src = int(source_row if source_row is not None else n_pml + 4)
    mon = int(monitor_row if monitor_row is not None else nz - n_pml - 4)
    if not 0 < src < nz or not 0 <= mon < nz:
        raise ValueError(
            f"source_row and monitor_row must lie inside the {nz}-row grid, "
            f"got source_row={src} and monitor_row={mon}"
        )
    # A monitor *above* the source is not a mistake — it is how a reflective
    # mask is read. Below the injection plane the stored field is total; above
    # it, purely scattered, which for an EUV mask is the reflected near field
    # already separated from the illumination.

    # Incident plane wave, evaluated with the *numerical* z-wavenumber so the
    # injected field is a solution of the discrete equations rather than the
    # continuous ones. Getting this wrong leaks a few percent of the incident
    # amplitude into the scattered-field region and contaminates the near field.
    kz = _numerical_kz(omega, kx, ky, dx, dz, dt)
    x = np.arange(nx) * dx
    phase_x = np.exp(1j * kx * x).astype(dtype)
    # Only the transverse components are injected. A plane wave's Ez is fixed
    # by transversality (k.E = 0), so specifying it independently would
    # over-determine the source; it appears in the solution on its own.
    e0x, e0y = complex(polarisation[0]), complex(polarisation[1])

    # The magnetic update needs no lossy counterpart: masks are non-magnetic,
    # so mu is real everywhere and one constant does.
    ch = dt / mu0
    # Reciprocals hoisted: the loop divides six full arrays by a cell size every
    # step, and division is several times the cost of the multiply it becomes.
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz

    # One transit of the domain, in optical periods, using the slowest medium
    # present — a high-index film both slows the wave and lengthens the settle.
    n_max = float(np.max(np.abs(np.asarray(index))))
    transit = (nz * dz * max(n_max, 1.0)) / (C0 * period)
    min_periods = ramp_periods + 2.0 * transit

    prev = None
    acc_ex = np.zeros(nx, dtype=np.complex128)
    acc_ey = np.zeros(nx, dtype=np.complex128)
    acc_ez = np.zeros(nx, dtype=np.complex128)
    residual = np.inf
    converged = False
    step = 0
    period_idx = 0

    for period_idx in range(max_periods):
        acc_ex[:] = 0.0
        acc_ey[:] = 0.0
        acc_ez[:] = 0.0

        for _ in range(steps_per_period):
            t = step * dt
            envelope = _envelope(step, steps_per_period, ramp_periods, omega, dt)

            # --- H update -------------------------------------------------
            d_ey_dz = dz_fwd(ey) * inv_dz
            psi_hx[:] = b_z * psi_hx + a_z * d_ey_dz
            curl_x = 1j * ky * ez - (inv_kap * d_ey_dz + psi_hx)

            d_ex_dz = dz_fwd(ex) * inv_dz
            psi_hy[:] = b_z * psi_hy + a_z * d_ex_dz
            curl_y = (inv_kap * d_ex_dz + psi_hy) - dx_fwd(ez) * inv_dx

            curl_z = dx_fwd(ey) * inv_dx - 1j * ky * ex

            hx -= ch * curl_x
            hy -= ch * curl_y
            hz -= ch * curl_z

            # Total-field / scattered-field. Below the source plane the stored
            # field is *total*; above it, only the scattered part — which for a
            # reflective mask is exactly the answer wanted, already separated
            # from the illumination.
            #
            # The correction is one term per update that straddles the plane.
            # Here the H at src-1/2 is scattered but its stencil reaches E at
            # row src, which is stored as total, so the incident part of that E
            # has to come back out.
            e_inc_src = (envelope * phase_x).astype(dtype)
            hx[src - 1, :] -= ch * (e0y * e_inc_src) / dz
            hy[src - 1, :] += ch * (e0x * e_inc_src) / dz

            # --- E update -------------------------------------------------
            d_hy_dz = dz_bwd(hy) * inv_dz
            psi_ex[:] = b_z * psi_ex + a_z * d_hy_dz
            curl_hx = 1j * ky * hz - (inv_kap * d_hy_dz + psi_ex)

            d_hx_dz = dz_bwd(hx) * inv_dz
            psi_ey[:] = b_z * psi_ey + a_z * d_hx_dz
            curl_hy = (inv_kap * d_hx_dz + psi_ey) - dx_bwd(hz) * inv_dx

            curl_hz = dx_bwd(hy) * inv_dx - 1j * ky * hx

            if has_metal:
                # Drude polarisation current, advanced on E^n and then
                # subtracted from the curl — the standard auxiliary-differential
                # -equation form. Zero coefficients everywhere non-metallic, so
                # dielectric cells are untouched.
                jx[:] = cj1 * jx + cj2 * ex
                jy[:] = cj1 * jy + cj2 * ey
                jz[:] = cj1 * jz + cj2 * ez
                curl_hx = curl_hx - jx
                curl_hy = curl_hy - jy
                curl_hz = curl_hz - jz

            ex[:] = ca * ex + cb * curl_hx
            ey[:] = ca * ey + cb * curl_hy
            ez[:] = ca * ez + cb * curl_hz

            # The matching half of the TF/SF pair: E at row src is total, but
            # its stencil reaches H at src-1/2, which is stored as scattered.
            # The incident H there is evaluated half a step later in time and
            # half a cell higher in z, which is where the leapfrog and the Yee
            # offsets put it — dropping either offset leaks a few percent of
            # the illumination into the scattered region.
            amp_half = _envelope(step + 0.5, steps_per_period, ramp_periods, omega, dt)
            h_inc = (amp_half * phase_x * np.exp(-1j * kz * dz / 2.0)).astype(dtype)
            hx_inc = -kz * e0y / (omega * mu0) * h_inc
            hy_inc = kz * e0x / (omega * mu0) * h_inc
            ey[src, :] -= cb[src, :] * hx_inc / dz
            ex[src, :] += cb[src, :] * hy_inc / dz

            # --- running DFT on the monitor plane -------------------------
            osc = np.exp(1j * omega * t)
            acc_ex += ex[mon, :] * osc
            acc_ey += ey[mon, :] * osc
            acc_ez += ez[mon, :] * osc
            step += 1

        cur = np.stack(
            (acc_ex / steps_per_period, acc_ey / steps_per_period,
             acc_ez / steps_per_period)
        )
        # Do not test for steadiness before the wave can physically have
        # reached the monitor and back. Without this the residual is computed
        # on numerical noise, which is already "steady", and the solver
        # declares victory holding a field of 1e-18. Two transits covers the
        # arrival plus the first reflection off whatever it hits.
        if prev is not None:
            scale = np.abs(cur).max()
            if scale > 0:
                residual = float(np.abs(cur - prev).max() / scale)
                if residual < steady_tol and period_idx > min_periods:
                    converged = True
                    prev = cur
                    break
        prev = cur

    if not converged:
        logger.warning(
            "FDTD did not reach steady state: residual %.2e after %d periods "
            "(%d steps). Returning the field anyway — check for a trapped mode "
            "or an order grazing the horizon.",
            residual, period_idx + 1, step,
        )

    assert prev is not None
    return FDTDResult(
        ex=prev[0].astype(np.complex128),
        ey=prev[1].astype(np.complex128),
        ez=prev[2].astype(np.complex128),
        steps=step,
        residual=residual,
        converged=converged,
    )


def _numerical_kz(
    omega: float, kx: float, ky: float, dx: float, dz: float, dt: float
) -> complex:
    """Longitudinal wavenumber that satisfies the *discrete* dispersion relation.

    The Yee scheme's plane waves do not travel at exactly ``c``; solving for the
    wavenumber the grid actually supports, rather than the continuous ``k0``,
    is what keeps total-field/scattered-field injection clean to round-off
    instead of leaking a percent or so of the incident wave.
    """
    lhs = (np.sin(omega * dt / 2.0) / (C0 * dt)) ** 2
    lhs -= (np.sin(kx * dx / 2.0) / dx) ** 2
    lhs -= (ky / 2.0) ** 2
    s = lhs * dz * dz
    s = np.clip(s, -1.0, 1.0)
    return (2.0 / dz) * np.arcsin(np.sqrt(s.astype(complex)))


def _envelope(
    step: float, steps_per_period: int, ramp_periods: float, omega: float, dt: float
) -> complex:
    """Incident amplitude at a (possibly half-integer) time step.

    A continuous-wave source switched on abruptly is broadband, and the
    transient it launches takes far longer to leave the domain than a smooth
    turn-on costs. The smoothstep ramp has zero first derivative at both ends,
    so the spectrum it injects stays tight around the operating frequency.
    """
    frac = (step / steps_per_period) / ramp_periods
    r = min(1.0, max(0.0, frac))
    ramp = r * r * (3.0 - 2.0 * r)
    return ramp * np.exp(-1j * omega * step * dt)
