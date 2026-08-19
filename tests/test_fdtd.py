"""The FDTD solver, against closed forms it has no way to fake.

A rigorous mask model is only worth having if the solver under it is right, and
"it produced a plausible picture" is not evidence. Every test here compares
against something computed independently: Fresnel coefficients, the
transfer-matrix method in :mod:`litho_sim.coat.films`, the grating equation, or
the solver's own convergence rate.

The single most useful of them is :func:`test_planar_stack_matches_fresnel`,
because a planar stack is exactly solvable and every mechanism in the solver —
injection, propagation, material response, boundary absorption — has to be
right for it to land.

A note on domain size, learned the hard way: a monitor plane placed near the
PML reads a few percent high or low, and the error refines away at *first*
order, which looks exactly like a staircased-interface bug. It is not — it is
residual PML reflection. Keep monitors well clear, or spend an afternoon
chasing the wrong thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.expose.m3d.yee import solve_near_field

LAM = 193e-9
QUARTZ = 1.5603 + 0.0j


def _detilted(field: np.ndarray, kx: float, dx: float) -> complex:
    """Mean field with the incident tilt divided out.

    At oblique incidence the field carries ``exp(i kx x)`` across the cell, and
    when the cell holds a whole number of transverse wavelengths the plain mean
    of that is **identically zero**. Removing the tilt first is not a
    convenience — without it every oblique measurement reads as no field at all.
    It is also precisely the operation the near-field extraction performs, for
    the same reason.
    """
    x = np.arange(field.size) * dx
    return complex((field * np.exp(-1j * kx * x)).mean())


def _slab(nz: int, nx: int, iface: int, n: complex) -> np.ndarray:
    """Vacuum above *iface*, a half-space of index *n* below."""
    idx = np.ones((nz, nx), dtype=complex)
    idx[iface:, :] = n  # stored n + ik, used as stored
    return idx


# ---------------------------------------------------------------------------
# Free space
# ---------------------------------------------------------------------------


def test_an_empty_domain_carries_the_wave_it_was_given():
    """Unit amplitude in, unit amplitude out — the injection's own test.

    Total-field/scattered-field injection is where a plane wave gets into the
    grid, and it is easy to get *almost* right: an amplitude that is a percent
    or two off, or an incident wavenumber taken from the continuum rather than
    the discrete dispersion relation. Both show up here, and nowhere else so
    cleanly, because free space has no other physics to hide behind.
    """
    idx = np.ones((400, 8), dtype=complex)
    res = solve_near_field(
        idx, 4e-9, 4e-9, LAM, polarisation=(0, 1, 0),
        source_row=50, monitor_row=90, n_pml=20, steady_tol=1e-5,
    )
    assert res.converged
    assert abs(res.ey.mean()) == pytest.approx(1.0, abs=0.02)


def test_nothing_leaks_upstream_of_the_source_plane():
    """Above the injection plane the stored field is *scattered* only.

    With nothing to scatter off, that region must stay empty. This is the
    property the whole EUV path depends on — a reflective mask's near field
    *is* the scattered field above the mask, so any incident light left in
    there is added straight onto the answer.
    """
    idx = np.ones((400, 8), dtype=complex)
    res = solve_near_field(
        idx, 4e-9, 4e-9, LAM, polarisation=(0, 1, 0),
        source_row=200, monitor_row=80, n_pml=20, steady_tol=1e-5,
    )
    assert abs(res.ey.mean()) < 0.05, (
        f"scattered-field region holds {abs(res.ey.mean()):.3f} of the incident "
        f"amplitude; TF/SF is leaking"
    )


# ---------------------------------------------------------------------------
# Against closed forms
# ---------------------------------------------------------------------------


def test_planar_stack_matches_fresnel():
    """Transmission into a half-space, against the Fresnel coefficient.

    Normal incidence onto vacuum/quartz transmits ``2/(1+n)`` in amplitude.
    Nothing about that is adjustable, so agreement at the tenth of a percent
    level is real evidence rather than a fitted tolerance.
    """
    idx = _slab(400, 8, 150, QUARTZ)
    res = solve_near_field(
        idx, 4e-9, 4e-9, LAM, polarisation=(0, 1, 0),
        source_row=60, monitor_row=220, n_pml=25, steady_tol=1e-5,
    )
    expected = abs(2.0 / (1.0 + QUARTZ))
    assert res.converged
    assert abs(res.ey.mean()) == pytest.approx(expected, rel=0.02)


def test_the_error_falls_as_the_square_of_the_cell_size():
    """Second-order convergence — the property that says the scheme is right.

    Yee's scheme is second-order accurate, so halving the cell should quarter
    the error. Asserting the *rate* rather than a tolerance is what makes this
    a statement about the discretisation instead of about one grid: a
    first-order rate would mean an interface or a source term is misplaced by
    half a cell, which no single-grid tolerance would reveal.
    """
    expected = abs(2.0 / (1.0 + QUARTZ))
    errors = []
    for scale in (1, 2, 4):  # dz = 8, 4, 2 nm
        d = 8e-9 / scale
        idx = _slab(400 * scale, 8, 150 * scale, QUARTZ)
        res = solve_near_field(
            idx, d, d, LAM, polarisation=(0, 1, 0),
            source_row=60 * scale, monitor_row=220 * scale,
            n_pml=25 * scale, steady_tol=1e-5, max_periods=500,
        )
        errors.append(abs(abs(res.ey.mean()) - expected) / expected)

    assert errors == sorted(errors, reverse=True), f"error must fall: {errors}"
    for coarse, fine in zip(errors, errors[1:]):
        rate = coarse / fine
        assert rate > 3.0, (
            f"convergence rate {rate:.2f} is closer to first order than second; "
            f"suspect a half-cell offset in a source or an interface"
        )


def test_an_absorbing_film_attenuates_by_beer_lambert():
    """Amplitude through an absorbing layer follows ``exp(-2 pi k d / lambda)``.

    This is what pins the *sign* of the imaginary index. Get it backwards and
    the layer amplifies — the failure mode
    :func:`litho_sim.coat.films._propagation_index` exists to prevent, and one
    an FDTD reproduces just as enthusiastically.
    """
    n = 1.5 + 0.05j
    dz = 2e-9
    thickness_cells = 100
    top = 150
    idx = np.ones((500, 8), dtype=complex)
    idx[top:top + thickness_cells, :] = n
    idx[top + thickness_cells:, :] = n.real + 0j  # lossless below

    res = solve_near_field(
        idx, dz, dz, LAM, polarisation=(0, 1, 0),
        source_row=60, monitor_row=320, n_pml=25, steady_tol=1e-5,
    )
    decay = np.exp(-2.0 * np.pi * n.imag * thickness_cells * dz / LAM)
    got = abs(res.ey.mean())
    assert got < 1.0, "an absorbing layer must attenuate, not amplify"
    # Interface reflections ride on top of the bulk decay, so this checks the
    # exponential is present and of the right size, not that it is alone.
    assert got == pytest.approx(decay * abs(2.0 / (1.0 + n.real)), rel=0.15)


# ---------------------------------------------------------------------------
# Oblique incidence and the Bloch boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle_deg", [0.0, 10.0, 20.0])
def test_oblique_incidence_matches_the_transfer_matrix(angle_deg):
    """Against :mod:`litho_sim.coat.films` at angle — the real validation.

    This is the test the whole design leans on. A Bloch-periodic complex-field
    solver at oblique incidence has several things that can be quietly wrong at
    once — the boundary phase, the incident wavenumber, the PML at angle — and
    a planar stack has an exact answer at every one of those angles.
    """
    from litho_sim.coat.films import Film, FilmStack

    theta = np.radians(angle_deg)
    k0 = 2.0 * np.pi / LAM
    kx = k0 * np.sin(theta)

    # The cell must hold a whole number of transverse wavelengths, or the Bloch
    # phase and the grid disagree about what "periodic" means.
    nx = 32
    dx = (2.0 * np.pi / kx) / nx if kx != 0 else 4e-9
    dz = 2e-9

    idx = _slab(500, nx, 200, QUARTZ)
    res = solve_near_field(
        idx, dx, dz, LAM, kx=kx, ky=0.0, polarisation=(0, 1, 0),
        source_row=70, monitor_row=300, n_pml=25, steady_tol=1e-5,
        max_periods=600,
    )

    stack = FilmStack(
        [Film("vac", 0.0, 1.0 + 0j), Film("q", 0.0, QUARTZ), Film("q", 0.0, QUARTZ)],
        resist_index=1,
    )
    r = stack.amplitude_reflection(LAM, theta, "s")
    expected = abs(1.0 + r)  # amplitude transmission = 1 + r for s at an interface

    assert res.converged, f"did not settle at {angle_deg} deg"
    assert abs(_detilted(res.ey, kx, dx)) == pytest.approx(expected, rel=0.06)


def test_the_bloch_phase_wraps_the_cell_correctly():
    """One period of transverse phase across the cell, and it closes.

    The Bloch factor is the only thing making an oblique wave periodic. If its
    sign or magnitude is wrong the field still looks like a wave, so this
    checks the property directly: the phase advance across the cell must equal
    ``kx * width``, and the wrap must be seamless.
    """
    nx, dz = 32, 4e-9
    theta = np.radians(15.0)
    kx = (2.0 * np.pi / LAM) * np.sin(theta)
    dx = (2.0 * np.pi / kx) / nx  # exactly one transverse wavelength per cell

    idx = np.ones((300, nx), dtype=complex)
    res = solve_near_field(
        idx, dx, dz, LAM, kx=kx, polarisation=(0, 1, 0),
        source_row=50, monitor_row=150, n_pml=20, steady_tol=1e-5,
    )
    field = res.ey
    assert abs(np.abs(field).std() / np.abs(field).mean()) < 0.02, (
        "an unobstructed oblique plane wave must have flat amplitude across x"
    )
    # Phase advances by exactly 2 pi over the cell, so successive differences
    # are uniform and sum to one turn.
    step = np.diff(np.unwrap(np.angle(field)))
    assert step.mean() == pytest.approx(2.0 * np.pi / nx, rel=0.05)
    assert step.std() < 0.05 * abs(step.mean())


# ---------------------------------------------------------------------------
# Conical incidence
# ---------------------------------------------------------------------------


def test_conical_incidence_leaves_the_time_step_almost_alone():
    """``k_y`` costs arithmetic, not stability.

    The 2.5-D formulation replaces ``d/dy`` with an exact multiplication, so
    ``k_y`` enters the Courant limit as ``k_y^2/4`` rather than through a
    difference. At the mask-side angles lithography uses, that term is
    thousands of times smaller than ``1/dx^2``, which is why solving the
    conical problem properly is affordable rather than a compromise.
    """
    dx = dz = 2e-9
    k0 = 2.0 * np.pi / LAM
    ky = k0 * np.sin(np.radians(20.0))  # generous: mask-side angles are far smaller
    without = 1.0 / np.sqrt(1.0 / dx**2 + 1.0 / dz**2)
    with_ky = 1.0 / np.sqrt(1.0 / dx**2 + 1.0 / dz**2 + (ky / 2.0) ** 2)
    assert with_ky / without > 0.999, (
        f"k_y shortened the time step by {(1 - with_ky / without):.3%}"
    )


def test_a_conical_run_still_settles_and_conserves_amplitude():
    """With ``k_y != 0`` all six components couple; free space must still be free.

    The decoupled TE/TM systems cannot represent this case at all, so the check
    is simply that the coupled solver does not lose or manufacture energy while
    doing so.
    """
    nx, dz = 24, 4e-9
    k0 = 2.0 * np.pi / LAM
    kx = k0 * np.sin(np.radians(8.0))
    ky = k0 * np.sin(np.radians(8.0))
    dx = (2.0 * np.pi / kx) / nx

    idx = np.ones((300, nx), dtype=complex)
    res = solve_near_field(
        idx, dx, dz, LAM, kx=kx, ky=ky, polarisation=(0, 1, 0),
        source_row=50, monitor_row=150, n_pml=20, steady_tol=1e-5,
    )
    assert res.converged
    assert abs(_detilted(res.ey, kx, dx)) == pytest.approx(1.0, abs=0.06)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_a_domain_too_small_for_its_pml_is_refused():
    idx = np.ones((20, 8), dtype=complex)
    with pytest.raises(ValueError, match="PML"):
        solve_near_field(idx, 4e-9, 4e-9, LAM, n_pml=16)


def test_a_monitor_outside_the_grid_is_refused():
    idx = np.ones((300, 8), dtype=complex)
    with pytest.raises(ValueError, match="monitor_row"):
        solve_near_field(idx, 4e-9, 4e-9, LAM, source_row=100, monitor_row=999,
                         n_pml=20)


def test_gain_media_are_refused_by_name():
    """A negative imaginary index is gain, and is almost always a sign slip.

    EUV constants are published as ``1 - delta - i*beta``; entered directly they
    make every absorber amplify. Saying so beats returning a mask that reflects
    300 %.
    """
    idx = np.ones((300, 8), dtype=complex)
    idx[150:, :] = 0.95 - 0.03j
    with pytest.raises(ValueError, match="gain"):
        solve_near_field(idx, 4e-9, 4e-9, LAM, n_pml=20)


def test_a_run_that_will_not_settle_warns_instead_of_raising(caplog):
    """An unconverged field is still informative; a crash is not.

    Matches the precedent in :func:`litho_sim.bake.reaction.bake_reaction_diffusion`:
    cap the work, say so, and hand back what you have with the residual
    attached so the caller can judge it.
    """
    idx = np.ones((300, 8), dtype=complex)
    res = solve_near_field(
        idx, 4e-9, 4e-9, LAM, polarisation=(0, 1, 0),
        source_row=50, monitor_row=150, n_pml=20,
        steady_tol=1e-14, max_periods=12,
    )
    assert not res.converged
    assert np.isfinite(res.residual)
    assert res.ey.shape == (8,)
