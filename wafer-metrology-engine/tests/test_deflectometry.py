"""
Unit tests for phase-measuring deflectometry.

Pins the plan's Tier-0 validation gate (<1 % shape recovery on synthetics),
the slope-integration accuracy, the temporal unwrap, the gamma calibration,
and -- most importantly -- the *parity* behaviour of reversal calibration:
what a 180-degree rotation genuinely separates, what it cannot, and how the
reference-flat subtraction closes the gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafer_metrology.deflectometry import (
    DeflectometrySetup,
    absolute_calibrate,
    fit_gamma,
    integrate_slopes,
    measure_deflectometry,
    plane_aligned_difference,
    reconstruct_slopes,
    reversal_calibrate,
    rotate_surface_180,
    simulate_deflectograms,
    simulate_gamma_sweep,
    surface_slopes,
    unwrap_temporal,
)
from wafer_metrology.synthesize import (
    apply_mask,
    make_flat_surface,
    make_wafer_grid,
    synthesize_wafer,
)
from wafer_metrology.zernike import reconstruct

WAFER_150 = 150e-3
RADIUS_150 = 75e-3


def rms(values: np.ndarray) -> float:
    """RMS over the finite entries of *values*."""
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2)))


@pytest.fixture(scope="module")
def wafer150():
    """A 150 mm synthetic wafer matching the hardware build."""
    return synthesize_wafer(n_pixels=160, diameter=WAFER_150, seed=0)


# ---------------------------------------------------------------------------
# Slope integration
# ---------------------------------------------------------------------------


def test_integration_recovers_an_analytic_surface():
    """Analytic slopes integrate back to the surface to well under 1 %."""
    x, y, r, theta, mask = make_wafer_grid(160, WAFER_150)
    k = 2.0 * np.pi / 0.05
    z = 1e-6 * np.sin(k * x) * np.cos(0.7 * k * y) + 20e-6 * (r / RADIUS_150) ** 2
    sx = 1e-6 * k * np.cos(k * x) * np.cos(0.7 * k * y) + 40e-6 * x / RADIUS_150**2
    sy = -1e-6 * 0.7 * k * np.sin(k * x) * np.sin(0.7 * k * y) + 40e-6 * y / RADIUS_150**2

    integrated = integrate_slopes(
        apply_mask(sx, mask), apply_mask(sy, mask), mask, WAFER_150 / 160
    )

    truth = apply_mask(z - np.nanmean(z[mask]), mask)
    error = integrated - truth
    assert rms(error) < 0.005 * rms(truth)


def test_integration_is_zero_mean_and_masked():
    """The gauge convention holds and NaN stays outside the aperture."""
    x, y, r, theta, mask = make_wafer_grid(96, WAFER_150)
    sx = apply_mask(np.full_like(x, 1e-4), mask)
    sy = apply_mask(np.zeros_like(x), mask)

    z = integrate_slopes(sx, sy, mask, WAFER_150 / 96)

    assert abs(np.nanmean(z)) < 1e-15
    assert np.isnan(z[~mask]).all()
    # A constant slope integrates to a plane: x-gradient recovered, y flat.
    gy, gx = np.gradient(np.where(mask, z, np.nan), WAFER_150 / 96)
    assert np.nanmedian(gx) == pytest.approx(1e-4, rel=1e-6)
    assert abs(np.nanmedian(gy)) < 1e-12


def test_integration_needs_pixels():
    """An empty mask is refused."""
    empty = np.zeros((8, 8), dtype=bool)
    with pytest.raises(ValueError):
        integrate_slopes(np.zeros((8, 8)), np.zeros((8, 8)), empty, 1e-3)


# ---------------------------------------------------------------------------
# Temporal unwrap
# ---------------------------------------------------------------------------


def test_temporal_unwrap_recovers_a_multi_period_coordinate():
    """A coordinate spanning many fine periods unwraps exactly."""
    coordinate = np.linspace(-0.08, 0.08, 4001)
    periods = (0.35, 0.08, 0.015)
    phases = [
        np.angle(np.exp(1j * 2.0 * np.pi * coordinate / p)) for p in periods
    ]

    recovered = unwrap_temporal(phases, periods)

    assert np.abs(recovered - coordinate).max() < 1e-12


def test_temporal_unwrap_survives_phase_noise():
    """Fringe-order decisions stay correct with realistic phase noise."""
    rng = np.random.default_rng(3)
    coordinate = np.linspace(-0.07, 0.07, 2001)
    periods = (0.35, 0.08, 0.015)
    phases = [
        np.angle(np.exp(1j * (2.0 * np.pi * coordinate / p + rng.normal(0, 0.01, coordinate.size))))
        for p in periods
    ]

    recovered = unwrap_temporal(phases, periods)

    error = recovered - coordinate
    assert np.abs(error).max() < periods[-1] / 4          # no order slips
    assert rms(error) < periods[-1] * 0.01 / (2 * np.pi) * 3


def test_temporal_unwrap_validates_inputs():
    """Mismatched lengths and non-decreasing periods are refused."""
    phase = np.zeros(4)
    with pytest.raises(ValueError):
        unwrap_temporal([phase], (0.1, 0.05))
    with pytest.raises(ValueError):
        unwrap_temporal([phase, phase], (0.05, 0.1))


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------


def test_gamma_fit_recovers_the_display_gamma():
    """The weighted log-log fit lands on the true gamma despite dark noise."""
    for true_gamma in (1.8, 2.2, 2.6):
        commanded, captured = simulate_gamma_sweep(true_gamma, noise=0.002, seed=1)
        assert fit_gamma(commanded, captured) == pytest.approx(true_gamma, abs=0.05)


def test_eight_step_shifting_suppresses_gamma_harmonics():
    """8-step phase extraction is robust to an uncorrected gamma; 4-step is not.

    Uniform N-step shifting aliases only harmonics m = kN +/- 1, so at N = 8
    the first contaminating harmonic of the gamma-distorted fringe is the 7th
    (~100x below the 3rd that N = 4 admits).  Tested directly on the recovered
    phase of a synthetic ramp -- in the full pipeline slope integration
    low-passes the ripple, which would mask exactly the effect under test.
    This is why the plan can run without a perfect LUT as long as it keeps 8
    steps, and why dropping to 4 steps for speed would need the LUT.
    """
    from wafer_metrology.interferometry import recover_phase, wrap

    phi = np.linspace(-np.pi, np.pi, 2001)

    def phase_error(n_steps: int, apply_lut: bool) -> float:
        deltas = 2.0 * np.pi * np.arange(n_steps) / n_steps
        frames = np.stack([
            (0.5 * (1.0 + np.cos(phi + d))) ** (1.0 if apply_lut else 2.2)
            for d in deltas
        ])
        recovered = recover_phase(frames, deltas)
        return float(np.abs(wrap(recovered - phi)).max())

    err_4 = phase_error(4, apply_lut=False)
    err_8 = phase_error(8, apply_lut=False)
    err_8_lut = phase_error(8, apply_lut=True)

    assert err_4 > 20.0 * err_8           # N=4 admits the 3rd harmonic
    assert err_8 < 1e-3                   # N=8 is already below a millirad
    assert err_8_lut < 1e-9               # a perfect LUT removes the rest


# ---------------------------------------------------------------------------
# Forward + inverse round trip (the Tier-0 validation gate)
# ---------------------------------------------------------------------------


def test_forward_phase_matches_the_sensitivity_relation(wafer150):
    """The forward model realises delta_phi = 4 pi d_s s / p exactly."""
    setup = DeflectometrySetup(noise=0.0, gamma=1.0, periods=(0.35, 0.015))
    deflectograms = simulate_deflectograms(wafer150, setup, seed=0)
    sx, sy = reconstruct_slopes(deflectograms)

    sx_true, sy_true, inner = surface_slopes(
        wafer150.z, wafer150.mask, wafer150.pixel_size
    )
    both = inner & np.isfinite(sx)
    assert rms((sx - sx_true)[both]) < 1e-9
    assert rms((sy - sy_true)[both]) < 1e-9


def test_validation_gate_under_one_percent(wafer150):
    """Full noisy round trip recovers the surface to <1 % relative RMS."""
    setup = DeflectometrySetup(noise=0.005, gamma=2.2, lut_gamma=2.2)
    measurement = measure_deflectometry(wafer150, setup, seed=7)

    truth = wafer150.z[measurement.mask]
    relative = measurement.rms_error / float(np.std(truth))
    assert relative < 0.01


def test_measurement_is_deterministic(wafer150):
    """Same seed, same reconstruction."""
    setup = DeflectometrySetup(noise=0.01)
    a = measure_deflectometry(wafer150, setup, seed=5)
    b = measure_deflectometry(wafer150, setup, seed=5)
    assert a.rms_error == pytest.approx(b.rms_error, rel=1e-15)


def test_undersized_coarse_period_warns(wafer150, caplog):
    """A coarse period smaller than the pattern span is flagged, not silent."""
    setup = DeflectometrySetup(periods=(0.05, 0.015), noise=0.0)
    with caplog.at_level("WARNING"):
        simulate_deflectograms(wafer150, setup, seed=0)
    assert "temporal unwrap" in caplog.text


# ---------------------------------------------------------------------------
# Reversal and absolute calibration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parity_setup(wafer150):
    """A rig with a known mixed-parity system error, plus its parts."""
    even = reconstruct(
        {(2, 0): 5e-6, (2, 2): 3e-6}, wafer150.x, wafer150.y, wafer150.radius,
        wafer150.mask,
    )
    odd = reconstruct(
        {(3, 1): 4e-6, (3, -3): 2e-6}, wafer150.x, wafer150.y, wafer150.radius,
        wafer150.mask,
    )
    system = apply_mask(np.nan_to_num(even) + np.nan_to_num(odd), wafer150.mask)
    setup = DeflectometrySetup(noise=0.002, system_zmap=system)
    return setup, even, odd


def test_reversal_separates_only_the_odd_parity_system(wafer150, parity_setup):
    """The documented parity limit, demonstrated end to end.

    The odd system part lands in the system estimate to nanometres; the even
    part (power + astigmatism -- a bowed screen) leaks into the wafer estimate
    essentially in full.  This is why the plan's claim that reversal separates
    "ALL fixed system error" is wrong, and why the flat matters.
    """
    setup, even, odd = parity_setup
    m0 = measure_deflectometry(wafer150, setup, seed=10)
    m180 = measure_deflectometry(rotate_surface_180(wafer150), setup, seed=11)

    wafer_est, system_est = reversal_calibrate(
        m0.z_measured, m180.z_measured, m0.mask
    )

    # System estimate == odd part, to measurement noise.
    odd_residual = plane_aligned_difference(
        system_est, odd, wafer150.x, wafer150.y, np.isfinite(system_est)
    )
    assert rms(odd_residual) < 0.02 * rms(odd[np.isfinite(system_est)])

    # Wafer estimate error == even part, to a few percent.
    wafer_error = plane_aligned_difference(
        wafer_est, wafer150.z, wafer150.x, wafer150.y, np.isfinite(wafer_est)
    )
    even_centred = plane_aligned_difference(
        even, np.zeros_like(even), wafer150.x, wafer150.y, np.isfinite(wafer_est)
    )
    assert rms(wafer_error) == pytest.approx(rms(even_centred), rel=0.05)


def test_flat_subtraction_removes_the_full_system(wafer150, parity_setup):
    """Absolute calibration against the reference flat beats parity limits."""
    setup, _, _ = parity_setup
    m_wafer = measure_deflectometry(wafer150, setup, seed=20)
    flat = make_flat_surface(wafer150.n_pixels, WAFER_150)
    m_flat = measure_deflectometry(flat, setup, seed=21)

    calibrated = absolute_calibrate(m_wafer.z_measured, m_flat.z_measured)

    error = plane_aligned_difference(
        calibrated, wafer150.z, wafer150.x, wafer150.y, np.isfinite(calibrated)
    )
    truth_rms = float(np.nanstd(wafer150.z[np.isfinite(calibrated)]))
    assert rms(error) < 0.01 * truth_rms


def test_rotation_helper_rotates_map_and_defects(wafer150):
    """Height map and defect ground truth rotate together."""
    rotated = rotate_surface_180(wafer150)

    assert np.allclose(
        rotated.z[rotated.mask], np.rot90(wafer150.z, 2)[rotated.mask], equal_nan=True
    )
    assert rotated.defects[0].x == pytest.approx(-wafer150.defects[0].x)
    assert rotated.defects[0].y == pytest.approx(-wafer150.defects[0].y)
    # Rotating twice restores the original.
    twice = rotate_surface_180(rotated)
    assert np.allclose(twice.z[twice.mask], wafer150.z[twice.mask], equal_nan=True)
