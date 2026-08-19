"""
Unit tests for phase-shifting interferometry.

Exercises the forward fringe model, the N-step phase estimator, both unwrapping
back-ends, and the end-to-end height round trip -- including the sampling limit
that decides whether unwrapping can work at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafer_metrology.interferometry import (
    HAVE_SKIMAGE,
    _dct_ls_unwrap,
    check_sampling,
    height_to_phase,
    measure_surface,
    phase_to_height,
    reconstruct_surface,
    recover_phase,
    simulate_fringes,
    synthetic_wavelength,
    unwrap_phase_2d,
    wrap,
)
from wafer_metrology.synthesize import apply_mask, make_wafer_grid, synthesize_wafer
from wafer_metrology.zernike import flatten

from .conftest import make_surface

HENE = 632.8e-9

# Both back-ends when scikit-image is installed, the bundled one otherwise.
UNWRAPPERS = [False, True] if HAVE_SKIMAGE else [False]


@pytest.fixture(scope="module")
def smooth_surface():
    """A defect-free, shape-removed wafer: nanotopography only, ~18 nm RMS.

    This is the regime single-wavelength interferometry actually operates in,
    and the only one where a 633 nm round trip is unambiguous.
    """
    clean = synthesize_wafer(n_pixels=192, seed=3, n_particles=0, n_scratches=0)
    smooth, _ = flatten(
        clean, nmax=6, remove_power=True,
        extra_terms=[(2, -2), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3)],
    )
    return smooth


# ---------------------------------------------------------------------------
# Height <-> phase
# ---------------------------------------------------------------------------


def test_height_phase_round_trip():
    """``phase_to_height`` inverts ``height_to_phase`` exactly."""
    z = np.linspace(-1e-6, 1e-6, 101)
    assert np.allclose(phase_to_height(height_to_phase(z, HENE), HENE), z, atol=1e-18)


def test_half_wave_of_height_is_a_full_fringe():
    """A lambda/2 height step is one 2 pi round-trip phase cycle."""
    assert height_to_phase(np.array([HENE / 2]), HENE)[0] == pytest.approx(2 * np.pi)


def test_wrap_maps_into_the_principal_interval():
    """``wrap`` folds arbitrary phase into ``(-pi, pi]``."""
    phi = np.array([0.0, 3.0, 7.0, -7.0, 100.0])
    wrapped = wrap(phi)
    assert np.all(wrapped > -np.pi - 1e-12)
    assert np.all(wrapped <= np.pi + 1e-12)
    # Wrapping only ever changes phase by whole cycles.
    cycles = (phi - wrapped) / (2 * np.pi)
    assert np.allclose(cycles, np.round(cycles))


def test_synthetic_wavelength_formula():
    """Two close lines behave like one much longer wavelength."""
    assert synthetic_wavelength(632.8e-9, 640.0e-9) == pytest.approx(56.2489e-6, rel=1e-4)
    assert synthetic_wavelength(640.0e-9, 632.8e-9) == pytest.approx(56.2489e-6, rel=1e-4)
    with pytest.raises(ValueError):
        synthetic_wavelength(HENE, HENE)


# ---------------------------------------------------------------------------
# Forward model and phase recovery
# ---------------------------------------------------------------------------


def test_fringes_follow_the_cosine_model():
    """Noise-free intensities match ``I0 (1 + V cos(phi + delta))``."""
    x, y, r, theta, mask = make_wafer_grid(64)
    z = apply_mask(1e-7 * x / 0.15, mask)
    frames = simulate_fringes(z, HENE, n_steps=4, visibility=0.8, noise=0.0, mask=mask)

    phi = height_to_phase(np.nan_to_num(z, nan=0.0), HENE)
    for k, delta in enumerate(frames.deltas_actual):
        expected = 1.0 * (1.0 + 0.8 * np.cos(phi + delta))
        assert np.allclose(frames.frames[k][mask], expected[mask], atol=1e-12)


def test_recover_phase_is_exact_without_noise():
    """The N-step estimator returns the true phase, modulo 2 pi."""
    x, y, r, theta, mask = make_wafer_grid(64)
    z = apply_mask(5e-8 * x / 0.15, mask)
    for n_steps in (3, 4, 5, 8):
        frames = simulate_fringes(z, HENE, n_steps=n_steps, noise=0.0, mask=mask)
        recovered = recover_phase(frames.frames, frames.deltas_nominal)
        truth = wrap(height_to_phase(np.nan_to_num(z, nan=0.0), HENE))
        assert np.allclose(wrap(recovered[mask] - truth[mask]), 0.0, atol=1e-9)


def test_too_few_steps_is_rejected():
    """Three unknowns need at least three frames."""
    with pytest.raises(ValueError):
        simulate_fringes(np.zeros((8, 8)), HENE, n_steps=2)


def test_step_error_is_recorded_and_applied():
    """A calibration error separates the actual steps from the nominal ones."""
    frames = simulate_fringes(np.zeros((8, 8)), HENE, n_steps=4, step_error=0.1, noise=0.0)
    assert not np.allclose(frames.deltas_actual, frames.deltas_nominal)
    assert np.allclose(frames.deltas_actual, frames.deltas_nominal * 1.1)


# ---------------------------------------------------------------------------
# Unwrapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefer_skimage", UNWRAPPERS)
def test_unwrap_round_trip_on_a_smooth_ramp(prefer_skimage):
    """Unwrapping recovers a multi-fringe ramp up to a constant offset."""
    x, y, r, theta, mask = make_wafer_grid(96)
    truth = 12.0 * np.pi * x / 0.15          # ~6 fringes across the wafer
    wrapped = apply_mask(wrap(truth), mask)

    unwrapped = unwrap_phase_2d(wrapped, mask, prefer_skimage=prefer_skimage)

    error = (unwrapped - truth)[mask]
    error -= error.mean()
    assert np.abs(error).max() < 1e-6


@pytest.mark.parametrize("prefer_skimage", UNWRAPPERS)
def test_unwrap_offset_is_a_whole_number_of_cycles(prefer_skimage):
    """The unwrap ambiguity is exactly a fringe order, never a fraction."""
    x, y, r, theta, mask = make_wafer_grid(96)
    truth = 8.0 * np.pi * y / 0.15
    unwrapped = unwrap_phase_2d(apply_mask(wrap(truth), mask), mask,
                                prefer_skimage=prefer_skimage)

    offset = float(np.nanmean((unwrapped - truth)[mask]))
    assert offset / (2 * np.pi) == pytest.approx(round(offset / (2 * np.pi)), abs=1e-6)


def test_bundled_unwrapper_is_congruent_with_its_input():
    """The DCT solution is snapped back onto the measured wrapped values."""
    x, y, r, theta, mask = make_wafer_grid(64)
    wrapped = wrap(10.0 * np.pi * (x + y) / 0.15)

    unwrapped = _dct_ls_unwrap(wrapped)

    residue = (unwrapped - wrapped) / (2 * np.pi)
    assert np.allclose(residue, np.round(residue), atol=1e-9)


def test_unwrap_preserves_the_mask(smooth_surface):
    """Pixels outside the aperture stay NaN through unwrapping."""
    frames = simulate_fringes(
        smooth_surface.z, HENE, noise=0.0, mask=smooth_surface.mask, seed=0
    )
    _, wrapped = reconstruct_surface(frames)
    unwrapped = unwrap_phase_2d(wrapped, smooth_surface.mask)
    assert np.isnan(unwrapped[~smooth_surface.mask]).all()
    assert np.isfinite(unwrapped[smooth_surface.mask]).all()


# ---------------------------------------------------------------------------
# Sampling limit
# ---------------------------------------------------------------------------


def test_sampling_check_flags_an_aliased_surface():
    """A warped wafer is aliased at 633 nm but fine at the synthetic wavelength."""
    wafer = synthesize_wafer(n_pixels=192, seed=0)

    worst_hene, hene_ok = check_sampling(wafer.z, HENE, wafer.mask)
    lam = synthetic_wavelength(632.8e-9, 640.0e-9)
    worst_syn, syn_ok = check_sampling(wafer.z, lam, wafer.mask)

    assert not hene_ok and worst_hene > np.pi
    assert syn_ok and worst_syn < np.pi
    # Phase scales inversely with wavelength.
    assert worst_hene / worst_syn == pytest.approx(lam / HENE, rel=1e-6)


def test_smooth_surface_is_within_the_sampling_limit(smooth_surface):
    """Once shape is removed, nanotopography is comfortably unwrappable."""
    worst, ok = check_sampling(smooth_surface.z, HENE, smooth_surface.mask)
    assert ok and worst < 1.0


# ---------------------------------------------------------------------------
# End-to-end measurement
# ---------------------------------------------------------------------------


def test_noiseless_measurement_is_essentially_exact(smooth_surface):
    """Without noise the reconstruction reproduces the surface to picometres."""
    measurement = measure_surface(smooth_surface, HENE, noise=0.0, seed=1)
    assert measurement.sampling_ok
    assert measurement.rms_error < 1e-12


def test_full_wafer_round_trip_at_the_synthetic_wavelength():
    """A 48 µm-PV wafer reconstructs exactly once the wavelength spans it."""
    wafer = synthesize_wafer(n_pixels=192, seed=0)
    lam = synthetic_wavelength(632.8e-9, 640.0e-9)

    measurement = measure_surface(wafer, lam, noise=0.0, seed=2)

    assert measurement.sampling_ok
    assert measurement.rms_error < 1e-11


def test_noise_degrades_the_measurement_monotonically(smooth_surface):
    """More detector noise means more reconstruction error."""
    errors = [
        measure_surface(smooth_surface, HENE, noise=level, seed=5).rms_error
        for level in (0.002, 0.01, 0.05)
    ]
    assert errors[0] < errors[1] < errors[2]
    assert errors[0] > 0.0


def test_noise_scales_the_height_error_as_expected(smooth_surface):
    """Height noise is linear in detector noise and in wavelength.

    Phase noise from additive intensity noise is proportional to the noise
    fraction, and height is ``lambda * phase / 4 pi`` -- so doubling either
    doubles the error.  That linearity is the whole cost of the synthetic
    wavelength trick.
    """
    low = measure_surface(smooth_surface, HENE, noise=0.01, seed=7).rms_error
    high = measure_surface(smooth_surface, HENE, noise=0.02, seed=7).rms_error
    assert high / low == pytest.approx(2.0, rel=0.15)


def test_phase_step_error_biases_the_result(smooth_surface):
    """A miscalibrated actuator produces error even with a noise-free detector."""
    clean = measure_surface(smooth_surface, HENE, noise=0.0, step_error=0.0, seed=9)
    biased = measure_surface(smooth_surface, HENE, noise=0.0, step_error=0.05, seed=9)
    assert biased.rms_error > 100 * max(clean.rms_error, 1e-15)


def test_measurement_reports_undersampling(caplog):
    """Measuring an aliased surface warns rather than silently returning junk."""
    wafer = synthesize_wafer(n_pixels=96, seed=0)
    with caplog.at_level("WARNING"):
        measurement = measure_surface(wafer, HENE, noise=0.0, seed=3)
    assert not measurement.sampling_ok
    assert "undersampled" in caplog.text


def test_measurement_is_deterministic(smooth_surface):
    """The same seed reproduces the same measurement exactly."""
    a = measure_surface(smooth_surface, HENE, noise=0.02, seed=42)
    b = measure_surface(smooth_surface, HENE, noise=0.02, seed=42)
    assert np.allclose(a.z_measured[a.z_measured == a.z_measured],
                       b.z_measured[b.z_measured == b.z_measured])
    assert a.rms_error == pytest.approx(b.rms_error, rel=1e-15)


def test_reconstruction_uses_the_nominal_steps():
    """Recovery assumes the steps the tool believes it took, so bias is visible."""
    x, y, r, theta, mask = make_wafer_grid(64)
    surface = make_surface(x, y, 2e-8 * x / 0.15, mask)
    frames = simulate_fringes(
        surface.z, HENE, noise=0.0, step_error=0.08, mask=mask, seed=0
    )
    z_biased, _ = reconstruct_surface(frames)

    # Recovering with the *actual* steps removes the bias entirely.
    ideal = recover_phase(frames.frames, frames.deltas_actual)
    z_ideal = phase_to_height(unwrap_phase_2d(ideal, mask), HENE)

    def centred(arr):
        return (arr - np.nanmean(arr[mask]))[mask]

    assert np.abs(centred(z_biased) - centred(surface.z)).max() > 1e-10
    assert np.abs(centred(z_ideal) - centred(surface.z)).max() < 1e-12
