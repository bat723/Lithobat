"""
Unit tests for the fibre-coupling models and the warp-to-loss capstone.

The closed-form loss models are pinned to the plan's defended numbers (1 dB at
2.5 µm, 3 dB at 4.3 µm, ~56 µm gap, ~2.6 degrees) and the combined
lateral-plus-gap formula is verified against a brute-force numerical overlap
integral, so the formula is checked against physics, not against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafer_metrology.coupling import (
    SMF28_W0_1550,
    WAVELENGTH_1550,
    angular_loss_db,
    attach_loss_budget,
    combined_loss_db,
    divergence_half_angle,
    etalon_transmission_db,
    fit_lateral_scan,
    gap_loss_db,
    lateral_loss_db,
    rayleigh_range,
)
from wafer_metrology.synthesize import make_flat_surface, synthesize_wafer


# ---------------------------------------------------------------------------
# Closed-form models against the plan's defended numbers
# ---------------------------------------------------------------------------


def test_beam_constants_match_smf28():
    """z_R ~ 55 µm and theta0 ~ 95 mrad at 1550 nm."""
    assert rayleigh_range() == pytest.approx(54.8e-6, rel=0.01)
    assert divergence_half_angle() == pytest.approx(94.9e-3, rel=0.01)


def test_lateral_loss_thresholds():
    """1 dB at 2.5 µm, 3 dB at 4.3 µm, 10 dB at 7.9 µm."""
    assert float(lateral_loss_db(2.5e-6)) == pytest.approx(1.0, abs=0.02)
    assert float(lateral_loss_db(4.3e-6)) == pytest.approx(3.0, abs=0.05)
    assert float(lateral_loss_db(7.9e-6)) == pytest.approx(10.0, abs=0.1)
    assert float(lateral_loss_db(0.0)) == 0.0


def test_gap_and_angular_thresholds():
    """1 dB at ~56 µm of gap and ~2.6 degrees of tilt."""
    assert float(gap_loss_db(55.8e-6)) == pytest.approx(1.0, abs=0.02)
    assert float(angular_loss_db(np.deg2rad(2.6))) == pytest.approx(1.0, abs=0.03)


def test_combined_reduces_to_the_pure_cases():
    """The combined formula owns both limits."""
    assert float(combined_loss_db(3e-6, 0.0)) == pytest.approx(
        float(lateral_loss_db(3e-6)), rel=1e-12
    )
    assert float(combined_loss_db(0.0, 40e-6)) == pytest.approx(
        float(gap_loss_db(40e-6)), rel=1e-12
    )


def test_gap_widens_the_lateral_tolerance():
    """At a large gap the arriving beam is bigger, so offsets cost less extra."""
    excess_close = combined_loss_db(4e-6, 0.0) - combined_loss_db(0.0, 0.0)
    excess_far = combined_loss_db(4e-6, 100e-6) - combined_loss_db(0.0, 100e-6)
    assert excess_far < excess_close


def test_combined_formula_matches_a_numerical_overlap_integral():
    """Brute-force Gaussian-beam overlap agrees with the closed form.

    Propagates the emitted Gaussian (expanded radius and phase curvature) to
    the second facet, offsets it, and evaluates the mode-overlap integral on a
    grid -- checking the formula against physics rather than against itself.
    """
    w0 = SMF28_W0_1550
    lam = WAVELENGTH_1550
    z_r = np.pi * w0**2 / lam
    k = 2.0 * np.pi / lam

    half = 40e-6
    n = 601
    axis = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(axis, axis)
    mode = np.exp(-(xx**2 + yy**2) / w0**2)

    for offset, gap in ((0.0, 30e-6), (3e-6, 0.0), (3e-6, 60e-6), (5e-6, 100e-6)):
        if gap > 0:
            w1 = w0 * np.sqrt(1.0 + (gap / z_r) ** 2)
            curvature = gap * (1.0 + (z_r / gap) ** 2)
            phase = np.exp(-1j * k * ((xx - offset) ** 2 + yy**2) / (2.0 * curvature))
        else:
            w1 = w0
            phase = 1.0
        beam = np.exp(-((xx - offset) ** 2 + yy**2) / w1**2) * phase

        overlap = np.abs(np.sum(beam * np.conj(mode))) ** 2
        norm = np.sum(np.abs(beam) ** 2) * np.sum(np.abs(mode) ** 2)
        numeric_db = -10.0 * np.log10(overlap / norm)

        assert numeric_db == pytest.approx(
            float(combined_loss_db(offset, gap)), abs=0.02
        ), f"mismatch at offset {offset}, gap {gap}"


def test_etalon_ripple_amplitude_and_period():
    """R = 3.5 % gives 0.61 dB pk-pk ripple with period lambda/2 in gap."""
    gaps = np.linspace(20e-6, 20e-6 + 3e-6, 6001)
    ripple = etalon_transmission_db(gaps)

    assert float(ripple.max() - ripple.min()) == pytest.approx(0.607, abs=0.01)
    assert float(ripple.max()) == pytest.approx(0.0, abs=1e-9)
    # Period: transmission repeats every lambda/2 of gap.
    shifted = etalon_transmission_db(gaps + WAVELENGTH_1550 / 2)
    assert np.allclose(ripple, shifted, atol=1e-9)


# ---------------------------------------------------------------------------
# Fitting a measured scan
# ---------------------------------------------------------------------------


def test_fit_recovers_w0_center_and_floor():
    """The plan's +/-10 µm, 1 µm-step scan pins w0 to a few percent."""
    rng = np.random.default_rng(0)
    offsets = np.arange(-10, 11, dtype=np.float64) * 1e-6
    truth = lateral_loss_db(offsets - 0.4e-6) + 0.35
    measured = truth + rng.normal(0.0, 0.05, offsets.size)

    fit = fit_lateral_scan(offsets, measured)

    assert fit.w0 == pytest.approx(SMF28_W0_1550, rel=0.05)
    assert fit.center == pytest.approx(0.4e-6, abs=0.3e-6)
    assert fit.floor_db == pytest.approx(0.35, abs=0.1)
    assert fit.residual_rms_db < 0.1
    assert np.isfinite(fit.model(offsets)).all()


def test_fit_needs_enough_points():
    """Three parameters cannot come from three points."""
    with pytest.raises(ValueError):
        fit_lateral_scan(np.zeros(3), np.zeros(3))


# ---------------------------------------------------------------------------
# Capstone
# ---------------------------------------------------------------------------


def test_flat_wafer_costs_nothing():
    """A perfectly flat wafer produces a ~zero-dB budget everywhere."""
    flat = make_flat_surface(192, 150e-3)
    budget = attach_loss_budget(flat)

    assert abs(budget.worst_db) < 1e-9
    assert abs(budget.median_db) < 1e-9


def test_warped_wafer_produces_a_positive_ordered_budget():
    """A real synthetic wafer yields worst >= p95 >= median, all finite."""
    wafer = synthesize_wafer(n_pixels=192, diameter=150e-3, seed=0)
    budget = attach_loss_budget(wafer)

    assert budget.worst_db >= budget.p95_db >= budget.median_db
    assert budget.worst_db > 0.05
    assert np.isfinite(budget.table["loss_total_db"]).all()
    assert np.isfinite(budget.penalty_map[np.isfinite(budget.penalty_map)]).all()
    assert "dB" in budget.headline


def test_longer_lever_costs_more():
    """The lateral term scales with the lever arm, so the budget must too."""
    wafer = synthesize_wafer(n_pixels=192, diameter=150e-3, seed=1)
    short = attach_loss_budget(wafer, lever_arm=1e-3)
    long = attach_loss_budget(wafer, lever_arm=5e-3)

    assert long.worst_db > short.worst_db
    assert long.table["loss_lateral_db"].max() > short.table["loss_lateral_db"].max()


def test_angular_term_is_negligible_at_warp_scale():
    """Warp-scale tilts are far inside the ~95 mrad divergence cone.

    This is a *result*, not an accident: theta0 is huge compared to any
    milliradian-scale wafer tilt, which is why the capstone's angular column
    reads ~zero and the budget is carried by lateral and gap.
    """
    wafer = synthesize_wafer(n_pixels=192, diameter=150e-3, seed=0)
    budget = attach_loss_budget(wafer)

    assert budget.table["loss_angular_db"].max() < 0.01
    assert budget.table["loss_lateral_db"].max() > budget.table["loss_angular_db"].max()


def test_oversized_die_is_refused():
    """A die bigger than the wafer leaves no complete site."""
    wafer = synthesize_wafer(n_pixels=128, diameter=150e-3, seed=0)
    with pytest.raises(ValueError, match="complete"):
        attach_loss_budget(wafer, die_size=200e-3)
