"""
Unit tests for wafer synthesis.

Checks grid geometry, the circular aperture and NaN convention, determinism,
and that each shape component contributes what it claims to.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafer_metrology.synthesize import (
    WAFER_DIAMETER_150MM,
    WAFER_DIAMETER_300MM,
    WAFER_THICKNESS_150MM,
    WAFER_THICKNESS_300MM,
    apply_mask,
    gravity_sag_horizontal,
    make_flat_surface,
    make_wafer_grid,
    nanotopography,
    parabolic_bow,
    synthesize_wafer,
    synthesize_wafer_pair,
)

from .conftest import WAFER_RADIUS


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


def test_grid_spans_one_wafer_diameter():
    """The aperture is inscribed in the square grid, centred on the origin."""
    x, y, r, theta, mask = make_wafer_grid(256)

    assert x.shape == (256, 256)
    assert float(np.abs(x.mean())) < 1e-15
    assert float(np.abs(y.mean())) < 1e-15
    assert r[mask].max() <= WAFER_RADIUS + 1e-12
    # The masked fraction approaches pi/4, the disk-in-square area ratio.
    assert mask.mean() == pytest.approx(np.pi / 4, rel=0.01)


def test_edge_exclusion_shrinks_the_aperture():
    """An exclusion zone removes an annulus of the requested width."""
    _, _, r, _, full = make_wafer_grid(256)
    _, _, _, _, trimmed = make_wafer_grid(256, edge_exclusion=5e-3)

    assert trimmed.sum() < full.sum()
    assert r[trimmed].max() <= WAFER_RADIUS - 5e-3 + 1e-12


def test_edge_exclusion_cannot_consume_the_wafer():
    """An exclusion wider than the radius is refused."""
    with pytest.raises(ValueError, match="edge_exclusion"):
        make_wafer_grid(64, edge_exclusion=WAFER_DIAMETER_300MM)


def test_apply_mask_writes_nan_outside():
    """``apply_mask`` is the single place the NaN convention is enforced."""
    _, _, _, _, mask = make_wafer_grid(64)
    z = apply_mask(np.ones((64, 64)), mask)

    assert np.isnan(z[~mask]).all()
    assert np.isfinite(z[mask]).all()


# ---------------------------------------------------------------------------
# Shape components
# ---------------------------------------------------------------------------


def test_parabolic_bow_is_zero_mean_over_the_disk():
    """The bow term contributes shape but no piston."""
    _, _, r, _, mask = make_wafer_grid(256)
    rho = r / WAFER_RADIUS

    bow = parabolic_bow(rho, sag=30e-6)

    assert float(np.mean(bow[mask])) == pytest.approx(0.0, abs=2e-7)
    # Centre-to-edge sag is the requested value.
    assert bow[mask].max() - float(bow[128, 128]) == pytest.approx(30e-6, rel=0.02)


def test_nanotopography_hits_its_target_rms():
    """The random component is scaled to exactly the requested RMS."""
    rng = np.random.default_rng(0)
    field = nanotopography((256, 256), 1.17e-3, rms=20e-9, correlation_length=3e-3, rng=rng)

    assert float(np.sqrt(np.mean(field**2))) == pytest.approx(20e-9, rel=1e-9)
    assert float(np.mean(field)) == pytest.approx(0.0, abs=1e-12)


def test_zero_nanotopography_is_exactly_flat():
    """Requesting zero RMS short-circuits to zeros, not to tiny noise."""
    rng = np.random.default_rng(0)
    field = nanotopography((32, 32), 1e-3, rms=0.0, correlation_length=3e-3, rng=rng)
    assert np.all(field == 0.0)


def test_smoothing_makes_neighbours_correlated():
    """Nanotopography is spatially correlated, unlike white noise."""
    rng = np.random.default_rng(1)
    field = nanotopography((256, 256), 1.17e-3, rms=20e-9, correlation_length=5e-3, rng=rng)

    neighbour_diff = np.abs(np.diff(field, axis=1)).mean()
    assert neighbour_diff < 0.2 * float(np.sqrt(np.mean(field**2)))


# ---------------------------------------------------------------------------
# Whole-wafer synthesis
# ---------------------------------------------------------------------------


def test_synthesis_is_deterministic():
    """The same seed and grid reproduce the surface bit for bit."""
    a = synthesize_wafer(n_pixels=128, seed=7)
    b = synthesize_wafer(n_pixels=128, seed=7)
    c = synthesize_wafer(n_pixels=128, seed=8)

    assert np.array_equal(a.z[a.mask], b.z[b.mask])
    assert not np.array_equal(a.z[a.mask], c.z[c.mask])
    assert [d.x for d in a.defects] == [d.x for d in b.defects]


def test_surface_unpacks_as_x_y_z_mask(wafer):
    """The documented 4-tuple contract holds."""
    x, y, z, mask = wafer

    assert x.shape == y.shape == z.shape == mask.shape
    assert mask.dtype == np.bool_
    assert np.isnan(z[~mask]).all()


def test_defect_truth_matches_the_request():
    """Injected defect counts and signs are as requested."""
    surface = synthesize_wafer(n_pixels=192, seed=2, n_particles=4, n_scratches=2)

    kinds = [d.kind for d in surface.defects]
    assert kinds.count("particle") == 4
    assert kinds.count("scratch") == 2
    assert all(d.height > 0 for d in surface.defects if d.kind == "particle")
    assert all(d.height < 0 for d in surface.defects if d.kind == "scratch")
    # Every defect lands inside the wafer.
    assert all(np.hypot(d.x, d.y) < WAFER_RADIUS for d in surface.defects)


def test_defects_can_be_suppressed():
    """A clean wafer has no defect records and a much smaller residual tail."""
    clean = synthesize_wafer(n_pixels=192, seed=2, n_particles=0, n_scratches=0)
    dirty = synthesize_wafer(n_pixels=192, seed=2, n_particles=6, n_scratches=2)

    assert clean.defects == []
    assert dirty.pv() > clean.pv()


def test_summary_statistics_ignore_the_masked_region(wafer):
    """PV and RMS are computed over the aperture only."""
    assert np.isfinite(wafer.pv())
    assert np.isfinite(wafer.rms())
    assert wafer.pv() == pytest.approx(
        float(np.nanmax(wafer.z) - np.nanmin(wafer.z)), rel=1e-12
    )
    assert wafer.values.size == int(wafer.mask.sum())


def test_with_z_preserves_grid_metadata(wafer):
    """Swapping the height map keeps the grid, mask and defect truth."""
    replaced = wafer.with_z(np.zeros_like(wafer.z))

    assert replaced.radius == wafer.radius
    assert replaced.pixel_size == wafer.pixel_size
    assert np.array_equal(replaced.mask, wafer.mask)
    assert replaced.defects == wafer.defects


# ---------------------------------------------------------------------------
# Front/back pairs
# ---------------------------------------------------------------------------


def test_pair_thickness_is_positive_and_near_nominal(wafer_pair):
    """The front face sits above the back face everywhere."""
    thickness = wafer_pair.thickness[wafer_pair.mask]

    assert np.all(thickness > 0)
    assert float(np.mean(thickness)) == pytest.approx(775e-6, rel=1e-3)


def test_median_surface_is_independent_of_thickness():
    """Changing the thickness field leaves the median surface untouched."""
    thin = synthesize_wafer_pair(n_pixels=128, seed=4, ttv_amplitude=0.2e-6,
                                 thickness_noise_rms=0.0)
    thick = synthesize_wafer_pair(n_pixels=128, seed=4, ttv_amplitude=4.0e-6,
                                  thickness_noise_rms=0.0)

    mask = thin.mask
    assert np.allclose(thin.median[mask], thick.median[mask], atol=1e-15)
    assert float(np.ptp(thick.thickness[mask])) > float(np.ptp(thin.thickness[mask]))


def test_pair_front_matches_a_standalone_surface(wafer_pair):
    """``pair.front`` is a fully-formed WaferSurface sharing the grid."""
    front = wafer_pair.front

    assert np.array_equal(front.z[front.mask], wafer_pair.z_front[wafer_pair.mask])
    assert front.radius == wafer_pair.radius
    assert front.defects == wafer_pair.defects


def test_flat_surface_factory_is_flat_and_gridded():
    """The reference-flat factory returns a zero surface with full metadata."""
    flat = make_flat_surface(96, WAFER_DIAMETER_150MM)

    assert flat.radius == pytest.approx(75e-3)
    assert flat.pixel_size == pytest.approx(WAFER_DIAMETER_150MM / 96)
    assert np.all(flat.z[flat.mask] == 0.0)
    assert np.isnan(flat.z[~flat.mask]).all()
    assert flat.defects == []


def test_gravity_sag_justifies_the_vertical_mount():
    """Horizontal sag is the same order as the bow being measured.

    ~7 µm (edge-supported lower bound) for 150 mm x 675 µm, ~90 µm for
    300 mm x 775 µm -- comparable to 30-60 µm warp specs, which is the
    quantitative argument for the vertical 3-point mount.
    """
    sag_150 = gravity_sag_horizontal(WAFER_DIAMETER_150MM, WAFER_THICKNESS_150MM)
    sag_300 = gravity_sag_horizontal(WAFER_DIAMETER_300MM, WAFER_THICKNESS_300MM)

    assert 4e-6 < sag_150 < 12e-6
    assert 60e-6 < sag_300 < 130e-6
    # Plate scaling: the load q = rho g h itself grows with thickness, so
    # sag ~ (q a^4) / (E h^3) ~ a^4 / h^2.
    expected_ratio = (300 / 150) ** 4 / (775 / 675) ** 2
    assert sag_300 / sag_150 == pytest.approx(expected_ratio, rel=1e-6)


def test_ttv_amplitude_is_respected():
    """The deterministic thickness range tracks the requested amplitude."""
    pair = synthesize_wafer_pair(n_pixels=192, seed=5, ttv_amplitude=2e-6,
                                 thickness_noise_rms=0.0)
    assert float(np.ptp(pair.thickness[pair.mask])) == pytest.approx(2e-6, rel=1e-6)
