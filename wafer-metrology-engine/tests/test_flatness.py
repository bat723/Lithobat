"""
Unit tests for SEMI flatness metrology.

Bow, warp and SFQR are checked against analytic surfaces whose values can be
derived in closed form, so the tests pin the definitions, not just the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafer_metrology.flatness import (
    DEFAULT_SITE_SIZE,
    compute_flatness,
    fit_reference_plane,
    plane_deviation,
    site_flatness,
)
from wafer_metrology.synthesize import WaferPair, apply_mask, make_wafer_grid

from .conftest import WAFER_RADIUS, make_surface


@pytest.fixture(scope="module")
def fine_grid():
    """A 384 px grid: fine enough for the analytic tolerances below."""
    return make_wafer_grid(384)


# ---------------------------------------------------------------------------
# Reference planes
# ---------------------------------------------------------------------------


def test_plane_fit_recovers_a_known_plane(fine_grid):
    """A pure plane is fitted exactly and leaves no deviation."""
    x, y, r, theta, mask = fine_grid
    z = apply_mask(3e-6 * x / WAFER_RADIUS - 1e-6 * y / WAFER_RADIUS + 5e-6, mask)

    plane, (a, b, c) = fit_reference_plane(x, y, z, mask)

    assert a == pytest.approx(3e-6 / WAFER_RADIUS, rel=1e-9)
    assert b == pytest.approx(-1e-6 / WAFER_RADIUS, rel=1e-9)
    assert c == pytest.approx(5e-6, abs=1e-15)
    assert np.abs((plane - z)[mask]).max() < 1e-15
    assert np.abs(plane_deviation(x, y, z, mask)[mask]).max() < 1e-15


def test_plane_fit_needs_three_points(fine_grid):
    """Fitting a plane to fewer than three pixels raises."""
    x, y, r, theta, mask = fine_grid
    tiny = np.zeros_like(mask)
    tiny[0, 0] = True
    with pytest.raises(ValueError):
        fit_reference_plane(x, y, apply_mask(x, tiny), tiny)


# ---------------------------------------------------------------------------
# Bow and warp
# ---------------------------------------------------------------------------


def test_bow_and_warp_of_an_analytic_paraboloid(fine_grid):
    """For ``z = a r^2`` on a disk of radius R the closed forms are exact.

    The least-squares plane through a rotationally symmetric surface is
    horizontal at the mean height, and ``mean(r^2) = R^2 / 2`` over a disk, so
    the deviation is ``a (r^2 - R^2/2)``: bow is ``-a R^2 / 2`` at the centre
    and warp is the full ``a R^2``.
    """
    x, y, r, theta, mask = fine_grid
    target_warp = 100e-6
    a = target_warp / WAFER_RADIUS**2
    surface = make_surface(x, y, a * r**2, mask)

    metrics = compute_flatness(surface, edge_exclusion=0.0).metrics

    assert metrics.bow == pytest.approx(-0.5 * target_warp, rel=2e-3)
    assert metrics.warp == pytest.approx(target_warp, rel=2e-3)
    assert metrics.warp >= abs(metrics.bow)


def test_bow_sign_flips_with_the_surface(fine_grid):
    """A concave-up wafer has negative bow; inverting it flips the sign."""
    x, y, r, theta, mask = fine_grid
    a = 100e-6 / WAFER_RADIUS**2

    up = compute_flatness(make_surface(x, y, a * r**2, mask), edge_exclusion=0.0).metrics
    down = compute_flatness(make_surface(x, y, -a * r**2, mask), edge_exclusion=0.0).metrics

    assert up.bow < 0 < down.bow
    assert up.bow == pytest.approx(-down.bow, rel=1e-6)
    assert up.warp == pytest.approx(down.warp, rel=1e-6)


def test_tilt_does_not_affect_bow_or_warp(fine_grid):
    """The reference plane absorbs tilt, so shape metrics are unchanged by it."""
    x, y, r, theta, mask = fine_grid
    a = 60e-6 / WAFER_RADIUS**2
    flat = compute_flatness(make_surface(x, y, a * r**2, mask), edge_exclusion=0.0).metrics
    tilted = compute_flatness(
        make_surface(x, y, a * r**2 + 40e-6 * x / WAFER_RADIUS, mask), edge_exclusion=0.0
    ).metrics

    assert tilted.bow == pytest.approx(flat.bow, rel=1e-6)
    assert tilted.warp == pytest.approx(flat.warp, rel=1e-6)


# ---------------------------------------------------------------------------
# Thickness
# ---------------------------------------------------------------------------


def test_ttv_from_a_known_thickness_wedge(fine_grid):
    """TTV equals the range of an imposed thickness field."""
    x, y, r, theta, mask = fine_grid
    nominal = 775e-6
    slope = 2e-6 / WAFER_RADIUS          # 2 µm from centre to edge
    thickness = nominal + slope * x
    median = 10e-6 * (r / WAFER_RADIUS) ** 2

    pair = WaferPair(
        x=x, y=y,
        z_front=apply_mask(median + 0.5 * thickness, mask),
        z_back=apply_mask(median - 0.5 * thickness, mask),
        mask=mask, radius=WAFER_RADIUS, pixel_size=2 * WAFER_RADIUS / 384,
    )
    metrics = compute_flatness(pair, edge_exclusion=0.0).metrics

    expected = float(np.ptp((slope * x)[mask]))
    assert metrics.ttv == pytest.approx(expected, rel=1e-6)
    assert metrics.thickness_mean == pytest.approx(nominal, abs=1e-9)


def test_thickness_cancels_out_of_the_median_surface(fine_grid):
    """Bow and warp are blind to thickness, which is why they use the median."""
    x, y, r, theta, mask = fine_grid
    median = 10e-6 * (r / WAFER_RADIUS) ** 2

    thin = WaferPair(
        x=x, y=y,
        z_front=apply_mask(median + 0.5 * 775e-6, mask),
        z_back=apply_mask(median - 0.5 * 775e-6, mask),
        mask=mask, radius=WAFER_RADIUS, pixel_size=2 * WAFER_RADIUS / 384,
    )
    wedged_thickness = 775e-6 + 5e-6 * x / WAFER_RADIUS
    wedged = WaferPair(
        x=x, y=y,
        z_front=apply_mask(median + 0.5 * wedged_thickness, mask),
        z_back=apply_mask(median - 0.5 * wedged_thickness, mask),
        mask=mask, radius=WAFER_RADIUS, pixel_size=2 * WAFER_RADIUS / 384,
    )

    a = compute_flatness(thin, edge_exclusion=0.0).metrics
    b = compute_flatness(wedged, edge_exclusion=0.0).metrics

    assert b.bow == pytest.approx(a.bow, rel=1e-9)
    assert b.warp == pytest.approx(a.warp, rel=1e-9)
    assert b.ttv > a.ttv                       # but thickness metrics do change


def test_single_surface_has_no_ttv(wafer):
    """A one-sided measurement reports NaN thickness rather than a wrong number."""
    metrics = compute_flatness(wafer).metrics
    assert np.isnan(metrics.ttv)
    assert np.isnan(metrics.thickness_mean)
    assert np.isfinite(metrics.warp)


# ---------------------------------------------------------------------------
# Site flatness (SFQR)
# ---------------------------------------------------------------------------


def test_sites_tile_on_the_requested_pitch(fine_grid):
    """Site centres land on the site grid and complete sites lie inside the QA."""
    x, y, r, theta, mask = fine_grid
    z = apply_mask(np.zeros_like(x), mask)
    edge_exclusion = 3e-3

    table, _ = site_flatness(
        x, y, z, mask, WAFER_RADIUS,
        site_size=DEFAULT_SITE_SIZE, edge_exclusion=edge_exclusion,
    )

    # Centres are exactly index * pitch.
    centres = table["x_center_mm"].to_numpy() * 1e-3
    assert np.allclose(centres, table["ix"].to_numpy() * DEFAULT_SITE_SIZE)
    # One site is centred on the wafer centre by default.
    assert ((table["ix"] == 0) & (table["iy"] == 0)).sum() == 1
    # Every complete site fits wholly inside the quality area.
    complete = table.loc[table["complete"]]
    half = 0.5 * DEFAULT_SITE_SIZE
    corner = np.hypot(
        complete["x_center_mm"].abs() * 1e-3 + half,
        complete["y_center_mm"].abs() * 1e-3 + half,
    )
    assert (corner <= WAFER_RADIUS - edge_exclusion + 1e-12).all()
    assert len(complete) > 0
    assert len(complete) < len(table)          # partial edge sites are excluded


def test_planar_surface_has_zero_site_flatness(fine_grid):
    """A tilted plane is perfectly flat within every site."""
    x, y, r, theta, mask = fine_grid
    z = apply_mask(4e-6 * x / WAFER_RADIUS - 2e-6 * y / WAFER_RADIUS, mask)

    table, sfqr_map = site_flatness(x, y, z, mask, WAFER_RADIUS)

    assert table.loc[table["complete"], "sfqr"].max() < 1e-15
    assert np.nanmax(sfqr_map) < 1e-15


def test_sfqr_of_an_analytic_quadratic():
    """For ``z = a x^2`` every site sees ``a u^2`` locally, so SFQR is ``a s^2 / 4``.

    Within a site the constant and linear parts of ``a (c + u)^2`` are absorbed
    by that site's own plane, leaving ``a u^2``.  Its least-squares residual
    over ``u`` in ``[-s/2, s/2]`` is ``a (u^2 - s^2/12)``, running from
    ``-a s^2 / 12`` at the centre to ``a s^2 / 6`` at the edge -- a range of
    ``a s^2 / 4``, identical for every site.

    A 768 px grid puts 64 samples across a 25 mm site; the outermost sample
    sits half a pixel inside the site edge, which biases the measured range
    about 1.6 % low against the continuum value.
    """
    x, y, r, theta, mask = make_wafer_grid(768)
    site = DEFAULT_SITE_SIZE
    a = 1.0e-3                                  # curvature scale [m^-1]
    z = apply_mask(a * x**2, mask)

    table, _ = site_flatness(x, y, z, mask, WAFER_RADIUS, site_size=site)

    expected = a * site**2 / 4.0
    measured = table.loc[table["complete"], "sfqr"].to_numpy()
    # Every complete site must return the same value -- the surface is uniform.
    assert measured.std() < 1e-18
    assert np.allclose(measured, expected, rtol=0.03)


def test_larger_sites_are_less_flat(wafer):
    """SFQR grows with site size: a bigger field spans more of the wafer's shape."""
    small = compute_flatness(wafer, site_size=10e-3).metrics
    large = compute_flatness(wafer, site_size=40e-3).metrics
    assert large.sfqr_mean > small.sfqr_mean


def test_site_size_must_be_positive(fine_grid):
    """A non-positive site size is refused."""
    x, y, r, theta, mask = fine_grid
    with pytest.raises(ValueError):
        site_flatness(x, y, apply_mask(x, mask), mask, WAFER_RADIUS, site_size=0.0)


def test_metrics_round_trip_through_the_display_table(wafer_pair):
    """The display table carries every metric with consistent units."""
    result = compute_flatness(wafer_pair)
    frame = result.metrics.to_frame()

    assert list(frame.columns) == ["parameter", "value", "unit", "description"]
    warp_row = frame.loc[frame["parameter"] == "Warp", "value"].iloc[0]
    assert warp_row == pytest.approx(result.metrics.warp * 1e6, rel=1e-12)
    assert result.metrics.to_series()["warp"] == pytest.approx(result.metrics.warp)
