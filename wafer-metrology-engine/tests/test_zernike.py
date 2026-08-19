"""
Unit tests for the Zernike decomposition module.

Covers the orthonormality of the basis on the unit disk, exact recovery of a
surface built from known coefficients, and the flatten operation.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafer_metrology.synthesize import apply_mask
from wafer_metrology.zernike import (
    PISTON,
    POWER,
    TILT,
    ansi_index,
    fit_surface,
    fit_zernikes,
    flatten,
    reconstruct,
    zernike,
    zernike_design_matrix,
    zernike_radial,
    zernike_terms,
)

from .conftest import WAFER_RADIUS, make_surface


# ---------------------------------------------------------------------------
# Basis properties
# ---------------------------------------------------------------------------


def test_term_enumeration_is_ansi_ordered():
    """Terms come back in ascending ANSI index with the expected count."""
    terms = zernike_terms(6)
    assert len(terms) == (6 + 1) * (6 + 2) // 2 == 28
    indices = [ansi_index(t) for t in terms]
    assert indices == sorted(indices)
    assert indices == list(range(len(terms)))
    assert terms[0] == PISTON
    assert terms[1:3] == [(1, -1), (1, 1)]


def test_radial_polynomial_boundary_values():
    """``R_n^m(1) == 1`` for every valid order, a defining property."""
    for n, m in zernike_terms(6):
        value = zernike_radial(n, m, np.array([1.0]))
        assert value[0] == pytest.approx(1.0, abs=1e-12)


def test_radial_polynomial_rejects_invalid_orders():
    """Orders with odd ``n - |m|`` or ``|m| > n`` are refused."""
    with pytest.raises(ValueError):
        zernike_radial(2, 1, np.array([0.5]))
    with pytest.raises(ValueError):
        zernike_radial(1, 3, np.array([0.5]))


def test_basis_is_orthonormal_on_the_disk(grid):
    """The Gram matrix over the disk approaches the identity.

    Orthonormality is exact only in the continuous limit, so the tolerance
    reflects the discretisation of a 192 px grid.
    """
    x, y, r, theta, mask = grid
    terms = zernike_terms(4)
    design = zernike_design_matrix(r[mask] / WAFER_RADIUS, theta[mask], terms)
    gram = design.T @ design / int(mask.sum())

    assert np.allclose(np.diag(gram), 1.0, atol=5e-3)
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.abs(off_diagonal).max() < 5e-3


def test_negative_m_is_the_sine_lobe(grid):
    """Positive ``m`` selects cosine, negative selects sine."""
    x, y, r, theta, mask = grid
    rho = r / WAFER_RADIUS
    cosine = zernike(2, 2, rho, theta)
    sine = zernike(2, -2, rho, theta)
    # A quarter-turn maps one lobe onto the other.
    rotated = zernike(2, 2, rho, theta - np.pi / 4)
    assert np.allclose(sine[mask], rotated[mask], atol=1e-9)
    assert not np.allclose(sine[mask], cosine[mask])


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_fit_recovers_known_coefficients(grid):
    """A surface built from known coefficients fits back to those coefficients."""
    x, y, r, theta, mask = grid
    truth = {(2, 0): 3.0e-6, (2, 2): -1.2e-6, (3, 1): 0.7e-6, (4, 0): 0.25e-6}
    z = reconstruct(truth, x, y, WAFER_RADIUS, mask)

    fit = fit_zernikes(x, y, z, WAFER_RADIUS, nmax=6, mask=mask)

    for term, expected in truth.items():
        assert fit[term] == pytest.approx(expected, abs=2e-9)
    # Terms that were not in the truth stay near zero.
    for term in fit.terms:
        if term not in truth:
            assert abs(fit[term]) < 2e-9
    assert fit.rms_residual < 1e-9
    assert fit.variance_explained > 0.999


def test_fit_residual_is_orthogonal_to_the_fitted_terms(wafer):
    """Least squares leaves a residual with no projection on the basis."""
    fit = fit_surface(wafer, nmax=4)
    rho = np.hypot(wafer.x, wafer.y) / wafer.radius
    theta = np.arctan2(wafer.y, wafer.x)
    valid = wafer.mask & np.isfinite(fit.residual)

    for n, m in fit.terms:
        basis = zernike(n, m, rho, theta)[valid]
        projection = float(np.mean(basis * fit.residual[valid]))
        assert abs(projection) < 1e-12


def test_dominant_terms_are_sorted_and_skip_piston(wafer):
    """``dominant`` returns the largest magnitudes, excluding piston."""
    fit = fit_surface(wafer, nmax=6)
    top = fit.dominant(5)
    magnitudes = [abs(v) for _, v in top]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(term != PISTON for term, _ in top)
    # The synthetic wafer is bow-dominated, so power leads.
    assert top[0][0] == POWER


def test_fit_raises_when_underdetermined(grid):
    """Fitting more terms than valid pixels is refused, not silently wrong."""
    x, y, r, theta, mask = grid
    tiny = mask & (r < 0.002)
    z = apply_mask(np.zeros_like(x), tiny)
    with pytest.raises(ValueError, match="valid pixels"):
        fit_zernikes(x, y, z, WAFER_RADIUS, nmax=6, mask=tiny)


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def test_flatten_removes_piston_tilt_and_power(grid):
    """After flattening, the removed terms refit to ~zero and others survive."""
    x, y, r, theta, mask = grid
    truth = {
        PISTON: 5.0e-6,
        TILT[0]: 2.0e-6,
        TILT[1]: -1.5e-6,
        POWER: 8.0e-6,
        (2, 2): 1.0e-6,     # astigmatism must be preserved
        (3, 3): -0.4e-6,    # trefoil must be preserved
    }
    surface = make_surface(x, y, reconstruct(truth, x, y, WAFER_RADIUS, mask), mask)

    flattened, _ = flatten(surface, nmax=6, remove_power=True)
    refit = fit_surface(flattened, nmax=6)

    for term in (PISTON, *TILT, POWER):
        assert abs(refit[term]) < 1e-9, f"{term} should have been removed"
    assert refit[(2, 2)] == pytest.approx(truth[(2, 2)], abs=2e-9)
    assert refit[(3, 3)] == pytest.approx(truth[(3, 3)], abs=2e-9)


def test_flatten_keeps_power_when_not_requested(grid):
    """With ``remove_power=False`` the bow term survives."""
    x, y, r, theta, mask = grid
    surface = make_surface(x, y, reconstruct({POWER: 8e-6}, x, y, WAFER_RADIUS, mask), mask)

    flattened, _ = flatten(surface, nmax=4, remove_power=False)

    assert fit_surface(flattened, nmax=4)[POWER] == pytest.approx(8e-6, abs=2e-9)


def test_flatten_reduces_peak_to_valley(wafer):
    """Flattening a real synthetic wafer strictly reduces PV."""
    flattened, fit = flatten(wafer, nmax=6, remove_power=True)
    assert flattened.pv() < wafer.pv()
    assert np.isnan(flattened.z[~wafer.mask]).all()
    assert fit.nmax == 6


def test_extra_terms_are_removed_too(grid):
    """``extra_terms`` removes further shape orders on request."""
    x, y, r, theta, mask = grid
    surface = make_surface(
        x, y, reconstruct({(2, 2): 4e-6, (3, 3): 2e-6}, x, y, WAFER_RADIUS, mask), mask
    )

    flattened, _ = flatten(surface, nmax=4, remove_power=True, extra_terms=[(2, 2)])
    refit = fit_surface(flattened, nmax=4)

    assert abs(refit[(2, 2)]) < 1e-9
    assert refit[(3, 3)] == pytest.approx(2e-6, abs=2e-9)
