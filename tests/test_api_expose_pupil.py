"""
API-level tests for the pupil subfeature of ``litho_sim.expose``.

Exercises ``pupil_grid`` and ``zernike_noll`` strictly through the public
package interface, plus one small end-to-end check that Zernike aberration
coefficients from ``OpticsConfig`` actually reach the imaging path.

Run with::

    .venv/bin/python -m pytest tests/test_api_expose_pupil.py -q
"""

from __future__ import annotations

# Ensure src/ is on the path when running tests directly
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core import GridConfig, OpticsConfig
from litho_sim.expose import compute_aerial_image, pupil_grid, zernike_noll
from litho_sim.mask import lines_and_spaces

# ---------------------------------------------------------------------------
# pupil_grid
# ---------------------------------------------------------------------------


def test_pupil_grid_shapes_and_layout():
    """All four arrays are (n, n); rows = eta, columns = xi, spanning [-1, 1]."""
    n = 65
    rho, phi, xi, eta = pupil_grid(n)

    for arr in (rho, phi, xi, eta):
        assert arr.shape == (n, n)

    # Columns carry xi: every row is the same ramp from -1 to +1.
    assert np.allclose(xi[0], xi[-1])
    assert np.allclose(xi[:, 0], -1.0)
    assert np.allclose(xi[:, -1], 1.0)

    # Rows carry eta: every column is the same ramp from -1 to +1.
    assert np.allclose(eta[:, 0], eta[:, -1])
    assert np.allclose(eta[0, :], -1.0)
    assert np.allclose(eta[-1, :], 1.0)


def test_pupil_grid_polar_consistency():
    """rho and phi are consistent with the Cartesian arrays."""
    n = 65
    rho, phi, xi, eta = pupil_grid(n)

    assert np.allclose(rho, np.hypot(xi, eta))
    assert np.allclose(phi, np.arctan2(eta, xi))

    # Odd n puts a sample exactly on the optical axis.
    c = n // 2
    assert rho[c, c] == pytest.approx(0.0)
    # Corner of the [-1, 1] square.
    assert rho[0, 0] == pytest.approx(math.sqrt(2.0))
    # phi = 0 on the positive-xi axis.
    assert phi[c, -1] == pytest.approx(0.0)


def test_pupil_grid_unit_circle_fraction():
    """The unit disc fills ~pi/4 of the square grid."""
    n = 101
    rho, *_ = pupil_grid(n)
    fraction = np.mean(rho <= 1.0)
    assert fraction == pytest.approx(math.pi / 4.0, abs=0.02)


# ---------------------------------------------------------------------------
# zernike_noll — low-index properties
# ---------------------------------------------------------------------------


@pytest.fixture
def grid65():
    rho, phi, xi, eta = pupil_grid(65)
    inside = rho <= 1.0
    return rho, phi, xi, eta, inside


def test_zernike_piston_is_constant(grid65):
    """Z1 (piston) is 1 everywhere on the (clipped) pupil."""
    rho, phi = grid65[0], grid65[1]
    z1 = zernike_noll(1, rho, phi)
    assert z1.shape == rho.shape
    assert np.allclose(z1, 1.0)


def test_zernike_tilts_are_linear(grid65):
    """Z2 = 2*xi and Z3 = 2*eta inside the pupil (Noll normalisation)."""
    rho, phi, xi, eta, inside = grid65
    z2 = zernike_noll(2, rho, phi)
    z3 = zernike_noll(3, rho, phi)
    assert np.allclose(z2[inside], 2.0 * xi[inside], atol=1e-12)
    assert np.allclose(z3[inside], 2.0 * eta[inside], atol=1e-12)


def test_zernike_defocus_radial(grid65):
    """Z4 = sqrt(3)(2 rho^2 - 1): rotationally symmetric, known extremes."""
    rho, phi, _, _, inside = grid65
    z4 = zernike_noll(4, rho, phi)
    assert np.allclose(z4[inside], math.sqrt(3.0) * (2.0 * rho[inside] ** 2 - 1.0))
    c = rho.shape[0] // 2
    assert z4[c, c] == pytest.approx(-math.sqrt(3.0))
    # Depends only on rho, and the grid is symmetric under xi <-> eta swap.
    assert np.allclose(z4, z4.T)


def test_zernike_parity_orthogonality(grid65):
    """Pairs orthogonal by parity vanish exactly even on a discrete grid."""
    rho, phi, _, _, inside = grid65
    z2 = zernike_noll(2, rho, phi)[inside]
    z3 = zernike_noll(3, rho, phi)[inside]
    z4 = zernike_noll(4, rho, phi)[inside]

    def normed_dot(a, b):
        return abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

    assert normed_dot(z2, z3) < 1e-10  # odd in xi vs odd in eta
    assert normed_dot(z2, z4) < 1e-10  # odd in xi vs even
    assert normed_dot(z3, z4) < 1e-10  # odd in eta vs even


# ---------------------------------------------------------------------------
# zernike_noll — edge cases (documented behaviour: warn and return zeros)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_j", [0, -3, 22, 999])
def test_zernike_unimplemented_index_returns_zeros(grid65, bad_j, caplog):
    """Indices outside Noll 1..21 return an all-zero array and log a warning."""
    rho, phi, *_ = grid65
    with caplog.at_level(logging.WARNING, logger="litho_sim.expose.pupil"):
        z = zernike_noll(bad_j, rho, phi)
    assert z.shape == rho.shape
    assert not np.any(z)
    assert any("not implemented" in rec.message for rec in caplog.records)


def test_zernike_incompatible_shapes_raise():
    """Non-broadcastable rho/phi shapes fail loudly for azimuthal terms."""
    rho = np.zeros((8, 8))
    phi = np.zeros((7, 5))
    with pytest.raises(ValueError):
        zernike_noll(2, rho, phi)  # m != 0, so phi participates


# ---------------------------------------------------------------------------
# End-to-end: config coefficients reach the imaging path
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_imaging():
    grid = GridConfig(n_pixels=64, pixel_size=8e-9)
    mask = lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=200e-9, cd=100e-9)
    return grid, mask


def _optics(**kwargs) -> OpticsConfig:
    return OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.7, **kwargs)


def test_coma_aberration_changes_aerial_image(tiny_imaging):
    """A coma coefficient in OpticsConfig measurably alters the image."""
    grid, mask = tiny_imaging
    base = compute_aerial_image(mask, _optics(), grid)
    coma = compute_aerial_image(mask, _optics(zernike_coeffs={7: 0.05}), grid)

    assert base.shape == coma.shape == (64, 64)
    assert np.all(np.isfinite(coma)) and np.all(coma >= 0.0)
    assert np.max(np.abs(coma - base)) > 1e-3


def test_piston_aberration_leaves_image_invariant(tiny_imaging):
    """Piston is a constant phase over the aperture: intensity must not move."""
    grid, mask = tiny_imaging
    base = compute_aerial_image(mask, _optics(), grid)
    piston = compute_aerial_image(mask, _optics(zernike_coeffs={1: 0.25}), grid)
    assert np.allclose(piston, base, atol=1e-9)
