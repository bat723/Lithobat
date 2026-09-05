"""
Interface-level tests for the developed-profile metrics.

``tests/test_profile_metrics.py`` pins the arithmetic on synthetic volumes
where the answer is known by construction. These tests come at the same
functions from the other side: through the public package surface
(``litho_sim.develop``), on volumes produced by the real develop pipeline
(``dill_exposure`` → ``develop_3d``), the way ``app/compute.py`` and
``viz/viz3d.py`` actually use them. What is checked is not the arithmetic but
the physics-facing contract: a printed line/space profile is neither sealed
nor cleared, an unexposed film is sealed with nothing lost from the top, a
flood-exposed film clears and has no top left to have lost — and the
predicates can never disagree with the fractions and named thresholds they
are defined in terms of.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from litho_sim.core.config import GridConfig, ResistConfig
from litho_sim.develop import (
    CLEARED_FRACTION,
    SEALED_FRACTION,
    develop_3d,
    dill_exposure,
    film_remaining,
    is_cleared,
    is_sealed,
    top_loss,
)

GRID = GridConfig(n_pixels=48, pixel_size=4e-9, dz=2e-9)
RESIST = ResistConfig()  # positive tone, 100 nm film, Mth = 0.5
NZ = int(round(RESIST.thickness / GRID.dz))
FILM_HEIGHT_NM = NZ * GRID.dz * 1e9
PITCH = 64e-9  # three line/space periods across the 192 nm field


def _intensity_volume(aerial_1d: np.ndarray) -> np.ndarray:
    """Extrude a 1-D aerial profile into the (nz, ny, nx) exposure volume."""
    return np.broadcast_to(
        aerial_1d[None, None, :], (NZ, GRID.n_pixels, GRID.n_pixels)
    ).copy()


def _develop(intensity: np.ndarray, **cfg_overrides) -> np.ndarray:
    """The public path a real caller takes: expose, then develop in 3-D."""
    cfg = dataclasses.replace(RESIST, **cfg_overrides) if cfg_overrides else RESIST
    pac = dill_exposure(intensity, dose=cfg.dose_nominal, dill_C=cfg.dill_C)
    return develop_3d(pac, cfg, GRID, model="threshold")


@pytest.fixture(scope="module")
def line_space_profile() -> np.ndarray:
    """A printed line/space pattern at nominal dose — the working regime."""
    x = np.arange(GRID.n_pixels) * GRID.pixel_size
    aerial = 0.5 * (1.0 - np.cos(2 * np.pi * x / PITCH))
    return _develop(_intensity_volume(aerial))


@pytest.fixture(scope="module")
def unexposed_profile() -> np.ndarray:
    """No light at all — the film must not develop."""
    return _develop(_intensity_volume(np.zeros(GRID.n_pixels)))


@pytest.fixture(scope="module")
def flood_exposed_profile() -> np.ndarray:
    """Uniform full exposure — the film must clear everywhere."""
    return _develop(_intensity_volume(np.ones(GRID.n_pixels)))


# ---------------------------------------------------------------------------
# The happy path: a profile the pipeline actually prints
# ---------------------------------------------------------------------------


def test_a_printed_pattern_is_neither_sealed_nor_cleared(line_space_profile):
    """The working regime is the one where neither diagnosis fires."""
    assert not is_sealed(line_space_profile)
    assert not is_cleared(line_space_profile)


def test_a_printed_pattern_keeps_a_sane_fraction_of_the_film(line_space_profile):
    """Half the field is line, half space; the surviving fraction must sit
    well inside the open interval the two thresholds bound."""
    frac = film_remaining(line_space_profile)
    assert CLEARED_FRACTION < frac < SEALED_FRACTION
    assert 0.2 < frac < 0.8  # equal lines and spaces, so nowhere near either edge


def test_a_printed_pattern_has_a_finite_top_loss_inside_the_film(line_space_profile):
    loss = top_loss(line_space_profile, GRID)
    assert np.isfinite(loss)
    assert 0.0 <= loss < FILM_HEIGHT_NM


def test_the_predicates_cannot_disagree_with_the_fraction(
    line_space_profile, unexposed_profile, flood_exposed_profile
):
    """is_sealed / is_cleared are *defined* as comparisons of film_remaining
    against the named constants; on real pipeline output the three exports
    must tell one consistent story."""
    for volume in (line_space_profile, unexposed_profile, flood_exposed_profile):
        frac = film_remaining(volume)
        assert is_sealed(volume) == (frac > SEALED_FRACTION)
        assert is_cleared(volume) == (frac < CLEARED_FRACTION)
        assert not (is_sealed(volume) and is_cleared(volume))


# ---------------------------------------------------------------------------
# The degenerate regimes, reached the way a user reaches them
# ---------------------------------------------------------------------------


def test_an_unexposed_film_is_sealed_and_intact(unexposed_profile):
    """Dark field, positive resist: nothing dissolves. The metrics must call
    it sealed, essentially all present, with nothing lost from the top."""
    assert is_sealed(unexposed_profile)
    assert not is_cleared(unexposed_profile)
    assert film_remaining(unexposed_profile) == pytest.approx(1.0)
    assert top_loss(unexposed_profile, GRID) == pytest.approx(0.0)


def test_a_flood_exposed_film_clears(flood_exposed_profile):
    """Uniform full dose: everything dissolves, and a film with no surviving
    voxel has no top to have lost — top loss is NaN, not a number."""
    assert is_cleared(flood_exposed_profile)
    assert not is_sealed(flood_exposed_profile)
    assert film_remaining(flood_exposed_profile) == pytest.approx(0.0)
    assert np.isnan(top_loss(flood_exposed_profile, GRID))


def test_an_overdeveloped_mack_profile_shows_real_top_loss():
    """Top loss is the metric's reason to exist: a finite-rate develop of a
    depth-graded exposure (film top sees the most light, plus flare) thins
    the film from above. The reported loss must be strictly positive yet
    smaller than the film, on a profile that still printed."""
    x = np.arange(GRID.n_pixels) * GRID.pixel_size
    aerial = 0.5 * (1.0 - np.cos(2 * np.pi * x / PITCH))
    depth_gain = np.linspace(0.6, 1.0, NZ)  # iz = 0 is the film bottom
    intensity = depth_gain[:, None, None] * (aerial[None, None, :] + 0.18)

    cfg = dataclasses.replace(RESIST, develop_time=10.0)
    pac = dill_exposure(intensity, dose=cfg.dose_nominal, dill_C=cfg.dill_C)
    remaining = develop_3d(pac, cfg, GRID, model="mack")

    assert not is_sealed(remaining)
    assert not is_cleared(remaining)
    frac = film_remaining(remaining)
    assert CLEARED_FRACTION < frac < SEALED_FRACTION
    loss = top_loss(remaining, GRID)
    assert 0.0 < loss < FILM_HEIGHT_NM


# ---------------------------------------------------------------------------
# Edge of the contract: a volume with no voxels at all
# ---------------------------------------------------------------------------


def test_a_zero_thickness_volume_yields_nan_not_a_crash():
    """A (0, ny, nx) volume has no film to measure. top_loss answers NaN by
    the same rule as a cleared film; film_remaining propagates numpy's
    empty-mean NaN (with the stock RuntimeWarning), and the predicates then
    decline both diagnoses — NaN compares False either way."""
    empty = np.zeros((0, 8, 8), dtype=bool)

    assert np.isnan(top_loss(empty, GRID))

    with pytest.warns(RuntimeWarning):
        assert np.isnan(film_remaining(empty))
    with pytest.warns(RuntimeWarning):
        assert is_sealed(empty) is False
    with pytest.warns(RuntimeWarning):
        assert is_cleared(empty) is False


def test_the_package_and_the_module_export_the_same_objects():
    """app/compute.py imports from the package, viz3d.py from the module;
    they must be the same functions, not diverging copies."""
    from litho_sim.develop import profile as profile_module

    assert film_remaining is profile_module.film_remaining
    assert top_loss is profile_module.top_loss
    assert is_sealed is profile_module.is_sealed
    assert is_cleared is profile_module.is_cleared
    assert SEALED_FRACTION == profile_module.SEALED_FRACTION
    assert CLEARED_FRACTION == profile_module.CLEARED_FRACTION
