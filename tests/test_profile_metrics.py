"""
Tests for the developed-profile metrics.

These are small functions, but they were previously unreachable except by
building a matplotlib figure — which is why the top-loss formula had already
been copy-pasted into the Streamlit app rather than imported. The tests below
pin the arithmetic on synthetic volumes where the answer is known by
construction, so the extraction cannot drift from the original.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.core.config import GridConfig
from litho_sim.develop.profile import (
    CLEARED_FRACTION,
    SEALED_FRACTION,
    film_remaining,
    is_cleared,
    is_sealed,
    top_loss,
)

GRID = GridConfig(n_pixels=16, pixel_size=4e-9, dz=2e-9)


def _volume(n_z: int = 20, top_full: int = 20) -> np.ndarray:
    """A film of *n_z* voxels, solid up to (excluding) *top_full*."""
    v = np.zeros((n_z, 16, 16), dtype=bool)
    v[:top_full, :, :] = True
    return v


# ---------------------------------------------------------------------------
# top loss
# ---------------------------------------------------------------------------


def test_a_full_film_has_lost_nothing():
    assert top_loss(_volume(20, 20), GRID) == pytest.approx(0.0)


@pytest.mark.parametrize("missing", [1, 3, 7])
def test_top_loss_counts_the_voxels_missing_from_the_top(missing):
    """Known by construction: remove *missing* layers, expect missing × dz."""
    v = _volume(20, 20 - missing)
    assert top_loss(v, GRID) == pytest.approx(missing * GRID.dz * 1e9)


def test_top_loss_is_nan_when_nothing_survived():
    """A fully cleared film has no top to have lost, and reporting the full
    thickness would read as a measurement rather than an absence."""
    assert np.isnan(top_loss(np.zeros((20, 16, 16), dtype=bool), GRID))


def test_top_loss_uses_the_highest_surviving_voxel_anywhere():
    """It is the film's top that matters, not any particular column — a
    single surviving pillar means the film has not lost height."""
    v = np.zeros((20, 16, 16), dtype=bool)
    v[:20, 0, 0] = True          # one full-height pillar
    v[:5, :, :] = True           # everything else heavily eroded
    assert top_loss(v, GRID) == pytest.approx(0.0)


def test_top_loss_scales_with_dz():
    v = _volume(20, 16)          # 4 voxels missing
    fine = GridConfig(n_pixels=16, pixel_size=4e-9, dz=1e-9)
    coarse = GridConfig(n_pixels=16, pixel_size=4e-9, dz=8e-9)
    assert top_loss(v, fine) == pytest.approx(4.0)
    assert top_loss(v, coarse) == pytest.approx(32.0)


def test_it_matches_the_inline_formula_it_replaced():
    """Byte-for-byte agreement with the expression lifted out of viz3d.

    The point of extracting it was to stop it being copied; this is the check
    that the copy and the original actually agreed in the first place.
    """
    rng = np.random.default_rng(0)
    for _ in range(8):
        v = rng.random((20, 16, 16)) > 0.5
        nz = v.shape[0]
        occupied = np.nonzero(v.any(axis=(1, 2)))[0]
        expected = (
            (nz - 1 - int(occupied.max())) * GRID.dz * 1e9
            if occupied.size else float("nan")
        )
        got = top_loss(v, GRID)
        assert (np.isnan(got) and np.isnan(expected)) or got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# film remaining, and the degenerate outcomes
# ---------------------------------------------------------------------------


def test_film_remaining_is_the_occupied_fraction():
    assert film_remaining(_volume(20, 20)) == pytest.approx(1.0)
    assert film_remaining(_volume(20, 10)) == pytest.approx(0.5)
    assert film_remaining(np.zeros((20, 16, 16), dtype=bool)) == pytest.approx(0.0)


def test_sealed_and_cleared_are_mutually_exclusive_and_usually_both_false():
    """The interesting case is the one where neither fires."""
    full = _volume(20, 20)
    empty = np.zeros((20, 16, 16), dtype=bool)
    half = _volume(20, 10)

    assert is_sealed(full) and not is_cleared(full)
    assert is_cleared(empty) and not is_sealed(empty)
    assert not is_sealed(half) and not is_cleared(half)


def test_the_thresholds_are_named_not_scattered():
    """They were bare literals in two places before; a diagnosis and the
    predicate that triggers it must not be able to disagree."""
    assert 0.9 < SEALED_FRACTION < 1.0
    assert 0.0 < CLEARED_FRACTION < 0.1


def test_metrics_reach_the_public_package():
    """The whole point is that these are importable without a figure."""
    from litho_sim.develop import film_remaining as fr
    from litho_sim.develop import sidewall_angle
    from litho_sim.develop import top_loss as tl

    assert callable(fr) and callable(tl) and callable(sidewall_angle)


# ---------------------------------------------------------------------------
# The 2-D profile — the field that used to be computed and discarded
# ---------------------------------------------------------------------------


def test_remaining_thickness_is_continuous_not_binary():
    """The whole point: a footprint has 2 levels, a profile has many."""
    from litho_sim.core.config import ResistConfig
    from litho_sim.develop import remaining_thickness

    x = np.linspace(0, 1, 256)
    latent = (0.5 * (1 + np.cos(2 * np.pi * 2 * x)))[None, :].repeat(8, 0)
    cfg = ResistConfig()

    h = remaining_thickness(latent, cfg)
    assert len(np.unique(np.round(h, 3))) > 10, (
        "remaining thickness came back quantised — it is meant to be the "
        "continuous field the binary path throws away"
    )


def test_remaining_thickness_stays_inside_the_film():
    from litho_sim.core.config import ResistConfig
    from litho_sim.develop import remaining_thickness

    rng = np.random.default_rng(0)
    latent = rng.random((16, 16))
    cfg = ResistConfig()
    h = remaining_thickness(latent, cfg)
    assert h.min() >= 0.0
    assert h.max() <= cfg.thickness * 1e9 + 1e-9


def test_more_develop_time_never_adds_resist():
    from dataclasses import replace

    from litho_sim.core.config import ResistConfig
    from litho_sim.develop import remaining_thickness

    x = np.linspace(0, 1, 128)
    latent = (0.5 * (1 + np.cos(2 * np.pi * 2 * x)))[None, :].repeat(4, 0)
    cfg = ResistConfig()
    profiles = [
        remaining_thickness(latent, replace(cfg, develop_time=t))
        for t in (1.0, 2.0, 5.0, 20.0)
    ]
    for earlier, later in zip(profiles, profiles[1:]):
        assert np.all(later <= earlier + 1e-9)


def test_just_clearing_time_is_film_over_rate():

    from litho_sim.core.config import ResistConfig
    from litho_sim.develop import just_clearing_time

    cfg = ResistConfig()
    assert just_clearing_time(cfg) == pytest.approx(
        cfg.thickness * 1e9 / cfg.mack_Rmax
    )


def test_the_default_develop_time_is_a_sane_over_develop():
    """Pins the recalibration.

    A raw number of seconds says nothing; what matters is the multiple of the
    just-clearing time. The default was 30x, which shrank a 100 nm line to
    ~50 nm and took 16.5 nm off the top. Anything in the low single digits is
    a normal process; this guards against drifting back.
    """
    from litho_sim.core.config import ResistConfig
    from litho_sim.develop import just_clearing_time

    cfg = ResistConfig()
    over = cfg.develop_time / just_clearing_time(cfg)
    assert 1.5 <= over <= 10.0, f"default develop_time is a {over:.0f}x over-develop"
