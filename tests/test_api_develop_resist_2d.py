"""
Interface-level tests for the 2-D develop pipeline.

Everything here goes through the public package surface —
``from litho_sim.develop import ...`` — never the ``resist`` module, so a
later internal split cannot break a caller without breaking these first.

The unit level (shapes, monotonicity of each physics step, exact pixel
arithmetic) is ``tests/test_resist.py``'s job. These tests instead run the
three models end-to-end on a *synthetic* aerial-like input whose designed
geometry is known analytically — a raised-cosine line/space profile — and
check the physics comes out the right way round: resist stays where the
image is dark (positive tone), clears where it is bright, and the measured
CD lands near the designed line.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, SimulationConfig
from litho_sim.develop import (
    feature_edges,
    just_clearing_time,
    measure_cd_1d,
    measure_cd_2d,
    remaining_thickness,
    simulate_resist,
)

N = 64                  # 64 px ...
PX = 4e-9               # ... of 4 nm -> 256 nm field
PITCH = 128e-9          # two full periods across the field
MODELS = ("threshold", "mack", "car")


@pytest.fixture(scope="module")
def cfg() -> SimulationConfig:
    c = SimulationConfig.from_tech_node("ArF")
    c.grid = GridConfig(n_pixels=N, pixel_size=PX)
    return c


@pytest.fixture(scope="module")
def aerial() -> np.ndarray:
    """Smooth line/space intensity in [0, 1], darkest at the field centre.

    ``I(x) = 0.5 − 0.5·cos(2πx/pitch)`` with x measured from the centre:
    a dark line sits mid-field, so the "centre feature" the CD helpers
    select is the designed one. At threshold 0.5 the dark region is exactly
    ``pitch/2`` wide — the analytic CD the tests measure against.
    """
    x = (np.arange(N) - N // 2) * PX
    row = 0.5 - 0.5 * np.cos(2.0 * np.pi * x / PITCH)
    return np.tile(row, (N, 1))


# ---------------------------------------------------------------------------
# Happy path: aerial-like input -> simulate_resist -> physically sane output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_positive_tone_prints_the_dark_line(aerial, cfg, model):
    """All three models keep resist where dark and clear where bright."""
    pac_exp, pac_peb, resist = simulate_resist(
        aerial, cfg.resist, cfg.grid, model=model
    )
    for arr in (pac_exp, pac_peb, resist):
        assert arr.shape == aerial.shape
        assert np.isfinite(arr).all()
    assert set(np.unique(resist)) <= {0.0, 1.0}

    mid = N // 2
    row, cut = resist[mid, :], aerial[mid, :]
    assert row[int(np.argmin(cut))] == 1.0, f"{model}: dark centre developed out"
    assert row[int(np.argmax(cut))] == 0.0, f"{model}: bright space did not clear"

    # The printed line must be a line, not a sliver or a slab.
    cd = measure_cd_2d(resist, cfg.grid.pixel_size, threshold=0.5, feature="above")
    assert 0.15 * PITCH < cd < 0.85 * PITCH, f"{model}: CD {cd*1e9:.1f} nm"


@pytest.mark.parametrize("model", ("mack", "car"))
def test_chemistry_fields_read_unexposed_as_one(aerial, cfg, model):
    """pac_exp/pac_peb: 1 means unexposed in every model, dark above bright."""
    pac_exp, pac_peb, _ = simulate_resist(aerial, cfg.resist, cfg.grid, model=model)
    mid = N // 2
    dark = int(np.argmin(aerial[mid]))
    bright = int(np.argmax(aerial[mid]))
    for arr in (pac_exp, pac_peb):
        assert -1e-9 <= float(arr.min()) and float(arr.max()) <= 1.0 + 1e-9
        assert arr[mid, dark] > arr[mid, bright]


def test_threshold_model_cd_matches_the_designed_line(aerial, cfg):
    """At threshold 0.5 the cosine's dark line is exactly pitch/2 wide."""
    rcfg = dataclasses.replace(cfg.resist, threshold=0.5)
    _, _, resist = simulate_resist(aerial, rcfg, cfg.grid, model="threshold")
    cd = measure_cd_2d(resist, cfg.grid.pixel_size, threshold=0.5, feature="above")
    assert cd == pytest.approx(PITCH / 2.0, abs=2.0 * PX)


def test_negative_tone_keeps_the_bright_region(aerial, cfg):
    neg = dataclasses.replace(cfg.resist, tone="negative")
    _, _, resist = simulate_resist(aerial, neg, cfg.grid, model="mack")
    mid = N // 2
    assert resist[mid, int(np.argmax(aerial[mid]))] == 1.0
    assert resist[mid, int(np.argmin(aerial[mid]))] == 0.0


# ---------------------------------------------------------------------------
# CD helpers on the continuous profile
# ---------------------------------------------------------------------------


def test_cd_is_subpixel_on_the_continuous_profile(aerial, cfg):
    """Measured on the smooth cutline, CD reproduces the analytic width."""
    cut = aerial[N // 2, :]
    edges = feature_edges(cut, threshold=0.5, feature="below")
    assert edges is not None
    rise, fall = edges
    assert fall > rise
    cd = measure_cd_1d(cut, PX, threshold=0.5, feature="below")
    assert cd == pytest.approx(PITCH / 2.0, abs=0.05e-9)


def test_remaining_thickness_tracks_the_image(aerial, cfg):
    """Dark keeps the film, bright loses it, and nothing leaves [0, film]."""
    t_nm = remaining_thickness(aerial, cfg.resist)
    film = cfg.resist.thickness * 1e9
    assert t_nm.shape == aerial.shape
    assert float(t_nm.min()) >= 0.0
    assert float(t_nm.max()) <= film + 1e-9

    mid = N // 2
    dark = int(np.argmin(aerial[mid]))
    bright = int(np.argmax(aerial[mid]))
    assert t_nm[mid, dark] > t_nm[mid, bright]
    assert t_nm[mid, dark] == pytest.approx(film, rel=0.05)
    assert t_nm[mid, bright] < 0.5 * film

    # develop_time is a sane multiple of the just-clearing time — the unit
    # the config documents it in.
    jct = just_clearing_time(cfg.resist)
    assert jct > 0.0
    assert 1.0 < cfg.resist.develop_time / jct < 50.0


# ---------------------------------------------------------------------------
# Edge and error cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_uniform_dark_field_leaves_the_film_intact(cfg, model):
    """A non-printing (all-dark) exposure must not develop anything out."""
    dark = np.zeros((N, N))
    _, _, resist = simulate_resist(dark, cfg.resist, cfg.grid, model=model)
    assert float(resist.mean()) == 1.0, f"{model} developed an unexposed film"


@pytest.mark.parametrize("model", MODELS)
def test_uniform_bright_field_clears_the_film(cfg, model):
    """A flood exposure clears everything, and CD reports the absence."""
    bright = np.ones((N, N))
    _, _, resist = simulate_resist(bright, cfg.resist, cfg.grid, model=model)
    assert float(resist.mean()) == 0.0, f"{model} left resist under flood exposure"
    assert measure_cd_2d(resist, PX) == 0.0


def test_zero_thickness_film_clears_trivially(aerial, cfg):
    flat = dataclasses.replace(cfg.resist, thickness=0.0)
    _, _, resist = simulate_resist(aerial, flat, cfg.grid, model="mack")
    assert float(resist.mean()) == 0.0
    assert just_clearing_time(flat) == 0.0
    assert np.all(remaining_thickness(aerial, flat) == 0.0)


def test_unknown_model_raises_through_the_package(aerial, cfg):
    with pytest.raises(ValueError, match="Unknown resist model"):
        simulate_resist(aerial, cfg.resist, cfg.grid, model="dissolve")


def test_bad_tone_is_rejected(aerial, cfg):
    bad = dataclasses.replace(cfg.resist, tone="sideways")
    with pytest.raises(ValueError, match="tone"):
        simulate_resist(aerial, bad, cfg.grid, model="threshold")
