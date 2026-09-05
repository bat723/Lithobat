"""
Roughness is read off the continuous develop-depth field, not the binary image.

A binary edge can only sit on a half-pixel, so any roughness measured on
it is the grid's rather than the resist's. These tests pin the field the
trials now carry and the two properties that make it the right thing to
measure: it is exactly the level set the binary image was cut from, and
the roughness read off it converges as the grid is refined, where the
binary reading swings by an order of magnitude.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from litho_sim.analysis import (
    edge_positions,
    measure_lcdu,
    measure_line_roughness,
    pool_roughness,
    trial_roughness,
)
from litho_sim.core.config import GridConfig, SimulationConfig
from litho_sim.develop import cleared_depth, stochastic_trials
from litho_sim.expose import compute_aerial_image
from litho_sim.mask import lines_and_spaces


def _euv_case(pixel_size: float, n_pixels: int, trials: int, seed: int = 5,
              dissolution: float | None = None):
    cfg = SimulationConfig.from_tech_node("EUV", resist_name="EUV CAR (Positive)")
    cfg.optics = dataclasses.replace(cfg.optics, normalisation="clear", source_grid=7)
    if dissolution is not None:
        cfg.resist = dataclasses.replace(cfg.resist, dissolution_sigma=dissolution)
    grid = GridConfig(n_pixels=n_pixels, pixel_size=pixel_size)
    mask = lines_and_spaces(n_pixels, pixel_size, pitch=64e-9, cd=32e-9)
    aerial = compute_aerial_image(mask, cfg.optics, grid, dose=0.7)
    res = stochastic_trials(aerial, cfg.resist, grid, cfg.optics.wavelength,
                            trials=trials, seed=seed)
    return cfg, grid, res


def test_depth_is_the_level_set_the_binary_image_was_cut_from():
    # Dissolution noise off, so the depth is the pure function of the latent.
    cfg, grid, res = _euv_case(4e-9, 48, trials=2, dissolution=0.0)
    assert res.depth.shape == res.resist.shape
    assert res.level == pytest.approx(cfg.resist.thickness * 1e9)
    assert res.feature == "below"
    np.testing.assert_array_equal(res.depth < res.level, res.resist > 0.5)
    # ...and it is the same field cleared_depth computes from the latent.
    np.testing.assert_allclose(res.depth[0], cleared_depth(res.protected, cfg.resist))


def test_roughness_on_the_depth_field_converges_where_the_binary_one_cannot():
    """Same physics on a 2 nm and a 1 nm grid — both resolve the preset's
    6 nm acid diffusion length — and the depth reading agrees. A 4 nm grid
    does not resolve it and reads high, but its *binary* reading hides the
    roughness under the half-pixel altogether."""
    _, g2, r2 = _euv_case(2e-9, 96, trials=6)
    _, g1, r1 = _euv_case(1e-9, 192, trials=6)
    cont2 = trial_roughness(r2, g2.pixel_size)
    cont1 = trial_roughness(r1, g1.pixel_size)
    assert np.isfinite(cont2.ler) and np.isfinite(cont1.ler)
    assert cont2.ler == pytest.approx(cont1.ler, rel=0.3)
    _, g4, r4 = _euv_case(4e-9, 48, trials=6)
    cont4 = trial_roughness(r4, g4.pixel_size)
    binary4 = pool_roughness([measure_line_roughness(t, g4.pixel_size) for t in r4.resist])
    # The mechanism: a binary edge can only sit on a half-pixel, so its
    # positions are quantised exactly, and quantisation whitens the series
    # — the correlation length collapses toward the pixel.
    left, right = edge_positions(r4.resist[0], g4.pixel_size)
    frac = np.mod(np.concatenate([left, right]) / g4.pixel_size, 1.0)
    np.testing.assert_allclose(frac[np.isfinite(frac)], 0.5, atol=1e-9)
    assert binary4.corr_length < cont4.corr_length


def test_correlation_length_lands_on_the_acid_diffusion_length():
    cfg, grid, res = _euv_case(2e-9, 96, trials=6)
    xi = trial_roughness(res, grid.pixel_size).corr_length
    diffusion = np.sqrt(2.0 * cfg.resist.D_acid * cfg.resist.bake_time)
    assert 0.5 * diffusion < xi < 1.6 * diffusion, (xi * 1e9, diffusion * 1e9)


def test_lcdu_on_depth_is_sub_pixel():
    _, grid, res = _euv_case(4e-9, 48, trials=4)
    quantised = measure_lcdu(res.resist, grid.pixel_size)
    fine = measure_lcdu(res.depth, grid.pixel_size, threshold=res.level, feature="below")
    assert fine.n_failed == quantised.n_failed == 0
    # Binary CDs are whole pixels; the depth CDs are not.
    assert np.allclose(quantised.cds / grid.pixel_size, np.round(quantised.cds / grid.pixel_size))
    assert not np.allclose(fine.cds / grid.pixel_size, np.round(fine.cds / grid.pixel_size))
    assert abs(fine.cd_mean - quantised.cd_mean) < grid.pixel_size
