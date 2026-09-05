"""
The 3-D profile's anchor: dose and develop time found, not guessed.

Two contracts. The continuous arrival-time field is exactly what the binary
develop thresholds, for both finite-rate models — so anything read off it
sub-voxel is a refinement of the developed volume, never a different
answer. And the calibration lands the mid-film width on the target with a
develop time that is a stated multiple of the time to clear, which is what
turns a preset's stump into a line.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from litho_sim.core.config import GridConfig, ResistConfig, SimulationConfig
from litho_sim.develop import (
    arrival_field,
    calibrate_profile,
    develop_3d,
    develop_front,
    print_resist_3d,
    profile_cd,
)
from litho_sim.mask import lines_and_spaces


@pytest.fixture
def euv():
    cfg = SimulationConfig.from_tech_node("EUV", resist_name="EUV CAR (Positive)")
    cfg.optics = dataclasses.replace(cfg.optics, source_grid=7, normalisation="clear")
    cfg.grid = GridConfig(n_pixels=64, pixel_size=4e-9, dz=2e-9, n_z_slices=7)
    mask = lines_and_spaces(64, 4e-9, pitch=64e-9, cd=32e-9)
    return cfg, mask


@pytest.mark.parametrize("model", ["mack", "front"])
def test_arrival_field_is_what_develop_3d_thresholds(euv, model):
    cfg, mask = euv
    out = print_resist_3d(mask, cfg.optics, cfg.grid, cfg.resist, dose=1.2,
                          standing_waves=False, develop_model="threshold")
    latent = out["latent"]
    T, level, feature = arrival_field(latent, cfg.resist, cfg.grid, model=model)
    assert T.shape == latent.shape and feature == "above"
    assert level == cfg.resist.develop_time
    np.testing.assert_array_equal(
        T > level, develop_3d(latent, cfg.resist, cfg.grid, model=model)
    )
    # Further from the developer means later: arrival never decreases downward.
    assert np.all(np.diff(T, axis=0) <= 1e-9)


def test_threshold_arrival_field_is_the_latent():
    latent = np.random.default_rng(0).random((5, 8, 8))
    resist = ResistConfig(mack_Mth=0.4)
    field, level, feature = arrival_field(latent, resist, GridConfig(n_pixels=8), "threshold")
    assert field is not latent and np.array_equal(field, latent)
    assert level == 0.4 and feature == "above"
    _, _, neg = arrival_field(latent, dataclasses.replace(resist, tone="negative"),
                              GridConfig(n_pixels=8), "threshold")
    assert neg == "below"


def test_calibration_hits_the_target_at_mid_film(euv):
    cfg, mask = euv
    cal = calibrate_profile(mask, cfg.optics, cfg.grid, cfg.resist, 32.0, tol_nm=0.25)
    assert cal.converged
    assert cal.cd_nm["mid"] == pytest.approx(32.0, abs=0.3)
    assert cal.develop_time == pytest.approx(2.0 * cal.clear_time)
    assert cal.resist.develop_time == cal.develop_time
    assert 0.2 < cal.dose < 6.0
    # The spaces open to the substrate and the line stands: a line, not a stump.
    out = print_resist_3d(mask, cfg.optics, cfg.grid, cal.resist, dose=cal.dose,
                          standing_waves=False, develop_model="mack")
    remaining = out["remaining"]
    assert 0.2 < remaining.mean() < 0.8
    assert remaining[-1].any(), "the top of the line was developed away"
    assert not remaining[0].all(), "the spaces never cleared at the substrate"


def test_calibration_sizes_the_depth_it_was_asked_for(euv):
    cfg, mask = euv
    top = calibrate_profile(mask, cfg.optics, cfg.grid, cfg.resist, 32.0, depth_fraction=1.0)
    assert top.cd_nm["top"] == pytest.approx(32.0, abs=0.3)
    assert top.dose != pytest.approx(
        calibrate_profile(mask, cfg.optics, cfg.grid, cfg.resist, 32.0).dose, rel=1e-3
    )


def test_calibration_refuses_the_threshold_model(euv):
    cfg, mask = euv
    with pytest.raises(ValueError, match="finite-rate"):
        calibrate_profile(mask, cfg.optics, cfg.grid, cfg.resist, 32.0, develop_model="threshold")


def test_profile_cd_reads_sub_voxel():
    # A tent: rises 0..4 over four pixels and falls back, on every row.
    row = np.array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0], dtype=float)
    field = np.broadcast_to(row, (3, 4, 10)).copy()
    grid = GridConfig(n_pixels=10, pixel_size=4e-9)
    # "above" 2.25 runs from x = 2.25 to x = 6.75: 4.5 pixels, not 4 or 5.
    cd = profile_cd(field, 2.25, "above", grid, depth_fraction=0.5)
    assert cd == pytest.approx(4.5 * 4e-9)


def test_develop_front_no_longer_inverts_a_rate_field():
    rate = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="exposed fraction"):
        develop_front(rate, 2.0, 4.0, 1.0, tone="negative")
