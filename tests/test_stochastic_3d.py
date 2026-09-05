"""
Stochastic resist profiles: the chemistry through the film, with the counts.

Pinned here: the 3-D reaction–diffusion bake reduces to the 3-D Gaussian
when its chemistry is switched off (so the ``"car"`` bake is a strict
generalisation of the path every profile used before); a batch of 3-D
trials carries the arrival-time field its solids are the level set of; the
profile statistics are finite at every height and the batch mean tracks
the deterministic profile; and the calibration works through the CAR bake.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from litho_sim.analysis import profile_roughness
from litho_sim.bake import apply_peb_3d, bake_reaction_diffusion
from litho_sim.core.config import GridConfig, SimulationConfig
from litho_sim.develop import (
    arrival_field,
    calibrate_profile,
    latent_volume,
    print_resist_3d,
    stochastic_trials_3d,
)
from litho_sim.expose import generate_acid_3d, sample_species_3d
from litho_sim.mask import lines_and_spaces


@pytest.fixture
def euv():
    cfg = SimulationConfig.from_tech_node("EUV", resist_name="EUV CAR (Positive)")
    cfg.optics = dataclasses.replace(cfg.optics, source_grid=7, normalisation="clear")
    cfg.grid = GridConfig(n_pixels=48, pixel_size=4e-9, dz=4e-9, n_z_slices=5)
    mask = lines_and_spaces(48, 4e-9, pitch=64e-9, cd=32e-9)
    return cfg, mask


def test_inert_car_bake_in_3d_is_the_gaussian_bake(euv):
    """No quencher, no deprotection, no loss: the acid just diffuses, and
    diffusing ``1 − PAC`` for the same length is diffusing PAC."""
    cfg, mask = euv
    out = print_resist_3d(mask, cfg.optics, cfg.grid, cfg.resist, dose=1.0,
                          standing_waves=False, develop_model="threshold")
    pac = out["pac"]
    g = cfg.grid
    # Match the Gaussian's diffusion length exactly: σ² = 2 D t.
    sigma = cfg.resist.diffusion_sigma
    inert = dataclasses.replace(cfg.resist, quencher_ratio=0.0, k_amp=0.0, k_loss=0.0,
                                D_acid=sigma ** 2 / (2.0 * cfg.resist.bake_time),
                                electron_blur_sigma=0.0)
    acid = 1.0 - pac
    baked = bake_reaction_diffusion(
        acid, g.pixel_size, inert.bake_time, inert.D_acid,
        spacing=(g.dz, g.pixel_size, g.pixel_size),
    )["acid"]
    gaussian = 1.0 - apply_peb_3d(pac, inert, g)
    # The interior: the two boundary treatments differ within a diffusion
    # length of the faces (the Gaussian reflects and wraps, the PDE is
    # zero-flux on every face), the physics does not.
    inner = (slice(3, -3), slice(3, -3), slice(3, -3))
    np.testing.assert_allclose(baked[inner], gaussian[inner], atol=0.03)


def test_species_3d_converges_to_the_mean_field(euv):
    cfg, mask = euv
    out = print_resist_3d(mask, cfg.optics, cfg.grid, cfg.resist, dose=1.0,
                          standing_waves=False, develop_model="threshold")
    g = cfg.grid
    exposure = latent_volume(out["intensity"], cfg.resist, g)["exposure"]
    heavy = dataclasses.replace(cfg.resist, pag_density=cfg.resist.pag_density * 400)
    rng = np.random.default_rng(1)
    draws = np.mean([
        sample_species_3d(exposure, heavy, cfg.optics.wavelength, g.pixel_size, g.dz, rng).acid
        for _ in range(32)
    ], axis=0)
    mean = generate_acid_3d(exposure, heavy, g.pixel_size, g.dz)
    # Photon noise is not suppressed by the PAG loading — 1.8 absorbed
    # photons per voxel is the EUV regime — so the comparison is plane by
    # plane, where it averages out. Shot noise depresses the mean slightly
    # (conversion is concave), so a small negative bias is physics; anything
    # larger is a bug.
    per_plane = np.abs(draws.mean(axis=(1, 2)) - mean.mean(axis=(1, 2)))
    assert per_plane.max() < 0.01
    assert np.abs(draws - mean).max() < 0.1
    assert float((draws - mean).mean()) <= 0.0


def test_profile_trials_carry_their_level_set(euv):
    cfg, mask = euv
    res = stochastic_trials_3d(mask, cfg.optics, cfg.grid, cfg.resist, dose=1.0,
                               trials=3, seed=2)
    nz = int(round(cfg.resist.thickness / cfg.grid.dz))
    assert res.arrival.shape == (3, nz, 48, 48)
    assert res.level == cfg.resist.develop_time and res.feature == "above"
    np.testing.assert_array_equal(res.remaining, res.arrival > res.level)
    assert res.z.shape == (nz,)
    assert res.sample.acid.shape == (nz, 48, 48)
    assert 0.0 < res.sample.mean_pag < 100.0, "per-voxel counts, not per-column"
    # Reproducible, and different trials differ.
    again = stochastic_trials_3d(mask, cfg.optics, cfg.grid, cfg.resist, dose=1.0,
                                 trials=3, seed=2)
    np.testing.assert_array_equal(res.arrival, again.arrival)
    assert not np.array_equal(res.arrival[0], res.arrival[1])


def test_profile_roughness_is_finite_through_the_film_and_tracks_the_mean(euv):
    cfg, mask = euv
    cal = calibrate_profile(mask, cfg.optics, cfg.grid, cfg.resist, 32.0, bake="car")
    assert cal.converged and cal.cd_nm["mid"] == pytest.approx(32.0, abs=0.3)
    det = print_resist_3d(mask, cfg.optics, cfg.grid, cal.resist, dose=cal.dose,
                          standing_waves=False, develop_model="mack", bake="car")
    ref, _, _ = arrival_field(det["latent"], cal.resist, cfg.grid, "mack")
    res = stochastic_trials_3d(mask, cfg.optics, cfg.grid, cal.resist, dose=cal.dose,
                               trials=8, seed=3)
    pr = profile_roughness(res, cfg.grid, reference=ref)
    assert pr.z_nm.shape == pr.lwr.shape == pr.cd.shape
    assert np.all(np.isfinite(pr.lwr)) and np.all(pr.lwr > 0.0)
    assert np.all(np.isfinite(pr.cd)) and np.all(pr.cd > 0.0)
    # Batch mean at mid-film sits on the deterministic width.
    mid = pr.z_nm.size // 2
    assert pr.cd[mid] * 1e9 == pytest.approx(cal.cd_nm["mid"], abs=3.0)
    assert pr.lcdu_mid.n_failed == 0
    assert pr.fails_base is not None and pr.fails_base.n_trials == 8
    # Residue and the odd bridge at the base are outcomes, not errors; what
    # must hold is that they are counted as what they are and stay rare.
    assert pr.bridges_base <= 2 and 0.0 <= pr.residue_base < 0.2
    s = pr.summary()
    assert set(s) >= {"lwr_3s_bottom_nm", "lwr_3s_mid_nm", "lwr_3s_top_nm", "top_loss_mean_nm"}
    assert np.isfinite(s["lwr_3s_mid_nm"])


def test_car_bake_changes_the_profile(euv):
    cfg, mask = euv
    gauss = print_resist_3d(mask, cfg.optics, cfg.grid, cfg.resist, dose=1.0,
                            standing_waves=False, develop_model="threshold")
    car = print_resist_3d(mask, cfg.optics, cfg.grid, cfg.resist, dose=1.0,
                          standing_waves=False, develop_model="threshold", bake="car")
    assert not np.array_equal(gauss["latent"], car["latent"])
    with pytest.raises(ValueError, match="bake must be"):
        print_resist_3d(mask, cfg.optics, cfg.grid, cfg.resist, bake="microwave")
    with pytest.raises(ValueError, match="finite-rate"):
        stochastic_trials_3d(mask, cfg.optics, cfg.grid, cfg.resist, develop_model="threshold")
