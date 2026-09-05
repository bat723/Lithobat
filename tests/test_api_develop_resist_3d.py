"""
Interface-level tests for the depth-resolved (3-D) resist pipeline.

Everything here goes through the public package surface — ``from
litho_sim.develop import ...`` — the way ``cli.py``, ``app/compute.py``,
``app/pipeline.py`` and ``patterning/steps.py`` consume it. The module-level
physics (fringe periods, Dill coupling, focus mapping) is covered by
``tests/test_resist3d.py``; this file instead pins the *contract*:

* the exported names exist and compose,
* the manual chain the real consumers write (exposure_volume →
  apply_vertical_interference → apply_absorption → PEB → develop_3d)
  reproduces ``print_resist_3d`` exactly,
* an aerial-like input prints a physically sane solid — resist standing
  where the mask is dark, cleared where it is bright, tapered by absorption,
* the edges behave: bad arguments raise, a dark field leaves the film
  intact, a zero-thickness film degenerates instead of crashing, and
  ``write_resist_to_stack`` refuses a stack with nothing to develop.

``develop_model="front"`` specifics are exercised elsewhere and deliberately
not duplicated here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import litho_sim.develop as develop_pkg
from litho_sim.bake import apply_peb_3d
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.develop import (
    add_standing_waves,
    apply_absorption,
    apply_vertical_interference,
    develop_3d,
    effective_defocus,
    exposure_volume,
    focus_reference_depth,
    print_resist_3d,
    sidewall_angle,
    tmm_standing_waves,
    write_resist_to_stack,
)
from litho_sim.mask.patterns import lines_and_spaces
from litho_sim.wafer import VACUUM, Stack, get_material

# ---------------------------------------------------------------------------
# Small, fast configuration: 64 px laterally, 25 voxels of depth.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=64, pixel_size=5e-9, dz=4e-9, n_z_slices=7)


@pytest.fixture(scope="module")
def optics() -> OpticsConfig:
    return OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=15)


@pytest.fixture(scope="module")
def resist() -> ResistConfig:
    return ResistConfig(
        thickness=100e-9, n_resist=1.70, dill_A=0.8, dill_B=0.05, dill_C=0.04,
        dose_nominal=30.0, mack_Mth=0.5, diffusion_sigma=8e-9,
    )


@pytest.fixture(scope="module")
def mask(grid) -> np.ndarray:
    """Two 160 nm periods on a 320 nm field — k1 ≈ 0.39, prints comfortably."""
    return lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=160e-9, cd=80e-9)


@pytest.fixture(scope="module")
def printed(mask, optics, grid, resist) -> dict:
    """One shared end-to-end print, reused by every happy-path assertion."""
    return print_resist_3d(mask, optics, grid, resist, dose=1.0, standing_waves=False)


def _deepest_column(row_is_bright: np.ndarray, want_bright: bool) -> int:
    """Index of the pixel farthest inside its own (bright or dark) region.

    Distance is periodic: the aerial image is computed with FFTs, so light
    from a bright run at one field edge wraps around and exposes the other.
    """
    region = row_is_bright if want_bright else ~row_is_bright
    n = region.size
    idx = np.arange(n)
    others = idx[~region]
    d = np.abs(idx[:, None] - others[None, :])
    d = np.minimum(d, n - d)
    depth = d.min(axis=1)
    depth[~region] = -1
    return int(np.argmax(depth))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_all_resist3d_exports_resolve_at_package_level():
    """Every advertised name must exist on the package and be callable."""
    exported = [
        "add_standing_waves", "apply_absorption", "apply_vertical_interference",
        "develop_3d", "effective_defocus", "exposure_volume",
        "focus_reference_depth", "print_resist_3d", "sidewall_angle",
        "tmm_standing_waves", "write_resist_to_stack",
    ]
    for name in exported:
        assert name in develop_pkg.__all__, f"{name} missing from __all__"
        assert callable(getattr(develop_pkg, name)), f"{name} is not callable"


def test_focus_helpers_answer_through_the_package(optics, resist):
    """The two focus helpers are exported; the package route must agree."""
    d_ref = focus_reference_depth(resist)  # default focus_reference="mid"
    assert d_ref == pytest.approx(resist.thickness / 2)
    assert effective_defocus(d_ref, optics, resist) == pytest.approx(optics.defocus)


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def test_print_resist_3d_prints_the_mask(printed, mask, grid, resist):
    """Aerial-like input → developed solid with the mask's geometry in it."""
    for key in ("intensity", "pac", "latent", "remaining", "z"):
        assert key in printed
    remaining = printed["remaining"]
    nz = int(round(resist.thickness / grid.dz))
    assert remaining.dtype == np.bool_
    assert remaining.shape == (nz, grid.n_pixels, grid.n_pixels)
    for key in ("intensity", "pac", "latent"):
        assert printed[key].shape == remaining.shape
    z = printed["z"]
    assert z.shape == (nz,)
    assert np.all(np.diff(z) > 0), "z must be bottom-up"
    assert z[-1] == pytest.approx(resist.thickness, abs=grid.dz)

    frac = float(remaining.mean())
    assert 0.05 < frac < 0.95, f"{frac:.1%} remaining — no printed pattern"

    # Resist stands where the mask is dark, clears where it is bright.
    r = grid.n_pixels // 2
    row_bright = mask[r] > 0.5
    assert row_bright.any() and (~row_bright).any()
    x_bright = _deepest_column(row_bright, want_bright=True)
    x_dark = _deepest_column(row_bright, want_bright=False)

    dark_col = remaining[:, r, x_dark]
    bright_col = remaining[:, r, x_bright]
    assert dark_col.mean() > 0.9, "resist under the dark mask should survive"
    assert not bright_col[-1], "top of the film under a bright space should clear"
    assert bright_col.sum() < dark_col.sum(), "bright column retains as much as dark"


def test_absorption_tapers_the_profile_top_down(printed):
    """Light is consumed on the way down, so the top bleaches and clears most."""
    pac = printed["pac"]
    assert float(pac[-1].mean()) < float(pac[0].mean()), (
        "top of the film should be more bleached than the bottom"
    )
    remaining = printed["remaining"]
    nz = remaining.shape[0]
    top = int(remaining[int(0.9 * (nz - 1))].sum())
    bottom = int(remaining[int(0.1 * (nz - 1))].sum())
    assert top <= bottom, "profile is undercut — absorption direction inverted"


def test_sidewall_angle_of_the_print_is_physical(printed, grid):
    angle = sidewall_angle(printed["remaining"], grid)
    assert 45.0 < angle <= 90.0, f"sidewall angle {angle:.1f}° is not physical"


def test_manual_chain_reproduces_print_resist_3d(mask, optics, grid, resist):
    """The composition every real consumer writes must equal the wrapper.

    ``app/pipeline.py`` and ``patterning/steps.py`` both hand-assemble
    exposure_volume → apply_vertical_interference → apply_absorption(dose=1)
    → PEB → develop_3d instead of calling ``print_resist_3d``. If the
    wrapper ever drifts from that chain (an extra dose factor, a reordered
    step), the app and the patterning flows silently diverge from the CLI.
    """
    r = dataclasses.replace(resist, substrate_reflectance=0.3)
    dose = 1.2

    intensity, _ = exposure_volume(mask, optics, grid, r, dose=dose)
    intensity = apply_vertical_interference(intensity, r, optics, grid)
    pac, _ = apply_absorption(intensity, r, grid.dz, dose=1.0)
    latent = apply_peb_3d(pac, r, grid)
    remaining = develop_3d(latent, r, grid, model="threshold")

    ref = print_resist_3d(mask, optics, grid, r, dose=dose, standing_waves=True)
    np.testing.assert_allclose(latent, ref["latent"], atol=1e-12)
    assert np.array_equal(remaining, ref["remaining"]), (
        "hand-assembled pipeline and print_resist_3d developed different solids"
    )


def test_mack_develop_model_prints_end_to_end(mask, optics, grid, resist):
    """The finite-rate model must also print through the same entry point.

    The default develop_time of 5 s is a 5x over-develop calibrated for the
    just-clearing dose; at this pitch and dose 1.0 the front outruns the
    100 nm film entirely, so the test develops for 1 s to land mid-window.
    """
    r = dataclasses.replace(resist, develop_time=1.0)
    res = print_resist_3d(
        mask, optics, grid, r, dose=1.0,
        standing_waves=False, develop_model="mack",
    )
    frac = float(res["remaining"].mean())
    assert 0.05 < frac < 0.95, f"{frac:.1%} remaining under mack develop"
    angle = sidewall_angle(res["remaining"], grid)
    assert np.isfinite(angle) and 0.0 < angle <= 90.0


# ---------------------------------------------------------------------------
# Standing-wave dispatch
# ---------------------------------------------------------------------------


def test_vertical_interference_dispatches_twobeam(optics, grid, resist):
    """optical_model='twobeam' must route to add_standing_waves, unchanged."""
    r = dataclasses.replace(resist, substrate_reflectance=0.4, optical_model="twobeam")
    intensity = np.ones((25, 8, 8))
    via_dispatch = apply_vertical_interference(intensity, r, optics, grid)
    direct = add_standing_waves(intensity, r, optics, grid.dz)
    np.testing.assert_array_equal(via_dispatch, direct)
    assert not np.allclose(via_dispatch, intensity), "reflectance 0.4 must modulate"


def test_vertical_interference_dispatches_tmm(optics, grid, resist):
    """optical_model='tmm' must modulate in depth while conserving dose."""
    r = dataclasses.replace(resist, optical_model="tmm", film_stack=[])
    intensity = np.ones((25, 8, 8))
    via_dispatch = apply_vertical_interference(intensity, r, optics, grid)
    direct = tmm_standing_waves(intensity, r, optics, grid)
    np.testing.assert_allclose(via_dispatch, direct)
    assert np.isfinite(via_dispatch).all() and via_dispatch.min() >= 0.0
    assert via_dispatch.mean() == pytest.approx(1.0, rel=1e-6), "dose not conserved"
    assert via_dispatch.max() > via_dispatch.min(), (
        "bare Si at 193 nm reflects strongly — the fringe cannot be flat"
    )


def test_vertical_interference_rejects_unknown_model(resist):
    with pytest.raises(ValueError, match="optical_model"):
        dataclasses.replace(resist, optical_model="crystal-ball")


# ---------------------------------------------------------------------------
# Edges and errors
# ---------------------------------------------------------------------------


def test_uniform_dark_mask_leaves_the_film_intact(optics, grid, resist):
    """No light, no development: a dark field must not print anything."""
    dark = np.zeros((grid.n_pixels, grid.n_pixels))
    res = print_resist_3d(dark, optics, grid, resist, dose=1.0, standing_waves=False)
    frac = float(res["remaining"].mean())
    assert frac > 0.99, f"a dark field developed away {100 * (1 - frac):.1f}% of the film"


def test_print_resist_3d_rejects_unknown_develop_model(mask, optics, grid, resist):
    with pytest.raises(ValueError, match="develop model"):
        print_resist_3d(
            mask, optics, grid, resist, standing_waves=False,
            develop_model="dissolve-harder",
        )


def test_zero_thickness_degenerates_to_a_single_plane(mask, optics, grid, resist):
    """A zero-thickness film collapses to one voxel plane rather than crashing."""
    flat = dataclasses.replace(resist, thickness=0.0)
    intensity, z = exposure_volume(mask, optics, grid, flat)
    assert intensity.shape == (1, grid.n_pixels, grid.n_pixels)
    assert z.shape == (1,)
    assert np.isfinite(intensity).all() and intensity.min() >= 0.0


def test_sidewall_angle_edge_cases(grid):
    """No feature → nan; a full untouched film → a vertical 90° wall."""
    empty = np.zeros((20, 16, 16), dtype=bool)
    assert np.isnan(sidewall_angle(empty, grid))
    full = np.ones((20, 16, 16), dtype=bool)
    assert sidewall_angle(full, grid) == pytest.approx(90.0)


def test_write_resist_to_stack_requires_a_coated_film(printed, grid):
    """A stack with no resist must be refused, not silently skipped."""
    bare = Stack.blank(grid, dz=grid.dz, substrate_thickness=40e-9, headroom=120e-9)
    with pytest.raises(ValueError, match="photoresist"):
        write_resist_to_stack(bare, printed["remaining"])


def test_write_resist_to_stack_carves_the_developed_profile(printed, grid):
    """The developed solid lands in the stack: cleared voxels become vacuum."""
    remaining = printed["remaining"]
    nz_vol = remaining.shape[0]

    stack = Stack.blank(grid, dz=grid.dz, substrate_thickness=40e-9, headroom=200e-9)
    resist_id = get_material("photoresist").id
    n_sub = int(round(40e-9 / grid.dz))
    stack.mat[n_sub:n_sub + nz_vol] = resist_id  # "coat" the film

    write_resist_to_stack(stack, remaining, material="photoresist")

    film = stack.mat[n_sub:n_sub + nz_vol]
    assert np.array_equal(film == resist_id, remaining), (
        "stack film does not match the developed solid"
    )
    assert (film[~remaining] == VACUUM).all()
    assert (stack.mat[:n_sub] != VACUUM).all(), "substrate must be untouched"
    assert any("develop" in h for h in stack.history)
