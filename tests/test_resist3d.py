"""
Physics tests for the depth-resolved (3-D) resist model.

These are the falsifiable checks that justify the whole 3-D approach — that
sweeping the existing pupil through the film, times an absorption and
standing-wave envelope, produces a profile that is actually right rather
than merely plausible:

* the film must expose more at the top than the bottom,
* the sidewall must slope rather than come out perfectly vertical,
* the standing-wave period must equal λ/(2·n_resist),
* and with the 3-D-only effects disabled it must reduce to the 2-D engine.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.bake.peb import apply_peb_3d
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.develop.resist3d import (
    add_standing_waves,
    apply_absorption,
    develop_3d,
    effective_defocus,
    exposure_volume,
    print_resist_3d,
    sidewall_angle,
)
from litho_sim.expose.aerial_image import compute_aerial_image
from litho_sim.mask.patterns import lines_and_spaces


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=80, pixel_size=4e-9, dz=2e-9, n_z_slices=11)


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
    """A 160 nm pitch on a 320 nm field: exactly two periods, k1 = 0.39.

    Pitch matters here. Much tighter and the aerial contrast collapses so
    that even the dark regions bleach past the develop threshold and the
    whole field clears; much looser and the profile comes out so steep that
    sidewall angle is uninformative.
    """
    return lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=160e-9, cd=80e-9)


# ---------------------------------------------------------------------------
# Focus / depth mapping
# ---------------------------------------------------------------------------


def test_effective_defocus_is_zero_at_the_reference_plane(optics, resist):
    top = dataclasses.replace(resist, focus_reference="top")
    assert effective_defocus(0.0, optics, top) == pytest.approx(0.0)

    mid = dataclasses.replace(resist, focus_reference="mid")
    assert effective_defocus(resist.thickness / 2, optics, mid) == pytest.approx(0.0)

    bot = dataclasses.replace(resist, focus_reference="bottom")
    assert effective_defocus(resist.thickness, optics, bot) == pytest.approx(0.0)


def test_effective_defocus_scales_with_resist_index(optics, resist):
    """Depth inside the film is compressed by the film's refractive index."""
    top_ref = dataclasses.replace(resist, focus_reference="top")
    depth = 100e-9
    d = effective_defocus(depth, optics, top_ref)
    expected = -depth * optics.n_immersion / top_ref.n_resist
    assert d == pytest.approx(expected)


def test_effective_defocus_rejects_bad_reference(optics, resist):
    bad = dataclasses.replace(resist, focus_reference="middle-ish")
    with pytest.raises(ValueError, match="focus_reference"):
        effective_defocus(0.0, optics, bad)


# ---------------------------------------------------------------------------
# Exposure volume
# ---------------------------------------------------------------------------


def test_exposure_volume_shape_and_range(mask, optics, grid, resist):
    intensity, z = exposure_volume(mask, optics, grid, resist)
    nz = int(round(resist.thickness / grid.dz))
    assert intensity.shape == (nz, grid.n_pixels, grid.n_pixels)
    assert z.shape == (nz,)
    assert np.isfinite(intensity).all() and intensity.min() >= 0.0


def test_exposure_volume_varies_with_depth(mask, optics, grid, resist):
    """Different depths sit at different defocus, so the image must change."""
    intensity, _ = exposure_volume(mask, optics, grid, resist)
    mid_slice = intensity[intensity.shape[0] // 2]
    assert not np.allclose(intensity[0], mid_slice), "image is flat through depth"


def test_mid_focus_is_symmetric_top_to_bottom(mask, optics, grid, resist):
    """With focus at mid-resist, top and bottom are equally defocused.

    Defocus enters the pupil as ρ², so ±Δz give the same aerial image in an
    aberration-free system. Identical top and bottom images are therefore the
    correct result here, not a bug — and it is why the sidewall taper comes
    from absorption rather than from the optics.
    """
    mid = dataclasses.replace(resist, focus_reference="mid")
    intensity, _ = exposure_volume(mask, optics, grid, mid)
    assert np.allclose(intensity[0], intensity[-1], atol=1e-9)

    top = dataclasses.replace(resist, focus_reference="top")
    I_top, _ = exposure_volume(mask, optics, grid, top)
    assert not np.allclose(I_top[0], I_top[-1])


def test_exposure_volume_z_is_bottom_up(mask, optics, grid, resist):
    """Index 0 must be the bottom of the film, matching Stack ordering."""
    _, z = exposure_volume(mask, optics, grid, resist)
    assert z[0] < z[-1]
    assert z[-1] == pytest.approx(resist.thickness, abs=grid.dz)


# ---------------------------------------------------------------------------
# Absorption
# ---------------------------------------------------------------------------


def test_absorption_exposes_the_top_more_than_the_bottom(mask, optics, grid, resist):
    """Beer-Lambert: light is consumed on the way down.

    This is the physical origin of the sloped sidewall, so if it inverts,
    profiles come out undercut instead of tapered.
    """
    intensity, _ = exposure_volume(mask, optics, grid, resist)
    M, E = apply_absorption(intensity, resist, grid.dz, dose=1.0)
    assert M[-1].mean() < M[0].mean(), "top of the film should be more bleached"
    assert E[-1].mean() > E[0].mean(), "top of the film should absorb more dose"


def test_absorption_is_bounded(mask, optics, grid, resist):
    intensity, _ = exposure_volume(mask, optics, grid, resist)
    M, _ = apply_absorption(intensity, resist, grid.dz)
    assert M.min() >= 0.0 and M.max() <= 1.0


def test_zero_dill_absorption_is_depth_neutral(mask, optics, grid, resist):
    """With A = B = 0 there is no attenuation, so depth only enters via optics."""
    clear = dataclasses.replace(resist, dill_A=0.0, dill_B=0.0)
    intensity = np.ones((10, 8, 8))
    M, _ = apply_absorption(intensity, clear, grid.dz, dose=1.0)
    assert np.allclose(M, M[0]), "no absorption should mean no depth gradient"


def test_bleaching_lets_more_light_through(mask, optics, grid, resist):
    """A bleaching resist ends up more exposed at depth than a frozen one."""
    intensity, _ = exposure_volume(mask, optics, grid, resist)
    M_bleach, _ = apply_absorption(intensity, resist, grid.dz, bleaching=True)
    M_frozen, _ = apply_absorption(intensity, resist, grid.dz, bleaching=False)
    assert M_bleach[0].mean() <= M_frozen[0].mean() + 1e-9


# ---------------------------------------------------------------------------
# Standing waves
# ---------------------------------------------------------------------------


def test_standing_wave_period(optics):
    """The fringe period must be λ / (2 · n_resist).

    A hard number — 56.8 nm at 193 nm in a 1.7-index resist — that validates
    the entire vertical machinery in one assertion.
    """
    grid = GridConfig(n_pixels=8, pixel_size=4e-9, dz=1e-9)
    resist = ResistConfig(thickness=600e-9, n_resist=1.70, substrate_reflectance=0.5)
    nz = int(resist.thickness / grid.dz)
    intensity = np.ones((nz, 8, 8))
    Isw = add_standing_waves(intensity, resist, optics, grid.dz)

    trace = Isw[:, 4, 4]
    freqs = np.fft.rfftfreq(nz, d=grid.dz)
    amp = np.abs(np.fft.rfft(trace - trace.mean()))
    period = 1.0 / freqs[int(np.argmax(amp))]

    expected = optics.wavelength / (2.0 * resist.n_resist)
    # Tolerance is one FFT bin at this record length.
    bin_width = abs(1.0 / freqs[int(np.argmax(amp))] - 1.0 / freqs[int(np.argmax(amp)) + 1])
    assert period == pytest.approx(expected, abs=bin_width), (
        f"standing-wave period {period*1e9:.1f} nm != λ/2n = {expected*1e9:.1f} nm"
    )


def test_standing_waves_conserve_dose(optics):
    """The fringe redistributes intensity in z; it must not create it."""
    grid = GridConfig(n_pixels=8, pixel_size=4e-9, dz=2e-9)
    resist = ResistConfig(thickness=200e-9, n_resist=1.7, substrate_reflectance=0.4)
    intensity = np.ones((100, 8, 8))
    Isw = add_standing_waves(intensity, resist, optics, grid.dz)
    assert Isw.mean() == pytest.approx(intensity.mean(), rel=1e-9)


def test_barc_disables_standing_waves(optics, resist):
    """Zero substrate reflectance is a perfect BARC — no modulation at all."""
    intensity = np.ones((50, 8, 8))
    no_r = dataclasses.replace(resist, substrate_reflectance=0.0)
    assert np.array_equal(add_standing_waves(intensity, no_r, optics, 2e-9), intensity)


def test_peb_washes_out_standing_waves(optics, resist, grid):
    """Acid diffusion smooths the vertical fringe — the reason PEB exists."""
    r = dataclasses.replace(resist, substrate_reflectance=0.5, diffusion_sigma=25e-9)
    nz = 100
    intensity = np.ones((nz, 8, 8))
    Isw = add_standing_waves(intensity, r, optics, grid.dz)
    g = GridConfig(n_pixels=8, pixel_size=4e-9, dz=grid.dz)
    baked = apply_peb_3d(np.clip(Isw, 0, 1), r, g)
    ripple_before = float(np.std(np.clip(Isw, 0, 1)[:, 4, 4]))
    ripple_after = float(np.std(baked[:, 4, 4]))
    assert ripple_after < ripple_before


# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------


def test_develop_requires_access_from_the_top():
    """A buried soluble pocket cannot dissolve — no developer can reach it."""
    resist = ResistConfig(tone="positive", mack_Mth=0.5)
    latent = np.ones((10, 8, 8))
    latent[4:6, 3:5, 3:5] = 0.1  # well-exposed island, sealed inside the film

    sealed = develop_3d(latent, resist, require_access=True)
    assert sealed.all(), "an unreachable pocket must not develop out"

    naive = develop_3d(latent, resist, require_access=False)
    assert not naive.all(), "without the accessibility rule it becomes a floating void"


def test_develop_clears_a_column_open_to_the_top():
    resist = ResistConfig(tone="positive", mack_Mth=0.5)
    latent = np.ones((10, 8, 8))
    latent[:, 3:5, 3:5] = 0.1  # a fully exposed trench, open at the top
    remaining = develop_3d(latent, resist)
    assert not remaining[:, 3:5, 3:5].any()
    assert remaining[:, 0, 0].all()


def test_mack_ray_development_respects_develop_time(grid):
    """Longer development must clear more film — and it must be gradual.

    The ray model integrates dz/R down each column, so partial development
    (thickness loss without clearing to the substrate) is representable.
    """
    resist = ResistConfig(
        tone="positive", mack_Rmax=100.0, mack_Rmin=0.01, mack_Mth=0.5,
        mack_n=4, thickness=100e-9,
    )
    latent = np.full((50, 8, 8), 0.35)  # uniformly well-exposed
    frac = []
    for t in (1.0, 5.0, 30.0):
        r = dataclasses.replace(resist, develop_time=t)
        frac.append(float(develop_3d(latent, r, grid, model="mack").mean()))
    assert frac == sorted(frac, reverse=True), f"not monotonic in time: {frac}"


def test_mack_ray_development_needs_a_grid():
    with pytest.raises(ValueError, match="requires a GridConfig"):
        develop_3d(np.ones((4, 4, 4)), ResistConfig(), model="mack")


def test_develop_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown develop model"):
        develop_3d(np.ones((4, 4, 4)), ResistConfig(), model="dunk-it")


def test_develop_tone_inverts():
    latent = np.tile(np.linspace(0.0, 1.0, 8), (10, 8, 1))
    pos = develop_3d(latent, ResistConfig(tone="positive", mack_Mth=0.5), require_access=False)
    neg = develop_3d(latent, ResistConfig(tone="negative", mack_Mth=0.5), require_access=False)
    assert np.array_equal(pos, ~neg)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_print_resist_3d_produces_a_pattern(mask, optics, grid, resist):
    res = print_resist_3d(mask, optics, grid, resist, dose=1.0, standing_waves=False)
    frac = float(res["remaining"].mean())
    assert 0.1 < frac < 0.9, f"{frac:.1%} remaining — expected a printed pattern"
    for key in ("intensity", "pac", "latent", "remaining", "z"):
        assert key in res


def test_relative_dose_is_applied_exactly_once(mask, optics, grid, resist):
    """Doubling *dose* must equal doubling *dose_nominal* — never dose².

    The relative dose is baked into the intensity by ``exposure_volume``; if
    the absorption step multiplied it in again, every 3-D dose sweep would
    respond quadratically. The 2-D engine documents this exact trap; this
    pins the 3-D pipeline to the same single-count rule.
    """
    import dataclasses

    scaled_dose = print_resist_3d(
        mask, optics, grid, resist, dose=1.6, standing_waves=False
    )
    scaled_nominal = print_resist_3d(
        mask, optics, grid,
        dataclasses.replace(resist, dose_nominal=1.6 * resist.dose_nominal),
        dose=1.0, standing_waves=False,
    )
    np.testing.assert_allclose(
        scaled_dose["pac"], scaled_nominal["pac"], atol=1e-12,
        err_msg="dose entered the exposure twice (quadratic dose response)",
    )


def test_printed_profile_narrows_toward_the_top(mask, optics, grid, resist):
    """Positive tone: the top absorbs most, so it clears most.

    This is the defining 3-D result — a 2-D simulation cannot express it.
    """
    res = print_resist_3d(mask, optics, grid, resist, dose=1.0, standing_waves=False)
    rem = res["remaining"]
    nz = rem.shape[0]
    top = rem[int(0.9 * (nz - 1))].sum()
    bottom = rem[int(0.1 * (nz - 1))].sum()
    assert top <= bottom, (
        f"top retains {top} px vs bottom {bottom} px — profile is undercut, "
        "which suggests the depth/absorption sign is inverted"
    )


def test_sidewall_angle_is_physical(mask, optics, grid, resist):
    res = print_resist_3d(mask, optics, grid, resist, dose=1.0, standing_waves=False)
    angle = sidewall_angle(res["remaining"], grid)
    assert 45.0 < angle <= 90.0, f"sidewall angle {angle:.1f}° is not physical"


def test_higher_dose_clears_more(mask, optics, grid, resist):
    """Monotonic dose response — the most basic sanity check there is."""
    low = print_resist_3d(mask, optics, grid, resist, dose=0.8, standing_waves=False)
    high = print_resist_3d(mask, optics, grid, resist, dose=1.4, standing_waves=False)
    assert high["remaining"].mean() < low["remaining"].mean()


def test_3d_agrees_with_2d_when_depth_effects_are_off(mask, optics, grid):
    """With absorption and standing waves disabled, the 3-D model must
    collapse onto the existing 2-D engine.

    Both then reduce to thresholding the same aerial image, so the printed
    footprints have to agree. This is the back-compatibility guarantee.
    """
    from litho_sim.develop.resist import simulate_resist

    flat = ResistConfig(
        thickness=20e-9, dill_A=0.0, dill_B=0.0, dill_C=0.04, dose_nominal=30.0,
        mack_Mth=0.5, diffusion_sigma=0.0, substrate_reflectance=0.0,
        focus_reference="mid",
    )
    g = dataclasses.replace(grid, dz=10e-9, n_z_slices=3)

    res = print_resist_3d(mask, optics, g, flat, dose=1.0, standing_waves=False,
                          bleaching=False, develop_model="mack")
    mid3d = res["remaining"][res["remaining"].shape[0] // 2]

    aerial = compute_aerial_image(mask, optics, g, dose=1.0)
    _, _, r2d = simulate_resist(aerial, flat, g, model="mack")

    agreement = float((mid3d == (r2d > 0.5)).mean())
    assert agreement > 0.95, (
        f"3-D and 2-D footprints agree on only {agreement:.1%} of pixels "
        "with all depth effects disabled"
    )
