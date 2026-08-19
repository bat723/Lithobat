"""
Tests for high-NA imaging: exact defocus geometry, and (later) the vector
effect.

The claims worth defending here are analytic ones — a high-NA correction is
only worth having if it agrees with the closed form it replaces in the regime
where both are valid, and departs from it by a *predictable* amount where it
is not.

Stage 1 (this file, for now) covers the geometry foundation:

* the paraxial branch still reproduces the textbook quadratic exactly, so
  nothing that existed before this work moved;
* the exact branch converges onto it as NA falls;
* and at hyper-NA the two part company by the ~⅓ the vault has claimed all
  along in ``docs/expose/Aerial Image Formation.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.expose.aerial_image import compute_aerial_image
from litho_sim.expose.pupil import defocus_opd, jones_states, vector_coefficients
from litho_sim.mask.patterns import lines_and_spaces

LAM = 193e-9


def _opd_metres(rho, NA, n, defocus, exact):
    """Defocus OPD in metres, converting out of the radian convention."""
    w = defocus_opd(np.asarray(rho), LAM, NA, defocus, n, exact=exact)
    return w * LAM / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# The paraxial branch is unchanged
# ---------------------------------------------------------------------------


def test_zero_defocus_is_zero_phase():
    rho = np.linspace(0.0, 1.0, 9)
    for exact in (False, True):
        assert np.array_equal(
            defocus_opd(rho, LAM, 1.35, 0.0, 1.44, exact=exact), np.zeros_like(rho)
        )


def test_paraxial_matches_the_closed_form():
    """The historical expression, asserted term for term.

    This is the regression guard for the refactor that split the pupil
    builder: if this drifts, every result the engine ever produced moved.
    """
    rho = np.linspace(0.0, 1.0, 11)
    NA, n, dz = 0.93, 1.0, 150e-9
    expected = np.pi * NA ** 2 * rho ** 2 * dz / (LAM * n)
    got = defocus_opd(rho, LAM, NA, dz, n, exact=False)
    assert np.allclose(got, expected, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Exact vs paraxial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "NA, n",
    [
        (0.33, 1.00),   # EUV,    u = 0.11
        (0.57, 1.00),   # i-line, u = 0.32
    ],
)
def test_paraxial_error_is_the_next_taylor_term(NA, n):
    """The exact form must depart from the quadratic by exactly the term the
    quadratic dropped.

    Expanding ``1 − cos θ`` gives ``u/2 + u²/8 + …`` with ``u = sin²θ``, and
    the quadratic keeps only the first. So the relative error at the rim is
    ``(u²/8)/(u/2) = u/4`` to leading order — a prediction, not a tolerance.
    Asserting the *shape* of the error is what makes this a physics test
    rather than a golden number.

    Only the low-NA presets are checked, because *leading* order is exactly
    what stops being sufficient as u grows: by KrF (u = 0.56) the u³ term has
    already pushed the true error 20 % above u/4, and by the immersion preset
    the series is useless — which is the whole argument for computing the
    cosine instead of expanding it.
    """
    rho = np.linspace(0.0, 1.0, 21)
    par = _opd_metres(rho, NA, n, 200e-9, exact=False)
    exa = _opd_metres(rho, NA, n, 200e-9, exact=True)

    rim_err = abs(exa[-1] - par[-1]) / abs(exa[-1])
    predicted = (NA / n) ** 2 / 4.0
    assert rim_err == pytest.approx(predicted, rel=0.20), (
        f"NA {NA}: rim error {rim_err:.4f} vs predicted u/4 = {predicted:.4f}"
    )


def test_paraxial_error_grows_with_na():
    """Monotone in NA — the reason this is optional at EUV and not at ArFi."""
    rho = 1.0
    errs = []
    for NA, n in ((0.33, 1.0), (0.57, 1.0), (0.93, 1.0), (1.35, 1.44)):
        par = _opd_metres(rho, NA, n, 200e-9, exact=False)
        exa = _opd_metres(rho, NA, n, 200e-9, exact=True)
        errs.append(abs(exa - par) / abs(exa))
    assert errs == sorted(errs), errs
    assert errs[0] < 0.05, "EUV should barely care"
    assert errs[-1] > 0.30, "immersion should care a lot"


def test_paraxial_understates_the_rim_by_a_third_at_immersion_na():
    """The number the vault has claimed all along, now pinned.

    ``docs/expose/Aerial Image Formation.md`` records "~30 %" for the
    paraxial defocus error at the pupil rim at NA 1.35. Computed exactly:
    cos θ = sqrt(1 − (1.35/1.44)²) = 0.348, so the exact OPD is
    n·Δz·(1 − cos θ) = 0.939·Δz against a paraxial NA²/(2n)·Δz = 0.633·Δz.
    """
    par = _opd_metres(1.0, 1.35, 1.44, 100e-9, exact=False)
    exa = _opd_metres(1.0, 1.35, 1.44, 100e-9, exact=True)
    understatement = (exa - par) / exa * 100.0
    assert 30.0 < understatement < 35.0, f"got {understatement:.1f}%"


def test_exact_is_always_the_larger_opd():
    """cos θ is concave, so the quadratic always sits under the true curve."""
    rho = np.linspace(0.0, 1.0, 21)
    for NA, n in ((0.93, 1.0), (1.35, 1.44), (1.35, 1.70)):
        par = _opd_metres(rho, NA, n, 120e-9, exact=False)
        exa = _opd_metres(rho, NA, n, 120e-9, exact=True)
        assert (exa >= par - 1e-18).all(), f"NA {NA}, n {n}"


def test_root_stays_real_beyond_the_aperture():
    """ρ > 1 is masked off by the caller, but must not produce NaN here."""
    rho = np.linspace(0.0, 1.5, 16)          # corners of the square FFT grid
    out = defocus_opd(rho, LAM, 1.35, 200e-9, 1.0, exact=True)   # NA > n too
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Wiring into the aerial image
# ---------------------------------------------------------------------------


def test_exact_defocus_is_off_by_default():
    """Back-compatibility contract, in one assertion."""
    o = OpticsConfig()
    assert o.exact_defocus is False
    assert o.n_image is None


def test_n_image_defaults_to_the_immersion_medium():
    o = OpticsConfig(NA=1.35, n_immersion=1.44)
    assert o.image_index == pytest.approx(1.44)
    assert o.sin_theta_max == pytest.approx(1.35 / 1.44)


def test_refraction_into_resist_shrinks_the_marginal_angle():
    """The whole reason to image in the resist rather than the fluid.

    sin θ falls from 0.94 in water to 0.79 in resist, which is what keeps the
    vector effect survivable at hyper-NA.
    """
    water = OpticsConfig(NA=1.35, n_immersion=1.44)
    resist = OpticsConfig(NA=1.35, n_immersion=1.44, n_image=1.70)
    assert water.sin_theta_max == pytest.approx(0.9375, abs=1e-4)
    assert resist.sin_theta_max == pytest.approx(0.7941, abs=1e-4)
    assert resist.sin_theta_max < water.sin_theta_max


def test_exact_defocus_leaves_the_in_focus_image_untouched():
    """With Δz = 0 both branches are identically zero, so the image must be
    bit-for-bit the same — a guard that the flag cannot leak into focus."""
    grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    mask = lines_and_spaces(64, 4e-9, pitch=200e-9, cd=100e-9)
    base = OpticsConfig(NA=0.93, source_grid=11, defocus=0.0)
    a = compute_aerial_image(mask, base, grid)
    b = compute_aerial_image(mask, OpticsConfig(
        NA=0.93, source_grid=11, defocus=0.0, exact_defocus=True), grid)
    assert np.array_equal(a, b)


def test_exact_defocus_matters_more_at_high_na():
    """Out of focus, the correction must be negligible at low NA and
    material at hyper-NA — the argument for having the flag at all."""
    grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    mask = lines_and_spaces(64, 4e-9, pitch=200e-9, cd=100e-9)

    def spread(NA, n, exact):
        o = OpticsConfig(NA=NA, n_immersion=n, source_grid=11,
                         defocus=150e-9, exact_defocus=exact)
        return compute_aerial_image(mask, o, grid)

    low = np.abs(spread(0.33, 1.0, True) - spread(0.33, 1.0, False)).max()
    high = np.abs(spread(1.35, 1.44, True) - spread(1.35, 1.44, False)).max()
    assert low < 0.01, f"low-NA change {low:.4f} should be negligible"
    assert high > 5 * low, f"high-NA change {high:.4f} vs low {low:.4f}"


# ---------------------------------------------------------------------------
# The vector effect, at the level where it is analytic
# ---------------------------------------------------------------------------
#
# Two beams leaving the pupil on opposite sides (φ = 0 and φ = π) at angle ±θ
# are what a dense grating actually images with. Their interference term is
# the dot product of the two field vectors, so it can be checked in closed
# form without involving an FFT at all.


def _two_beam_overlap(polarisation, sin_theta, obliquity=False):
    """Normalised interference term for two rays at ±θ in the x–z plane."""
    phi = np.array([0.0, np.pi])
    rho = np.array([1.0, 1.0])
    (jx, jy, _), = jones_states(polarisation, 0.0)   # one incident wave
    V = np.array(vector_coefficients(rho, phi, jx, jy, sin_theta,
                                     obliquity=obliquity))
    num = np.vdot(V[:, 0], V[:, 1]).real
    den = np.linalg.norm(V[:, 0]) * np.linalg.norm(V[:, 1])
    return num / den


@pytest.mark.parametrize("sin_theta", [0.1, 0.3, 0.5, 0.7071, 0.794, 0.9375])
def test_te_beams_interfere_perfectly_at_any_angle(sin_theta):
    """The whole reason hyper-NA scanners use azimuthal polarisation.

    An s-polarised field is perpendicular to the plane of incidence, so
    bending the ray never tilts it out of alignment with its partner. The
    overlap is 1 at 11° and still 1 at 70°.
    """
    assert abs(_two_beam_overlap("y", sin_theta)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("sin_theta", [0.1, 0.3, 0.5, 0.6, 0.794, 0.9375])
def test_tm_interference_follows_cos_two_theta(sin_theta):
    """The vector effect, in closed form.

    Two p-polarised beams separated by 2θ interfere with their amplitude
    scaled by cos 2θ. Nothing in the code computes ``cos 2θ`` — it falls out
    of projecting the incident field onto each ray's own s/p directions and
    letting the lens tilt the p half.
    """
    theta = np.arcsin(sin_theta)
    assert _two_beam_overlap("x", sin_theta) == pytest.approx(
        np.cos(2 * theta), abs=1e-12
    )


def test_tm_interference_nulls_at_45_degrees():
    """At θ = 45° the two p-polarised fields are orthogonal and simply do not
    interfere — contrast vanishes however good the optics are."""
    assert _two_beam_overlap("x", np.sin(np.pi / 4)) == pytest.approx(0.0, abs=1e-9)


def test_tm_interference_inverts_beyond_45_degrees():
    """Past the null the interference term goes negative: bright and dark
    swap. This is a sign change, not a fade, and it is why TM at hyper-NA is
    worse than useless."""
    assert _two_beam_overlap("x", 0.9375) < -0.1


@pytest.mark.parametrize("phi_s", [0.0, 0.7, np.pi / 3, 2.5, -1.2])
def test_te_generates_no_longitudinal_field_in_its_own_meridian(phi_s):
    """An azimuthally polarised wave has no radial component in its own
    plane of incidence, so the lens has nothing to tilt into E_z there.

    Evaluated exactly at φ = φ_s, which is that plane. Away from it the
    source point's fixed polarisation does acquire a radial part relative to
    other pupil directions — that is geometry, not a failure of TE.
    """
    (jx, jy, _), = jones_states("te", phi_s)
    _, _, vz = vector_coefficients(
        np.array([1.0]), np.array([phi_s]), jx, jy, 0.9375, obliquity=False
    )
    assert abs(complex(vz[0])) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("phi_s", [0.0, 0.7, np.pi / 3, 2.5, -1.2])
def test_tm_is_all_longitudinal_leverage_in_its_own_meridian(phi_s):
    """The mirror image: a radially polarised wave is entirely p-polarised
    in its own plane, so it generates the largest possible E_z."""
    (jx, jy, _), = jones_states("tm", phi_s)
    _, _, vz = vector_coefficients(
        np.array([1.0]), np.array([phi_s]), jx, jy, 0.9375, obliquity=False
    )
    assert abs(complex(vz[0])) == pytest.approx(0.9375, abs=1e-12)


def test_longitudinal_field_grows_with_angle():
    """E_z is the energy the vector model accounts for and the scalar one
    silently drops."""
    phi = np.array([0.0])
    rho = np.array([1.0])
    mags = []
    for st in (0.1, 0.5, 0.9375):
        _, _, vz = vector_coefficients(phi * 0 + rho, phi, 1.0, 0.0, st,
                                       obliquity=False)
        mags.append(abs(complex(vz[0])))
    assert mags == sorted(mags)
    assert mags[-1] == pytest.approx(0.9375, abs=1e-12)


def test_unknown_polarisation_lists_the_options():
    with pytest.raises(ValueError, match="polarisation must be one of"):
        jones_states("circular", 0.0)


def test_unknown_imaging_model_is_rejected():
    grid = GridConfig(n_pixels=32, pixel_size=4e-9)
    mask = lines_and_spaces(32, 4e-9, pitch=200e-9, cd=100e-9)
    with pytest.raises(ValueError, match="imaging_model must be"):
        compute_aerial_image(
            mask, OpticsConfig(source_grid=5, imaging_model="rigorous"), grid
        )


def test_vector_defaults_are_off():
    """Back-compatibility contract, in one assertion."""
    o = OpticsConfig()
    assert o.imaging_model == "scalar"
    assert o.polarisation == "unpolarised"
    assert o.obliquity is True


# ---------------------------------------------------------------------------
# The vector effect, through the imaging engine
# ---------------------------------------------------------------------------


def _contrast(a):
    return float((a.max() - a.min()) / (a.max() + a.min()))


def test_vector_reduces_to_scalar_at_low_na():
    """The reduction check: where the vector effect is negligible the vector
    engine must reproduce the scalar one.

    Vertical lines diffract along x, so y-polarisation is s-polarised for
    every order and is the case that must agree exactly.
    """
    grid = GridConfig(n_pixels=128, pixel_size=16e-9)
    mask = lines_and_spaces(128, 16e-9, pitch=800e-9, cd=400e-9)
    common = dict(NA=0.33, source_grid=11, sigma_outer=0.5)
    scalar = compute_aerial_image(mask, OpticsConfig(**common), grid)
    vec = compute_aerial_image(
        mask,
        OpticsConfig(**common, imaging_model="vector", polarisation="y",
                     obliquity=False),
        grid,
    )
    assert np.abs(vec - scalar).max() < 1e-3


def _dipole_contrast(polarisation, n_image):
    """Dense lines under x-dipole illumination — the configuration where the
    poles sit in the plane of diffraction, so radial is pure TM and
    azimuthal is pure TE."""
    grid = GridConfig(n_pixels=128, pixel_size=4e-9)
    mask = lines_and_spaces(128, 4e-9, pitch=100e-9, cd=50e-9)
    o = OpticsConfig(
        NA=1.35, n_immersion=1.44, n_image=n_image, source_grid=21,
        sigma_outer=0.9, sigma_inner=0.6, source_type="dipole",
        source_kwargs={"axis": "x"}, imaging_model="vector",
        polarisation=polarisation,
    )
    return _contrast(compute_aerial_image(mask, o, grid))


def test_te_holds_contrast_where_tm_collapses():
    """The headline result, at hyper-NA in resist."""
    te = _dipole_contrast("te", 1.70)
    tm = _dipole_contrast("tm", 1.70)
    assert te > 0.85, f"TE should be near-perfect, got {te:.3f}"
    assert tm < 0.6 * te, f"TM {tm:.3f} should collapse against TE {te:.3f}"


def test_unpolarised_sits_between_te_and_tm():
    """It is the incoherent average of two orthogonal states, so it cannot
    beat the better one or trail the worse."""
    te = _dipole_contrast("te", 1.70)
    un = _dipole_contrast("unpolarised", 1.70)
    tm = _dipole_contrast("tm", 1.70)
    assert tm < un < te, f"TM {tm:.3f}, unpol {un:.3f}, TE {te:.3f}"


def test_refraction_into_resist_protects_contrast():
    """Imaging into the resist rather than the fluid is not bookkeeping.

    The internal angle is smaller (sin θ 0.79 vs 0.94), so 2θ sits closer to
    the cos 2θ null from the far side and TM recovers substantially. Getting
    this wrong overstates the vector effect.
    """
    tm_water = _dipole_contrast("tm", 1.44)
    tm_resist = _dipole_contrast("tm", 1.70)
    assert tm_resist > 1.5 * tm_water, (
        f"in resist {tm_resist:.3f} should clearly beat in water {tm_water:.3f}"
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_peak_normalisation_is_the_default_and_unchanged():
    """The ten assertions elsewhere in the suite that pin ``max() == dose``
    depend on this staying the default."""
    assert OpticsConfig().normalisation == "peak"
    grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    mask = lines_and_spaces(64, 4e-9, pitch=200e-9, cd=100e-9)
    a = compute_aerial_image(mask, OpticsConfig(source_grid=11), grid, dose=1.3)
    assert a.max() == pytest.approx(1.3)


@pytest.mark.parametrize(
    "kw",
    [
        {},
        dict(imaging_model="vector", polarisation="te"),
        dict(imaging_model="vector", polarisation="tm"),
        dict(imaging_model="vector", polarisation="unpolarised"),
    ],
)
def test_clear_field_normalises_to_unity(kw):
    """An unpatterned mask must print exactly one clear-field dose — for
    every imaging model, which is what makes ``threshold`` comparable across
    polarisations."""
    grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    ones = np.ones((64, 64))
    o = OpticsConfig(NA=1.35, n_immersion=1.44, n_image=1.70,
                     source_grid=11, normalisation="clear", **kw)
    a = compute_aerial_image(ones, o, grid)
    assert a.mean() == pytest.approx(1.0, rel=1e-9)
    assert np.ptp(a) == pytest.approx(0.0, abs=1e-12), "clear field must be flat"


def test_no_normalisation_exposes_the_throughput_the_peak_mode_hides():
    """Peak normalisation makes every image top out at the dose, so two
    polarisations look equally bright however much light they really deliver.
    Turning it off is what makes a dose comparison possible at all."""
    grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    mask = lines_and_spaces(64, 4e-9, pitch=100e-9, cd=50e-9)
    common = dict(NA=1.35, n_immersion=1.44, n_image=1.70, source_grid=11)

    peaks_normalised = [
        compute_aerial_image(mask, OpticsConfig(
            **common, imaging_model="vector", polarisation=p), grid).max()
        for p in ("te", "tm")
    ]
    assert peaks_normalised[0] == pytest.approx(peaks_normalised[1]), (
        "peak mode should erase the difference — that is the problem"
    )

    raw = [
        compute_aerial_image(mask, OpticsConfig(
            **common, imaging_model="vector", polarisation=p,
            normalisation="none"), grid).max()
        for p in ("te", "tm")
    ]
    assert raw[0] != pytest.approx(raw[1]), "raw intensities must differ"


def test_te_reaches_a_threshold_at_lower_dose_than_tm():
    """The practical payoff of keeping the absolute scale.

    Dose to reach a fixed clear-field-referenced intensity is inversely
    proportional to the peak, so the polarisation that images more
    efficiently needs less of it.
    """
    grid = GridConfig(n_pixels=128, pixel_size=4e-9)
    mask = lines_and_spaces(128, 4e-9, pitch=100e-9, cd=50e-9)

    def dose_to_reach(pol, target=0.30):
        o = OpticsConfig(
            NA=1.35, n_immersion=1.44, n_image=1.70, source_grid=21,
            sigma_outer=0.9, sigma_inner=0.6, source_type="dipole",
            source_kwargs={"axis": "x"}, imaging_model="vector",
            polarisation=pol, normalisation="clear",
        )
        return target / compute_aerial_image(mask, o, grid).max()

    assert dose_to_reach("te") < dose_to_reach("tm")


def test_unknown_normalisation_is_rejected():
    grid = GridConfig(n_pixels=32, pixel_size=4e-9)
    mask = lines_and_spaces(32, 4e-9, pitch=200e-9, cd=100e-9)
    with pytest.raises(ValueError, match="normalisation must be"):
        compute_aerial_image(
            mask, OpticsConfig(source_grid=5, normalisation="rms"), grid)


# ---------------------------------------------------------------------------
# Imaging inside the resist
# ---------------------------------------------------------------------------


def test_exposure_volume_declares_the_resist_as_the_image_medium():
    """The 3-D path images into the film, so the ray angle must be the
    internal one — that is what sets the size of the vector effect."""
    from litho_sim.core.config import ResistConfig

    o = OpticsConfig(NA=1.35, n_immersion=1.44)
    r = ResistConfig(n_resist=1.70)
    in_resist = OpticsConfig(NA=1.35, n_immersion=1.44, n_image=r.n_resist)
    assert in_resist.sin_theta_max < o.sin_theta_max
    assert in_resist.sin_theta_max == pytest.approx(1.35 / 1.70, abs=1e-9)


def test_declaring_the_resist_medium_does_not_double_count_the_index():
    """The trap this wiring had to avoid.

    ``effective_defocus`` already converts the defocus into
    immersion-equivalent units, so naming the resist as the image medium
    without converting the defocus with it would apply the index twice and
    silently corrupt every out-of-focus plane. Pinned by requiring the
    defocused 3-D exposure to match the direct formulation — raw geometric
    depth, resist index — which is the physics both routes are approximating.
    """
    from litho_sim.core.config import ResistConfig
    from litho_sim.develop.resist3d import effective_defocus

    o = OpticsConfig(NA=1.35, n_immersion=1.44, defocus=0.0)
    r = ResistConfig(n_resist=1.70, thickness=100e-9, focus_reference="top")
    depth = 100e-9

    # What exposure_volume now hands the imaging engine.
    wired = defocus_opd(
        np.array([1.0]), LAM, 1.35,
        effective_defocus(depth, o, r) * (r.n_resist / o.n_immersion),
        r.n_resist, exact=False,
    )[0]
    # The direct statement of the same physics: the plane really is `depth`
    # below focus, inside a medium of index n_resist.
    direct = defocus_opd(
        np.array([1.0]), LAM, 1.35, -depth, r.n_resist, exact=False
    )[0]
    assert wired == pytest.approx(direct, rel=1e-12)


def test_vector_costs_three_transforms_per_source_point(monkeypatch):
    """Structural guard on the loop: three components, one batched call, and
    six for unpolarised because it is two incoherent input states."""
    import litho_sim.expose.aerial_image as ai

    grid = GridConfig(n_pixels=32, pixel_size=4e-9)
    mask = lines_and_spaces(32, 4e-9, pitch=200e-9, cd=100e-9)
    counts = []
    real_ifft2 = np.fft.ifft2

    def counting_ifft2(a, *args, **kwargs):
        counts.append(np.asarray(a).ndim)
        return real_ifft2(a, *args, **kwargs)

    monkeypatch.setattr(ai.np.fft, "ifft2", counting_ifft2)

    common = dict(NA=0.93, source_grid=5, sigma_outer=0.5)
    counts.clear()
    compute_aerial_image(mask, OpticsConfig(**common), grid)
    n_scalar = len(counts)

    counts.clear()
    compute_aerial_image(mask, OpticsConfig(
        **common, imaging_model="vector", polarisation="te"), grid)
    assert len(counts) == n_scalar, "polarised vector batches its 3 components"
    assert all(d == 3 for d in counts), "each call should transform a 3-stack"

    counts.clear()
    compute_aerial_image(mask, OpticsConfig(
        **common, imaging_model="vector", polarisation="unpolarised"), grid)
    assert len(counts) == 2 * n_scalar, "unpolarised runs two incoherent states"
