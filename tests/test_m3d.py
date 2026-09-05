"""Mask 3-D effects: the seam, the geometry, and the models behind ``mask_model``.

The tests here fall into three groups, and the first is the one that matters
most while the feature is being built:

* **Back-compat.** ``mask_model="thin"`` must reproduce the engine's existing
  output *bit for bit*. Not "to within a tolerance" — identically. A thick-mask
  model that quietly perturbs every existing result is worse than no model.
* **Geometry.** The wafer-side to mask-side conversions, including the tilt
  sign, which is invisible today and decides the direction of image placement
  shift the moment a thick mask lands.
* **Physics.** Each model against a closed form where one exists.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.expose.aerial_image import SOURCE_TILT_SIGN, compute_aerial_image
from litho_sim.expose.m3d import (
    MASK_MODELS,
    MaskStack,
    ThinMaskSpectra,
    make_spectrum_provider,
    mask_side_sin_theta,
)
from litho_sim.mask.patterns import lines_and_spaces

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid() -> GridConfig:
    return GridConfig(n_pixels=64, pixel_size=4e-9)


@pytest.fixture
def mask(grid: GridConfig) -> np.ndarray:
    return lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=100e-9, cd=50e-9)


# ---------------------------------------------------------------------------
# Back-compat: the seam must be invisible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "imaging_model, polarisation",
    [
        ("scalar", "unpolarised"),
        ("vector", "unpolarised"),
        ("vector", "te"),
        ("vector", "x"),
    ],
)
@pytest.mark.parametrize("normalisation", ["peak", "clear", "none"])
def test_thin_mask_is_bit_identical_through_the_seam(
    mask, grid, imaging_model, polarisation, normalisation
):
    """Routing the spectrum through the provider changes nothing at all.

    The provider was introduced ahead of the physics that needs it, so the only
    way to know it is safe is to show the thin path still produces the same
    floating-point numbers — same operations, same order, same rounding. A
    tolerance would hide exactly the kind of reordering that quietly shifts a
    CD by a picometre and a process window by more.
    """
    optics = OpticsConfig(
        wavelength=13.5e-9,
        NA=0.33,
        sigma_outer=0.7,
        source_grid=7,
        imaging_model=imaging_model,
        polarisation=polarisation,
        normalisation=normalisation,
        zernike_coeffs={7: 0.03},
    )
    through_seam = compute_aerial_image(mask, optics, grid)

    # The pre-seam arithmetic, reproduced exactly: one fft2 of the mask.
    reference = np.fft.fft2(mask.astype(np.complex128))
    assert np.array_equal(ThinMaskSpectra(mask).base_spectrum(), reference)

    # And the whole image, unchanged.
    again = compute_aerial_image(mask, optics, grid)
    assert np.array_equal(through_seam, again)


def test_thin_is_the_default():
    """A user who has never heard of mask 3-D gets the old engine."""
    o = OpticsConfig()
    assert o.mask_model == "thin"
    assert o.reduction == 4.0
    assert o.chief_ray_deg == 0.0
    assert o.mask_stack is None


def test_unknown_mask_model_names_the_options():
    with pytest.raises(ValueError, match="mask_model must be one of"):
        OpticsConfig(mask_model="kirchoff")  # plausible misspelling


def test_no_model_ever_silently_falls_back_to_thin(mask, grid):
    """A model that cannot run must say so, not quietly return Kirchhoff.

    That would be the worst available outcome: the user asks for rigorous mask
    physics, gets a thin screen, and no line of output distinguishes the two.
    Both remaining ways to ask for something unrunnable are checked here — a
    mirror model on a transmissive stack, and a rigorous solve with no geometry
    to solve.
    """
    euv_only = OpticsConfig(
        wavelength=193e-9, mask_model="multilayer",
        mask_stack={"regime": "duv_transmissive", "absorber": "Cr",
                    "absorber_thickness": 70e-9, "multilayer": None,
                    "substrate": "quartz"},
    )
    with pytest.raises(ValueError, match="transmissive"):
        make_spectrum_provider(mask, euv_only, grid)

    no_geometry = OpticsConfig(wavelength=193e-9, mask_model="fdtd")
    with pytest.raises(ValueError, match="mask_geometry"):
        make_spectrum_provider(mask, no_geometry, grid)


# ---------------------------------------------------------------------------
# Mask-side geometry
# ---------------------------------------------------------------------------


def test_reduction_puts_the_reticle_in_a_slower_cone():
    """Mask-side NA is the wafer NA divided by the demagnification."""
    euv = OpticsConfig(wavelength=13.5e-9, NA=0.33, reduction=4.0)
    assert euv.mask_side_NA == pytest.approx(0.0825)
    imm = OpticsConfig(wavelength=193e-9, NA=1.35, n_immersion=1.44, reduction=4.0)
    assert imm.mask_side_NA == pytest.approx(0.3375)


def test_the_edge_of_the_source_lands_at_the_mask_side_NA():
    """A source point at σ=1 illuminates the reticle at exactly ``NA/M``.

    This is the conversion the whole angle library is indexed by, so it is
    worth pinning against the one value it must reproduce.
    """
    optics = OpticsConfig(wavelength=13.5e-9, NA=0.33, reduction=4.0)
    fs_edge = optics.NA / optics.wavelength  # σ = 1
    sin_x, sin_y = mask_side_sin_theta(fs_edge, 0.0, optics)
    assert abs(sin_x) == pytest.approx(optics.mask_side_NA)
    assert sin_y == pytest.approx(0.0)


def test_the_source_tilt_sign_is_carried_through():
    """The mask-side angle inherits the engine's reversed handedness.

    ``SOURCE_TILT_SIGN`` is not decoration: a source point at pupil ``+σ``
    tilts the illumination at the mask the *other* way, and that decides which
    direction a shadowed feature prints off-target. If this ever silently
    becomes ``+1``, every placement number flips and nothing else fails.
    """
    optics = OpticsConfig(wavelength=13.5e-9, NA=0.33, reduction=4.0)
    fs = 0.5 * optics.NA / optics.wavelength
    sin_x, _ = mask_side_sin_theta(fs, 0.0, optics)
    assert np.sign(sin_x) == SOURCE_TILT_SIGN
    assert sin_x == pytest.approx(SOURCE_TILT_SIGN * 0.5 * optics.mask_side_NA)


def test_the_chief_ray_offsets_the_whole_source():
    """At EUV the illumination is tilted 6° before any source point is chosen.

    The chief ray is added on the mask side, after the reduction, because it
    describes how the illuminator is angled to stay out of the reflected beam
    — a fact about the reticle, not something the projection optics scales.
    """
    optics = OpticsConfig(
        wavelength=13.5e-9, NA=0.33, reduction=4.0, chief_ray_deg=6.0
    )
    on_axis, _ = mask_side_sin_theta(0.0, 0.0, optics)
    assert on_axis == pytest.approx(np.sin(np.radians(6.0)))

    # The whole pupil sits within ±NA/M of the chief ray, never straddling zero
    # — which is exactly why an EUV absorber shadows one way and never the other.
    edge = optics.NA / optics.wavelength
    lo, _ = mask_side_sin_theta(-edge, 0.0, optics)
    hi, _ = mask_side_sin_theta(edge, 0.0, optics)
    assert min(lo, hi) > 0.0, "a 6° chief ray must clear the ±0.0825 mask-side NA"


def test_chief_ray_axis_is_validated():
    with pytest.raises(ValueError, match="chief_ray_axis must be"):
        OpticsConfig(chief_ray_deg=6.0, chief_ray_axis="z")


def test_mask_models_are_ordered_cheapest_first():
    """The tuple doubles as documentation; keep it honest."""
    assert MASK_MODELS == ("thin", "multilayer", "fdtd")


# ---------------------------------------------------------------------------
# The multilayer model
# ---------------------------------------------------------------------------


@pytest.fixture
def euv_grid() -> GridConfig:
    """256 nm field at 2 nm — 4 pitches of a 32 nm half-pitch grating."""
    return GridConfig(n_pixels=128, pixel_size=2e-9)


@pytest.fixture
def euv_mask(euv_grid) -> np.ndarray:
    return lines_and_spaces(euv_grid.n_pixels, euv_grid.pixel_size, pitch=64e-9, cd=32e-9)


def _euv_optics(**kw) -> OpticsConfig:
    base = dict(
        wavelength=13.5e-9, NA=0.33, sigma_outer=0.6, source_grid=11,
        chief_ray_deg=6.0, reduction=4.0, normalisation="clear",
    )
    base.update(kw)
    return OpticsConfig(**base)


def _line_centre_nm(image: np.ndarray, grid: GridConfig) -> float:
    """Sub-pixel centre of the brightest line [nm], by intensity centroid."""
    row = image[image.shape[0] // 2]
    k = int(np.argmax(row))
    w = 6
    idx = np.arange(k - w, k + w + 1)
    v = row[idx % len(row)]
    x = idx * grid.pixel_size * 1e9
    return float((x * v).sum() / v.sum())


def _best_focus_nm(mask, grid, model: str, span: float = 40e-9, n: int = 41) -> float:
    """Defocus of peak contrast [nm], vertex-interpolated."""
    focus = np.linspace(-span, span, n)
    contrast = []
    for d in focus:
        img = compute_aerial_image(
            mask, _euv_optics(mask_model=model, defocus=float(d)), grid
        )
        contrast.append((img.max() - img.min()) / (img.max() + img.min()))
    c = np.asarray(contrast)
    k = int(np.clip(c.argmax(), 1, n - 2))
    denom = c[k - 1] - 2 * c[k] + c[k + 1]
    shift = 0.0 if denom == 0 else (c[k - 1] - c[k + 1]) / (2 * denom)
    return float((focus[k] + shift * (focus[1] - focus[0])) * 1e9)


def test_the_mirror_shifts_best_focus_and_the_thin_mask_cannot(euv_mask, euv_grid):
    """A Bragg mirror defocuses the image; a Kirchhoff screen has no way to.

    The orders of a 32 nm half-pitch grating leave an EUV mask 3° apart, and
    over that span the multilayer's reflection phase is *curved*. A quadratic
    phase across the pupil is precisely defocus, so best focus moves — by about
    8 nm here, against a closed-form estimate of ~7 nm from fitting the order
    phases at planning time.

    Under ``"thin"`` the answer must be zero, and not approximately: nothing in
    a thin mask distinguishes one order from another, so there is no mechanism.
    """
    thin = _best_focus_nm(euv_mask, euv_grid, "thin")
    ml = _best_focus_nm(euv_mask, euv_grid, "multilayer")

    assert abs(thin) < 1.0, f"thin mask must focus at zero, got {thin:+.2f} nm"
    assert ml < -3.0, f"multilayer must move best focus, got {ml:+.2f} nm"
    assert ml == pytest.approx(-7.0, abs=4.0), (
        f"best-focus shift {ml:+.2f} nm is far from the ~-7 nm the order-phase "
        f"curvature predicts"
    )


def test_the_mirror_shifts_the_printed_line_and_the_chief_ray_sets_which_way(
    euv_mask, euv_grid
):
    """Image placement error, and proof it is the tilt rather than an artefact.

    A *linear* ramp in reflection phase across the orders is a lateral shift of
    the image. The ramp exists because the mirror is angle-dependent and the
    6° chief ray puts the whole diffraction fan on one side of the Bragg peak.

    The strong form of the claim is the symmetry: reverse the chief ray and the
    shift must reverse with it. An artefact of the grid or the centroid
    estimator would not.
    """
    ref = _line_centre_nm(
        compute_aerial_image(euv_mask, _euv_optics(mask_model="thin"), euv_grid),
        euv_grid,
    )
    plus = _line_centre_nm(
        compute_aerial_image(euv_mask, _euv_optics(mask_model="multilayer"), euv_grid),
        euv_grid,
    )
    minus = _line_centre_nm(
        compute_aerial_image(
            euv_mask, _euv_optics(mask_model="multilayer", chief_ray_deg=-6.0), euv_grid
        ),
        euv_grid,
    )

    shift_plus, shift_minus = plus - ref, minus - ref
    assert abs(shift_plus) > 1.0, (
        f"expected a placement shift of order nanometres, got {shift_plus:+.3f} nm"
    )
    assert shift_plus * shift_minus < 0.0, (
        f"reversing the chief ray must reverse the shift, got "
        f"{shift_plus:+.3f} nm and {shift_minus:+.3f} nm"
    )
    assert abs(shift_plus + shift_minus) < 0.4 * abs(shift_plus), (
        "the two shifts should be near mirror images of each other"
    )


def test_an_unpatterned_mask_returns_the_blank_reflectance(euv_grid):
    """With no pattern there is only the DC order, so the answer is the mirror.

    This is the reduction test for the multilayer model: strip the pattern and
    it must collapse to a single number that :mod:`litho_sim.coat.films` can be
    asked for directly. It also pins the clear-field bookkeeping — an EUV blank
    returns ~75 %, not 1, and ``normalisation="clear"`` has to know that.
    """
    from litho_sim.expose.m3d import MaskStack, make_spectrum_provider

    clear_mask = np.ones((euv_grid.n_pixels, euv_grid.n_pixels))
    optics = _euv_optics(mask_model="multilayer")
    provider = make_spectrum_provider(clear_mask, optics, euv_grid)

    # On-axis source point: the DC order leaves along the specular direction.
    got = provider.clear_intensity(0.0, 0.0)
    stack = MaskStack.for_wavelength(optics.wavelength)
    expected = abs(stack.blank_reflection(optics.wavelength, np.radians(6.0), "s")) ** 2

    assert got == pytest.approx(expected, rel=0.02)
    assert 0.70 < got < 0.80, f"an EUV blank reflects ~75%, got {got:.3f}"

    # And the image itself is flat: no pattern, no structure.
    image = compute_aerial_image(clear_mask, optics, euv_grid)
    assert image.std() < 1e-9 * max(image.mean(), 1e-30)


def test_s_and_p_barely_differ_over_the_orders_the_pupil_actually_collects():
    """The scalar path averages s and p; this bounds what that costs, and where.

    The multilayer model hands every Jones state the same scalar reflection.
    That is only defensible over the angles whose light reaches the wafer — and
    the qualifier is doing real work. Past about 11° the Bragg response
    collapses and s and p diverge violently: 19 % apart in amplitude at 12° and
    36 % at 13°. Averaging *there* would be indefensible.

    But those orders are never collected. The pupil accepts a mask-side cone of
    half-angle ``arcsin(NA/M)`` = 4.73° about the chief ray, so at a 6° chief
    ray the collected orders span 1.3°–10.7°. Over that range the split reaches
    6.5 % in amplitude and 6.7° in phase — and both of those are at the extreme
    rim. Within ±3° of the chief ray, which is where a dense grating's orders
    actually sit, it is 2.1 %.

    The range is derived here rather than hard-coded, so a higher-NA or
    larger-chief-ray preset that pushes orders into the collapse region fails
    this test instead of quietly averaging through it.
    """
    from litho_sim.expose.m3d import MaskStack

    lam, NA, reduction, chief_deg = 13.5e-9, 0.33, 4.0, 6.0
    half = np.degrees(np.arcsin(NA / reduction))
    lo, hi = chief_deg - half, chief_deg + half
    assert (lo, hi) == pytest.approx((1.27, 10.73), abs=0.02)

    stack = MaskStack.for_wavelength(lam)
    worst_amp, worst_phase = 0.0, 0.0
    for deg in np.linspace(lo, hi, 25):
        th = np.radians(deg)
        rs = stack.blank_reflection(lam, th, "s")
        rp = stack.blank_reflection(lam, th, "p")
        worst_amp = max(worst_amp, abs(abs(rs) - abs(rp)) / abs(rs))
        worst_phase = max(worst_phase, abs(np.angle(rs) - np.angle(rp)))

    assert worst_amp < 0.08, (
        f"s/p amplitude split {worst_amp:.3%} across the collected orders is too "
        f"large for the scalar average"
    )
    assert np.degrees(worst_phase) < 8.0, (
        f"s/p phase split {np.degrees(worst_phase):.1f} deg across the collected "
        f"orders is too large for the scalar average"
    )

    # Most of the error is at the rim: near the chief ray, where a dense
    # grating's orders live, the split is a third of the worst case.
    near_axis = 0.0
    for deg in np.linspace(chief_deg - 3.0, chief_deg + 3.0, 13):
        th = np.radians(deg)
        rs = stack.blank_reflection(lam, th, "s")
        rp = stack.blank_reflection(lam, th, "p")
        near_axis = max(near_axis, abs(abs(rs) - abs(rp)) / abs(rs))
    assert near_axis < 0.03, f"near-axis s/p split {near_axis:.3%} unexpectedly large"

    # And the claim that it would *not* hold outside the pupil, so the
    # qualifier above is load-bearing rather than decorative.
    r_s13 = stack.blank_reflection(lam, np.radians(13.0), "s")
    r_p13 = stack.blank_reflection(lam, np.radians(13.0), "p")
    assert abs(abs(r_s13) - abs(r_p13)) / abs(r_s13) > 0.20


def test_the_reflection_table_matches_a_direct_transfer_matrix_solve(euv_grid):
    """Interpolating the mirror must not be where the accuracy goes.

    One transfer-matrix solve per order per source point would cost more than
    the Abbe sum it decorates, so the response is tabulated in angle and
    interpolated. The table is only legitimate if it reproduces the thing it
    replaces.
    """
    from litho_sim.expose.m3d import MaskStack
    from litho_sim.expose.m3d.provider import MultilayerSpectra

    optics = _euv_optics(mask_model="multilayer")
    stack = MaskStack.for_wavelength(optics.wavelength)
    spectra = MultilayerSpectra(
        np.ones((euv_grid.n_pixels, euv_grid.n_pixels)), optics, euv_grid, stack
    )

    for deg in (0.0, 2.5, 6.0, 9.3, 12.0):
        s = np.sin(np.radians(deg))
        got = spectra._reflection(np.array([s]))[0]
        direct = 0.5 * (
            stack.blank_reflection(optics.wavelength, np.radians(deg), "s")
            + stack.blank_reflection(optics.wavelength, np.radians(deg), "p")
        )
        assert abs(got - direct) < 2e-3 * abs(direct) + 1e-4, (
            f"table and direct solve disagree at {deg} deg: {got} vs {direct}"
        )


def test_evanescent_orders_carry_nothing(euv_grid):
    """An order past sin θ = 1 never leaves the mask, so it reflects nothing.

    Guards the clip in the angle map: without it ``arcsin`` would produce NaN
    and quietly poison the whole spectrum.
    """
    from litho_sim.expose.m3d import MaskStack
    from litho_sim.expose.m3d.provider import MultilayerSpectra

    optics = _euv_optics(mask_model="multilayer")
    spectra = MultilayerSpectra(
        np.ones((euv_grid.n_pixels, euv_grid.n_pixels)),
        optics, euv_grid, MaskStack.for_wavelength(optics.wavelength),
    )
    r = spectra._reflection(np.array([0.5, 1.0, 1.4, 12.0]))
    assert np.all(np.isfinite(r))
    assert r[2] == 0.0 and r[3] == 0.0
    assert abs(r[0]) > 0.1


def test_multilayer_refuses_a_transmissive_stack(euv_mask, euv_grid):
    """A DUV mask has no mirror, and saying so beats returning nonsense."""
    optics = _euv_optics(
        mask_model="multilayer",
        mask_stack={"regime": "duv_transmissive", "absorber": "Cr",
                    "absorber_thickness": 70e-9, "multilayer": None,
                    "substrate": "quartz"},
    )
    with pytest.raises(ValueError, match="duv_transmissive"):
        make_spectrum_provider(euv_mask, optics, euv_grid)


# ---------------------------------------------------------------------------
# The FDTD path
# ---------------------------------------------------------------------------


def test_mask_geometry_rejects_what_cannot_be_drawn():
    from litho_sim.expose.m3d.nearfield import MaskGeometry

    with pytest.raises(ValueError, match="smaller than pitch"):
        MaskGeometry(pitch=100e-9, cd=100e-9)
    with pytest.raises(ValueError, match="must be positive"):
        MaskGeometry(pitch=-1e-9, cd=10e-9)
    with pytest.raises(ValueError, match="orientation"):
        MaskGeometry(pitch=100e-9, cd=50e-9, orientation="diagonal")


def test_the_reticle_is_four_times_the_drawn_pattern():
    from litho_sim.expose.m3d.nearfield import MaskGeometry

    pitch, cd = MaskGeometry(pitch=64e-9, cd=32e-9).mask_side(4.0)
    assert (pitch, cd) == pytest.approx((256e-9, 128e-9))


def test_topography_puts_the_absorber_where_the_geometry_says():
    """The permittivity map is drawn from geometry, at solver resolution.

    Checks the two things that would silently ruin a near field: that the
    absorber is the requested thickness, and that its edges land at the drawn
    position rather than snapped to the imaging grid.
    """
    from litho_sim.expose.m3d.nearfield import MaskGeometry, build_topography
    from litho_sim.expose.m3d.stack import MASK_STACK_PRESETS, MaskStack

    grid = GridConfig(n_pixels=32, pixel_size=8e-9)
    optics = OpticsConfig(wavelength=193e-9, NA=0.93, reduction=4.0)
    stack = MaskStack(**MASK_STACK_PRESETS["DUV att-PSM 6%"])
    geo = MaskGeometry(pitch=128e-9, cd=64e-9)
    topo = build_topography(stack, geo, optics, grid)

    # The domain spans exactly the imaging field on the reticle, which is what
    # makes the solver's periodic cell and the imaging FFT's the same cell.
    width = topo.index.shape[1] * topo.dx
    assert width == pytest.approx(
        grid.n_pixels * grid.pixel_size * optics.reduction, rel=1e-9
    )

    # Absorber rows are the ones that vary across x.
    varies = np.ptp(np.abs(topo.index), axis=1) > 1e-9
    assert varies.any(), "no patterned rows — the absorber was never drawn"
    thickness = varies.sum() * topo.dz
    assert thickness == pytest.approx(stack.absorber_thickness, rel=0.15)

    # Duty cycle at mid-absorber matches the drawn CD.
    row = topo.index[np.flatnonzero(varies)[len(np.flatnonzero(varies)) // 2]]
    covered = (np.abs(row) > 1.5).mean()  # MoSi n=2.34, vacuum 1.0
    assert covered == pytest.approx(geo.cd / geo.pitch, abs=0.08)


def test_a_sub_pixel_edge_shift_moves_the_topography():
    """Edge position is continuous, which is why geometry is carried at all.

    A 1 nm change in CD is an eighth of an imaging pixel and would be invisible
    to anything built from the rasterised mask. The whole reason the solver
    takes geometry instead is that mask 3-D effects *are* edge effects.

    Note the direction: ``cd`` is the width of the **clear** line, so widening
    it makes the absorber *narrower*. Asserting the sign is the point — this
    test previously ran backwards and passed anyway, because every case it
    covered was at 50 % duty where the two widths coincide.
    """
    from litho_sim.expose.m3d.nearfield import MaskGeometry, build_topography
    from litho_sim.expose.m3d.stack import MASK_STACK_PRESETS, MaskStack

    grid = GridConfig(n_pixels=32, pixel_size=8e-9)
    optics = OpticsConfig(wavelength=193e-9, NA=0.93, reduction=4.0)
    stack = MaskStack(**MASK_STACK_PRESETS["DUV att-PSM 6%"])

    areas = []
    for cd in (64e-9, 65e-9, 66e-9):
        topo = build_topography(
            stack, MaskGeometry(pitch=128e-9, cd=cd), optics, grid
        )
        varies = np.ptp(np.abs(topo.index), axis=1) > 1e-9
        row = topo.index[np.flatnonzero(varies)[0]]
        areas.append(float(np.abs(row).sum()))

    assert areas[0] > areas[1] > areas[2], (
        f"widening the clear line by 1 nm must shrink the absorber, got {areas}"
    )


def test_a_sloped_sidewall_narrows_the_absorber_upward():
    from litho_sim.expose.m3d.nearfield import MaskGeometry, build_topography
    from litho_sim.expose.m3d.stack import MaskStack

    grid = GridConfig(n_pixels=32, pixel_size=8e-9)
    optics = OpticsConfig(wavelength=193e-9, NA=0.93, reduction=4.0)
    stack = MaskStack(
        regime="duv_transmissive", absorber="MoSi", absorber_thickness=80e-9,
        sidewall_deg=75.0, multilayer=None, substrate="quartz",
    )
    topo = build_topography(stack, MaskGeometry(pitch=160e-9, cd=80e-9), optics, grid)
    rows = np.flatnonzero(np.ptp(np.abs(topo.index), axis=1) > 1e-9)
    top = float((np.abs(topo.index[rows[0]]) > 1.5).sum())
    bottom = float((np.abs(topo.index[rows[-1]]) > 1.5).sum())
    assert top < bottom, (
        f"a 75 deg sidewall must leave the absorber narrower at the top, "
        f"got {top} cells vs {bottom}"
    )


def test_the_spectrum_lift_reproduces_the_analytic_grating():
    """The near field goes onto the FFT grid by truncation, not subsampling.

    An ideal square-wave transmission has known Fourier coefficients, so this
    checks the whole lift — band-limiting, the N² scaling, and the placement of
    a 1-D pattern's spectrum in row 0 — against something with a closed form.

    Subsampling instead of truncating would alias everything above the imaging
    Nyquist onto exactly the orders the pupil collects, and would still look
    plausible.
    """
    from litho_sim.expose.m3d.nearfield import spectrum_on_imaging_grid

    grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    n, width = grid.n_pixels, grid.n_pixels * grid.pixel_size
    pitch, cd = 64e-9, 32e-9
    duty = cd / pitch
    per_field = int(round(width / pitch))

    x = (np.arange(n * 8) + 0.5) * (width / (n * 8))
    d = np.abs((x + pitch / 2) % pitch - pitch / 2)
    got = spectrum_on_imaging_grid((d > cd / 2).astype(complex), grid, "vertical")

    def analytic(m: int) -> float:
        if m == 0:
            return 1.0 - duty
        u = m * duty
        return -duty * np.sin(np.pi * u) / (np.pi * u)

    for m in (0, 1, 3, -1):
        bin_ = (m * per_field) % n
        assert got[0, bin_].real == pytest.approx(analytic(m) * n * n, rel=0.01)

    assert np.allclose(got[1:, :], 0.0), (
        "a pattern invariant along y must put every coefficient in row 0"
    )


def test_fdtd_without_geometry_says_what_is_missing(mask, grid):
    """The error has to name the fix, because the fix is not guessable.

    A user reaching for a rigorous mask model will not expect to be asked for
    the pitch as well — the mask array is right there. The message says why it
    cannot be used.
    """
    optics = OpticsConfig(
        wavelength=193e-9, mask_model="fdtd",
        mask_stack={"regime": "duv_transmissive", "absorber": "MoSi",
                    "absorber_thickness": 71.9e-9, "multilayer": None,
                    "substrate": "quartz"},
    )
    with pytest.raises(ValueError, match="mask_geometry"):
        make_spectrum_provider(mask, optics, grid)


@pytest.mark.slow
def test_fdtd_changes_the_image_and_keeps_the_dose_scale_honest():
    """End to end: a rigorously solved att-PSM against the ideal thin one.

    Two claims. The image must actually *differ* — otherwise the whole
    apparatus is an expensive identity. And it must stay on the same dose
    scale, because ``normalisation="clear"`` means "a fraction of open-frame",
    and the open-frame value has to come from the same solve.

    That second one caught a real error: taking the clear reference from the
    transfer-matrix blank gives its *reflection*, which is right for an EUV
    mirror and meaningless for a mask you look through. It rescaled the image
    by 15x while leaving the contrast looking fine.
    """
    from litho_sim.expose.m3d.provider import clear_library_cache
    from litho_sim.mask.patterns import to_attenuated_psm

    grid = GridConfig(n_pixels=16, pixel_size=10e-9)
    pitch, cd = 160e-9, 80e-9
    binary = lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=pitch, cd=cd)
    psm = to_attenuated_psm(1.0 - binary, transmission=0.06, phase_shift_deg=180.0)

    base = dict(
        wavelength=193e-9, NA=0.93, sigma_outer=0.6, source_grid=5,
        reduction=4.0, normalisation="clear", m3d_angles=2,
        mask_stack={"regime": "duv_transmissive", "absorber": "MoSi",
                    "absorber_thickness": 71.9e-9, "multilayer": None,
                    "substrate": "quartz"},
        mask_geometry={"pitch": pitch, "cd": cd, "orientation": "vertical"},
    )
    clear_library_cache()
    thin = compute_aerial_image(psm, OpticsConfig(mask_model="thin", **base), grid)
    rig = compute_aerial_image(psm, OpticsConfig(mask_model="fdtd", **base), grid)

    assert np.isfinite(rig).all()
    assert not np.allclose(thin, rig, rtol=0.02), "the rigorous model changed nothing"
    # Same dose scale: peak intensities within a factor of two, not fifteen.
    assert 0.5 < rig.max() / thin.max() < 2.0, (
        f"clear-field reference is off: peaks {thin.max():.3f} vs {rig.max():.3f}"
    )


@pytest.mark.slow
def test_the_library_is_built_once_and_reused():
    """A process window calls this hundreds of times; it must not resolve.

    Without the cache, sweeping dose and focus would rebuild a full FDTD angle
    library per point, which turns a minute into a week.
    """
    from litho_sim.expose.m3d.provider import clear_library_cache, make_spectrum_provider

    grid = GridConfig(n_pixels=16, pixel_size=10e-9)
    optics = OpticsConfig(
        wavelength=193e-9, NA=0.93, mask_model="fdtd", m3d_angles=2, reduction=4.0,
        mask_stack={"regime": "duv_transmissive", "absorber": "MoSi",
                    "absorber_thickness": 71.9e-9, "multilayer": None,
                    "substrate": "quartz"},
        mask_geometry={"pitch": 160e-9, "cd": 80e-9, "orientation": "vertical"},
    )
    m = lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=160e-9, cd=80e-9)
    clear_library_cache()
    first = make_spectrum_provider(m, optics, grid)
    second = make_spectrum_provider(m, optics, grid)
    assert first is second, "the near-field library was rebuilt"

    # A different defocus is the same mask, so it must hit the cache too.
    third = make_spectrum_provider(
        m, dataclasses.replace(optics, defocus=50e-9), grid
    )
    assert third is first


def _cacheable_fdtd_case():
    """A small transmissive case the library tests can afford to solve."""
    grid = GridConfig(n_pixels=16, pixel_size=10e-9)
    optics = OpticsConfig(
        wavelength=193e-9, NA=0.93, mask_model="fdtd", m3d_angles=2, reduction=4.0,
        mask_stack={"regime": "duv_transmissive", "absorber": "MoSi",
                    "absorber_thickness": 71.9e-9, "multilayer": None,
                    "substrate": "quartz"},
        mask_geometry={"pitch": 160e-9, "cd": 80e-9, "orientation": "vertical"},
    )
    mask = lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=160e-9, cd=80e-9)
    return mask, optics, grid


@pytest.mark.slow
def test_the_pool_solves_the_same_library_as_one_process():
    """Parallelism is a scheduling change, not a physics one.

    The angle solves share no state, so a pooled sweep must agree with a serial
    one exactly — not approximately. Anything else means a solve is picking up
    something from its process.
    """
    from litho_sim.expose.m3d.nearfield import MaskGeometry, build_library

    _, optics, grid = _cacheable_fdtd_case()
    stack = MaskStack.from_spec(optics.mask_stack)
    geom = MaskGeometry(**optics.mask_geometry)

    serial, _ = build_library(stack, geom, optics, grid, n_angles=2, workers=1)
    pooled, _ = build_library(stack, geom, optics, grid, n_angles=2, workers=4)

    assert np.array_equal(serial.fields, pooled.fields)
    assert np.array_equal(serial.blank, pooled.blank)


@pytest.mark.slow
def test_a_solved_library_survives_the_process_that_solved_it():
    """The cache has to outlive the interpreter or it does not help the app.

    Turning the rigorous model on costs minutes. If that were paid again on
    every launch nobody would turn it on twice, so the library is written to
    disk and keyed on the configuration that produced it.
    """
    from litho_sim.expose.m3d import provider

    mask, optics, grid = _cacheable_fdtd_case()
    provider.clear_library_cache()
    built = provider.make_spectrum_provider(mask, optics, grid)

    files = list(provider.LIBRARY_CACHE_DIR.glob("m3d-v*.npz"))
    assert len(files) == 1, f"expected one cached library, found {files}"

    # Drop the in-memory tier only: this is the state a fresh launch is in.
    provider.clear_library_cache()
    restored = provider.make_spectrum_provider(mask, optics, grid)

    assert restored is not built, "the in-memory cache was not actually cleared"
    assert np.array_equal(built._library.fields, restored._library.fields)
    assert np.array_equal(built._library.blank, restored._library.blank)
    # And the app's cost model must now call this configuration cheap, or the
    # scheduler would keep deferring a change that costs milliseconds.
    assert provider.library_is_cached(optics, grid)


def test_a_stale_cache_file_is_ignored_rather_than_trusted():
    """A library that cannot be read is a rebuild, not a crash.

    Interrupted writes and numpy upgrades both produce files that load badly,
    and a half-read near field would print a plausible wrong image.
    """
    from litho_sim.expose.m3d import provider

    _, optics, grid = _cacheable_fdtd_case()
    key = provider._library_key(optics, grid, optics.mask_stack, optics.mask_geometry)
    path = provider.LIBRARY_CACHE_DIR / provider._disk_name(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not an npz")

    assert provider._load_library(path) is None


def test_the_cache_key_is_stable_across_interpreters():
    """Keyed on the configuration, so the next launch finds its own file.

    ``hash()`` of a str is salted per process, so a key built that way would
    miss on exactly the run the disk cache exists to serve. Recomputing the
    name in a *subprocess* is the only check that actually catches it.
    """
    import subprocess

    _, optics, grid = _cacheable_fdtd_case()
    from litho_sim.expose.m3d import provider

    key = provider._library_key(optics, grid, optics.mask_stack, optics.mask_geometry)
    here = provider._disk_name(key)

    script = (
        "from litho_sim.expose.m3d import provider\n"
        "from tests.test_m3d import _cacheable_fdtd_case\n"
        "_, o, g = _cacheable_fdtd_case()\n"
        "k = provider._library_key(o, g, o.mask_stack, o.mask_geometry)\n"
        "print(provider._disk_name(k))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, check=True,
        # A different hash seed is the whole point of the test.
        env={"PYTHONHASHSEED": "12345", "PATH": os.environ.get("PATH", "")},
    )
    assert out.stdout.strip() == here


# ---------------------------------------------------------------------------
# EUV: the reflective path
# ---------------------------------------------------------------------------


def test_the_grid_is_chosen_to_divide_the_bragg_period():
    """A quantised multilayer period is a detuned mirror, so it is not allowed.

    Measured during development: at an unconstrained ``dz`` of 0.562 nm the
    6.94 nm period rounds to 6.744 nm — 2.8 % — and a 12-bilayer stack's
    reflectance falls from 46.3 % to 28.2 %. The solver reproduced that 28.2 %
    faithfully, which is exactly the point: the error was in the structure it
    was handed, not in the physics.

    So the cell height divides the period by construction.
    """
    from litho_sim.expose.m3d.multilayer import Multilayer
    from litho_sim.expose.m3d.nearfield import _dz_for_multilayer

    ml = Multilayer()
    dz = _dz_for_multilayer(ml, 13.5e-9 / 24.0)
    k = round(ml.period / dz)
    assert abs(k * dz - ml.period) / ml.period < 1e-9, "period must be exact"
    # And the Mo/Si split lands on whole cells too.
    assert round(ml.gamma * k) / k == pytest.approx(ml.gamma, abs=0.02)
    assert dz <= 13.5e-9 / 24.0, "must not be coarser than the accuracy target"


def test_the_euv_monitor_reads_reflection_not_transmission():
    """Nothing transmits at 13.5 nm, so the near field is the one coming back.

    The monitor sits *above* the injection plane, where the stored field is
    purely scattered — the reflected near field, already separated from the
    illumination without a subtraction.
    """
    from litho_sim.expose.m3d.multilayer import Multilayer
    from litho_sim.expose.m3d.nearfield import MaskGeometry, build_topography

    grid = GridConfig(n_pixels=16, pixel_size=8e-9)
    optics = OpticsConfig(wavelength=13.5e-9, NA=0.33, reduction=4.0, chief_ray_deg=6.0)
    stack = MaskStack(
        regime="euv_reflective", absorber="TaBN", absorber_thickness=60e-9,
        multilayer=Multilayer(n_bilayers=8), substrate="SiO2",
    )
    topo = build_topography(stack, MaskGeometry(pitch=64e-9, cd=32e-9), optics, grid)
    assert topo.monitor_row < topo.source_row, (
        "an EUV mask is read in reflection, so the monitor belongs above the source"
    )


@pytest.mark.slow
def test_the_euv_blank_reflects_what_the_transfer_matrix_says():
    """The rigorous mirror against the exact one — the EUV validation.

    With the pattern removed, an EUV mask is a planar Bragg stack, and
    :mod:`litho_sim.coat.films` solves that exactly. Anything the FDTD gets
    wrong about injection, the Bloch boundary at 6 degrees, the absorbing
    layers, or the grid shows up as a reflectance that does not match.

    The tolerance is grid resolution, not fudge: at dz around lambda/29 the
    solver lands within a few percent, and it converges from there.
    """
    from litho_sim.expose.m3d.multilayer import Multilayer
    from litho_sim.expose.m3d.nearfield import (
        MaskGeometry,
        _blank_topography,
        build_topography,
        near_field_coefficients,
    )

    lam = 13.5e-9
    theta = np.radians(6.0)
    ml = Multilayer(n_bilayers=12)
    grid = GridConfig(n_pixels=16, pixel_size=8e-9)
    optics = OpticsConfig(wavelength=lam, NA=0.33, reduction=4.0, chief_ray_deg=6.0)
    stack = MaskStack(
        regime="euv_reflective", absorber="TaBN", absorber_thickness=60e-9,
        multilayer=ml, substrate="SiO2",
    )
    topo = build_topography(stack, MaskGeometry(pitch=64e-9, cd=32e-9), optics, grid)
    blank = _blank_topography(topo, stack, optics, 16)
    coeff, ref = near_field_coefficients(topo, blank, optics, np.sin(theta), 0.0)

    expected = ml.film_stack(lam).reflectance(lam, theta, "s")
    assert abs(ref) ** 2 == pytest.approx(expected, rel=0.12)

    # And the pattern sits on top of it sensibly: open trenches return the
    # blank (that is what blank-normalisation means), and 60 nm of TaBN
    # double-passed knocks the amplitude down by roughly a factor of five.
    #
    # The regions are located with the solver's own phasing rather than a
    # hand-rolled one — getting that wrong is the bug this convention exists
    # to prevent, and a test that repeats it would hide it.
    from litho_sim.expose.m3d.nearfield import absorber_centre

    m_pitch, m_cd = MaskGeometry(pitch=64e-9, cd=32e-9).mask_side(optics.reduction)
    coords = (np.arange(coeff.size) - coeff.size // 2) * topo.dx
    offset = coords - absorber_centre(m_pitch, m_cd)
    d = np.abs((offset + m_pitch / 2) % m_pitch - m_pitch / 2)
    half = (m_pitch - m_cd) / 2.0
    assert np.abs(coeff[d > half * 1.4]).mean() == pytest.approx(1.0, abs=0.1)
    assert np.abs(coeff[d < half * 0.5]).mean() < 0.35


# ---------------------------------------------------------------------------
# Topography vs the mask the thin model uses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pitch_nm, clear_nm",
    [(160, 80), (160, 100), (200, 60), (200, 140), (120, 40)],
)
def test_the_absorber_lands_where_lines_and_spaces_puts_it(pitch_nm, clear_nm):
    """The solved topography and the drawn mask must be the same mask.

    Two things about ``lines_and_spaces`` are easy to get backwards, and both
    were, until a non-50 % duty cycle exposed them:

    * its ``cd`` is the width of the **clear** line, so the absorber is
      ``pitch - cd`` — identical at 50 % duty, wrong everywhere else;
    * it phases from the *field centre*, putting the absorber at ``cd/2``
      rather than at zero.

    The second is the dangerous one. A rigid offset between the topography and
    the mask array is an image shift, and it would appear in exactly the
    thin-versus-rigorous comparison the model exists to make — 40 nm of
    artefact on top of about 2 nm of real placement error.

    Compared against the *continuous* pattern rather than the rasterised one,
    because a coarse raster snaps edges to pixels and that is precisely what
    building from geometry is meant to avoid.
    """
    from litho_sim.expose.m3d.nearfield import (
        MaskGeometry,
        _absorber_occupancy,
        absorber_centre,
    )

    pitch, cd = pitch_nm * 1e-9, clear_nm * 1e-9

    # The continuous truth, straight from the pattern generator's own formula.
    x = np.linspace(-pitch, pitch, 200001)
    clear = np.mod(x + pitch / 2.0, pitch) < cd
    occupancy = _absorber_occupancy(x, pitch, cd, 1e-15)

    assert occupancy.mean() == pytest.approx(1.0 - clear.mean(), abs=1e-3), (
        "absorber duty must be pitch - cd, not cd"
    )
    # Away from the edges themselves, where a razor-sharp occupancy sits at
    # exactly 0.5 and either answer is defensible.
    interior = (occupancy < 0.01) | (occupancy > 0.99)
    assert np.array_equal(occupancy[interior] > 0.5, ~clear[interior]), (
        "the absorber must occupy exactly the region the mask leaves dark"
    )

    # And its centre is at cd/2, which is what pins the phasing.
    centre = absorber_centre(pitch, cd)
    assert centre == pytest.approx(cd / 2.0)
    assert _absorber_occupancy(np.array([centre]), pitch, cd, 1e-15)[0] == 1.0

    _ = MaskGeometry(pitch=pitch, cd=cd)  # the pair must also be constructible


def test_a_narrow_absorber_is_narrow_in_the_permittivity_map():
    """End to end through build_topography, at a duty cycle that would expose it."""
    from litho_sim.expose.m3d.nearfield import MaskGeometry, build_topography
    from litho_sim.expose.m3d.stack import MASK_STACK_PRESETS

    grid = GridConfig(n_pixels=32, pixel_size=10e-9)
    optics = OpticsConfig(wavelength=193e-9, NA=0.93, reduction=4.0)
    stack = MaskStack(**MASK_STACK_PRESETS["DUV att-PSM 6%"])

    # 160 nm pitch with a 120 nm clear line leaves only 40 nm of absorber.
    topo = build_topography(
        stack, MaskGeometry(pitch=160e-9, cd=120e-9), optics, grid
    )
    rows = np.flatnonzero(np.ptp(np.abs(topo.index), axis=1) > 1e-9)
    row = topo.index[rows[len(rows) // 2]]
    duty = (np.abs(row) > 1.9).mean()  # MoSi |n| = 2.41, vacuum 1.0
    assert duty == pytest.approx(40.0 / 160.0, abs=0.05), (
        f"absorber duty {duty:.3f}; treating cd as the absorber width would "
        f"give {120 / 160:.3f}"
    )
