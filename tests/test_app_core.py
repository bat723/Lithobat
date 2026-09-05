"""
Tests for the application's engine-side layer.

Everything here runs without a display and without Qt, which is the point of
splitting the app that way: the parts that decide *what* to simulate and
*when* are the parts worth testing, and they are testable only if they do not
drag a widget toolkit in with them.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from litho_sim.app.compute import (
    build_mask,
    compute_imaging,
    compute_profile_3d,
    estimate_cost_ms,
)
from litho_sim.app.params import (
    DEVELOP3D_ONLY,
    GROUPS,
    LATENT3D_IGNORES,
    PATTERNS,
    SPECS,
    SPECS_BY_KEY,
    STAGES,
    ParameterModel,
    ParamSpec,
)
from litho_sim.app.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Parameter specs
# ---------------------------------------------------------------------------


def test_every_spec_is_well_formed():
    for s in SPECS:
        assert s.label, f"{s.key} has no label"
        assert s.help, f"{s.key} has no tooltip — every knob should explain itself"
        assert s.group in GROUPS, f"{s.key} is in unknown group '{s.group}'"
        assert s.target in ("optics", "resist", "grid", "mask", "view"), s.key


def test_spec_keys_are_unique():
    keys = [s.key for s in SPECS]
    assert len(keys) == len(set(keys))


def test_defaults_are_inside_their_own_ranges():
    """A default outside its slider is a control that jumps on first touch."""
    for s in SPECS:
        if s.kind in ("float", "int"):
            assert s.lo <= s.default <= s.hi, f"{s.key}={s.default} outside [{s.lo}, {s.hi}]"
        elif s.kind == "choice":
            assert s.default in s.choices, s.key


def test_malformed_specs_are_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        ParamSpec("k", "K", "colour", 1.0)
    with pytest.raises(ValueError, match="lists no choices"):
        ParamSpec("k", "K", "choice", "a")
    with pytest.raises(ValueError, match="hi <= lo"):
        ParamSpec("k", "K", "float", 1.0, lo=2.0, hi=1.0)


def test_values_are_clamped_not_rejected():
    """A slider cannot leave its track, so out-of-range means clamp."""
    p = ParameterModel()
    assert p.set("NA", 99.0) == SPECS_BY_KEY["NA"].hi
    assert p.set("NA", -5.0) == SPECS_BY_KEY["NA"].lo
    assert p.set("n_pixels", 100) == 100  # ints round, they do not truncate


def test_choices_are_rejected_rather_than_clamped():
    p = ParameterModel()
    with pytest.raises(ValueError, match="must be one of"):
        p.set("polarisation", "circular")


def test_unknown_parameter_is_an_error():
    with pytest.raises(KeyError, match="unknown parameter"):
        ParameterModel().set("nonexistent", 1)


# ---------------------------------------------------------------------------
# Assembling engine configs
# ---------------------------------------------------------------------------


def test_display_units_are_converted_to_si():
    """Sliders are in nanometres; the engine only ever sees metres."""
    p = ParameterModel()
    p.set("pitch", 200.0)
    p.set("defocus", -100.0)
    assert p.si("pitch") == pytest.approx(200e-9)
    assert p.optics().defocus == pytest.approx(-100e-9)
    assert p.grid().pixel_size == pytest.approx(4e-9)


def test_defaults_reproduce_the_engine_defaults():
    """The app must not quietly start from somewhere else."""
    from litho_sim.core.config import OpticsConfig

    o = ParameterModel().optics()
    d = OpticsConfig()
    for fieldname in ("NA", "n_immersion", "sigma_outer", "source_type",
                      "source_grid", "imaging_model", "polarisation",
                      "normalisation", "exact_defocus"):
        assert getattr(o, fieldname) == getattr(d, fieldname), fieldname
    # Wavelength goes through a nm→m conversion, so compare as a float.
    assert o.wavelength == pytest.approx(d.wavelength)


def test_sigma_inner_cannot_overtake_sigma_outer():
    """The source builders raise if it does, so the UI must not allow it.

    Two independent sliders can always be dragged into an invalid pair; the
    model has to be the thing that stops it, not the user's care.
    """
    p = ParameterModel()
    p.set("sigma_outer", 0.40)
    p.set("sigma_inner", 0.90)
    o = p.optics()
    assert o.sigma_inner < o.sigma_outer
    assert o.sigma_inner >= 0.0


def test_na_cannot_exceed_the_immersion_index():
    """Two independent sliders can always be dragged into an impossible pair.

    NA = n·sinθ, so a dry lens cannot have NA 1.3. The engine clips sinθ
    internally and carries on, which produces plausible-looking nonsense —
    found by actually driving the app, where NA 1.30 in air reported
    sinθ = 1.300 and a CD of 4 nm.
    """
    p = ParameterModel()
    p.set("NA", 1.30)                 # left dry on purpose
    assert p.optics().sin_theta_max <= 1.0
    assert p.optics().NA == pytest.approx(1.0)

    p.set("n_immersion", 1.44)        # now it is reachable
    assert p.optics().NA == pytest.approx(1.30)


def test_dipole_gets_its_axis():
    p = ParameterModel()
    p.set("source_type", "dipole")
    assert p.optics().source_kwargs == {"axis": "x"}
    p.set("source_type", "conventional")
    assert p.optics().source_kwargs == {}


def test_signature_changes_only_when_a_value_does():
    p = ParameterModel()
    before = p.signature()
    p.set("NA", p["NA"])            # same value
    assert p.signature() == before
    p.set("NA", p["NA"] + 0.05)
    assert p.signature() != before


# ---------------------------------------------------------------------------
# Computing
# ---------------------------------------------------------------------------


@pytest.fixture
def small():
    """A cheap configuration that is well sampled *and* actually prints.

    128 px at 4 nm is a 512 nm field, so a 256 nm pitch fits exactly twice —
    the FFT wraps the field, and a fractional number of periods images its own
    seam. The 128 nm feature sits at k₁ ≈ 0.62, comfortably resolved: pushed
    much denser the resist simply clears everywhere and every CD reads zero,
    which makes for tests that pass by accident.

    Function-scoped on purpose. Several tests below move knobs, and a shared
    model would make them order-dependent.
    """
    p = ParameterModel()
    p.set("n_pixels", 128)
    p.set("pixel_size", 4.0)
    p.set("pitch", 256.0)
    p.set("cd", 128.0)
    p.set("source_grid", 7)
    return p


@pytest.mark.parametrize("pattern", PATTERNS)
def test_every_pattern_builds_and_images(small, pattern):
    """Each entry in the pattern menu must actually produce an image."""
    n = small.grid().n_pixels
    small.set("pattern", pattern)
    mask = build_mask(small)
    assert mask.shape == (n, n)
    assert np.isfinite(mask).all()
    r = compute_imaging(small)
    assert np.isfinite(r.aerial).all()
    small.set("pattern", "lines and spaces")


def test_result_carries_everything_the_panel_draws(small):
    r = compute_imaging(small)
    n = small.grid().n_pixels
    assert r.mask.shape == r.aerial.shape == r.resist.shape == (n, n)
    assert r.cut_aerial.shape == r.cut_resist.shape == r.x_nm.shape == (n,)
    assert r.elapsed_ms > 0
    assert "CD" in r.summary and "NILS" in r.summary


def test_result_signature_matches_the_parameters_it_came_from(small):
    r = compute_imaging(small)
    assert r.signature == small.signature()


def test_result_carries_the_threshold_it_was_computed_with(small):
    """So a view can draw the threshold line that belongs to *this* image.

    An expensive image can land after the slider has moved on; drawing the
    current threshold over an older image states something untrue about it.
    """
    small.set("threshold", 0.42)
    r = compute_imaging(small)
    small.set("threshold", 0.77)          # moved on since
    assert r.threshold == pytest.approx(0.42)


def test_threshold_moves_the_printed_cd(small):
    """The calibration knob has to do something, or the panel is lying.

    This is the regression class the legacy app shipped twice: a slider that
    changed only a cache key.
    """
    small.set("threshold", 0.25)
    lo = compute_imaging(small).cd_nm
    small.set("threshold", 0.60)
    hi = compute_imaging(small).cd_nm
    assert lo > 0 and hi > 0, "neither threshold printed anything"
    assert hi > lo, f"threshold 0.60 gave {hi:.1f} nm, 0.25 gave {lo:.1f} nm"


def test_peb_diffusion_actually_moves_the_printed_cd(small):
    """The control has to do something, or it is lying to the user.

    The engine's threshold model works in intensity space and deliberately
    bypasses diffusion, so before this the PEB slider was inert — the
    developed resist was bit-identical at 0, 20 and 60 nm. That is the exact
    regression class the vault already records once ("the PEB slider fed a
    config the resist step ignored"), which is why it gets a test rather than
    a comment.
    """
    small.set("diffusion_sigma", 0.0)
    sharp = compute_imaging(small).cd_nm
    small.set("diffusion_sigma", 20.0)
    diffused = compute_imaging(small).cd_nm
    assert sharp > 0
    assert diffused < sharp, (
        f"diffusion should shrink the printed feature: {diffused:.1f} nm "
        f"against {sharp:.1f} nm sharp"
    )


def test_zero_diffusion_reproduces_the_engine_exactly(small):
    """The app's default must not quietly disagree with the CLI.

    apply_peb returns a copy below a thousandth of a pixel, so at the default
    sigma of 0 the app's pipeline is the engine's pipeline.
    """
    from litho_sim.develop import simulate_resist
    from litho_sim.expose import compute_aerial_image

    small.set("diffusion_sigma", 0.0)
    r = compute_imaging(small)

    aerial = compute_aerial_image(
        build_mask(small), small.optics(), small.grid(), dose=small.dose
    )
    _, _, resist = simulate_resist(
        aerial, small.resist(), small.grid(), model="threshold"
    )
    assert np.array_equal(r.resist, resist)


def test_peb_defaults_to_off():
    """So the app and the command line agree out of the box."""
    assert ParameterModel()["diffusion_sigma"] == 0.0


def test_vector_model_reaches_the_engine():
    """Switching the control must change the image, not just the config.

    Uses x-dipole illumination on dense lines: with the poles in the plane of
    diffraction, radial polarisation is pure TM and azimuthal is pure TE, so
    the split is unambiguous. Under a conventional disc the two sit within a
    percent of each other on a well-resolved pattern, which would make this a
    coin toss rather than a test.
    """
    p = ParameterModel()
    p.set("n_pixels", 128)
    p.set("pitch", 100.0)
    p.set("cd", 50.0)
    p.set("NA", 1.35)
    p.set("n_immersion", 1.44)
    p.set("n_image", 1.70)
    p.set("source_type", "dipole")
    p.set("sigma_outer", 0.90)
    p.set("sigma_inner", 0.60)
    p.set("source_grid", 11)
    p.set("imaging_model", "vector")

    p.set("polarisation", "te")
    te = compute_imaging(p).contrast
    p.set("polarisation", "tm")
    tm = compute_imaging(p).contrast
    assert te > 1.3 * tm, f"TE {te:.3f} should clearly beat TM {tm:.3f}"


def test_cost_estimate_tracks_measured_time(small):
    """The estimate decides what runs live, so it must not be fantasy.

    Within a factor of two is plenty — it only has to sort cheap from
    expensive, and it must never claim something slow is fast.
    """
    r = compute_imaging(small)
    est = estimate_cost_ms(small)
    assert 0.4 * r.elapsed_ms < est < 3.0 * r.elapsed_ms, (
        f"estimate {est:.0f} ms vs measured {r.elapsed_ms:.0f} ms"
    )


def test_cost_estimate_knows_vector_is_dearer():
    p = ParameterModel()
    scalar = estimate_cost_ms(p)
    p.set("imaging_model", "vector")
    p.set("polarisation", "te")
    polarised = estimate_cost_ms(p)
    p.set("polarisation", "unpolarised")
    unpolarised = estimate_cost_ms(p)
    assert polarised == pytest.approx(3 * scalar)
    assert unpolarised == pytest.approx(6 * scalar)


# ---------------------------------------------------------------------------
# Staged caching
# ---------------------------------------------------------------------------


def test_every_spec_declares_a_stage():
    for s in SPECS:
        assert s.stage in STAGES, f"{s.key} has stage '{s.stage}'"


def test_bad_stage_is_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        ParamSpec("k", "K", "float", 0.5, stage="develop")


def test_stage_signature_ignores_later_stages():
    """A stage must not notice knobs it cannot possibly depend on.

    This is the whole mechanism: if the aerial signature moved when the
    threshold did, the cache would miss and the Abbe sum would run anyway.
    """
    p = ParameterModel()
    before = p.stage_signature("aerial")
    p.set("threshold", 0.77)     # resist stage
    p.set("dose", 2.0)           # scale stage
    assert p.stage_signature("aerial") == before
    p.set("NA", 1.1)             # aerial stage
    assert p.stage_signature("aerial") != before


def test_stage_signature_includes_earlier_stages():
    """The pipeline is sequential, so a stage does depend on its ancestors."""
    p = ParameterModel()
    before = p.stage_signature("resist")
    p.set("pitch", 300.0)        # mask stage, upstream of resist
    assert p.stage_signature("resist") != before


@pytest.mark.parametrize(
    "key, value, expect_rerun",
    [
        ("threshold", 0.55, False),
        ("tone", "negative", False),
        ("dose", 1.7, False),
        ("normalisation", "clear", False),
        ("diffusion_sigma", 15.0, False),
        ("NA", 1.10, True),
        ("source_grid", 15, True),
        ("defocus", 100.0, True),
        ("pitch", 260.0, True),
    ],
)
def test_only_upstream_changes_rerun_the_abbe_sum(key, value, expect_rerun):
    """The measured claim, asserted: six of the knobs are effectively free.

    The Abbe sum is 98 % of the pipeline, so what matters is not how fast a
    stage is but whether a given control can avoid triggering it at all.
    """
    p = ParameterModel()
    p.set("n_pixels", 64)
    p.set("source_grid", 11)
    pipe = Pipeline()
    compute_imaging(p, pipe)
    before = pipe.runs["aerial"]

    p.set(key, value)
    compute_imaging(p, pipe)

    after = pipe.runs["aerial"]
    if expect_rerun:
        assert after == before + 1, f"'{key}' should have rebuilt the aerial image"
    else:
        assert after == before, f"'{key}' rebuilt the aerial image unnecessarily"


@pytest.mark.parametrize("key, value", [
    ("dose", 1.6), ("normalisation", "none"), ("threshold", 0.5),
])
def test_cached_results_are_identical_to_fresh_ones(key, value):
    """Caching must be invisible in the output, or it is just a bug factory."""
    p = ParameterModel()
    p.set("n_pixels", 64)
    p.set("source_grid", 11)
    pipe = Pipeline()
    compute_imaging(p, pipe)          # prime

    p.set(key, value)
    cached = compute_imaging(p, pipe)
    fresh = compute_imaging(p)        # no pipeline: everything recomputed
    assert np.array_equal(cached.aerial, fresh.aerial)
    assert np.array_equal(cached.resist, fresh.resist)


def test_dose_and_normalisation_are_exactly_a_rescale():
    """The identity the cache relies on, checked against the engine.

    Dose is applied once, after normalisation, so the finished image is the
    raw Abbe sum times a single constant. If that ever stops holding, the
    cached `scale` stage silently produces wrong images.
    """
    from litho_sim.expose.aerial_image import (
        compute_aerial_image,
        normalisation_scale,
    )

    p = ParameterModel()
    p.set("n_pixels", 64)
    p.set("source_grid", 11)
    grid, optics = p.grid(), p.optics()
    mask = build_mask(p)

    raw, peak, clear = compute_aerial_image(mask, optics, grid, return_raw=True)
    for mode in ("peak", "clear", "none"):
        for dose in (0.4, 1.0, 2.5):
            o = dataclasses.replace(optics, normalisation=mode)
            direct = compute_aerial_image(mask, o, grid, dose=dose)
            assert np.array_equal(
                direct, raw * normalisation_scale(mode, peak, clear, dose)
            ), f"{mode} @ dose {dose}"


def test_pipeline_invalidate_forces_a_rerun():
    p = ParameterModel()
    p.set("n_pixels", 64)
    p.set("source_grid", 11)
    pipe = Pipeline()
    compute_imaging(p, pipe)
    assert pipe.runs["aerial"] == 1
    compute_imaging(p, pipe)
    assert pipe.runs["aerial"] == 1, "unchanged params should hit the cache"
    pipe.invalidate()
    compute_imaging(p, pipe)
    assert pipe.runs["aerial"] == 2


# ---------------------------------------------------------------------------
# The 3-D profile
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny3d():
    """Small enough to develop in depth quickly, big enough to print."""
    p = ParameterModel()
    p.set("n_pixels", 48)
    p.set("source_grid", 7)
    p.set("pitch", 192.0)
    p.set("cd", 96.0)
    p.set("n_z_slices", 5)
    p.set("thickness", 60.0)
    return p


def test_profile_carries_the_metrics_and_the_cuts(tiny3d):
    r = compute_profile_3d(tiny3d)
    nz, ny, nx = r.remaining.shape
    assert r.cut_remaining.shape == (nz, nx)
    assert r.cut_latent.shape == (nz, nx)
    assert r.x_nm.shape == (nx,)
    assert 0.0 <= r.film_remaining_pct <= 100.0
    assert np.isfinite(r.top_loss_nm) or r.cleared
    assert "remaining" in r.summary and "sidewall" in r.summary
    assert str(int(tiny3d["thickness"])) in r.label


def test_profile_cuts_are_taken_from_the_volume(tiny3d):
    """Sliced on the worker so the GUI never indexes a 20 MB array."""
    r = compute_profile_3d(tiny3d)
    assert np.array_equal(r.cut_remaining, r.remaining[:, r.row, :])


def test_a_sealed_film_says_so_rather_than_drawing_nothing():
    """Degenerate outcomes are the ones that look like bugs, so they carry a
    flag and an explanation instead of an empty picture."""
    from litho_sim.app.compute import Profile3DResult

    full = np.ones((8, 8, 8), dtype=bool)
    r = Profile3DResult(
        remaining=full, cut_remaining=full[:, 4, :],
        cut_latent=np.zeros((8, 8)), x_nm=np.arange(8.0), height_nm=16.0,
        row=4, sidewall_deg=float("nan"), film_remaining_pct=100.0,
        top_loss_nm=0.0, sealed=True, cleared=False, label="x",
        elapsed_ms=1.0, signature=(),
    )
    assert "did not develop" in r.diagnosis
    assert r.diagnosis != ""


def test_3d_knobs_do_not_disturb_the_2d_image():
    """The 3-D branch hangs off the mask, not off the 2-D aerial image, so a
    thickness or plane-count change must leave the imaging cache alone."""
    p = ParameterModel()
    p.set("n_pixels", 64)
    p.set("source_grid", 7)
    pipe = Pipeline()
    compute_imaging(p, pipe)
    before = pipe.runs["aerial"]

    for key, value in (("thickness", 200.0), ("n_z_slices", 21),
                       ("develop_model", "mack"), ("standing_waves", True),
                       ("z_exaggeration", 4.0)):
        p.set(key, value)
        compute_imaging(p, pipe)

    assert pipe.runs["aerial"] == before, "3-D and view knobs rebuilt the aerial image"


def test_view_knobs_are_not_physics():
    """z-exaggeration must invalidate nothing at all."""
    assert ParameterModel.stages_invalidated_by("z_exaggeration") == ()
    assert ParameterModel.stages_invalidated_by("thickness") == ("profile3d",)
    assert "aerial" not in ParameterModel.stages_invalidated_by("thickness")


# ---------------------------------------------------------------------------
# The Qt boundary
# ---------------------------------------------------------------------------


def test_the_core_never_imports_qt():
    """The split is only worth having if it holds.

    Everything tested in this file must stay importable on a machine with no
    Qt and no display — that is what makes the logic testable at all. If
    PySide6 leaks into params/compute/pipeline, this fails.
    """
    import litho_sim.app.compute as c
    import litho_sim.app.fem as f
    import litho_sim.app.params as p
    import litho_sim.app.pipeline as pl
    import litho_sim.app.stochastics as st

    for module in (p, c, pl, f, st):
        src = Path(module.__file__).read_text(encoding="utf-8")
        assert "PySide6" not in src, f"{module.__name__} imports Qt"
        assert "QtWidgets" not in src, f"{module.__name__} imports Qt"


def _qt_binding_available() -> bool:
    import importlib.util

    return any(
        importlib.util.find_spec(m) is not None
        for m in ("PySide6", "PyQt6", "PySide2", "PyQt5")
    )


def test_missing_qt_explains_itself():
    """Without a Qt binding the app module must say what to install, not
    raise a bare ImportError about a module nobody asked for."""
    if _qt_binding_available():
        pytest.skip("a Qt binding is installed, so the guard cannot fire")

    with pytest.raises(ImportError, match="pip install PySide6"):
        import litho_sim.app.main  # noqa: F401


# ---------------------------------------------------------------------------
# The 3-D latent cache
# ---------------------------------------------------------------------------


def _small_3d_model():
    m = ParameterModel()
    m.set("n_pixels", 64)
    m.set("n_z_slices", 3)
    return m


def test_latent3d_signature_ignores_develop_only_knobs():
    """Nothing the develop step reads may enter the latent image."""
    m = _small_3d_model()
    before = m.latent3d_signature()
    for key in LATENT3D_IGNORES:
        spec = SPECS_BY_KEY[key]
        m.set(key, spec.choices[-1] if spec.kind == "choice"
              else (not m[key]) if spec.kind == "bool" else m[key] + spec.step)
    assert m.latent3d_signature() == before


def test_latent3d_signature_tracks_everything_else():
    """A physical knob the latent depends on must change the signature."""
    for key in ("NA", "pitch", "dose", "diffusion_sigma", "thickness",
                "standing_waves", "focus_reference"):
        m = _small_3d_model()
        before = m.latent3d_signature()
        spec = SPECS_BY_KEY[key]
        m.set(key, spec.choices[-1] if spec.kind == "choice"
              else (not m[key]) if spec.kind == "bool" else m[key] + spec.step)
        assert m.latent3d_signature() != before, key


def test_develop_only_change_reuses_the_latent():
    """The 730 ms path runs once; a threshold change costs only develop_3d."""
    m = _small_3d_model()
    pipeline = Pipeline()
    compute_profile_3d(m, pipeline)
    assert pipeline.runs["profile_latent"] == 1

    for thr in (0.35, 0.50, 0.65):
        m.set("threshold", thr)
        compute_profile_3d(m, pipeline)
    m.set("develop_model", "mack")
    compute_profile_3d(m, pipeline)

    assert pipeline.runs["profile_latent"] == 1


def test_expensive_3d_change_does_recompute_the_latent():
    m = _small_3d_model()
    pipeline = Pipeline()
    compute_profile_3d(m, pipeline)
    m.set("thickness", m["thickness"] + 20.0)
    compute_profile_3d(m, pipeline)
    assert pipeline.runs["profile_latent"] == 2


def test_cached_3d_profile_matches_the_uncached_one_exactly():
    """The cache must be an optimisation only — never a different answer."""
    m = _small_3d_model()
    m.set("standing_waves", True)
    pipeline = Pipeline()
    for model in ("threshold", "mack"):
        for thr in (0.35, 0.55):
            m.set("develop_model", model)
            m.set("threshold", thr)
            fresh = compute_profile_3d(m)
            cached = compute_profile_3d(m, pipeline)
            assert np.array_equal(fresh.remaining, cached.remaining)
            assert np.array_equal(fresh.cut_latent, cached.cut_latent)
            assert fresh.film_remaining_pct == cached.film_remaining_pct


def test_3d_branch_does_not_disturb_the_2d_cache():
    """The two branches share the mask and nothing else."""
    m = _small_3d_model()
    pipeline = Pipeline()
    compute_imaging(m, pipeline)
    aerial_runs = pipeline.runs["aerial"]
    compute_profile_3d(m, pipeline)
    assert pipeline.runs["aerial"] == aerial_runs


def test_every_develop3d_knob_actually_moves_the_profile():
    """The control-that-lies check, made routine.

    Twice now a slider has fed a config the stage below it ignored — the PEB
    diffusion control, then the 2-D threshold standing in for a PAC threshold
    the 3-D path never read. Both were invisible to tests that only counted
    cache hits, because a cached wrong answer matches an uncached wrong one.
    So: assert the picture moves.
    """
    def configured():
        """A base where every live knob is somewhere it can actually move.

        Surface inhibition is a *pair* — a depth with no rate reduction, or a
        reduction over no depth, is off either way, the same coupling
        sigma_inner/sigma_outer have. Starting each knob at its default and
        pushing it to a limit would drive `inhibition_rate` to 1.0, which is
        "no inhibition", and the test would conclude a working control was
        dead.
        """
        m = _small_3d_model()
        m.set("develop_model", "mack")         # so develop_time is live too
        m.set("inhibition_depth", 12.0)
        m.set("inhibition_rate", 0.5)
        return m

    reference = compute_profile_3d(configured()).remaining

    for key in DEVELOP3D_ONLY:
        m = configured()
        spec = SPECS_BY_KEY[key]
        if spec.kind == "choice":
            m.set(key, next(c for c in spec.choices if c != m[key]))
        elif spec.kind == "bool":
            m.set(key, not m[key])
        else:
            # Move to whichever end of the range is further from where it sits.
            m.set(key, spec.lo if abs(m[key] - spec.lo) > abs(m[key] - spec.hi)
                  else spec.hi)
        assert not np.array_equal(compute_profile_3d(m).remaining, reference), (
            f"'{key}' changed nothing in the 3-D profile — it is wired to a "
            f"config field the develop step does not read"
        )


def test_2d_threshold_is_deliberately_inert_in_3d():
    """Documents why `threshold` is not in DEVELOP3D_ONLY.

    It calibrates the 2-D intensity model; the depth-resolved path cuts on
    PAC concentration through mack_Mth. Refreshing 3-D when it moves would be
    ~290 ms spent redrawing an identical picture.
    """
    m = _small_3d_model()
    first = compute_profile_3d(m).remaining
    m.set("threshold", 0.90)
    assert np.array_equal(compute_profile_3d(m).remaining, first)
    assert "threshold" not in DEVELOP3D_ONLY


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_layout_solver_is_skipped_when_nothing_moved(monkeypatch):
    """Constrained layout calls get_tightbbox on every artist — colourbars,
    captions, a legend on the title row — to re-derive a layout that only
    changes when the window does."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        from matplotlib.layout_engine import ConstrainedLayoutEngine

        view = window.profile_view.panels
        view._relayout = True
        view._paint()
        assert isinstance(view.figure.get_layout_engine(), ConstrainedLayoutEngine)
        view._paint()
        # "none" leaves a PlaceHolderLayoutEngine, whose execute() is a no-op —
        # the solver is what costs, and it no longer runs.
        assert not isinstance(
            view.figure.get_layout_engine(), ConstrainedLayoutEngine
        )

        from matplotlib.backends.qt_compat import QtCore, QtGui

        view.resizeEvent(
            QtGui.QResizeEvent(QtCore.QSize(900, 700), QtCore.QSize(800, 600))
        )
        assert view._relayout, "a resize must re-solve the layout"
    finally:
        window.close()


# ---------------------------------------------------------------------------
# The Wafer Stack tab
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_stack_tab_builds_and_runs_a_recipe(monkeypatch):
    """The canonical loop: deposit, deposit, etch stopping on the layer below,
    polish flat — and a snapshot per step to scrub through."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.app.stepspecs import build_step

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        for step in (
            build_step("deposit", {"material": "poly-Si", "thickness": 60.0,
                                   "conformal": False}),
            build_step("deposit", {"material": "a-C", "thickness": 40.0,
                                   "conformal": False}),
            build_step("etch", {"targets": "a-C", "depth": 200.0,
                                "anisotropy": 0.5, "stop_on": "poly-Si",
                                "stop_rate": 0.001}),
            build_step("cmp", {"height": 70.0}),
        ):
            tab.session.append(step)
        tab._refresh_list()
        assert tab.recipe.count() == 4

        # Run synchronously — the worker thread is exercised by the app, but a
        # test should not race it.
        tab.session.run_to()
        tab.on_finished(None)
        app.processEvents()

        snaps = tab.session.snapshots()
        assert len(snaps) == 4
        assert snaps[0].thickness_of("poly-Si").max() > 0
        assert snaps[1].thickness_of("a-C").max() > 0
        assert snaps[2].thickness_of("a-C").max() == 0.0, "mandrel should clear"
        assert snaps[2].thickness_of("poly-Si").min() >= 55e-9, (
            "the etch went through the layer it was told to stop on"
        )
        assert snaps[3].top_height().std() == pytest.approx(0.0, abs=1e-12)
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_scrubber_has_a_slot_for_the_bare_wafer(monkeypatch):
    """`snapshots()` has no entry for the wafer as loaded, so the scrubber
    carries one more position than there are steps."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.app.stepspecs import build_step

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.session.append(build_step("deposit", {"material": "SiO2",
                                                  "thickness": 20.0,
                                                  "conformal": False}))
        tab.session.run_to()
        tab.on_finished(None)
        app.processEvents()

        assert tab.scrubber.maximum() == 1, "one step plus 'as loaded'"
        tab.scrubber.setValue(0)
        app.processEvents()
        assert tab.scrub_label.text() == "as loaded"
        assert tab.session.stack_at(-1).thickness_of("SiO2").max() == 0.0
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_selecting_a_step_does_not_edit_it(monkeypatch):
    """Clicking through the recipe must be read-only.

    The form repopulates from the selected step, and writing back on every
    change means a round-trip mismatch would silently rewrite the recipe as
    you browsed it.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.app.stepspecs import build_step

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        for step in (
            build_step("deposit", {"material": "SiN", "thickness": 33.0,
                                   "conformal": True}),
            build_step("etch", {"targets": "SiN", "depth": 77.0,
                                "anisotropy": 0.4, "stop_on": "Si",
                                "stop_rate": 0.02}),
            build_step("strip", {"material": "photoresist"}),
        ):
            tab.session.append(step)
        tab._refresh_list()
        before = [s.to_dict() for s in tab.session.steps]

        for row in (0, 1, 2, 1, 0):
            tab.recipe.setCurrentRow(row)
            app.processEvents()

        assert [s.to_dict() for s in tab.session.steps] == before
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_importing_a_developed_profile_gives_a_patterned_wafer(monkeypatch):
    """The litho result becomes the flow's starting wafer, patterned."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        # 64 px, not 32: at 4 nm pixels a 32 px field is 128 nm, less than
        # the default 200 nm pitch, so nothing patterns and the film comes out
        # either wholly cleared or wholly intact.
        window.model.set("n_pixels", 64)
        window.model.set("n_z_slices", 3)
        window.model.set("mack_Mth", 0.42)
        profile = compute_profile_3d(window.model)
        # Guard the premise: at some settings the film clears entirely and
        # there is nothing to import, which would make the assertions below
        # pass or fail for the wrong reason.
        assert 5.0 < profile.film_remaining_pct < 95.0, (
            f"fixture prints nothing useful ({profile.film_remaining_pct:.0f}% left)"
        )

        window.stack_tab.import_profile(profile.remaining, window.model.grid())
        base = window.stack_tab.session.stack_at(-1)

        resist = base.thickness_of("photoresist")
        assert resist.max() > 0, "no resist on the imported wafer"
        assert resist.min() < resist.max(), (
            "the imported resist is uniform — the pattern did not survive"
        )
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_profile_button_is_disabled_until_there_is_a_profile(monkeypatch):
    """The affordance has to match the state, not just apologise afterwards.

    This used to be an always-enabled button: clicking it with no profile put
    one sentence in the status bar and changed nothing on screen. That reads as
    a dead control — it was reported as one, by someone reasonably expecting a
    file dialog from a button labelled "Import resist profile". The button is
    now greyed out until the Develop tab has produced something, says why in
    its tooltip, and points at File ▸ Open wafer for the from-disk case.

    The status-bar guard is still asserted: something calling the slot directly
    must no-op with an explanation rather than raise.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        assert window._profile is None
        btn = window.stack_tab.import_btn
        assert not btn.isEnabled(), "nothing to import, so nothing to offer"
        assert "Simulate" in btn.toolTip(), "must say where the profile comes from"
        assert "File" in btn.toolTip(), "must point at the from-disk route"
        assert "import" not in btn.text().lower(), (
            f"{btn.text()!r} still reads like a file dialog"
        )

        window._import_profile()
        app.processEvents()
        assert "3-D resist profile" in window.status.currentMessage()
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_file_menu_offers_open_save_and_the_device_presets(monkeypatch):
    """The half that was missing: anything arriving from disk comes through here."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.tech.devices import DEVICES

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        labels = [a.text() for a in window.file_menu.actions()]
        assert any("Open wafer" in a for a in labels), labels
        assert any("Save wafer" in a for a in labels), labels

        presets = [a.text() for a in window.devices_menu.actions()]
        for spec in DEVICES.values():
            assert spec.title in presets, presets
        assert window.act_section.isChecked(), "sectioned by default"
    finally:
        window.close()


@pytest.mark.slow
def test_a_device_preset_builds_caches_and_sections(tmp_path):
    """Build once, load from disk thereafter — and the section is data.

    nfet rather than gaa: same code path, a third of the wall clock.
    """
    from litho_sim.tech.devices import DEVICES, build, cache_path, sectioned
    from litho_sim.wafer import get_material

    stack, label = build("nfet", preset_dir=tmp_path)
    assert stack.mat.any()
    assert "nFET" in label and "nm" in label, label
    assert cache_path("nfet", tmp_path).is_file(), "preset was not cached"

    cached, _ = build("nfet", preset_dir=tmp_path)
    assert cached.mat.shape == stack.mat.shape
    assert (cached.mat == stack.mat).all(), "cache round-trip changed the wafer"

    cut = sectioned("nfet", stack)
    assert cut.mat.size < stack.mat.size, "sectioning removed nothing"
    for m in ("Si", "SiO2", "SiN", "poly-Si", "TiN"):
        assert (cut.mat == get_material(m).id).any(), f"section lost {m}"

    with pytest.raises(KeyError):
        build("no-such-device", preset_dir=tmp_path)
    assert set(DEVICES) == {"gaa", "nfet"}


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_an_imported_wafer_can_be_etched_through(monkeypatch):
    """The whole point of importing: pattern transfer.

    Resist masks the columns it covers; the etch opens the rest. Nothing is
    told the resist is a mask — it masks by being there.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.app.stepspecs import build_step

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        # 64 px, not 32: at 4 nm pixels a 32 px field is 128 nm, less than
        # the default 200 nm pitch, so nothing patterns and the film comes out
        # either wholly cleared or wholly intact.
        window.model.set("n_pixels", 64)
        window.model.set("n_z_slices", 3)
        window.model.set("mack_Mth", 0.42)
        profile = compute_profile_3d(window.model)
        tab = window.stack_tab
        tab.import_profile(profile.remaining, window.model.grid())

        tab.session.append(build_step("strip", {"material": "photoresist"}))
        stack = tab.session.run_to()
        assert stack.thickness_of("photoresist").max() == 0.0
    finally:
        window.close()


# ---------------------------------------------------------------------------
# The Wafer Stack tab: edits have to land, and reach the picture
# ---------------------------------------------------------------------------


def _settle(app, tab, timeout=15.0):
    """Pump until the tab is idle and the recipe is fully simulated.

    Drives the real worker thread rather than standing in for it: the point of
    these tests is the hand-off, and a stub would test the stub.
    """
    import time

    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if (not tab._busy_flow and not tab._edit_settle.isActive()
                and tab.session.valid_upto == len(tab.session)):
            return
    raise AssertionError("the flow never settled")


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_adding_a_step_targets_the_wafer_that_step_will_actually_see(monkeypatch):
    """The reported bug, verbatim.

    Defaults were read from the last *simulated* wafer, and while a recipe is
    being built there isn't one — so every step saw the bare substrate. Coat
    resist, add an etch, press Run: the etch proposed etching the substrate
    through 90 nm of resist, did nothing, and said nothing.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        for kind in ("spincoat", "etch"):
            tab.palette_box.setCurrentText(kind)
            tab.add_btn.click()
            _settle(app, tab)

        assert tab.session.steps[1].targets == "photoresist"
        assert tab.session.did_nothing() == [], (
            "the etch was a no-op again: "
            f"{[s.describe() for s in tab.session.steps]}"
        )
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_editing_a_step_redraws_without_pressing_run(monkeypatch):
    """A wall angle you can drag but that changes nothing is the whole report.

    Two things had to be true at once: the flow must re-run on its own, and
    the wafer must have a mask edge for the shape controls to act on.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        for kind in ("deposit", "pattern", "etch"):
            tab.palette_box.setCurrentText(kind)
            tab.add_btn.click()
            _settle(app, tab)

        # The etch must be aimed through the opening, not at the mask on top.
        assert tab.session.steps[2].targets == "spacer-oxide"

        tab.recipe.setCurrentRow(2)
        before = tab.session.stack_at(2).mat.copy()
        wall = tab.form._widgets["sidewall_deg"]
        wall.setValue(wall.value() - 20)
        _settle(app, tab)

        assert not np.array_equal(before, tab.session.stack_at(2).mat), (
            "the wall angle reached the recipe but not the wafer"
        )
        assert tab.scrubber.value() == tab.scrubber.maximum(), (
            "the view should be showing the step that just changed"
        )
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_a_step_that_did_nothing_is_marked_after_a_run(monkeypatch):
    """`_refresh_list` is the only thing that paints the marker, and it was
    never called on the path that produces one."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.app.stepspecs import build_step

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.session.append(build_step("deposit", {"material": "poly-Si",
                                                  "thickness": 40.0,
                                                  "conformal": False}))
        tab.session.append(build_step("etch", {"targets": "SOC"}))  # not present
        tab.session.run_to()
        tab.on_finished(None)
        app.processEvents()

        assert "no change" in tab.recipe.item(1).text()
        assert "unchanged" in tab.notes.text(), (
            f"the tab's readout explained nothing: {tab.notes.text()!r}"
        )
        # And the engine's own sentence has to reach the user, not just a flag —
        # naming the specific cause, because "not exposed" covers four of them.
        tab.recipe.setCurrentRow(1)
        assert "no SOC anywhere" in tab.notes.text(), tab.notes.text()
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_recipe_is_not_edited_while_a_run_is_in_flight(monkeypatch):
    """The concurrency contract, written down as something that executes.

    `Worker.run_flow` takes the session by reference, so a GUI-thread
    `replace()` would `del` snapshots out from under the worker's own
    `append`. Editing during a run therefore buffers instead of mutating.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.app.stepspecs import build_step

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.session.append(build_step("deposit", {"material": "poly-Si",
                                                  "thickness": 40.0,
                                                  "conformal": False}))
        tab.session.append(build_step("etch", {"targets": "poly-Si",
                                               "depth": 20.0}))
        tab.session.run_to()
        tab.on_finished(None)
        _settle(app, tab)

        tab.recipe.setCurrentRow(1)
        tab._busy_flow = True                      # pretend a run is in flight
        tab._set_editing_enabled(False)
        assert not tab.add_btn.isEnabled(), "the recipe stayed editable mid-run"
        assert not tab.del_btn.isEnabled()

        was = tab.session.steps[1]
        depth = tab.form._widgets["depth"]
        depth.setValue(depth.value() + 10)

        assert tab.session.steps[1] is was, "the session was mutated mid-run"
        assert 1 in tab._deferred, "the edit was dropped rather than buffered"
        assert "nm" in tab.recipe.item(1).text(), "the row did not show the edit"
        # Selecting the row again must show the pending edit, not the old value.
        tab._on_selected(1)
        assert tab.form.values()["depth"] == pytest.approx(
            tab._deferred[1].depth * 1e9)

        tab.on_finished(None)                      # the run lands
        _settle(app, tab)
        assert tab.session.steps[1].depth == pytest.approx(30e-9)
        assert not tab._deferred, "the buffer was not drained"
        assert tab.add_btn.isEnabled(), "editing was not re-enabled"
    finally:
        window.close()


# ---------------------------------------------------------------------------
# Mask 3-D: the panel must not offer impossible combinations
# ---------------------------------------------------------------------------


def test_mask_model_availability_depends_on_wavelength_and_pattern():
    """``multilayer`` needs an EUV mirror; ``fdtd`` needs a cross-section.

    Both conditions live in a different panel section from the control they
    gate, so a user changing "Mask model" alone cannot see them. The rule is
    kept Qt-free and tested here so the panel and the engine cannot drift.
    """
    from litho_sim.app.params import mask_model_availability

    duv = mask_model_availability(193e-9, "lines and spaces")
    assert duv["thin"] is None
    assert duv["fdtd"] is None
    assert duv["multilayer"] and "EUV" in duv["multilayer"]

    euv = mask_model_availability(13.5e-9, "lines and spaces")
    assert euv["multilayer"] is None

    contacts = mask_model_availability(13.5e-9, "contacts")
    assert contacts["multilayer"] is None
    assert contacts["fdtd"] and "2.5-D" in contacts["fdtd"]

    # An isolated line is a grating whose pitch is the field, so it qualifies.
    assert mask_model_availability(193e-9, "isolated line")["fdtd"] is None


def test_every_offered_mask_model_actually_runs():
    """Nothing the panel leaves enabled may raise.

    The complaint that prompted this: switching Mask model while the rest of
    the panel sat at its defaults produced an exception, because 10 of the 24
    reachable (pattern, model, wavelength) combinations were impossible. The
    sweep is kept so that stays at zero.
    """
    import itertools

    from litho_sim.app.compute import build_mask, estimate_cost_ms
    from litho_sim.app.params import (
        SPECS_BY_KEY,
        ParameterModel,
        mask_model_availability,
    )
    from litho_sim.expose.m3d import make_spectrum_provider
    from litho_sim.expose.m3d.nearfield import MaskGeometry

    offered = 0
    for pattern, model, lam_nm in itertools.product(
        SPECS_BY_KEY["pattern"].choices,
        SPECS_BY_KEY["mask_model"].choices,
        (193.0, 13.5),
    ):
        params = ParameterModel()
        params.set("pattern", pattern)
        params.set("wavelength", lam_nm)
        if lam_nm == 13.5:
            params.set("NA", 0.33)
            params.set("chief_ray_deg", 6.0)
        params.set("n_pixels", 32)
        params.set("source_grid", 5)

        if mask_model_availability(params.si("wavelength"), pattern)[model]:
            continue  # the panel greys this one out
        offered += 1

        params.set("mask_model", model)
        optics, grid = params.optics(), params.grid()
        estimate_cost_ms(params)
        if model == "fdtd":
            # The library build is minutes; the validation is what used to fail.
            assert optics.mask_geometry is not None, (pattern, model, lam_nm)
            MaskGeometry(**optics.mask_geometry)
        else:
            make_spectrum_provider(build_mask(params), optics, grid)

    assert offered >= 14, f"only {offered} combinations were exercised"


# ---------------------------------------------------------------------------
# A loaded device shows the flow that built it
# ---------------------------------------------------------------------------


def _gaa_flow():
    """The cached GAA recipe, or a skip. Building it here would cost 7 s."""
    from litho_sim.tech.devices import build, flow_for

    stack, label = build("gaa")
    flow = flow_for("gaa")
    if flow is None:                                    # pragma: no cover
        pytest.skip("no cached gaa flow")
    return stack, label, flow


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_loading_a_device_fills_the_recipe_list(monkeypatch):
    """The panel used to sit empty beside a plot listing 37 process steps.

    Two stores, one populated: the plot's legend reads `Stack.history`, the
    list reads `FlowSession.steps`, and loading a device wrote only the first.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    stack, label, flow = _gaa_flow()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        assert tab.recipe.count() == 0, "premise: it starts empty"

        tab.load_stack(stack, label=label, flow=flow)
        app.processEvents()

        assert tab.recipe.count() == len(flow) > 30, (
            "the list must hold one row per step of the device's own recipe"
        )
        assert tab.session.as_built_upto == len(flow), "provenance is recorded"
        assert tab.session.locked_upto == 0, (
            "and nothing is locked: the recipe came with the layouts and "
            "optics it ran with, so this session can replay it"
        )
        assert not tab._busy_flow, (
            "an adopted flow has every snapshot already; scheduling a run for "
            "it freezes the panel for a round trip with nothing to compute"
        )
        # The rows say what ran, in the engine's own words.
        assert "Deposit SiGe 10 nm" in tab.recipe.item(0).text()
        assert "Expose fin" in tab.recipe.item(7).text()
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_selecting_a_device_step_shows_the_settings_it_ran_with(monkeypatch):
    """The precise settings, all of them, as controls where a control can hold
    the value and as text where none can.

    Every field stays visible either way. What changed when the panel became
    editable is only *which widget* each one gets — a value shown but not
    editable is honest, a value quietly narrowed to fit a dropdown is not.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.app.stepspecs import readout_for

    stack, label, flow = _gaa_flow()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.load_stack(stack, label=label, flow=flow)
        fin_etch = next(i for i, s in enumerate(flow.steps)
                        if s.kind == "etch" and not isinstance(s.targets, str))
        tab.recipe.setCurrentRow(fin_etch)
        app.processEvents()

        assert not tab.form.isHidden(), "the editor shows; the flow is editable"
        assert tab.readout.isHidden()

        shown = {s.key for s in tab.form._specs}
        step = tab.session.steps[fin_etch]
        # Nothing the readout would report may be missing from the editor.
        assert shown >= {"depth", "anisotropy", "targets", "stop_on"}
        assert {"sidewall_deg", "footer_height", "footer_extent",
                "top_radius", "bottom_radius", "bias"} <= shown, (
            "footing and rounding have to be reachable"
        )
        # The two the widgets cannot hold are present, and inert.
        assert set(tab.form._readonly) == {"targets", "stop_on", "stop_rate"}
        assert step.targets == ["Si", "SiGe"]
        assert step.selectivity == {"Si": 1.0, "SiGe": 1.0}
        assert readout_for(step), "and the full readout still formats it"
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_scrubber_walks_the_device_build(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    stack, label, flow = _gaa_flow()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.load_stack(stack, label=label, flow=flow)
        app.processEvents()

        assert tab.scrubber.maximum() == len(flow), (
            "one slot per step, plus slot 0 for the wafer before any of them"
        )
        # Selecting a step moves the picture to that step: the settings on the
        # left and the wafer on the right have to be describing one thing.
        tab.recipe.setCurrentRow(9)
        app.processEvents()
        assert tab.scrubber.value() == 10
        assert tab.scrub_label.text() == flow.steps[9].describe()

        tab.scrubber.setValue(0)
        app.processEvents()
        assert tab.scrub_label.text() == "bare wafer", (
            "slot 0 holds the substrate the flow started on, not the device"
        )
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_an_as_built_step_can_be_edited_removed_and_built_on(monkeypatch):
    """A device's recipe is a recipe, not an exhibit.

    It arrives with the layouts and optics it ran with, so this session can
    honestly replay it — and where it can replay, it may edit.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    stack, label, flow = _gaa_flow()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.load_stack(stack, label=label, flow=flow)
        tab.recipe.setCurrentRow(3)
        app.processEvents()

        assert tab.del_btn.isEnabled(), "an editable step is removable"
        assert tab.up_btn.isEnabled() and tab.down_btn.isEnabled()

        before = len(tab.session)
        tab.del_btn.click()
        app.processEvents()
        assert len(tab.session) == before - 1, "the step was removed"
        assert tab.session.valid_upto <= 3, "and the tail is now out of date"

        # Adding still works, and lands after the device.
        tab.palette_box.setCurrentText("strip")
        tab.add_btn.click()
        app.processEvents()
        assert len(tab.session) == before
        assert tab.recipe.currentRow() == len(tab.session) - 1
        assert tab.session.grid.pixel_size == stack.grid.pixel_size, (
            "the device's grid is the authority; adding a step must not swap "
            "it for the parameter dock's"
        )
        assert tab.session.grid.n_z_slices == flow.grid.n_z_slices, (
            "and the recipe's own resist slicing has to survive too, or a "
            "replayed exposure resolves differently from the one on screen"
        )
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_cross_section_of_a_scrubbed_step_is_not_empty(monkeypatch):
    """The device's section is a 3-D cut, and applying it to the 2-D view
    emptied the plot.

    A cross-section takes the middle slice of what it is handed, so cropping
    first *moves* that slice. The GAA cut keeps y from 45%, whose midpoint sits
    just outside the fin: fine on the finished device, where the STI and ILD
    have filled that region back in, and blank at the fin etch, where
    everything off-fin has just been etched to the substrate. It rendered an
    empty pair of axes and every test still passed.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.tech.devices import sectioned

    stack, label, flow = _gaa_flow()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.load_stack(stack, label=label, flow=flow,
                       section=lambda s: sectioned("gaa", s))
        tab.mode_section.setChecked(True)
        app.processEvents()

        drawn = []
        original = tab.view.show_stack
        tab.view.show_stack = (
            lambda s, *a, **kw: (drawn.append(s), original(s, *a, **kw))[1]
        )

        for step in (6, 10, 20, 36):
            drawn.clear()
            tab.scrubber.setValue(step)
            app.processEvents()
            assert drawn, f"step {step} drew nothing at all"
            wafer = drawn[-1]
            # The slice the cross-section actually paints.
            sl = wafer.mat[:, wafer.mat.shape[1] // 2, :]
            filled = float((sl != 0).mean())
            assert filled > 0.05, (
                f"step {step}: the drawn cross-section is {filled:.1%} "
                f"material — an all-but-empty plot"
            )

        # And the other half: the solid view still gets the device cut open,
        # or "not empty" would be satisfied by never sectioning anything.
        drawn.clear()
        tab.mode_solid.setChecked(True)
        app.processEvents()
        assert drawn, "switching to 3-D drew nothing"
        assert drawn[-1].mat.shape != stack.mat.shape, (
            "the 3-D view needs the device cut open — uncut it is a solid "
            "block with every interesting feature interior"
        )
        assert np.array_equal(drawn[-1].mat, sectioned("gaa", stack).mat)
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_process_flow_strip_can_be_put_away(monkeypatch):
    """It says what the recipe list says, and costs a quarter of the width.

    Both halves matter: the wafer has to actually get the space back (hiding
    the axes would leave its column reserved), and the strip has to come back.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow
    from litho_sim.app.stepspecs import build_step

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.session.append(
            build_step("deposit", {"material": "SiN", "thickness": 40.0,
                                   "conformal": False})
        )
        tab.session.run_to()
        tab.on_finished(None)
        tab.mode_section.setChecked(True)
        app.processEvents()

        def wafer_width():
            tab.view.canvas.draw()
            return tab.view.ax.get_position().width

        assert tab.show_flow.isChecked(), "on by default — it is useful"
        assert tab.view.ax_panel is not None
        with_strip = wafer_width()

        tab.show_flow.setChecked(False)          # the real checkbox
        app.processEvents()
        assert tab.view.ax_panel is None, "the strip's axes must be gone"
        without = wafer_width()
        assert without > with_strip * 1.3, (
            f"the wafer got {without:.3f} of the figure against "
            f"{with_strip:.3f} — the width was not actually reclaimed"
        )

        tab.show_flow.setChecked(True)
        app.processEvents()
        assert tab.view.ax_panel is not None, "and it has to come back"
        # Not equality with the first reading: that one was taken before the
        # constrained layout had settled, so it is a little narrower than the
        # steady state. What matters is that the column was handed back.
        assert wafer_width() < without * 0.9, (
            "the strip returned but took none of the width back"
        )
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_the_flow_strip_toggle_is_greyed_where_it_does_nothing(monkeypatch):
    """The solid view has no strip, so the checkbox must not stay live there.

    A control with nothing to act on is this app's recurring bug — the eight
    inert etch controls, then the always-enabled profile button.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        assert tab.show_flow.isEnabled(), "premise: live in the cross-section"

        tab.mode_solid.setChecked(True)
        app.processEvents()
        assert not tab.show_flow.isEnabled()
        assert tab.view._pages.currentWidget() is tab.view.solid, "the solid is showing"

        tab.mode_section.setChecked(True)
        app.processEvents()
        assert tab.show_flow.isEnabled()
        assert tab.view.ax_panel is not None, "the strip returns with the view"

        # A preference taken while it was greyed still has to stick.
        tab.show_flow.setChecked(False)
        app.processEvents()
        tab.mode_solid.setChecked(True)
        app.processEvents()
        tab.mode_section.setChecked(True)
        app.processEvents()
        assert tab.view.ax_panel is None, (
            "coming back from 3-D restored a strip the user had put away"
        )
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_clicking_through_a_device_recipe_never_changes_it(monkeypatch):
    """Browsing must be read-only even now that editing is allowed.

    The form repopulates from the selected step and writes back on change, so
    a round-trip mismatch would rewrite the recipe as you looked at it. The
    device flows are the hard case: a two-material etch, five-row selectivity
    tables, and a 254 nm film past its slider's ceiling.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    stack, label, flow = _gaa_flow()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.load_stack(stack, label=label, flow=flow)
        app.processEvents()
        before = [s.to_dict() for s in tab.session.steps]

        for row in range(tab.recipe.count()):
            tab.recipe.setCurrentRow(row)
            app.processEvents()

        assert [s.to_dict() for s in tab.session.steps] == before, (
            "merely selecting steps rewrote the recipe"
        )
        assert tab.session.valid_upto == len(tab.session), (
            "and nothing was invalidated, so nothing needs re-running"
        )
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_editing_a_device_step_marks_the_tail_stale_without_running(monkeypatch):
    """Manual re-run: editing step 8 of 36 replays 29, which is a button press
    rather than something to pay on every slider release."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    stack, label, flow = _gaa_flow()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.load_stack(stack, label=label, flow=flow)
        app.processEvents()
        assert tab.session.locked_upto == 0, "the context came with it"

        fin = next(i for i, s in enumerate(tab.session.steps)
                   if s.kind == "expose")
        tab.recipe.setCurrentRow(fin)
        app.processEvents()

        # Through the real widget, not the handler.
        spec = next(s for s in tab.form._specs if s.key == "dose")
        tab.form._widgets["dose"].setValue(
            int(round((1.20 - spec.lo) / spec.step))
        )
        app.processEvents()

        step = tab.session.steps[fin]
        assert step.dose == pytest.approx(1.20, abs=0.05)
        assert step.layout == "fin", "the field nobody touched survived"
        assert not tab._edit_settle.isActive() and not tab._busy_flow, (
            "an adopted flow must not auto-run"
        )
        assert tab.session.valid_upto == fin, "the tail was invalidated"
        stale = [i for i in range(tab.recipe.count())
                 if "stale" in tab.recipe.item(i).text()]
        assert len(stale) == len(tab.session) - fin
        assert "out of date" in tab.notes.text()
    finally:
        window.close()


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_an_etch_profile_control_reaches_the_step(monkeypatch):
    """Footing and rounding, asked for by name, and inert until now."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    stack, label, flow = _gaa_flow()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    try:
        tab = window.stack_tab
        tab.load_stack(stack, label=label, flow=flow)
        etch = next(i for i, s in enumerate(tab.session.steps)
                    if s.kind == "etch" and not isinstance(s.targets, str))
        tab.recipe.setCurrentRow(etch)
        app.processEvents()

        for key, target in (("sidewall_deg", 78.0), ("footer_height", 10.0)):
            spec = next(s for s in tab.form._specs if s.key == key)
            tab.form._widgets[key].setValue(
                int(round((target - spec.lo) / spec.step))
            )
            app.processEvents()

        step = tab.session.steps[etch]
        assert step.sidewall_deg == pytest.approx(78.0, abs=1.0)
        assert step.footer_height == pytest.approx(10e-9, abs=2e-9)
        # And the fields no widget can hold came through untouched.
        assert step.targets == ["Si", "SiGe"]
        assert step.selectivity == {"Si": 1.0, "SiGe": 1.0}
        assert "targets" in tab.form._readonly
    finally:
        window.close()


# ---------------------------------------------------------------------------
# The Process Window tab
# ---------------------------------------------------------------------------


def test_compute_fem_end_to_end_headless():
    """The whole FEM path with no Qt: request in, measured result out.

    Guards the tab's compute contract — a sweep that runs, anchors its dose
    axis, extracts a window, and reports progress in the promised units.
    """
    from litho_sim.app.fem import FemRequest, compute_fem

    model = ParameterModel()
    model.set("source_grid", 5)          # keep the Abbe sums cheap

    ticks: list[tuple[int, int]] = []
    req = FemRequest(
        params=model,
        focus_half_range_nm=150.0, n_focus=3,
        dose_half_range=0.30, n_dose=3,
        target_cd_nm=100.0, tolerance_pct=10.0,
        auto_centre_dose=True, with_meef=True,
    )
    result = compute_fem(req, progress=lambda i, n: ticks.append((i, n)))

    assert len(result.bossung_df) == 9, (
        f"3 focus × 3 dose should give 9 rows, got {len(result.bossung_df)}."
    )
    assert "nils" in result.bossung_df.columns
    assert result.dose_anchored, "dose-to-size should anchor at app defaults"
    assert result.pw["prints"], "the default L/S must print at its anchor"
    assert result.pw["EL_pct"] > 0 and result.pw["DOF_nm"] > 0
    assert np.isfinite(result.meef), "MEEF should be measurable here"
    assert result.diagnosis == ""
    assert "EL" in result.summary and "MEEF" in result.summary
    # anchor (1) + focus columns (3) + MEEF (2) = 6 progress units
    assert ticks[-1] == (6, 6), f"progress ended at {ticks[-1]}, not (6, 6)."


def test_compute_fem_reports_a_diagnosis_when_nothing_prints():
    from litho_sim.app.fem import FemRequest, compute_fem

    model = ParameterModel()
    model.set("source_grid", 5)
    # An unreachable target: the anchor fails, the un-anchored clear-dose
    # axis prints nothing in spec, and the result must say so in words.
    req = FemRequest(
        params=model,
        focus_half_range_nm=100.0, n_focus=3,
        dose_half_range=0.10, n_dose=3,
        target_cd_nm=500.0, tolerance_pct=5.0,
        auto_centre_dose=True, with_meef=False,
    )
    result = compute_fem(req)
    # The pattern itself prints — the failure is that no printed CD lands
    # within spec of an absurd target, and the diagnosis must say which
    # failure it is (a target problem, not a dose problem).
    assert not result.dose_anchored
    assert result.window_empty
    assert "No (dose, focus) point printed in spec" in result.diagnosis
    assert "Adjust the target CD or tolerance" in result.diagnosis
    assert "no process window" in result.summary


@pytest.mark.skipif(not _qt_binding_available(), reason="needs a Qt binding")
def test_process_window_tab_runs_a_sweep_offscreen():
    """Drive the real tab: click Run, wait for the worker, check the panels.

    This is the wiring test — request signal to worker thread, progress back
    to the bar, result into four drawn axes, controls released. Everything a
    unit test of compute_fem cannot see.
    """
    import os
    import time as _time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.backends.qt_compat import QtWidgets

    from litho_sim.app.main import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    try:
        win.model.set("source_grid", 5)
        tab = win.pw_tab
        win.tabs.setCurrentWidget(win.simulate_tab)
        win.simulate_tab.select("Focus-exposure matrix")
        app.processEvents()

        assert len(tab.figure.axes) == 0, "placeholder should hold no axes"
        assert "images" in tab.cost_label.text(), "cost estimate never drawn"

        tab.n_focus.setValue(3)
        tab.n_dose.setValue(3)
        tab.focus_range.setValue(150.0)
        tab.run_btn.click()
        assert not tab.run_btn.isEnabled(), "Run must latch while busy"

        deadline = _time.monotonic() + 30.0
        while tab._busy and _time.monotonic() < deadline:
            app.processEvents()
            _time.sleep(0.01)
        assert not tab._busy, "sweep never finished within 30 s"

        assert tab._result is not None, "no result reached the tab"
        # Four panels; the heatmap's colorbar makes it at least five axes.
        assert len(tab.figure.axes) >= 4, (
            f"expected the 2×2 grid, found {len(tab.figure.axes)} axes"
        )
        assert tab.run_btn.isEnabled(), "controls never re-enabled"
        assert tab.progress.value() == tab.progress.maximum()
        # Strict on purpose: at these settings the default L/S anchors and
        # prints, so the summary must carry the full readout. A generic
        # "EL or diagnosis" check let a disabled-before-read MEEF checkbox
        # slip through — the sweep ran, with MEEF silently off.
        assert "EL" in tab.notes.text(), tab.notes.text()
        assert "MEEF" in tab.notes.text(), tab.notes.text()
        assert "dose-to-size" in tab.notes.text(), tab.notes.text()

        # The fourth-panel switch redraws from the held result, no recompute.
        tab.fourth.setCurrentIndex(1)
        app.processEvents()
        assert len(tab.figure.axes) >= 4
    finally:
        win.thread.quit()
        win.thread.wait(2000)
        win.close()
