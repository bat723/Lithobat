"""
Unit tests for the process window analysis module.

Tests use a lightweight simulation (small grid, threshold model) so that
the test suite runs in seconds, not minutes.

Run with::

    pytest tests/test_analysis.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.analysis import (
    calibrate_dose_to_size,
    compute_depth_of_focus,
    compute_el_dof_curve,
    compute_exposure_latitude,
    compute_meef,
    compute_nils,
    compute_process_window,
    evaluate_cd,
    in_spec,
    sweep_dose_focus,
)
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig, SimulationConfig
from litho_sim.develop.resist import measure_cd_1d, measure_cd_2d
from litho_sim.mask.patterns import lines_and_spaces

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cfg() -> SimulationConfig:
    """Tiny grid config for fast sweep tests."""
    return SimulationConfig(
        optics=OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.85),
        resist=ResistConfig(tone="positive", threshold=0.30),
        grid=GridConfig(n_pixels=64, pixel_size=4e-9),
    )


@pytest.fixture(scope="module")
def tiny_mask(tiny_cfg: SimulationConfig) -> np.ndarray:
    cfg = tiny_cfg
    return lines_and_spaces(
        cfg.grid.n_pixels,
        cfg.grid.pixel_size,
        pitch=200e-9,
        cd=100e-9,
    )


@pytest.fixture(scope="module")
def bossung_df(tiny_cfg, tiny_mask):
    """Pre-computed Bossung sweep for analysis tests."""
    return sweep_dose_focus(
        mask=tiny_mask,
        optics=tiny_cfg.optics,
        grid=tiny_cfg.grid,
        resist=tiny_cfg.resist,
        doses=[0.8, 1.0, 1.2],
        defoci_nm=[-100.0, 0.0, 100.0],
        model="threshold",
    )


# ---------------------------------------------------------------------------
# measure_cd helpers
# ---------------------------------------------------------------------------


def test_measure_cd_1d_known_profile():
    """CD measurement on a synthetic profile with known answer."""
    n = 100
    profile = np.zeros(n)
    profile[30:71] = 1.0  # feature width = 41 pixels
    pixel_size = 2e-9
    cd = measure_cd_1d(profile, pixel_size)
    expected = 41 * pixel_size
    assert abs(cd - expected) <= pixel_size, (
        f"Expected CD ≈ {expected*1e9:.1f} nm, got {cd*1e9:.1f} nm."
    )


def test_measure_cd_1d_no_feature():
    """All-zero profile should return CD = 0."""
    profile = np.zeros(64)
    assert measure_cd_1d(profile, 2e-9) == 0.0


def test_measure_cd_2d(tiny_cfg, tiny_mask):
    """2-D CD measurement should return a positive value for an L/S mask."""
    from litho_sim.develop.resist import simulate_resist
    from litho_sim.expose.aerial_image import compute_aerial_image

    aerial = compute_aerial_image(tiny_mask, tiny_cfg.optics, tiny_cfg.grid)
    _, _, resist = simulate_resist(aerial, tiny_cfg.resist, tiny_cfg.grid, model="threshold")
    cd = measure_cd_2d(resist, tiny_cfg.grid.pixel_size)
    assert cd > 0, "CD should be positive for a printed L/S pattern."


# ---------------------------------------------------------------------------
# in_spec helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cd, target, tol, expected", [
    (100.0, 100.0, 10.0, True),   # exact match
    (108.0, 100.0, 10.0, True),   # within 10%
    (112.0, 100.0, 10.0, False),  # outside 10%
    (0.0, 100.0, 10.0, False),    # zero CD
    (100.0, 0.0, 10.0, False),    # zero target
])
def test_in_spec(cd, target, tol, expected):
    assert in_spec(cd, target, tol) == expected


# ---------------------------------------------------------------------------
# Bossung sweep
# ---------------------------------------------------------------------------


def test_bossung_df_shape(bossung_df):
    """Sweep should have 3 doses × 3 defoci = 9 rows."""
    assert len(bossung_df) == 9, f"Expected 9 rows, got {len(bossung_df)}."


def test_bossung_df_columns(bossung_df):
    for col in ("dose", "defocus_nm", "cd_nm"):
        assert col in bossung_df.columns, f"Missing column '{col}'."


def test_bossung_cd_nonnegative(bossung_df):
    assert (bossung_df["cd_nm"] >= 0).all(), "CD values should be non-negative."


# ---------------------------------------------------------------------------
# Process window metrics
# ---------------------------------------------------------------------------


def test_compute_process_window_keys(bossung_df):
    pw = compute_process_window(bossung_df, target_cd_nm=100.0, tolerance_pct=10.0)
    for key in ("EL_pct", "DOF_nm", "area", "best_dose", "best_focus_nm", "window_df"):
        assert key in pw, f"Missing key '{key}' in process window result."


def test_el_nonnegative(bossung_df):
    el = compute_exposure_latitude(bossung_df, target_cd_nm=100.0, tolerance_pct=20.0)
    assert el >= 0.0, f"EL should be ≥ 0, got {el}."


def test_dof_nonnegative(bossung_df):
    dof = compute_depth_of_focus(bossung_df, target_cd_nm=100.0, tolerance_pct=20.0)
    assert dof >= 0.0, f"DOF should be ≥ 0, got {dof}."


# ---------------------------------------------------------------------------
# Closed-form metrology fixtures (audit acceptance criteria for H2/H5)
#
# A cosine aerial image has analytic threshold crossings, so both the
# sub-pixel CD and the full-scale NILS can be pinned to a formula rather
# than to whatever the code happened to return last week.
# ---------------------------------------------------------------------------


def _cosine_profile(n=256, pixel_size=1e-9, period=128e-9):
    """I(x) = 0.5·(1 + cos(2πx/p)), peak at the array centre."""
    x = (np.arange(n) - n // 2) * pixel_size
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * x / period)), x


def test_subpixel_cd_matches_cosine_closed_form():
    """CD of a cosine bright fringe: width = (p/π)·arccos(2t−1)."""
    from litho_sim.develop.resist import measure_cd_1d

    n, px, period, t = 256, 1e-9, 128e-9, 0.3
    profile, _ = _cosine_profile(n, px, period)
    cd = measure_cd_1d(profile, px, threshold=t, feature="above")
    expected = (period / np.pi) * np.arccos(2.0 * t - 1.0)
    err_px = abs(cd - expected) / px
    assert err_px < 0.05, (
        f"Sub-pixel CD off by {err_px:.3f} px: got {cd*1e9:.3f} nm, "
        f"closed form {expected*1e9:.3f} nm."
    )


def test_subpixel_cd_below_measures_the_dark_line():
    """feature='below' measures the complementary dark region of the period."""
    from litho_sim.develop.resist import measure_cd_1d

    n, px, period, t = 256, 1e-9, 128e-9, 0.3
    profile, _ = _cosine_profile(n, px, period)
    # Shift by half a period so the dark fringe sits at the array centre.
    dark_centred = np.roll(profile, int(period / 2 / px))
    cd_dark = measure_cd_1d(dark_centred, px, threshold=t, feature="below")
    bright = (period / np.pi) * np.arccos(2.0 * t - 1.0)
    expected = period - bright
    assert abs(cd_dark - expected) / px < 0.05, (
        f"Dark-line CD {cd_dark*1e9:.3f} nm ≠ p − bright = {expected*1e9:.3f} nm."
    )


def test_nils_matches_cosine_closed_form_full_scale():
    """NILS = CD·|dlnI/dx| at the crossing — full linewidth, not half (H2).

    For the cosine, |dlnI/dx| at the threshold crossing is
    (π/p)·sqrt(1−(2t−1)²)/t; a surviving /2 in the implementation would
    miss the closed form by 2× and fail loudly.
    """
    n, px, period, t = 256, 1e-9, 128e-9, 0.3
    profile, _ = _cosine_profile(n, px, period)
    nominal_cd = 100e-9
    nils = compute_nils(profile, px, threshold=t, nominal_cd=nominal_cd)
    ils = (np.pi / period) * np.sqrt(1.0 - (2.0 * t - 1.0) ** 2) / t
    expected = ils * nominal_cd
    assert abs(nils - expected) / expected < 0.02, (
        f"NILS {nils:.3f} vs closed form {expected:.3f} (>2% off)."
    )


# ---------------------------------------------------------------------------
# Analytic Bossung frame — EL/DOF extraction pinned without a simulator
#
# cd(d, f) = target·(1 + a·(1−d) − b·|f|) is linear along both axes, so the
# linearly-interpolated spec crossings are *exact* and EL/DOF have closed
# forms: d ∈ [1 − tol/(100a), 1 + tol/(100a)], |f| ≤ tol/(100b).
# ---------------------------------------------------------------------------

_TARGET = 100.0
_A = 0.45          # dose sensitivity → EL = 2·0.1/0.45·100 = 44.444…%
_B = 4.5e-4        # per nm → DOF = 2·0.1/4.5e-4 = 444.444… nm


def _analytic_frame():
    import pandas as pd

    doses = np.round(np.arange(0.70, 1.301, 0.05), 4)
    foci = np.arange(-300.0, 301.0, 50.0)
    rows = []
    for d in doses:
        for f in foci:
            cd = _TARGET * (1.0 + _A * (1.0 - d) - _B * abs(f))
            rows.append(dict(dose=float(d), defocus_nm=float(f), cd_nm=cd, cd_m=cd * 1e-9))
    return pd.DataFrame(rows)


def test_el_closed_form_with_nominal_dose_denominator():
    """EL = (d_hi − d_lo)/nominal·100 with interpolated crossings.

    Closed form: in-spec doses are [0.7778, 1.2222], so EL = 44.444 %.
    The old mean-of-in-spec-doses denominator would return a different
    number, and node-resolution extraction would return 40 %.
    """
    df = _analytic_frame()
    el = compute_exposure_latitude(
        df, target_cd_nm=_TARGET, tolerance_pct=10.0,
        defocus_nm=0.0, nominal_dose=1.0,
    )
    expected = 2.0 * (10.0 / 100.0) / _A * 100.0  # 44.444…
    assert el == pytest.approx(expected, rel=1e-6), (
        f"EL {el:.4f}% vs closed form {expected:.4f}%."
    )


def test_dof_closed_form_with_interpolated_crossings():
    """DOF at dose 1.0: |f| ≤ 222.22 nm → 444.44 nm, exactly interpolable."""
    df = _analytic_frame()
    dof = compute_depth_of_focus(
        df, target_cd_nm=_TARGET, tolerance_pct=10.0, dose=1.0
    )
    expected = 2.0 * (10.0 / 100.0) / _B  # 444.444… nm
    assert dof == pytest.approx(expected, rel=1e-6), (
        f"DOF {dof:.2f} nm vs closed form {expected:.2f} nm."
    )


def test_process_window_ties_break_to_zero_focus():
    """The frame's CD spread across dose is focus-independent, so every
    printing column ties on the flatness criterion — the tie must go to the
    smallest |defocus|, not to whichever dict key came first."""
    df = _analytic_frame()
    pw = compute_process_window(df, target_cd_nm=_TARGET, tolerance_pct=10.0)
    assert pw["prints"] is True
    assert pw["best_focus_nm"] == pytest.approx(0.0)
    assert pw["best_dose"] == pytest.approx(1.0)
    assert pw["EL_pct"] == pytest.approx(2.0 * 0.1 / _A * 100.0, rel=1e-6)
    assert pw["DOF_nm"] == pytest.approx(2.0 * 0.1 / _B, rel=1e-6)


def test_best_focus_excludes_non_printing_points():
    """cd = 0 columns have zero spread and used to win the flattest-Bossung
    criterion, parking best focus at a sweep edge where nothing prints."""
    import pandas as pd

    doses = [0.8, 1.0, 1.2]
    foci = [-300.0, -150.0, 0.0, 150.0, 300.0]
    rows = []
    for d in doses:
        for f in foci:
            cd = 0.0 if f == -300.0 else 100.0 + 5.0 * (1.0 - d)
            rows.append(dict(dose=d, defocus_nm=f, cd_nm=cd, cd_m=cd * 1e-9))
    pw = compute_process_window(pd.DataFrame(rows), target_cd_nm=100.0)
    assert pw["prints"] is True
    assert pw["best_focus_nm"] != -300.0, (
        "Best focus landed on the non-printing column."
    )
    assert pw["best_focus_nm"] == pytest.approx(0.0)  # tie-break to centre


def test_best_focus_skips_degenerate_flat_columns():
    """Deep defocus compresses every dose onto nearly the same wrong CD.

    That column is flatter than any real one — printing-point exclusion does
    not catch it, because the points *do* print. Found by running the real
    app: best focus landed at −300 nm where nothing was in spec, EL came out
    0, and the tab declared 'no window' while in-spec points sat in plain
    view at focus 0.
    """
    import pandas as pd

    doses = [0.8, 1.0, 1.2]
    foci = [-300.0, 0.0, 300.0]
    rows = []
    for d in doses:
        for f in foci:
            if abs(f) == 300.0:
                cd = 50.0                      # printing, flat, out of spec
            else:
                cd = 100.0 + 20.0 * (1.0 - d)  # in spec, dose-sensitive
            rows.append(dict(dose=d, defocus_nm=f, cd_nm=cd, cd_m=cd * 1e-9))
    pw = compute_process_window(pd.DataFrame(rows), target_cd_nm=100.0,
                                tolerance_pct=10.0)
    assert pw["best_focus_nm"] == pytest.approx(0.0), (
        f"Best focus {pw['best_focus_nm']} landed on a degenerate flat "
        "column instead of the in-spec one."
    )
    assert pw["EL_pct"] > 0.0


def test_all_dark_frame_reports_prints_false():
    import pandas as pd

    rows = [
        dict(dose=d, defocus_nm=f, cd_nm=0.0, cd_m=0.0)
        for d in (0.8, 1.0, 1.2) for f in (-100.0, 0.0, 100.0)
    ]
    pw = compute_process_window(pd.DataFrame(rows), target_cd_nm=100.0)
    assert pw["prints"] is False
    assert pw["EL_pct"] == 0.0 and pw["DOF_nm"] == 0.0
    assert np.isnan(pw["best_focus_nm"]) and np.isnan(pw["best_dose"])


def test_dof_two_islands_never_merge():
    """An interrupted in-spec window is two windows; DOF reports the one
    holding the best CD, not the max−min span across the gap."""
    import pandas as pd

    foci = np.arange(-300.0, 301.0, 50.0)
    rows = []
    for f in foci:
        if f in (-300.0, -250.0):
            cd = 108.0            # island 1: in spec, off target
        elif -200.0 <= f <= 0.0:
            cd = 130.0            # gap: prints, out of spec
        else:
            cd = 100.0            # island 2: on target
        rows.append(dict(dose=1.0, defocus_nm=float(f), cd_nm=cd, cd_m=cd * 1e-9))
    dof = compute_depth_of_focus(
        pd.DataFrame(rows), target_cd_nm=100.0, tolerance_pct=10.0, dose=1.0
    )
    # Island 2 spans nodes 50…300; its left bound interpolates to where CD
    # crosses 110 between f=0 (130) and f=50 (100): 50 − 50·(1/3) = 33.33.
    expected = 300.0 - (50.0 - 50.0 * (110.0 - 100.0) / (130.0 - 100.0))
    assert dof == pytest.approx(expected, rel=1e-6), (
        f"DOF {dof:.2f} nm; two islands must not merge (max−min would be 600)."
    )


# ---------------------------------------------------------------------------
# Sweep economics — one Abbe sum per focus, zero per dose
# ---------------------------------------------------------------------------


def test_sweep_costs_one_abbe_sum_per_focus(tiny_cfg, tiny_mask, monkeypatch):
    import litho_sim.analysis.process_window as pw_mod

    calls: list[bool] = []
    real = pw_mod.compute_aerial_image

    def counting(*args, **kwargs):
        calls.append(bool(kwargs.get("return_raw", False)))
        return real(*args, **kwargs)

    monkeypatch.setattr(pw_mod, "compute_aerial_image", counting)
    pw_mod.sweep_dose_focus(
        mask=tiny_mask,
        optics=tiny_cfg.optics,
        grid=tiny_cfg.grid,
        resist=tiny_cfg.resist,
        doses=[0.8, 1.0, 1.2],
        defoci_nm=[-100.0, -50.0, 0.0, 50.0],
    )
    assert len(calls) == 4, (
        f"3 doses × 4 defoci should cost 4 Abbe sums, not {len(calls)}."
    )
    assert all(calls), "Every sweep image must be requested with return_raw=True."


def test_sweep_matches_evaluate_cd_at_shared_point(tiny_cfg, tiny_mask):
    """The rescale shortcut must be exactly the full computation."""
    import dataclasses

    df = sweep_dose_focus(
        mask=tiny_mask, optics=tiny_cfg.optics, grid=tiny_cfg.grid,
        resist=tiny_cfg.resist, doses=[1.0], defoci_nm=[0.0],
        normalisation="clear",
    )
    clear_optics = dataclasses.replace(tiny_cfg.optics, normalisation="clear")
    cd_eval = evaluate_cd(
        tiny_mask, clear_optics, tiny_cfg.grid, tiny_cfg.resist,
        dose=1.0, defocus_m=0.0,
    )
    assert df["cd_m"].iloc[0] == pytest.approx(cd_eval, rel=1e-12), (
        "Sweep rescale path and evaluate_cd disagree at the same point."
    )


def test_sweep_progress_callback_fires_per_focus(tiny_cfg, tiny_mask):
    ticks: list[tuple[int, int]] = []
    sweep_dose_focus(
        mask=tiny_mask, optics=tiny_cfg.optics, grid=tiny_cfg.grid,
        resist=tiny_cfg.resist, doses=[1.0], defoci_nm=[-50.0, 0.0, 50.0],
        progress=lambda i, n: ticks.append((i, n)),
    )
    assert ticks == [(1, 3), (2, 3), (3, 3)]


def test_sweep_has_nils_column(bossung_df):
    assert "nils" in bossung_df.columns
    assert (bossung_df["nils"] >= 0).all()


# ---------------------------------------------------------------------------
# Dose-to-size, EL-vs-DOF curve, MEEF
# ---------------------------------------------------------------------------


def test_dose_to_size_round_trips(tiny_cfg, tiny_mask):
    # The 64-px fixture's image never widens the printed line to the full
    # 100 nm drawn CD at any dose, so calibrate to a size it can reach —
    # the machinery under test (scan, bracket, bisection) is identical.
    import dataclasses

    target = 75.0
    dose = calibrate_dose_to_size(
        tiny_mask, tiny_cfg.optics, tiny_cfg.grid, tiny_cfg.resist,
        target_cd_nm=target, tol_nm=0.05,
    )
    assert np.isfinite(dose), "Calibration failed to bracket the target CD."
    clear_optics = dataclasses.replace(tiny_cfg.optics, normalisation="clear")
    cd_nm = evaluate_cd(
        tiny_mask, clear_optics, tiny_cfg.grid, tiny_cfg.resist,
        dose=dose, defocus_m=0.0,
    ) * 1e9
    assert abs(cd_nm - target) < 0.1, (
        f"Dose-to-size dose {dose:.4f} prints {cd_nm:.3f} nm, not {target} nm."
    )


def test_dose_to_size_returns_nan_when_target_unreachable(tiny_cfg, tiny_mask):
    # The widest line this image ever prints is ~103 nm, right at the
    # emergence discontinuity; 150 nm is unreachable at any dose.
    dose = calibrate_dose_to_size(
        tiny_mask, tiny_cfg.optics, tiny_cfg.grid, tiny_cfg.resist,
        target_cd_nm=150.0,
    )
    assert np.isnan(dose), (
        "The fixture cannot print 150 nm at any dose; calibration must say "
        "so with NaN rather than return a dose that lies."
    )


def test_el_dof_curve_shape_on_analytic_frame():
    df = _analytic_frame()
    curve = compute_el_dof_curve(df, target_cd_nm=_TARGET, tolerance_pct=10.0)
    assert not curve.empty
    assert curve["dof_nm"].is_monotonic_increasing
    # cd falls with |f|, so a wider focus window admits a narrower dose band.
    assert curve["el_pct"].is_monotonic_decreasing
    # At DOF 0 the band is node-resolution: doses 0.80…1.20 → 40 %.
    assert curve.loc[curve["dof_nm"] == 0.0, "el_pct"].iloc[0] == pytest.approx(40.0)


def test_meef_sane_on_lines_and_spaces(tiny_cfg):
    # A well-resolved, commensurate fixture: two full 256 nm periods with the
    # 100 nm dark line interior to the array. The 64-px tiny fixture operates
    # at its resolution cliff, where an 8 nm bias genuinely stops the pattern
    # printing — real physics, but MEEF is undefined there.
    from litho_sim.core.config import GridConfig
    from litho_sim.mask.patterns import lines_and_spaces

    grid = GridConfig(n_pixels=128, pixel_size=4e-9)
    mask = lines_and_spaces(128, 4e-9, pitch=256e-9, cd=156e-9)
    # Evaluate at the dose that actually sizes the feature — raw dose 1.0
    # under clear normalisation may not print at all, and that is exactly
    # how the app tab anchors its MEEF call too.
    dose = calibrate_dose_to_size(
        mask, tiny_cfg.optics, grid, tiny_cfg.resist, target_cd_nm=100.0,
    )
    assert np.isfinite(dose)
    meef = compute_meef(mask, tiny_cfg.optics, grid, tiny_cfg.resist, dose=dose)
    assert np.isfinite(meef), "MEEF returned NaN on a printing L/S pattern."
    assert 0.2 < meef < 8.0, (
        f"MEEF {meef:.2f} outside any physically plausible band for L/S."
    )


def test_meef_rejects_sub_pixel_bias(tiny_cfg, tiny_mask):
    with pytest.raises(ValueError, match="below one grid pixel"):
        compute_meef(
            tiny_mask, tiny_cfg.optics, tiny_cfg.grid, tiny_cfg.resist,
            bias_nm=1.0,  # grid pixel is 4 nm
        )

