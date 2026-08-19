"""
Unit tests for the DOE / Gage R&R module.

The ANOVA and variance-component maths is checked against synthetic data with
*known* components, so the tests pin the estimator rather than the pipeline.
A small end-to-end study then confirms the whole chain runs and is repeatable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafer_metrology.doe import (
    DEFAULT_SYNTHETIC_WAVELENGTH,
    RESPONSES,
    anova_table,
    export_for_jmp,
    gauge_rr,
    gauge_rr_all,
    measure_once,
    repeatability_summary,
    run_gauge_rr,
    run_noise_sweep,
)
from wafer_metrology.synthesize import synthesize_wafer


def synthetic_study(
    *,
    n_parts: int = 8,
    n_operators: int = 3,
    n_trials: int = 4,
    part_sd: float = 10.0,
    operator_sd: float = 0.0,
    repeat_sd: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a crossed study with known variance components.

    Part and operator effects are drawn randomly and then *standardised* to have
    exactly the requested sample standard deviation.  Without that step a design
    with only three operators would realise an operator spread far from its
    population value -- three draws estimate a standard deviation very poorly --
    and the test would be checking the RNG rather than the estimator.  Only the
    within-cell noise is left random, and it has plenty of degrees of freedom.

    Parameters
    ----------
    n_parts, n_operators, n_trials : int
        Design size.
    part_sd : float
        Realised part-to-part standard deviation.
    operator_sd : float
        Realised operator offset standard deviation.
    repeat_sd : float
        True within-cell (repeatability) standard deviation.
    seed : int
        RNG seed.

    Returns
    -------
    DataFrame
        Long-format measurements with a ``value`` response.
    """
    rng = np.random.default_rng(seed)

    def standardised(size: int, target_sd: float) -> np.ndarray:
        """Centred effects whose ddof=1 sample SD is exactly *target_sd*."""
        if target_sd <= 0.0 or size < 2:
            return np.zeros(size)
        raw = rng.normal(0.0, 1.0, size=size)
        raw -= raw.mean()
        return raw * (target_sd / raw.std(ddof=1))

    part_effect = standardised(n_parts, part_sd)
    operator_effect = standardised(n_operators, operator_sd)

    rows = []
    for p in range(n_parts):
        for o in range(n_operators):
            for t in range(n_trials):
                rows.append(
                    {
                        "part": f"P{p}",
                        "operator": f"O{o}",
                        "trial": t,
                        "value": 100.0 + part_effect[p] + operator_effect[o]
                        + rng.normal(0.0, repeat_sd),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------


def test_sum_of_squares_decomposes_the_total():
    """Part, operator, interaction and error sum exactly to the total SS."""
    data = synthetic_study(operator_sd=2.0, seed=1)

    table = anova_table(data, "value").set_index("source")
    parts = table.loc[["Part", "Operator", "Part*Operator", "Repeatability"], "SS"].sum()

    assert parts == pytest.approx(table.loc["Total", "SS"], rel=1e-10)


def test_degrees_of_freedom_are_correct():
    """df follow the standard crossed-design formulas."""
    data = synthetic_study(n_parts=5, n_operators=3, n_trials=4)

    table = anova_table(data, "value").set_index("source")

    assert table.loc["Part", "df"] == 4
    assert table.loc["Operator", "df"] == 2
    assert table.loc["Part*Operator", "df"] == 8
    assert table.loc["Repeatability", "df"] == 5 * 3 * 3
    assert table.loc["Total", "df"] == 5 * 3 * 4 - 1


def test_part_effect_is_detected_as_significant():
    """A large part effect yields a tiny p-value and dominates the mean squares.

    The operator term is checked by its mean square rather than its p-value: a
    p-value under a true null is uniformly distributed, so asserting it exceeds
    0.05 would fail one run in twenty.
    """
    data = synthetic_study(part_sd=10.0, operator_sd=0.0, repeat_sd=1.0, seed=2)

    table = anova_table(data, "value").set_index("source")

    assert table.loc["Part", "p"] < 1e-6
    assert table.loc["Operator", "MS"] < 0.05 * table.loc["Part", "MS"]


def test_unbalanced_design_is_rejected():
    """An unbalanced design is refused rather than silently mis-analysed."""
    data = synthetic_study(n_trials=3).iloc[:-1]      # drop one row
    with pytest.raises(ValueError, match="balanced"):
        anova_table(data, "value")


def test_no_replication_is_rejected():
    """One trial per cell leaves repeatability inestimable."""
    data = synthetic_study(n_trials=1)
    with pytest.raises(ValueError, match="at least 2 trials"):
        anova_table(data, "value")


# ---------------------------------------------------------------------------
# Variance components
# ---------------------------------------------------------------------------


def test_components_recover_known_standard_deviations():
    """EV, AV and PV estimate the standard deviations they were generated from."""
    data = synthetic_study(
        n_parts=12, n_operators=3, n_trials=6,
        part_sd=10.0, operator_sd=3.0, repeat_sd=1.0, seed=3,
    )

    result = gauge_rr(data, "value")

    assert result.ev == pytest.approx(1.0, rel=0.10)
    assert result.av == pytest.approx(3.0, rel=0.20)
    assert result.pv == pytest.approx(10.0, rel=0.10)


def test_no_operator_effect_gives_near_zero_reproducibility():
    """With identical operators, AV collapses and GRR is essentially EV.

    Repeatability is set well below part variation so %GRR lands near 5 %,
    clear of the 10 % verdict boundary -- at repeat_sd = 1.0 against
    part_sd = 10.0 the ratio is 9.95 %, and the verdict would flip on noise.
    """
    data = synthetic_study(
        n_parts=12, n_operators=3, n_trials=6,
        part_sd=10.0, operator_sd=0.0, repeat_sd=0.5, seed=4,
    )

    result = gauge_rr(data, "value")

    assert result.av < 0.25
    assert result.grr == pytest.approx(result.ev, rel=0.2)
    assert result.pct_grr < 10.0
    assert result.verdict == "acceptable"


def test_a_noisy_gauge_is_reported_as_unacceptable():
    """When repeatability rivals part variation, %GRR fails and ndc collapses."""
    data = synthetic_study(
        n_parts=10, n_operators=3, n_trials=5,
        part_sd=1.0, operator_sd=0.0, repeat_sd=3.0, seed=5,
    )

    result = gauge_rr(data, "value")

    assert result.pct_grr > 30.0
    assert result.verdict == "unacceptable"
    assert result.ndc <= 1


def test_percentages_are_consistent_with_the_standard_deviations():
    """The reported percentages are the components divided by total variation."""
    data = synthetic_study(operator_sd=2.0, seed=6)
    result = gauge_rr(data, "value")

    assert result.tv == pytest.approx(np.hypot(result.grr, result.pv), rel=1e-12)
    assert result.grr == pytest.approx(np.hypot(result.ev, result.av), rel=1e-12)
    assert result.pct_grr == pytest.approx(100.0 * result.grr / result.tv, rel=1e-12)
    frame = result.to_frame()
    assert list(frame["component"])[0] == "Repeatability (EV)"
    assert frame["pct_of_total"].iloc[-1] == pytest.approx(100.0)


def test_variance_components_are_never_negative():
    """Negative variance estimates are clamped to zero, as AIAG prescribes."""
    # Part variation far below the noise drives the raw estimate negative.
    data = synthetic_study(n_parts=3, n_operators=3, n_trials=3,
                           part_sd=1e-6, operator_sd=1e-6, repeat_sd=5.0, seed=7)

    result = gauge_rr(data, "value")

    assert result.pv >= 0.0
    assert result.av >= 0.0
    assert result.ev > 0.0
    assert result.ndc >= 0


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_measure_once_returns_every_response():
    """A single simulated measurement reports all tracked responses."""
    surface = synthesize_wafer(n_pixels=128, seed=0)

    metrics = measure_once(surface, noise=0.01, step_error=0.0, seed=1)

    for response in RESPONSES:
        assert response in metrics
        assert np.isfinite(metrics[response])
    assert metrics["rms_error_nm"] > 0.0


def test_measure_once_is_deterministic():
    """The same seed gives the same numbers; a different seed does not."""
    surface = synthesize_wafer(n_pixels=128, seed=0)

    a = measure_once(surface, noise=0.02, step_error=0.0, seed=5)
    b = measure_once(surface, noise=0.02, step_error=0.0, seed=5)
    c = measure_once(surface, noise=0.02, step_error=0.0, seed=6)

    assert a["warp_um"] == pytest.approx(b["warp_um"], rel=1e-15)
    assert a["warp_um"] != pytest.approx(c["warp_um"], rel=1e-12)


def test_gauge_rr_study_has_the_expected_shape():
    """The crossed design produces one row per part x operator x trial."""
    study = run_gauge_rr(n_parts=2, n_operators=2, n_trials=2, n_pixels=96)

    assert len(study) == 2 * 2 * 2
    assert set(study["part"]) == {"W1", "W2"}
    assert set(study["operator"]) == {"Op1", "Op2"}
    # Operators differ by their phase-step calibration error.
    assert study.groupby("operator")["step_error"].nunique().eq(1).all()
    assert study["step_error"].nunique() == 2


def test_gauge_rr_needs_replication():
    """A single trial per cell is refused up front."""
    with pytest.raises(ValueError, match="n_trials"):
        run_gauge_rr(n_parts=2, n_operators=2, n_trials=1, n_pixels=64)


def test_gauge_rr_all_covers_every_response():
    """The summary table has one row per response with a verdict."""
    study = run_gauge_rr(n_parts=2, n_operators=2, n_trials=2, n_pixels=96)

    summary = gauge_rr_all(study)

    assert list(summary["response"]) == list(RESPONSES)
    assert summary["verdict"].isin({"acceptable", "marginal", "unacceptable"}).all()
    assert (summary["pct_grr"] >= 0).all()


def test_noise_sweep_repeatability_grows_with_noise():
    """More detector noise widens the spread of repeated measurements."""
    sweep = run_noise_sweep(
        noise_levels=(0.002, 0.05), n_repeats=5, n_pixels=128,
        wavelength=DEFAULT_SYNTHETIC_WAVELENGTH,
    )

    summary = repeatability_summary(sweep)

    assert len(summary) == 2
    quiet = summary.loc[summary["noise"] == 0.002, "warp_um_std"].iloc[0]
    loud = summary.loc[summary["noise"] == 0.05, "warp_um_std"].iloc[0]
    assert loud > quiet
    assert (summary["n"] == 5).all()


def test_export_for_jmp_writes_factors_first(tmp_path):
    """The exported CSV leads with the factor columns JMP expects."""
    study = run_gauge_rr(n_parts=2, n_operators=2, n_trials=2, n_pixels=96)

    path = export_for_jmp(study, tmp_path / "runs.csv")
    reloaded = pd.read_csv(path)

    assert path.exists()
    assert list(reloaded.columns)[:5] == ["part", "operator", "trial", "noise", "step_error"]
    assert len(reloaded) == len(study)
