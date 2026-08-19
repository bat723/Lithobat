"""
Design of experiments: repeatability, reproducibility and Gage R&R.

A metrology number is worthless without knowing how much of it is the wafer and
how much is the gauge.  This module runs the whole pipeline -- synthesise,
interfere, reconstruct, measure flatness -- many times over a crossed
part x operator x trial design and decomposes the observed variation.

Design
------
* **Part** -- a distinct wafer (distinct synthesis seed).  Contributes the
  *real* variation a good gauge must resolve.
* **Operator** -- a distinct tool setup, simulated as a different phase-step
  calibration error.  A miscalibrated actuator biases every measurement that
  operator takes, which is exactly what reproducibility (AV) captures.
* **Trial** -- a repeat measurement of the same wafer by the same operator,
  differing only in detector noise seed.  Contributes repeatability (EV).

Variance components follow the standard crossed ANOVA method (AIAG MSA):

===========  ===========================================================
Term         Meaning
===========  ===========================================================
``EV``       Equipment variation -- repeatability, from the residual MS.
``AV``       Appraiser variation -- reproducibility, operator + interaction.
``GRR``      ``sqrt(EV^2 + AV^2)`` -- total gauge error.
``PV``       Part variation -- genuine wafer-to-wafer spread.
``TV``       Total variation, ``sqrt(GRR^2 + PV^2)``.
``%GRR``     ``100 * GRR / TV``.  Under 10 % is acceptable, over 30 % fails.
``ndc``      Number of distinct categories, ``1.41 * PV / GRR``; >= 5 wanted.
===========  ===========================================================

Every table is a tidy :class:`pandas.DataFrame`, and :func:`export_for_jmp`
writes a long-format CSV ready to drop into JMP's Variability/Gauge platform.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import stats

from .flatness import DEFAULT_EDGE_EXCLUSION, DEFAULT_SITE_SIZE, compute_flatness
from .interferometry import measure_surface, synthetic_wavelength
from .plotting import PALETTE, series_style
from .synthesize import WaferSurface, synthesize_wafer

logger = logging.getLogger(__name__)

DEFAULT_SYNTHETIC_WAVELENGTH: float = synthetic_wavelength(632.8e-9, 640.0e-9)
"""Synthetic wavelength [m] (~56 µm) able to span a warped wafer unambiguously."""

RESPONSES: Tuple[str, ...] = ("warp_um", "bow_um", "sfqr_max_nm", "sfqr_mean_nm")
"""Response variables tracked through the study.

Both SFQR statistics are carried deliberately: the *max* is an extreme value
over ~90 sites, so it inherits the worst noise excursion on the wafer and makes
a jumpy gauge, while the *mean* averages that noise down.  The Gage R&R report
shows the two side by side, which is the argument for monitoring a robust
statistic rather than a max.
"""

RESPONSE_LABELS: Dict[str, str] = {
    "warp_um": "Warp [µm]",
    "bow_um": "Bow [µm]",
    "sfqr_max_nm": "SFQR max [nm]",
    "sfqr_mean_nm": "SFQR mean [nm]",
}


# ---------------------------------------------------------------------------
# Single measurement run
# ---------------------------------------------------------------------------


def measure_once(
    surface: WaferSurface,
    *,
    noise: float,
    step_error: float,
    seed: int,
    wavelength: float = DEFAULT_SYNTHETIC_WAVELENGTH,
    n_steps: int = 4,
    site_size: float = DEFAULT_SITE_SIZE,
    edge_exclusion: float = DEFAULT_EDGE_EXCLUSION,
) -> Dict[str, float]:
    """Measure one wafer once and return its flatness metrics.

    The chain is the production one: interferometric acquisition, phase
    recovery, unwrapping, then SEMI flatness on the *reconstructed* surface --
    so every metric inherits the measurement's noise and bias.

    Parameters
    ----------
    surface : WaferSurface
        The wafer being measured (the "part").
    noise : float
        Detector noise as a fraction of mean intensity.
    step_error : float
        Fractional phase-step calibration error (the operator effect).
    seed : int
        RNG seed for this acquisition.
    wavelength : float
        Illumination or synthetic wavelength [m].
    n_steps : int
        Number of phase steps.
    site_size : float
        Exposure site edge length [m].
    edge_exclusion : float
        Fixed quality area edge exclusion [m].

    Returns
    -------
    dict
        ``warp_um``, ``bow_um``, ``sfqr_max_nm``, ``sfqr_mean_nm`` and
        ``rms_error_nm`` (the reconstruction error against truth).
    """
    measurement = measure_surface(
        surface, wavelength, n_steps=n_steps, noise=noise, step_error=step_error, seed=seed
    )
    reconstructed = surface.with_z(measurement.z_measured)
    result = compute_flatness(
        reconstructed, site_size=site_size, edge_exclusion=edge_exclusion
    )
    metrics = result.metrics
    return {
        "warp_um": metrics.warp * 1e6,
        "bow_um": metrics.bow * 1e6,
        "sfqr_max_nm": metrics.sfqr_max * 1e9,
        "sfqr_mean_nm": metrics.sfqr_mean * 1e9,
        "rms_error_nm": measurement.rms_error * 1e9,
    }


# ---------------------------------------------------------------------------
# Study drivers
# ---------------------------------------------------------------------------


def run_gauge_rr(
    *,
    n_parts: int = 3,
    n_operators: int = 3,
    n_trials: int = 3,
    n_pixels: int = 256,
    noise: float = 0.01,
    operator_step_errors: Optional[Sequence[float]] = None,
    wavelength: float = DEFAULT_SYNTHETIC_WAVELENGTH,
    seed: int = 0,
) -> pd.DataFrame:
    """Run a crossed part x operator x trial Gage R&R study.

    Each part is synthesised once and measured by every operator on every
    trial, so part variation is genuinely common across the design.

    Parameters
    ----------
    n_parts : int
        Number of distinct wafers.
    n_operators : int
        Number of simulated operators / tool setups.
    n_trials : int
        Repeat measurements per part-operator cell.  Must be at least 2 for
        the residual (repeatability) term to be estimable.
    n_pixels : int
        Grid size for the study; smaller keeps the run quick.
    noise : float
        Detector noise fraction, common to all operators.
    operator_step_errors : sequence of float, optional
        Per-operator phase-step calibration error.  Defaults to a spread
        centred on zero, so operators differ but none is grossly wrong.
    wavelength : float
        Measurement wavelength [m].
    seed : int
        Base seed; part and acquisition seeds derive from it deterministically.

    Returns
    -------
    DataFrame
        Long-format results, one row per measurement, with columns ``part``,
        ``operator``, ``trial``, ``step_error``, ``noise`` and the responses.

    Raises
    ------
    ValueError
        If *n_trials* is less than 2.
    """
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2 to estimate repeatability, got {n_trials}")

    if operator_step_errors is None:
        operator_step_errors = np.linspace(-0.02, 0.02, n_operators)
    if len(operator_step_errors) != n_operators:
        raise ValueError("operator_step_errors length must equal n_operators")

    rows: List[dict] = []
    for part in range(n_parts):
        # One physical wafer per part, measured repeatedly.
        surface = synthesize_wafer(n_pixels=n_pixels, seed=seed * 1000 + part)
        for operator in range(n_operators):
            for trial in range(n_trials):
                acquisition_seed = (
                    (seed + 1) * 1_000_000 + part * 10_000 + operator * 100 + trial
                )
                metrics = measure_once(
                    surface,
                    noise=noise,
                    step_error=float(operator_step_errors[operator]),
                    seed=acquisition_seed,
                    wavelength=wavelength,
                )
                rows.append(
                    {
                        "part": f"W{part + 1}",
                        "operator": f"Op{operator + 1}",
                        "trial": trial + 1,
                        "step_error": float(operator_step_errors[operator]),
                        "noise": noise,
                        **metrics,
                    }
                )
        logger.info("Gage R&R: part %d/%d complete", part + 1, n_parts)

    frame = pd.DataFrame(rows)
    logger.info(
        "Gage R&R study: %d measurements (%d parts x %d operators x %d trials)",
        len(frame), n_parts, n_operators, n_trials,
    )
    return frame


def run_noise_sweep(
    *,
    noise_levels: Sequence[float] = (0.002, 0.005, 0.01, 0.02, 0.05),
    n_repeats: int = 8,
    n_pixels: int = 256,
    wafer_seed: int = 0,
    wavelength: float = DEFAULT_SYNTHETIC_WAVELENGTH,
    step_error: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Repeat-measure one wafer across several detector noise levels.

    Holding the wafer fixed isolates pure repeatability: every bit of spread in
    the responses is gauge error.

    Parameters
    ----------
    noise_levels : sequence of float
        Detector noise fractions to sweep.
    n_repeats : int
        Repeat measurements per noise level.
    n_pixels : int
        Grid size.
    wafer_seed : int
        Seed of the single wafer under test.
    wavelength : float
        Measurement wavelength [m].
    step_error : float
        Phase-step calibration error held during the sweep.
    seed : int
        Base seed for the acquisition noise.

    Returns
    -------
    DataFrame
        One row per measurement with columns ``noise``, ``repeat`` and the
        responses.
    """
    surface = synthesize_wafer(n_pixels=n_pixels, seed=wafer_seed)
    rows: List[dict] = []
    for level_index, level in enumerate(noise_levels):
        for repeat in range(n_repeats):
            metrics = measure_once(
                surface,
                noise=float(level),
                step_error=step_error,
                seed=(seed + 7) * 100_000 + level_index * 1_000 + repeat,
                wavelength=wavelength,
            )
            rows.append({"noise": float(level), "repeat": repeat + 1, **metrics})
        logger.info("Noise sweep: level %.4g complete", level)
    return pd.DataFrame(rows)


def repeatability_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """Summarise a noise sweep into per-level means and standard deviations.

    Parameters
    ----------
    sweep : DataFrame
        Output of :func:`run_noise_sweep`.

    Returns
    -------
    DataFrame
        Columns ``noise, n`` plus ``<response>_mean``, ``<response>_std`` and
        ``<response>_3sigma`` for each response.
    """
    grouped = sweep.groupby("noise")
    out = pd.DataFrame({"noise": sorted(sweep["noise"].unique())}).set_index("noise")
    out["n"] = grouped.size()
    for response in RESPONSES:
        out[f"{response}_mean"] = grouped[response].mean()
        out[f"{response}_std"] = grouped[response].std(ddof=1)
        out[f"{response}_3sigma"] = 3.0 * grouped[response].std(ddof=1)
    return out.reset_index()


# ---------------------------------------------------------------------------
# ANOVA and variance components
# ---------------------------------------------------------------------------


def anova_table(
    data: pd.DataFrame,
    response: str,
    *,
    part_col: str = "part",
    operator_col: str = "operator",
) -> pd.DataFrame:
    """Two-way crossed ANOVA with interaction for a balanced design.

    Parameters
    ----------
    data : DataFrame
        Long-format measurements.
    response : str
        Response column to analyse.
    part_col, operator_col : str
        Factor column names.

    Returns
    -------
    DataFrame
        Rows ``Part, Operator, Part*Operator, Repeatability, Total`` with
        columns ``source, SS, df, MS, F, p``.

    Raises
    ------
    ValueError
        If the design is unbalanced or has no replication.
    """
    counts = data.groupby([part_col, operator_col]).size()
    if counts.nunique() != 1:
        raise ValueError("Gage R&R ANOVA requires a balanced design")
    n_trials = int(counts.iloc[0])
    if n_trials < 2:
        raise ValueError("need at least 2 trials per cell to estimate repeatability")

    n_parts = data[part_col].nunique()
    n_operators = data[operator_col].nunique()
    values = data[response].to_numpy(dtype=np.float64)
    grand = values.mean()

    part_means = data.groupby(part_col)[response].mean()
    op_means = data.groupby(operator_col)[response].mean()
    cell_means = data.groupby([part_col, operator_col])[response].mean()

    ss_total = float(((values - grand) ** 2).sum())
    ss_part = float(n_operators * n_trials * ((part_means - grand) ** 2).sum())
    ss_op = float(n_parts * n_trials * ((op_means - grand) ** 2).sum())

    interaction = 0.0
    for (part, operator), cell_mean in cell_means.items():
        interaction += (cell_mean - part_means[part] - op_means[operator] + grand) ** 2
    ss_int = float(n_trials * interaction)
    ss_err = float(ss_total - ss_part - ss_op - ss_int)

    df_part = n_parts - 1
    df_op = n_operators - 1
    df_int = df_part * df_op
    df_err = n_parts * n_operators * (n_trials - 1)

    ms_part = ss_part / df_part if df_part else np.nan
    ms_op = ss_op / df_op if df_op else np.nan
    ms_int = ss_int / df_int if df_int else np.nan
    ms_err = ss_err / df_err if df_err else np.nan

    # Parts and operators are tested against the interaction, interaction
    # against the residual -- the standard random-effects expected mean squares.
    f_part = ms_part / ms_int if (df_int and ms_int) else np.nan
    f_op = ms_op / ms_int if (df_int and ms_int) else np.nan
    f_int = ms_int / ms_err if (df_err and ms_err) else np.nan

    def pvalue(f: float, dfn: int, dfd: int) -> float:
        if not np.isfinite(f) or dfn <= 0 or dfd <= 0 or f <= 0:
            return float("nan")
        return float(stats.f.sf(f, dfn, dfd))

    return pd.DataFrame(
        [
            ("Part", ss_part, df_part, ms_part, f_part, pvalue(f_part, df_part, df_int)),
            ("Operator", ss_op, df_op, ms_op, f_op, pvalue(f_op, df_op, df_int)),
            ("Part*Operator", ss_int, df_int, ms_int, f_int, pvalue(f_int, df_int, df_err)),
            ("Repeatability", ss_err, df_err, ms_err, np.nan, np.nan),
            ("Total", ss_total, len(values) - 1, np.nan, np.nan, np.nan),
        ],
        columns=["source", "SS", "df", "MS", "F", "p"],
    )


@dataclass
class GaugeRR:
    """Variance-component breakdown of a Gage R&R study, in response units.

    Attributes
    ----------
    response : str
        Response analysed.
    ev : float
        Equipment variation (repeatability) standard deviation.
    av : float
        Appraiser variation (reproducibility) standard deviation.
    grr : float
        Total gauge R&R standard deviation.
    pv : float
        Part variation standard deviation.
    tv : float
        Total variation standard deviation.
    pct_grr : float
        ``100 * grr / tv``.
    pct_ev, pct_av, pct_pv : float
        Component percentages of total variation.
    ndc : int
        Number of distinct categories the gauge can resolve.
    """

    response: str
    ev: float
    av: float
    grr: float
    pv: float
    tv: float
    pct_grr: float
    pct_ev: float
    pct_av: float
    pct_pv: float
    ndc: int

    @property
    def verdict(self) -> str:
        """AIAG acceptability verdict from ``pct_grr``."""
        if not np.isfinite(self.pct_grr):
            return "undetermined"
        if self.pct_grr < 10.0:
            return "acceptable"
        if self.pct_grr < 30.0:
            return "marginal"
        return "unacceptable"

    def to_frame(self) -> pd.DataFrame:
        """Return the classic Gage R&R component table.

        Returns
        -------
        DataFrame
            Columns ``component, std_dev, pct_of_total``.
        """
        rows = [
            ("Repeatability (EV)", self.ev, self.pct_ev),
            ("Reproducibility (AV)", self.av, self.pct_av),
            ("Gauge R&R", self.grr, self.pct_grr),
            ("Part variation (PV)", self.pv, self.pct_pv),
            ("Total variation (TV)", self.tv, 100.0),
        ]
        return pd.DataFrame(rows, columns=["component", "std_dev", "pct_of_total"])


def gauge_rr(
    data: pd.DataFrame,
    response: str,
    *,
    part_col: str = "part",
    operator_col: str = "operator",
) -> GaugeRR:
    """Estimate Gage R&R variance components by the ANOVA method.

    Negative variance estimates -- which happen when a true effect is near zero
    and noise makes the mean square smaller than the term below it -- are
    clamped to zero, as AIAG prescribes.

    Parameters
    ----------
    data : DataFrame
        Long-format measurements from :func:`run_gauge_rr`.
    response : str
        Response column to analyse.
    part_col, operator_col : str
        Factor column names.

    Returns
    -------
    GaugeRR
        The variance-component breakdown.

    Examples
    --------
    >>> frame = run_gauge_rr(n_parts=2, n_operators=2, n_trials=2, n_pixels=96)
    >>> result = gauge_rr(frame, "warp_um")
    >>> bool(0.0 <= result.pct_grr <= 100.0)
    True
    """
    table = anova_table(data, response, part_col=part_col, operator_col=operator_col)
    ms = table.set_index("source")["MS"]

    counts = data.groupby([part_col, operator_col]).size()
    n_trials = int(counts.iloc[0])
    n_parts = data[part_col].nunique()
    n_operators = data[operator_col].nunique()

    var_repeat = float(ms["Repeatability"])
    var_int = max(0.0, float(ms["Part*Operator"] - ms["Repeatability"]) / n_trials)
    var_op = max(0.0, float(ms["Operator"] - ms["Part*Operator"]) / (n_parts * n_trials))
    var_part = max(0.0, float(ms["Part"] - ms["Part*Operator"]) / (n_operators * n_trials))

    ev = float(np.sqrt(max(var_repeat, 0.0)))
    av = float(np.sqrt(var_op + var_int))
    grr = float(np.sqrt(ev**2 + av**2))
    pv = float(np.sqrt(var_part))
    tv = float(np.sqrt(grr**2 + pv**2))

    def pct(value: float) -> float:
        return 100.0 * value / tv if tv > 0 else float("nan")

    ndc = int(np.floor(1.41 * pv / grr)) if grr > 0 else 0
    result = GaugeRR(
        response=response, ev=ev, av=av, grr=grr, pv=pv, tv=tv,
        pct_grr=pct(grr), pct_ev=pct(ev), pct_av=pct(av), pct_pv=pct(pv), ndc=max(ndc, 0),
    )
    logger.info(
        "Gage R&R [%s]: %%GRR %.1f%% (%s), EV %.4g, AV %.4g, PV %.4g, ndc %d",
        response, result.pct_grr, result.verdict, ev, av, pv, result.ndc,
    )
    return result


def gauge_rr_all(
    data: pd.DataFrame, responses: Sequence[str] = RESPONSES
) -> pd.DataFrame:
    """Run :func:`gauge_rr` for several responses and stack the summaries.

    Parameters
    ----------
    data : DataFrame
        Long-format measurements.
    responses : sequence of str
        Response columns to analyse.

    Returns
    -------
    DataFrame
        One row per response with every component and the verdict.
    """
    rows = []
    for response in responses:
        result = gauge_rr(data, response)
        row = asdict(result)
        row["verdict"] = result.verdict
        rows.append(row)
    return pd.DataFrame(rows)


def export_for_jmp(data: pd.DataFrame, path: str | Path) -> Path:
    """Write the study to a long-format CSV for JMP's Variability/Gauge platform.

    JMP wants one row per measurement with the factors as columns, which is
    exactly the shape :func:`run_gauge_rr` already produces; this helper just
    orders the columns predictably.

    Parameters
    ----------
    data : DataFrame
        Long-format measurements.
    path : str or Path
        Destination CSV.

    Returns
    -------
    Path
        The resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    leading = [c for c in ("part", "operator", "trial", "noise", "step_error")
               if c in data.columns]
    ordered = data[leading + [c for c in data.columns if c not in leading]]
    ordered.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows for JMP)", out, len(ordered))
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_doe_summary(
    sweep: pd.DataFrame,
    study: pd.DataFrame,
    components: pd.DataFrame,
    *,
    response: str = "warp_um",
    title: str = "Metrology repeatability and Gage R&R",
    figsize: Tuple[float, float] = (13.5, 9.0),
) -> Tuple[Figure, NDArray]:
    """Four-panel DOE summary.

    Panels: repeatability vs noise for two responses (separate panels, never a
    shared twin axis), the variance-component breakdown, and a run chart of the
    crossed study.

    Parameters
    ----------
    sweep : DataFrame
        Output of :func:`run_noise_sweep`.
    study : DataFrame
        Output of :func:`run_gauge_rr`.
    components : DataFrame
        Output of :func:`gauge_rr_all`.
    response : str
        Response featured in the run chart and component panel.
    title : str
        Figure suptitle.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    import matplotlib.pyplot as plt

    summary = repeatability_summary(sweep)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    # Panels 1-2: measurement sigma vs detector noise, one response each.
    for ax, resp in zip(axes[:2], ("warp_um", "sfqr_mean_nm")):
        ax.plot(
            summary["noise"] * 100.0, summary[f"{resp}_std"],
            **series_style(0), label=f"1σ of {RESPONSE_LABELS[resp]}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Detector noise [% of mean intensity]")
        ax.set_ylabel(f"Repeatability 1σ [{RESPONSE_LABELS[resp].split('[')[1].rstrip(']')}]")
        ax.set_title(f"Repeatability vs. noise — {RESPONSE_LABELS[resp]}")
        ax.legend(loc="upper left")

    # Panel 3: variance components for the featured response.
    ax = axes[2]
    row = components.loc[components["response"] == response]
    if len(row):
        row = row.iloc[0]
        names = ["EV\n(repeat)", "AV\n(reprod)", "GRR", "PV\n(part)"]
        values = [row["pct_ev"], row["pct_av"], row["pct_grr"], row["pct_pv"]]
        colours = [PALETTE[0], PALETTE[1], PALETTE[5], PALETTE[3]]
        ax.bar(names, values, color=colours, edgecolor="white", linewidth=0.6)
        ax.axhline(10.0, color=PALETTE[6], ls="--", lw=1.2, label="10 % acceptable")
        ax.axhline(30.0, color=PALETTE[5], ls=":", lw=1.2, label="30 % unacceptable")
        for i, value in enumerate(values):
            ax.annotate(f"{value:.1f}%", (i, value), ha="center",
                        textcoords="offset points", xytext=(0, 3), fontsize=8.5)
        ax.set_ylabel("% of total variation")
        ax.set_title(
            f"Variance components — {RESPONSE_LABELS.get(response, response)}\n"
            f"%GRR {row['pct_grr']:.1f}% ({row['verdict']}), ndc {int(row['ndc'])}"
        )
        ax.legend(loc="upper center", fontsize=8)

    # Panel 4: run chart, operators as distinct colour+marker series.
    ax = axes[3]
    parts = sorted(study["part"].unique())
    positions = {part: i for i, part in enumerate(parts)}
    for idx, operator in enumerate(sorted(study["operator"].unique())):
        subset = study.loc[study["operator"] == operator]
        cell_means = subset.groupby("part")[response].mean()
        ax.plot(
            [positions[p] for p in cell_means.index], cell_means.to_numpy(),
            label=operator, linestyle="-", **series_style(idx),
        )
        ax.scatter(
            [positions[p] + (idx - 1) * 0.06 for p in subset["part"]],
            subset[response],
            color=series_style(idx)["color"], s=14, alpha=0.55, zorder=2,
        )
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(parts)
    ax.set_xlabel("Part (wafer)")
    ax.set_ylabel(RESPONSE_LABELS.get(response, response))
    ax.set_title("Run chart: part × operator cell means")
    ax.legend(title="Operator", loc="best", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig, axes
