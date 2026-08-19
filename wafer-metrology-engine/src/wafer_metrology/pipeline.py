"""
End-to-end pipeline: synthesise -> decompose -> measure -> analyse -> qualify.

Runs every module in order, writes each figure and table to the results
directory, and prints the metric tables.  This is both the demo and the
regression harness: if the physics is wrong, the printed numbers move.

Run it from the project root::

    python run.py                 # full run, 512 px grid
    python run.py --quick         # 256 px, smaller DOE, no autoencoder
    python run.py --help

or, once installed, via the ``wafer-metrology`` console script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import HENE_WAVELENGTH, WAFER_DIAMETER_150MM
from .coupling import LossBudget, attach_loss_budget, plot_loss_budget
from .defects import (
    HAVE_TORCH,
    DefectReport,
    detect_defects,
    detect_defects_ml,
    plot_defect_map,
    score_detections,
    train_autoencoder,
)
from .deflectometry import (
    DeflectometryMeasurement,
    DeflectometrySetup,
    absolute_calibrate,
    fit_gamma,
    measure_deflectometry,
    plane_aligned_difference,
    plot_deflectograms,
    plot_deflectometry_recovery,
    plot_reversal,
    reversal_calibrate,
    rotate_surface_180,
    simulate_gamma_sweep,
)
from .doe import (
    DEFAULT_SYNTHETIC_WAVELENGTH,
    export_for_jmp,
    gauge_rr_all,
    plot_doe_summary,
    repeatability_summary,
    run_gauge_rr,
    run_noise_sweep,
)
from .flatness import FlatnessResult, compute_flatness, plot_flatness
from .interferometry import (
    check_sampling,
    measure_surface,
    plot_interferogram,
    plot_reconstruction,
)
from .plotting import plot_wafer_panels, save_figure
from .synthesize import make_flat_surface, synthesize_wafer, synthesize_wafer_pair
from .zernike import (
    fit_surface,
    flatten,
    plot_coefficient_spectrum,
    plot_flatten_summary,
    reconstruct,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything the pipeline produced, for notebook and test consumption.

    Attributes
    ----------
    figures : list of Path
        Figure files written, in order.
    tables : dict
        Named CSV paths written.
    flatness_true : FlatnessResult
        Flatness of the ground-truth wafer (front and back known, so TTV is
        available).
    flatness_measured : FlatnessResult
        Flatness of the interferometrically reconstructed front surface.
    defects : DefectReport
        Classical defect extraction on the reconstructed surface.
    defects_ml : DefectReport or None
        Autoencoder extraction, when torch is available and enabled.
    gauge : DataFrame
        Gage R&R component summary.
    study : DataFrame
        Raw crossed-design measurements.
    sweep : DataFrame
        Raw noise-sweep measurements.
    rms_reconstruction_error : float
        RMS interferometric reconstruction error [m].
    """

    figures: List[Path] = field(default_factory=list)
    tables: Dict[str, Path] = field(default_factory=dict)
    flatness_true: Optional[FlatnessResult] = None
    flatness_measured: Optional[FlatnessResult] = None
    defects: Optional[DefectReport] = None
    defects_ml: Optional[DefectReport] = None
    gauge: Optional[pd.DataFrame] = None
    study: Optional[pd.DataFrame] = None
    sweep: Optional[pd.DataFrame] = None
    rms_reconstruction_error: float = float("nan")
    deflectometry: Optional[DeflectometryMeasurement] = None
    deflectometry_gate: float = float("nan")
    loss_budget: Optional[LossBudget] = None


def _banner(step: int, text: str) -> None:
    """Print a numbered section banner."""
    print(f"\n{'=' * 78}\n[{step}] {text}\n{'=' * 78}")


def run_pipeline(
    results_dir: str | Path = "results",
    *,
    n_pixels: int = 512,
    seed: int = 0,
    wavelength: float = DEFAULT_SYNTHETIC_WAVELENGTH,
    site_size: float = 25e-3,
    edge_exclusion: float = 3e-3,
    noise: float = 0.01,
    run_ml: bool = True,
    run_doe: bool = True,
    run_deflectometry: bool = True,
    run_capstone: bool = True,
    deflectometry_pixels: int = 256,
    doe_pixels: int = 256,
    doe_parts: int = 3,
    doe_operators: int = 3,
    doe_trials: int = 3,
    doe_repeats: int = 10,
) -> PipelineResult:
    """Execute the full metrology pipeline and write every artefact.

    Parameters
    ----------
    results_dir : str or Path
        Output directory for figures and tables.
    n_pixels : int
        Grid size for the headline wafer.
    seed : int
        Master seed; every stage derives deterministically from it.
    wavelength : float
        Measurement wavelength [m].  Defaults to the ~56 µm synthetic
        wavelength, which spans a warped 300 mm wafer without ambiguity.
    site_size : float
        Exposure site edge length [m].
    edge_exclusion : float
        Fixed quality area edge exclusion [m].
    noise : float
        Detector noise fraction for the headline measurement.
    run_ml : bool
        Train and apply the autoencoder defect path when torch is available.
    run_doe : bool
        Run the DOE / Gage R&R study.
    run_deflectometry : bool
        Run the hardware-matched deflectometry simulation (Leg A) with
        reversal and absolute calibration on a 150 mm wafer.
    run_capstone : bool
        Run the warp-to-loss capstone on the deflectometry result (requires
        *run_deflectometry*).
    deflectometry_pixels : int
        Grid size for the deflectometry stages.
    doe_pixels : int
        Grid size used inside the DOE (smaller keeps it quick).
    doe_parts, doe_operators, doe_trials : int
        Crossed design size.
    doe_repeats : int
        Repeats per noise level in the sweep.

    Returns
    -------
    PipelineResult
        Handles to every artefact and intermediate result.
    """
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = PipelineResult()

    def figure(fig, name: str) -> None:
        result.figures.append(save_figure(fig, out / name))

    # ------------------------------------------------------------------
    _banner(1, "Synthesise a 300 mm wafer (bow + warp + nanotopography + defects)")
    pair = synthesize_wafer_pair(n_pixels=n_pixels, seed=seed)
    surface = pair.front
    print(
        f"grid {n_pixels}×{n_pixels} px, pixel {pair.pixel_size * 1e3:.3f} mm, "
        f"PV {surface.pv() * 1e6:.2f} µm, RMS {surface.rms() * 1e6:.2f} µm, "
        f"{len(pair.defects)} injected defects"
    )
    fig, _ = plot_wafer_panels(
        pair.x, pair.y,
        [pair.z_front, pair.median, pair.thickness - np.nanmean(pair.thickness)],
        ["Front surface", "Median surface (shape)", "Thickness variation"],
        scale=1e6, label="Height [µm]", signed=True, radius=pair.radius,
        shared_scale=False, suptitle="Synthetic 300 mm wafer",
    )
    figure(fig, "01_wafer_synthesis.png")

    # ------------------------------------------------------------------
    _banner(2, "Zernike decomposition and flattening")
    fit = fit_surface(surface, nmax=6)
    flattened, _ = flatten(surface, nmax=6, remove_power=True)
    print(f"{len(fit.terms)} terms fitted, {100 * fit.variance_explained:.3f}% of variance explained")
    print("dominant terms (RMS contribution):")
    for term, value in fit.dominant(6):
        print(f"   Z{term}  {value * 1e6:+8.3f} µm")
    print(f"PV before flatten {surface.pv() * 1e6:.2f} µm -> after {flattened.pv() * 1e6:.3f} µm")

    fig, _ = plot_coefficient_spectrum(fit)
    figure(fig, "02_zernike_spectrum.png")
    fig, _ = plot_flatten_summary(surface, flattened, fit)
    figure(fig, "03_zernike_flatten.png")

    # ------------------------------------------------------------------
    _banner(3, "Interferometric measurement")
    worst_hene, hene_ok = check_sampling(surface.z, HENE_WAVELENGTH, surface.mask)
    worst_syn, syn_ok = check_sampling(surface.z, wavelength, surface.mask)
    print(
        f"sampling check — HeNe 633 nm: {worst_hene:.1f} rad/px "
        f"({'OK' if hene_ok else 'ALIASED, unwrapping would fail'})"
    )
    print(
        f"sampling check — synthetic λ = {wavelength * 1e6:.1f} µm: {worst_syn:.3f} rad/px "
        f"({'OK' if syn_ok else 'ALIASED'})"
    )

    measurement = measure_surface(
        surface, wavelength, n_steps=4, noise=noise, seed=seed + 11
    )
    result.rms_reconstruction_error = measurement.rms_error
    print(
        f"reconstruction: RMS error {measurement.rms_error * 1e9:.2f} nm, "
        f"PV error {measurement.pv_error * 1e9:.2f} nm"
    )

    fig, _ = plot_interferogram(surface, measurement.frames, measurement.wrapped)
    figure(fig, "04_interferogram.png")
    fig, _ = plot_reconstruction(surface, measurement)
    figure(fig, "05_reconstruction.png")

    # A single-wavelength HeNe measurement resolves nanotopography to sub-nm,
    # but only once the shape is removed and no steep defects remain.
    clean = synthesize_wafer(n_pixels=n_pixels, seed=seed, n_particles=0, n_scratches=0)
    smooth, _ = flatten(
        clean, nmax=6, remove_power=True,
        extra_terms=[(2, -2), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3)],
    )
    hene = measure_surface(smooth, HENE_WAVELENGTH, noise=noise / 2, seed=seed + 12)
    print(
        f"HeNe nanotopography scan: surface RMS {smooth.rms() * 1e9:.2f} nm measured to "
        f"{hene.rms_error * 1e9:.3f} nm RMS error"
    )
    fig, _ = plot_reconstruction(
        smooth, hene, title="HeNe nanotopography scan (shape removed, defect-free)"
    )
    figure(fig, "06_nanotopography_hene.png")

    # ------------------------------------------------------------------
    _banner(4, "SEMI flatness metrology")
    flatness_true = compute_flatness(
        pair, site_size=site_size, edge_exclusion=edge_exclusion
    )
    reconstructed = surface.with_z(measurement.z_measured)
    flatness_measured = compute_flatness(
        reconstructed, site_size=site_size, edge_exclusion=edge_exclusion
    )
    result.flatness_true = flatness_true
    result.flatness_measured = flatness_measured

    truth_table = flatness_true.metrics.to_frame()
    measured = flatness_measured.metrics
    comparison = truth_table.copy()
    comparison["measured"] = [
        measured.bow * 1e6, measured.warp * 1e6, measured.ttv * 1e6, measured.sori * 1e6,
        measured.sfqr_max * 1e9, measured.sfqr_mean * 1e9, measured.sfqr_p99 * 1e9,
        measured.thickness_mean * 1e6, measured.site_size * 1e3, measured.edge_exclusion * 1e3,
    ]
    comparison = comparison.rename(columns={"value": "truth"})
    print(
        comparison[["parameter", "truth", "measured", "unit", "description"]]
        .to_string(index=False, float_format=lambda v: f"{v:9.3f}")
    )
    print(
        "\n(TTV needs both faces; a reflection interferometer sees only the front, "
        "so the measured column has no TTV.)"
    )

    result.tables["flatness_metrics"] = flatness_true.to_csv(out / "flatness_metrics.csv")
    result.tables["site_flatness"] = flatness_true.sites_to_csv(out / "site_flatness.csv")
    fig, _ = plot_flatness(flatness_true)
    figure(fig, "07_flatness_summary.png")

    # ------------------------------------------------------------------
    _banner(5, "Defect extraction")
    result.defects = detect_defects(reconstructed)
    scores = score_detections(result.defects, pair.defects)
    print(
        f"classical: robust σ {result.defects.sigma * 1e9:.2f} nm, "
        f"threshold {result.defects.threshold * 1e9:.2f} nm, "
        f"{result.defects.count} detections"
    )
    if len(scores):
        print(f"recall: {int(scores['detected'].sum())}/{len(scores)} injected defects found")
    if result.defects.count:
        print(
            result.defects.table[
                ["defect_id", "x_mm", "y_mm", "equiv_diameter_mm", "peak_height_nm", "kind"]
            ].to_string(index=False, float_format=lambda v: f"{v:8.2f}")
        )
    print("\nsize distribution:")
    print(result.defects.size_distribution().to_string(index=False))

    result.tables["defects"] = out / "defects.csv"
    result.defects.table.to_csv(result.tables["defects"], index=False)
    fig, _ = plot_defect_map(reconstructed, result.defects, truth=pair.defects)
    figure(fig, "08_defect_map.png")

    if run_ml and HAVE_TORCH:
        print("\ntraining the convolutional autoencoder on clean wafers...")
        # Train on clean wafers put through the *same* measurement chain as the
        # wafer under test: a model trained on noise-free truth and scored on a
        # noisy reconstruction sees a distribution it never learned.
        clean_set = []
        for i in range(3):
            reference = synthesize_wafer(
                n_pixels=n_pixels, seed=seed + 500 + i, n_particles=0, n_scratches=0
            )
            reference_measurement = measure_surface(
                reference, wavelength, n_steps=4, noise=noise, seed=seed + 900 + i
            )
            clean_set.append(reference.with_z(reference_measurement.z_measured))
        model, scale = train_autoencoder(clean_set, epochs=10, seed=seed)
        result.defects_ml = detect_defects_ml(model, reconstructed, scale)
        ml_scores = score_detections(result.defects_ml, pair.defects)
        print(
            f"autoencoder: {result.defects_ml.count} detections, recall "
            f"{int(ml_scores['detected'].sum())}/{len(ml_scores)}"
        )
        fig, _ = plot_defect_map(reconstructed, result.defects_ml, truth=pair.defects)
        figure(fig, "09_defect_map_autoencoder.png")
    elif run_ml:
        print("\nPyTorch not installed — skipping the autoencoder path "
              "(pip install 'wafer-metrology-engine[ml]')")

    # ------------------------------------------------------------------
    if run_doe:
        _banner(6, "DOE: repeatability and Gage R&R")
        result.study = run_gauge_rr(
            n_parts=doe_parts, n_operators=doe_operators, n_trials=doe_trials,
            n_pixels=doe_pixels, noise=noise, wavelength=wavelength, seed=seed,
        )
        result.sweep = run_noise_sweep(
            n_repeats=doe_repeats, n_pixels=doe_pixels, wafer_seed=seed,
            wavelength=wavelength, seed=seed,
        )
        result.gauge = gauge_rr_all(result.study)

        print("\nrepeatability vs. detector noise (1σ over repeats):")
        summary = repeatability_summary(result.sweep)
        print(
            summary[["noise", "n", "warp_um_std", "sfqr_mean_nm_std", "sfqr_max_nm_std"]]
            .to_string(index=False, float_format=lambda v: f"{v:10.4f}")
        )
        print("\nGage R&R variance components:")
        print(
            result.gauge[
                ["response", "ev", "av", "grr", "pv", "pct_grr", "ndc", "verdict"]
            ].to_string(index=False, float_format=lambda v: f"{v:9.3f}")
        )

        result.tables["gauge_rr"] = out / "gauge_rr_components.csv"
        result.gauge.to_csv(result.tables["gauge_rr"], index=False)
        result.tables["doe_runs"] = export_for_jmp(result.study, out / "doe_runs_for_jmp.csv")
        result.tables["noise_sweep"] = out / "noise_sweep.csv"
        result.sweep.to_csv(result.tables["noise_sweep"], index=False)

        fig, _ = plot_doe_summary(result.sweep, result.study, result.gauge)
        figure(fig, "10_doe_gauge_rr.png")

    # ------------------------------------------------------------------
    wafer150 = None
    z_absolute = None
    if run_deflectometry:
        _banner(7, "Deflectometry (hardware-matched Leg A simulation, 150 mm wafer)")
        wafer150 = synthesize_wafer(
            n_pixels=deflectometry_pixels, diameter=WAFER_DIAMETER_150MM, seed=seed + 1
        )
        # A fixed instrument systematic: screen bow (rotation-even power +
        # astigmatism) plus a rotation-odd coma/trefoil pose component, so the
        # calibration stage can show which method removes which half.
        system_zmap = reconstruct(
            {(2, 0): 4e-6, (2, 2): 2e-6, (3, 1): 3e-6, (3, -3): 1.5e-6},
            wafer150.x, wafer150.y, wafer150.radius, wafer150.mask,
        )
        gamma_hat = fit_gamma(*simulate_gamma_sweep(2.2, noise=0.002, seed=seed + 31))
        setup = DeflectometrySetup(
            noise=noise / 2, gamma=2.2, lut_gamma=gamma_hat, system_zmap=system_zmap
        )
        print(
            f"rig: d_s = {setup.screen_distance:.2f} m, periods "
            f"{[round(p * 1e3) for p in setup.periods]} mm, {setup.n_steps} steps, "
            f"display γ 2.2 calibrated to γ̂ = {gamma_hat:.3f}"
        )
        print(
            "injected system error (screen bow + pose): "
            f"PV {float(np.nanmax(system_zmap) - np.nanmin(system_zmap)) * 1e6:.1f} µm"
        )

        deflect = measure_deflectometry(wafer150, setup, seed=seed + 32)
        result.deflectometry = deflect
        print(
            f"raw measurement error (uncalibrated, incl. systematic): "
            f"RMS {deflect.rms_error * 1e9:.0f} nm"
        )
        fig, _ = plot_deflectograms(wafer150, deflect)
        figure(fig, "11_deflectometry_fringes.png")

        _banner(8, "Reversal vs. absolute calibration")
        deflect_180 = measure_deflectometry(
            rotate_surface_180(wafer150), setup, seed=seed + 33
        )
        wafer_est, system_est = reversal_calibrate(
            deflect.z_measured, deflect_180.z_measured, deflect.mask
        )
        flat = make_flat_surface(deflectometry_pixels, WAFER_DIAMETER_150MM)
        flat_measured = measure_deflectometry(flat, setup, seed=seed + 34)
        z_absolute = absolute_calibrate(deflect.z_measured, flat_measured.z_measured)

        def aligned_rms(estimate: np.ndarray) -> float:
            err = plane_aligned_difference(
                estimate, wafer150.z, wafer150.x, wafer150.y, np.isfinite(estimate)
            )
            return float(np.sqrt(np.nanmean(err[np.isfinite(err)] ** 2)))

        truth_rms = float(np.nanstd(wafer150.z[deflect.mask]))
        rms_reversal = aligned_rms(wafer_est)
        rms_absolute = aligned_rms(z_absolute)
        result.deflectometry_gate = rms_absolute / truth_rms
        print(
            f"reversal wafer estimate error:  RMS {rms_reversal * 1e9:7.0f} nm  "
            "(the rotation-even system error -- screen bow -- survives reversal)"
        )
        print(
            f"absolute (flat-subtracted):     RMS {rms_absolute * 1e9:7.0f} nm  "
            f"-> {100 * result.deflectometry_gate:.3f}% of surface RMS "
            f"({'PASS' if result.deflectometry_gate < 0.01 else 'FAIL'} vs the <1% Tier-0 gate)"
        )
        fig, _ = plot_deflectometry_recovery(
            wafer150,
            deflect,
            title="Deflectometry height recovery (raw, before calibration)",
        )
        figure(fig, "12_deflectometry_recovery.png")
        fig, _ = plot_reversal(wafer150, wafer_est, system_est, system_zmap, z_absolute)
        figure(fig, "13_reversal_calibration.png")

        # Flatness of the calibrated measurement vs truth, on the 150 mm wafer.
        calibrated_surface = wafer150.with_z(z_absolute)
        truth_metrics = compute_flatness(wafer150).metrics
        measured_metrics = compute_flatness(calibrated_surface).metrics
        print(
            f"bow  truth {truth_metrics.bow * 1e6:+8.2f} µm | measured "
            f"{measured_metrics.bow * 1e6:+8.2f} µm\n"
            f"warp truth {truth_metrics.warp * 1e6:8.2f} µm | measured "
            f"{measured_metrics.warp * 1e6:8.2f} µm\n"
            f"SFQR max truth {truth_metrics.sfqr_max * 1e9:6.0f} nm | measured "
            f"{measured_metrics.sfqr_max * 1e9:6.0f} nm "
            f"({truth_metrics.n_sites} sites)"
        )

    if run_capstone and z_absolute is not None and wafer150 is not None:
        _banner(9, "Capstone: measured warp → fibre-attach loss budget (1550 nm)")
        measured_surface = wafer150.with_z(z_absolute)
        result.loss_budget = attach_loss_budget(measured_surface)
        budget = result.loss_budget
        print(budget.headline)
        by_mechanism = budget.table[
            ["loss_angular_db", "loss_lateral_db", "loss_gap_db"]
        ].abs().max()
        print(
            "worst single-mechanism contributions: "
            f"angular {by_mechanism['loss_angular_db']:.4f} dB, "
            f"lateral (tilt × lever) {by_mechanism['loss_lateral_db']:.3f} dB, "
            f"gap (standoff) {by_mechanism['loss_gap_db']:.3f} dB"
        )
        result.tables["loss_budget"] = budget.to_csv(out / "loss_budget.csv")
        fig, _ = plot_loss_budget(measured_surface, budget)
        figure(fig, "14_loss_budget.png")
    elif run_capstone:
        print("\ncapstone skipped: it consumes the deflectometry measurement "
              "(enable run_deflectometry)")

    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"Wrote {len(result.figures)} figures and {len(result.tables)} tables to {out.resolve()}")
    for path in result.figures:
        print(f"   {path.name}")
    for name, path in result.tables.items():
        print(f"   {Path(path).name}")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to :data:`sys.argv`.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="wafer-metrology",
        description="Wafer surface metrology and warpage characterisation pipeline.",
    )
    parser.add_argument("--results", default="results", help="output directory")
    parser.add_argument("--n-pixels", type=int, default=512, help="grid size for the main wafer")
    parser.add_argument("--seed", type=int, default=0, help="master RNG seed")
    parser.add_argument("--noise", type=float, default=0.01, help="detector noise fraction")
    parser.add_argument("--site-size", type=float, default=25.0, help="exposure site edge [mm]")
    parser.add_argument("--edge-exclusion", type=float, default=3.0, help="edge exclusion [mm]")
    parser.add_argument("--quick", action="store_true",
                        help="small grid, small DOE, no autoencoder")
    parser.add_argument("--no-ml", action="store_true", help="skip the autoencoder path")
    parser.add_argument("--no-doe", action="store_true", help="skip the DOE / Gage R&R study")
    parser.add_argument("--no-deflectometry", action="store_true",
                        help="skip the deflectometry (Leg A) simulation")
    parser.add_argument("--no-capstone", action="store_true",
                        help="skip the warp-to-loss capstone")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    kwargs: Dict[str, Any] = {
        "n_pixels": 256 if args.quick else args.n_pixels,
        "seed": args.seed,
        "noise": args.noise,
        "site_size": args.site_size * 1e-3,
        "edge_exclusion": args.edge_exclusion * 1e-3,
        "run_ml": not (args.no_ml or args.quick),
        "run_doe": not args.no_doe,
        "run_deflectometry": not args.no_deflectometry,
        "run_capstone": not args.no_capstone,
    }
    if args.quick:
        kwargs.update(doe_pixels=128, doe_parts=2, doe_operators=2, doe_trials=2,
                      doe_repeats=4, deflectometry_pixels=192)

    run_pipeline(args.results, **kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
