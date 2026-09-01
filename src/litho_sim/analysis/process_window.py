"""
Process window and CD analysis module.

Provides functions for:

* Bossung curve generation (CD vs. defocus sweeps across dose)
* Exposure latitude (EL) and depth-of-focus (DOF) extraction
* Full 2-D process window contour calculation
* Normalised Image Log-Slope (NILS) computation
* Best-focus and best-dose identification
* Dose-to-size calibration, EL-vs-DOF trade-off curve, MEEF

All physical quantities use SI units internally; helper functions convert
to convenient display units (nm, %) for DataFrames and return values.

Two conventions worth knowing before reading numbers out of this module:

* **Dose is baked into the aerial image.** Every measurement here goes
  through :func:`_measure_point`, which never forwards dose to the resist
  chemistry — passing it twice would square it (the double-count hazard
  documented at :func:`litho_sim.develop.resist.simulate_resist`).
* **Sweeps default to clear-field normalisation.** The peak intensity is
  focus-dependent, so under peak normalisation "dose 1.0" delivers a
  different physical energy at every defocus and Bossung curves flatten
  artificially. Clear-field normalisation is the lithographic convention
  that makes dose mean the same thing at every focus.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from litho_sim.bake.peb import apply_peb
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig, SimulationConfig
from litho_sim.develop.resist import (
    dill_exposure,
    mack_development_rate,
    measure_cd_1d,
    measure_cd_2d,
)
from litho_sim.expose.aerial_image import compute_aerial_image, normalisation_scale
from litho_sim.mask.patterns import apply_bias, lines_and_spaces

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NILS / ILS
# ---------------------------------------------------------------------------


def compute_nils(
    aerial_1d: NDArray[np.float64],
    pixel_size: float,
    threshold: float,
    nominal_cd: float,
) -> float:
    """Compute the Normalised Image Log-Slope (NILS) at the feature edge.

    NILS = CD · |d(ln I) / dx|  evaluated at the threshold crossing
    (Mack, *Fundamental Principles of Optical Lithography*, Eq. 3.92 —
    the normalisation length is the full linewidth, not the half-width;
    this function returned half the standard value until the audit's H2
    finding was fixed).

    A NILS > 2 is generally required for robust lithographic printing.

    The crossing is located sub-pixel by linear interpolation of the
    intensity between the bracketing samples, and the log-slope is
    interpolated to the same fractional position.

    Parameters
    ----------
    aerial_1d : NDArray
        1-D aerial image intensity profile (centre cross-section).
    pixel_size : float
        Physical pixel size [m].
    threshold : float
        Intensity threshold at the resist edge (0–1).
    nominal_cd : float
        Nominal CD (full feature linewidth) used as normalisation length [m].

    Returns
    -------
    float
        NILS value (dimensionless).  Returns 0.0 if computation fails.
    """
    intensity = np.asarray(aerial_1d, dtype=np.float64)
    n = len(intensity)
    # Gradient of ln(I)  [units: m⁻¹]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_intensity = np.where(intensity > 1e-12, np.log(intensity), -28.0)  # clip log(0)
    log_slope = np.gradient(log_intensity, pixel_size)

    # Find threshold crossings (gap index i names the interval (i, i+1))
    diff = np.diff((intensity > threshold).astype(int))
    crossings = np.where(np.abs(diff) == 1)[0]
    if len(crossings) == 0:
        return 0.0

    # Select crossing closest to centre
    centre = n // 2
    i = int(crossings[np.argmin(np.abs(crossings - centre))])

    # Fractional position of the crossing inside the gap, slope interpolated
    # to it — the same sub-pixel treatment measure_cd_1d applies to CD.
    if intensity[i + 1] == intensity[i]:
        t = 0.5
    else:
        t = float(np.clip(
            (threshold - intensity[i]) / (intensity[i + 1] - intensity[i]), 0.0, 1.0
        ))
    ils = abs((1.0 - t) * log_slope[i] + t * log_slope[i + 1])
    return float(ils * nominal_cd)


# ---------------------------------------------------------------------------
# Single-point CD measurement
# ---------------------------------------------------------------------------


def _measure_point(
    aerial: NDArray[np.float64],
    resist_cfg: ResistConfig,
    grid: GridConfig,
    model: str = "threshold",
) -> float:
    """Printed CD [m] from an aerial image — the single measurement kernel.

    Every CD this module reports (Bossung sweeps, dose-to-size, MEEF) comes
    through here, so the measurement conventions live in one place:

    * **Dose rule**: any dose is already baked into *aerial* and is never
      forwarded to the resist chemistry — the Dill exposure would multiply
      it in again (the double-count hazard documented at
      :func:`litho_sim.develop.resist.simulate_resist`).
    * **Sub-pixel**: the *continuous* field is measured with interpolated
      threshold crossings rather than binarised first. Binarising quantises
      CD to whole pixels (4 nm at the default grid — the audit's H5
      finding), which is too coarse to resolve a Bossung curve.
    * **threshold model**: measures the PEB-diffused latent image at the
      develop threshold — the diffused-aerial-image treatment, matching the
      desktop app's live path. At ``diffusion_sigma = 0`` the blur is a
      no-op and this reduces to thresholding the aerial directly.
    * **mack model**: runs Dill → PEB → Mack rate and measures the develop
      depth field against the film thickness — where the front fails to
      reach the substrate, resist survives.
    * **car model**: runs acid generation → acid/quencher reaction–diffusion
      → Mack rate on the *protected* fraction, measured the same way. Note
      the bake makes the dose axis genuinely non-linear (that is the
      quencher's whole point), and each grid point pays for a PDE bake —
      a car sweep costs seconds per point, not milliseconds.
    """
    mid = aerial.shape[0] // 2
    if model == "threshold":
        latent = apply_peb(aerial, resist_cfg.diffusion_sigma, grid.pixel_size)
        cut = latent[mid, :]
        # Positive tone: resist survives where intensity stays *under* the
        # threshold — the printed line is the dark region of the image.
        feature = "below" if resist_cfg.tone == "positive" else "above"
        return measure_cd_1d(
            cut, grid.pixel_size, threshold=resist_cfg.threshold, feature=feature
        )
    if model == "mack":
        pac = dill_exposure(aerial, resist_cfg.dose_nominal, resist_cfg.dill_C)
        pac_peb = apply_peb(pac, resist_cfg.diffusion_sigma, grid.pixel_size)
        rate = mack_development_rate(
            pac_peb, resist_cfg.mack_Rmax, resist_cfg.mack_Rmin,
            resist_cfg.mack_Mth, resist_cfg.mack_n,
        )
        cleared_nm = rate * resist_cfg.develop_time
        cut = cleared_nm[mid, :]
        thickness_nm = resist_cfg.thickness * 1e9
        feature = "below" if resist_cfg.tone == "positive" else "above"
        return measure_cd_1d(
            cut, grid.pixel_size, threshold=thickness_nm, feature=feature
        )
    if model == "car":
        from litho_sim.bake.reaction import bake_reaction_diffusion
        from litho_sim.expose.photochem import generate_acid

        # The bake diffuses in 2-D, so the whole field is baked and the cut
        # taken afterwards — cutting first would turn lateral diffusion off.
        acid = generate_acid(aerial, resist_cfg, grid.pixel_size, dose=1.0)
        baked = bake_reaction_diffusion(
            acid,
            grid.pixel_size,
            resist_cfg.bake_time,
            resist_cfg.D_acid,
            quencher=resist_cfg.quencher_ratio,
            D_quencher=resist_cfg.D_quencher,
            k_quench=resist_cfg.k_quench,
            k_loss=resist_cfg.k_loss,
            k_amp=resist_cfg.k_amp,
        )
        rate = mack_development_rate(
            baked["protected"], resist_cfg.mack_Rmax, resist_cfg.mack_Rmin,
            resist_cfg.mack_Mth, resist_cfg.mack_n,
        )
        cleared_nm = rate * resist_cfg.develop_time
        cut = cleared_nm[mid, :]
        thickness_nm = resist_cfg.thickness * 1e9
        feature = "below" if resist_cfg.tone == "positive" else "above"
        return measure_cd_1d(
            cut, grid.pixel_size, threshold=thickness_nm, feature=feature
        )
    raise ValueError(
        f"Unknown resist model: '{model}'. Choose 'threshold', 'mack' or 'car'."
    )


def evaluate_cd(
    mask: NDArray[np.float64],
    optics: OpticsConfig,
    grid: GridConfig,
    resist: ResistConfig,
    dose: float,
    defocus_m: float,
    model: str = "threshold",
) -> float:
    """Compute the printed CD for one (dose, defocus) operating point.

    Parameters
    ----------
    mask : NDArray
        Binary mask transmittance array.
    optics : OpticsConfig
        Optical system parameters.  *defocus* is overridden by *defocus_m*.
    grid : GridConfig
        Simulation grid.
    resist : ResistConfig
        Resist parameters.
    dose : float
        Normalised dose.
    defocus_m : float
        Defocus [m].
    model : str
        Resist development model (``"mack"`` or ``"threshold"``).

    Returns
    -------
    float
        Measured CD [m].
    """
    # Override defocus without mutating the original config
    local_optics = dataclasses.replace(optics, defocus=defocus_m)
    aerial = compute_aerial_image(mask, local_optics, grid, dose=dose)
    return _measure_point(aerial, resist, grid, model=model)


# ---------------------------------------------------------------------------
# Bossung sweep
# ---------------------------------------------------------------------------


def sweep_dose_focus(
    mask: NDArray[np.float64],
    optics: OpticsConfig,
    grid: GridConfig,
    resist: ResistConfig,
    doses: list[float],
    defoci_nm: list[float],
    model: str = "threshold",
    normalisation: str | None = "clear",
    target_cd_nm: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Sweep dose × defocus and record printed CD and NILS at each point.

    Costs one Abbe sum per *focus* value, not per grid point: dose is a pure
    post-hoc rescale of the raw sum (``compute_aerial_image(return_raw=True)``
    plus :func:`normalisation_scale`), so the dose axis is nearly free.

    Parameters
    ----------
    mask : NDArray
        Binary mask pattern.
    optics : OpticsConfig
        Base optical parameters.
    grid : GridConfig
        Simulation grid.
    resist : ResistConfig
        Resist parameters.
    doses : list of float
        Normalised dose levels to evaluate.
    defoci_nm : list of float
        Defocus positions to evaluate [nm].
    model : str
        Resist development model.
    normalisation : str, optional
        Normalisation mode for the sweep. Defaults to ``"clear"`` — the peak
        is focus-dependent, so under ``"peak"`` normalisation the same dose
        number delivers different energy at every defocus and the Bossung
        curves flatten artificially. Pass ``None`` to use ``optics`` as-is.
    target_cd_nm : float, optional
        Normalisation length for the NILS column [nm]. When omitted, each
        point's own measured CD is used — fine for plotting trends, but a
        fixed nominal CD is the standard definition.
    progress : callable, optional
        Called as ``progress(done_focus, n_focus)`` after each completed
        focus column — the hook a GUI progress bar wants.

    Returns
    -------
    pd.DataFrame
        Columns: ``dose``, ``defocus_nm``, ``cd_nm``, ``cd_m``, ``nils``.
    """
    if normalisation is not None and normalisation != optics.normalisation:
        logger.info(
            "Sweep overrides normalisation '%s' → '%s' so dose delivers the "
            "same energy at every focus.", optics.normalisation, normalisation,
        )
        base_optics = dataclasses.replace(optics, normalisation=normalisation)
    else:
        base_optics = optics
    norm_mode = base_optics.normalisation

    records = []
    n_focus = len(defoci_nm)
    mid = grid.n_pixels // 2
    logger.info(
        "Bossung sweep: %d doses × %d defoci = %d points (%d Abbe sums)",
        len(doses), n_focus, len(doses) * n_focus, n_focus,
    )

    for j_def, def_nm in enumerate(defoci_nm):
        local_optics = _optics_at_focus(base_optics, def_nm)
        raw, peak, clear = compute_aerial_image(
            mask, local_optics, grid, return_raw=True
        )
        for dose in doses:
            aerial = raw * normalisation_scale(norm_mode, peak, clear, dose)
            cd_m = _measure_point(aerial, resist, grid, model=model)
            nils_len = target_cd_nm * 1e-9 if target_cd_nm is not None else cd_m
            nils = compute_nils(
                aerial[mid, :], grid.pixel_size,
                threshold=resist.threshold, nominal_cd=nils_len,
            )
            records.append(
                dict(
                    dose=float(dose),
                    defocus_nm=float(def_nm),
                    cd_nm=cd_m * 1e9,
                    cd_m=cd_m,
                    nils=float(nils),
                )
            )
            logger.debug(
                "  dose=%.3f, Δz=%.1f nm → CD=%.1f nm, NILS=%.2f",
                dose, def_nm, cd_m * 1e9, nils,
            )
        if progress is not None:
            progress(j_def + 1, n_focus)

    df = pd.DataFrame(records)
    logger.info(
        "Bossung sweep complete. CD range: %.1f – %.1f nm",
        df["cd_nm"].min(), df["cd_nm"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Process window metrics
# ---------------------------------------------------------------------------


def in_spec(
    cd_nm: float,
    target_nm: float,
    tolerance_pct: float,
) -> bool:
    """Return True if CD is within ±tolerance_pct% of target."""
    if target_nm <= 0 or cd_nm <= 0:
        return False
    return abs(cd_nm - target_nm) / target_nm * 100.0 <= tolerance_pct


def _optics_at_focus(optics, defocus_nm: float, normalisation: str | None = None):
    """A local optics at this defocus [nm], optionally normalisation-overridden.

    The one spelling of "give me these optics, focused there" the per-condition
    analyses share — the original is never mutated.
    """
    local = dataclasses.replace(optics, defocus=defocus_nm * 1e-9)
    if normalisation is not None:
        local = dataclasses.replace(local, normalisation=normalisation)
    return local


def _in_spec_runs(
    slice_df: pd.DataFrame,
    axis_col: str,
    target_cd_nm: float,
    tolerance_pct: float,
):
    """Sorted axis values, CDs, in-spec flags, and the contiguous in-spec runs.

    The shared front half of the EL and DOF cuts: order one Bossung slice
    along *axis_col*, flag which nodes print in spec, and find the runs.
    """
    ordered = slice_df.sort_values(axis_col)
    axis = ordered[axis_col].to_numpy(dtype=np.float64)
    cds = ordered["cd_nm"].to_numpy(dtype=np.float64)
    flags = np.array(
        [in_spec(c, target_cd_nm, tolerance_pct) for c in cds], dtype=bool
    )
    return axis, cds, flags, _contiguous_runs(flags)


def _contiguous_runs(flags: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Inclusive ``(start, stop)`` index pairs of each contiguous True run."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _pick_run(
    runs: list[tuple[int, int]],
    x: NDArray[np.float64],
    x_ref: float,
) -> tuple[int, int]:
    """The run whose x-interval is nearest *x_ref*; ties go to the longest.

    Two in-spec islands either side of a gap are two windows, not one — the
    caller works within a single run, and this picks which.
    """
    def distance(run: tuple[int, int]) -> float:
        lo, hi = x[run[0]], x[run[1]]
        if lo <= x_ref <= hi:
            return 0.0
        return min(abs(x_ref - lo), abs(x_ref - hi))

    return min(runs, key=lambda r: (distance(r), -(r[1] - r[0])))


def _interpolated_bounds(
    x: NDArray[np.float64],
    cd: NDArray[np.float64],
    run: tuple[int, int],
    target_cd_nm: float,
    tolerance_pct: float,
) -> tuple[float, float]:
    """Sub-node bounds of an in-spec run, by interpolating the spec crossings.

    Between the boundary in-spec node and its out-of-spec neighbour, the CD
    crosses one of the two spec limits; the crossing position is linear-
    interpolated in x. A neighbour that did not print (``cd <= 0``) carries
    no usable CD, so the bound clamps at the in-spec node — never
    extrapolate through a non-printing point.
    """
    lo_spec = target_cd_nm * (1.0 - tolerance_pct / 100.0)
    hi_spec = target_cd_nm * (1.0 + tolerance_pct / 100.0)
    i0, i1 = run

    def cross(inner: int, outer: int) -> float:
        c_in, c_out = cd[inner], cd[outer]
        if c_out <= 0 or c_out == c_in:
            return float(x[inner])
        limit = hi_spec if c_out > hi_spec else lo_spec
        t = float(np.clip((limit - c_in) / (c_out - c_in), 0.0, 1.0))
        return float(x[inner] + t * (x[outer] - x[inner]))

    lower = cross(i0, i0 - 1) if i0 > 0 else float(x[i0])
    upper = cross(i1, i1 + 1) if i1 < len(x) - 1 else float(x[i1])
    return lower, upper


def compute_exposure_latitude(
    bossung_df: pd.DataFrame,
    target_cd_nm: float,
    tolerance_pct: float = 10.0,
    defocus_nm: float = 0.0,
    nominal_dose: float = 1.0,
) -> float:
    """Compute exposure latitude (EL) at a given defocus.

    EL = (dose_high − dose_low) / dose_nominal × 100 %

    where dose_high / dose_low are the doses at which the printed CD crosses
    *target_cd_nm* ± *tolerance_pct* %, linearly interpolated between sweep
    nodes. Only a single contiguous in-spec dose band counts — the one
    nearest *nominal_dose* (ties go to the widest); an interrupted window is
    not one window.

    Parameters
    ----------
    bossung_df : pd.DataFrame
        Output of :func:`sweep_dose_focus`.
    target_cd_nm : float
        Nominal CD [nm].
    tolerance_pct : float
        Allowed CD variation [%].
    defocus_nm : float
        Defocus slice to evaluate [nm].  Uses the closest available row.
    nominal_dose : float
        The dose the EL percentage is referenced to — the process's centre
        dose, not a property of whichever nodes happened to land in spec.

    Returns
    -------
    float
        EL in percent.  Returns 0.0 if no in-spec window exists.
    """
    available = bossung_df["defocus_nm"].unique()
    closest_def = float(available[np.argmin(np.abs(available - defocus_nm))])
    slice_df = bossung_df[np.isclose(bossung_df["defocus_nm"], closest_def)]

    doses, cds, _, runs = _in_spec_runs(
        slice_df, "dose", target_cd_nm, tolerance_pct
    )
    if not runs:
        return 0.0

    run = _pick_run(runs, doses, nominal_dose)
    dose_lo, dose_hi = _interpolated_bounds(
        doses, cds, run, target_cd_nm, tolerance_pct
    )
    if nominal_dose <= 0:
        return 0.0
    return float((dose_hi - dose_lo) / nominal_dose * 100.0)


def compute_depth_of_focus(
    bossung_df: pd.DataFrame,
    target_cd_nm: float,
    tolerance_pct: float = 10.0,
    dose: float | None = None,
) -> float:
    """Compute depth of focus (DOF) at a given dose.

    DOF = the single contiguous defocus range over which CD stays within
    spec, with the spec crossings linearly interpolated between sweep nodes.
    Two in-spec islands either side of a gap never merge into one window.

    Parameters
    ----------
    bossung_df : pd.DataFrame
        Output of :func:`sweep_dose_focus`.
    target_cd_nm : float
        Nominal CD [nm].
    tolerance_pct : float
        Allowed CD variation [%].
    dose : float, optional
        Dose level to evaluate.  Defaults to the dose whose Bossung peak is
        closest to *target_cd_nm*, considering printing points only.

    Returns
    -------
    float
        DOF in nm.  Returns 0.0 if no in-spec window exists.
    """
    available_doses = bossung_df["dose"].unique()

    if dose is None:
        # Auto-select dose whose best printing CD is closest to target.
        # Non-printing rows (cd = 0) are excluded — zero is an absence of a
        # feature, not a candidate CD.
        best_cds = []
        for d in available_doses:
            sub = bossung_df[np.isclose(bossung_df["dose"], d)]
            sub = sub[sub["cd_nm"] > 0]
            if sub.empty:
                best_cds.append(np.inf)
            else:
                best_cds.append(float((sub["cd_nm"] - target_cd_nm).abs().min()))
        if not np.isfinite(best_cds).any():
            return 0.0
        dose = float(available_doses[int(np.argmin(best_cds))])

    closest_dose = float(available_doses[np.argmin(np.abs(available_doses - dose))])
    slice_df = bossung_df[np.isclose(bossung_df["dose"], closest_dose)]

    foci, cds, flags, runs = _in_spec_runs(
        slice_df, "defocus_nm", target_cd_nm, tolerance_pct
    )
    if not runs:
        return 0.0

    # Reference the run to the in-spec focus whose CD is nearest target.
    in_idx = np.where(flags)[0]
    f_ref = float(foci[in_idx[np.argmin(np.abs(cds[in_idx] - target_cd_nm))]])
    run = _pick_run(runs, foci, f_ref)
    f_lo, f_hi = _interpolated_bounds(foci, cds, run, target_cd_nm, tolerance_pct)
    return float(f_hi - f_lo)


def compute_process_window(
    bossung_df: pd.DataFrame,
    target_cd_nm: float,
    tolerance_pct: float = 10.0,
    nominal_dose: float = 1.0,
) -> dict:
    """Compute the full 2-D lithographic process window.

    The process window is the set of (dose, defocus) pairs for which the
    printed CD is within *target_cd_nm* ± *tolerance_pct* %.

    Parameters
    ----------
    bossung_df : pd.DataFrame
        Output of :func:`sweep_dose_focus`.
    target_cd_nm : float
        Nominal (target) CD [nm].
    tolerance_pct : float
        CD tolerance [%].
    nominal_dose : float
        Centre dose the EL percentage is referenced to.

    Returns
    -------
    dict with keys:
        * ``"window_df"``  – copy of *bossung_df* with an ``"in_spec"`` bool column.
        * ``"EL_pct"``     – exposure latitude at best focus [%].
        * ``"DOF_nm"``     – depth of focus at best dose [nm].
        * ``"area"``       – approximate window area [nm·%].
        * ``"best_dose"``  – dose giving CD closest to target at best focus.
        * ``"best_focus_nm"`` – defocus that minimises CD-vs-dose variation,
          judged over printing points only.
        * ``"prints"``     – False when no defocus column has at least two
          printing doses; the other metrics are then 0 / NaN and a consumer
          should show a diagnosis, not a plot.
        * ``"nominal_dose"`` – echoed back for labelling.

    Notes
    -----
    Best focus is the flattest Bossung criterion — the defocus whose CD has
    the smallest spread across dose — but computed over printing points only.
    Non-printing points enter the frame as ``cd_nm = 0``, whose near-zero
    spread would otherwise beat any real column and park "best focus" at a
    sweep edge where nothing prints.
    """
    df = bossung_df.copy()
    df["in_spec"] = df["cd_nm"].apply(
        lambda cd: in_spec(cd, target_cd_nm, tolerance_pct)
    )

    # Best focus: flattest CD-vs-dose column, printing points only — and
    # among columns holding at least one in-spec point when any exist.
    # Excluding non-printing points is not enough on its own: at deep
    # defocus the contrast collapse compresses every dose onto nearly the
    # same (wrong) CD, and that degenerate flatness beats any real column.
    # A best focus where nothing is in spec makes the whole window read as
    # empty while in-spec points sit in plain view at other defoci.
    defoci = np.sort(df["defocus_nm"].unique())
    spread_per_defocus: dict[float, float] = {}
    in_spec_defoci = set()
    for d in defoci:
        col = df[np.isclose(df["defocus_nm"], d)]
        col = col[col["cd_nm"] > 0]
        if len(col) >= 2:
            spread_per_defocus[float(d)] = float(col["cd_nm"].std())
            if col["in_spec"].any():
                in_spec_defoci.add(float(d))
    if in_spec_defoci:
        spread_per_defocus = {
            f: s for f, s in spread_per_defocus.items() if f in in_spec_defoci
        }

    if not spread_per_defocus:
        logger.info("Process window: nothing prints across the sweep.")
        return {
            "window_df": df,
            "EL_pct": 0.0,
            "DOF_nm": 0.0,
            "area": 0.0,
            "best_dose": float("nan"),
            "best_focus_nm": float("nan"),
            "prints": False,
            "nominal_dose": float(nominal_dose),
        }

    # Spreads that are equal in exact arithmetic (a symmetric Bossung has
    # spread(f) == spread(−f)) differ in the last float bits, so ties are
    # broken within a relative tolerance rather than on raw std order —
    # otherwise rounding noise, not physics, picks the best focus.
    min_spread = min(spread_per_defocus.values())
    tie_tol = min_spread * 1e-9 + 1e-12
    best_focus_nm = min(
        (f for f, s in spread_per_defocus.items() if s <= min_spread + tie_tol),
        key=lambda f: (abs(f), f),
    )

    # Best dose: at best focus, the printing dose that gives CD closest to target
    at_best_focus = df[np.isclose(df["defocus_nm"], best_focus_nm)]
    at_best_focus = at_best_focus[at_best_focus["cd_nm"] > 0]
    best_dose = float(
        at_best_focus.loc[
            (at_best_focus["cd_nm"] - target_cd_nm).abs().idxmin(), "dose"
        ]
    )

    el = compute_exposure_latitude(
        df, target_cd_nm, tolerance_pct,
        defocus_nm=best_focus_nm, nominal_dose=nominal_dose,
    )
    dof = compute_depth_of_focus(df, target_cd_nm, tolerance_pct, dose=best_dose)
    area = el * dof  # [% · nm]

    logger.info(
        "Process Window: EL=%.1f%%, DOF=%.0f nm, area=%.0f %%·nm, "
        "best focus=%.0f nm, best dose=%.3f",
        el, dof, area, best_focus_nm, best_dose,
    )
    return {
        "window_df": df,
        "EL_pct": el,
        "DOF_nm": dof,
        "area": area,
        "best_dose": best_dose,
        "best_focus_nm": float(best_focus_nm),
        "prints": True,
        "nominal_dose": float(nominal_dose),
    }


# ---------------------------------------------------------------------------
# Dose-to-size calibration
# ---------------------------------------------------------------------------


def calibrate_dose_to_size(
    mask: NDArray[np.float64],
    optics: OpticsConfig,
    grid: GridConfig,
    resist: ResistConfig,
    target_cd_nm: float,
    defocus_nm: float = 0.0,
    model: str = "threshold",
    normalisation: str | None = "clear",
    dose_bounds: tuple[float, float] = (0.25, 4.0),
    tol_nm: float = 0.05,
    max_iter: int = 40,
) -> float:
    """Find the dose that prints *target_cd_nm* at the given focus.

    One Abbe sum total: the raw image is computed once and every bisection
    step is a scalar rescale plus a sub-pixel CD measurement, so the whole
    calibration costs barely more than a single aerial image.

    Parameters
    ----------
    mask, optics, grid, resist
        As for :func:`sweep_dose_focus`.
    target_cd_nm : float
        The CD the dose should size to [nm].
    defocus_nm : float
        Focus at which to calibrate [nm].
    model : str
        Resist development model.
    normalisation : str, optional
        Normalisation mode (default ``"clear"``; ``None`` uses *optics* as-is).
    dose_bounds : (float, float)
        Bisection bracket.
    tol_nm : float
        Convergence tolerance on |CD − target| [nm].
    max_iter : int
        Bisection iteration cap.

    Returns
    -------
    float
        The calibrated dose, or NaN when the bracket does not size — the
        feature never prints, or never reaches target, anywhere in bounds.
    """
    local = _optics_at_focus(optics, defocus_nm, normalisation)
    raw, peak, clear = compute_aerial_image(mask, local, grid, return_raw=True)
    target_m = target_cd_nm * 1e-9

    def cd_error(dose: float) -> float:
        aerial = raw * normalisation_scale(local.normalisation, peak, clear, dose)
        return _measure_point(aerial, resist, grid, model=model) - target_m

    # CD is not monotone across the full bracket: an underexposed image has
    # no edges at all (CD 0) and an overexposed one clears completely
    # (CD 0 again), with the printable regime in between — and the interval
    # where CD passes the target can be narrow. A dense scan locates a
    # sign-change interval — preferring the one nearest dose 1 — and
    # bisection refines inside it. Every probe is a rescale, not an Abbe
    # sum, so even 65 of them are effectively free.
    probes = np.linspace(float(dose_bounds[0]), float(dose_bounds[1]), 65)
    errors = [cd_error(float(p)) for p in probes]
    for p, e in zip(probes, errors):
        if abs(e) < tol_nm * 1e-9:
            return float(p)

    brackets = [
        i for i in range(len(probes) - 1)
        if np.sign(errors[i]) != np.sign(errors[i + 1])
    ]
    if not brackets:
        logger.warning(
            "dose-to-size: CD never crosses %.1f nm over dose ∈ [%.2f, %.2f]; "
            "returning NaN.",
            target_cd_nm, probes[0], probes[-1],
        )
        return float("nan")

    # CD(dose) is discontinuous at emergence — the printed width jumps from
    # nothing straight to roughly the bright-fringe separation. A sign change
    # across that cliff is not a sizing dose, and bisection there converges
    # to the cliff edge. Prefer brackets whose both endpoints print (the
    # crossing lies inside the smooth printable regime), and validate
    # convergence so a discontinuous bracket falls through to the next.
    def both_print(i: int) -> bool:
        return errors[i] > -target_m and errors[i + 1] > -target_m

    candidates = sorted(
        brackets,
        key=lambda i: (
            not both_print(i),
            abs(0.5 * (probes[i] + probes[i + 1]) - 1.0),
        ),
    )
    for pick in candidates:
        lo, hi = float(probes[pick]), float(probes[pick + 1])
        f_lo = errors[pick]
        for _ in range(max_iter):
            mid_dose = 0.5 * (lo + hi)
            f_mid = cd_error(mid_dose)
            if abs(f_mid) < tol_nm * 1e-9:
                return float(mid_dose)
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid_dose, f_mid
            else:
                hi = mid_dose

    logger.warning(
        "dose-to-size: CD crosses %.1f nm only at an emergence discontinuity; "
        "no dose sizes to within %.2f nm. Returning NaN.",
        target_cd_nm, tol_nm,
    )
    return float("nan")


# ---------------------------------------------------------------------------
# EL-vs-DOF trade-off curve
# ---------------------------------------------------------------------------


def compute_el_dof_curve(
    bossung_df: pd.DataFrame,
    target_cd_nm: float,
    tolerance_pct: float = 10.0,
    nominal_dose: float = 1.0,
) -> pd.DataFrame:
    """The exposure-latitude available at every depth-of-focus demand.

    For every contiguous focus window in the sweep, finds the widest
    contiguous dose band that stays in spec across the *whole* window, and
    keeps the best EL seen at each DOF. Wider focus windows admit narrower
    dose bands, so the curve falls — the shape a process engineer trades
    along.

    Resolution note: this curve works on sweep nodes (no sub-node
    interpolation), so treat the scalar EL/DOF from
    :func:`compute_process_window` — which does interpolate — as the
    headline numbers, and this as the trade-off's shape.

    Parameters
    ----------
    bossung_df : pd.DataFrame
        Output of :func:`sweep_dose_focus`.
    target_cd_nm : float
        Nominal CD [nm].
    tolerance_pct : float
        Allowed CD variation [%].
    nominal_dose : float
        Centre dose the EL percentages are referenced to.

    Returns
    -------
    pd.DataFrame
        Columns ``dof_nm``, ``el_pct``, sorted by ``dof_nm``. Empty when
        nothing is in spec anywhere.
    """
    mat = bossung_df.pivot_table(index="dose", columns="defocus_nm", values="cd_nm")
    mat = mat.sort_index().sort_index(axis=1)
    doses = mat.index.to_numpy(dtype=np.float64)
    foci = mat.columns.to_numpy(dtype=np.float64)
    cd = mat.to_numpy(dtype=np.float64)

    ok = np.zeros_like(cd, dtype=bool)
    for a in range(cd.shape[0]):
        for b in range(cd.shape[1]):
            ok[a, b] = in_spec(float(cd[a, b]), target_cd_nm, tolerance_pct)

    best: dict[float, float] = {}
    n_focus = len(foci)
    for i0 in range(n_focus):
        for i1 in range(i0, n_focus):
            holds = ok[:, i0:i1 + 1].all(axis=1)
            runs = _contiguous_runs(holds)
            if not runs:
                continue
            r0, r1 = max(runs, key=lambda r: doses[r[1]] - doses[r[0]])
            el = float((doses[r1] - doses[r0]) / nominal_dose * 100.0)
            dof = float(foci[i1] - foci[i0])
            if el > best.get(dof, -1.0):
                best[dof] = el

    rows = sorted(best.items())
    return pd.DataFrame(rows, columns=["dof_nm", "el_pct"])


# ---------------------------------------------------------------------------
# MEEF
# ---------------------------------------------------------------------------


def compute_meef(
    mask: NDArray[np.float64],
    optics: OpticsConfig,
    grid: GridConfig,
    resist: ResistConfig,
    *,
    bias_nm: float | None = None,
    dose: float = 1.0,
    defocus_nm: float = 0.0,
    model: str = "threshold",
    normalisation: str | None = "clear",
) -> float:
    """Mask Error Enhancement Factor: d(wafer CD) / d(mask CD).

    Central difference over a ± mask bias — two extra Abbe sums. The mask CD
    increment in the denominator is *measured from the biased masks*, not
    assumed from *bias_nm*: :func:`~litho_sim.mask.patterns.apply_bias`
    rounds to whole pixels, so the requested and realised bias differ, and
    only the realised one keeps the derivative honest.

    Mask arrays in this codebase are specified at wafer scale (the grid is
    the image-plane grid; ``OpticsConfig.reduction`` only rescales the
    mask-side NA), so the increment is already a wafer-scale increment —
    the standard MEEF convention, with no ×4 correction.

    Parameters
    ----------
    mask, optics, grid, resist
        As for :func:`sweep_dose_focus`.
    bias_nm : float, optional
        Half-step of the central difference, per edge [nm]. Defaults to one
        pixel — the smallest bias :func:`apply_bias` can realise.
    dose : float
        Dose at which to evaluate (typically the dose-to-size anchor).
    defocus_nm : float
        Focus at which to evaluate [nm] (typically best focus).
    model : str
        Resist development model.
    normalisation : str, optional
        Normalisation mode (default ``"clear"``; ``None`` uses *optics* as-is).

    Returns
    -------
    float
        MEEF (dimensionless), or NaN when either biased point fails to print.

    Raises
    ------
    ValueError
        When *bias_nm* is below one grid pixel — the bias would round to
        nothing and the derivative would be 0/0.
    """
    px_nm = grid.pixel_size * 1e9
    if bias_nm is None:
        bias_nm = px_nm
    if bias_nm < px_nm - 1e-9:
        raise ValueError(
            f"bias_nm={bias_nm:.2f} nm is below one grid pixel ({px_nm:.2f} nm); "
            "apply_bias rounds to whole pixels and the bias would do nothing."
        )

    local = _optics_at_focus(optics, defocus_nm, normalisation)

    # The printed feature polarity, shared by the mask and wafer measurements
    # so numerator and denominator carry consistent signs. Positive tone
    # prints the mask's dark region.
    feature = "below" if resist.tone == "positive" else "above"

    mask_plus = apply_bias(mask, +bias_nm, grid.pixel_size)
    mask_minus = apply_bias(mask, -bias_nm, grid.pixel_size)

    cd_mask_plus = measure_cd_2d(mask_plus, grid.pixel_size, threshold=0.5, feature=feature)
    cd_mask_minus = measure_cd_2d(mask_minus, grid.pixel_size, threshold=0.5, feature=feature)
    d_mask = cd_mask_plus - cd_mask_minus
    if d_mask == 0.0:
        raise ValueError(
            "Realised mask ΔCD is zero — the bias saturated or the feature "
            "vanished; increase bias_nm or check the mask."
        )

    cd_wafer = []
    for biased in (mask_plus, mask_minus):
        aerial = compute_aerial_image(biased, local, grid, dose=dose)
        cd_wafer.append(_measure_point(aerial, resist, grid, model=model))
    if min(cd_wafer) <= 0.0:
        logger.warning("MEEF: a biased point does not print; returning NaN.")
        return float("nan")

    meef = (cd_wafer[0] - cd_wafer[1]) / d_mask
    logger.info(
        "MEEF = %.2f (mask ΔCD %.1f nm → wafer ΔCD %.1f nm)",
        meef, d_mask * 1e9, (cd_wafer[0] - cd_wafer[1]) * 1e9,
    )
    return float(meef)


# ---------------------------------------------------------------------------
# Convenience: full analysis from a SimulationConfig
# ---------------------------------------------------------------------------


def run_full_analysis(
    cfg: SimulationConfig,
    pitch_nm: float = 200.0,
    target_cd_nm: float = 100.0,
    n_doses: int = 5,
    n_defoci: int = 11,
    defocus_range_nm: float = 300.0,
    dose_range: float = 0.3,
    tolerance_pct: float = 10.0,
    model: str = "threshold",
    normalisation: str | None = "clear",
) -> tuple[pd.DataFrame, dict]:
    """One-shot helper: build mask, sweep, and compute process window.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation configuration.
    pitch_nm : float
        L/S pitch [nm].
    target_cd_nm : float
        Target / nominal CD [nm].
    n_doses : int
        Number of dose levels in the sweep.
    n_defoci : int
        Number of defocus steps in the sweep.
    defocus_range_nm : float
        Half-range of defocus sweep [nm].  Sweep spans
        −range … +range.
    dose_range : float
        Fractional dose variation around 1.0
        (e.g. 0.3 → doses from 0.7 to 1.3).
    tolerance_pct : float
        CD tolerance for process window [%].
    model : str
        Resist development model.
    normalisation : str, optional
        Sweep normalisation mode (see :func:`sweep_dose_focus`).

    Returns
    -------
    bossung_df : pd.DataFrame
        Full dose × defocus sweep results.
    pw_results : dict
        Process window metrics from :func:`compute_process_window`.
    """
    # Build mask
    mask = lines_and_spaces(
        n_pixels=cfg.grid.n_pixels,
        pixel_size=cfg.grid.pixel_size,
        pitch=pitch_nm * 1e-9,
        cd=target_cd_nm * 1e-9,
    )

    doses = list(np.linspace(1.0 - dose_range, 1.0 + dose_range, n_doses))
    defoci = list(np.linspace(-defocus_range_nm, defocus_range_nm, n_defoci))

    bossung_df = sweep_dose_focus(
        mask=mask,
        optics=cfg.optics,
        grid=cfg.grid,
        resist=cfg.resist,
        doses=doses,
        defoci_nm=defoci,
        model=model,
        normalisation=normalisation,
        target_cd_nm=target_cd_nm,
    )

    pw_results = compute_process_window(
        bossung_df, target_cd_nm, tolerance_pct, nominal_dose=1.0
    )
    return bossung_df, pw_results
