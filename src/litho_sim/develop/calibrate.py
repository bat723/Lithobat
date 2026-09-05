"""
Anchoring a 3-D profile: dose-to-size and develop time, found rather than set.

A finite-rate develop has two knobs that a preset cannot know in advance:
the dose that prints the drawn width, which depends on the pattern, the
optics and the film, and the time in the developer, which is only
meaningful against the time it takes to clear the spaces. Left at their
defaults they produced stumps (the dose was wrong) and empty fields (the
develop was long enough to eat through the lines). Every real process sets
both by measurement; this module does the same by simulation.

:func:`calibrate_profile` runs the exposure once, then bisects dose on the
rest of the chain — absorption, bake, rate law, arrival time — reading the
printed width at a chosen depth off the continuous arrival-time field with
a sub-pixel crossing, exactly as the process window reads a Bossung curve.
The develop time is set at every candidate dose as a multiple of the time
the fastest column takes to reach the substrate: a 1.5× over-develop by
default, which is where a process is normally run.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.develop.resist import measure_cd_1d
from litho_sim.develop.resist3d import (
    apply_vertical_interference,
    arrival_field,
    exposure_volume,
    latent_volume,
)

logger = logging.getLogger(__name__)

__all__ = ["Calibration", "calibrate_profile", "profile_cd"]


@dataclass
class Calibration:
    """What :func:`calibrate_profile` found.

    Attributes
    ----------
    dose : float
        Relative dose that prints the target width at the chosen depth.
    develop_time : float
        Seconds in the developer: *overdevelop* × the time to clear.
    clear_time : float
        Time for the fastest column to reach the substrate at *dose* [s].
    cd_nm : dict
        Printed width at the bottom, middle and top of the film [nm].
    resist : ResistConfig
        The input resist with ``develop_time`` set — what to print with.
    converged : bool
        Whether the bisection met its tolerance.
    """

    dose: float
    develop_time: float
    clear_time: float
    cd_nm: dict[str, float]
    resist: ResistConfig
    converged: bool


def profile_cd(
    field: NDArray[np.float64],
    level: float,
    feature: str,
    grid: GridConfig,
    depth_fraction: float = 0.5,
    axis: int = 1,
) -> float:
    """Printed width [m] at a depth through the film, read off a continuous field.

    *depth_fraction* is 0 at the substrate and 1 at the film top; the row
    (or column) through the field centre is measured with the same
    interpolated crossing the 2-D CD uses, so the answer is sub-voxel.
    """
    nz = field.shape[0]
    iz = int(round(np.clip(depth_fraction, 0.0, 1.0) * (nz - 1)))
    plane = field[iz]
    mid = plane.shape[1 - axis] // 2
    cut = plane[mid, :] if axis == 1 else plane[:, mid]
    return measure_cd_1d(cut, grid.pixel_size, threshold=level, feature=feature)


def _clear_time(field: NDArray[np.float64], level_feature: str) -> float:
    """Seconds for the fastest column to finish the substrate plane."""
    bottom = field[0]
    return float(bottom.min()) if level_feature == "above" else float("nan")


def calibrate_profile(
    mask: NDArray,
    optics: OpticsConfig,
    grid: GridConfig,
    resist: ResistConfig,
    target_cd_nm: float,
    *,
    develop_model: str = "mack",
    overdevelop: float = 2.0,
    depth_fraction: float = 0.5,
    axis: int = 1,
    standing_waves: bool = False,
    bleaching: bool = True,
    bake: str = "gaussian",
    dose_bounds: tuple[float, float] = (0.2, 6.0),
    max_develop: float = 120.0,
    tol_nm: float = 0.25,
    max_iter: int = 40,
) -> Calibration:
    """Find the dose and develop time that print *target_cd_nm* through the film.

    Parameters
    ----------
    mask, optics, grid, resist
        The process. ``resist.develop_time`` is ignored and replaced.
    target_cd_nm : float
        Width of the resist line at *depth_fraction*, in nanometres. Measured
        along *axis* through the field centre.
    develop_model : str
        ``"mack"`` (ray march, the default: fast, and the calibration is
        dominated by the vertical component of development anyway) or
        ``"front"``.
    overdevelop : float
        Develop time as a multiple of the time the fastest column takes to
        clear to the substrate. Anything from 1.5 to 6 prints the line;
        measured on the ArF preset the wall steepens by only a few degrees
        across that range, and the line's top holds because the recalibrated
        presets dissolve unexposed resist at a fraction of a nanometre per
        second. 2 is a normal process margin.
    depth_fraction : float
        Where to size the line: 0 substrate, 0.5 mid-film (default), 1 top.
    standing_waves, bleaching, bake
        Forwarded to the exposure chain — ``bake="car"`` calibrates the
        reaction–diffusion path, at its cost per bisection step.
    dose_bounds : (lo, hi)
        Bisection bracket on relative dose.
    max_develop : float
        A develop time above this [s] means the spaces never really opened:
        the dose is treated as too low rather than the develop as too long.
        Without it an under-exposed candidate produced a clear time of hours
        and the bisection walked to the bracket floor.
    tol_nm : float
        Stop when the width is within this of the target.
    max_iter : int
        Bisection cap.

    Returns
    -------
    Calibration

    Notes
    -----
    Only the exposure pays for the Abbe sums; every bisection step re-runs
    absorption, bake and rate law, about 50 ms at the default grid. The
    threshold develop model has no develop time and is not calibrated here.
    """
    if develop_model not in ("mack", "front"):
        raise ValueError(
            f"calibrate_profile needs a finite-rate develop model ('mack' or 'front'), "
            f"got {develop_model!r}"
        )
    if not 0.0 <= depth_fraction <= 1.0:
        raise ValueError(f"depth_fraction must be in [0, 1], got {depth_fraction}")

    intensity1, _z = exposure_volume(mask, optics, grid, resist, dose=1.0)
    if standing_waves:
        intensity1 = apply_vertical_interference(intensity1, resist, optics, grid)
    target_m = target_cd_nm * 1e-9

    def evaluate(dose: float):
        latent = latent_volume(intensity1 * dose, resist, grid, bake=bake,
                               bleaching=bleaching)["latent"]
        T, _level, feature = arrival_field(latent, resist, grid, model=develop_model)
        t_clear = _clear_time(T, feature)
        t_dev = overdevelop * t_clear
        cd = profile_cd(T, t_dev, feature, grid, depth_fraction, axis)
        return cd, t_dev, t_clear, T, feature

    # Positive tone: more dose, narrower line. A width of zero means the line
    # was lost (too much dose); a width spanning the field means the spaces
    # never opened at the measured depth (too little).
    lo, hi = dose_bounds
    sign = 1.0 if resist.tone == "positive" else -1.0

    def err(w: float, t_dev: float) -> float:
        # Signed distance from target, oriented so that err(lo) > 0 > err(hi):
        # positive means "more dose". A lost line is too much dose; spaces
        # that take longer than max_develop to open are too little, whatever
        # the width read at that depth says.
        if t_dev > max_develop:
            return sign * grid.grid_size
        if w <= 0.0:
            return -sign * grid.grid_size
        return sign * (w - target_m)

    def evaluate_err(dose: float) -> tuple[float, float]:
        cd, t_dev, *_ = evaluate(dose)
        return cd, err(cd, t_dev)

    w_lo, e_lo = evaluate_err(lo)
    w_hi, e_hi = evaluate_err(hi)
    if not (e_lo > 0.0 > e_hi):
        logger.warning(
            "calibrate_profile: target %.1f nm is not bracketed by doses %.2f "
            "(%.1f nm) and %.2f (%.1f nm)", target_cd_nm, lo, w_lo * 1e9, hi, w_hi * 1e9,
        )
    converged = False
    dose = 0.5 * (lo + hi)
    for _ in range(max_iter):
        dose = 0.5 * (lo + hi)
        w, e = evaluate_err(dose)
        if abs(e) <= tol_nm * 1e-9 and w > 0.0:
            converged = True
            break
        if e > 0.0:
            lo = dose
        else:
            hi = dose

    cd_mid, t_dev, t_clear, T, feature = evaluate(dose)
    cds = {
        name: profile_cd(T, t_dev, feature, grid, f, axis) * 1e9
        for name, f in (("bottom", 0.0), ("mid", 0.5), ("top", 1.0))
    }
    logger.info(
        "calibrate_profile: dose %.3f, develop %.2f s (%.1fx the %.2f s clear time), "
        "CD bottom/mid/top %.1f/%.1f/%.1f nm for target %.1f",
        dose, t_dev, overdevelop, t_clear, cds["bottom"], cds["mid"], cds["top"], target_cd_nm,
    )
    return Calibration(
        dose=float(dose),
        develop_time=float(t_dev),
        clear_time=float(t_clear),
        cd_nm=cds,
        resist=dataclasses.replace(resist, develop_time=float(t_dev)),
        converged=converged,
    )
