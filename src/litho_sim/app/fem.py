"""
The process-window computation behind the app's Process Window tab.

Qt-free on purpose, like the rest of the app core: the tab hands a
:class:`FemRequest` to the worker thread, :func:`compute_fem` turns it into a
:class:`ProcessWindowResult`, and the tab draws whatever comes back. Tests
drive this module headless.

Deliberately **no Pipeline involvement**: :func:`~litho_sim.analysis.sweep_dose_focus`
already reuses the raw Abbe sum across the dose axis internally, so the staged
cache has nothing to add — and, more to the point, a sweep pushed through the
worker's shared Pipeline would evict the live view's ``aerial`` entry with a
dozen foreign defocus signatures, making the next slider drag pay a full
Abbe sum for no reason.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from litho_sim.analysis import (
    calibrate_dose_to_size,
    compute_el_dof_curve,
    compute_meef,
    compute_process_window,
    sweep_dose_focus,
)
from litho_sim.app.compute import build_mask
from litho_sim.app.params import ParameterModel

logger = logging.getLogger(__name__)


@dataclass
class FemRequest:
    """Everything one press of the Run button asks for.

    ``params`` is a deep copy of the dock's model, snapshotted on the GUI
    side — the sweep must describe the optics the user was looking at when
    they pressed Run, not whatever the sliders say by the time the worker
    gets around to it. The sweep ranges live here rather than in the dock
    because they parameterise this tab alone (see StackTab's docstring for
    why tab-local knobs stay out of the global parameter model).
    """

    params: ParameterModel
    focus_half_range_nm: float = 300.0
    n_focus: int = 11
    dose_half_range: float = 0.30
    n_dose: int = 7
    target_cd_nm: float = 100.0
    tolerance_pct: float = 10.0
    auto_centre_dose: bool = True
    with_meef: bool = True


@dataclass
class ProcessWindowResult:
    """One FEM sweep, measured and summarised."""

    bossung_df: pd.DataFrame        # dose, defocus_nm, cd_nm, cd_m, nils
    pw: dict                        # compute_process_window output
    el_dof: pd.DataFrame            # dof_nm, el_pct
    nominal_dose: float             # centre of the dose axis
    dose_anchored: bool             # True when dose-to-size found the centre
    meef: float                     # NaN when skipped or undefined
    target_cd_nm: float
    tolerance_pct: float
    label: str                      # what was swept, for the figure title
    elapsed_ms: float

    @property
    def summary(self) -> str:
        if self.window_empty:
            return f"no process window    [{self.elapsed_ms:.0f} ms]"
        bits = [
            f"EL {self.pw['EL_pct']:.1f}%",
            f"DOF {self.pw['DOF_nm']:.0f} nm",
            f"best focus {self.pw['best_focus_nm']:+.0f} nm",
        ]
        if self.dose_anchored:
            bits.append(f"dose-to-size {self.nominal_dose:.3f}")
        if np.isfinite(self.meef):
            bits.append(f"MEEF {self.meef:.2f}")
        return "    ".join(bits) + f"    [{self.elapsed_ms:.0f} ms]"

    @property
    def window_empty(self) -> bool:
        """True when there is no in-spec window to draw."""
        return (
            not self.pw.get("prints", False)
            or self.pw["EL_pct"] <= 0.0
            or self.pw["DOF_nm"] <= 0.0
        )

    @property
    def diagnosis(self) -> str:
        """Why the panel shows words instead of a window, or ``""``.

        Two distinct failures get two distinct explanations: nothing printed
        at all (a dose-range problem), and printing that never lands in spec
        (a target/tolerance problem) — telling a user to widen the dose range
        when their target CD is simply wrong sends them the wrong way.
        """
        if not self.window_empty:
            return ""
        printing = self.bossung_df[self.bossung_df["cd_nm"] > 0]
        if printing.empty:
            lo = float(self.bossung_df["dose"].min())
            hi = float(self.bossung_df["dose"].max())
            return (
                "No (dose, focus) point printed in spec — nothing printed at "
                f"all. The swept doses ({lo:.2f}–{hi:.2f}, clear-field "
                "normalised) may sit below the printing threshold — try "
                "'centre dose on target CD', widen the dose range, or lower "
                "the resist threshold."
            )
        window = self.pw.get("window_df")
        if window is not None and bool(window["in_spec"].any()):
            k = int(window["in_spec"].sum())
            return (
                f"{k} of {len(window)} points print in spec, but they form "
                "no usable window at the selected best focus and dose — "
                "widen the sweep so the in-spec band has out-of-spec "
                "neighbours to interpolate against, or loosen the tolerance."
            )
        row = printing.loc[
            (printing["cd_nm"] - self.target_cd_nm).abs().idxmin()
        ]
        return (
            "No (dose, focus) point printed in spec — the pattern prints, "
            f"but never within ±{self.tolerance_pct:.0f}% of "
            f"{self.target_cd_nm:.0f} nm. Closest was {row['cd_nm']:.1f} nm "
            f"at dose {row['dose']:.3f}, defocus {row['defocus_nm']:+.0f} nm. "
            "Adjust the target CD or tolerance."
        )


def compute_fem(
    req: FemRequest,
    progress: Callable[[int, int], None] | None = None,
) -> ProcessWindowResult:
    """Run the full focus-exposure matrix for one parameter snapshot.

    Cost is ``n_focus`` Abbe sums for the sweep, plus one for the
    dose-to-size anchor and two for MEEF — the dose axis is a rescale of
    cached raw sums and is effectively free. *progress* is called with
    ``(done, total)`` in those same units.
    """
    t0 = time.perf_counter()
    params = req.params
    grid = params.grid()
    optics = params.optics()
    resist = params.resist()
    mask = build_mask(params)

    total = req.n_focus + (1 if req.auto_centre_dose else 0) + (2 if req.with_meef else 0)
    done = 0

    def tick(units: int = 1) -> None:
        nonlocal done
        done += units
        if progress is not None:
            progress(done, total)

    nominal_dose = 1.0
    dose_anchored = False
    if req.auto_centre_dose:
        anchored = calibrate_dose_to_size(
            mask, optics, grid, resist,
            target_cd_nm=req.target_cd_nm, defocus_nm=0.0,
        )
        if np.isfinite(anchored):
            nominal_dose = float(anchored)
            dose_anchored = True
        else:
            logger.warning(
                "FEM: dose-to-size found no dose printing %.0f nm; the dose "
                "axis stays centred on 1.0.", req.target_cd_nm,
            )
        tick()

    doses = list(
        nominal_dose
        * np.linspace(1.0 - req.dose_half_range, 1.0 + req.dose_half_range, req.n_dose)
    )
    defoci = list(
        np.linspace(-req.focus_half_range_nm, req.focus_half_range_nm, req.n_focus)
    )

    bossung_df = sweep_dose_focus(
        mask=mask, optics=optics, grid=grid, resist=resist,
        doses=doses, defoci_nm=defoci,
        target_cd_nm=req.target_cd_nm,
        progress=lambda _i, _n: tick(),
    )
    pw = compute_process_window(
        bossung_df, req.target_cd_nm, req.tolerance_pct, nominal_dose=nominal_dose
    )
    el_dof = compute_el_dof_curve(
        bossung_df, req.target_cd_nm, req.tolerance_pct, nominal_dose=nominal_dose
    )

    meef = float("nan")
    if req.with_meef and pw["prints"]:
        try:
            meef = compute_meef(
                mask, optics, grid, resist,
                dose=pw["best_dose"], defocus_nm=pw["best_focus_nm"],
            )
        except ValueError as exc:
            logger.warning("FEM: MEEF skipped — %s", exc)
        tick(2)

    label = (
        f"{params['pattern']} · pitch {params['pitch']:.0f} nm · "
        f"target {req.target_cd_nm:.0f} nm ±{req.tolerance_pct:.0f}% · "
        f"NA {params['NA']:.2f} · σ {params['sigma_outer']:.2f} · "
        f"{params['imaging_model']}, clear-normalised"
    )

    return ProcessWindowResult(
        bossung_df=bossung_df,
        pw=pw,
        el_dof=el_dof,
        nominal_dose=nominal_dose,
        dose_anchored=dose_anchored,
        meef=meef,
        target_cd_nm=req.target_cd_nm,
        tolerance_pct=req.tolerance_pct,
        label=label,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
