"""Analysis: process windows, Bossung curves, NILS, and results I/O.

``process_window`` is the sweep/window machinery; ``data_utils`` validates
and pivots the resulting DataFrames.
"""

from litho_sim.analysis.data_utils import (
    pivot_cd_matrix,
    save_results_csv,
    validate_dataframe,
)
from litho_sim.analysis.process_window import (
    calibrate_dose_to_size,
    compute_depth_of_focus,
    compute_el_dof_curve,
    compute_exposure_latitude,
    compute_meef,
    compute_nils,
    compute_process_window,
    evaluate_cd,
    in_spec,
    run_full_analysis,
    sweep_dose_focus,
)

__all__ = [
    "pivot_cd_matrix", "save_results_csv", "validate_dataframe",
    "calibrate_dose_to_size", "compute_depth_of_focus", "compute_el_dof_curve",
    "compute_exposure_latitude", "compute_meef", "compute_nils",
    "compute_process_window", "evaluate_cd", "in_spec",
    "run_full_analysis", "sweep_dose_focus",
]
