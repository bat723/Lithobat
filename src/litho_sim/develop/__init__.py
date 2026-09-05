"""Development: latent image → resist profile, in 2-D and 3-D.

``resist`` is the 2-D pipeline (Dill exposure, development models, CD
measurement); ``resist3d`` is the depth-resolved pipeline (exposure volume,
absorption, standing waves, 3-D develop, profile metrics). The exposure-side
physics inside ``resist3d`` is scheduled to migrate to ``litho_sim.expose``
when the vector-imaging work lands; import from this package, not the
modules, to stay insulated from that split.
"""

from litho_sim.develop.calibrate import Calibration, calibrate_profile, profile_cd
from litho_sim.develop.front import (
    arrival_time,
    arrival_time_front,
    develop_front,
)
from litho_sim.develop.profile import (
    CLEARED_FRACTION,
    SEALED_FRACTION,
    film_remaining,
    is_cleared,
    is_sealed,
    top_loss,
)
from litho_sim.develop.resist import (
    cleared_depth,
    develop_field,
    dill_exposure,
    feature_edges,
    just_clearing_time,
    mack_development_rate,
    measure_cd_1d,
    measure_cd_2d,
    remaining_thickness,
    simulate_resist,
    threshold_development,
)
from litho_sim.develop.resist3d import (
    BAKE_MODELS,
    add_standing_waves,
    apply_absorption,
    apply_vertical_interference,
    arrival_field,
    develop_3d,
    dissolution_rate_3d,
    effective_defocus,
    exposure_volume,
    focus_reference_depth,
    latent_volume,
    print_resist_3d,
    sidewall_angle,
    tmm_standing_waves,
    write_resist_to_stack,
)
from litho_sim.develop.stochastic import (
    StochasticProfileResult,
    StochasticResult,
    add_edge_roughness,
    stochastic_trials,
    stochastic_trials_3d,
)

__all__ = [
    "remaining_thickness", "just_clearing_time",
    "cleared_depth", "develop_field", "dill_exposure", "feature_edges", "mack_development_rate",
    "measure_cd_1d", "measure_cd_2d",
    "simulate_resist", "threshold_development",
    "arrival_time", "arrival_time_front", "develop_front",
    "Calibration", "calibrate_profile", "profile_cd", "arrival_field", "dissolution_rate_3d",
    "StochasticResult", "add_edge_roughness", "stochastic_trials",
    "StochasticProfileResult", "stochastic_trials_3d", "latent_volume", "BAKE_MODELS",
    "add_standing_waves", "apply_absorption", "apply_vertical_interference",
    "develop_3d", "effective_defocus", "exposure_volume", "focus_reference_depth",
    "print_resist_3d", "sidewall_angle", "tmm_standing_waves", "write_resist_to_stack",
    "film_remaining", "top_loss", "is_sealed", "is_cleared",
    "SEALED_FRACTION", "CLEARED_FRACTION",
]
