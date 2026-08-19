"""Shared foundations: configuration dataclasses, grids, and logging.

Everything in the engine imports downward from here; ``core`` imports nothing
from the rest of the package.
"""

from litho_sim.core.config import (
    GridConfig,
    OpticsConfig,
    ResistConfig,
    SimulationConfig,
)
from litho_sim.core.utils import setup_logging

__all__ = [
    "GridConfig",
    "OpticsConfig",
    "ResistConfig",
    "SimulationConfig",
    "setup_logging",
]
