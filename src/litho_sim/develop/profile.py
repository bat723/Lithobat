"""
Metrics that describe a developed resist profile.

A 3-D develop returns a boolean volume; these turn it into the handful of
numbers a lithographer actually quotes. They live here rather than inside a
plotting function because a number you can only obtain by building a figure
is not really available: the top-loss formula below existed only as four
inline lines inside ``viz3d.resist_profile_3d_figure``, and had already been
copied verbatim into the Streamlit app rather than imported.

Conventions match the rest of the engine: volumes are indexed ``[iz, iy, ix]``
with **iz = 0 at the bottom of the resist**, and every length is in metres in,
nanometres out where the docstring says so.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from litho_sim.core.config import GridConfig
from litho_sim.develop.resist3d import sidewall_angle

logger = logging.getLogger(__name__)

#: Above this fraction the film has not developed at all; below the second, it
#: has cleared completely. Both are degenerate outcomes that make every other
#: metric meaningless, so they are named once here rather than being repeated
#: as bare literals wherever a diagnosis is printed.
SEALED_FRACTION = 0.995
CLEARED_FRACTION = 0.005

__all__ = [
    "film_remaining",
    "top_loss",
    "sidewall_angle",
    "is_sealed",
    "is_cleared",
    "SEALED_FRACTION",
    "CLEARED_FRACTION",
]


def film_remaining(remaining: NDArray[np.bool_]) -> float:
    """Fraction of the resist volume still standing, in [0, 1]."""
    return float(np.asarray(remaining).mean())


def top_loss(remaining: NDArray[np.bool_], grid: GridConfig) -> float:
    """Height of resist lost from the top of the film [nm].

    Development does not only cut downwards where it should; it also thins the
    film everywhere, because the top of the resist sees the most light. The
    difference between the nominal film top and the highest voxel that
    survived anywhere is the top loss — a direct read on how much budget the
    process has left before the pattern stops transferring.

    Parameters
    ----------
    remaining : NDArray[bool]
        ``(nz, ny, nx)`` developed volume, ``iz = 0`` at the bottom.
    grid : GridConfig
        Supplies ``dz``.

    Returns
    -------
    float
        Loss in nanometres, or ``nan`` if nothing survived — a fully cleared
        film has no top to have lost.
    """
    remaining = np.asarray(remaining)
    nz = remaining.shape[0]
    occupied = np.nonzero(remaining.any(axis=(1, 2)))[0]
    if occupied.size == 0:
        return float("nan")
    return float((nz - 1 - int(occupied.max())) * grid.dz * 1e9)


def is_sealed(remaining: NDArray[np.bool_]) -> bool:
    """True when the film essentially did not develop.

    Standing-wave nodes can seal the top under the threshold model, which
    produces a picture that looks like a bug and is not one.
    """
    return film_remaining(remaining) > SEALED_FRACTION


def is_cleared(remaining: NDArray[np.bool_]) -> bool:
    """True when the film cleared everywhere — dose too high, or too long a
    develop for the contrast available at this pitch."""
    return film_remaining(remaining) < CLEARED_FRACTION
