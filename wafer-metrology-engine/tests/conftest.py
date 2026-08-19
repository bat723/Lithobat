"""
Shared pytest fixtures and path setup.

Puts ``src`` on the path so the suite runs against the working tree without an
editable install, matching how ``run.py`` bootstraps itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafer_metrology.synthesize import (  # noqa: E402
    WaferSurface,
    apply_mask,
    make_wafer_grid,
    synthesize_wafer,
    synthesize_wafer_pair,
)

WAFER_RADIUS = 0.15
"""Radius of a 300 mm wafer [m], used throughout the suite."""


@pytest.fixture(scope="session")
def grid():
    """A 192 px wafer grid: ``(x, y, r, theta, mask)``."""
    return make_wafer_grid(192)


@pytest.fixture(scope="session")
def wafer():
    """A deterministic 192 px synthetic wafer with defects."""
    return synthesize_wafer(n_pixels=192, seed=0)


@pytest.fixture(scope="session")
def wafer_pair():
    """A deterministic 192 px front/back wafer pair."""
    return synthesize_wafer_pair(n_pixels=192, seed=0)


def make_surface(x, y, z, mask, radius: float = WAFER_RADIUS) -> WaferSurface:
    """Wrap analytic arrays into a :class:`WaferSurface` for the metrology calls.

    Parameters
    ----------
    x, y : NDArray
        Coordinate grids [m].
    z : NDArray
        Height map [m], masked or not.
    mask : NDArray of bool
        Wafer aperture.
    radius : float
        Wafer radius [m].

    Returns
    -------
    WaferSurface
        Surface with the grid metadata the analysis functions expect.
    """
    n = z.shape[0]
    return WaferSurface(
        x=x, y=y, z=apply_mask(z, mask), mask=mask,
        radius=radius, pixel_size=2.0 * radius / n,
    )
