"""
Unit tests for illumination sources and their coupling into the Abbe sum.

The headline concern here is *source sampling*. The Abbe sum costs one FFT
per source point, so how the source is sampled sets both accuracy and
runtime. These tests pin the convergence behaviour that
``OpticsConfig.source_grid`` controls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.expose.aerial_image import compute_aerial_image
from litho_sim.expose.illumination import annular, build_source, conventional, dipole, quadrupole
from litho_sim.mask.patterns import lines_and_spaces

# ---------------------------------------------------------------------------
# Source shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        conventional(21, 0.8),
        annular(21, 0.8, 0.5),
        dipole(21, 0.8, 0.5),
        quadrupole(21, 0.8, 0.5),
    ],
)
def test_sources_are_normalised(src):
    assert src.sum() == pytest.approx(1.0)
    assert (src >= 0.0).all()


def test_annular_rejects_inverted_sigmas():
    with pytest.raises(ValueError):
        annular(21, sigma_outer=0.4, sigma_inner=0.6)


def test_annular_is_hollow():
    """An annular source must have nothing at the centre."""
    src = annular(21, 0.8, 0.5)
    assert src[10, 10] == 0.0


def test_dipole_axis_is_honoured():
    """dipole(axis=...) was unreachable before source_kwargs was forwarded."""
    x = dipole(21, 0.8, 0.4, axis="x")
    y = dipole(21, 0.8, 0.4, axis="y")
    assert not np.array_equal(x, y)
    assert np.allclose(x, y.T)


def test_build_source_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown source type"):
        build_source(21, "trefoil", 0.8)


# ---------------------------------------------------------------------------
# Source sampling in the Abbe sum
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=64, pixel_size=4e-9)


@pytest.fixture(scope="module")
def mask(grid) -> np.ndarray:
    return lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=200e-9, cd=100e-9)


def test_source_grid_converges(mask, grid):
    """A coarse source grid must approach the dense result as it is refined.

    This is what justifies the default of 21: it is ~40x cheaper than
    sampling the source on the mask grid, for well under 1% intensity error.
    """
    dense = compute_aerial_image(
        mask, OpticsConfig(sigma_outer=0.8, source_grid=grid.n_pixels), grid
    )
    errors = []
    for sg in (11, 21, 41):
        a = compute_aerial_image(mask, OpticsConfig(sigma_outer=0.8, source_grid=sg), grid)
        errors.append(float(np.abs(a - dense).max()))

    assert errors == sorted(errors, reverse=True), (
        f"Refining the source grid should reduce error monotonically, got {errors}. "
        "Non-monotonic convergence means source points are being quantised onto "
        "the FFT lattice instead of placed exactly."
    )
    assert errors[1] < 0.01, f"source_grid=21 error {errors[1]:.4f} exceeds 1%"


def test_source_grid_does_not_change_normalisation(mask, grid):
    for sg in (11, 21, 41):
        a = compute_aerial_image(mask, OpticsConfig(sigma_outer=0.8, source_grid=sg), grid)
        assert a.max() == pytest.approx(1.0)


def test_source_kwargs_reach_the_builder(mask, grid):
    """OpticsConfig.source_kwargs must actually influence the aerial image.

    compute_aerial_image used to call build_source without forwarding kwargs,
    which made dipole(axis=...) and quadrupole(rotation_deg=...) dead code.
    """
    common = dict(source_type="dipole", sigma_outer=0.8, sigma_inner=0.4, source_grid=21)
    ax = compute_aerial_image(mask, OpticsConfig(**common, source_kwargs={"axis": "x"}), grid)
    ay = compute_aerial_image(mask, OpticsConfig(**common, source_kwargs={"axis": "y"}), grid)
    assert not np.allclose(ax, ay), "dipole axis had no effect on the aerial image"


def test_dipole_beats_conventional_for_dense_lines(grid):
    """Off-axis illumination should improve contrast at aggressive pitch.

    This is the reason multi-patterning flows want dipole for mandrel layers.
    """
    dense_mask = lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=110e-9, cd=55e-9)

    def contrast(optics):
        a = compute_aerial_image(dense_mask, optics, grid)
        return (a.max() - a.min()) / (a.max() + a.min())

    conv = contrast(OpticsConfig(source_type="conventional", sigma_outer=0.8, source_grid=21))
    dip = contrast(
        OpticsConfig(
            source_type="dipole", sigma_outer=0.9, sigma_inner=0.6,
            source_grid=21, source_kwargs={"axis": "x"},
        )
    )
    assert dip > conv, f"dipole contrast {dip:.3f} did not beat conventional {conv:.3f}"
