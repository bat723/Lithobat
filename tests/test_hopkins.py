"""
The two imaging paths that reuse the Abbe sum's work, pinned against it.

* :func:`compute_aerial_planes` must give, plane for plane, what
  :func:`compute_aerial_image` gives one plane at a time — the batching and
  the shared geometry are implementation, not physics.
* :class:`SOCSKernels` is the Abbe sum reassembled. With every eigenvector
  kept it must reproduce it to rounding, for any source, defocus and
  aberration set; and it must refuse the two processes it cannot represent
  rather than approximate them.
* The OPC print model chooses between the two by cost, and prints the same
  field either way.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.expose import (
    SOCSKernels,
    compute_aerial_image,
    compute_aerial_planes,
    source_points,
)
from litho_sim.mask import Layout, line_array, lines_and_spaces


@pytest.fixture
def grid():
    return GridConfig(n_pixels=64, pixel_size=4e-9)


@pytest.fixture
def mask(grid):
    return lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=128e-9, cd=64e-9)


OPTICS = {
    "conventional": dict(NA=0.93, sigma_outer=0.8, source_grid=9),
    "annular, defocused, aberrated": dict(
        NA=0.93, source_type="annular", sigma_outer=0.85, sigma_inner=0.5,
        source_grid=11, defocus=120e-9, zernike_coeffs={7: 0.03, 11: -0.02},
    ),
    "dipole, exact defocus, clear-normalised": dict(
        NA=1.35, n_immersion=1.44, n_image=1.7, source_type="dipole",
        source_kwargs={"axis": "x"}, sigma_outer=0.9, sigma_inner=0.6,
        source_grid=11, defocus=-60e-9, exact_defocus=True, normalisation="clear",
    ),
}


# ---------------------------------------------------------------------------
# Planes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(OPTICS))
def test_planes_match_one_image_at_a_time(mask, grid, name):
    optics = OpticsConfig(**OPTICS[name])
    planes = [(-100e-9, 1.7), (0.0, 1.7), (45e-9, 1.7), (150e-9, 1.44)]
    stack = compute_aerial_planes(mask, optics, grid, planes, dose=1.3)
    assert stack.shape == (len(planes), grid.n_pixels, grid.n_pixels)
    for p, (defocus, n_image) in enumerate(planes):
        one = compute_aerial_image(
            mask, dataclasses.replace(optics, defocus=defocus, n_image=n_image), grid, dose=1.3
        )
        np.testing.assert_allclose(stack[p], one, rtol=0, atol=1e-12 * one.max())


def test_planes_match_under_the_vector_model(mask, grid):
    optics = OpticsConfig(
        NA=1.35, n_immersion=1.44, imaging_model="vector", polarisation="te",
        source_type="dipole", source_kwargs={"axis": "x"}, sigma_outer=0.9,
        sigma_inner=0.6, source_grid=9, normalisation="clear",
    )
    planes = [(0.0, 1.7), (80e-9, 1.7)]
    stack = compute_aerial_planes(mask, optics, grid, planes)
    for p, (defocus, n_image) in enumerate(planes):
        one = compute_aerial_image(
            mask, dataclasses.replace(optics, defocus=defocus, n_image=n_image), grid
        )
        np.testing.assert_allclose(stack[p], one, rtol=0, atol=1e-12 * one.max())


def test_source_points_sum_to_the_rendered_source():
    optics = OpticsConfig(source_type="annular", sigma_outer=0.8, sigma_inner=0.5, source_grid=15)
    src = source_points(optics)
    assert len(src) > 0
    assert src.weight.sum() == pytest.approx(1.0)
    k_max = optics.NA / optics.wavelength
    assert np.all(np.hypot(src.fx, src.fy) <= k_max * (1.0 + 1e-12))
    assert np.all(np.hypot(src.fx, src.fy) >= 0.5 * k_max * (1.0 - 1e-12))


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(OPTICS))
def test_all_kernels_reproduce_the_abbe_sum_to_rounding(mask, grid, name):
    optics = OpticsConfig(**OPTICS[name])
    kernels = SOCSKernels.build(optics, grid, tol=0.0)
    raw_k, peak_k, clear_k = kernels.raw(mask)
    raw_a, peak_a, clear_a = compute_aerial_image(mask, optics, grid, return_raw=True)
    np.testing.assert_allclose(raw_k, raw_a, rtol=0, atol=1e-12 * peak_a)
    assert peak_k == pytest.approx(peak_a, rel=1e-12)
    assert clear_k == pytest.approx(clear_a, rel=1e-12)
    assert kernels.energy_kept == pytest.approx(1.0)
    # The normalised, dosed image goes through the same scalar.
    np.testing.assert_allclose(
        kernels.image(mask, dose=0.8), compute_aerial_image(mask, optics, grid, dose=0.8),
        rtol=0, atol=1e-12,
    )


def test_default_tolerance_keeps_the_image_to_a_part_in_1e8(mask, grid):
    optics = OpticsConfig(**OPTICS["dipole, exact defocus, clear-normalised"])
    full = SOCSKernels.build(optics, grid, tol=0.0)
    trimmed = SOCSKernels.build(optics, grid)
    assert len(trimmed) < len(full), "the default tolerance dropped nothing"
    ref = full.image(mask)
    assert np.abs(trimmed.image(mask) - ref).max() < 1e-8 * ref.max()
    assert trimmed.energy_kept > 1.0 - 1e-8


def test_kernel_count_is_bounded_by_band_and_source(grid):
    optics = OpticsConfig(**OPTICS["conventional"])
    kernels = SOCSKernels.build(optics, grid, tol=0.0)
    assert len(kernels) <= min(kernels.n_band, kernels.n_source)
    assert np.all(np.diff(kernels.weights) <= 0), "weights must descend"
    capped = SOCSKernels.build(optics, grid, max_kernels=3)
    assert len(capped) == 3


def test_kernels_refuse_what_they_cannot_represent(grid):
    vector = OpticsConfig(imaging_model="vector", source_grid=5)
    assert not SOCSKernels.supports(vector)
    with pytest.raises(ValueError, match="scalar thin-mask"):
        SOCSKernels.build(vector, grid)
    thick = OpticsConfig(mask_model="multilayer", wavelength=13.5e-9, NA=0.33, source_grid=5)
    with pytest.raises(ValueError, match="scalar thin-mask"):
        SOCSKernels.build(thick, grid)


def test_kernels_check_the_mask_shape(grid):
    kernels = SOCSKernels.build(OpticsConfig(**OPTICS["conventional"]), grid)
    with pytest.raises(ValueError, match="mask must be"):
        kernels.raw(np.ones((grid.n_pixels + 2, grid.n_pixels)))


# ---------------------------------------------------------------------------
# The print model's choice
# ---------------------------------------------------------------------------


@pytest.fixture
def layout(grid):
    return Layout(line_array(3, pitch=128e-9, cd=64e-9, length=0.8 * grid.grid_size))


def test_print_model_prints_the_same_field_either_way(layout, grid):
    from litho_sim.opc import PrintModel

    optics = OpticsConfig(**OPTICS["conventional"])
    resist = ResistConfig()
    kw = dict(dose=1.1, tone="dark", model="threshold")
    via_kernels = PrintModel(optics, resist, grid, imaging="socs", socs_tol=0.0, **kw)
    via_abbe = PrintModel(optics, resist, grid, imaging="abbe", **kw)
    assert via_kernels.uses_kernels and not via_abbe.uses_kernels
    a, b = via_kernels.print(layout), via_abbe.print(layout)
    np.testing.assert_allclose(a.aerial, b.aerial, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.field, b.field, rtol=0, atol=1e-12)
    assert a.level == b.level


def test_print_model_auto_takes_the_cheaper_engine(layout, grid):
    from litho_sim.opc import PrintModel

    optics = OpticsConfig(**OPTICS["conventional"])
    auto = PrintModel(optics, ResistConfig(), grid)
    assert auto.uses_kernels, "a scalar thin-mask process goes through the kernels"
    # A process the kernels cannot represent falls back without complaint...
    vector = PrintModel(dataclasses.replace(optics, imaging_model="vector"), ResistConfig(), grid)
    assert not vector.uses_kernels
    # ...and insisting on them for that process is refused.
    with pytest.raises(ValueError, match="scalar thin-mask"):
        PrintModel(dataclasses.replace(optics, imaging_model="vector"), ResistConfig(), grid,
                   imaging="socs")


def test_print_model_rebuilds_kernels_when_the_optics_change(layout, grid):
    from litho_sim.opc import PrintModel

    model = PrintModel(OpticsConfig(**OPTICS["conventional"]), ResistConfig(), grid, imaging="socs")
    first = model.kernels()
    assert model.kernels() is first, "same optics must reuse the kernels"
    model.optics = dataclasses.replace(model.optics, defocus=100e-9)
    assert model.kernels() is not first, "new optics must rebuild them"
