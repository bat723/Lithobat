"""
API tests for the pixel pattern builders, through the public package surface.

Everything here imports from :mod:`litho_sim.mask` — never the deep
``litho_sim.mask.patterns`` path — because that is the interface the app
(:func:`litho_sim.app.compute.build_mask`), the CLI demo, and the
process-window analysis actually consume.  Numbers are chosen so features
divide the grid exactly, letting assertions be exact rather than approximate.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.core.config import GridConfig, SimulationConfig
from litho_sim.expose import compute_aerial_image
from litho_sim.mask import (
    apply_bias,
    checkerboard,
    contact_array,
    isolated_line,
    lines_and_spaces,
    to_attenuated_psm,
)

# Grid used throughout: 4 nm pixels, so 64 nm pitch = 16 px and 32 nm CD = 8 px.
PX = 4e-9


# ---------------------------------------------------------------------------
# lines_and_spaces
# ---------------------------------------------------------------------------


class TestLinesAndSpaces:
    # The sampler quantises analytic edges onto the pixel grid, and because
    # nanometre values are not binary-exact each period boundary can land one
    # pixel early or late.  Assertions therefore allow +/- 1 px per feature —
    # exactly the guarantee real consumers (CLI demo, process window) rely on.
    def test_shape_binary_and_duty_cycle(self):
        n = 128
        mask = lines_and_spaces(n, PX, pitch=64e-9, cd=30e-9)
        assert mask.shape == (n, n)
        assert mask.dtype == np.float64
        assert set(np.unique(mask)) == {0.0, 1.0}
        # 8 lines of ideally 7.5 px each: total bright within 1 px per line.
        ideal_bright = n * 30e-9 / 64e-9  # 60 px per row
        assert abs(mask[0].sum() - ideal_bright) <= 8

    def test_line_widths_and_pitch(self):
        n, pitch_px, cd_px = 128, 16, 30e-9 / PX  # 7.5 px ideal width
        mask = lines_and_spaces(n, PX, pitch=64e-9, cd=30e-9)
        row = mask[0].astype(int)
        edges = np.diff(np.concatenate(([0], row, [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        widths = ends - starts
        # Every full line is cd wide to within one pixel of quantisation.
        assert np.all(np.abs(widths[1:-1] - cd_px) <= 1)
        # Centre-to-centre spacing of consecutive lines is the pitch, +/- 1 px.
        centres = (starts + ends) / 2
        assert np.all(np.abs(np.diff(centres) - pitch_px) <= 1)
        # Vertical lines: every row identical.
        assert np.array_equal(mask, np.tile(mask[0], (n, 1)))

    def test_horizontal_is_transpose_of_vertical(self):
        v = lines_and_spaces(64, PX, pitch=64e-9, cd=30e-9, orientation="vertical")
        h = lines_and_spaces(64, PX, pitch=64e-9, cd=30e-9, orientation="horizontal")
        assert np.array_equal(h, v.T)

    def test_invalid_orientation_raises(self):
        with pytest.raises(ValueError, match="orientation"):
            lines_and_spaces(64, PX, pitch=64e-9, cd=32e-9, orientation="diagonal")

    def test_zero_pitch_degenerates_to_dark_mask(self):
        # No validation exists for pitch <= 0; the documented-by-test outcome
        # of pitch=0 is an all-opaque mask (nothing prints), not an exception.
        mask = lines_and_spaces(16, PX, pitch=0.0, cd=8e-9)
        assert mask.shape == (16, 16)
        assert np.all(mask == 0.0)

    def test_degenerate_single_pixel_grid(self):
        mask = lines_and_spaces(1, PX, pitch=64e-9, cd=32e-9)
        assert mask.shape == (1, 1)
        assert mask[0, 0] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# contact_array
# ---------------------------------------------------------------------------


class TestContactArray:
    def test_shape_binary_and_area_fraction(self):
        n = 128
        mask = contact_array(n, PX, pitch_x=64e-9, pitch_y=64e-9, cd_x=30e-9)
        assert mask.shape == (n, n)
        assert set(np.unique(mask)) == {0.0, 1.0}
        # Openings are separable, so check each axis: ideally 60 of 128
        # pixels open, +/- 1 px per each of the 8 periods (quantisation).
        for axis in (0, 1):
            open_count = mask.any(axis=axis).sum()
            assert abs(open_count - 60) <= 8

    def test_cd_y_defaults_to_cd_x(self):
        square = contact_array(64, PX, pitch_x=64e-9, pitch_y=64e-9, cd_x=32e-9)
        explicit = contact_array(
            64, PX, pitch_x=64e-9, pitch_y=64e-9, cd_x=32e-9, cd_y=32e-9
        )
        assert np.array_equal(square, explicit)

    def test_mask_is_separable_outer_product(self):
        mask = contact_array(64, PX, pitch_x=64e-9, pitch_y=32e-9, cd_x=32e-9, cd_y=16e-9)
        open_x = mask.any(axis=0)
        open_y = mask.any(axis=1)
        assert np.array_equal(mask, np.outer(open_y, open_x).astype(np.float64))

    def test_rectangular_contacts_are_wider_than_tall(self):
        mask = contact_array(128, PX, pitch_x=64e-9, pitch_y=64e-9, cd_x=30e-9, cd_y=14e-9)
        open_x = mask.any(axis=0).sum()  # ideally 60 px
        open_y = mask.any(axis=1).sum()  # ideally 28 px
        assert abs(open_x - 60) <= 8
        assert abs(open_y - 28) <= 8
        assert open_y < open_x


# ---------------------------------------------------------------------------
# isolated_line
# ---------------------------------------------------------------------------


class TestIsolatedLine:
    def test_single_centred_line_of_expected_width(self):
        n = 64
        # cd = 30 nm with 4 nm pixels: |coord| < 15 nm selects coords
        # -12, -8, -4, 0, 4, 8, 12 nm -> exactly 7 pixels (off-boundary on
        # purpose so float rounding cannot flip an edge pixel).
        mask = isolated_line(n, PX, cd=30e-9)
        assert mask.shape == (n, n)
        assert set(np.unique(mask)) == {0.0, 1.0}
        row = mask[0]
        assert np.array_equal(mask, np.tile(row, (n, 1)))
        bright = np.flatnonzero(row)
        assert bright.size == 7
        # Contiguous and centred on the grid origin (index n//2).
        assert np.array_equal(bright, np.arange(bright[0], bright[0] + 7))
        assert bright[3] == n // 2

    def test_horizontal_orientation(self):
        v = isolated_line(64, PX, cd=30e-9, orientation="vertical")
        h = isolated_line(64, PX, cd=30e-9, orientation="horizontal")
        assert np.array_equal(h, v.T)


# ---------------------------------------------------------------------------
# checkerboard
# ---------------------------------------------------------------------------


class TestCheckerboard:
    N = 64
    PITCH = 32e-9  # 8 px cells -> 8x8 cells on the grid, exactly.
    CD = 14e-9  # hole edges 1 nm off pixel centres, so no boundary flips

    def _mask(self):
        return checkerboard(self.N, PX, pitch=self.PITCH, cd=self.CD)

    def test_shape_binary_nonempty(self):
        mask = self._mask()
        assert mask.shape == (self.N, self.N)
        assert set(np.unique(mask)) == {0.0, 1.0}
        assert mask.sum() > 0

    def test_adjacent_cells_never_both_hold_holes(self):
        mask = self._mask()
        cell_px = 8
        # Shifting by one cell lands every hole on an inactive cell.
        assert np.all(mask * np.roll(mask, cell_px, axis=1) == 0)
        assert np.all(mask * np.roll(mask, cell_px, axis=0) == 0)

    def test_periodic_with_two_cell_pitch(self):
        mask = self._mask()
        # The alternation period is 2 cells = 16 px, and 64/16 divides exactly.
        assert np.array_equal(mask, np.roll(mask, 16, axis=1))
        assert np.array_equal(mask, np.roll(mask, 16, axis=0))


# ---------------------------------------------------------------------------
# apply_bias
# ---------------------------------------------------------------------------


class TestApplyBias:
    def _line(self):
        return isolated_line(64, PX, cd=30e-9)  # 7 px wide (see above)

    @staticmethod
    def _width(mask):
        # Measured on the centre row: erosion treats pixels beyond the array
        # edge as opaque (border_value=0), so rows touching the border thin out.
        return int(mask[32].sum())

    def test_positive_bias_dilates_by_whole_pixels(self):
        biased = apply_bias(self._line(), bias_nm=4.0, pixel_size=PX)
        # 4 nm on a 4 nm grid = 1 px each side: 7 -> 9.
        assert self._width(biased) == 9
        assert set(np.unique(biased)) <= {0.0, 1.0}

    def test_negative_bias_erodes(self):
        biased = apply_bias(self._line(), bias_nm=-4.0, pixel_size=PX)
        assert self._width(biased) == 5

    def test_zero_and_subpixel_bias_return_unchanged_copy(self):
        mask = self._line()
        for bias in (0.0, 1.0):  # 1 nm rounds to 0 px on a 4 nm grid
            out = apply_bias(mask, bias_nm=bias, pixel_size=PX)
            assert np.array_equal(out, mask)
            assert out is not mask  # a copy, as process_window relies on

    def test_bias_round_trip_on_dense_lines(self):
        # The process-window flow biases +/- around nominal; on a periodic
        # L/S pattern dilate-then-erode must restore the original.
        mask = lines_and_spaces(128, PX, pitch=64e-9, cd=30e-9)
        widened = apply_bias(mask, bias_nm=4.0, pixel_size=PX)
        restored = apply_bias(widened, bias_nm=-4.0, pixel_size=PX)
        assert widened.sum() > mask.sum()
        # Compare away from the array border, where erosion's border_value=0
        # eats one pixel that dilation had pushed outside the grid.
        crop = np.s_[2:-2, 2:-2]
        assert np.array_equal(restored[crop], mask[crop])


# ---------------------------------------------------------------------------
# to_attenuated_psm
# ---------------------------------------------------------------------------


class TestToAttenuatedPSM:
    def test_default_conversion_values(self):
        mask = lines_and_spaces(64, PX, pitch=64e-9, cd=32e-9)
        psm = to_attenuated_psm(mask)
        assert psm.dtype == np.complex128
        assert psm.shape == mask.shape
        chrome = mask > 0.5
        # 6 % intensity transmission with a 180 degree phase flip.
        assert np.allclose(psm[chrome], -np.sqrt(0.06))
        assert np.allclose(np.abs(psm[chrome]) ** 2, 0.06)
        assert np.allclose(psm[~chrome], 1.0 + 0.0j)

    def test_custom_transmission_and_phase(self):
        mask = np.ones((4, 4))
        psm = to_attenuated_psm(mask, transmission=0.25, phase_shift_deg=0.0)
        assert np.allclose(psm, 0.5 + 0.0j)


# ---------------------------------------------------------------------------
# End-to-end: patterns feeding the real imaging engine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> SimulationConfig:
    c = SimulationConfig.from_tech_node("ArF")
    c.grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    return c


class TestImagingConsumption:
    def test_binary_mask_prints_through_aerial_image(self, cfg):
        mask = lines_and_spaces(
            cfg.grid.n_pixels, cfg.grid.pixel_size, pitch=200e-9, cd=100e-9
        )
        aerial = compute_aerial_image(mask, cfg.optics, cfg.grid, dose=1.0)
        assert aerial.shape == mask.shape
        assert np.isfinite(aerial).all()
        assert aerial.min() >= 0.0
        # The image must modulate, and light must land where the mask is clear.
        assert aerial.max() - aerial.min() > 0.2
        assert aerial[mask == 1.0].mean() > aerial[mask == 0.0].mean()

    def test_attenuated_psm_mask_is_accepted_by_imaging(self, cfg):
        binary = lines_and_spaces(
            cfg.grid.n_pixels, cfg.grid.pixel_size, pitch=200e-9, cd=100e-9
        )
        psm = to_attenuated_psm(1.0 - binary)  # chrome where the binary mask is dark
        aerial = compute_aerial_image(psm, cfg.optics, cfg.grid, dose=1.0)
        assert aerial.dtype == np.float64  # intensity comes back real
        assert np.isfinite(aerial).all()
        assert aerial.min() >= 0.0
        # The 6 % leakage background must differ from the binary-mask image.
        binary_aerial = compute_aerial_image(binary, cfg.optics, cfg.grid, dose=1.0)
        assert not np.allclose(aerial, binary_aerial)
