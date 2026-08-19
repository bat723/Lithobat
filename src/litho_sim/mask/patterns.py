"""
Mask pattern generation module.

All functions return 2-D **transmittance** arrays with values in [0, 1]:
1 = clear / bright (light passes), 0 = opaque / dark. This is exactly how
``compute_aerial_image`` consumes them — the array is multiplied into the
field, so a 1 prints and a 0 does not. (``to_attenuated_psm`` is the odd one
out: it treats 1-regions as the *feature* to convert and returns a complex
transmittance.)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def lines_and_spaces(
    n_pixels: int,
    pixel_size: float,
    pitch: float,
    cd: float,
    orientation: str = "vertical",
) -> NDArray[np.float64]:
    """Generate a periodic line/space (L/S) mask pattern.

    Parameters
    ----------
    n_pixels : int
        Grid size (square: n × n pixels).
    pixel_size : float
        Physical pixel size [m].
    pitch : float
        Centre-to-centre period [m].
    cd : float
        Width of the bright (transmitting) line [m] — the feature that
        prints in a positive-tone resist is the *space* between these.
    orientation : str
        ``"vertical"`` – lines run vertically (vary in x);
        ``"horizontal"`` – lines run horizontally (vary in y).

    Returns
    -------
    NDArray[np.float64]
        Binary mask of shape ``(n_pixels, n_pixels)``.
    """
    coords = (np.arange(n_pixels) - n_pixels // 2) * pixel_size
    phase = np.mod(coords + pitch / 2, pitch)  # centre a line at origin
    line_1d = (phase < cd).astype(np.float64)

    if orientation == "vertical":
        mask = np.tile(line_1d[np.newaxis, :], (n_pixels, 1))
    elif orientation == "horizontal":
        mask = np.tile(line_1d[:, np.newaxis], (1, n_pixels))
    else:
        raise ValueError(f"orientation must be 'vertical' or 'horizontal', got '{orientation}'")

    logger.debug(
        "L/S mask: pitch=%.1f nm, CD=%.1f nm, orientation=%s",
        pitch * 1e9, cd * 1e9, orientation,
    )
    return mask


def contact_array(
    n_pixels: int,
    pixel_size: float,
    pitch_x: float,
    pitch_y: float,
    cd_x: float,
    cd_y: float | None = None,
) -> NDArray[np.float64]:
    """Generate a 2-D periodic contact-hole array (clear on opaque).

    Parameters
    ----------
    n_pixels, pixel_size : int, float
        Grid definition.
    pitch_x, pitch_y : float
        Periods in x and y [m].
    cd_x : float
        Contact opening width in x [m].
    cd_y : float, optional
        Contact opening height in y [m].  Defaults to *cd_x*.

    Returns
    -------
    NDArray[np.float64]
        Binary mask (1 = contact opening, 0 = opaque background).
    """
    if cd_y is None:
        cd_y = cd_x
    coords = (np.arange(n_pixels) - n_pixels // 2) * pixel_size
    px = np.mod(coords + pitch_x / 2, pitch_x)
    py = np.mod(coords + pitch_y / 2, pitch_y)
    open_x = (px < cd_x).astype(np.float64)
    open_y = (py < cd_y).astype(np.float64)
    mask = np.outer(open_y, open_x)
    logger.debug(
        "Contact array: pitch=(%.1f, %.1f) nm, CD=(%.1f, %.1f) nm",
        pitch_x * 1e9, pitch_y * 1e9, cd_x * 1e9, cd_y * 1e9,
    )
    return mask


def isolated_line(
    n_pixels: int,
    pixel_size: float,
    cd: float,
    orientation: str = "vertical",
) -> NDArray[np.float64]:
    """Generate a single isolated opaque line centred on the grid.

    Parameters
    ----------
    n_pixels, pixel_size : int, float
        Grid definition.
    cd : float
        Line width [m].
    orientation : str
        ``"vertical"`` or ``"horizontal"``.

    Returns
    -------
    NDArray[np.float64]
        Binary mask.
    """
    coords = (np.arange(n_pixels) - n_pixels // 2) * pixel_size
    line_1d = (np.abs(coords) < cd / 2.0).astype(np.float64)

    if orientation == "vertical":
        mask = np.tile(line_1d[np.newaxis, :], (n_pixels, 1))
    else:
        mask = np.tile(line_1d[:, np.newaxis], (1, n_pixels))

    logger.debug("Isolated line: CD=%.1f nm, orientation=%s", cd * 1e9, orientation)
    return mask


def apply_bias(
    mask: NDArray[np.float64],
    bias_nm: float,
    pixel_size: float,
) -> NDArray[np.float64]:
    """Apply a CD bias by morphological dilation (positive) or erosion (negative).

    Parameters
    ----------
    mask : NDArray
        Input binary mask.
    bias_nm : float
        Bias amount in nm.  Positive enlarges features; negative shrinks them.
    pixel_size : float
        Physical pixel size [m].

    Returns
    -------
    NDArray[np.float64]
        Biased binary mask.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    bias_px = int(round(abs(bias_nm) * 1e-9 / pixel_size))
    if bias_px == 0:
        return mask.copy()

    struct = np.ones((2 * bias_px + 1, 2 * bias_px + 1), dtype=bool)
    binary = mask.astype(bool)
    result = binary_dilation(binary, struct) if bias_nm > 0 else binary_erosion(binary, struct)
    logger.debug("Mask bias: %.1f nm (%d px)", bias_nm, bias_px)
    return result.astype(np.float64)


def to_attenuated_psm(
    mask: NDArray[np.float64],
    transmission: float = 0.06,
    phase_shift_deg: float = 180.0,
) -> NDArray[np.complex128]:
    """Convert a binary chrome mask to an attenuated phase-shifting mask.

    Chrome regions become partially transmissive with a phase shift of
    *phase_shift_deg* (typically 180° for att-PSM).

    Parameters
    ----------
    mask : NDArray
        Binary amplitude mask (1 = chrome).
    transmission : float
        Fractional amplitude transmission through chrome (e.g. 0.06 = 6 %).
    phase_shift_deg : float
        Phase applied to chrome regions [degrees].

    Returns
    -------
    NDArray[np.complex128]
        Complex-valued mask transmittance.
    """
    phase_rad = np.deg2rad(phase_shift_deg)
    t = np.where(
        mask > 0.5,
        np.sqrt(transmission) * np.exp(1j * phase_rad),
        1.0 + 0.0j,
    )
    return t.astype(np.complex128)


def checkerboard(
    n_pixels: int,
    pixel_size: float,
    pitch: float,
    cd: float,
) -> np.ndarray:
    """
    Generate an alternating contact hole pattern (SRAM-like checkerboard).

    Args:
        n_pixels:   Grid side length in pixels.
        pixel_size: Physical size of each pixel [m].
        pitch:      Cell pitch [m].
        cd:         Contact hole side length [m].

    Returns:
        2-D mask array with values in {0, 1}.
    """
    x = np.arange(n_pixels) * pixel_size
    y = np.arange(n_pixels) * pixel_size
    xx, yy = np.meshgrid(x, y)

    ix = np.floor(xx / pitch).astype(int)
    iy = np.floor(yy / pitch).astype(int)
    active_cell = (ix + iy) % 2 == 0

    mx = xx % pitch
    my = yy % pitch
    cx, cy = pitch / 2, pitch / 2
    in_hole = (np.abs(mx - cx) <= cd / 2) & (np.abs(my - cy) <= cd / 2)

    return np.where(active_cell & in_hole, 1.0, 0.0).astype(np.float64)
