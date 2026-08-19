"""
The shape of an etched sidewall.

A real etch does not cut a rectangle. The opening is biased away from the mask
edge, the wall is tapered, both corners are rounded, and a foot of unetched
material is often left at the base. Those are five separate causes — mask
erosion, ion angular spread, redeposition, incomplete clearing — and describing
them mechanistically would mean five models.

Instead this describes the *result*: one lateral offset per depth,

    offset(d) > 0   the opening is wider  than the mask edge at that depth
    offset(d) < 0   the opening is narrower

which is enough to place the etch front anywhere a real profile goes. It is a
shape model, honestly, and calibrated rather than derived — the same bargain
:func:`~litho_sim.develop.resist.surface_inhibition` makes for the induction
period.

Applying it costs one distance transform. Given the signed distance from the
mask opening, a voxel at depth ``d`` is etched where ``signed_distance <=
offset(d)`` — so the whole profile is a per-plane threshold on a field computed
once, rather than a per-voxel geometric test.

Sign conventions, which are the easy thing to get wrong
-------------------------------------------------------
``sidewall_deg`` follows the rest of the codebase (see
:func:`~litho_sim.develop.resist3d.sidewall_angle`): it is the angle of the
**remaining material's** wall from horizontal, so **90° is vertical** and less
than 90° slopes outward at the base. For an etched hole that reads as tapered —
narrower at the bottom — which is the normal result and the opposite of how the
same number reads on a resist line. Worth stating, because the two describe
complementary solids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["EtchProfile"]


@dataclass(frozen=True)
class EtchProfile:
    """Lateral offset of the etch front as a function of depth.

    Every length is in metres. All zero (and ``sidewall_deg = 90``) is a
    perfectly vertical cut exactly on the mask edge, which is what the engine
    did before this existed.

    Parameters
    ----------
    bias : float
        Offset at the top surface. Positive widens the opening — the etch
        undercuts the mask; negative pulls it in.
    sidewall_deg : float
        Wall angle from horizontal, 90° vertical. Below 90° the hole narrows
        with depth; above 90° it flares, which is re-entrant and the shape that
        makes a subsequent fill pinch off.
    top_radius : float
        Radius of the rounded corner where the wall meets the top surface. The
        arc is tangent to both, so the opening flares by ``top_radius`` at the
        very top and rejoins the wall smoothly.
    bottom_radius : float
        The same fillet where the wall meets the floor, narrowing the opening
        as it approaches the base.
    footer_height : float
        Height of the foot left at the base.
    footer_extent : float
        How far that foot juts into the opening, at the floor. Tapers linearly
        to nothing at ``footer_height``.
    """

    bias: float = 0.0
    sidewall_deg: float = 90.0
    top_radius: float = 0.0
    bottom_radius: float = 0.0
    footer_height: float = 0.0
    footer_extent: float = 0.0

    @property
    def is_identity(self) -> bool:
        """True when this profile does nothing, so callers can skip the work."""
        return (
            self.bias == 0.0
            and self.sidewall_deg == 90.0
            and self.top_radius == 0.0
            and self.bottom_radius == 0.0
            and self.footer_extent == 0.0
        )

    def offsets(
        self, depth_below_top: NDArray[np.float64], total_depth: float
    ) -> NDArray[np.float64]:
        """Lateral offset [m] at each depth below the surface the etch started from.

        Parameters
        ----------
        depth_below_top : NDArray
            Distances below the starting surface [m], any shape.
        total_depth : float
            Full etch depth [m], which is what the bottom features are measured
            up from. Pass the depth actually reached, not the budget, or the
            foot lands somewhere there is no floor.
        """
        d = np.asarray(depth_below_top, dtype=np.float64)
        offset = np.full(d.shape, float(self.bias))

        # Taper. cot(90°) is 0, so a vertical wall contributes nothing and the
        # common case costs one multiply by zero rather than a special case.
        angle = math.radians(float(np.clip(self.sidewall_deg, 1.0, 179.0)))
        offset -= d / math.tan(angle)

        # Rounded top corner: an arc tangent to the surface and to the wall, so
        # the opening is `top_radius` wider at d = 0 and rejoins at d = r.
        if self.top_radius > 0.0:
            r = self.top_radius
            inside = d < r
            arc = r - np.sqrt(np.maximum(r * r - (r - d) ** 2, 0.0))
            offset += np.where(inside, arc, 0.0)

        above_floor = total_depth - d

        # Rounded bottom corner: the same fillet, narrowing toward the floor.
        if self.bottom_radius > 0.0:
            r = self.bottom_radius
            inside = (above_floor >= 0.0) & (above_floor < r)
            arc = r - np.sqrt(np.maximum(r * r - (r - above_floor) ** 2, 0.0))
            offset -= np.where(inside, arc, 0.0)

        # The foot: a wedge at the base, full extent at the floor and gone by
        # footer_height.
        if self.footer_height > 0.0 and self.footer_extent != 0.0:
            frac = np.clip(1.0 - above_floor / self.footer_height, 0.0, 1.0)
            offset -= np.where(above_floor >= 0.0, self.footer_extent * frac, 0.0)

        return offset

    def describe(self) -> str:
        """One line, for a step's description. Empty when it does nothing."""
        if self.is_identity:
            return ""
        bits = []
        if self.bias:
            bits.append(f"bias {self.bias*1e9:+.0f} nm")
        if self.sidewall_deg != 90.0:
            bits.append(f"{self.sidewall_deg:.0f}° wall")
        if self.top_radius:
            bits.append(f"top r{self.top_radius*1e9:.0f}")
        if self.bottom_radius:
            bits.append(f"bot r{self.bottom_radius*1e9:.0f}")
        if self.footer_extent:
            bits.append(
                f"foot {self.footer_extent*1e9:.0f}×{self.footer_height*1e9:.0f} nm"
            )
        return ", ".join(bits)
