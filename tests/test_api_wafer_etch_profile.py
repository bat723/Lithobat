"""
Interface-level tests for etch-profile shapes, through ``litho_sim.wafer``.

``tests/test_stack.py`` already pins the closed forms of each profile term and
the sign of the applied behaviours. These tests exercise the *public package
interface* instead — ``from litho_sim.wafer import EtchProfile`` — and check
the applied geometry quantitatively: a taper angle must narrow the opening at
the rate ``cot(angle)`` per unit depth, a re-entrant wall must flare at the
same rate with the opposite sign, and a bias must widen uniformly. Plus the
error and degenerate-input contracts a consumer can hit.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig
from litho_sim.wafer import VACUUM, EtchProfile, Stack

PX = 4e-9
DZ = 4e-9
FILM = 80e-9          # oxide film thickness [m]
N_SUB = 6             # substrate voxels (24 nm at dz = 4 nm)
TRENCH_LO, TRENCH_HI = 12, 36   # open columns [lo, hi), 24 px = 96 nm wide


def _film_stack() -> tuple[Stack, np.ndarray]:
    """A small SiO2 film on Si, and an open-mask stripe for a trench etch.

    The film top sits at voxel ``N_SUB + FILM/DZ - 1``; the trench mask is a
    full-height stripe in y so the sidewall geometry is one-dimensional in x
    and widths can be read off a single row.
    """
    grid = GridConfig(n_pixels=48, pixel_size=PX)
    st = Stack.blank(grid, dz=DZ, substrate_thickness=N_SUB * DZ, headroom=120e-9)
    st.deposit_blanket("SiO2", FILM)
    mask = np.zeros(st.shape_xy, dtype=bool)
    mask[:, TRENCH_LO:TRENCH_HI] = True
    return st, mask


def _open_width(st: Stack, iz: int) -> float:
    """Width of the opening [m] at one z level, read along the centre row."""
    row = st.shape_xy[0] // 2
    return float(np.count_nonzero(st.mat[iz, row, :] == VACUUM)) * st.pixel_size


def _surf_iz(st_before_etch_film_top: Stack) -> int:
    """Index of the film's top voxel, where depth-below-surface is zero."""
    return N_SUB + int(round(FILM / DZ)) - 1


# ---------------------------------------------------------------------------
# The public interface itself
# ---------------------------------------------------------------------------


def test_etch_profile_is_exported_from_the_wafer_package():
    """Consumers build profiles and hand them to Stack.etch; both halves of
    that handshake must be reachable from ``litho_sim.wafer`` without touching
    a private module path."""
    import litho_sim.wafer as wafer

    assert "EtchProfile" in wafer.__all__
    p = wafer.EtchProfile(sidewall_deg=85.0)
    assert not p.is_identity
    # Same class the internal module defines — one type, not a re-export copy.
    from litho_sim.wafer.etch_profile import EtchProfile as _Internal

    assert wafer.EtchProfile is _Internal


def test_profile_is_immutable_and_defaults_to_identity():
    p = EtchProfile()
    assert p.is_identity
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.bias = 1e-9  # type: ignore[misc]
    # A foot height with no extent juts nowhere: still the identity.
    assert EtchProfile(footer_height=20e-9).is_identity
    assert not EtchProfile(sidewall_deg=89.5).is_identity


# ---------------------------------------------------------------------------
# Applied geometry: sign AND magnitude, end to end through Stack.etch
# ---------------------------------------------------------------------------


def test_taper_angle_sets_the_sidewall_slope():
    """A 75 degree wall pulls each side in by cot(75) per unit depth.

    test_stack.py checks the sign (narrower at the bottom); this checks the
    engine actually honours the *number* — the width lost between two depths
    must match 2 * (d2 - d1) / tan(75 deg) to within pixel quantisation.
    """
    st, mask = _film_stack()
    st.etch("SiO2", depth=FILM, open_mask=mask,
            profile=EtchProfile(sidewall_deg=75.0))

    top = _surf_iz(st)
    iz1, iz2 = top - 1, N_SUB + 1        # depths 4 nm and 72 nm below surface
    d1, d2 = (top - iz1) * DZ, (top - iz2) * DZ
    w1, w2 = _open_width(st, iz1), _open_width(st, iz2)

    assert w2 < w1, "a sub-90 wall must narrow with depth"
    expected_shrink = 2.0 * (d2 - d1) / math.tan(math.radians(75.0))
    assert w1 - w2 == pytest.approx(expected_shrink, abs=2.5 * PX), (
        f"sidewall slope is off: width went {w1*1e9:.0f} -> {w2*1e9:.0f} nm, "
        f"expected a shrink of {expected_shrink*1e9:.1f} nm"
    )
    # And monotonic in between — a taper, not merely two lucky samples.
    widths = [_open_width(st, iz) for iz in range(iz2, iz1 + 1)]
    assert all(a <= b + 1e-15 for a, b in zip(widths, widths[1:])), (
        "opening width must not increase with depth under a tapered wall"
    )


def test_reentrant_wall_flares_at_the_complementary_rate():
    """Above 90 degrees the hole widens with depth — the fill-pinch-off shape.

    Same magnitude law as the taper with the sign flipped: cot(105 deg) is
    negative, so the offset grows outward by |cot| per unit depth, undercutting
    the masked film while the top opening stays at the mask width.
    """
    st, mask = _film_stack()
    st.etch("SiO2", depth=FILM, open_mask=mask,
            profile=EtchProfile(sidewall_deg=105.0))

    top = _surf_iz(st)
    iz1, iz2 = top - 1, N_SUB + 1
    d1, d2 = (top - iz1) * DZ, (top - iz2) * DZ
    w1, w2 = _open_width(st, iz1), _open_width(st, iz2)

    assert w2 > w1, "a re-entrant wall must widen with depth"
    expected_flare = 2.0 * (d2 - d1) * abs(1.0 / math.tan(math.radians(105.0)))
    assert w2 - w1 == pytest.approx(expected_flare, abs=2.5 * PX), (
        f"re-entrant flare is off: width went {w1*1e9:.0f} -> {w2*1e9:.0f} nm, "
        f"expected a flare of {expected_flare*1e9:.1f} nm"
    )
    # Near the surface the opening is still the mask's: the flare grows down
    # and under, it does not blow out the top.
    mask_width = (TRENCH_HI - TRENCH_LO) * PX
    assert w1 == pytest.approx(mask_width, abs=1.5 * PX)


def test_bias_widens_the_opening_uniformly():
    """A positive bias undercuts the mask by the same amount at every depth.

    The bias is deliberately *not* an exact pixel multiple: a threshold that
    lands exactly on a voxel boundary flips on float epsilons, which is a
    property of the discretisation, not of the interface under test.
    """
    bias = 10e-9  # 2.5 pixels a side
    st, mask = _film_stack()
    plain = st.copy()
    plain.etch("SiO2", depth=FILM, open_mask=mask)
    biased = st.copy()
    biased.etch("SiO2", depth=FILM, open_mask=mask,
                profile=EtchProfile(bias=bias))

    top = _surf_iz(st)
    widenings = [
        _open_width(biased, iz) - _open_width(plain, iz)
        for iz in (top - 1, (top + N_SUB) // 2, N_SUB + 1)
    ]
    for w in widenings:
        assert w == pytest.approx(2.0 * bias, abs=1.5 * PX), (
            f"bias must widen by 2*bias; measured {w*1e9:.1f} nm"
        )
    assert max(widenings) - min(widenings) < 0.5 * PX, (
        f"a pure bias must be depth-independent, got {widenings}"
    )


# ---------------------------------------------------------------------------
# Error and degenerate-input contracts
# ---------------------------------------------------------------------------


def test_a_lateral_etch_refuses_a_shaped_profile():
    """exposure='any' has no single mask edge, so a profile is a contradiction
    the interface must reject rather than quietly ignore."""
    st, _ = _film_stack()
    with pytest.raises(ValueError, match="profile"):
        st.etch("SiO2", depth=20e-9, exposure="any",
                profile=EtchProfile(sidewall_deg=80.0))


def test_a_lateral_etch_accepts_the_identity_profile():
    """The default EtchProfile() must never invalidate a recipe — consumers
    (patterning EtchStep) rely on the identity being interchangeable with
    'no profile'."""
    st, _ = _film_stack()
    before = st.thickness_of("SiO2").max()
    st.etch("SiO2", depth=12e-9, exposure="any", profile=EtchProfile())
    assert st.thickness_of("SiO2").max() < before, (
        "the identity profile must not stop a lateral etch from etching"
    )


def test_extreme_angles_are_clamped_not_propagated():
    """0 and 180 degrees are walls with no vertical run — a naive cot() is
    infinite. The interface clamps to [1, 179] and stays finite."""
    d = np.linspace(0.0, FILM, 9)
    for deg in (0.0, -45.0, 180.0, 360.0):
        off = EtchProfile(sidewall_deg=deg).offsets(d, FILM)
        assert np.isfinite(off).all(), f"non-finite offsets at {deg} degrees"
    # The clamp lands exactly on the documented limits.
    lo = EtchProfile(sidewall_deg=0.0).offsets(d, FILM)
    assert np.allclose(lo, EtchProfile(sidewall_deg=1.0).offsets(d, FILM))
    hi = EtchProfile(sidewall_deg=180.0).offsets(d, FILM)
    assert np.allclose(hi, EtchProfile(sidewall_deg=179.0).offsets(d, FILM))
    # And a vertical wall contributes exactly nothing.
    assert np.allclose(EtchProfile(sidewall_deg=90.0).offsets(d, FILM), 0.0)


def test_degenerate_dimensions_stay_finite():
    """Zero total depth, empty input, and depths above the surface are all
    states a caller can reach; none may produce NaN or a crash."""
    full = EtchProfile(bias=5e-9, sidewall_deg=80.0, top_radius=10e-9,
                       bottom_radius=8e-9, footer_height=20e-9,
                       footer_extent=6e-9)
    # Empty in, empty out.
    assert full.offsets(np.array([]), FILM).shape == (0,)
    # A zero-depth etch: every bottom feature is measured from a floor at the
    # surface itself.
    off0 = full.offsets(np.linspace(0.0, 20e-9, 6), 0.0)
    assert np.isfinite(off0).all()
    # Negative depth (above the surface) is asked for by the shaping pass on
    # masked columns; it must be a number, not an error.
    assert np.isfinite(full.offsets(np.array([-10e-9, 0.0, 10e-9]), FILM)).all()


def test_describe_names_every_shape_term():
    """The one-line description feeds step listings in the app; each non-zero
    term must show up so a recipe reads back what it does."""
    text = EtchProfile(bias=5e-9, sidewall_deg=85.0, top_radius=10e-9,
                       bottom_radius=8e-9, footer_height=20e-9,
                       footer_extent=6e-9).describe()
    for needle in ("+5", "85", "r10", "r8", "foot"):
        assert needle in text, f"'{needle}' missing from: {text!r}"
    assert EtchProfile().describe() == ""
