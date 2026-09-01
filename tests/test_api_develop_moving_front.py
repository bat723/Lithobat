"""
Interface-level tests for the moving-front develop model.

Everything here goes through the public package surface, ``litho_sim.develop``
— primarily :func:`develop_3d(model="front")`, because that is the exact call
the application makes (``app/compute.py`` passes ``params.develop_model``
straight through, and ``patterning/steps.py``'s Develop step does the same).
The direct names ``develop_front`` and ``arrival_time`` are also imported from
the package, not the module, to pin the re-export.

The unit-level physics (Godunov exactness, Dijkstra bounds, convergence) lives
in ``tests/test_front.py`` and is deliberately not repeated. What belongs
here is the contract a consumer sees:

* an aerial-like latent image develops into a physically sane profile —
  resist standing where dark, cleared where bright;
* with surface inhibition, the front produces a T-top — resist wider at the
  top plane than at mid-depth — which the vertical ray march is structurally
  incapable of;
* the documented error and edge cases at the ``develop_3d`` boundary.

Grids are kept small (48 px lateral, 24 z-voxels) so the eikonal solves stay
fast.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, ResistConfig
from litho_sim.develop import arrival_time, develop_3d, develop_front

# ---------------------------------------------------------------------------
# A small, shared scene: one 64 nm dark line in bright surroundings
# ---------------------------------------------------------------------------

NZ, NY, NX = 24, 6, 48          # 96 nm film, 4 nm voxels, 192 nm field
LINE_LO, LINE_HI = 16, 32       # drawn line: 16 px = 64 nm


def _grid() -> GridConfig:
    return GridConfig(n_pixels=NX, pixel_size=4e-9, dz=4e-9)


def _latent_line() -> np.ndarray:
    """A PAC field the way an aerial image leaves it.

    Positive tone: bright areas (the spaces) are exposed, so their PAC is
    low; the dark line keeps its PAC high. The edge is smoothed over a few
    pixels, which is what a band-limited image plus a bake always produces —
    and is the regime the front solver documents itself as valid for.
    """
    x = np.arange(NX, dtype=np.float64)
    d = np.abs(x - 0.5 * (LINE_LO + LINE_HI - 1))       # px from line centre
    half_width = 0.5 * (LINE_HI - LINE_LO)
    pac_x = 0.02 + 0.94 / (1.0 + np.exp(-(half_width - d) / 1.5))
    return np.broadcast_to(pac_x, (NZ, NY, NX)).copy()


def _widths(remaining: np.ndarray) -> np.ndarray:
    """Standing columns per z-plane, averaged over y."""
    return remaining.mean(axis=1).sum(axis=1)


# ---------------------------------------------------------------------------
# Happy path: the app's own call
# ---------------------------------------------------------------------------


def test_front_model_prints_a_line_where_the_mask_is_dark():
    """develop_3d(model="front") — the exact call app/compute.py makes."""
    resist = ResistConfig()                      # positive, develop_time=5 s
    remaining = develop_3d(_latent_line(), resist, _grid(), model="front")

    assert remaining.shape == (NZ, NY, NX)
    assert remaining.dtype == np.bool_

    # Resist stands where the mask was dark...
    core = slice(LINE_LO + 4, LINE_HI - 4)
    assert remaining[:, :, core].all(), "line core did not survive develop"
    # ...and is gone where it was bright, all the way to the substrate.
    assert not remaining[:, :, :8].any(), "left space failed to clear"
    assert not remaining[:, :, -8:].any(), "right space failed to clear"

    # The printed CD at mid-height is near the drawn 16 px, not collapsed
    # and not smeared across the field.
    mid_width = float(remaining[NZ // 2].mean(axis=0).sum())
    assert 10.0 <= mid_width <= 22.0, f"mid-height CD {mid_width:.1f} px"


def test_longer_develop_only_removes_resist():
    """Profiles at increasing develop_time must nest, seen from develop_3d."""
    latent, grid = _latent_line(), _grid()
    short = develop_3d(latent, replace(ResistConfig(), develop_time=2.0),
                       grid, model="front")
    long = develop_3d(latent, replace(ResistConfig(), develop_time=8.0),
                      grid, model="front")
    assert np.all(long <= short), "developing longer put resist back"
    assert long.sum() < short.sum(), "extra develop time removed nothing"


def test_negative_tone_inverts_which_side_survives():
    """The tone plumbing through develop_3d reaches the front model."""
    latent, grid = _latent_line(), _grid()
    pos = develop_3d(latent, ResistConfig(tone="positive"), grid, model="front")
    neg = develop_3d(latent, ResistConfig(tone="negative"), grid, model="front")
    # Negative tone: the exposed (bright, low-PAC) regions are what remains.
    assert neg[:, :, :8].all(), "negative tone lost its exposed spaces"
    assert not neg[:, :, LINE_LO + 4:LINE_HI - 4].any(), (
        "negative tone kept the unexposed line"
    )
    # And the two tones disagree where it matters.
    assert not np.array_equal(pos, neg)


# ---------------------------------------------------------------------------
# The signature behaviour: surface inhibition + lateral front = T-top
# ---------------------------------------------------------------------------


def test_surface_inhibition_gives_the_front_a_t_top_the_ray_march_cannot():
    """The model's reason to exist, observed through the public interface.

    With a strongly inhibited surface layer, the front punches down through
    the bright spaces, then eats sideways *under* the slow cap at the line
    edge faster than it eats through the cap itself. The result is resist
    wider at the very top plane than at mid-depth — an overhang.

    The Mack ray march is structurally incapable of this: each column is a
    single bottom-anchored block, so wherever its top voxel stands the whole
    column stands, and the standing width can only grow (never shrink) with
    depth. Both halves are asserted, on the same latent image and the same
    resist chemistry.
    """
    latent, grid = _latent_line(), _grid()
    resist = replace(
        ResistConfig(),
        inhibition_depth=20.0,      # nm
        inhibition_rate=0.02,       # heavily quenched surface
        develop_time=5.0,
    )

    front = develop_3d(latent, resist, grid, model="front")
    mack = develop_3d(latent, resist, grid, model="mack")

    w_front = _widths(front)
    w_mack = _widths(mack)
    top, mid = NZ - 1, NZ // 2

    # The front's overhang: strictly wider at the top plane than at
    # mid-depth, i.e. cleared space exists directly below standing resist.
    assert w_front[top] > w_front[mid], (
        f"no T-top: top width {w_front[top]:.1f} px vs mid {w_front[mid]:.1f}"
    )
    overhung = front[top] & ~front[mid]
    assert overhung.any(), "no column is overhung by the inhibited cap"

    # The ray march on identical inputs: monotone with depth, no overhang.
    assert np.all(w_mack[:-1] >= w_mack[1:] - 1e-9), (
        "mack standing width grew toward the top, which its column "
        "integral cannot do"
    )
    assert not (mack[top] & ~mack[mid]).any(), (
        "the vertical ray march produced an overhang"
    )

    # And the spaces did clear under the front, so the T-top is not just
    # an under-developed film.
    assert not front[:NZ - 6, :, :6].any(), "space never cleared under front"


# ---------------------------------------------------------------------------
# Edge and error cases at the public boundary
# ---------------------------------------------------------------------------


def test_uniformly_dark_field_leaves_the_film_intact():
    """A non-printing exposure develops nothing at all."""
    dark = np.full((NZ, NY, NX), 0.97)
    remaining = develop_3d(dark, ResistConfig(), _grid(), model="front")
    assert remaining.all(), "unexposed film lost resist"


def test_zero_develop_time_removes_nothing():
    resist = replace(ResistConfig(), develop_time=0.0)
    remaining = develop_3d(_latent_line(), resist, _grid(), model="front")
    assert remaining.all(), "zero seconds of developer removed resist"


def test_front_model_requires_a_grid():
    with pytest.raises(ValueError, match="requires a GridConfig"):
        develop_3d(_latent_line(), ResistConfig(), grid=None, model="front")


def test_unknown_model_is_rejected_by_name():
    with pytest.raises(ValueError, match="Unknown develop model"):
        develop_3d(_latent_line(), ResistConfig(), _grid(), model="sideways")


# ---------------------------------------------------------------------------
# Packaging: the names are on the package, and they are the real ones
# ---------------------------------------------------------------------------


def test_front_api_is_exported_from_the_develop_package():
    """The package docstring says "import from this package, not the modules";
    these two names must therefore live on it, and be the same objects the
    module defines (not shadows)."""
    import litho_sim.develop as develop_pkg
    from litho_sim.develop import front as front_mod

    assert develop_pkg.develop_front is front_mod.develop_front
    assert develop_pkg.arrival_time is front_mod.arrival_time
    assert "develop_front" in develop_pkg.__all__
    assert "arrival_time" in develop_pkg.__all__

    # And the package-level name behaves: a trivial uniform solve.
    rate = np.full((6, 4, 4), 50.0)
    remaining = develop_front(rate, dz_nm=4.0, pixel_nm=4.0, develop_time=0.1)
    assert remaining.shape == rate.shape
    T = arrival_time(rate, spacing=(4.0, 4.0, 4.0))
    assert T.shape == rate.shape
