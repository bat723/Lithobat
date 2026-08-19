"""
The develop front, checked against cases with known answers.

The eikonal solver is the kind of code that produces plausible pictures while
being wrong, so most of this file is analytic: a planar front, a layered stack
and a laterally-uniform field all have arithmetic answers, and the pathological
case is checked against an independent shortest-path computation.
"""

import numpy as np
import pytest

from litho_sim.develop.front import arrival_time, develop_front


def _ray_march(rate, dz_nm):
    """Vertical-only travel time — the model `front` must reduce to."""
    return np.cumsum((dz_nm / np.maximum(rate, 1e-12))[::-1], axis=0)[::-1]


def _blocks_per_column(remaining):
    """How many separate runs of resist each column holds. >1 means overhang."""
    d = np.diff(remaining.astype(np.int8), axis=0)
    return (d == 1).sum(axis=0) + remaining[0]


# ---------------------------------------------------------------------------
# Analytic cases
# ---------------------------------------------------------------------------


def test_planar_front_in_a_uniform_medium_is_exact():
    """T = depth / rate, to machine precision. Nothing lateral to get wrong."""
    rate = np.full((40, 16, 16), 50.0)
    T = arrival_time(rate, spacing=(2.0, 4.0, 4.0))
    depth_nm = (np.arange(40)[::-1]) * 2.0
    assert np.abs(T - (depth_nm / 50.0)[:, None, None]).max() < 1e-12


def test_a_layered_stack_adds_up():
    """Time through two layers is the sum of the two transit times."""
    rate = np.full((30, 8, 8), 10.0)
    rate[:15] = 100.0                       # fast layer at the bottom
    T = arrival_time(rate, spacing=(1.0, 1.0, 1.0))
    # 14 slow voxels below the pinned top plane, then 15 fast ones.
    assert T[0, 4, 4] == pytest.approx(14 * 0.1 + 15 * 0.01, abs=1e-9)


def test_reduces_to_the_ray_march_without_lateral_structure():
    """The strongest statement available: where the old model is right, the
    new one must agree with it *exactly*, not merely closely."""
    rng = np.random.default_rng(0)
    column = rng.uniform(5.0, 80.0, (30, 1, 1))
    rate = np.repeat(np.repeat(column, 6, axis=1), 6, axis=2)
    for t in (0.2, 0.5, 1.0):
        assert np.array_equal(
            develop_front(rate, 2.0, 4.0, t), _ray_march(rate, 2.0) > t
        )


def test_a_faster_lateral_route_is_taken():
    """A slow column beside a fast one is reached around the side."""
    rate = np.full((30, 3, 9), 0.5)
    rate[:, :, 4] = 0.5                     # slow chimney
    rate[:, :, :4] = 50.0                   # fast on both sides
    rate[:, :, 5:] = 50.0
    T = arrival_time(rate, spacing=(1.0, 1.0, 1.0))
    straight_down = 29 * 1.0 / 0.5
    assert T[0, 1, 4] < straight_down, "front ignored the cheap detour"


def test_the_solver_converges_rather_than_stalling():
    """A guessed start and a cold start must reach the same answer.

    This is the check that caught the real bug: a Godunov update that returns
    'unreachable' when its quadratic has no real root strands the relaxation
    at a fixed point far above the true arrival time. It converges, it reports
    success, and it is wrong — but only where the rate contrast is strong, so
    smooth test data never sees it.
    """
    rng = np.random.default_rng(1)
    rate = rng.uniform(0.5, 60.0, (24, 12, 12))
    guess = _ray_march(rate, 1.0)
    cold = arrival_time(rate, spacing=(1.0, 4.0, 4.0))
    warm = arrival_time(rate, spacing=(1.0, 4.0, 4.0), initial=guess)
    assert np.abs(cold - warm).max() < 1e-6


def test_no_result_exceeds_an_independent_shortest_path():
    """Checked against Dijkstra, which cannot be subtly wrong.

    An 8-connected graph only permits travel along its own edge directions, so
    its answer is an upper bound on a front free to move at any angle. The
    solver may come in under it; it must never come in over it by more than
    the first-order scheme's known diagonal bias.
    """
    sparse = pytest.importorskip("scipy.sparse")
    from scipy.ndimage import gaussian_filter
    from scipy.sparse.csgraph import dijkstra

    rng = np.random.default_rng(2)
    # Smoothed on purpose. A real dissolution-rate field is band-limited by
    # the optics and smoothed again by the bake, so it is resolved by the
    # grid. White noise per voxel is not, and a first-order scheme is ~98 %
    # slow on it because the true optimal path zigzags below the voxel scale
    # — a real limit of the discretisation, and not one lithography hits.
    rate = np.clip(gaussian_filter(rng.uniform(0.5, 40.0, (20, 24)), 2.0), 0.5, None)
    nz, nx = rate.shape
    hz = hx = 1.0
    idx = np.arange(nz * nx).reshape(nz, nx)
    slow = 1.0 / rate

    rows, cols, w = [], [], []
    for dk, di in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        a0 = slice(max(dk, 0), nz + min(dk, 0))
        a1 = slice(max(di, 0), nx + min(di, 0))
        b0 = slice(max(-dk, 0), nz + min(-dk, 0))
        b1 = slice(max(-di, 0), nx + min(-di, 0))
        rows.append(idx[a0, a1].ravel())
        cols.append(idx[b0, b1].ravel())
        w.append(np.hypot(dk * hz, di * hx)
                 * 0.5 * (slow[a0, a1].ravel() + slow[b0, b1].ravel()))
    G = sparse.coo_matrix(
        (np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nz * nx, nz * nx),
    ).tocsr()
    reference = dijkstra(G, indices=idx[nz - 1, :], min_only=True).reshape(nz, nx)

    seed = np.zeros((nz, nx), dtype=bool)
    seed[-1] = True
    T = arrival_time(rate, spacing=(hz, hx), seed=seed)

    finite = np.isfinite(reference) & (reference > 0)
    over = (T - reference)[finite] / reference[finite]
    assert over.max() < 0.05, f"solver runs {over.max():.1%} slow against Dijkstra"


# ---------------------------------------------------------------------------
# What the front buys that the ray march cannot
# ---------------------------------------------------------------------------


def test_a_ray_march_can_never_undercut():
    """Not a limitation of tuning — a structural one, worth pinning.

    Each column integrates its own ∫dz/R, so resist below a cleared voxel is
    unreachable by construction. No dose, time or rate law changes that.
    """
    rate = np.full((30, 4, 16), 40.0)
    rate[-4:, :, :] = 0.05                  # heavily inhibited cap
    rate[:, :, 7:9] = 60.0                  # a fast channel through it
    rate[-4:, :, 7:9] = 60.0
    for t in (0.5, 1.0, 2.0, 5.0):
        assert (_blocks_per_column(_ray_march(rate, 1.0) > t) > 1).sum() == 0


def test_the_front_undercuts_a_slow_cap():
    """Punch through the fast channel, then eat sideways underneath."""
    rate = np.full((30, 4, 16), 40.0)
    rate[-4:, :, :] = 0.05
    rate[:, :, 7:9] = 60.0
    rate[-4:, :, 7:9] = 60.0
    overhangs = [
        int((_blocks_per_column(develop_front(rate, 1.0, 1.0, t)) > 1).sum())
        for t in (0.5, 1.0, 2.0, 5.0)
    ]
    assert max(overhangs) > 0, (
        "the front never undercut the cap, so it is behaving like a ray march"
    )


def test_one_solve_serves_every_develop_time():
    """The level-set formulation gives a develop-time sweep for free, and the
    profiles it yields must nest — more time can only remove resist."""
    rng = np.random.default_rng(3)
    rate = rng.uniform(1.0, 50.0, (20, 8, 8))
    times = [0.2, 0.5, 1.0, 2.0]
    profiles = [develop_front(rate, 1.0, 4.0, t) for t in times]
    for earlier, later in zip(profiles, profiles[1:]):
        assert np.all(later <= earlier), "developing longer put resist back"


def test_spacing_must_match_the_field():
    with pytest.raises(ValueError, match="spacing has"):
        arrival_time(np.ones((4, 4, 4)), spacing=(1.0, 1.0))
