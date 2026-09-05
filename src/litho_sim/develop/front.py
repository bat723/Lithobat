"""
Development as a moving front, not a column of independent rays.

The engine's Mack model marches straight down each column, integrating
``∫dz/R`` until the elapsed time reaches ``develop_time``. That is exact when
the front moves vertically, and it is structurally incapable of anything else:
a column cannot be undercut by its neighbour, so no dose, no time and no rate
law will ever produce an overhang, a T-top or a foot.

The physical statement is that the developer front moves **normal to itself**
at the local dissolution rate. Writing ``T(x)`` for the time the front arrives
at a point, that is the eikonal equation

.. math::

    |\\nabla T(x)| \\; R(x) = 1,

with ``T = 0`` on the resist's top surface. The profile after developing for
``t`` seconds is then the level set ``{T > t}`` — and because one solve gives
*every* develop time at once, a develop-time sweep costs nothing extra.

Not a Stefan problem
--------------------
Worth stating plainly, because the two get conflated. A Stefan problem has a
field that must be *solved* in the bulk (heat, or solute), a moving interface,
and an interface condition tying the boundary's velocity to that field's
**flux**. Development has the moving interface but not the other two: by the
time the developer arrives, the PAC field is frozen — exposure and bake have
already happened — so ``R`` is a prescribed function of position, not something
coupled to an unknown.

The Damköhler number says how safe that is. With ``R ≈ 100 nm/s`` over a 100 nm
feature, ``Da = R·L/D`` is ~1e-5 against bulk developer diffusion: transport
equilibrates five orders of magnitude faster than the interface moves, so the
field drops out exactly rather than by approximation.

The **one** place that fails is the unswollen glassy polymer right at the
surface, where ``D`` is ~1e-16 m²/s and ``Da`` is ~100 — genuinely transport
limited, and the origin of the induction period. That is what
:func:`~litho_sim.develop.resist.surface_inhibition` models empirically, the
same way the literature does, rather than solving a second moving-boundary
problem inside this one.

Where a coupled field genuinely belongs in this engine is the **bake**, where
acid diffuses while being consumed by quencher — see
:mod:`litho_sim.bake.peb`.

Accuracy
--------
First-order upwind Godunov, which is **exact for axis-aligned propagation**
(verified to 1e-14 on a planar front, and it reproduces the vertical ray march
bit-for-bit when the rate field has no lateral structure) and anisotropic on
the diagonals — about 4.6 % on a face diagonal and 7.4 % on a body diagonal at
typical resolution, converging under refinement. That error budget is arranged
the right way round for this problem: development travels mostly straight down
and only a little sideways, so the accurate direction is the one that carries
the profile and the lossy one only trims the undercut.

The scheme does assume the **rate field is resolved by the grid**. Against an
independent shortest-path reference it lands within 3 % on a smoothed field and
runs ~98 % slow on white noise, where the true optimal path zigzags below the
voxel scale. Lithography only ever produces the former — a latent image is
band-limited by the optics and smoothed again by the bake — but a rate field
built some other way should be checked before it is trusted.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["arrival_time", "arrival_time_front", "develop_front"]

#: Arrival times start here — large enough to act as "unreached", small enough
#: that squaring it will not overflow float64.
_UNREACHED = 1e30


def _godunov(
    neighbours: Sequence[NDArray[np.float64]],
    spacing: Sequence[float],
    slowness: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the local eikonal update at every point at once.

    For the upwind neighbours ``a_i`` along each axis, the Godunov scheme
    solves

        Σ_i max(T − a_i, 0)² / h_i² = slowness²

    The ``max(·, 0)`` is what makes it upwind, and it means the answer uses
    only the axes the front actually came from — which is not known until
    ``T`` is. The standard resolution is to sort the candidates and take them
    in order: try the nearest alone, and admit the next only if the solution
    has already run past it.

    Getting this wrong is subtle and expensive. Including a term the front did
    not come from can leave the quadratic with **no real root**, and treating
    that as "unreachable" instead of falling back to fewer terms strands the
    solver at a fixed point far above the true arrival time. It only shows up
    where the rate contrast is strong — a smooth rate field never triggers it,
    which makes it exactly the bug that survives smooth test cases.
    """
    d = len(neighbours)
    a = np.stack([np.asarray(x, dtype=np.float64) for x in neighbours])
    h = np.empty_like(a)
    for i, sp in enumerate(spacing):
        h[i] = sp

    # Ascending, so the front's nearest source is considered first.
    order = np.argsort(a, axis=0)
    a = np.take_along_axis(a, order, axis=0)
    w = 1.0 / np.take_along_axis(h, order, axis=0) ** 2

    T = np.full(slowness.shape, _UNREACHED)
    solved = np.zeros(slowness.shape, dtype=bool)
    A = np.zeros_like(slowness)
    B = np.zeros_like(slowness)
    C = -slowness * slowness

    for k in range(d):
        known = a[k] < _UNREACHED
        av = np.where(known, a[k], 0.0)
        A = A + np.where(known, w[k], 0.0)
        B = B - 2.0 * np.where(known, w[k] * av, 0.0)
        C = C + np.where(known, w[k] * av * av, 0.0)

        disc = B * B - 4.0 * A * C
        ok = known & (A > 0) & (disc >= 0)
        root = np.divide(
            -B + np.sqrt(np.where(ok, disc, 0.0)),
            2.0 * np.where(A > 0, A, 1.0),
            out=np.full(slowness.shape, _UNREACHED), where=ok,
        )
        # Admit this many sources only if the solution really does sit beyond
        # the ones included and short of the next one out.
        nxt = a[k + 1] if k + 1 < d else np.full_like(a[k], _UNREACHED)
        accept = ok & ~solved & (root >= a[k]) & (root <= nxt)
        T = np.where(accept, root, T)
        solved |= accept

    return T


def _sweep(
    T: NDArray[np.float64],
    axis: int,
    direction: int,
    spacing: Sequence[float],
    slowness: NDArray[np.float64],
    seed: NDArray[np.bool_],
) -> None:
    """One Gauss-Seidel sweep along *axis*, updating ``T`` in place.

    Sweeping rather than relaxing is the whole game. A Jacobi iteration moves
    information exactly one voxel per pass, so a 100-voxel-deep film needs at
    least 100 of them and the per-iteration change goes small long before the
    answer is right — which is a stall that looks exactly like convergence. A
    sweep marches the full length of the axis in one pass, carrying the front
    with it, and the standard result is that a handful of sweeps in alternating
    directions suffices.

    Sequential along the swept axis (so each slab sees its predecessor already
    updated) and vectorised across the other two, which keeps the Python loop
    to the length of one axis rather than the size of the volume.
    """
    n = T.shape[axis]
    others = [a for a in range(T.ndim) if a != axis]
    order = range(n) if direction > 0 else range(n - 1, -1, -1)

    for i in order:
        here_l: list[Any] = [slice(None)] * T.ndim
        here_l[axis] = i
        here = tuple(here_l)

        along = []
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                s_: list[Any] = list(here)
                s_[axis] = j
                along.append(T[tuple(s_)])
        a_axis = (
            np.minimum.reduce(along) if along
            else np.full(T[here].shape, _UNREACHED)
        )

        slab = T[here]
        neighbours = [a_axis]
        sp = [spacing[axis]]
        for a in others:
            in_slab = a - (1 if a > axis else 0)
            lo = np.roll(slab, 1, axis=in_slab)
            hi = np.roll(slab, -1, axis=in_slab)
            edge: list[Any] = [slice(None)] * slab.ndim
            edge[in_slab] = 0
            lo[tuple(edge)] = _UNREACHED
            edge[in_slab] = -1
            hi[tuple(edge)] = _UNREACHED
            neighbours.append(np.minimum(lo, hi))
            sp.append(spacing[a])

        updated = np.minimum(slab, _godunov(neighbours, sp, slowness[here]))
        T[here] = np.where(seed[here], 0.0, updated)


def arrival_time(
    rate: NDArray[np.float64],
    spacing: Sequence[float],
    seed: NDArray[np.bool_] | None = None,
    initial: NDArray[np.float64] | None = None,
    tol: float = 1e-6,
    max_rounds: int = 24,
) -> NDArray[np.float64]:
    """When the develop front reaches every voxel.

    Solves ``|∇T| · rate = 1`` by the fast sweeping method: repeated
    Gauss-Seidel sweeps in every axis direction, each of which propagates the
    front along one family of characteristics. Because the update is monotone
    and only ever lowers ``T``, starting from any upper bound converges down
    onto the viscosity solution.

    Parameters
    ----------
    rate : NDArray
        Dissolution rate at every voxel [nm/s].
    spacing : sequence of float
        Voxel size along each axis [nm]. Anisotropic grids are fine — the
        2 nm × 4 nm voxels this engine uses by default are the normal case.
    seed : NDArray[bool], optional
        Where the developer starts. Defaults to the top plane of axis 0, which
        is where it comes from here (``iz = 0`` is the substrate).
    initial : NDArray, optional
        Starting guess. **Must be an upper bound** on the true arrival time,
        or the relaxation converges to something that is not the answer. The
        vertical ray march qualifies: restricting the front to vertical paths
        can only make it slower than letting it choose freely.
    tol : float
        Stop once a whole round of sweeps moves nothing by more than this [s].
    max_rounds : int
        Safety net. A smooth rate field converges to machine precision in
        about six; a strongly contrasted one (a quenched surface layer, say)
        wants roughly twice that. Hitting the limit is logged, not raised — a partly-swept field is
        still an upper bound, so the profile errs toward under-developed
        rather than wrong.

    Returns
    -------
    NDArray
        Arrival time [s]. Unreachable voxels hold a very large finite number
        rather than ``inf``, so comparisons and arithmetic stay well behaved.
    """
    rate = np.asarray(rate, dtype=np.float64)
    if len(spacing) != rate.ndim:
        raise ValueError(
            f"spacing has {len(spacing)} entries for a {rate.ndim}-D rate field"
        )
    slowness = 1.0 / np.maximum(rate, 1e-12)

    if seed is None:
        top = np.zeros(rate.shape, dtype=bool)
        top[-1] = True                        # developer sits on the top surface
        seed = top

    T = np.full(rate.shape, _UNREACHED) if initial is None else np.array(
        initial, dtype=np.float64, copy=True
    )
    T[seed] = 0.0

    delta = np.inf
    for rnd in range(max_rounds):
        before = T.copy()
        for axis in range(rate.ndim):
            for direction in (1, -1):
                _sweep(T, axis, direction, spacing, slowness, seed)
        delta = float(np.max(np.abs(before - T)))
        if delta < tol:
            logger.debug("arrival_time: converged after %d rounds", rnd + 1)
            break
    else:
        logger.warning(
            "arrival_time: %d sweep rounds without converging (last change "
            "%.3g s); the field is still an upper bound.", max_rounds, delta,
        )
    return T


def arrival_time_front(
    rate: NDArray[np.float64],
    dz_nm: float,
    pixel_nm: float,
    ray_march_guess: bool = True,
) -> NDArray[np.float64]:
    """When the laterally mobile front finishes dissolving each voxel [s].

    The continuous field :func:`develop_front` thresholds: resist remains
    where this exceeds the develop time. Kept separate so a develop-time
    sweep, a sub-pixel CD at any depth, or a roughness measurement can read
    the crossing off the field instead of a binary volume.

    Parameters
    ----------
    rate : NDArray
        ``(nz, ny, nx)`` dissolution rate [nm/s], ``iz = 0`` at the
        substrate, already evaluated for the tone in hand — the rate the
        developer actually meets.
    dz_nm, pixel_nm : float
        Voxel height and lateral pitch [nm].
    ray_march_guess : bool
        Seed the relaxation with the vertical ray march. It is a valid upper
        bound and it is already nearly right wherever the front is vertical,
        which is most of the volume — worth roughly an order of magnitude in
        iterations. Turning it off is for tests that want the solver judged on
        its own.
    """
    rate = np.asarray(rate, dtype=np.float64)

    # A ghost plane above the film carries the boundary condition, so that
    # T counts the time to *dissolve* a voxel rather than to arrive at its
    # centre. Without it the front starts half a voxel too deep and every
    # profile sits one voxel low — and, more to the point, it would no longer
    # agree with the ray march in the case where the two must agree.
    rate_g = np.concatenate([rate, rate[-1:]], axis=0)
    seed = np.zeros(rate_g.shape, dtype=bool)
    seed[-1] = True

    guess = None
    if ray_march_guess:
        # Vertical-only travel time: a candidate path, therefore never faster
        # than the true front, therefore a legal upper bound.
        guess = np.concatenate([
            np.cumsum((dz_nm / np.maximum(rate, 1e-12))[::-1], axis=0)[::-1],
            np.zeros_like(rate[-1:]),
        ], axis=0)

    return arrival_time(
        rate_g, spacing=(dz_nm, pixel_nm, pixel_nm), seed=seed, initial=guess
    )[:-1]


def develop_front(
    rate: NDArray[np.float64],
    dz_nm: float,
    pixel_nm: float,
    develop_time: float,
    tone: str = "positive",
    ray_march_guess: bool = True,
) -> NDArray[np.bool_]:
    """Develop a 3-D rate field with a laterally mobile front.

    The drop-in replacement for the vertical ray march in
    :func:`~litho_sim.develop.resist3d.develop_3d`, differing in exactly one
    way that matters: the front may travel sideways, so resist can be undercut.

    Parameters
    ----------
    rate : NDArray
        ``(nz, ny, nx)`` dissolution rate [nm/s], ``iz = 0`` at the
        substrate — the rate the developer meets, so for a negative resist
        the caller evaluates the rate law on the *exposed* fraction (as
        :func:`~litho_sim.develop.resist3d.develop_3d` does) before calling.
    dz_nm, pixel_nm : float
        Voxel height and lateral pitch [nm].
    develop_time : float
        Seconds in the developer.
    tone : str
        Must be ``"positive"``. The rate field already encodes the tone;
        an earlier ``"negative"`` branch inverted the field around its own
        extremes, a data-dependent transformation that is not a rate law
        and disagreed with every other negative-tone path in the engine.
    ray_march_guess : bool
        Seed the relaxation with the vertical ray march — see
        :func:`arrival_time_front`.

    Returns
    -------
    NDArray[bool]
        True where resist **remains**, matching ``develop_3d``'s convention.
    """
    if tone != "positive":
        raise ValueError(
            "develop_front takes the rate the developer meets; evaluate the "
            "Mack law on the exposed fraction for a negative resist "
            "(develop_3d does this) rather than passing tone='negative'."
        )
    T = arrival_time_front(rate, dz_nm, pixel_nm, ray_march_guess=ray_march_guess)
    remaining = T > develop_time
    logger.debug(
        "develop_front: %.1f%% remains after %.1f s",
        100.0 * float(remaining.mean()), develop_time,
    )
    return remaining
