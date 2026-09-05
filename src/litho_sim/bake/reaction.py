"""
Post-exposure bake as reaction–diffusion, not as a blur.

``apply_peb`` convolves the latent image with a Gaussian. That is the exact
Green's function of the linear diffusion equation, and for a *chemically
amplified* resist it is the wrong equation — not because diffusion is wrong,
but because acid is not conserved while it diffuses:

* **Quencher.** Formulated base neutralises acid on contact. It is what sets
  the effective threshold and much of the resist's contrast, and a blur has no
  representation of it at all.
* **Loss.** Acid evaporates from the surface and is consumed by side
  reactions, so total acid falls during the bake.
* **Non-linearity.** Deprotection consumes acid catalytically, so the reaction
  rate depends on the local concentration of *both* species. A convolution is
  linear by construction and cannot produce that.

The consequence is not subtle. A blur is symmetric in dose: brightening and
dimming by the same factor move the latent image by the same amount either
way. With a quencher present, acid below the neutralisation level is
annihilated while acid above it survives, which is a threshold — and that
asymmetry is where much of a real resist's contrast comes from.

The system
----------
Writing ``H`` for acid, ``Q`` for quencher and ``M`` for protected sites:

.. math::

    \\partial_t H &= D_H \\nabla^2 H - k_q H Q - k_{loss} H \\\\
    \\partial_t Q &= D_Q \\nabla^2 Q - k_q H Q \\\\
    \\partial_t M &= -k_{amp} H M

Integrated by operator splitting: diffusion explicitly, with the step size
chosen from its stability limit, and the neutralisation **exactly** — second-
order kinetics has a closed form over a step, so the fast quench never sets
the step size. Before 2026-09-05 the quench was explicit too, and with
``k_quench = 20`` it forced 3,000 steps where diffusion needed 70; in 3-D that
was the difference between a bake in seconds and one in minutes. This is
genuinely a coupled two-species field with a non-linear sink — the thing the
develop step turned out *not* to be. Development has a moving
boundary whose speed is prescribed by a frozen field (see
:mod:`litho_sim.develop.front`); the bake has no moving boundary at all but
does have fields that must be solved. The two are easy to mix up and they want
opposite machinery.

Reducing to the old behaviour
-----------------------------
With ``quencher = 0``, ``k_loss = 0`` and ``k_amp = 0`` this is pure linear
diffusion, and :func:`bake_reaction_diffusion` reproduces the Gaussian blur to
the accuracy of the time stepping — asserted in the test suite, so the new path
is a strict generalisation rather than a different model wearing the same name.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["bake_reaction_diffusion", "diffusion_length"]

#: Explicit diffusion is stable while D·dt/h² stays under 1/(2·ndim). Backing
#: off to 40 % of the limit costs a few extra steps and removes any argument
#: about marginal stability.
_SAFETY = 0.4


def _quench_exact(
    H: NDArray[np.float64], Q: NDArray[np.float64], kt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Acid and quencher after neutralising for ``k·t``, exactly.

    ``dH/dt = dQ/dt = −k H Q`` conserves ``d = H − Q``. With ``d ≠ 0``::

        H(t) = d · H₀ / (H₀ − Q₀ · exp(−k d t)),   Q(t) = H(t) − d

    and with equal concentrations ``H(t) = H₀ / (1 + k H₀ t)``. Both are
    monotone and stay non-negative for any step, which is what lets the
    step size ignore the quench rate entirely. Where ``|k d t|`` is tiny
    the two forms agree to rounding and the equal-concentration one is used
    to avoid a 0/0.
    """
    d = H - Q
    x = np.clip(kt * d, -700.0, 700.0)
    equal = np.abs(x) < 1e-9
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denom = H - Q * np.exp(-x)
        unequal = np.where(np.abs(denom) > 0.0, d * H / denom, 0.0)
        same = H / (1.0 + kt * H)
    H_new = np.where(equal, same, unequal)
    H_new = np.clip(H_new, 0.0, np.maximum(H, Q))
    Q_new = np.maximum(H_new - d, 0.0)
    return H_new, Q_new


def diffusion_length(D: float, time: float) -> float:
    """The ``σ = sqrt(2 D t)`` a diffusion coefficient and bake time imply [m].

    The bridge between the two ways of parameterising a bake: this module
    wants ``D`` and ``t``, while :func:`~litho_sim.bake.peb.apply_peb` and the
    app's controls speak in diffusion length. Same physics, and this converts.
    """
    return float(np.sqrt(2.0 * D * time))


def bake_reaction_diffusion(
    acid: NDArray[np.float64],
    pixel_size: float,
    bake_time: float,
    D_acid: float,
    quencher: float | NDArray[np.float64] = 0.0,
    D_quencher: float = 0.0,
    k_quench: float = 0.0,
    k_loss: float = 0.0,
    k_amp: float = 0.0,
    protected: float = 1.0,
    spacing: tuple[float, ...] = (),
    max_steps: int = 20000,
) -> dict:
    """Bake an acid distribution, with quenching, loss and deprotection.

    Parameters
    ----------
    acid : NDArray
        Acid concentration generated by exposure. Any dimensionality — 2-D
        latent images and 3-D volumes both work.
    pixel_size : float
        Lateral voxel size [m]. Used for every axis unless *spacing* overrides.
    bake_time : float
        Bake duration [s].
    D_acid, D_quencher : float
        Diffusivities [m²/s]. Quencher is usually far less mobile than acid;
        leaving ``D_quencher`` at 0 pins it where it was formulated.
    quencher : float or NDArray
        Initial quencher loading, in the same units as *acid*. A scalar means
        uniform — the deterministic case, where base is formulated in evenly.
        An array is a per-voxel loading, which is what the stochastic path
        supplies: sampled molecule counts are anything but uniform.
    k_quench : float
        Acid–base neutralisation rate. This is the term that turns a blur into
        a threshold.
    k_loss : float
        First-order acid loss (evaporation, side reactions) [1/s].
    k_amp : float
        Deprotection rate constant. With ``k_amp = 0`` the protected fraction
        is left untouched and only the acid field evolves.
    protected : float or NDArray
        Initial protected fraction — scalar for uniform, array for per-voxel.
    spacing : tuple of float, optional
        Per-axis voxel size [m], when the grid is anisotropic — which it is in
        3-D here, where ``dz`` is not ``pixel_size``.
    max_steps : int
        Guard against a pathological time step. When only reaction *accuracy*
        is at stake the cap is applied and logged; when the diffusion
        *stability* limit cannot be honoured within the budget the call
        raises instead, because integrating anyway diverges to NaN.

    Returns
    -------
    dict
        ``acid``, ``quencher``, ``protected`` and ``steps``. The latent image
        the develop step wants is ``protected`` when ``k_amp > 0``, and the
        normalised ``acid`` otherwise.
    """
    acid = np.asarray(acid, dtype=np.float64)
    h = np.asarray(spacing if spacing else (pixel_size,) * acid.ndim, dtype=float)
    if h.size != acid.ndim:
        raise ValueError(f"spacing has {h.size} entries for a {acid.ndim}-D field")
    if np.any(h <= 0.0):
        raise ValueError(f"voxel sizes must be positive, got {tuple(h)}")
    if bake_time < 0.0:
        raise ValueError(f"bake_time must be non-negative, got {bake_time}")

    H = acid.copy()
    # Scalars broadcast to uniform fields; arrays pass through as per-voxel
    # loadings. Copy so the integration never writes into caller memory.
    Q = np.broadcast_to(np.asarray(quencher, dtype=np.float64), H.shape).copy()
    M = np.broadcast_to(np.asarray(protected, dtype=np.float64), H.shape).copy()

    if bake_time == 0.0:
        # The t → 0 limit is the identity, and the step-size arithmetic below
        # cannot express it (0/0) — return the fields untouched.
        return {"acid": H, "quencher": Q, "protected": M, "steps": 0}

    D_max = max(D_acid, D_quencher)
    h_min = float(h.min())
    if D_max > 0:
        dt_diff = _SAFETY * h_min * h_min / (2.0 * acid.ndim * D_max)
    else:
        dt_diff = bake_time
    # Neutralisation and loss are integrated exactly within a step, so only
    # the deprotection's accuracy (acid changing during a step) bounds dt
    # beyond diffusion — and with k_amp × H of order 0.05 /s it never binds.
    rate_scale = max(k_amp * float(H.max()), 1e-30)
    dt = min(dt_diff, _SAFETY / rate_scale, bake_time)
    n_steps = max(int(np.ceil(bake_time / dt)), 1)
    if n_steps > max_steps:
        # The two dt constraints fail differently when capped. Over-stepping
        # the reaction terms is merely inaccurate — the concentration floors
        # and the exponential keep everything finite. Over-stepping the
        # diffusion limit is *divergent*: the explicit stencil amplifies
        # grid-scale modes and the field goes NaN, which is not a result any
        # caller can use.
        if bake_time / max_steps > dt_diff:
            n_diff = int(np.ceil(bake_time / dt_diff))
            raise ValueError(
                f"bake needs {n_diff} steps to stay inside the diffusion "
                f"stability limit but max_steps={max_steps}; integrating "
                "anyway would diverge to NaN. Reduce bake_time or the "
                "diffusivities, or raise max_steps."
            )
        logger.warning(
            "PEB reaction-diffusion wanted %d steps; capping at %d. The bake "
            "will be under-resolved in time — reduce bake_time, D or the rate "
            "constants if the result matters.", n_steps, max_steps,
        )
        n_steps = max_steps
    dt = bake_time / n_steps

    # scipy's `laplace` is the unit-spacing stencil, so each axis carries its
    # own 1/h² and an anisotropic grid stays correct.
    inv_h2 = 1.0 / (h * h)

    def _lap(field):
        out = np.zeros_like(field)
        for axis in range(field.ndim):
            w = np.zeros(field.ndim, dtype=int)
            w[axis] = 1
            shifted = (
                np.roll(field, 1, axis=axis)
                + np.roll(field, -1, axis=axis)
                - 2.0 * field
            )
            # Zero-flux (Neumann) edges: nothing leaves the simulated film,
            # which is what a bake on a wafer actually does.
            lo = [slice(None)] * field.ndim
            hi = [slice(None)] * field.ndim
            first = [slice(None)] * field.ndim
            last = [slice(None)] * field.ndim
            lo[axis], hi[axis] = 0, -1
            first[axis], last[axis] = 1, -2
            shifted[tuple(lo)] = field[tuple(first)] - field[tuple(lo)]
            shifted[tuple(hi)] = field[tuple(last)] - field[tuple(hi)]
            out += shifted * inv_h2[axis]
        return out

    # Acid loss is linear and decoupled, so it integrates exactly. Doing it
    # explicitly instead would make the answer depend on a step size chosen
    # for the *diffusion* stability limit — three steps of forward Euler turn
    # exp(-1.2) = 0.301 into 0.6^3 = 0.216, a 28 % error in a term with a
    # closed form sitting right there.
    loss_factor = np.exp(-k_loss * dt) if k_loss > 0.0 else 1.0

    for _ in range(n_steps):
        # Diffusion, explicit: this is what the step size was chosen for.
        if D_acid > 0.0:
            H = np.maximum(H + dt * D_acid * _lap(H), 0.0)
        if D_quencher > 0.0:
            Q = np.maximum(Q + dt * D_quencher * _lap(Q), 0.0)
        # Neutralisation, exact over the step: H + Q → nothing is second-
        # order kinetics with H − Q conserved, and that has a closed form.
        if k_quench > 0.0:
            H, Q = _quench_exact(H, Q, k_quench * dt)
        if k_loss > 0.0:
            H *= loss_factor
        if k_amp > 0.0:
            M = np.maximum(M * np.exp(-k_amp * H * dt), 0.0)

    logger.debug(
        "PEB reaction-diffusion: %d steps of %.3g s, acid %.3g -> %.3g, "
        "quencher left %.3g",
        n_steps, dt, float(acid.max()), float(H.max()), float(Q.mean()),
    )
    return {"acid": H, "quencher": Q, "protected": M, "steps": n_steps}
