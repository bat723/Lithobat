"""
Recomputing only what changed.

The 2-D pipeline is mask → aerial → resist, and the middle step is **98 % of
the cost** (47 ms against 0.008 and 0.015). Recomputing all three whenever any
control moves therefore means paying for the Abbe sum to answer questions that
have nothing to do with it — dragging the threshold slider re-ran a hundred
FFTs to change where one number crossed a curve.

This module caches each stage against the parameters *that stage* depends on,
so a change invalidates only its own stage and the ones downstream. Measured
consequences at 128 px:

======================  ================  ==========
change                  recomputes        cost
======================  ================  ==========
threshold, tone         resist + metrics  0.04 ms
dose, normalisation     a scalar rescale  0.04 ms
NA, sigma, source_grid  aerial + resist   47.6 ms
pitch, CD, grid         everything        48.3 ms
======================  ================  ==========

Two structural facts shape it. Dose and normalisation are a single
multiplicative constant applied *after* the Abbe sum, so the cache holds the
raw sum and rescales it — see :func:`~litho_sim.expose.aerial_image.
normalisation_scale`. And the 3-D branch does **not** hang off the 2-D aerial
image: ``exposure_volume`` rebuilds the optics per depth plane with its own
defocus and image index, so 3-D is a parallel branch off the mask, and is
modelled that way here rather than looking deceptively reusable.

Qt-free, like the rest of the app's core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from litho_sim.app.params import STAGES, ParameterModel
from litho_sim.expose.aerial_image import compute_aerial_image, normalisation_scale


@dataclass
class _Entry:
    """One cached stage output and the signature it was computed from."""

    signature: tuple
    value: Any


class Pipeline:
    """Staged cache over the 2-D imaging path.

    Not a scheduler and not a thread — it decides *what* must be recomputed,
    not when or where. Call :meth:`aerial` or :meth:`mask` and it returns a
    cached result or produces a fresh one.
    """

    def __init__(self) -> None:
        self._cache: dict[str, _Entry] = {}
        #: Counts of genuine (non-cached) evaluations, for tests and telemetry.
        #: `profile_latent` is not one of the 2-D stages but is cached alike.
        self.runs: dict[str, int] = {s: 0 for s in (*STAGES, "profile_latent")}

    # -- cache plumbing ------------------------------------------------
    def _get(self, stage: str, signature: tuple):
        entry = self._cache.get(stage)
        if entry is not None and entry.signature == signature:
            return entry.value
        return None

    def _put(self, stage: str, signature: tuple, value: Any):
        self._cache[stage] = _Entry(signature, value)
        self.runs[stage] += 1
        return value

    def invalidate(self, stage: str | None = None) -> None:
        """Drop one stage's cache, or all of them."""
        if stage is None:
            self._cache.clear()
        else:
            self._cache.pop(stage, None)

    # -- stages --------------------------------------------------------
    def mask(self, params: ParameterModel) -> NDArray[np.float64]:
        """The drawn pattern."""
        sig = params.stage_signature("mask")
        hit = self._get("mask", sig)
        if hit is not None:
            return hit

        from litho_sim.app.compute import build_mask

        return self._put("mask", sig, build_mask(params))

    def aerial_raw(self, params: ParameterModel):
        """``(raw_sum, peak, clear)`` — the Abbe sum before normalisation.

        Cached against the optics signature *excluding* dose and
        normalisation, because neither enters the sum.
        """
        sig = params.stage_signature("aerial")
        hit = self._get("aerial", sig)
        if hit is not None:
            return hit

        value = compute_aerial_image(
            self.mask(params), params.optics(), params.grid(), return_raw=True
        )
        return self._put("aerial", sig, value)

    def aerial(self, params: ParameterModel) -> NDArray[np.float64]:
        """The finished aerial image, dosed and normalised.

        Changing only dose or normalisation lands here without touching the
        transform: it is one multiply over the cached raw sum.
        """
        raw, peak, clear = self.aerial_raw(params)
        scale = normalisation_scale(
            params["normalisation"], peak, clear, params.dose
        )
        return raw * scale

    # -- the 3-D branch ------------------------------------------------
    def profile_latent(self, params: ParameterModel):
        """``(latent, z)`` — the depth-resolved image just before developing.

        This is where the 3-D cost lives: the Abbe sums are 71 % of it and the
        absorption march another 25 %, against 0.6 % for the develop itself.
        Caching here means changing the develop model, the threshold or the
        tone re-runs only that last step.

        Not derived from the 2-D aerial image, and it cannot be:
        ``exposure_volume`` rebuilds the optics per depth plane with its own
        defocus and image index, so the 3-D branch hangs off the mask.
        """
        sig = params.latent3d_signature()
        hit = self._get("profile_latent", sig)
        if hit is not None:
            return hit

        from litho_sim.bake import apply_peb_3d
        from litho_sim.develop.resist3d import (
            apply_absorption,
            apply_vertical_interference,
            exposure_volume,
        )

        grid, optics, resist = params.grid(), params.optics(), params.resist()
        intensity, z = exposure_volume(
            self.mask(params), optics, grid, resist, dose=params.dose
        )
        if params.standing_waves:
            intensity = apply_vertical_interference(intensity, resist, optics, grid)
        # `intensity` already carries the dose, so absorption must not apply
        # it again — passing it twice makes every dose sweep quadratic. Same
        # rule print_resist_3d follows.
        pac, _ = apply_absorption(intensity, resist, grid.dz, dose=1.0)
        latent = apply_peb_3d(pac, resist, grid)
        return self._put("profile_latent", sig, (latent, z))
