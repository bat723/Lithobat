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
        #: `profile_latent` and `opc` are not 2-D stages but are cached alike.
        self.runs: dict[str, int] = {
            s: 0 for s in (*STAGES, "profile_latent", "opc", "print_model")
        }

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
        """The pattern as it goes on the mask: drawn, or corrected.

        With OPC on this is the corrected layout rasterised, and its
        signature is the whole 2-D process — ``stage_signature`` already
        says so — because that is what the correction depends on.
        """
        sig = params.stage_signature("mask")
        hit = self._get("mask", sig)
        if hit is not None:
            return hit

        from litho_sim.app.compute import build_mask

        corrected = self.opc(params).corrected if params.opc_enabled else None
        return self._put("mask", sig, build_mask(params, corrected))

    # -- the correction ------------------------------------------------
    def print_model(self, params: ParameterModel):
        """The process as a print model, its SOCS kernels kept across runs.

        Keyed on the aerial stage. A change there, or anywhere with OPC on,
        builds a fresh model — cheap — but hands it the old kernel cache,
        which the model re-keys on its own optics and grid and reuses when
        those are unchanged. The kernels are the only part worth keeping.
        """
        from litho_sim.app.opc import print_model

        sig = params.stage_signature("aerial")
        hit = self._get("print_model", sig)
        if hit is not None:
            return hit
        previous = self._cache.get("print_model")
        cache = previous.value._raw_cache if previous is not None else None
        return self._put("print_model", sig, print_model(params, cache=cache))

    def opc(self, params: ParameterModel):
        """The correction for the current process — an ``OPCResult``.

        Cached on :meth:`ParameterModel.opc_signature`, which leaves the
        ``opc`` switch out: a correction run on its own from the Simulate
        tab is the one the next Print uses once the switch is on.
        """
        from litho_sim.app.opc import compute_opc

        sig = params.opc_signature()
        hit = self._get("opc", sig)
        if hit is not None:
            return hit
        return self._put("opc", sig, compute_opc(params, model=self.print_model(params)))

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

        from litho_sim.develop.resist3d import (
            apply_vertical_interference,
            exposure_volume,
            latent_volume,
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
        # The car resist model bakes in 3-D with the same acid/quencher
        # chemistry the 2-D path runs; the other models take the Gaussian.
        bake = "car" if params.resist_model == "car" else "gaussian"
        latent = latent_volume(intensity, resist, grid, bake=bake)["latent"]
        return self._put("profile_latent", sig, (latent, z))
