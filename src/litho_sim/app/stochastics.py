"""
The Monte-Carlo computation behind the app's Stochastics tab.

Qt-free, like ``fem``: the tab hands a :class:`StochRequest` to the worker
thread, :func:`compute_stochastics` turns it into a :class:`StochResult`, and
the tab draws whatever comes back. Tests drive this module headless.

What one press of Run buys: the current dock configuration printed once
deterministically through the ``car`` chain (the design intent), then
*trials* more times through :func:`~litho_sim.develop.stochastic.
stochastic_trials` — same chemistry, sampled photon and molecule counts —
and the trial-to-trial scatter reduced to the numbers a lithographer quotes:
LER, LWR, correlation length, LCDU, and the bridge/break failure rate
against the deterministic print. The wavelength comes from the dock's
optics, which is the entire difference between a quiet ArF panel and a loud
EUV one.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from litho_sim.analysis import (
    CDUniformity,
    FailureStats,
    LineRoughness,
    failure_rate,
    measure_lcdu,
    measure_line_roughness,
    pool_roughness,
)
from litho_sim.app.compute import build_mask
from litho_sim.app.params import ParameterModel
from litho_sim.develop import simulate_resist, stochastic_trials
from litho_sim.expose import compute_aerial_image

logger = logging.getLogger(__name__)


@dataclass
class StochRequest:
    """Everything one press of the Run button asks for.

    ``params`` is a deep copy of the dock's model, snapshotted on the GUI
    side — the same contract as :class:`~litho_sim.app.fem.FemRequest`.
    Trials and seed live here rather than in the dock because they
    parameterise this tab alone.
    """

    params: ParameterModel
    trials: int = 16
    seed: int = 0


@dataclass
class StochResult:
    """One Monte-Carlo batch, measured and reduced for drawing."""

    reference: NDArray[np.float64]     # deterministic car print (ny, nx)
    example: NDArray[np.float64]       # the first stochastic trial
    prob_map: NDArray[np.float64]      # per-pixel print probability
    roughness: LineRoughness           # pooled over trials
    lcdu: CDUniformity
    fails: FailureStats
    mean_photons: float                # absorbed photons per voxel
    mean_pag: float                    # PAG molecules per voxel
    x_nm: NDArray[np.float64]
    label: str
    elapsed_ms: float

    @property
    def summary(self) -> str:
        def nm3(v: float) -> str:
            return "—" if not np.isfinite(v) else f"{v * 1e9:.2f} nm"

        cd = ("—" if not np.isfinite(self.lcdu.cd_mean)
              else f"{self.lcdu.cd_mean * 1e9:.1f} nm")
        return (
            f"LER(3σ) {nm3(self.roughness.ler_3s)}    "
            f"LWR(3σ) {nm3(self.roughness.lwr_3s)}    "
            f"ξ {nm3(self.roughness.corr_length)}    "
            f"CD {cd} · LCDU {nm3(self.lcdu.lcdu)}    "
            f"fails {self.fails.n_failed}/{self.fails.n_trials}    "
            f"[{self.elapsed_ms:.0f} ms]"
        )

    @property
    def diagnosis(self) -> str:
        """Why the numbers are dashes, when they are."""
        if self.lcdu.n_failed == self.lcdu.n_trials:
            return (
                "No trial printed a measurable feature. The deterministic "
                "car print is the place to start: tune dose, Develop "
                "threshold and the quencher until it prints, then come back."
            )
        if not np.isfinite(self.roughness.ler):
            return (
                "Feature printed but line roughness is undefined — LER wants "
                "a line running down the field (lines and spaces, or an "
                "isolated line). LCDU and failures above still apply."
            )
        return ""


def compute_stochastics(
    req: StochRequest,
    progress: Callable[[int, int], None] | None = None,
) -> StochResult:
    """Run the deterministic reference plus *trials* sampled prints.

    Cost is one car develop for the reference and one per trial — the
    reaction–diffusion bake dominates, so wall clock is roughly
    ``(trials + 1) × car develop time``. *progress* is called with
    ``(done, total)`` in those units.
    """
    t0 = time.perf_counter()
    params = req.params
    grid = params.grid()
    optics = params.optics()
    # The reference is the *design intent*, so cosmetic edge roughness must
    # not blur it — a noisy reference would count its own noise as failures.
    resist = dataclasses.replace(params.resist(), use_stochastic=False)

    total = req.trials + 1
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total)

    mask = build_mask(params)
    aerial = compute_aerial_image(mask, optics, grid, dose=params.dose)

    _, _, reference = simulate_resist(aerial, resist, grid, model="car")
    tick()

    # One realisation per call, seeds spaced from the request seed — the
    # batch is reproducible and each trial reports progress as it lands,
    # which a single stochastic_trials(trials=N) call could not.
    trials = np.empty((req.trials, *aerial.shape), dtype=np.float64)
    depths = np.empty_like(trials)
    level = float(resist.thickness * 1e9)
    first_sample = None
    for i in range(req.trials):
        res = stochastic_trials(
            aerial, resist, grid, optics.wavelength,
            trials=1, seed=req.seed + i,
        )
        trials[i] = res.resist[0]
        depths[i] = res.depth[0]
        if i == 0:
            first_sample = res.sample
        tick()
    assert first_sample is not None

    # Geometry is read off the continuous develop-depth field with sub-pixel
    # crossings; the binary images serve topology only. A binary edge sits
    # on a half-pixel, and at the app's 4 nm grid that read a 1.3 nm LER as
    # 0.1 nm — the panel used to report the grid, not the resist.
    lcdu = measure_lcdu(depths, grid.pixel_size, threshold=level, feature="below")
    fails = failure_rate(trials, reference)
    pooled = pool_roughness([
        measure_line_roughness(d, grid.pixel_size, threshold=level, feature="below")
        for d in depths
    ])

    label = (
        f"{params['pattern']} · pitch {params['pitch']:.0f} nm · "
        f"λ {params['wavelength']:.1f} nm · dose {params.dose:.2f} · "
        f"{req.trials} trials · "
        f"{first_sample.mean_photons:.0f} photons, "
        f"{first_sample.mean_pag:.0f} PAG per voxel"
    )

    return StochResult(
        reference=reference,
        example=trials[0],
        prob_map=trials.mean(axis=0),
        roughness=pooled,
        lcdu=lcdu,
        fails=fails,
        mean_photons=float(first_sample.mean_photons),
        mean_pag=float(first_sample.mean_pag),
        x_nm=np.asarray(np.arange(grid.n_pixels) * grid.pixel_size * 1e9, dtype=np.float64),
        label=label,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
