"""
What the app computes, and nothing about how it displays it.

One function, :func:`compute_imaging`, turns a :class:`ParameterModel` into
everything the imaging panel shows. It is deliberately a plain function over
plain data: no Qt, no figures, no globals — so it runs on a worker thread
without ceremony and is testable without a display.

Thread-safety note: the engine holds no mutable global state, and numpy's FFT
releases the GIL, so several of these genuinely run in parallel on separate
threads. The one thing not to do is share a mutable ``Stack`` across threads;
nothing here does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from litho_sim.app.pipeline import Pipeline

from litho_sim.analysis import compute_nils
from litho_sim.app.params import ParameterModel
from litho_sim.bake import apply_peb
from litho_sim.develop import (
    develop_3d,
    film_remaining,
    is_cleared,
    is_sealed,
    mack_development_rate,
    measure_cd_2d,
    print_resist_3d,
    remaining_thickness,
    sidewall_angle,
    simulate_resist,
    top_loss,
)
from litho_sim.expose import compute_aerial_image
from litho_sim.mask import (
    checkerboard,
    contact_array,
    isolated_line,
    lines_and_spaces,
    to_attenuated_psm,
)


@dataclass
class ImagingResult:
    """Everything the imaging panel draws, plus the numbers under it."""

    mask: NDArray[np.float64]
    aerial: NDArray[np.float64]
    latent: NDArray[np.float64]
    resist: NDArray[np.float64]
    cut_aerial: NDArray[np.float64]
    cut_latent: NDArray[np.float64]
    cut_resist: NDArray[np.float64]
    thickness_nm: NDArray[np.float64]      # remaining resist [nm]
    cut_thickness: NDArray[np.float64]
    film_nm: float
    x_nm: NDArray[np.float64]
    cd_nm: float
    nils: float
    contrast: float
    threshold: float
    elapsed_ms: float
    signature: tuple

    @property
    def summary(self) -> str:
        cd = "—" if not np.isfinite(self.cd_nm) or self.cd_nm <= 0 else f"{self.cd_nm:.1f} nm"
        return (
            f"CD {cd}    NILS {self.nils:.2f}    "
            f"contrast {self.contrast:.3f}    [{self.elapsed_ms:.0f} ms]"
        )


def build_mask(params: ParameterModel) -> NDArray:
    """The drawn pattern, as a transmittance array.

    Complex-valued when the mask type is att-psm — the dark regions carry a
    6 % amplitude at 180°, and the imaging path takes complex masks as-is.
    """
    grid = params.grid()
    n, px = grid.n_pixels, grid.pixel_size
    pitch = params.si("pitch")
    cd = params.si("cd")
    pattern = params["pattern"]

    if pattern == "lines and spaces":
        mask = lines_and_spaces(n, px, pitch=pitch, cd=cd)
    elif pattern == "contacts":
        # Square array: one pitch and one CD control drive both axes.
        mask = contact_array(n, px, pitch_x=pitch, pitch_y=pitch, cd_x=cd)
    elif pattern == "isolated line":
        mask = isolated_line(n, px, cd=cd)
    elif pattern == "checkerboard":
        mask = checkerboard(n, px, pitch=pitch, cd=cd)
    else:
        raise ValueError(f"unknown pattern '{pattern}'")

    if params["mask_type"] == "att-psm":
        mask = to_attenuated_psm(mask)
    return mask


def compute_imaging(params: ParameterModel, pipeline=None) -> ImagingResult:
    """Run mask → aerial → resist and measure the result.

    Everything the panel needs comes out of one call so the worker thread has
    a single unit of work to do and a single object to hand back.

    Parameters
    ----------
    params : ParameterModel
        The current knob settings.
    pipeline : Pipeline, optional
        A staged cache. When given, the mask and the Abbe sum are reused if
        the parameters they depend on have not changed — which turns a
        threshold or dose change from 47 ms into 0.02 ms. Without one, every
        stage is computed fresh, which is what the tests and any one-shot
        caller want.
    """
    t0 = time.perf_counter()

    grid = params.grid()
    optics = params.optics()
    resist_cfg = params.resist()

    if pipeline is not None:
        mask = pipeline.mask(params)
        aerial = pipeline.aerial(params)
    else:
        mask = build_mask(params)
        aerial = compute_aerial_image(mask, optics, grid, dose=params.dose)

    resist_model = params.resist_model
    if resist_model == "threshold":
        # Post-exposure bake. The engine's threshold model works in intensity
        # space and therefore *deliberately* bypasses diffusion
        # (develop/resist.py) — which left the app's PEB control doing
        # literally nothing, the same class of defect the vault already
        # records once. Blurring the aerial image before thresholding is the
        # standard diffused-aerial-image treatment, and it is what makes the
        # control mean something. At sigma = 0 (the app's default) apply_peb
        # returns a copy, so the app still reproduces the CLI and the engine
        # exactly.
        latent = apply_peb(aerial, resist_cfg.diffusion_sigma, grid.pixel_size)
        _, _, resist = simulate_resist(latent, resist_cfg, grid, model="threshold")

        # The Mack rate law over the same latent, kept as a *thickness*
        # rather than binarised. The threshold model above answers "did it
        # clear"; this answers "how much is left", which is the difference
        # between a footprint and a profile — and it is the number the
        # engine was already computing and discarding.
        thickness_nm = remaining_thickness(latent, resist_cfg)

        # Measured from the continuous latent at the develop threshold
        # rather than from the binarised resist: binarising first quantises
        # CD to whole pixels (4 nm at defaults — the audit's H5 finding),
        # and the Process Window tab measures sub-pixel, so the status bar
        # must agree with it. For positive tone the printed line is the
        # region *under* threshold.
        cd_m = measure_cd_2d(
            latent, grid.pixel_size, threshold=resist_cfg.threshold,
            feature="below" if resist_cfg.tone == "positive" else "above",
        )
        shown_threshold = resist_cfg.threshold
    else:
        # Chemistry route — mack or car. simulate_resist owns exposure and
        # bake internally, so it gets the *aerial*: the dose is already in
        # it, and the threshold path's pre-blur would double-count the PEB.
        # The latent panel then shows chemistry, not intensity — PAC after
        # the Gaussian bake (mack) or the protected fraction after the
        # reaction–diffusion bake (car), 1 = unexposed in both.
        _, latent, resist = simulate_resist(
            aerial, resist_cfg, grid, model=resist_model
        )

        # Depth the developer clears through that latent, as a thickness map
        # — the same rule simulate_resist binarised by, kept continuous.
        rate = mack_development_rate(
            latent, resist_cfg.mack_Rmax, resist_cfg.mack_Rmin,
            resist_cfg.mack_Mth, resist_cfg.mack_n,
        )
        film_nm = resist_cfg.thickness * 1e9
        cleared_nm = rate * resist_cfg.develop_time
        thickness_nm = np.clip(film_nm - cleared_nm, 0.0, film_nm)

        # CD measured where the develop front fails to reach the substrate —
        # sub-pixel on the continuous cleared-depth field, the same rule the
        # Process Window's mack/car measurement uses, so the two agree.
        cd_m = measure_cd_2d(
            cleared_nm, grid.pixel_size, threshold=film_nm,
            feature="below" if resist_cfg.tone == "positive" else "above",
        )
        # The dashed line the cut plot draws sits on the chemistry latent,
        # where the meaningful level is the develop threshold.
        shown_threshold = resist_cfg.mack_Mth

    mid = grid.n_pixels // 2
    cut_aerial = aerial[mid, :]
    cut_latent = latent[mid, :]
    cut_resist = resist[mid, :]
    x_nm = np.arange(grid.n_pixels) * grid.pixel_size * 1e9
    nils = compute_nils(
        cut_aerial, grid.pixel_size,
        threshold=resist_cfg.threshold,
        nominal_cd=params.si("cd"),
    )
    lo, hi = float(aerial.min()), float(aerial.max())
    contrast = (hi - lo) / (hi + lo) if (hi + lo) > 0 else 0.0

    return ImagingResult(
        # Display-safe: an att-PSM mask is complex, and imshow of a complex
        # array is undefined. The real part shows the −24.5 % amplitude
        # background for what it is.
        mask=mask.real if np.iscomplexobj(mask) else mask,
        aerial=aerial,
        latent=latent,
        resist=resist,
        cut_aerial=cut_aerial,
        cut_latent=cut_latent,
        cut_resist=cut_resist,
        thickness_nm=thickness_nm,
        cut_thickness=thickness_nm[mid, :],
        film_nm=float(resist_cfg.thickness * 1e9),
        x_nm=x_nm,
        cd_nm=cd_m * 1e9,
        nils=float(nils),
        contrast=float(contrast),
        # Carried so a view can draw the threshold this image was actually
        # computed with, even if the control has moved on since. For the
        # chemistry models this is the develop threshold, drawn against the
        # chemistry-space latent it applies to.
        threshold=float(shown_threshold),
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        signature=params.signature(),
    )


@dataclass
class Profile3DResult:
    """A developed 3-D profile, reduced to what a panel actually draws.

    Deliberately does *not* carry all four volumes ``print_resist_3d``
    returns — that is 20 MB, and three of them are intermediate. What it does
    carry are the developed solid, the latent image, and the cuts through
    them, sliced here on the worker thread so the GUI never indexes a large
    array while painting.
    """

    remaining: NDArray[np.bool_]        # (nz, ny, nx) developed solid
    cut_remaining: NDArray[np.bool_]    # (nz, nx) through `row`
    cut_latent: NDArray[np.float64]     # (nz, nx)
    x_nm: NDArray[np.float64]
    height_nm: float
    row: int
    sidewall_deg: float
    film_remaining_pct: float
    top_loss_nm: float
    sealed: bool
    cleared: bool
    label: str
    elapsed_ms: float
    signature: tuple

    @property
    def summary(self) -> str:
        angle = "—" if not np.isfinite(self.sidewall_deg) else f"{self.sidewall_deg:.1f}°"
        loss = "—" if not np.isfinite(self.top_loss_nm) else f"{self.top_loss_nm:.1f} nm"
        return (
            f"remaining {self.film_remaining_pct:.1f}%    sidewall {angle}    "
            f"top loss {loss}    [{self.elapsed_ms:.0f} ms]"
        )

    @property
    def diagnosis(self) -> str:
        """Why the picture is empty, when it is."""
        if self.sealed:
            return ("Film did not develop. Standing-wave nodes can seal the top "
                    "under the threshold model — raise PEB diffusion, turn "
                    "standing waves off, or use the finite-rate 'mack' model.")
        if self.cleared:
            return ("Film cleared completely. Dose too high for the contrast at "
                    "this pitch, or the develop front outran the film.")
        return ""


def compute_profile_3d(
    params: ParameterModel, pipeline: Pipeline | None = None
) -> Profile3DResult:
    """Develop the resist in depth and measure the resulting solid.

    Roughly 25× the cost of the 2-D pipeline — it runs one full Abbe sum per
    optical plane — which is why the app puts it behind an explicit button
    rather than on the live path.

    With a *pipeline*, the latent image is cached and only the develop step
    re-runs when a develop-only knob moves: 0.6 % of the work instead of all
    of it. Without one, this is the whole path from scratch.
    """
    t0 = time.perf_counter()

    grid = params.grid()
    resist_cfg = params.resist()

    if pipeline is None:
        result = print_resist_3d(
            build_mask(params),
            params.optics(),
            grid,
            resist_cfg,
            dose=params.dose,
            standing_waves=params.standing_waves,
            develop_model=params.develop_model,
        )
        remaining, latent = result["remaining"], result["latent"]
    else:
        latent, _z = pipeline.profile_latent(params)
        remaining = develop_3d(latent, resist_cfg, grid, model=params.develop_model)

    nz, ny, nx = remaining.shape
    row = ny // 2

    frac = film_remaining(remaining)
    label = (
        f"{params['thickness']:.0f} nm film · {params['n_z_slices']:.0f} planes · "
        f"{params.develop_model}"
        + (" · standing waves" if params.standing_waves else "")
    )

    return Profile3DResult(
        remaining=remaining,
        # Sliced here, on the worker: the GUI thread should never reach into
        # a 20 MB volume while it is trying to paint.
        cut_remaining=remaining[:, row, :],
        cut_latent=latent[:, row, :],
        x_nm=np.arange(nx) * grid.pixel_size * 1e9,
        height_nm=nz * grid.dz * 1e9,
        row=row,
        sidewall_deg=float(sidewall_angle(remaining, grid, row=row)),
        film_remaining_pct=100.0 * frac,
        top_loss_nm=top_loss(remaining, grid),
        sealed=is_sealed(remaining),
        cleared=is_cleared(remaining),
        label=label,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        signature=params.stage_signature("profile3d"),
    )


#: Rough wall-clock of one FDTD near-field solve, in milliseconds. Measured at
#: 8.5 s for the default 128-pixel DUV field and more at EUV; the number only
#: has to be large enough that the scheduler never runs a library live, and
#: ``LIVE_BUDGET_MS`` is 120 ms, so one solve clears that by two orders of
#: magnitude however the estimate is divided up.
_FDTD_SOLVE_MS = 9_000.0

#: Effective speedup from the solver pool, not the worker count. The kernel is
#: memory-bandwidth bound, so concurrent solves contend: measured 4.2x on small
#: fields and 2.5x at the default 128-pixel one. 3 is the honest middle.
_FDTD_POOL_SPEEDUP = 3.0


def estimate_cost_ms(params: ParameterModel) -> float:
    """Rough cost of the next :func:`compute_imaging`, without running it.

    Used to decide whether a change can be applied live or should wait for
    the drag to settle. Calibrated against measurements on this machine: the
    Abbe sum is one FFT per source point, so cost goes as (source points) ×
    (n² log n), and the vector model multiplies it by three — six when
    unpolarised, which is two incoherent input states.
    """
    n = float(params["n_pixels"])
    sg = float(params["source_grid"])
    sigma = float(params["sigma_outer"])

    # Non-zero source points ≈ the disc area on the σ grid.
    points = max(np.pi * (sigma * sg / 2.0) ** 2, 1.0)
    fft = (n ** 2) * np.log2(max(n, 2.0))
    # Calibrated on this machine against two points an order of magnitude
    # apart: 7 ms at n=96/sg=11 and 50 ms at n=128/sg=21, both σ=0.8.
    base = points * fft * 1.85e-6

    if params["imaging_model"] == "vector":
        base *= 6.0 if params["polarisation"] == "unpolarised" else 3.0

    # A rigorous mask model is not a multiplier on the Abbe sum; it is a
    # separate, far larger job done once and then cached. Reporting the cached
    # cost would let the scheduler run a multi-minute FDTD library live on a
    # slider drag, so an uncached library is quoted at its real scale and the
    # scheduler defers it. Once the library exists the extra cost is an
    # interpolation per source point, which is noise beside the transforms.
    if params["mask_model"] == "fdtd":
        from litho_sim.expose.m3d.provider import library_is_cached

        if not library_is_cached(params.optics(), params.grid()):
            # The angle solves run in a process pool, so the serial total
            # overstates the wall clock. Not by the worker count, though: the
            # FDTD kernel is memory-bandwidth bound, and nine concurrent solves
            # on this machine measured 2.5x rather than 9x. Capping the assumed
            # speedup keeps the estimate on the right side of reality — 27 s
            # predicted against 30.8 s measured for a 3x3 library at n=128.
            from litho_sim.expose.m3d.nearfield import _DEFAULT_WORKERS

            angles = float(params["m3d_angles"]) ** 2
            speedup = min(max(_DEFAULT_WORKERS, 1), _FDTD_POOL_SPEEDUP)
            base += (angles / speedup) * _FDTD_SOLVE_MS

    return float(base)
