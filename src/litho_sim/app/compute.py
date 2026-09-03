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
    #: What the latent image *is*, in words, so the Bake panel can label its
    #: axes honestly: intensity for the threshold model, PAC for mack, the
    #: protected fraction for car — 1 = unexposed for the chemistry two.
    latent_kind: str = "intensity after PEB"

    @property
    def summary(self) -> str:
        cd = "—" if not np.isfinite(self.cd_nm) or self.cd_nm <= 0 else f"{self.cd_nm:.1f} nm"
        return (
            f"CD {cd}    NILS {self.nils:.2f}    "
            f"contrast {self.contrast:.3f}    [{self.elapsed_ms:.0f} ms]"
        )

    @property
    def cd_text(self) -> str:
        return "—" if not np.isfinite(self.cd_nm) or self.cd_nm <= 0 else f"{self.cd_nm:.1f} nm"


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
        latent_kind = "intensity after PEB"
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
        latent_kind = ("PAC after bake" if resist_model == "mack"
                       else "protected fraction after bake")

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
        latent_kind=latent_kind,
    )


# ---------------------------------------------------------------------------
# Live previews — the two pictures that are not simulations
# ---------------------------------------------------------------------------
#
# Nothing physical recomputes while a control moves: exposure and development
# run only when asked, from the Simulate tab. Two pictures are exempt because
# they are *drawings of the settings* rather than results — the mask as
# drawn, and the illumination as sampled — and both cost well under a
# millisecond, so the tab that holds the controls can show them as they change.


def mask_preview(params: ParameterModel) -> NDArray[np.float64]:
    """The drawn pattern as the Mask tab shows it — real-valued, 1 = clear."""
    mask = build_mask(params)
    return mask.real if np.iscomplexobj(mask) else mask


@dataclass
class SourcePreview:
    """The illumination as the Abbe sum will sample it, plus the numbers that
    say what it can resolve."""

    source: NDArray[np.float64]     # (n, n) on the σ grid, rows η, cols ξ
    n_points: int                   # source points the Abbe sum will visit
    order_shift: float | None       # λ/(pitch·NA) in pupil units; None if aperiodic
    k1: float                       # CD · NA / λ
    label: str


def source_preview(params: ParameterModel) -> SourcePreview:
    """Build the source on its own grid and work out where the ±1 orders land.

    The picture the Source tab draws is the actual sampling — one pixel per
    source point on the ``source_grid`` — because that grid is the cost knob
    and seeing it coarsen is the point. The two dashed circles are the pupil
    displaced by one diffraction order of the drawn pitch: a source point
    inside the overlap images the pattern with two beams, one outside it
    contributes only a DC background, which is the whole reason off-axis
    illumination exists.
    """
    from litho_sim.expose.illumination import build_source

    optics = params.optics()
    source = build_source(
        optics.source_grid, optics.source_type,
        optics.sigma_outer, optics.sigma_inner,
        **(optics.source_kwargs or {}),
    )
    n_points = int(np.count_nonzero(source))

    periodic = params["pattern"] in ("lines and spaces", "contacts", "checkerboard")
    pitch = params.si("pitch")
    shift = (optics.wavelength / (pitch * optics.NA)) if periodic and pitch > 0 else None
    k1 = params.si("cd") * optics.NA / optics.wavelength

    label = (
        f"{optics.source_type} · σ {optics.sigma_outer:.2f}"
        + (f"/{optics.sigma_inner:.2f}" if optics.sigma_inner > 0 else "")
        + f" · {n_points} source points · k₁ {k1:.2f}"
    )
    return SourcePreview(
        source=source, n_points=n_points, order_shift=shift, k1=float(k1),
        label=label,
    )


@dataclass
class FilmPreview:
    """The coated film as the Resist tab sketches it: its geometry, its
    discretisation, and the exposure curve its chemistry implies."""

    thickness_nm: float
    dz_nm: float
    n_voxels: int                   # voxel rows the 3-D develop will use
    n_planes: int                   # optical planes the 3-D exposure samples
    plane_z_nm: NDArray[np.float64]
    dose_axis: NDArray[np.float64]  # mJ/cm²
    pac: NDArray[np.float64]        # exp(-C·E): PAC left after exposure
    dose_to_clear: float            # mJ/cm² at relative dose 1
    dill_C: float                   # noqa: N815 - the engine's own field name
    resist_model: str
    tone: str
    label: str


#: Knobs the film sketch reads — the tab redraws it when one of these moves.
FILM_PREVIEW_KEYS: tuple[str, ...] = (
    "thickness", "dz", "n_z_slices", "dose_nominal", "dill_C",
    "resist_model", "tone",
)


def film_preview(params: ParameterModel) -> FilmPreview:
    """What the coat step produced, without simulating anything.

    Two facts are worth seeing before pressing Run. The film's discretisation
    — ``n_z_slices`` optical planes interpolated onto ``thickness/dz`` voxel
    rows — is the 3-D cost and accuracy knob, and is invisible as two
    numbers. And the Dill exposure curve, :math:`m(E) = e^{-CE}`, is the
    resist's whole sensitivity in one line: the dose slider on the Expose
    tab multiplies ``dose_nominal``, so where that lands on the curve is
    what "dose 1.0" means for this resist.
    """
    t_nm = float(params["thickness"])
    dz_nm = float(params["dz"])
    n_planes = int(params["n_z_slices"])
    e0 = float(params["dose_nominal"])
    c = float(params["dill_C"])
    dose_axis = np.linspace(0.0, 3.0 * e0, 121)
    return FilmPreview(
        thickness_nm=t_nm,
        dz_nm=dz_nm,
        n_voxels=max(int(round(t_nm / dz_nm)), 1),
        n_planes=n_planes,
        plane_z_nm=np.linspace(0.0, t_nm, n_planes),
        dose_axis=dose_axis,
        pac=np.exp(-c * dose_axis),
        dose_to_clear=e0,
        dill_C=c,
        resist_model=str(params["resist_model"]),
        tone=str(params["tone"]),
        label=(f"{t_nm:.0f} nm {params['tone']} resist · {params['resist_model']} model · "
               f"{n_planes} optical planes on {max(int(round(t_nm / dz_nm)), 1)} voxel rows"),
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
    optical plane — so the Simulate tab lists it as its own run.

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
#: 8.5 s for the default 128-pixel DUV field and more at EUV.
_FDTD_SOLVE_MS = 9_000.0

#: Effective speedup from the solver pool, not the worker count. The kernel is
#: memory-bandwidth bound, so concurrent solves contend: measured 4.2x on small
#: fields and 2.5x at the default 128-pixel one. 3 is the honest middle.
_FDTD_POOL_SPEEDUP = 3.0


def estimate_cost_ms(params: ParameterModel) -> float:
    """Rough cost of the next :func:`compute_imaging`, without running it.

    Shown beside the Run buttons, so the price of a source grid or of
    switching to the vector model is visible before it is paid. Calibrated
    against measurements on this machine: the Abbe sum is one FFT per source
    point, so cost goes as (source points) × (n² log n), and the vector model
    multiplies it by three — six when unpolarised, which is two incoherent
    input states.
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
    # separate, far larger job done once and then cached, so an uncached
    # library is quoted at its real scale — the number a user should see
    # before pressing Run. Once the library exists the extra cost is an
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
