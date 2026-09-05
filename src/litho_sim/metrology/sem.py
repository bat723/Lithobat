"""
A CD-SEM image of what the engine printed.

The rest of the engine answers *what prints*. This answers what an inspection
tool would *see* — which is not the same picture, and the gap between the two
is where metrology lives. A scanning electron microscope rasters a focused
probe over the wafer and counts the secondary electrons (SEs) it kicks out.
Three things shape the image, and this module has one term for each:

**Material contrast.** The SE yield depends on what the beam lands on. Resist
and silicon differ by tens of percent, so the exposed substrate reads darker
than the resist top. :attr:`SEMConfig.substrate_yield` is that ratio.

**Edge bloom.** Yield rises steeply with surface tilt — Seiler's
:math:`\\delta(\\theta) \\propto \\sec^{n}\\theta` — because SEs generated in a
sidewall have a short path out through the wall. Every step edge therefore
draws a bright line, the "white edge" a CD-SEM measures from. Modelled here as
an emission proportional to the *height crossed* by each pixel (so a
full-thickness wall is brighter than a partially developed slope), spread
laterally over an SE escape length (:attr:`SEMConfig.escape_length`) and scaled
by :attr:`SEMConfig.edge_yield`.

**The probe.** A Gaussian of :attr:`SEMConfig.beam_fwhm` blurs everything,
material contrast and edge bloom alike, and the finite number of primary
electrons per pixel makes the count Poisson — which is why a fast scan is
grainy and why frames are averaged.

The model is a signal model, not a Monte-Carlo transport code: no beam
penetration depth, no charging, no trench shadowing. It reproduces the two
things a lithographer reads off a CD-SEM — where the edges are and how rough
they look — and :func:`measure_cd_sem` measures the CD back from the image
the way a tool does, by finding the edge peaks. Comparing that number with
the simulation's own CD shows the measurement bias the edge bloom introduces.

**Wafer stacks.** The same three terms image a multi-material wafer — a
device out of the process-flow engine rather than a resist print. Material
contrast comes from each :class:`~litho_sim.wafer.materials.Material`'s
``se_yield`` instead of the one ``substrate_yield`` ratio, so a cleaved
nanosheet stack shows its oxides bright, its silicon dark and its metal gate
between; edge bloom stays where the topography is, at solid/vacuum
boundaries, because a buried interface on a flat cleaved face has no
sidewall to emit from. :func:`stack_xsection_sem` and
:func:`stack_topdown_sem` take a :class:`~litho_sim.wafer.stack.Stack`
directly; the ``material_*`` functions under them take arrays.

All lengths are metres, all images ``(rows, cols)`` with ``origin="lower"``
conventions; signals are in secondary electrons per primary, normalised so
the flat resist top emits 1.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks

from litho_sim.wafer.materials import MATERIAL_LIBRARY, VACUUM, Material, get_material

if TYPE_CHECKING:
    from litho_sim.wafer.stack import Stack

__all__ = [
    "SEMConfig", "SEMImage", "SEMMeasurement",
    "measure_cd_sem", "top_surface",
    "topdown_sem", "topdown_signal", "xsection_sem", "xsection_signal",
    "material_yields",
    "material_topdown_sem", "material_topdown_signal",
    "material_xsection_sem", "material_xsection_signal",
    "stack_topdown_sem", "stack_xsection_sem",
    "GeometryBuffers", "tilt_signal", "tilt_sem", "screen_occlusion",
]

#: Gaussian FWHM → σ.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

#: What the vacuum above a cleaved sample emits, relative to the resist top.
#: Not zero: a real X-SEM background is a faint grey, and a zero-signal region
#: would have zero shot noise, which reads as a painted-on black.
VACUUM_YIELD = 0.04


@dataclass(frozen=True)
class SEMConfig:
    """The instrument.

    Parameters
    ----------
    beam_fwhm : float
        Probe diameter [m]. Blurs the whole image.
    escape_length : float
        Lateral extent of edge bloom beyond the probe [m] — the SE escape
        length at a sidewall. Widens only the edges.
    edge_yield : float
        SEs emitted per primary from a full-height sidewall, relative to the
        flat resist top. Sets how bright edges are against the interior.
    substrate_yield : float
        SE yield of the exposed substrate relative to the resist top. Below 1
        the developed floor reads dark, above 1 bright.
    electrons_per_pixel : float
        Mean primary electrons per pixel per frame. The shot-noise knob.
    frames : int
        Frames averaged. Noise falls as the square root.
    seed : int
        RNG seed, so an image is reproducible.
    secant_power : float
        The exponent in Seiler's tilt law, ``δ ∝ sec^n θ`` — how much
        brighter a surface gets as it turns away from the beam. Only the
        tilt view sees the surface at an angle, so only it reads this;
        the top-down and cross-section models carry the same physics as
        the edge bloom instead. 0.8 sits inside the measured 0.6–1.3.
    directionality : float
        The Everhart–Thornley detector sits to one side of the column, so
        faces turned towards it collect more of their electrons than faces
        turned away. 0 is an ideal in-lens detector that sees every
        direction alike; 1 makes a face pointing straight away from the
        detector go dark. Tilt view only.
    shadowing : float
        How strongly a trench floor darkens because the walls around it
        block secondary electrons on their way to the detector. 0 turns the
        term off; 1 lets a fully enclosed floor go to the vacuum level.
        Tilt view only.
    """

    beam_fwhm: float = 3.0e-9
    escape_length: float = 2.0e-9
    edge_yield: float = 3.0
    substrate_yield: float = 0.6
    electrons_per_pixel: float = 100.0
    frames: int = 4
    seed: int = 0
    secant_power: float = 0.8
    directionality: float = 0.5
    shadowing: float = 0.5

    def __post_init__(self) -> None:
        if self.beam_fwhm < 0 or self.escape_length < 0:
            raise ValueError("beam_fwhm and escape_length must be >= 0")
        if self.edge_yield < 0 or self.substrate_yield < 0:
            raise ValueError("yields must be >= 0")
        if self.electrons_per_pixel <= 0:
            raise ValueError("electrons_per_pixel must be > 0")
        if int(self.frames) < 1:
            raise ValueError("frames must be >= 1")
        if self.secant_power < 0:
            raise ValueError("secant_power must be >= 0")
        if not 0.0 <= self.directionality <= 1.0:
            raise ValueError("directionality must be in [0, 1]")
        if not 0.0 <= self.shadowing <= 1.0:
            raise ValueError("shadowing must be in [0, 1]")

    @property
    def electrons(self) -> float:
        """Primaries per pixel over all frames — what the noise depends on."""
        return float(self.electrons_per_pixel) * int(self.frames)


@dataclass
class SEMImage:
    """One SEM frame stack, noisy and noiseless, with its geometry."""

    image: NDArray[np.float64]      # Poisson-sampled, SE per primary
    signal: NDArray[np.float64]     # the same without shot noise
    pixel_size: float               # column pitch [m]
    row_size: float                 # row pitch [m] — dz for a cross-section
    mode: str                       # "topdown" | "xsection" | "tilt"
    electrons: float                # primaries per pixel behind `image`
    row_origin: float = 0.0         # height of row 0 [m]; negative under a film

    @property
    def extent_nm(self) -> tuple[float, float, float, float]:
        """``(x0, x1, y0, y1)`` for ``imshow``, in nanometres.

        A cross-section puts the film's bottom at z = 0 and the substrate
        rows below it, so the axis reads as depth into the wafer.
        """
        rows, cols = self.image.shape
        y0 = self.row_origin * 1e9
        return (0.0, cols * self.pixel_size * 1e9, y0, y0 + rows * self.row_size * 1e9)

    @property
    def snr(self) -> float:
        """Signal-to-noise on the resist top, as measured from the frame."""
        noise = float(np.std(self.image - self.signal))
        return float(np.mean(self.signal)) / noise if noise > 0 else float("inf")


@dataclass
class SEMMeasurement:
    """A CD read off an SEM image by its edge peaks."""

    cd: float                        # [m]; NaN when no feature was found
    left: float                      # edge positions [m]
    right: float
    feature: str                     # "line" | "space"
    profile: NDArray[np.float64]     # the row-averaged signal the edges came from
    x: NDArray[np.float64]           # its axis [m]
    edges: NDArray[np.float64]       # every edge found [m]

    @property
    def found(self) -> bool:
        return bool(np.isfinite(self.cd))


# ---------------------------------------------------------------------------
# Signal formation
# ---------------------------------------------------------------------------


def _central_diff(a: NDArray, axis: int, wrap: bool) -> NDArray:
    """Half the height change between a pixel's two neighbours.

    Across a single step of height H between pixels i and i+1, both pixels
    report H/2 — the wall is shared between the two sides of the edge and
    the total is conserved, which is what makes the bloom's brightness a
    measure of wall height rather than of where the grid happened to fall.
    """
    if wrap:
        return 0.5 * np.abs(np.roll(a, -1, axis) - np.roll(a, 1, axis))
    fwd = np.diff(a, axis=axis, append=np.take(a, [-1], axis=axis))
    back = np.diff(a, axis=axis, prepend=np.take(a, [0], axis=axis))
    return 0.5 * (np.abs(fwd) + np.abs(back))


def top_surface(remaining: NDArray[np.bool_], dz: float) -> NDArray[np.float64]:
    """Height of the top of the resist in each column, from a developed volume.

    The *top*, not the amount: a column with an undercut still presents its
    cap to a top-down beam, and the beam sees nothing of the cavity below.
    Zero where the column cleared.
    """
    r = np.asarray(remaining, dtype=bool)
    nz = r.shape[0]
    any_left = r.any(axis=0)
    highest = nz - np.argmax(r[::-1], axis=0)       # 1-based index of the top voxel
    return np.where(any_left, highest, 0).astype(np.float64) * float(dz)


def topdown_signal(
    height: NDArray[np.float64],
    pixel_size: float,
    cfg: SEMConfig,
    film_thickness: float | None = None,
) -> NDArray[np.float64]:
    """The noiseless top-down SE signal from a resist height map.

    Parameters
    ----------
    height : (ny, nx)
        Remaining resist thickness per column [m]. Zero is cleared substrate.
    pixel_size : float
        Column pitch [m]; the field is taken as periodic, as the imaging is.
    cfg : SEMConfig
    film_thickness : float, optional
        The as-coated thickness, so a full-height wall scores exactly
        ``edge_yield``. Defaults to the tallest column present.
    """
    h = np.asarray(height, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"height must be (ny, nx), got shape {h.shape}")
    h_ref = float(film_thickness) if film_thickness else float(max(h.max(), 1e-12))
    lam = max(float(cfg.escape_length), 1e-12)

    # Which material the beam lands on. Soft over the escape length so a
    # few-nanometre remnant reads as partly substrate, as it would.
    base = cfg.substrate_yield + (1.0 - cfg.substrate_yield) * (1.0 - np.exp(-h / lam))
    return _topdown_from(base, h, h_ref, pixel_size, cfg)


def _topdown_from(
    base: NDArray[np.float64],
    h: NDArray[np.float64],
    h_ref: float,
    pixel_size: float,
    cfg: SEMConfig,
) -> NDArray[np.float64]:
    """Edge bloom over a base-yield map, then the probe.

    The part of a top-down image that does not care what the surface is
    made of: *base* says what each pixel emits flat, *h* says where the
    walls are. The resist path and the wafer-stack path differ only in how
    they arrive at *base*.
    """
    lam = max(float(cfg.escape_length), 1e-12)
    px = float(pixel_size)

    # Sidewall crossed by each pixel, in units of the reference height, then
    # spread over the distance SEs travel out of the wall.
    wall = np.hypot(_central_diff(h, 1, True), _central_diff(h, 0, True)) / h_ref
    if cfg.escape_length > 0:
        wall = gaussian_filter(wall, sigma=lam / px, mode="wrap")

    signal = base + cfg.edge_yield * wall
    if cfg.beam_fwhm > 0:
        signal = gaussian_filter(signal, sigma=cfg.beam_fwhm * _FWHM_TO_SIGMA / px,
                                 mode="wrap")
    return signal


def xsection_signal(
    remaining: NDArray[np.bool_],
    pixel_size: float,
    dz: float,
    cfg: SEMConfig,
    substrate_rows: int | None = None,
) -> NDArray[np.float64]:
    """The noiseless SE signal from a cleaved cross-section.

    Parameters
    ----------
    remaining : (nz, nx)
        One slice of the developed volume, ``True`` where resist survived,
        row 0 at the substrate.
    pixel_size, dz : float
        Column and row pitch [m].
    cfg : SEMConfig
    substrate_rows : int, optional
        Rows of substrate drawn under the film. Defaults to 15 % of the film.

    The cleaved face shows material contrast in the bulk and bloom along
    every boundary the face cuts — resist against vacuum, resist against
    substrate — because a boundary on the face is an edge the beam crosses.
    Vacuum is drawn above the film as well as in its openings, so the top of
    a full-height line is a boundary too: without it a line reaching the top
    row of the volume had no edge there and no bloom, which a test caught.
    """
    m = np.asarray(remaining, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"remaining must be (nz, nx), got shape {m.shape}")
    nz, nx = m.shape
    sub = _substrate_rows(nz, substrate_rows)
    vac = _vacuum_rows(nz)

    mat = np.full((sub + nz + vac, nx), VACUUM_YIELD, dtype=np.float64)
    mat[:sub] = cfg.substrate_yield
    film = mat[sub:sub + nz]
    film[m] = 1.0
    return _xsection_from(mat, pixel_size, dz, cfg)


def _xsection_from(
    mat: NDArray[np.float64], pixel_size: float, dz: float, cfg: SEMConfig
) -> NDArray[np.float64]:
    """Bloom along the solid/vacuum outline of a yield map, then the probe.

    *mat* is what each pixel of the cleaved face emits flat; anything above
    :data:`VACUUM_YIELD` is solid. Only the outline of the solid blooms — a
    boundary between two solids on a flat face has no sidewall, so it shows
    as a step in grey and nothing more. Periodic across, clamped up and
    down, as the face is.
    """
    solid = (mat > VACUUM_YIELD).astype(np.float64)
    edge = np.hypot(_central_diff(solid, 1, True), _central_diff(solid, 0, False))

    lam = max(float(cfg.escape_length), 1e-12)
    px, pz = float(pixel_size), float(dz)
    if cfg.escape_length > 0:
        edge = gaussian_filter(edge, sigma=(lam / pz, lam / px), mode=("nearest", "wrap"))
    signal = mat + cfg.edge_yield * edge
    if cfg.beam_fwhm > 0:
        s = cfg.beam_fwhm * _FWHM_TO_SIGMA
        signal = gaussian_filter(signal, sigma=(s / pz, s / px), mode=("nearest", "wrap"))
    return signal


def _substrate_rows(nz: int, requested: int | None) -> int:
    return int(requested) if requested is not None else max(int(round(0.15 * nz)), 2)


def _vacuum_rows(nz: int) -> int:
    """Headroom above the film, so its top surface is a boundary."""
    return max(int(round(0.1 * nz)), 2)


def _sample(signal: NDArray[np.float64], cfg: SEMConfig,
            rng: np.random.Generator | None) -> NDArray[np.float64]:
    """Count electrons: Poisson at ``signal × primaries`` per pixel, over all
    frames, back in SE-per-primary units."""
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    n = cfg.electrons
    counts = rng.poisson(np.clip(signal, 0.0, None) * n)
    return counts.astype(np.float64) / n


def topdown_sem(
    height: NDArray[np.float64],
    pixel_size: float,
    cfg: SEMConfig | None = None,
    film_thickness: float | None = None,
    rng: np.random.Generator | None = None,
) -> SEMImage:
    """A top-down CD-SEM frame of a resist height map. See :func:`topdown_signal`."""
    cfg = cfg or SEMConfig()
    signal = topdown_signal(height, pixel_size, cfg, film_thickness)
    return SEMImage(
        image=_sample(signal, cfg, rng), signal=signal,
        pixel_size=float(pixel_size), row_size=float(pixel_size),
        mode="topdown", electrons=cfg.electrons,
    )


def xsection_sem(
    remaining: NDArray[np.bool_],
    pixel_size: float,
    dz: float,
    cfg: SEMConfig | None = None,
    substrate_rows: int | None = None,
    rng: np.random.Generator | None = None,
) -> SEMImage:
    """A cross-section SEM frame of one slice of a developed volume. See
    :func:`xsection_signal`."""
    cfg = cfg or SEMConfig()
    signal = xsection_signal(remaining, pixel_size, dz, cfg, substrate_rows)
    sub = _substrate_rows(np.asarray(remaining).shape[0], substrate_rows)
    return SEMImage(
        image=_sample(signal, cfg, rng), signal=signal,
        pixel_size=float(pixel_size), row_size=float(dz),
        mode="xsection", electrons=cfg.electrons,
        row_origin=-sub * float(dz),
    )


# ---------------------------------------------------------------------------
# Wafer stacks: material contrast
# ---------------------------------------------------------------------------


def material_yields(
    materials: Iterable[Material] | None = None,
    overrides: Mapping[str | int | Material, float] | None = None,
) -> NDArray[np.float64]:
    """A 256-entry lookup from material ID to SE yield.

    ``lut[stack.mat]`` turns a voxel array into what each voxel would emit.
    Vacuum is :data:`VACUUM_YIELD`; every registered material carries its own
    ``se_yield``; an ID nothing registers reads as the resist top, 1, rather
    than as vacuum — a phantom hole would grow a phantom bloom around it.

    Parameters
    ----------
    materials : iterable of Material, optional
        The library to read. Defaults to the built-in one; a
        :class:`~litho_sim.wafer.stack.Stack` passes its own
        ``materials.values()`` so a user-registered film is honoured.
    overrides : mapping, optional
        ``{material: yield}`` applied last. The app maps its *Substrate
        yield* knob onto silicon this way.
    """
    lut = np.ones(256, dtype=np.float64)
    lut[VACUUM] = VACUUM_YIELD
    for m in (MATERIAL_LIBRARY.values() if materials is None else materials):
        if m.id != VACUUM:
            lut[m.id] = float(m.se_yield)
    for ref, y in (overrides or {}).items():
        lut[get_material(ref).id] = float(y)
    return lut


def material_topdown_signal(
    height: NDArray[np.float64],
    top_ids: NDArray[np.uint8],
    pixel_size: float,
    cfg: SEMConfig,
    yields: NDArray[np.float64] | None = None,
    height_ref: float | None = None,
) -> NDArray[np.float64]:
    """The noiseless top-down SE signal from a wafer's top surface.

    Parameters
    ----------
    height : (ny, nx)
        Height of the top surface per column [m] — :meth:`Stack.top_height`.
    top_ids : (ny, nx)
        Material ID exposed at the top of each column —
        :meth:`Stack.top_material`. What the beam lands on.
    pixel_size : float
        Column pitch [m]; the field is taken as periodic.
    cfg : SEMConfig
    yields : (256,), optional
        Yield per material ID, from :func:`material_yields`. Defaults to the
        built-in library.
    height_ref : float, optional
        The wall height that scores exactly ``edge_yield``. A resist print
        has a natural one — the film thickness — and a wafer does not, so
        this defaults to the tallest step on the wafer: the largest wall in
        view is as bright as a full-height resist wall would be. Pass the
        same reference across a sequence of steps to keep them comparable.

    A planarised wafer has no walls at all, and images as material contrast
    alone — which is what a top-down of a finished device looks like.
    """
    h = np.asarray(height, dtype=np.float64)
    ids = np.asarray(top_ids)
    if h.ndim != 2 or ids.shape != h.shape:
        raise ValueError(
            f"height and top_ids must both be (ny, nx), got {h.shape} and {ids.shape}"
        )
    lut = material_yields() if yields is None else np.asarray(yields, dtype=np.float64)
    base = lut[ids.astype(np.intp)]
    if height_ref is None:
        height_ref = float(h.max() - h.min())
    h_ref = max(float(height_ref), 1e-12)
    return _topdown_from(base, h, h_ref, pixel_size, cfg)


def material_topdown_sem(
    height: NDArray[np.float64],
    top_ids: NDArray[np.uint8],
    pixel_size: float,
    cfg: SEMConfig | None = None,
    yields: NDArray[np.float64] | None = None,
    height_ref: float | None = None,
    rng: np.random.Generator | None = None,
) -> SEMImage:
    """A top-down SEM frame of a wafer's top surface. See
    :func:`material_topdown_signal`."""
    cfg = cfg or SEMConfig()
    signal = material_topdown_signal(height, top_ids, pixel_size, cfg, yields, height_ref)
    return SEMImage(
        image=_sample(signal, cfg, rng), signal=signal,
        pixel_size=float(pixel_size), row_size=float(pixel_size),
        mode="topdown", electrons=cfg.electrons,
    )


def _trim_headroom(ids: NDArray, headroom: int | None) -> NDArray:
    """Cut a section down to its solid plus a little vacuum above.

    A process stack carries whatever headroom its deposits needed — often
    more than the device is tall — and imaging all of it squashes the films
    into the bottom of the frame. Keep the rows up to the highest solid one
    and :func:`_vacuum_rows` above that (or *headroom* rows, if given),
    padding with vacuum when the stack is shorter than that.
    """
    nz = ids.shape[0]
    solid_rows = np.nonzero((ids != VACUUM).any(axis=1))[0]
    top = int(solid_rows[-1]) + 1 if solid_rows.size else nz
    vac = _vacuum_rows(top) if headroom is None else max(int(headroom), 0)
    want = top + vac
    if want <= nz:
        return ids[:want]
    pad = np.full((want - nz, ids.shape[1]), VACUUM, dtype=ids.dtype)
    return np.concatenate([ids, pad], axis=0)


def material_xsection_signal(
    section: NDArray[np.uint8],
    pixel_size: float,
    dz: float,
    cfg: SEMConfig,
    yields: NDArray[np.float64] | None = None,
    headroom: int | None = None,
) -> NDArray[np.float64]:
    """The noiseless SE signal from a cleaved multi-material face.

    Parameters
    ----------
    section : (nz, n)
        Material IDs on the cut plane, row 0 at the bottom of the wafer —
        :meth:`Stack.cross_section`. The substrate is part of it, so no
        substrate rows are invented under it the way :func:`xsection_signal`
        does for a bare resist slice.
    pixel_size, dz : float
        Column and row pitch [m].
    cfg : SEMConfig
    yields : (256,), optional
        Yield per material ID, from :func:`material_yields`.
    headroom : int, optional
        Vacuum rows drawn above the highest solid. Defaults to a tenth of
        the solid's height; the stack's own headroom beyond that is cut off.

    Material contrast in the bulk, bloom along the solid's outline — top
    surface, trench walls, a released sheet's underside — and none at a
    buried interface, which on a flat face has nothing to emit from.
    """
    ids = np.asarray(section)
    if ids.ndim != 2:
        raise ValueError(f"section must be (nz, n), got shape {ids.shape}")
    ids = _trim_headroom(ids, headroom)
    lut = material_yields() if yields is None else np.asarray(yields, dtype=np.float64)
    mat = lut[ids.astype(np.intp)]
    return _xsection_from(mat, pixel_size, dz, cfg)


def material_xsection_sem(
    section: NDArray[np.uint8],
    pixel_size: float,
    dz: float,
    cfg: SEMConfig | None = None,
    yields: NDArray[np.float64] | None = None,
    headroom: int | None = None,
    rng: np.random.Generator | None = None,
) -> SEMImage:
    """A cross-section SEM frame of a cleaved multi-material face. See
    :func:`material_xsection_signal`. Row 0 is the bottom of the wafer, so
    the extent's z runs up from the substrate's underside."""
    cfg = cfg or SEMConfig()
    signal = material_xsection_signal(section, pixel_size, dz, cfg, yields, headroom)
    return SEMImage(
        image=_sample(signal, cfg, rng), signal=signal,
        pixel_size=float(pixel_size), row_size=float(dz),
        mode="xsection", electrons=cfg.electrons, row_origin=0.0,
    )


def stack_xsection_sem(
    stack: Stack,
    axis: str = "y",
    index: int | None = None,
    cfg: SEMConfig | None = None,
    overrides: Mapping[str | int | Material, float] | None = None,
    headroom: int | None = None,
    rng: np.random.Generator | None = None,
) -> SEMImage:
    """Cleave a wafer stack and image the face.

    ``axis="y"`` cuts at constant y and shows the x–z plane, ``"x"`` the
    y–z plane — :meth:`Stack.cross_section`'s convention — at *index*, or
    the centre. Yields come from the stack's own material table, with
    *overrides* applied on top.
    """
    lut = material_yields(stack.materials.values(), overrides)
    section = stack.cross_section(axis, index)
    return material_xsection_sem(
        section, float(stack.grid.pixel_size), float(stack.dz), cfg,
        yields=lut, headroom=headroom, rng=rng,
    )


def stack_topdown_sem(
    stack: Stack,
    cfg: SEMConfig | None = None,
    overrides: Mapping[str | int | Material, float] | None = None,
    height_ref: float | None = None,
    rng: np.random.Generator | None = None,
) -> SEMImage:
    """Look down on a wafer stack: the top material of every column, and
    the walls between them. See :func:`material_topdown_signal`."""
    lut = material_yields(stack.materials.values(), overrides)
    return material_topdown_sem(
        stack.top_height(), stack.top_material(), float(stack.grid.pixel_size),
        cfg, yields=lut, height_ref=height_ref, rng=rng,
    )


# ---------------------------------------------------------------------------
# Measuring the image
# ---------------------------------------------------------------------------


def _refine_peak(y: NDArray[np.float64], i: int) -> float:
    """Sub-pixel peak position by a parabola through three points."""
    n = y.shape[0]
    if not 0 < i < n - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2.0 * b + c
    if denom >= 0.0:
        return float(i)
    return float(i) + 0.5 * (a - c) / denom


def measure_cd_sem(
    image: NDArray[np.float64],
    pixel_size: float,
    feature: str = "line",
    smooth_px: float = 1.0,
    prominence: float = 0.15,
) -> SEMMeasurement:
    """Measure a CD from a top-down SEM image the way a CD-SEM does.

    Rows are averaged into one signal profile, its edge peaks are found, and
    the CD is the peak-to-peak distance across the requested feature — a
    ``"line"`` (bright interior: resist) or a ``"space"`` (dark interior:
    substrate) — nearest the centre of the field. Peak positions are refined
    to sub-pixel by a parabola.

    The field is treated as periodic, so an edge on the wrap seam is found
    too. Returns a NaN CD when fewer than two edges exist or no feature of the
    requested kind can be told apart — an SEM of a blank wafer has no CD.

    Parameters
    ----------
    image : (ny, nx)
        A top-down frame, noisy or noiseless.
    pixel_size : float
        Column pitch [m].
    feature : str
        ``"line"`` or ``"space"``.
    smooth_px : float
        Gaussian smoothing of the profile before peak finding, in pixels.
        Averaging along y already suppresses most of the shot noise.
    prominence : float
        Minimum peak prominence as a fraction of the profile's range.
    """
    if feature not in ("line", "space"):
        raise ValueError(f"feature must be 'line' or 'space', got {feature!r}")
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"image must be (ny, nx), got shape {img.shape}")
    nx = img.shape[1]
    px = float(pixel_size)
    x = np.arange(nx) * px

    profile = img.mean(axis=0)
    prof = gaussian_filter1d(profile, smooth_px, mode="wrap") if smooth_px > 0 else profile

    empty = SEMMeasurement(float("nan"), float("nan"), float("nan"), feature,
                           profile, np.asarray(x, dtype=np.float64), np.empty(0))
    span = float(prof.max() - prof.min())
    if span <= 0.0:
        return empty

    # Tile three times so a peak straddling the seam is a peak, not an end.
    tiled = np.concatenate([prof, prof, prof])
    peaks, _ = find_peaks(tiled, prominence=prominence * span)
    peaks = peaks[(peaks >= nx) & (peaks < 2 * nx)]
    if peaks.size < 2:
        return empty
    pos = np.array([_refine_peak(tiled, int(i)) for i in peaks]) - nx
    pos.sort()

    # Every segment between consecutive edges, cyclically. Its interior
    # brightness says what it is: resist top reads bright, substrate dark.
    segs = []
    for k in range(pos.size):
        left = pos[k]
        right = pos[(k + 1) % pos.size] + (nx if k == pos.size - 1 else 0)
        lo, hi = int(np.ceil(left)) + 1, int(np.floor(right)) - 1
        if hi < lo:
            continue
        idx = np.arange(lo, hi + 1) % nx
        segs.append((left, right, float(prof[idx].mean())))
    if not segs:
        return empty

    means = np.array([m for _, _, m in segs])
    if means.max() - means.min() < 0.05 * span:
        # Every interior looks the same — there is only one kind of feature
        # here, and we cannot say which. Refuse rather than guess.
        return empty
    cut = 0.5 * (means.max() + means.min())
    want_bright = feature == "line"
    candidates = [
        (left, right) for (left, right, m) in segs
        if (m > cut) == want_bright
    ]
    if not candidates:
        return empty

    centre = 0.5 * nx
    def _dist(seg):
        mid = 0.5 * (seg[0] + seg[1])
        return min(abs(mid - centre), abs(mid - nx - centre), abs(mid + nx - centre))
    left, right = min(candidates, key=_dist)
    return SEMMeasurement(
        cd=(right - left) * px, left=left * px, right=right * px,
        feature=feature, profile=profile, x=np.asarray(x, dtype=np.float64), edges=pos * px,
    )


# ---------------------------------------------------------------------------
# The tilt view: a cleaved sample on a tilted stage
# ---------------------------------------------------------------------------


@dataclass
class GeometryBuffers:
    """What the beam sees of a 3-D surface, pixel by pixel, before any physics.

    A tilt-stage micrograph is an image of a *surface* seen from an angle,
    and the three quantities the SE signal depends on are all geometric:
    which way each visible patch faces, how far along the beam it sits,
    and what it is made of. A renderer with a depth buffer produces all
    three in one pass (:func:`litho_sim.viz.render.render_buffers`), and
    this is the hand-off — plain arrays, so the physics in
    :func:`tilt_signal` stays numpy and testable on synthetic geometry.

    Attributes
    ----------
    normals : (rows, cols, 3)
        Unit surface normal at each pixel, world axes, pointing out of the
        solid. NaN where nothing is seen.
    depth : (rows, cols)
        Distance along the beam to the surface [nm]. NaN where nothing is
        seen — the vacuum.
    material : (rows, cols) uint8
        0 nothing, 1 substrate, 2 resist.
    view : (3,)
        The beam direction, world axes, unit — from the column towards the
        sample. The projection is parallel, as a SEM's is: the scan angles
        at these magnifications are a fraction of a degree.
    up, right : (3,)
        The image's row and column axes in world space, unit.
    focal : (3,)
        The world point [nm] at the image centre.
    pixel_nm : float
        The scan pitch: nanometres per pixel, the same across and down.
    film_nm : float
        The as-coated film thickness [nm], so a full-height silhouette
        blooms exactly ``edge_yield`` — the same reference the top-down
        model uses.
    """

    normals: NDArray[np.float64]
    depth: NDArray[np.float64]
    material: NDArray[np.uint8]
    view: NDArray[np.float64]
    up: NDArray[np.float64]
    right: NDArray[np.float64]
    focal: NDArray[np.float64]
    pixel_nm: float
    film_nm: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape  # type: ignore[return-value]

    @property
    def seen(self) -> NDArray[np.bool_]:
        return self.material > 0

    def project(self, points_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        """World points [nm] → ``(col, row)`` image coordinates, sub-pixel.

        Parallel projection makes this a dot product with the image axes:
        no perspective, so a length on the sample projects to the same
        number of pixels wherever it lies. A scale bar drawn with it is
        measured, not typed.
        """
        p = np.atleast_2d(np.asarray(points_nm, dtype=np.float64)) - self.focal
        rows, cols = self.shape
        col = cols / 2.0 + (p @ self.right) / self.pixel_nm
        row = rows / 2.0 - (p @ self.up) / self.pixel_nm
        return np.column_stack([col, row])


def screen_occlusion(
    depth_nm: NDArray[np.float64],
    pixel_nm: float,
    radii_px: tuple[int, ...] = (2, 4, 8, 16, 32),
    n_directions: int = 8,
    coarsen: int = 2,
) -> NDArray[np.float64]:
    """How much of the sky each surface pixel can see, from the depth buffer.

    Secondary electrons leave the surface and drift to the detector, and
    a floor between two walls loses the ones that hit a wall on the way.
    This is a horizon-based estimate of that: in each of ``n_directions``
    across the image, and at each radius, how far *above* the pixel's
    surface a neighbour sits — a neighbour closer to the beam by ``Δd``
    at a lateral distance ``r`` subtends an elevation ``atan(Δd / r)``.
    The largest elevation in a direction is that direction's horizon, and
    the occlusion is the mean horizon over all directions, 0 for an open
    plane and 1 for a pixel walled in on every side.

    ``depth_nm`` is distance along the beam, NaN in the vacuum. The vacuum
    is treated as infinitely far — it never occludes anything.

    The horizon is a smooth, low-frequency quantity, so it is found on a
    grid ``coarsen`` times coarser than the frame and interpolated back —
    at 2 that is a quarter of the work for no visible difference. The
    radii are in frame pixels either way.
    """
    d_full = np.asarray(depth_nm, dtype=np.float64)
    seen_full = np.isfinite(d_full)
    if not seen_full.any():
        return np.zeros_like(d_full)
    k = max(int(coarsen), 1)
    d = d_full[::k, ::k] if k > 1 else d_full
    seen = np.isfinite(d)
    far = float(np.nanmax(d_full))
    d = np.where(seen, d, far + 1e6)
    px = float(pixel_nm) * k
    radii = sorted({max(int(round(r / k)), 1) for r in radii_px})
    R = max(radii)
    rows, cols = d.shape
    # One padded copy; every neighbour is a slice of it, not a roll.
    dp = np.pad(d, R, mode="edge")
    angles = np.arange(n_directions) * (2.0 * np.pi / n_directions)
    total = np.zeros_like(d)
    for a in angles:
        dx, dy = np.cos(a), np.sin(a)
        best = np.zeros_like(d)                 # the steepest tan(elevation)
        for r in radii:
            sx, sy = int(round(r * dx)), int(round(r * dy))
            if sx == 0 and sy == 0:
                continue
            neighbour = dp[R + sy:R + sy + rows, R + sx:R + sx + cols]
            lateral = np.hypot(sx, sy) * px
            np.maximum(best, (d - neighbour) / lateral, out=best)
        total += np.arctan(best)
    horizon = np.clip(total / (n_directions * (np.pi / 2.0)), 0.0, 1.0)
    horizon = np.where(seen, horizon, 0.0)
    if k > 1:
        from scipy.ndimage import zoom

        full = zoom(horizon, k, order=1, mode="nearest")
        horizon = full[:d_full.shape[0], :d_full.shape[1]]
        if horizon.shape != d_full.shape:       # the coarse grid may fall short by k-1
            pad = ((0, d_full.shape[0] - horizon.shape[0]), (0, d_full.shape[1] - horizon.shape[1]))
            horizon = np.pad(horizon, pad, mode="edge")
    return np.where(seen_full, np.clip(horizon, 0.0, 1.0), 0.0)


def tilt_signal(
    buffers: GeometryBuffers,
    cfg: SEMConfig,
    detector: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """The noiseless SE signal of a tilted, cleaved sample.

    The same three terms as the other two views, applied to a surface seen
    at an angle, plus the two that only exist once there *is* an angle:

    * **Material** — 1 on resist, ``substrate_yield`` on the substrate,
      ``VACUUM_YIELD`` where the beam misses the sample.
    * **Tilt yield** — Seiler's ``sec^n θ`` between the surface normal
      and the beam, capped at grazing incidence where the law diverges but
      the real yield saturates.
    * **Edge bloom** — where the depth jumps between neighbouring pixels
      the beam is crossing a silhouette, and electrons born on the far
      side of it escape through the free surface behind. The bloom is the
      depth crossed per pixel in units of the film thickness, times
      ``edge_yield``, spread over ``escape_length``. A full-height wall
      seen edge-on scores the same ``edge_yield`` the top-down model gives
      it.
    * **Detector side** — the Everhart–Thornley detector sits off-axis, so
      a face turned towards it is brighter than one turned away, by
      ``directionality``.
    * **Shadowing** — a floor between walls sees less of the detector
      (:func:`screen_occlusion`), by ``shadowing``.

    Then the probe blurs everything by ``beam_fwhm``. Units are SE per
    primary with the flat resist top facing the beam at 1, so the same
    ``electrons_per_pixel`` gives the same noise as the other views.

    Parameters
    ----------
    buffers : GeometryBuffers
    cfg : SEMConfig
    detector : (3,), optional
        Direction from the sample towards the detector, world axes. The
        default puts it above and behind the column — up the image and
        towards the beam — which is where a chamber detector sits relative
        to a tilted stage.
    """
    n = np.asarray(buffers.normals, dtype=np.float64)
    depth = np.asarray(buffers.depth, dtype=np.float64)
    material = np.asarray(buffers.material)
    seen = material > 0
    px = float(buffers.pixel_nm)
    if n.shape[:2] != depth.shape or material.shape != depth.shape:
        raise ValueError("normals, depth and material must share the image shape")
    if px <= 0:
        raise ValueError("pixel_nm must be > 0")
    view = np.asarray(buffers.view, dtype=np.float64)
    view = view / np.linalg.norm(view)

    n_safe = np.where(seen[..., None], n, 0.0)
    # The angle between the surface normal and the beam. Outward normals
    # face the column, so -n·v is the cosine; the abs guards a renderer
    # that hands back the inward one.
    cos_t = np.abs(n_safe @ view)
    cos_t = np.where(seen, np.clip(cos_t, 0.08, 1.0), 1.0)
    tilt_yield = np.clip(cos_t ** (-float(cfg.secant_power)), 1.0, 4.0)

    base = np.where(material == 2, 1.0, cfg.substrate_yield)
    base = np.where(seen, base, VACUUM_YIELD)

    # Silhouettes: the depth crossed per pixel, in film thicknesses. The
    # vacuum is infinitely far; a wafer edge against it blooms too, as it
    # does on a real micrograph.
    far = float(np.nanmax(depth[seen])) if seen.any() else 0.0
    d = np.where(seen, depth, far + 4.0 * max(float(buffers.film_nm), 1.0))
    step = np.hypot(np.gradient(d, axis=0), np.gradient(d, axis=1))
    film = max(float(buffers.film_nm), 1e-9)
    bloom = cfg.edge_yield * np.clip(step / film, 0.0, 1.0)
    lam_px = float(cfg.escape_length) * 1e9 / px
    if lam_px > 0:
        bloom = gaussian_filter(bloom, lam_px, mode="nearest")
    bloom = np.where(seen, bloom, 0.0)

    if detector is None:
        up = np.asarray(buffers.up, dtype=np.float64)
        det = up - view
    else:
        det = np.asarray(detector, dtype=np.float64)
    det = det / max(np.linalg.norm(det), 1e-12)
    facing = np.clip(0.5 + 0.5 * (n_safe @ det), 0.0, 1.0)
    side = (1.0 - cfg.directionality) + cfg.directionality * facing

    shade = 1.0 - cfg.shadowing * screen_occlusion(depth, px) if cfg.shadowing > 0 else 1.0

    signal = np.where(seen, base * tilt_yield * (1.0 + bloom) * side * shade, VACUUM_YIELD)
    sigma_px = float(cfg.beam_fwhm) * 1e9 * _FWHM_TO_SIGMA / px
    if sigma_px > 0:
        signal = gaussian_filter(signal, sigma_px, mode="nearest")
    return signal


def tilt_sem(
    buffers: GeometryBuffers,
    cfg: SEMConfig | None = None,
    detector: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> SEMImage:
    """A tilt-stage SEM frame of a rendered surface. See :func:`tilt_signal`."""
    cfg = cfg or SEMConfig()
    signal = tilt_signal(buffers, cfg, detector)
    px = float(buffers.pixel_nm) * 1e-9
    return SEMImage(
        image=_sample(signal, cfg, rng), signal=signal,
        pixel_size=px, row_size=px, mode="tilt", electrons=cfg.electrons,
    )
