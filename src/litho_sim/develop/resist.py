"""
Resist process simulation module.

Implements the standard photoresist pipeline for both positive and negative
tone resists:

1. **Exposure**          – Dill photochemical model (PAC bleaching)
2. **Post-Exposure Bake** – acid diffusion via Gaussian blur, or full CAR
   acid/quencher reaction–diffusion (``model="car"``)
3. **Development**       – Mack 4-parameter or simple threshold model
4. **Stochastic LER**    – optional correlated edge roughness

The physical stochastic route — photon shot noise and molecular counting
propagated through the chemistry, rather than roughness stamped on at the
end — is :func:`litho_sim.develop.stochastic.stochastic_trials`.

Physical models
---------------
*Dill exposure* (thin-resist approximation)::

    M(x) = exp(−C · I(x) · dose)

where M is the photoactive compound (PAC) concentration, I is the
normalised aerial image, and C is the exposure rate constant.

*Mack development rate*::

    a  = [(n+1)/(n−1)] · (1 − M_th)ⁿ
    R  = R_max · (a+1)(1−M)ⁿ / [a + (1−M)ⁿ]  +  R_min
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from litho_sim.bake.peb import apply_peb
from litho_sim.core.config import GridConfig, ResistConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 – Exposure (Dill model)
# ---------------------------------------------------------------------------


def dill_exposure(
    aerial: NDArray[np.float64],
    dose: float,
    dill_C: float,
) -> NDArray[np.float64]:
    """Simulate resist exposure using the Dill thin-resist approximation.

    Parameters
    ----------
    aerial : NDArray
        Normalised aerial image intensity [0, 1], shape ``(ny, nx)``.
    dose : float
        Normalised exposure dose (dimensionless multiplier).
    dill_C : float
        Dill C coefficient – exposure rate constant.

    Returns
    -------
    NDArray[np.float64]
        PAC concentration M(x, y) after exposure, range [0, 1].
        M = 1 → unexposed; M → 0 → fully exposed.
    """
    E = np.clip(aerial, 0.0, None) * dose
    M = np.exp(-dill_C * E)
    M = np.clip(M, 0.0, 1.0)
    logger.debug(
        "Dill exposure: dose=%.3f, C=%.4f, M ∈ [%.4f, %.4f]",
        dose, dill_C, float(M.min()), float(M.max()),
    )
    return M.astype(np.float64)


# ---------------------------------------------------------------------------
# Step 2 – Post-Exposure Bake
# ---------------------------------------------------------------------------
#
# Diffusion belongs to the bake step: :func:`litho_sim.bake.peb.apply_peb`
# (Gaussian) and :func:`litho_sim.bake.reaction.bake_reaction_diffusion`
# (acid/quencher reaction–diffusion). :func:`_bake` below is the one place
# this module runs exposure and bake together.
#
# ---------------------------------------------------------------------------
# Step 3 – Development
# ---------------------------------------------------------------------------


def mack_development_rate(
    latent_image: NDArray[np.float64],
    Rmax: float,
    Rmin: float,
    Mth: float,
    n: int,
) -> NDArray[np.float64]:
    """Compute dissolution rate using the Mack 4-parameter model.

    Parameters
    ----------
    latent_image : NDArray
        PAC concentration M(x, y) after PEB, values in [0, 1].
    Rmax, Rmin : float
        Maximum / minimum dissolution rates [nm/s].
    Mth : float
        Threshold PAC concentration (0–1).
    n : int
        Contrast exponent (≥ 2).

    Returns
    -------
    NDArray[np.float64]
        Development rate R(x, y) [a.u. same units as Rmax/Rmin].
    """
    M = np.clip(latent_image, 0.0, 1.0)
    a = ((n + 1) / max(n - 1, 1e-9)) * (1.0 - Mth) ** n
    one_m_M_n = (1.0 - M) ** n
    R = Rmax * ((a + 1.0) * one_m_M_n) / (a + one_m_M_n) + Rmin
    return R.astype(np.float64)


def surface_inhibition(
    rate: NDArray[np.float64],
    dz_nm: float,
    depth_nm: float,
    relative_rate: float,
) -> NDArray[np.float64]:
    r"""Slow the dissolution rate in the top few nanometres of the film.

    The one place in development where the transport really is rate-limiting.
    Everywhere else the developer equilibrates ~1e5 times faster than the
    interface moves (Damköhler ~1e-5), so the rate depends only on the frozen
    PAC and the front has a prescribed speed. But solvent entering *unswollen,
    glassy* polymer has ``D`` around 1e-16 m²/s, which puts ``Da`` near 100 —
    genuinely transport limited, and the origin of the induction period.

    Rather than solve a second moving-boundary problem inside the first one,
    this follows the literature and absorbs it into an empirical depth
    dependence, Mack's surface-inhibition term:

    .. math::

        R_{\mathrm{eff}}(z) = R(M) \left[
            1 - (1 - r_0)\, e^{-z / \delta} \right]

    where ``z`` is measured **down from the top surface**. It is a fitted
    stand-in, calibrated against dissolution-rate-monitor data, not derived.

    This is what produces T-topping — but only in company. A slow cap alone
    changes nothing under the vertical ray march, because every column simply
    burns through it and the profile shifts down uniformly. It takes a front
    that can also move *sideways*, once through, to eat out from under the cap
    and leave the overhang. See :mod:`litho_sim.develop.front`.

    Parameters
    ----------
    rate : NDArray
        ``(nz, ...)`` dissolution rate, ``iz = 0`` at the substrate.
    dz_nm : float
        Voxel height [nm].
    depth_nm : float
        Inhibition depth ``δ`` [nm]. Zero or less disables the term entirely
        and returns the rate unchanged.
    relative_rate : float
        Rate at the very top surface as a fraction of the bulk rate, ``r₀``.
        1.0 is no inhibition; 0.05 is a strongly quenched surface.

    Returns
    -------
    NDArray
        The modified rate field.
    """
    if depth_nm <= 0.0 or relative_rate >= 1.0:
        return rate
    nz = rate.shape[0]
    # Depth measured downward from the top surface, which is the *last* index.
    z_from_top = (nz - 1 - np.arange(nz)) * dz_nm
    factor = 1.0 - (1.0 - relative_rate) * np.exp(-z_from_top / depth_nm)
    return rate * factor.reshape((nz,) + (1,) * (rate.ndim - 1))


def cleared_depth(
    latent: NDArray[np.float64],
    cfg: ResistConfig,
) -> NDArray[np.float64]:
    """Depth the developer clears at every point, in nanometres.

    ``rate × develop_time`` under the Mack rate law, evaluated on the
    *protection* the developer meets: the latent itself for a positive
    resist, and its complement for a negative one, where exposure is what
    makes the polymer insoluble. Resist remains wherever this falls short of
    the film thickness — in both tones — so this is the continuous field
    every binary developed image, every 2-D profile and every stochastic
    trial is a level set of.

    Parameters
    ----------
    latent : NDArray
        Post-bake latent image, ``(ny, nx)``: PAC after the Gaussian bake,
        or the protected fraction after the reaction–diffusion bake; 1 means
        unexposed in either.
    cfg : ResistConfig
        Supplies the Mack rate parameters, ``develop_time`` and ``tone``.

    Returns
    -------
    NDArray[np.float64]
        Cleared depth [nm], ``>= 0``, uncapped.
    """
    protection = latent if cfg.tone == "positive" else 1.0 - np.asarray(latent)
    rate = mack_development_rate(
        protection, cfg.mack_Rmax, cfg.mack_Rmin, cfg.mack_Mth, cfg.mack_n
    )
    return rate * cfg.develop_time


def remaining_thickness(
    latent: NDArray[np.float64],
    cfg: ResistConfig,
) -> NDArray[np.float64]:
    """How much resist is left at every point, in nanometres.

    The 2-D counterpart of a profile, and it costs nothing extra: the Mack path
    in :func:`simulate_resist` already computes the depth the developer clears
    and then throws the number away on the next line, keeping only whether it
    exceeded the film. That reduces a field with dozens of distinct levels —
    rounded tops, sloped edges, partial clearing — to one bit.

    Returned in nanometres rather than as a fraction so it can be plotted
    against the film thickness directly, and so a reader can see top loss for
    what it is.

    Note this is the *vertical ray* approximation: each column develops on its
    own, so the result is a height field and cannot represent an undercut. For
    that, the depth-resolved path with ``develop_model="front"`` is the one to
    use — see :mod:`litho_sim.develop.front`.

    Parameters
    ----------
    latent : NDArray
        Aerial or post-bake latent image, ``(ny, nx)``.
    cfg : ResistConfig
        Supplies the Mack rate parameters, ``develop_time`` and ``thickness``.
        The relative dose is already in *latent*; the Dill exposure here
        applies only the nominal dose that turns it into real units.

    Returns
    -------
    NDArray[np.float64]
        Remaining thickness [nm], clipped to ``[0, thickness]``.
    """
    pac = dill_exposure(latent, cfg.dose_nominal, cfg.dill_C)
    thickness_nm = cfg.thickness * 1e9
    return np.clip(thickness_nm - cleared_depth(pac, cfg), 0.0, thickness_nm)


def just_clearing_time(cfg: ResistConfig) -> float:
    """Seconds for the developer to clear the film at its maximum rate [s].

    The natural unit for ``develop_time``: a process is usually run at a small
    multiple of this, and quoting a raw number of seconds hides whether that
    multiple is 2 or 30. The engine's own default was **30x** this value, which
    is why every Mack profile came out as an over-developed dome rather than a
    line with a flat top.
    """
    return float(cfg.thickness * 1e9 / max(cfg.mack_Rmax, 1e-12))


def threshold_development(
    latent_image: NDArray[np.float64],
    threshold: float,
    tone: str = "positive",
) -> NDArray[np.float64]:
    """Binary threshold development model.

    Parameters
    ----------
    latent_image : NDArray
        PAC concentration or normalised aerial image.
    threshold : float
        Development threshold (0-1).
    tone : str
        "positive" - resist clears (=0) where intensity >= threshold.
        "negative" - resist remains (=1) where intensity >= threshold.

    Returns
    -------
    NDArray[np.float64]
        Binary resist image: 1 = resist remaining, 0 = cleared.
    """
    if tone == "positive":
        return np.where(latent_image >= threshold, 0.0, 1.0)
    elif tone == "negative":
        return np.where(latent_image >= threshold, 1.0, 0.0)
    else:
        raise ValueError(f"tone must be 'positive' or 'negative', got '{tone}'")

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def _bake(
    aerial: NDArray[np.float64],
    cfg: ResistConfig,
    grid: GridConfig,
    model: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exposure and bake for the chemistry models — the one copy of the chain.

    Returns ``(after_exposure, after_bake)``, both scaled so that 1 means
    "unexposed": PAC after Dill exposure and after the Gaussian bake for
    ``"mack"``; unconverted PAG and the protected fraction after the
    reaction–diffusion bake for ``"car"``. :func:`simulate_resist` binarises
    the second field, :func:`develop_field` turns it into a develop depth;
    neither re-derives the chemistry, so the two cannot drift apart.

    The relative dose is already in *aerial* (it is applied by
    :func:`~litho_sim.expose.aerial_image.compute_aerial_image`); only the
    nominal dose that converts it to real units is applied here.
    """
    if model == "mack":
        pac = dill_exposure(aerial, cfg.dose_nominal, cfg.dill_C)
        return pac, apply_peb(pac, cfg.diffusion_sigma, grid.pixel_size)
    if model == "car":
        from litho_sim.bake.reaction import bake_reaction_diffusion
        from litho_sim.expose.photochem import generate_acid

        # The bake diffuses in 2-D, so the whole field is baked and any cut
        # taken afterwards — cutting first would turn lateral diffusion off.
        acid = generate_acid(aerial, cfg, grid.pixel_size, dose=1.0)
        baked = bake_reaction_diffusion(
            acid,
            grid.pixel_size,
            cfg.bake_time,
            cfg.D_acid,
            quencher=cfg.quencher_ratio,
            D_quencher=cfg.D_quencher,
            k_quench=cfg.k_quench,
            k_loss=cfg.k_loss,
            k_amp=cfg.k_amp,
        )
        return 1.0 - acid, baked["protected"]
    raise ValueError(
        f"Unknown resist model: '{model}'. Choose 'threshold', 'mack' or 'car'."
    )


def _mack_binary(
    latent: NDArray[np.float64], cfg: ResistConfig
) -> NDArray[np.float64]:
    """Develop a latent image to a binary resist map under the Mack model.

    *latent* is whatever field plays the role of "how protected is this
    voxel" — PAC after a Gaussian bake, or the protected fraction after the
    CAR bake. Both develop by the same rule: the developer clears
    ``rate × develop_time`` of depth, and resist survives where that fails
    to reach the substrate.

    Both tones go through :func:`cleared_depth`: the developer meets the
    protection it meets — the latent for a positive resist, its complement
    for a negative one, where exposure is what makes the polymer insoluble
    — and resist remains wherever the depth it clears falls short of the
    film. Until 2026-09-05 negative tone was the positive result inverted,
    which is not a rate law and disagreed with the 2-D profile and the 3-D
    developer (audit finding 13); all three now say the same thing.
    """
    cleared_nm = cleared_depth(latent, cfg)
    return (cleared_nm < cfg.thickness * 1e9).astype(np.float64)


def simulate_resist(
    aerial: NDArray[np.float64],
    cfg: ResistConfig,
    grid: GridConfig,
    model: str = "threshold",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the full resist pipeline: exposure → PEB → development → LER.

    Parameters
    ----------
    aerial : NDArray
        Aerial image intensity, shape ``(ny, nx)``.  The relative dose applied
        by :func:`~litho_sim.expose.aerial_image.compute_aerial_image` is
        already in it, and nothing here applies one again — the chemistry
        only converts it to real units with ``cfg.dose_nominal``.
    cfg : ResistConfig
        Resist chemistry and process parameters.
    grid : GridConfig
        Simulation grid (supplies ``pixel_size`` for the PEB kernel).
    model : str
        ``"threshold"`` – binarise the aerial image directly at
        ``cfg.threshold``.  Fast, and what the interactive app has always
        used in practice.
        ``"mack"`` – physical route: Dill exposure → PEB diffusion →
        Mack 4-parameter dissolution, developed for ``cfg.develop_time``.
        ``"car"`` – chemically amplified route: Dill acid generation →
        acid/quencher reaction–diffusion with catalytic deprotection
        (:func:`litho_sim.bake.reaction.bake_reaction_diffusion`) → Mack
        dissolution of the *protected* fraction.  The quencher is what makes
        this differ from ``"mack"``: sub-threshold acid is annihilated
        rather than blurred, so dose response is asymmetric and contrast
        comes from chemistry rather than the Mack exponent alone.

    Returns
    -------
    (pac_exp, pac_peb, resist) : tuple of NDArray
        PAC concentration after exposure, after PEB, and the developed resist
        image (1 = resist remaining, 0 = cleared).  On the ``"threshold"``
        path the first two are copies of *aerial*, since that model never
        enters PAC space.  On the ``"car"`` path they are the analogous
        chemistry fields — unconverted PAG after exposure, protected fraction
        after the bake — chosen so 1 means "unexposed" in every model.

    Raises
    ------
    ValueError
        For an unrecognised *model*.
    """
    if model == "threshold":
        # Threshold is defined in intensity space, not PAC space, so Dill/PEB
        # are bypassed. pac_exp/pac_peb echo the aerial image for API symmetry.
        resist = threshold_development(aerial, cfg.threshold, cfg.tone)
        pac_exp = pac_peb = aerial.copy()
    else:
        pac_exp, pac_peb = _bake(aerial, cfg, grid, model)
        resist = _mack_binary(pac_peb, cfg)

    if cfg.use_stochastic:
        from litho_sim.develop.stochastic import add_edge_roughness

        resist = add_edge_roughness(
            resist,
            cfg.stochastic_sigma,
            cfg.stochastic_corr_length,
            grid.pixel_size,
        )

    logger.debug(
        "Resist (%s, %s): %.1f%% of area remaining",
        model, cfg.tone, 100.0 * float(resist.mean()),
    )
    return pac_exp, pac_peb, resist


def develop_field(
    aerial: NDArray[np.float64],
    cfg: ResistConfig,
    grid: GridConfig,
    model: str = "threshold",
) -> tuple[NDArray[np.float64], float]:
    """The continuous field the printed edge is a level set of, and its level.

    Every model in this engine ends the same way: some scalar field crosses
    some threshold, and the crossing is the resist edge. This returns that
    ``(field, threshold)`` pair without binarising, so a caller can locate
    the edge *between* pixels — the sub-pixel measurement that
    :func:`litho_sim.analysis.process_window._measure_point` needs for a
    Bossung curve and that OPC needs for an edge placement error. Both go
    through here so there is exactly one definition of "what printed".

    The field always increases with exposure: the bright side of a mask
    edge is the side where ``field > threshold``, whatever the model and
    whichever the tone. For a negative resist under the chemistry models
    the developer clears *less* where the light was, so the field returned
    is the film thickness minus the cleared depth against a level of zero —
    rising with exposure, resist where it is positive.

    Parameters
    ----------
    aerial : NDArray
        Aerial image, ``(ny, nx)``, dose already applied.
    cfg : ResistConfig
        Resist chemistry and process parameters.
    grid : GridConfig
        Supplies ``pixel_size`` for the bake kernels.
    model : str
        ``"threshold"`` — the PEB-diffused aerial image against
        ``cfg.threshold``.  At ``diffusion_sigma = 0`` this is the aerial
        image itself.
        ``"mack"`` — Dill exposure → PEB → Mack rate; the field is the depth
        the developer clears [nm] against the film thickness.
        ``"car"`` — acid generation → reaction–diffusion bake → Mack rate on
        the protected fraction; same depth field, same threshold.

    Returns
    -------
    (field, threshold) : tuple
        The field, same shape as *aerial*, and the level the resist edge
        sits at, in the field's own units.

    Raises
    ------
    ValueError
        For an unrecognised *model*.
    """
    if model == "threshold":
        return apply_peb(aerial, cfg.diffusion_sigma, grid.pixel_size), float(cfg.threshold)

    _, latent = _bake(aerial, cfg, grid, model)
    depth = cleared_depth(latent, cfg)
    thickness_nm = float(cfg.thickness * 1e9)
    if cfg.tone == "positive":
        return depth, thickness_nm
    return thickness_nm - depth, 0.0


# ---------------------------------------------------------------------------
# CD measurement helpers
# ---------------------------------------------------------------------------


def feature_edges(
    profile: NDArray[np.float64],
    threshold: float = 0.5,
    *,
    feature: str = "above",
) -> tuple[float, float] | None:
    """Sub-pixel (rise, fall) indices of the centre feature of a 1-D profile.

    Finds all threshold crossings by linear interpolation between the
    bracketing samples, pairs each rising edge with the next falling one, and
    returns the pair whose centre is closest to the midpoint of the array —
    or ``None`` when no closed feature exists, which is itself a result: on
    a resist cutline it means the line broke.

    This is the primitive under :func:`measure_cd_1d` (CD is the difference
    of the two numbers) and under the LER measurement in
    :mod:`litho_sim.analysis.stochastics` (edge position row by row *is* the
    roughness signal).

    Parameters
    ----------
    profile : NDArray
        1-D profile — binary resist, aerial intensity, or any continuous
        field the threshold is defined against.
    threshold : float
        Threshold for edge detection, in the same units as *profile*.
    feature : str
        ``"above"`` finds the region above threshold (a bright feature, or
        remaining resist in a binary image); ``"below"`` the region under it.

    Returns
    -------
    tuple of float, or None
        Fractional indices ``(x_rise, x_fall)`` of the feature's edges.
    """
    p = np.asarray(profile, dtype=np.float64)
    thr = float(threshold)
    if feature == "below":
        # Negation preserves interpolated crossing positions exactly.
        p, thr = -p, -thr
    elif feature != "above":
        raise ValueError(f"feature must be 'above' or 'below', got '{feature}'")

    above = (p > thr).astype(int)
    diff = np.diff(above)
    rising = np.where(diff == 1)[0]   # crossing inside gap (i, i+1), going up
    falling = np.where(diff == -1)[0]

    if rising.size == 0 or falling.size == 0:
        return None

    def crossing(i: int) -> float:
        """Fractional index where the profile crosses *thr* in gap (i, i+1)."""
        p0, p1 = p[i], p[i + 1]
        if p1 == p0:
            return float(i) + 0.5
        t = (thr - p0) / (p1 - p0)
        return float(i) + float(np.clip(t, 0.0, 1.0))

    # Pair each rising edge with the next falling edge
    features: list[tuple[float, float]] = []
    for r in rising:
        after = falling[falling > r]
        if after.size > 0:
            features.append((crossing(int(r)), crossing(int(after[0]))))

    if not features:
        return None

    # Select the feature closest to the array centre
    centre = len(p) / 2.0
    feat_centres = np.array([(x_r + x_f) / 2.0 for x_r, x_f in features])
    idx = int(np.argmin(np.abs(feat_centres - centre)))
    return features[idx]


def measure_cd_1d(
    profile: NDArray[np.float64],
    pixel_size: float,
    threshold: float = 0.5,
    *,
    feature: str = "above",
) -> float:
    """Measure the CD of the centre feature from a 1-D profile.

    The width of the feature :func:`feature_edges` selects, in metres.

    On a binary profile every crossing interpolates to exactly half a pixel
    past the gap index, so the result reproduces the historical integer-pixel
    answer bit-for-bit. On a continuous profile (an aerial cutline, a
    develop-depth field) the crossing lands sub-pixel, which is what makes
    CD-through-focus resolvable below the 4 nm grid quantum.

    Parameters
    ----------
    profile : NDArray
        1-D profile — binary resist, aerial intensity, or any continuous
        field the threshold is defined against.
    pixel_size : float
        Physical pixel size [m].
    threshold : float
        Threshold for edge detection, in the same units as *profile*.
    feature : str
        Forwarded to :func:`feature_edges` — ``"above"`` or ``"below"``.

    Returns
    -------
    float
        Measured CD [m].  Returns 0.0 if no closed feature is found.
    """
    edges = feature_edges(profile, threshold, feature=feature)
    if edges is None:
        return 0.0
    x_rise, x_fall = edges
    return float((x_fall - x_rise) * pixel_size)


def measure_cd_2d(
    resist_image: NDArray[np.float64],
    pixel_size: float,
    axis: int = 1,
    threshold: float = 0.5,
    *,
    feature: str = "above",
) -> float:
    """Measure CD from the centre cross-section of a 2-D resist image.

    Parameters
    ----------
    resist_image : NDArray
        2-D resist image — binary or continuous (see :func:`measure_cd_1d`).
    pixel_size : float
        Physical pixel size [m].
    axis : int
        Axis perpendicular to the feature orientation.
        1 → extract the centre *row* (vary across x, for vertical lines).
        0 → extract the centre *column* (vary across y, for horizontal lines).
    threshold : float
        Edge detection threshold.
    feature : str
        Forwarded to :func:`measure_cd_1d` — ``"above"`` or ``"below"``.

    Returns
    -------
    float
        CD [m].
    """
    n = resist_image.shape[1 - axis]
    midpoint = n // 2
    profile = resist_image[midpoint, :] if axis == 1 else resist_image[:, midpoint]
    return measure_cd_1d(profile, pixel_size, threshold, feature=feature)

