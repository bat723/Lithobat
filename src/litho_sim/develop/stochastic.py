"""
Stochastic resist printing: noise as an output, not an input.

Two tools at two levels of honesty:

* :func:`add_edge_roughness` — *cosmetic*. Stamps statistically plausible
  correlated roughness onto an already-developed binary image. Cheap, and
  the right thing when a picture needs texture; wired to
  ``ResistConfig.use_stochastic``. It replaced an edge-flip model whose
  noise was uncorrelated pixel salt — invisible to any CD measurement and
  unphysical on its face: real LER has a correlation length of tens of nm,
  and *correlated* excursions are what bridge lines and pinch contacts.

* :func:`stochastic_trials` — *physical*. Runs the chemically-amplified
  chain end to end with the integers put back in: Poisson photons →
  Poisson PAG and quencher molecules → binomial acid conversion
  (:func:`~litho_sim.expose.photochem.sample_species`) → the same
  reaction–diffusion bake the deterministic ``"car"`` model uses → Mack
  development. Nothing here injects noise; every rough edge and every failed
  feature is a consequence of counting statistics upstream. Run it at ArF
  and the trials all print alike; run it at EUV and they don't — same code,
  same dose, fourteen times fewer photons.

The second one is the reason this module exists. LER, LWR, LCDU and
stochastic failure rates are *measurements over trials* — see
:mod:`litho_sim.analysis.stochastics` — not parameters anyone chose.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter

from litho_sim.bake.reaction import bake_reaction_diffusion
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.develop.resist import cleared_depth
from litho_sim.develop.resist3d import (
    apply_absorption,
    apply_vertical_interference,
    arrival_field,
    exposure_volume,
    latent_volume,
)
from litho_sim.expose.photochem import SpeciesSample, sample_species, sample_species_3d

logger = logging.getLogger(__name__)

__all__ = [
    "correlated_noise",
    "dissolution_scatter",
    "add_edge_roughness",
    "stochastic_trials",
    "StochasticResult",
    "stochastic_trials_3d",
    "StochasticProfileResult",
]


def correlated_noise(
    shape: tuple[int, ...],
    corr_length_px: float | Sequence[float],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Unit-variance Gaussian noise with a chosen correlation length.

    White noise filtered by a Gaussian of width ``ξ/2`` has autocorrelation
    ``exp(−r²/ξ²)`` — down to 1/e at exactly ``r = ξ``, which is the
    convention roughness papers quote. Renormalised to unit variance after
    filtering, so the caller's σ means what it says regardless of ξ. Wrap
    boundaries match the periodic simulation grid.

    *corr_length_px* may be one number or one per axis, for a volume whose
    voxels are not cubes. A correlation length under a milli-pixel returns
    plain white noise.
    """
    white = rng.standard_normal(shape)
    xi = np.atleast_1d(np.asarray(corr_length_px, dtype=np.float64))
    if np.all(xi < 1e-3):
        return white
    sigma = (float(xi[0]) / 2.0) if xi.size == 1 else tuple(float(x) / 2.0 for x in xi)
    filtered = gaussian_filter(white, sigma=sigma, mode="wrap")
    std = float(filtered.std())
    return filtered / std if std > 0.0 else filtered


def dissolution_scatter(
    shape: tuple[int, ...],
    cfg: ResistConfig,
    spacing: Sequence[float],
    rng: np.random.Generator,
    column_height: float | None = None,
) -> NDArray[np.float64] | None:
    """A per-trial multiplier on the dissolution rate, or ``None`` when off.

    Log-normal with unit mean, ``exp(σξ − σ²/2)``, ξ correlated over
    ``cfg.dissolution_corr_length`` along every axis of *spacing* [m per
    voxel]. Development noise: the granularity of a polymer film dissolving,
    which no count upstream describes and which is what gives an ArF line
    its roughness. See ``ResistConfig.dissolution_sigma``.

    ``dissolution_sigma`` is the scatter of one granule, a region one
    correlation length across. What a voxel sees is the average over the
    granules it spans: a 2-D full-thickness voxel (*column_height* given)
    averages ``H/ξ`` of them down its column, and a 3-D voxel taller than
    ξ averages ``dz/ξ``, so σ is reduced by the square root of that count.
    Applying the granule scatter to a whole column instead put factor-of-six
    tails on the cleared depth and cut islands into the spaces.
    """
    sigma = float(cfg.dissolution_sigma)
    if sigma <= 0.0:
        return None
    xi = float(cfg.dissolution_corr_length)
    span = float(column_height) if column_height is not None else float(spacing[0])
    if span > xi:
        sigma *= np.sqrt(xi / span)
    xi_px = [xi / float(h) for h in spacing]
    field = correlated_noise(shape, xi_px, rng)
    return np.exp(sigma * field - 0.5 * sigma * sigma)


def add_edge_roughness(
    resist: NDArray[np.float64],
    sigma: float,
    corr_length: float,
    pixel_size: float,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Displace the edges of a binary image by correlated noise.

    Works in signed-distance space: every point knows how far it is from the
    nearest edge, a correlated noise field ``u`` with standard deviation
    *sigma* is added, and the zero level set is re-extracted. Because the
    distance function changes by one metre per metre, the edge moves by
    exactly ``u`` — so *sigma* IS the 1-σ edge displacement, on any geometry,
    with no per-feature bookkeeping. Displacements share the noise field's
    correlation length along the edge, which is what makes the result look
    like a SEM image instead of transmission static.

    Parameters
    ----------
    resist : NDArray
        Binary resist image (0/1).
    sigma : float
        1-σ edge displacement [m].  Non-positive returns the input unchanged.
    corr_length : float
        1/e correlation length of the displacement along the edge [m].
    pixel_size : float
        Physical pixel size [m].
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Binary resist image with rough edges.
    """
    if sigma <= 0.0:
        return resist
    binary = resist > 0.5
    if binary.all() or not binary.any():
        # No edges to roughen — and the distance transform of a uniform
        # field is not meaningful anyway.
        return resist

    rng = np.random.default_rng(seed)
    # EDT measures to the nearest opposite *pixel centre*, so boundary cells
    # read 1.0; shifting by half a pixel puts the interface itself at zero.
    d_in = distance_transform_edt(binary)
    d_out = distance_transform_edt(~binary)
    sdf = np.where(binary, d_in - 0.5, -(d_out - 0.5))

    u = correlated_noise(resist.shape, corr_length / pixel_size, rng)
    u *= sigma / pixel_size
    rough = (sdf + u) > 0.0

    logger.debug(
        "edge roughness: σ=%.1f nm, ξ=%.0f nm, changed %.2f%% of pixels",
        sigma * 1e9, corr_length * 1e9,
        100.0 * float(np.mean(rough != binary)),
    )
    return rough.astype(np.float64)


@dataclass
class StochasticResult:
    """A batch of stochastic printing trials of the same exposure.

    Attributes
    ----------
    resist : NDArray
        Developed binary images, ``(trials, ny, nx)``. What the topology
        metrics (bridges, breaks) reduce over — and only those. A binary
        edge sits on a half-pixel, so any roughness read off it is the grid's,
        not the resist's: at 4 nm pixels it hides a 1.3 nm 3σ LER entirely,
        at 2 nm it reports its own 1.7 nm quantisation floor, and it destroys
        the correlation length at every pixel size.
    depth : NDArray
        The cleared develop depth of each trial [nm], ``(trials, ny, nx)`` —
        the continuous field the binary image is a level set of
        (:func:`~litho_sim.develop.resist.cleared_depth`). Resist remains
        where ``depth < level``. Measured with sub-pixel crossings, this
        gives roughness that converges as the grid is refined and a
        correlation length that lands on the acid diffusion length; every
        roughness statistic should be read from it.
    level : float
        The film thickness [nm] — the level ``depth`` is measured against.
    protected : NDArray
        Post-bake protected fraction of the *first* trial — the latent image
        one realisation actually developed from, kept for inspection.
    sample : SpeciesSample
        The first trial's sampled species, carrying ``mean_pag`` and
        ``mean_photons`` — the two counts that decide how noisy the whole
        experiment was.
    """

    resist: NDArray[np.float64]
    depth: NDArray[np.float64]
    level: float
    protected: NDArray[np.float64]
    sample: SpeciesSample

    @property
    def trials(self) -> int:
        return int(self.resist.shape[0])

    #: Which side of ``level`` the resist is on in ``depth``: it remains where
    #: the developer cleared less than the film.
    feature: str = "below"


def stochastic_trials(
    aerial: NDArray[np.float64],
    cfg: ResistConfig,
    grid: GridConfig,
    wavelength: float,
    dose: float = 1.0,
    trials: int = 16,
    seed: int | None = None,
) -> StochasticResult:
    """Print the same aerial image *trials* times through sampled chemistry.

    Each trial draws its own photons, PAG, and quencher
    (:func:`~litho_sim.expose.photochem.sample_species`), bakes them through
    the full acid/quencher reaction–diffusion, and develops the protected
    fraction — exactly the deterministic ``model="car"`` pipeline of
    :func:`~litho_sim.develop.resist.simulate_resist`, except the initial
    conditions are one realisation instead of the mean.  Trial-to-trial
    variation is therefore pure counting statistics.

    The bake is the cost: expect roughly the deterministic ``"car"`` runtime
    times *trials*.  For process-window-style exploration, drop the grid to
    64–96 px first.

    Parameters
    ----------
    aerial : NDArray
        Normalised aerial image, ``(ny, nx)``.
    cfg : ResistConfig
        Chemistry and process parameters — the CAR block plus the Mack
        develop parameters.
    grid : GridConfig
        Supplies ``pixel_size``.
    wavelength : float
        Exposure wavelength [m], from ``OpticsConfig`` — it sets the photon
        energy, and with it the entire scale of the shot noise.
    dose : float
        Relative dose multiplier, same convention as ``simulate_resist``.
    trials : int
        Number of independent realisations.
    seed : int, optional
        Seed for the run.  One generator is drawn through sequentially, so a
        fixed seed reproduces the whole batch.

    Returns
    -------
    StochasticResult
    """
    rng = np.random.default_rng(seed)
    out = np.empty((trials, *aerial.shape), dtype=np.float64)
    depth = np.empty((trials, *aerial.shape), dtype=np.float64)
    first_sample: SpeciesSample | None = None
    first_protected: NDArray[np.float64] | None = None

    for i in range(trials):
        species = sample_species(
            aerial, cfg, wavelength, grid.pixel_size, dose=dose, rng=rng
        )
        baked = bake_reaction_diffusion(
            species.acid,
            grid.pixel_size,
            cfg.bake_time,
            cfg.D_acid,
            quencher=species.quencher,
            D_quencher=cfg.D_quencher,
            k_quench=cfg.k_quench,
            k_loss=cfg.k_loss,
            k_amp=cfg.k_amp,
        )
        d = cleared_depth(baked["protected"], cfg)
        scatter = dissolution_scatter(
            d.shape, cfg, (grid.pixel_size, grid.pixel_size), rng, column_height=cfg.thickness
        )
        if scatter is not None:
            d = d * scatter
        depth[i] = d
        out[i] = (d < cfg.thickness * 1e9).astype(np.float64)
        if i == 0:
            first_sample = species
            first_protected = baked["protected"]

    assert first_sample is not None and first_protected is not None
    logger.info(
        "stochastic_trials: %d trials, %.0f photons and %.0f PAG per voxel "
        "(mean), λ=%.1f nm",
        trials, first_sample.mean_photons, first_sample.mean_pag,
        wavelength * 1e9,
    )
    return StochasticResult(
        resist=out, depth=depth, level=float(cfg.thickness * 1e9),
        protected=first_protected, sample=first_sample,
    )


# ---------------------------------------------------------------------------
# Through the film: stochastic profiles
# ---------------------------------------------------------------------------


@dataclass
class StochasticProfileResult:
    """A batch of stochastic 3-D prints of the same exposure.

    Attributes
    ----------
    arrival : NDArray
        Arrival time of the develop front at every voxel, per trial,
        ``(trials, nz, ny, nx)`` [s], ``iz = 0`` at the substrate — the
        continuous field every profile statistic reads with sub-voxel
        crossings. Resist remains where it exceeds ``level``.
    level : float
        The develop time [s].
    remaining : NDArray[bool]
        ``arrival > level``, per trial: the developed solids.
    latent : NDArray
        The first trial's protected-fraction volume after the bake.
    sample : SpeciesSample
        The first trial's sampled species, with the per-voxel counts.
    z : NDArray
        Height of each plane above the substrate [m].
    """

    arrival: NDArray[np.float64]
    level: float
    remaining: NDArray[np.bool_]
    latent: NDArray[np.float64]
    sample: SpeciesSample
    z: NDArray[np.float64]
    feature: str = "above"

    @property
    def trials(self) -> int:
        return int(self.arrival.shape[0])


def stochastic_trials_3d(
    mask: NDArray,
    optics: OpticsConfig,
    grid: GridConfig,
    resist: ResistConfig,
    dose: float = 1.0,
    trials: int = 8,
    seed: int | None = None,
    develop_model: str = "mack",
    standing_waves: bool = False,
    bleaching: bool = True,
) -> StochasticProfileResult:
    """Print the same exposure *trials* times through sampled chemistry, in 3-D.

    The chain of :func:`stochastic_trials` with the film resolved in depth:
    one exposure volume; per trial, photons and molecules drawn in every
    voxel (:func:`~litho_sim.expose.photochem.sample_species_3d`), the 3-D
    acid/quencher reaction–diffusion bake, the Mack rate law and the
    arrival-time field of the chosen develop model
    (:func:`~litho_sim.develop.resist3d.arrival_field`). Trial-to-trial
    variation is counting statistics; what it produces here that the 2-D
    model cannot is roughness that varies with height, top-loss scatter,
    footing, and bridges at the substrate.

    Cost is the 3-D bake per trial. At 48 × 48 × 13 voxels a trial is well
    under a second; at the default 128 px grid and 2 nm planes it is tens
    of seconds, so start coarse.

    Parameters
    ----------
    mask, optics, grid, resist
        The process; the resist's CAR block (``pag_density``,
        ``quencher_ratio``, ``D_acid``, ``k_*``) and Mack parameters are
        what run.
    dose : float
        Relative dose.
    trials, seed
        Batch size and reproducibility, as for :func:`stochastic_trials`.
    develop_model : str
        ``"mack"`` (ray march) or ``"front"`` (eikonal, slower).
    standing_waves, bleaching
        Forwarded to the exposure chain.
    """
    if develop_model not in ("mack", "front"):
        raise ValueError(
            f"stochastic_trials_3d needs a finite-rate develop model, got {develop_model!r}"
        )
    rng = np.random.default_rng(seed)
    intensity, z = exposure_volume(mask, optics, grid, resist, dose=dose)
    if standing_waves:
        intensity = apply_vertical_interference(intensity, resist, optics, grid)
    # The local incident dose the sampler draws photons against: the
    # deterministic absorption chain, once per batch.
    _pac, exposure = apply_absorption(intensity, resist, grid.dz, dose=1.0, bleaching=bleaching)

    arrival = np.empty((trials, *intensity.shape), dtype=np.float64)
    first_sample: SpeciesSample | None = None
    first_latent: NDArray[np.float64] | None = None
    level = float(resist.develop_time)
    for i in range(trials):
        species = sample_species_3d(
            exposure, resist, optics.wavelength, grid.pixel_size, grid.dz, rng=rng
        )
        latent = latent_volume(
            intensity, resist, grid, bake="car", bleaching=bleaching, species=species
        )["latent"]
        scatter = dissolution_scatter(
            latent.shape, resist, (grid.dz, grid.pixel_size, grid.pixel_size), rng
        )
        T, level, _feature = arrival_field(
            latent, resist, grid, model=develop_model, rate_scale=scatter
        )
        arrival[i] = T
        if i == 0:
            first_sample, first_latent = species, latent

    assert first_sample is not None and first_latent is not None
    logger.info(
        "stochastic_trials_3d: %d trials on %s voxels, %.1f photons and %.1f PAG "
        "per voxel (mean), develop %s",
        trials, intensity.shape, first_sample.mean_photons, first_sample.mean_pag,
        develop_model,
    )
    return StochasticProfileResult(
        arrival=arrival, level=level, remaining=arrival > level,
        latent=first_latent, sample=first_sample, z=np.asarray(z, dtype=np.float64),
    )
