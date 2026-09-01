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
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter

from litho_sim.bake.reaction import bake_reaction_diffusion
from litho_sim.core.config import GridConfig, ResistConfig
from litho_sim.develop.resist import _mack_binary
from litho_sim.expose.photochem import SpeciesSample, sample_species

logger = logging.getLogger(__name__)

__all__ = [
    "correlated_noise",
    "add_edge_roughness",
    "stochastic_trials",
    "StochasticResult",
]


def correlated_noise(
    shape: tuple[int, ...],
    corr_length_px: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Unit-variance Gaussian noise with a chosen correlation length.

    White noise filtered by a Gaussian of width ``ξ/2`` has autocorrelation
    ``exp(−r²/ξ²)`` — down to 1/e at exactly ``r = ξ``, which is the
    convention roughness papers quote. Renormalised to unit variance after
    filtering, so the caller's σ means what it says regardless of ξ. Wrap
    boundaries match the periodic simulation grid.

    A correlation length under a milli-pixel returns plain white noise.
    """
    white = rng.standard_normal(shape)
    if corr_length_px < 1e-3:
        return white
    filtered = gaussian_filter(white, sigma=corr_length_px / 2.0, mode="wrap")
    std = float(filtered.std())
    return filtered / std if std > 0.0 else filtered


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
        Developed binary images, ``(trials, ny, nx)``.  The axis every
        statistic in :mod:`litho_sim.analysis.stochastics` reduces over.
    protected : NDArray
        Post-bake protected fraction of the *first* trial — the latent image
        one realisation actually developed from, kept for inspection.
    sample : SpeciesSample
        The first trial's sampled species, carrying ``mean_pag`` and
        ``mean_photons`` — the two counts that decide how noisy the whole
        experiment was.
    """

    resist: NDArray[np.float64]
    protected: NDArray[np.float64]
    sample: SpeciesSample

    @property
    def trials(self) -> int:
        return int(self.resist.shape[0])


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
        out[i] = _mack_binary(baked["protected"], cfg)
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
        resist=out, protected=first_protected, sample=first_sample
    )
