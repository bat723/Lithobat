"""
Exposure photochemistry: from an aerial image to an acid field.

Everything that happens *during* the exposure lives here — photon absorption
and acid generation — in two forms that are the same physics at different
resolutions:

* :func:`generate_acid` is the mean field: the smooth acid concentration a
  dose produces, the continuum limit the rest of the deterministic pipeline
  consumes.
* :func:`sample_species` is the same chemistry with the integers put back in.
  A voxel does not absorb ``38.7`` photons; it absorbs a Poisson draw around
  that, converts a binomial share of an integer number of PAG molecules, and
  sits next to an integer number of quencher molecules. Those counts are the
  *entire* origin of stochastic printing failures — no noise parameter is
  added anywhere, and the fluctuations vanish as counts grow, which is why
  the same model is quiet at ArF and loud at EUV.

Why EUV is the stochastic regime
--------------------------------
Dose is energy per area, so the photon count per area is ``dose / E_photon``.
A 13.5 nm photon carries 92 eV against 6.4 eV at 193 nm — fourteen times the
energy, so at equal dose a fourteenth the photons. Absorb ~20 % of them in a
thin film and a 30 mJ/cm² exposure leaves a (4 nm)² EUV voxel with tens of
absorbed photons where the ArF voxel has thousands. Relative shot noise goes
as ``1/sqrt(N)``: a few percent at DUV, tens of percent at EUV. That ratio is
the whole story, and it falls out of the constants rather than being asserted.

Units
-----
Doses are mJ/cm² (the industry unit, and what ``ResistConfig.dose_nominal``
holds); concentrations are dimensionless in units of the initial PAG loading,
which is what :func:`~litho_sim.bake.reaction.bake_reaction_diffusion` expects
and what makes ``acid = 1`` mean "every PAG converted".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from litho_sim.core.config import ResistConfig

logger = logging.getLogger(__name__)

__all__ = [
    "photon_energy",
    "absorbed_fraction",
    "mean_absorbed_photons",
    "mean_absorbed_photons_3d",
    "generate_acid",
    "generate_acid_3d",
    "sample_species",
    "sample_species_3d",
    "acids_per_photon",
    "SpeciesSample",
]

_PLANCK = 6.62607015e-34  # [J·s]
_C_LIGHT = 2.99792458e8   # [m/s]
#: 1 mJ/cm² expressed in SI energy density.
_MJ_CM2_TO_J_M2 = 10.0


def photon_energy(wavelength: float) -> float:
    """Energy of one photon at *wavelength* [J].

    The number that makes EUV stochastic: ``hc/λ`` is 1.03e-18 J at ArF and
    1.47e-17 J at EUV, so equal doses deliver 14x fewer photons at 13.5 nm.
    """
    return _PLANCK * _C_LIGHT / wavelength


def absorbed_fraction(resist: ResistConfig) -> float:
    """Fraction of incident light the film absorbs, from the Dill terms.

    Beer–Lambert through the full thickness with the *unbleached* absorbance
    ``A + B`` (Dill coefficients are per µm). Bleaching would reduce A's share
    as exposure proceeds, but CARs barely bleach and at EUV ``A = 0`` outright,
    so the unbleached value is the honest constant. This is a film-integrated
    number — the 2-D model's voxels span the full thickness, so depth
    variation of the absorption is deliberately averaged out.
    """
    alpha_per_m = (resist.dill_A + resist.dill_B) * 1e6
    return float(1.0 - np.exp(-alpha_per_m * resist.thickness))


def mean_absorbed_photons(
    aerial: NDArray[np.float64],
    resist: ResistConfig,
    wavelength: float,
    pixel_size: float,
    dose: float = 1.0,
) -> NDArray[np.float64]:
    """Expected absorbed photons per full-thickness voxel, ``(ny, nx)``.

    Parameters
    ----------
    aerial : NDArray
        Normalised aerial image intensity.
    resist : ResistConfig
        Supplies ``dose_nominal``, thickness and the Dill absorbance.
    wavelength : float
        Exposure wavelength [m] — from ``OpticsConfig``, not the resist.
    pixel_size : float
        Lateral voxel size [m].
    dose : float
        Relative dose multiplier on top of ``dose_nominal``, same convention
        as everywhere else in the engine.
    """
    energy_density = (
        np.clip(aerial, 0.0, None) * dose * resist.dose_nominal * _MJ_CM2_TO_J_M2
    )
    per_area = energy_density / photon_energy(wavelength)
    return per_area * pixel_size * pixel_size * absorbed_fraction(resist)


def _electron_blur(
    field: NDArray[np.float64],
    resist: ResistConfig,
    pixel_size: float,
    dz: float | None = None,
) -> NDArray[np.float64]:
    """Spread deposited energy over the secondary-electron range.

    Applied to the *photon/energy* field, before acid conversion: the cascade
    happens between absorption and acid generation, so it blurs where the
    acid appears, not the acid after the fact. Wrap boundaries match the
    periodic FFT grid the aerial image lives on; a 3-D volume (``dz`` given,
    depth on axis 0) is not periodic in depth and clamps there instead.
    """
    sigma_px = resist.electron_blur_sigma / pixel_size
    if sigma_px < 1e-3:
        return field
    if dz is None:
        return gaussian_filter(field, sigma=sigma_px, mode="wrap")
    return gaussian_filter(
        field,
        sigma=(resist.electron_blur_sigma / dz, sigma_px, sigma_px),
        mode=("nearest", "wrap", "wrap"),
    )


def mean_absorbed_photons_3d(
    exposure: NDArray[np.float64],
    resist: ResistConfig,
    wavelength: float,
    pixel_size: float,
    dz: float,
) -> NDArray[np.float64]:
    """Expected absorbed photons per voxel of a depth-resolved exposure.

    *exposure* is the local incident dose at every voxel [mJ/cm²] — the
    ``E`` :func:`~litho_sim.develop.resist3d.apply_absorption` returns, already
    attenuated by the film above — so each voxel absorbs its own slice of it:
    ``E · α · dz`` per unit area with the unbleached absorbance ``A + B``,
    the same constant the 2-D model integrates through the whole film.
    """
    alpha_per_m = (resist.dill_A + resist.dill_B) * 1e6
    energy = np.clip(exposure, 0.0, None) * _MJ_CM2_TO_J_M2
    per_area = energy / photon_energy(wavelength)
    return per_area * pixel_size * pixel_size * (1.0 - np.exp(-alpha_per_m * dz))


def generate_acid(
    aerial: NDArray[np.float64],
    resist: ResistConfig,
    pixel_size: float,
    dose: float = 1.0,
) -> NDArray[np.float64]:
    """Mean-field acid concentration after exposure, in units of PAG₀.

    First-order photolysis: a PAG molecule survives an exposure ``E`` with
    probability ``exp(−C·E)``, so the converted fraction is::

        h(x) = 1 − exp(−C · E(x)),   E = aerial × dose × dose_nominal

    — the same Dill C the classic path uses, acting on the same exposure, so
    the CAR chain and the ``"mack"`` chain agree about what the light did and
    differ only in what the chemistry does next. This is exactly the
    ``N → ∞`` limit of :func:`sample_species`, a correspondence the test
    suite asserts rather than assumes.
    """
    exposure = np.clip(aerial, 0.0, None) * dose * resist.dose_nominal
    exposure = _electron_blur(exposure, resist, pixel_size)
    return 1.0 - np.exp(-resist.dill_C * exposure)


def generate_acid_3d(
    exposure: NDArray[np.float64],
    resist: ResistConfig,
    pixel_size: float,
    dz: float,
) -> NDArray[np.float64]:
    """Mean-field acid per voxel from a depth-resolved exposure, in units of PAG₀.

    The 3-D counterpart of :func:`generate_acid`: the same first-order
    photolysis on the local incident dose at each voxel, with the electron
    blur applied through the volume. This is what the 3-D reaction–diffusion
    bake starts from.
    """
    blurred = _electron_blur(np.clip(exposure, 0.0, None), resist, pixel_size, dz)
    return 1.0 - np.exp(-resist.dill_C * blurred)


@dataclass
class SpeciesSample:
    """One stochastic realisation of the post-exposure chemistry.

    Concentration fields are in units of the mean initial PAG concentration,
    ready for :func:`~litho_sim.bake.reaction.bake_reaction_diffusion`.

    Attributes
    ----------
    acid, quencher : NDArray
        Sampled concentration fields, ``(ny, nx)``.
    photons : NDArray
        Absorbed photons per voxel actually drawn (after electron blur, so
        fractional values are possible at EUV).
    mean_pag : float
        Expected PAG molecules per voxel — the number whose smallness makes
        the resist stochastic.
    mean_photons : float
        Spatial mean of the expected absorbed-photon field, for the same
        at-a-glance judgement about shot noise.
    """

    acid: NDArray[np.float64]
    quencher: NDArray[np.float64]
    photons: NDArray[np.float64]
    mean_pag: float
    mean_photons: float


def sample_species(
    aerial: NDArray[np.float64],
    resist: ResistConfig,
    wavelength: float,
    pixel_size: float,
    dose: float = 1.0,
    rng: np.random.Generator | None = None,
) -> SpeciesSample:
    """Draw one realisation of photon absorption and acid generation.

    The chain, per full-thickness voxel:

    1. **Photons** — ``P ~ Poisson(n̄_ph)``: arrival statistics of light.
       The draw is then spread over the electron-cascade range at EUV.
    2. **PAG** — ``N ~ Poisson(n̄_pag)``: molecules were dissolved in a
       spin-coated film, so their count per voxel is Poisson too.
    3. **Acid** — ``Binomial(N, 1 − exp(−C·E_loc))`` with the *local, sampled*
       exposure ``E_loc = E × P/n̄_ph``: each of the N molecules independently
       survives the photon flux that actually arrived. Saturation is free —
       a voxel cannot yield more acid than it has PAG.
    4. **Quencher** — ``Poisson(quencher_ratio × n̄_pag)``, independent of
       exposure; base is formulated in, not generated.

    The sampled exposure field is unbiased, and the whole chain converges to
    :func:`generate_acid` as the counts grow — so LER, LCDU and failure rates
    measured over many draws are attributable to counting statistics alone.
    (At finite counts the *mean converted acid* sits slightly below the
    deterministic value: conversion is concave in exposure, so shot noise
    genuinely costs a little sensitivity. That is Jensen's inequality doing
    chemistry, not a bug.)
    """
    rng = np.random.default_rng() if rng is None else rng
    n_mean = mean_absorbed_photons(aerial, resist, wavelength, pixel_size, dose)
    exposure = np.clip(aerial, 0.0, None) * dose * resist.dose_nominal
    mean_pag = resist.pag_density * pixel_size * pixel_size * resist.thickness
    return _sample_counts(n_mean, exposure, resist, mean_pag, pixel_size, None, rng)


def sample_species_3d(
    exposure: NDArray[np.float64],
    resist: ResistConfig,
    wavelength: float,
    pixel_size: float,
    dz: float,
    rng: np.random.Generator | None = None,
) -> SpeciesSample:
    """Draw one realisation of the chemistry through the film depth.

    The chain of :func:`sample_species`, per voxel of a ``(nz, ny, nx)``
    volume: *exposure* is the local incident dose the film above lets
    through (:func:`~litho_sim.develop.resist3d.apply_absorption`'s ``E``),
    each voxel absorbs its own ``α · dz`` slice of it, holds
    ``pag_density × dx² × dz`` molecules, and converts them against the
    photons that actually arrived. The counts per voxel are an order of
    magnitude smaller than the 2-D model's full-thickness voxels — a 4 nm
    EUV voxel holds about 19 PAG molecules — which is the resolution at
    which vertical roughness, top-loss scatter and footing become
    consequences of counting rather than assumptions.

    Concentrations come out in units of the mean initial PAG per voxel,
    ready for the 3-D :func:`~litho_sim.bake.reaction.bake_reaction_diffusion`
    with ``spacing=(dz, dx, dx)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_mean = mean_absorbed_photons_3d(exposure, resist, wavelength, pixel_size, dz)
    mean_pag = resist.pag_density * pixel_size * pixel_size * dz
    return _sample_counts(
        n_mean, np.clip(exposure, 0.0, None), resist, mean_pag, pixel_size, dz, rng
    )


def acids_per_photon(resist: ResistConfig, wavelength: float) -> float:
    """Conversion events per absorbed photon the resist's constants imply.

    Events per volume are ``C · E · PAG₀`` (the Dill exponent times the
    loading); absorbed photons per volume are ``E · α / E_photon``. The dose
    cancels: ``φ = C · PAG₀ · E_photon / (10 α)`` with ``α = A + B`` per
    metre and the 10 converting mJ/cm² to J/m². A property of the resist at
    a wavelength, not of the pattern or the grid — about 5.5 for the EUV
    preset, where a 92 eV photon's cascade converts several PAG molecules.
    The DUV CAR preset comes out at 2.5, which one 6.4 eV photon cannot do;
    that is the preset's Dill C being high for its absorbance, and the
    number is logged so it can be seen.
    """
    alpha_per_m = (resist.dill_A + resist.dill_B) * 1e6
    if alpha_per_m <= 0.0:
        return float("nan")
    return float(
        resist.dill_C * resist.pag_density * photon_energy(wavelength)
        / (_MJ_CM2_TO_J_M2 * alpha_per_m)
    )


def _sample_counts(
    n_mean: NDArray[np.float64],
    exposure: NDArray[np.float64],
    resist: ResistConfig,
    mean_pag: float,
    pixel_size: float,
    dz: float | None,
    rng: np.random.Generator,
) -> SpeciesSample:
    """The counting chain shared by the 2-D and 3-D samplers.

    *n_mean* is the expected absorbed-photon count per voxel and *exposure*
    the deterministic local dose [mJ/cm²], on the same grid; *dz* is given
    for a volume so the electron blur knows its depth spacing.
    """
    photons = rng.poisson(n_mean).astype(np.float64)
    photons = _electron_blur(photons, resist, pixel_size, dz)

    # The sampled field rescales the deterministic exposure pointwise: where
    # the Poisson draw came in 20 % hot, the local dose is 20 % hot.
    exposure = _electron_blur(exposure, resist, pixel_size, dz)
    with np.errstate(invalid="ignore", divide="ignore"):
        blurred_mean = _electron_blur(n_mean, resist, pixel_size, dz)
        local = np.where(blurred_mean > 0.0, photons / blurred_mean, 0.0)
    exposure_loc = exposure * local

    n_pag = rng.poisson(mean_pag, size=exposure.shape)
    if resist.acid_yield_noise:
        # Acid is made in *events*, not by a smooth exposure: each absorbed
        # photon sets off a cascade that converts a random number of PAG
        # molecules — at EUV several, through the secondary electrons. The
        # expected number of conversion events per voxel is C·E·N̄ (that is
        # exactly the Dill exponent), so the events per absorbed photon are
        # φ = C·E·N̄ / n̄, a constant of the resist and wavelength; the
        # events actually fired are Poisson about φ times the photons that
        # actually arrived, and the fraction of PAG they reach is
        # 1 − exp(−k/N̄), as for hits with replacement. The mean is the
        # deterministic conversion; the variance carries both the photon
        # count and the yield per photon, which the old binomial on a
        # rescaled exposure left out.
        with np.errstate(invalid="ignore", divide="ignore"):
            events_mean = resist.dill_C * exposure_loc * mean_pag
            n_events = rng.poisson(np.clip(events_mean, 0.0, None))
        p_convert = np.clip(1.0 - np.exp(-n_events / mean_pag), 0.0, 1.0)
    else:
        p_convert = np.clip(1.0 - np.exp(-resist.dill_C * exposure_loc), 0.0, 1.0)
    n_acid = rng.binomial(n_pag, p_convert)
    n_quencher = rng.poisson(resist.quencher_ratio * mean_pag, size=exposure.shape)

    if mean_pag < 10.0:
        logger.warning(
            "sample_species: %.2f PAG molecules per voxel — the grid is so "
            "fine (or the loading so low) that single-molecule noise "
            "dominates everything. Check pag_density and pixel_size.",
            mean_pag,
        )
    return SpeciesSample(
        acid=n_acid / mean_pag,
        quencher=n_quencher / mean_pag,
        photons=photons,
        mean_pag=float(mean_pag),
        mean_photons=float(n_mean.mean()),
    )
