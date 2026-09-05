"""
Depth-resolved (3-D) resist exposure and development.

The 2-D engine in :mod:`litho_sim.develop.resist` collapses the resist to a single
plane, so a printed feature is a binary footprint with no sidewall, no top
loss, and no thickness. This module resolves the film through its depth and
produces an actual solid.

Pipeline
--------
1. **Depth-resolved aerial image** — the projected image is not the same at
   every depth in the film, because each depth sits at a different defocus.
   :func:`exposure_volume` sweeps
   :func:`~litho_sim.expose.aerial_image.compute_aerial_image` through the film and
   interpolates onto the voxel grid.
2. **Absorption and bleaching** — :func:`apply_absorption` integrates the
   coupled Dill equations down the film::

       ∂I/∂z = −(A·M + B)·I        (absorption, strongest where PAC survives)
       ∂M/∂t = −C·I·M              (bleaching, clears the way for later light)

   This is what makes the top of the film see more dose than the bottom, and
   therefore what produces a sloped sidewall rather than a vertical cut.
3. **Standing waves** — :func:`add_standing_waves` interferes the downward
   wave with its reflection off the substrate, giving the classic
   scalloped sidewall with period λ/(2·n_resist).
4. **Post-exposure bake** — a 3-D Gaussian, which is also what washes the
   standing waves back out (the real reason PEB exists).
5. **Development** — :func:`develop_3d` thresholds the latent image and then
   keeps only material the developer can actually *reach* from the top.

Units
-----
Dill ``A`` and ``B`` are in µm⁻¹ (the convention the published values use);
``C`` is in cm²/mJ and pairs with a dose in mJ/cm². Everything else is SI.

Coordinates
-----------
Returned volumes are indexed ``[iz, iy, ix]`` with **iz = 0 at the bottom of
the resist**, matching :class:`~litho_sim.wafer.stack.Stack` so they can be written
straight into a wafer stack.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label

from litho_sim.bake.peb import apply_peb_3d
from litho_sim.bake.reaction import bake_reaction_diffusion
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.develop.front import arrival_time_front
from litho_sim.develop.resist import mack_development_rate, surface_inhibition
from litho_sim.expose.aerial_image import compute_aerial_planes
from litho_sim.wafer import VACUUM, get_material

logger = logging.getLogger(__name__)

_MICRON = 1e-6


# ---------------------------------------------------------------------------
# Step 1 – depth-resolved aerial image
# ---------------------------------------------------------------------------


def focus_reference_depth(resist: ResistConfig) -> float:
    """Depth below the resist top that ``OpticsConfig.defocus`` refers to [m]."""
    ref = resist.focus_reference
    if ref == "top":
        return 0.0
    if ref == "mid":
        return resist.thickness / 2.0
    if ref == "bottom":
        return resist.thickness
    raise ValueError(
        f"focus_reference must be 'top', 'mid', or 'bottom', got '{ref}'"
    )


def effective_defocus(
    depth: float, optics: OpticsConfig, resist: ResistConfig
) -> float:
    """Defocus to use when imaging a plane *depth* metres below the resist top.

    Moving the observation plane deeper into the film moves it *away* from
    focus, but the film's own refractive index stretches the optical path:
    a physical step of ``Δd`` inside a medium of index ``n_resist`` costs the
    same wavefront error as a step of ``Δd·n_immersion/n_resist`` in the
    medium the scanner's focus is calibrated in.

    Parameters
    ----------
    depth : float
        Depth below the resist top surface [m], positive downward.
    optics : OpticsConfig
        Supplies the nominal defocus and immersion index.
    resist : ResistConfig
        Supplies the resist index and which plane focus refers to.

    Returns
    -------
    float
        Defocus [m] to hand to :func:`~litho_sim.expose.aerial_image.compute_aerial_image`.
    """
    d_ref = focus_reference_depth(resist)
    return optics.defocus - (depth - d_ref) * (optics.n_immersion / resist.n_resist)


def exposure_volume(
    mask: NDArray,
    optics: OpticsConfig,
    grid: GridConfig,
    resist: ResistConfig,
    dose: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the aerial image throughout the resist film.

    Each optical plane is a full Abbe sum, so the count is kept small
    (``grid.n_z_slices``) and the result is interpolated onto the finer voxel
    grid.  The image varies smoothly with defocus, so this costs very little
    accuracy — and the fast-varying part (standing waves) is applied
    analytically afterwards rather than sampled.

    Parameters
    ----------
    mask : NDArray
        2-D mask transmittance (real or complex).
    optics, grid, resist : config objects
        Optical system, numerical grid, and resist film parameters.
    dose : float
        Relative exposure dose.

    Returns
    -------
    (intensity, z) : tuple of NDArray
        ``intensity`` has shape ``(nz, ny, nx)`` with ``iz = 0`` at the
        **bottom** of the resist; ``z`` holds the corresponding heights above
        the resist bottom [m].
    """
    nz = max(int(round(resist.thickness / grid.dz)), 1)
    n_slices = int(np.clip(grid.n_z_slices, 2, nz))

    # Optical planes, expressed as depth below the resist top.
    depths = np.linspace(0.0, resist.thickness, n_slices)
    planes = np.empty((n_slices, grid.n_pixels, grid.n_pixels), dtype=np.float64)

    # The image forms *inside* the resist, so that is the index the imaging
    # geometry must use — it is what sets the ray angle, sin θ = NA/n_resist
    # rather than NA/n_immersion, and therefore the whole size of the vector
    # effect. Refraction into the denser film is what keeps hyper-NA
    # printable at all.
    #
    # `effective_defocus` returns a defocus already expressed in
    # immersion-equivalent units (that is what its n_immersion/n_resist factor
    # is for), so declaring the image medium to be resist means converting the
    # defocus into resist units too. Without the conversion the index would be
    # applied twice and every out-of-focus plane would be wrong.
    index_ratio = resist.n_resist / optics.n_immersion

    # One pass through the source for every plane: the mask spectrum and each
    # source point's pupil geometry are shared, only the defocus phase differs.
    planes[:] = compute_aerial_planes(
        mask, optics, grid,
        [(effective_defocus(float(d), optics, resist) * index_ratio, resist.n_resist)
         for d in depths],
        dose=dose,
    )

    # Interpolate onto the voxel grid, then flip to bottom-up (Stack order).
    z_vox_depth = (np.arange(nz) + 0.5) * grid.dz  # depth below top
    z_vox_depth = z_vox_depth[::-1]  # so index 0 ends up at the bottom
    flat = planes.reshape(n_slices, -1)
    out = np.empty((nz, flat.shape[1]), dtype=np.float64)
    for k in range(flat.shape[1]):
        out[:, k] = np.interp(z_vox_depth, depths, flat[:, k])
    intensity = out.reshape(nz, grid.n_pixels, grid.n_pixels)

    z = (np.arange(nz) + 0.5) * grid.dz
    logger.info(
        "Exposure volume: %d voxels deep from %d optical planes, "
        "I ∈ [%.3f, %.3f]",
        nz, n_slices, float(intensity.min()), float(intensity.max()),
    )
    return np.asarray(intensity, dtype=np.float64), np.asarray(z, dtype=np.float64)


# ---------------------------------------------------------------------------
# Step 2 – absorption with bleaching
# ---------------------------------------------------------------------------


def apply_absorption(
    intensity: NDArray[np.float64],
    resist: ResistConfig,
    dz: float,
    dose: float = 1.0,
    n_time_steps: int = 16,
    bleaching: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate the coupled Dill exposure equations through the film depth.

    Absorption and bleaching are coupled: absorbed light destroys photoactive
    compound, which reduces absorption, which lets later light reach deeper.
    Marching in time captures that; a single static exponential does not.

    Parameters
    ----------
    intensity : NDArray
        Depth-resolved aerial image ``(nz, ny, nx)``, bottom-up.
    resist : ResistConfig
        Supplies Dill A/B [µm⁻¹], C [cm²/mJ], and ``dose_nominal`` [mJ/cm²].
    dz : float
        Voxel height [m].
    dose : float
        Relative dose multiplier.
    n_time_steps : int
        Exposure sub-steps.  16 is ample; the update is exact for frozen
        intensity, so this only resolves the coupling.
    bleaching : bool
        Set False to freeze absorption at its unbleached value.

    Returns
    -------
    (M, E) : tuple of NDArray
        Final photoactive-compound concentration in [0, 1] and cumulative
        absorbed exposure dose, both ``(nz, ny, nx)``.
    """
    A = resist.dill_A / _MICRON  # µm⁻¹ → m⁻¹
    B = resist.dill_B / _MICRON
    C = resist.dill_C
    total_dose = dose * resist.dose_nominal

    M = np.ones_like(intensity)
    E = np.zeros_like(intensity)
    dt = 1.0 / max(n_time_steps, 1)

    for _ in range(max(n_time_steps, 1)):
        alpha = A * M + B if bleaching else np.full_like(M, A + B)
        # Optical depth accumulated from the top surface downward. The array
        # is bottom-up, so reverse, accumulate, and reverse back.
        tau = np.cumsum(alpha[::-1] * dz, axis=0)[::-1]
        # Exclude each voxel's own absorption from the light reaching it.
        tau -= alpha * dz
        I_local = intensity * np.exp(-tau)
        dE = I_local * total_dose * dt
        if bleaching:
            M *= np.exp(-C * dE)
        E += dE

    if not bleaching:
        M = np.exp(-C * E)
    np.clip(M, 0.0, 1.0, out=M)
    logger.debug(
        "Absorption: M ∈ [%.3f, %.3f], top/bottom dose ratio %.2f",
        float(M.min()), float(M.max()),
        float(E[-1].mean() / max(E[0].mean(), 1e-12)),
    )
    return M, E


# ---------------------------------------------------------------------------
# Step 3 – standing waves
# ---------------------------------------------------------------------------


def add_standing_waves(
    intensity: NDArray[np.float64],
    resist: ResistConfig,
    optics: OpticsConfig,
    dz: float,
) -> NDArray[np.float64]:
    """Modulate the intensity by substrate-reflection standing waves.

    The downward-travelling wave interferes with its own reflection off the
    substrate, producing a vertical intensity fringe of period
    ``λ / (2·n_resist)`` — about 57 nm for 193 nm light in a 1.7-index
    resist.  It is why unbaked resist sidewalls look scalloped, and why
    bottom anti-reflective coatings exist.

    Setting ``resist.substrate_reflectance`` to 0 (a perfect BARC) makes this
    a no-op. The reflectance is a magnitude; the reflection carries the π
    phase of a wave meeting a denser medium (silicon under resist), so the
    interface is a **node** — the standing wave's dark plane sits at the
    substrate, which is where the foot of a positive resist line comes from.
    Until 2026-09-05 the phase was omitted and the substrate was an
    antinode, inverting the polarity of the footing failure (audit finding
    14); the transfer-matrix path had it right, and the two now agree.

    Parameters
    ----------
    intensity : NDArray
        ``(nz, ny, nx)`` intensity, bottom-up.
    resist, optics : config objects
        Supply the resist index, substrate reflectance, and wavelength.
    dz : float
        Voxel height [m].

    Returns
    -------
    NDArray
        Modulated intensity, same shape.
    """
    r = float(resist.substrate_reflectance)
    if abs(r) < 1e-9:
        return intensity

    nz = intensity.shape[0]
    n = resist.n_resist
    k = 2.0 * np.pi * n / optics.wavelength
    thickness = nz * dz
    z = (np.arange(nz) + 0.5) * dz  # height above the resist bottom

    # Downward wave has travelled (thickness - z); the reflected wave has
    # travelled (thickness - z) + 2z. Their interference gives the fringe.
    down = np.exp(-1j * k * (thickness - z))
    # −r: the reflection off a denser substrate flips the field.
    up = -r * np.exp(-1j * k * (thickness + z))
    envelope = np.abs(down + up) ** 2
    envelope /= envelope.mean()  # preserve the dose, only redistribute it

    period = optics.wavelength / (2.0 * n)
    logger.debug(
        "Standing waves: period %.1f nm (%.1f voxels), contrast %.2f",
        period * 1e9, period / dz,
        (envelope.max() - envelope.min()) / (envelope.max() + envelope.min()),
    )
    return intensity * envelope[:, None, None]


# ---------------------------------------------------------------------------
# Step 4 – post-exposure bake
# ---------------------------------------------------------------------------


def tmm_standing_waves(
    intensity: NDArray[np.float64],
    resist: ResistConfig,
    optics: OpticsConfig,
    grid: GridConfig,
) -> NDArray[np.float64]:
    """Modulate the intensity by transfer-matrix interference through the stack.

    The rigorous alternative to :func:`add_standing_waves`. Where that model
    has one scalar reflectance, this one solves the full multilayer problem —
    so a bottom anti-reflective coating can be *designed* (thickness and
    absorption tuned until its two reflections cancel) rather than asserted.

    The envelope is averaged over the illumination's angular distribution:
    every source point sits at a different incidence angle, hence a slightly
    different fringe period, and summing them is what partially washes the
    fringe out in a real exposure.

    Parameters
    ----------
    intensity : NDArray
        ``(nz, ny, nx)`` intensity, bottom-up.
    resist : ResistConfig
        Supplies ``film_stack``, ``n_resist``, and the Dill coefficients from
        which the resist's extinction is derived.
    optics : OpticsConfig
        Supplies wavelength, NA, immersion index and the source description.
    grid : GridConfig
        Supplies ``dz``.

    Returns
    -------
    NDArray
        Modulated intensity, normalised so total dose is unchanged.
    """
    from litho_sim.coat.films import film_stack_from_records, resist_index_from_dill
    from litho_sim.expose.illumination import build_source

    nz = intensity.shape[0]
    thickness = nz * grid.dz
    n_res = resist_index_from_dill(
        resist.n_resist, resist.dill_A, resist.dill_B, optics.wavelength
    )
    stack = film_stack_from_records(
        resist.film_stack, thickness, n_res,
        ambient_n=complex(optics.n_immersion, 0.0),
    )

    # Angular distribution of the illumination, as (sigma, weight) pairs.
    src = build_source(
        n_pixels=max(int(optics.source_grid), 3),
        source_type=optics.source_type,
        sigma_outer=optics.sigma_outer,
        sigma_inner=optics.sigma_inner,
        spec=optics.source_spec,
        **(optics.source_kwargs or {}),
    )
    ns = src.shape[0]
    c = np.linspace(-1.0, 1.0, ns)
    xi, eta = np.meshgrid(c, c)
    rho = np.sqrt(xi ** 2 + eta ** 2)

    z = (np.arange(nz) + 0.5) * grid.dz
    envelope = np.zeros(nz, dtype=np.float64)
    total_w = 0.0
    for r, w in zip(rho[src > 0].ravel(), src[src > 0].ravel()):
        sin_theta = np.clip(r * optics.NA / optics.n_immersion, 0.0, 0.999)
        envelope += w * stack.field_profile(
            optics.wavelength, np.asarray(z, dtype=np.float64),
            theta0=float(np.arcsin(sin_theta))
        )
        total_w += w
    if total_w > 0:
        envelope /= total_w
    if envelope.mean() > 0:
        envelope /= envelope.mean()

    logger.debug(
        "TMM: %s, R=%.4f, fringe contrast %.3f",
        stack.describe(), stack.reflectance(optics.wavelength),
        (envelope.max() - envelope.min()) / (envelope.max() + envelope.min() + 1e-12),
    )
    return intensity * envelope[:, None, None]


def apply_vertical_interference(
    intensity: NDArray[np.float64],
    resist: ResistConfig,
    optics: OpticsConfig,
    grid: GridConfig,
) -> NDArray[np.float64]:
    """Dispatch to the configured optical model.

    ``"twobeam"`` keeps the original heuristic byte-for-byte; ``"tmm"`` swaps
    in the multilayer solution.
    """
    model = resist.optical_model
    if model == "twobeam":
        return add_standing_waves(intensity, resist, optics, grid.dz)
    if model == "tmm":
        return tmm_standing_waves(intensity, resist, optics, grid)
    raise ValueError(
        f"Unknown optical model '{model}'. Choose 'twobeam' or 'tmm'."
    )


# apply_peb_3d moved to litho_sim.bake.peb — imported at the top of this
# module for print_resist_3d's internal use.


# ---------------------------------------------------------------------------
# Step 4b – the latent volume, either bake
# ---------------------------------------------------------------------------

BAKE_MODELS = ("gaussian", "car")


def latent_volume(
    intensity: NDArray[np.float64],
    resist: ResistConfig,
    grid: GridConfig,
    bake: str = "gaussian",
    bleaching: bool = True,
    species=None,
) -> dict:
    """Absorb and bake a depth-resolved intensity into the latent volume.

    Parameters
    ----------
    intensity : NDArray
        ``(nz, ny, nx)`` intensity, dose included, bottom-up.
    resist, grid
        The film and its voxels.
    bake : str
        ``"gaussian"`` — Dill exposure, then the Gaussian PEB of
        :func:`~litho_sim.bake.peb.apply_peb_3d`: a conventional resist.
        ``"car"`` — acid generation from the local dose, then the 3-D
        acid/quencher reaction–diffusion bake with catalytic deprotection,
        the same chemistry the 2-D ``model="car"`` runs; the latent is the
        protected fraction. Until 2026-09-05 the 3-D path had only the
        Gaussian, so a CAR preset's quencher — the thing that sets its
        contrast — never reached a profile.
    bleaching : bool
        Solve the coupled Dill equations rather than freezing absorption.
    species : SpeciesSample, optional
        A sampled acid and quencher volume from
        :func:`~litho_sim.expose.photochem.sample_species_3d`, which the
        ``"car"`` bake starts from instead of the mean field — how a
        stochastic profile trial enters.

    Returns
    -------
    dict
        ``pac`` (after absorption), ``exposure`` (local incident dose,
        mJ/cm²), ``acid`` (``"car"`` only, before the bake) and ``latent``
        (after the bake — PAC or protected fraction, 1 = unexposed).
    """
    if bake not in BAKE_MODELS:
        raise ValueError(f"bake must be one of {list(BAKE_MODELS)}, got {bake!r}")
    pac, exposure = apply_absorption(intensity, resist, grid.dz, dose=1.0, bleaching=bleaching)
    out = {"pac": pac, "exposure": exposure}
    if bake == "gaussian":
        out["latent"] = apply_peb_3d(pac, resist, grid)
        return out
    from litho_sim.expose.photochem import generate_acid_3d

    if species is None:
        acid = generate_acid_3d(exposure, resist, grid.pixel_size, grid.dz)
        quencher: float | NDArray[np.float64] = resist.quencher_ratio
    else:
        acid, quencher = species.acid, species.quencher
    baked = bake_reaction_diffusion(
        acid,
        grid.pixel_size,
        resist.bake_time,
        resist.D_acid,
        quencher=quencher,
        D_quencher=resist.D_quencher,
        k_quench=resist.k_quench,
        k_loss=resist.k_loss,
        k_amp=resist.k_amp,
        spacing=(grid.dz, grid.pixel_size, grid.pixel_size),
    )
    out["acid"] = acid
    out["latent"] = baked["protected"]
    return out


# ---------------------------------------------------------------------------
# Step 5 – development
# ---------------------------------------------------------------------------


def dissolution_rate_3d(
    latent: NDArray[np.float64], resist: ResistConfig, grid: GridConfig
) -> NDArray[np.float64]:
    """The rate the developer meets at every voxel [nm/s].

    Mack's rate law on the protection the developer sees — the latent for a
    positive resist, its complement for a negative one, where exposure is
    what makes the polymer insoluble — with the surface-inhibition depth
    dependence on top. The one definition both finite-rate models share.
    """
    source = 1.0 - latent if resist.tone == "negative" else latent
    rate = mack_development_rate(
        source, resist.mack_Rmax, resist.mack_Rmin, resist.mack_Mth, resist.mack_n,
    )
    return surface_inhibition(
        rate, grid.dz * 1e9, resist.inhibition_depth, resist.inhibition_rate,
    )


def arrival_field(
    latent: NDArray[np.float64],
    resist: ResistConfig,
    grid: GridConfig,
    model: str = "mack",
    rate_scale: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, str]:
    """The continuous field a developed volume is a level set of.

    Returns ``(field, level, feature)``: resist remains where the field is
    on the *feature* side of *level*. For the finite-rate models the field
    is the time the developer takes to finish each voxel and the level is
    the develop time, so resist is ``"above"``; for the threshold model the
    field is the latent itself against ``mack_Mth`` (``"above"`` for a
    positive resist, ``"below"`` for a negative one).

    This is what a sub-pixel CD at any depth, a develop-time sweep, or a
    roughness measurement should read — a binary volume only knows an edge
    to the nearest voxel.

    Parameters
    ----------
    latent : NDArray
        ``(nz, ny, nx)`` post-bake latent image, bottom-up.
    resist, grid
        As for :func:`develop_3d`.
    model : str
        ``"mack"`` (vertical ray march: ``∫dz/R`` down each column),
        ``"front"`` (the eikonal front, which can undercut), or
        ``"threshold"``.
    rate_scale : NDArray, optional
        A per-voxel multiplier on the dissolution rate — the stochastic
        trials' development noise. Ignored by the threshold model.
    """
    if model == "threshold":
        feature = "above" if resist.tone == "positive" else "below"
        return np.array(latent, dtype=np.float64), float(resist.mack_Mth), feature
    if model not in ("mack", "front"):
        raise ValueError(
            f"Unknown develop model '{model}'. Choose 'threshold', 'mack' or 'front'."
        )
    rate = dissolution_rate_3d(latent, resist, grid)
    if rate_scale is not None:
        rate = rate * rate_scale
    dz_nm = grid.dz * 1e9
    if model == "front":
        # The front travels normal to itself, so it can undercut. Same rate
        # law, different geometry — and the only route to an overhang, since
        # a ray march develops each column in isolation.
        T = arrival_time_front(rate, dz_nm, grid.pixel_size * 1e9)
    else:
        t_voxel = dz_nm / np.maximum(rate, 1e-12)  # seconds to dissolve one voxel
        # Time for the front to finish this voxel = everything above it, plus
        # itself. The array is bottom-up, so accumulate from the top.
        T = np.cumsum(t_voxel[::-1], axis=0)[::-1]
    return T, float(resist.develop_time), "above"


def develop_3d(
    latent: NDArray[np.float64],
    resist: ResistConfig,
    grid: GridConfig | None = None,
    model: str = "threshold",
    require_access: bool = True,
) -> NDArray[np.bool_]:
    """Develop the latent image into a solid.

    Parameters
    ----------
    latent : NDArray
        ``(nz, ny, nx)`` PAC concentration after PEB, bottom-up.
    resist : ResistConfig
        Supplies ``mack_Mth``, ``tone``, and the Mack rate parameters.
    grid : GridConfig, optional
        Required for ``model="mack"`` (supplies ``dz``).
    model : str
        ``"threshold"`` — dissolve wherever PAC falls below ``mack_Mth``.
        Fast, and adequate when only the footprint matters.

        ``"front"`` — the same rate law advanced as a **moving front**,
        normal to itself, by solving the eikonal equation. The only model
        here that can undercut, and therefore the only one that can produce
        a T-top or a foot. See :mod:`litho_sim.develop.front`.

        ``"mack"`` — vertical ray development.  Each column develops
        downward at the local Mack dissolution rate, and the front stops
        wherever ``∫dz/R`` reaches ``resist.develop_time``.  This is the
        model that makes ``mack_Rmax``/``Rmin``/``Mth``/``n`` mean something
        in 3-D, and it is what matches the 2-D engine's criterion.
    require_access : bool
        Threshold model only: dissolve only material the developer can
        actually reach from the top surface.  Without it, an isolated pocket
        of well-exposed resist buried inside the film would "develop out"
        into a floating void, which no developer can do.  The ray model
        enforces this by construction.

    Returns
    -------
    NDArray[np.bool_]
        True where resist **remains**, shape ``(nz, ny, nx)``.
    """
    if model in ("mack", "front"):
        if grid is None:
            raise ValueError(
                f"develop_3d(model={model!r}) requires a GridConfig for dz"
            )
        T, level, _ = arrival_field(latent, resist, grid, model=model)
        remaining = T > level
        logger.debug(
            "Develop (%s): %.1f%% remains after %.0f s",
            model, 100.0 * float(remaining.mean()), resist.develop_time,
        )
        return remaining

    if model != "threshold":
        raise ValueError(
            f"Unknown develop model '{model}'. "
            f"Choose 'threshold', 'mack' or 'front'."
        )

    Mth = resist.mack_Mth
    # Positive tone: exposed resist (low PAC) dissolves.
    soluble = latent < Mth if resist.tone == "positive" else latent >= Mth

    if require_access and soluble.any():
        # A soluble voxel only clears if it connects to the top surface
        # through other soluble voxels.
        lab, n = label(soluble)
        if n > 0:
            open_labels = np.unique(lab[-1])  # components touching the top plane
            open_labels = open_labels[open_labels != 0]
            soluble = np.isin(lab, open_labels)

    remaining = ~soluble
    logger.debug(
        "Develop (threshold): %.1f%% of the film remains",
        100.0 * float(remaining.mean()),
    )
    return remaining


# ---------------------------------------------------------------------------
# Convenience: full 3-D print
# ---------------------------------------------------------------------------


def print_resist_3d(
    mask: NDArray,
    optics: OpticsConfig,
    grid: GridConfig,
    resist: ResistConfig,
    dose: float = 1.0,
    standing_waves: bool = True,
    bleaching: bool = True,
    develop_model: str = "threshold",
    bake: str = "gaussian",
) -> dict:
    """Run the whole 3-D resist pipeline for one exposure.

    Parameters
    ----------
    mask : NDArray
        2-D mask transmittance.
    optics, grid, resist : config objects
        Simulation parameters.
    dose : float
        Relative dose.
    standing_waves : bool
        Apply substrate-reflection standing waves.
    bleaching : bool
        Solve the coupled Dill equations rather than freezing absorption.
    develop_model : str
        ``"threshold"``, ``"mack"`` or ``"front"`` — see :func:`develop_3d`.
    bake : str
        ``"gaussian"`` or ``"car"`` — see :func:`latent_volume`.

    Returns
    -------
    dict
        ``intensity``, ``pac`` (after absorption), ``latent`` (after the
        bake), ``remaining`` (bool, True = resist left), and ``z``.
    """
    intensity, z = exposure_volume(mask, optics, grid, resist, dose=dose)
    if standing_waves:
        intensity = apply_vertical_interference(intensity, resist, optics, grid)
    # *intensity* already carries the relative dose (exposure_volume bakes it
    # into the aerial image), so the absorption step must not apply it again —
    # passing it in both places double-counts and makes every dose sweep
    # quadratic. Same rule as the 2-D engine's simulate_resist().
    vol = latent_volume(intensity, resist, grid, bake=bake, bleaching=bleaching)
    pac, latent = vol["pac"], vol["latent"]
    remaining = develop_3d(latent, resist, grid, model=develop_model)

    frac = float(remaining.mean())
    if frac > 0.995 and standing_waves and develop_model == "threshold":
        logger.warning(
            "Film did not develop (%.1f%% remains): a standing-wave node at "
            "the film top is insoluble under the threshold model, and the "
            "developer cannot reach past it. Raise diffusion_sigma, set "
            "substrate_reflectance=0 (BARC), or use develop_model='mack' — "
            "the finite-rate front passes slow layers instead of stopping.",
            100.0 * frac,
        )
    elif frac < 0.005:
        logger.warning(
            "Film fully cleared (%.1f%% remains): dose too high for this "
            "image contrast, or the Mack front outran develop_time.",
            100.0 * frac,
        )

    return {
        "intensity": intensity,
        "pac": pac,
        "latent": latent,
        "remaining": remaining,
        "z": z,
    }


def write_resist_to_stack(
    stack,
    remaining: NDArray[np.bool_],
    material: str = "photoresist",
) -> None:
    """Replace a stack's resist film with a developed 3-D profile.

    The developed volume is aligned by its **top** with the top of the resist
    film, because that is the surface the developer attacks.  Aligning tops
    rather than bottoms is what lets this work for resist spun over existing
    topography — as happens on the second exposure of every LELE flow, where
    the film is thicker in the trenches than over the lines.

    Parameters
    ----------
    stack : Stack
        Stack containing a coated film of *material*.  Mutated in place.
    remaining : NDArray[np.bool_]
        ``(nz, ny, nx)`` volume from :func:`develop_3d`, True where resist
        remains.
    material : str
        Name of the resist material in the stack.

    Raises
    ------
    ValueError
        If the stack contains no such film.

    Notes
    -----
    Where the coated film is deeper than the simulated volume, the excess at
    the bottom is left untouched — the developer would have had to chew
    through more film than was simulated.  Give ``ResistConfig.thickness``
    the full coated depth to avoid that.
    """
    mid = get_material(material).id
    occ = stack.mat == mid
    if not occ.any():
        raise ValueError(
            f"Stack has no '{material}' film to develop. Coat it first."
        )

    z_present = np.nonzero(occ.any(axis=(1, 2)))[0]
    z_top = int(z_present.max())
    nz_vol = int(remaining.shape[0])

    # Align the top of the developed volume with the top of the film.
    z_start = max(z_top - nz_vol + 1, 0)
    vol_lo = nz_vol - (z_top - z_start + 1)

    window = stack.mat[z_start:z_top + 1]
    window[(window == mid) & ~remaining[vol_lo:]] = VACUUM
    stack.history.append(
        f"develop {material}: {100.0 * float(remaining.mean()):.0f}% retained"
    )


def sidewall_angle(
    remaining: NDArray[np.bool_], grid: GridConfig, row: int | None = None
) -> float:
    """Estimate the sidewall angle of the centre feature [degrees].

    Measures the width of the feature nearest the field centre at 10 % and
    90 % of the remaining film height and converts the difference into an
    angle.  90° is a perfectly vertical wall; smaller values slope outward
    at the base (the normal positive-tone result, since the top of the film
    absorbs the most light).

    Until 2026-09-05 this summed the widths of *every* feature in the row,
    so on a lines-and-spaces field of two and a half lines the run was two
    and a half times too large and a 62° wall reported as 41°.

    Returns
    -------
    float
        Angle in degrees, or ``nan`` if no feature is found.
    """
    from litho_sim.develop.resist import feature_edges

    nz, ny, nx = remaining.shape
    r = ny // 2 if row is None else row
    prof = remaining[:, r, :]

    occupied = np.nonzero(prof.any(axis=1))[0]
    if occupied.size < 2:
        return float("nan")
    z_lo, z_hi = int(occupied.min()), int(occupied.max())
    height = (z_hi - z_lo + 1) * grid.dz
    if height <= 0:
        return float("nan")

    def width(iz: int) -> float:
        row = prof[iz].astype(np.float64)
        edges = feature_edges(row, 0.5, feature="above")
        if edges is None:
            # No crossing: the plane is either solid (an uncleared floor, or
            # an untouched film — full width) or empty.
            return nx * grid.pixel_size if row.all() else 0.0
        return (edges[1] - edges[0]) * grid.pixel_size

    i_lo = int(z_lo + 0.1 * (z_hi - z_lo))
    i_hi = int(z_lo + 0.9 * (z_hi - z_lo))
    w_lo, w_hi = width(i_lo), width(i_hi)
    if w_lo == 0.0 and w_hi == 0.0:
        return float("nan")

    run = (w_lo - w_hi) / 2.0
    rise = 0.8 * height
    if abs(run) < 1e-15:
        return 90.0
    return float(np.degrees(np.arctan2(rise, run)))
