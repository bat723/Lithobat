"""
Configuration module for the LithoPy simulation engine.

Defines all simulation parameters through typed dataclasses, plus
built-in presets for common technology nodes and resist chemistries.
All physical quantities are in SI units unless otherwise noted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Technology-node presets
# ---------------------------------------------------------------------------

TECH_NODE_PRESETS: dict[str, dict[str, Any]] = {
    "i-line": {
        "wavelength": 365e-9,
        "NA": 0.57,
        "k1": 0.50,
        "description": "365 nm i-line stepper",
    },
    "KrF": {
        "wavelength": 248e-9,
        "NA": 0.75,
        "k1": 0.40,
        "description": "248 nm KrF excimer scanner",
    },
    "ArF": {
        "wavelength": 193e-9,
        "NA": 0.93,
        "k1": 0.35,
        "description": "193 nm ArF dry scanner",
    },
    "ArF_immersion": {
        "wavelength": 193e-9,
        "NA": 1.35,
        "n_immersion": 1.44,
        "k1": 0.28,
        "description": "193 nm ArF immersion scanner",
    },
    "EUV": {
        "wavelength": 13.5e-9,
        "NA": 0.33,
        "k1": 0.40,
        "description": "13.5 nm EUV scanner",
    },
}

# ---------------------------------------------------------------------------
# Resist library presets
# ---------------------------------------------------------------------------

RESIST_LIBRARY: dict[str, dict[str, Any]] = {
    "Generic Positive": {
        "tone": "positive",
        "threshold": 0.30,
        "dill_A": 0.80,
        "dill_B": 0.05,
        "dill_C": 0.04,
        "mack_Rmax": 100.0,
        "mack_Rmin": 0.01,
        "mack_Mth": 0.50,
        "mack_n": 4,
        "diffusion_sigma": 20e-9,
        "use_stochastic": False,
        "stochastic_sigma": 1e-9,
    },
    "CAR (Positive)": {
        "tone": "positive",
        "threshold": 0.25,
        "dill_A": 0.50,
        "dill_B": 0.02,
        "dill_C": 0.06,
        "mack_Rmax": 200.0,
        "mack_Rmin": 0.001,
        "mack_Mth": 0.40,
        "mack_n": 6,
        "diffusion_sigma": 30e-9,
        "use_stochastic": False,
        "stochastic_sigma": 1e-9,
        # CAR chemistry (develop model "car"). A KrF/ArF-class formulation:
        # moderate PAG loading, base quencher at 10 % of it.
        "pag_density": 2.0e26,
        "quencher_ratio": 0.10,
        "bake_time": 60.0,
        "D_acid": 4.0e-18,
        "k_quench": 20.0,
        "k_amp": 0.05,
    },
    "EUV CAR (Positive)": {
        "tone": "positive",
        "threshold": 0.25,
        # No bleaching at 13.5 nm — absorption is atomic, not chromophore
        # chemistry — so Dill A is 0 and B carries the whole absorbance
        # (~4 /µm for an organic CAR at EUV).
        "dill_A": 0.0,
        "dill_B": 4.0,
        "dill_C": 0.05,
        "mack_Rmax": 150.0,
        "mack_Rmin": 0.001,
        "mack_Mth": 0.60,
        "mack_n": 8,
        "diffusion_sigma": 20e-9,
        "use_stochastic": False,
        "stochastic_sigma": 1e-9,
        # Thin film — EUV resists run 30–50 nm to hold aspect ratio.
        "thickness": 50e-9,
        # Heavier PAG and quencher loadings than DUV: photons are ~14x more
        # energetic and ~14x scarcer at equal dose, so the formulation fights
        # shot noise with more acid per absorption and more base to sharpen
        # the confinement.
        "pag_density": 3.0e26,
        "quencher_ratio": 0.20,
        "bake_time": 60.0,
        "D_acid": 3.0e-18,
        "k_quench": 20.0,
        "k_amp": 0.05,
        # Photoelectron / secondary-electron range: acid is generated where
        # the electron cascade ends, a few nm from the absorption site.
        "electron_blur_sigma": 4e-9,
    },
    "Generic Negative": {
        "tone": "negative",
        "threshold": 0.60,
        "dill_A": 0.70,
        "dill_B": 0.04,
        "dill_C": 0.03,
        "mack_Rmax": 80.0,
        "mack_Rmin": 0.01,
        "mack_Mth": 0.55,
        "mack_n": 3,
        "diffusion_sigma": 25e-9,
        "use_stochastic": False,
        "stochastic_sigma": 1e-9,
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

#: Bumped when a saved configuration's meaning changes. Written into every
#: JSON file so a reader can tell which rules the numbers were written under.
CONFIG_SCHEMA_VERSION = 1

IMAGING_MODELS = ("scalar", "vector")
POLARISATIONS = ("unpolarised", "x", "y", "te", "tm")
NORMALISATIONS = ("peak", "clear", "none")
MASK_MODELS = ("thin", "multilayer", "fdtd")
TONES = ("positive", "negative")
FOCUS_REFERENCES = ("top", "mid", "bottom")
OPTICAL_MODELS = ("twobeam", "tmm")


def _choice(owner: str, name: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(
            f"{owner}.{name} must be one of {list(choices)}, got {value!r}"
        )


def _bounded(
    owner: str,
    name: str,
    value: Any,
    lo: float | None = None,
    hi: float | None = None,
    *,
    lo_open: bool = False,
    hi_open: bool = False,
) -> None:
    """Raise unless ``lo <= value <= hi`` (bounds open where asked, or absent)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{owner}.{name} must be a number, got {value!r}") from None
    if v != v:  # NaN
        raise ValueError(f"{owner}.{name} must not be NaN")
    if lo is not None and (v < lo or (lo_open and v == lo)):
        raise ValueError(
            f"{owner}.{name} must be {'>' if lo_open else '>='} {lo:g}, got {value!r}"
        )
    if hi is not None and (v > hi or (hi_open and v == hi)):
        raise ValueError(
            f"{owner}.{name} must be {'<' if hi_open else '<='} {hi:g}, got {value!r}"
        )


def _known_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys *cls* does not declare, warning once about what was dropped.

    A configuration written by a newer build may carry fields this one does
    not know; refusing to load it would make every saved file a liability.
    Unknown keys are reported and ignored, and the schema version says which
    rules the rest was written under.
    """
    names = {f.name for f in fields(cls)}
    unknown = sorted(k for k in data if k not in names)
    if unknown:
        logger.warning(
            "%s: ignoring unknown field(s) %s", cls.__name__, ", ".join(unknown)
        )
    return {k: v for k, v in data.items() if k in names}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OpticsConfig:
    """Parameters describing the projection optical system.

    Attributes
    ----------
    wavelength : float
        Exposure wavelength [m].
    NA : float
        Numerical aperture of the projection lens.
    n_immersion : float
        Refractive index of the immersion medium (1.0 = dry, 1.44 = water).
    defocus : float
        Defocus distance [m].  Positive = above focal plane.
    sigma_outer : float
        Outer partial coherence factor (0 < σ ≤ 1).
    sigma_inner : float
        Inner partial coherence factor for annular / off-axis modes.
    source_type : str
        Illumination mode: ``"conventional"``, ``"annular"``,
        ``"dipole"``, or ``"quadrupole"``.
    source_grid : int
        Side length of the illumination sampling grid.  The Abbe sum costs one
        FFT per non-zero source point, so this — not ``GridConfig.n_pixels`` —
        is what sets simulation time.  21 gives ~200 source points and agrees
        with a fully dense source to better than 1 % intensity while running
        ~40× faster.  Raise it if a source shape has fine structure.
    source_kwargs : Dict[str, Any]
        Extra arguments forwarded to the source builder, e.g.
        ``{"axis": "y"}`` for dipole or ``{"rotation_deg": 0.0}`` for
        quadrupole.
    source_spec : Dict[str, Any], optional
        A serialised :class:`~litho_sim.expose.source.Source`.  When set it overrides
        ``source_type``/``sigma_*``/``source_kwargs`` entirely, which is how
        arbitrary pole sets, per-pole weights, pole blur, and free-form
        pixelated sources are carried.

        Stored as a plain dict rather than a ``Source`` object so that
        :meth:`SimulationConfig.to_json` still works — a numpy array in a
        dataclass field would make the config unserialisable.
    zernike_coeffs : Dict[int, float]
        Wavefront aberration coefficients {Noll_index: amplitude_in_waves}.
    exact_defocus : bool
        Use the exact (non-paraxial) defocus optical path difference instead
        of the quadratic approximation.  The two agree to a fraction of a
        percent at low NA and diverge badly at the rim of a hyper-NA pupil —
        at NA 1.35 the paraxial form understates the edge OPD by about a
        third.  Off by default so existing results are reproducible; the
        vector imaging path forces it on, since it needs the same cos θ.
    n_image : float, optional
        Refractive index of the medium the image is formed *in*.  ``None``
        means "same as ``n_immersion``", i.e. the image is considered to form
        in the immersion fluid.  Set it to the resist index to image inside
        the film, which is where the angles — and therefore any high-NA
        correction — are physically evaluated.  Refraction into a denser
        medium shrinks the internal angle: at NA 1.35 that is sin θ = 0.94 in
        water but only 0.79 in resist.
    imaging_model : str
        ``"scalar"`` treats the field as a single complex quantity, which is
        what this engine has always done and is accurate while the ray angles
        stay modest.  ``"vector"`` tracks all three components of the
        electric field, which is what makes the polarisation-dependent
        contrast loss at high NA appear.  Vector imaging costs three inverse
        transforms per source point instead of one (six when unpolarised),
        and forces ``exact_defocus`` on, since both need the same cos θ.
    polarisation : str
        Illumination polarisation, used only by the vector model:
        ``"unpolarised"``, ``"x"``, ``"y"``, ``"te"`` (azimuthal) or ``"tm"``
        (radial).  TE is what real hyper-NA scanners use, because it keeps
        every ray pair s-polarised with respect to its own plane of incidence
        and so preserves full contrast at any angle.
    obliquity : bool
        Apply the aplanatic ``sqrt(cos θ)`` radiometric factor in the vector
        model.  On by default; exposed so its contribution can be isolated
        from the polarisation effect.
    normalisation : str
        What the returned aerial image is measured against.

        ``"peak"`` rescales so the brightest pixel equals the dose.  It is
        the historical behaviour and it is convenient — every image lands in
        a known range — but it throws the absolute radiometric scale away,
        so two configurations can never be compared on how much light they
        actually deliver.

        ``"clear"`` divides by the open-frame intensity for the same optics
        and polarisation: what an unpatterned mask would print.  This is the
        lithographic convention, it makes ``ResistConfig.threshold`` mean a
        fixed fraction of clear-field dose, and it keeps the throughput
        differences between polarisation states visible.

        ``"none"`` returns the raw summed intensity.
    mask_model : str
        How the mask converts illumination into a diffraction spectrum.

        ``"thin"`` is the Kirchhoff screen this engine has always used: the
        mask is a 2-D complex transmittance with no thickness, so its spectrum
        is one FFT, computed once and shared by every source point.

        ``"multilayer"`` keeps the thin-mask spectrum but reflects each
        diffraction order off the EUV multilayer at *its own* angle, using the
        transfer-matrix solver in :mod:`litho_sim.coat.films`.  Costs
        essentially nothing and it is not a small correction: across the orders
        of a dense 32 nm-half-pitch grating the mirror's reflection phase
        varies by tens of degrees, which is an image placement shift and a
        best-focus shift that ``"thin"`` sets to zero by construction.

        ``"fdtd"`` solves Maxwell's equations around the absorber and uses the
        resulting near field as the mask transmission.  This is the only mode
        that captures absorber shadowing.  It is far more expensive and needs a
        cached angle library — see :mod:`litho_sim.expose.m3d`.
    reduction : float
        Mask-to-wafer demagnification, 4x on every production scanner.  Only
        ``"multilayer"`` and ``"fdtd"`` use it: the mask is drawn at wafer
        scale everywhere in this engine, but absorber thickness is a
        *reticle*-side quantity, and the incidence angle at the reticle is the
        wafer-side angle divided by this.
    chief_ray_deg : float
        Angle of the chief ray at the mask [degrees].  0 for a transmissive
        DUV mask; **6** for EUV, where the mask is reflective and the
        illumination has to get out of the way of the reflected beam.  That
        tilt is what makes the absorber cast a shadow, so it is the origin of
        most of what is distinctive about EUV mask 3-D effects.
    chief_ray_axis : str
        Which axis the chief ray is tilted along, ``"x"`` or ``"y"``.
    mask_stack : Dict[str, Any], optional
        Serialised mask film stack — absorber material and thickness, sidewall
        angle, multilayer definition.  Stored as a plain dict rather than an
        object for the same reason as ``source_spec``: so
        :meth:`SimulationConfig.to_json` still works.  ``None`` means "use the
        default stack for this wavelength".
    mask_geometry : Dict[str, Any], optional
        ``{"pitch": ..., "cd": ..., "orientation": ...}`` in **wafer-side**
        metres — the drawn pattern, as the pattern generator was called with.

        Only ``mask_model="fdtd"`` needs it, and it cannot be recovered from the
        mask array: the solver grid is sub-nanometre once divided by the
        reduction, so reading edges off a 4 nm raster would quantise every
        absorber edge to 16 nm on the reticle. Edge position is the whole
        subject of a thick-mask model, so the geometry is carried rather than
        inferred.
    m3d_angles : int
        Side length of the illumination-angle grid the FDTD near-field library
        is solved on.  Cost is its square, so 3 (nine solves) is the default.
    """

    wavelength: float = 193e-9
    NA: float = 0.93
    n_immersion: float = 1.0
    defocus: float = 0.0
    sigma_outer: float = 0.80
    sigma_inner: float = 0.0
    source_type: str = "conventional"
    source_grid: int = 21
    source_kwargs: dict[str, Any] = field(default_factory=dict)
    source_spec: dict[str, Any] | None = None
    zernike_coeffs: dict[int, float] = field(default_factory=dict)
    exact_defocus: bool = False
    n_image: float | None = None
    imaging_model: str = "scalar"
    polarisation: str = "unpolarised"
    obliquity: bool = True
    normalisation: str = "peak"
    mask_model: str = "thin"
    reduction: float = 4.0
    chief_ray_deg: float = 0.0
    chief_ray_axis: str = "x"
    mask_stack: dict[str, Any] | None = None
    mask_geometry: dict[str, Any] | None = None
    m3d_angles: int = 3

    def __post_init__(self) -> None:
        """Reject values the physics cannot take, at construction.

        Every check here is a physical bound, not a taste: an NA above the
        immersion index means ``sin θ > 1``, a coherence factor above 1 puts
        source points outside the pupil where the renderer silently clips
        them, and a misspelt model name would otherwise surface as a
        ``ValueError`` several calls deep, or not at all. ``dataclasses.replace``
        re-runs this, so a derived config is checked too.
        """
        o = "OpticsConfig"
        _bounded(o, "wavelength", self.wavelength, 0.0, lo_open=True)
        _bounded(o, "NA", self.NA, 0.0, lo_open=True)
        _bounded(o, "n_immersion", self.n_immersion, 0.0, lo_open=True)
        if self.n_image is not None:
            _bounded(o, "n_image", self.n_image, 0.0, lo_open=True)
        if self.NA > self.image_index:
            raise ValueError(
                f"{o}: NA {self.NA:g} exceeds the image-space index "
                f"{self.image_index:g}, which would make sin θ > 1. Raise "
                f"n_immersion (or n_image) or lower NA."
            )
        _bounded(o, "defocus", self.defocus)
        _bounded(o, "sigma_outer", self.sigma_outer, 0.0, 1.0, lo_open=True)
        _bounded(o, "sigma_inner", self.sigma_inner, 0.0, self.sigma_outer)
        _bounded(o, "source_grid", self.source_grid, 3)
        if self.source_spec is None:
            from litho_sim.expose.illumination import SOURCE_TYPES

            _choice(o, "source_type", self.source_type, SOURCE_TYPES)
        _choice(o, "imaging_model", self.imaging_model, IMAGING_MODELS)
        _choice(o, "polarisation", self.polarisation, POLARISATIONS)
        _choice(o, "normalisation", self.normalisation, NORMALISATIONS)
        _choice(o, "mask_model", self.mask_model, MASK_MODELS)
        _choice(o, "chief_ray_axis", self.chief_ray_axis, ("x", "y"))
        _bounded(o, "reduction", self.reduction, 0.0, lo_open=True)
        _bounded(o, "chief_ray_deg", self.chief_ray_deg, -90.0, 90.0)
        _bounded(o, "m3d_angles", self.m3d_angles, 1)
        for j, c in (self.zernike_coeffs or {}).items():
            if not isinstance(j, int) or isinstance(j, bool) or j < 1:
                raise ValueError(
                    f"{o}.zernike_coeffs keys are Noll indices (int ≥ 1), got {j!r}"
                )
            _bounded(o, f"zernike_coeffs[{j}]", c)

    @property
    def mask_side_NA(self) -> float:
        """Numerical aperture on the *mask* side, ``NA / reduction``.

        The reticle sits in a much slower cone than the wafer: 0.0825 at the
        EUV preset against 0.33 at the wafer, and 0.34 at ArF immersion
        against 1.35.  Every mask-side incidence angle is bounded by this.
        """
        return self.NA / self.reduction

    @property
    def k_max(self) -> float:
        """Maximum supported spatial frequency [m⁻¹]."""
        return self.NA / self.wavelength

    @property
    def image_index(self) -> float:
        """Refractive index of the image-space medium.

        Resolves :attr:`n_image` against its ``None`` default so callers never
        have to.  Everything angular in image space — defocus OPD, and later
        the vector-imaging geometry — is evaluated against this, not against
        ``n_immersion`` directly.
        """
        return self.n_immersion if self.n_image is None else self.n_image

    @property
    def sin_theta_max(self) -> float:
        """Sine of the marginal ray angle in the image medium.

        ``NA / n_image``.  This is the number the high-NA corrections turn on:
        it is 0.33 at the EUV preset, 0.93 dry at ArF, and 0.94 in water at
        the immersion preset — but only 0.79 once refracted into resist.
        """
        return self.NA / self.image_index


@dataclass
class ResistConfig:
    """Parameters describing the photoresist and process chemistry.

    Attributes
    ----------
    tone : str
        ``"positive"`` or ``"negative"`` resist.
    threshold : float
        Simple-model aerial-image threshold (0–1).
    dill_A, dill_B, dill_C : float
        Dill ABC coefficients for the exposure model.
    mack_Rmax, mack_Rmin : float
        Maximum / minimum development rate [nm/s].
    inhibition_depth : float
        Surface-inhibition depth δ [nm]. Solvent entering unswollen glassy
        polymer is genuinely transport-limited (Damköhler ~100), unlike the
        rest of development (~1e-5), and this is the standard empirical
        stand-in for it. 0 disables it. Typical values are 5–20 nm.
    inhibition_rate : float
        Dissolution rate at the very top surface, as a fraction of bulk.
        1.0 is no inhibition; 0.05 is a strongly quenched surface. Produces
        T-topping, but only with a develop model whose front can move
        laterally — see ``develop_model="front"``.
    mack_Mth : float
        Mack model threshold PAC concentration (0–1).
    mack_n : int
        Mack model contrast exponent.
    diffusion_sigma : float
        PEB acid diffusion length [m].  Drives the Gaussian-blur bake of the
        ``"mack"`` path; the ``"car"`` path bakes with ``D_acid × bake_time``
        instead and ignores this.
    use_stochastic : bool
        Add correlated line-edge roughness to the developed binary image.
        This is the *cosmetic* noise knob — a statistical texture stamped on
        after development.  The physical route, where roughness emerges from
        photon and molecular counting statistics, is
        :func:`litho_sim.develop.stochastic.stochastic_trials`.
    stochastic_sigma : float
        1-σ edge displacement of that roughness [m].
    stochastic_corr_length : float
        Correlation length of the roughness along the edge [m].  Real LER is
        not white: reported correlation lengths are tens of nanometres, and
        uncorrelated single-pixel noise averages away under any CD
        measurement, which is why the old edge-flip model changed nothing.
    pag_density : float
        Photoacid-generator loading [1/m³].  Sets the molecular count per
        voxel and with it the size of the counting noise — the deterministic
        chemistry never sees it.
    quencher_ratio : float
        Initial base-quencher loading as a fraction of the PAG loading.
        This is the knob that turns the bake non-linear: acid below the
        quencher level is annihilated, acid above it survives.
    bake_time : float
        PEB duration [s], for the ``"car"`` reaction–diffusion bake.
    D_acid, D_quencher : float
        Acid / quencher diffusivities during the bake [m²/s].  The default
        quencher is immobile, which is what bulky base molecules do.
    k_quench : float
        Acid–base neutralisation rate [1/s per unit concentration], with
        concentrations measured in units of the initial PAG loading.
    k_amp : float
        Catalytic deprotection rate [1/s per unit acid].  ``k_amp ×
        acid × bake_time`` of order a few is a well-amplified resist.
    k_loss : float
        First-order acid loss (evaporation, side reactions) [1/s].
    electron_blur_sigma : float
        Photoelectron blur [m].  At EUV a 92 eV absorption releases an
        electron cascade and the acid appears where the cascade thermalises,
        a few nm away.  0 disables it, which is right for DUV.
    thickness : float
        As-coated resist film thickness [m].  Sets the z extent of the 3-D
        exposure volume and the depth the develop front must clear.
    develop_time : float
        Seconds in the developer. Meaningful only against the time it takes to
        clear the film at full rate — ``develop.just_clearing_time`` — because
        a raw number of seconds hides whether that is a 2x over-develop or a
        30x one. The default is **5x**, measured as the point where the spaces
        clear with 2.7 nm of top loss and the printed CD is still near the
        drawn one. It was 30x, which shrank a 100 nm line to ~50 nm and cost
        16.5 nm off the top.
        Development time [s].  With Mack rates in nm/s, ``rate *
        develop_time`` is the depth cleared.
    dose_nominal : float
        Nominal exposure dose to clear [mJ/cm²].  The ``dose`` used elsewhere
        in the codebase is a dimensionless *relative* dose; the Dill model
        needs real units, so exposure energy is ``aerial × dose ×
        dose_nominal``.  Without this the default ``dill_C`` of 0.04 bleaches
        almost nothing at a relative dose of 1.0 and the Mack path never
        clears.
    n_resist : float
        Refractive index of the resist film.  Scales the defocus phase when
        imaging into the film, and sets the standing-wave period.
    substrate_reflectance : float
        Amplitude reflectance at the resist/substrate interface, in [0, 1].
        Drives standing-wave contrast; 0.0 disables the effect (which is
        what a bottom anti-reflective coating is for).
    focus_reference : str
        Which plane of the resist film ``OpticsConfig.defocus`` refers to:
        ``"top"``, ``"mid"``, or ``"bottom"``.  ``"mid"`` is both the usual
        practice and the choice that gives a physically correct tapered
        profile — focusing at the top leaves the bottom of the film so
        defocused that image blur widens the base enough to cancel the
        absorption taper.
    optical_model : str
        How the vertical intensity distribution is computed.

        ``"twobeam"`` (default) — the single-interface heuristic driven by
        ``substrate_reflectance``.  Right period, nothing else.

        ``"tmm"`` — transfer-matrix interference through ``film_stack``,
        averaged over the illumination's angular distribution.  This is what
        lets a BARC be *designed* rather than asserted.
    film_stack : List[Dict[str, Any]]
        Layers between the resist and the substrate, top-down, used only when
        ``optical_model="tmm"``.  Each record is either
        ``{"material": "SiARC", "thickness": 38e-9}`` — resolved against the
        material library — or ``{"n": 1.8, "k": 0.4, "thickness": 38e-9}``.
        Empty means bare substrate.
    """

    tone: str = "positive"
    threshold: float = 0.30
    dill_A: float = 0.80
    dill_B: float = 0.05
    dill_C: float = 0.04
    mack_Rmax: float = 100.0
    mack_Rmin: float = 0.01
    mack_Mth: float = 0.50
    mack_n: int = 4
    # Surface inhibition — the induction period, as an empirical depth
    # dependence rather than a second moving-boundary solve. Off by default,
    # so nothing that exists today moves.
    inhibition_depth: float = 0.0
    inhibition_rate: float = 1.0
    diffusion_sigma: float = 20e-9
    use_stochastic: bool = False
    stochastic_sigma: float = 1e-9
    stochastic_corr_length: float = 25e-9
    # CAR chemistry — consumed by develop model "car" and the stochastic
    # trial runner. Defaults are a generic DUV formulation; the deterministic
    # path never reads pag_density.
    pag_density: float = 2.0e26
    quencher_ratio: float = 0.10
    bake_time: float = 60.0
    D_acid: float = 4.0e-18
    D_quencher: float = 0.0
    k_quench: float = 20.0
    k_amp: float = 0.05
    k_loss: float = 0.0
    electron_blur_sigma: float = 0.0
    thickness: float = 100e-9
    develop_time: float = 5.0
    dose_nominal: float = 30.0
    n_resist: float = 1.70
    substrate_reflectance: float = 0.0
    focus_reference: str = "mid"
    optical_model: str = "twobeam"
    film_stack: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject values the chemistry cannot take, at construction.

        Zero is allowed wherever it has a meaning — a zero-thickness film, a
        zero develop time and a zero diffusion length are all legitimate
        degenerate cases the tests lean on — and refused where it does not
        (a zero nominal dose, a zero maximum rate). The Mack exponent must be
        at least 2 because the rate law divides by ``n − 1``.
        """
        r = "ResistConfig"
        _choice(r, "tone", self.tone, TONES)
        _bounded(r, "threshold", self.threshold, 0.0, 1.0, lo_open=True, hi_open=True)
        for name in ("dill_A", "dill_B", "dill_C"):
            _bounded(r, name, getattr(self, name), 0.0)
        _bounded(r, "mack_Rmax", self.mack_Rmax, 0.0, lo_open=True)
        _bounded(r, "mack_Rmin", self.mack_Rmin, 0.0, self.mack_Rmax)
        _bounded(r, "mack_Mth", self.mack_Mth, 0.0, 1.0, lo_open=True, hi_open=True)
        _bounded(r, "mack_n", self.mack_n, 2)
        _bounded(r, "inhibition_depth", self.inhibition_depth, 0.0)
        _bounded(r, "inhibition_rate", self.inhibition_rate, 0.0, 1.0, lo_open=True)
        for name in ("diffusion_sigma", "stochastic_sigma", "quencher_ratio",
                     "bake_time", "D_acid", "D_quencher", "k_quench", "k_amp",
                     "k_loss", "electron_blur_sigma", "thickness", "develop_time"):
            _bounded(r, name, getattr(self, name), 0.0)
        _bounded(r, "stochastic_corr_length", self.stochastic_corr_length, 0.0,
                 lo_open=True)
        _bounded(r, "pag_density", self.pag_density, 0.0, lo_open=True)
        _bounded(r, "dose_nominal", self.dose_nominal, 0.0, lo_open=True)
        _bounded(r, "n_resist", self.n_resist, 0.0, lo_open=True)
        _bounded(r, "substrate_reflectance", self.substrate_reflectance, 0.0, 1.0)
        _choice(r, "focus_reference", self.focus_reference, FOCUS_REFERENCES)
        _choice(r, "optical_model", self.optical_model, OPTICAL_MODELS)
        for i, layer in enumerate(self.film_stack):
            if not isinstance(layer, dict) or "thickness" not in layer:
                raise ValueError(
                    f"{r}.film_stack[{i}] must be a dict with a 'thickness' key, "
                    f"got {layer!r}"
                )
            _bounded(r, f"film_stack[{i}].thickness", layer["thickness"], 0.0)


@dataclass
class GridConfig:
    """Numerical simulation grid parameters.

    Attributes
    ----------
    n_pixels : int
        Number of pixels along each axis (square grid).
    pixel_size : float
        Physical size of one pixel [m].
    dz : float
        Vertical voxel height for 3-D simulation [m].  Must resolve the
        standing-wave period λ/(2·n_resist) ≈ 57 nm at 193 nm, so keep it
        well under ~5 nm if standing waves are enabled.
    n_z_slices : int
        Number of *optical* planes computed through the resist.  Each one
        costs a full Abbe sum, so this is the expensive knob; the intensity
        varies smoothly with depth, so ~21 planes interpolated onto the
        finer ``dz`` voxel grid is plenty.
    """

    n_pixels: int = 128
    pixel_size: float = 4e-9
    dz: float = 2e-9
    n_z_slices: int = 21

    def __post_init__(self) -> None:
        g = "GridConfig"
        if isinstance(self.n_pixels, bool) or int(self.n_pixels) != self.n_pixels:
            raise TypeError(f"{g}.n_pixels must be an integer, got {self.n_pixels!r}")
        _bounded(g, "n_pixels", self.n_pixels, 1)
        _bounded(g, "pixel_size", self.pixel_size, 0.0, lo_open=True)
        # dz = 0 is tolerated: purely 2-D work never reads it.
        _bounded(g, "dz", self.dz, 0.0)
        _bounded(g, "n_z_slices", self.n_z_slices, 1)

    @property
    def grid_size(self) -> float:
        """Total simulation domain width [m]."""
        return self.n_pixels * self.pixel_size


@dataclass
class SimulationConfig:
    """Top-level configuration aggregating optics, resist, and grid settings.

    Examples
    --------
    Create from a technology-node preset::

        cfg = SimulationConfig.from_tech_node("ArF", resist_name="CAR (Positive)")

    Load from / save to JSON::

        cfg.to_json(Path("my_sim.json"))
        cfg2 = SimulationConfig.from_json(Path("my_sim.json"))
    """

    optics: OpticsConfig = field(default_factory=OpticsConfig)
    resist: ResistConfig = field(default_factory=ResistConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    dose: float = 1.0
    name: str = "default_sim"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain, JSON-compatible dictionary.

        Carries ``schema_version`` so a file can say which rules it was
        written under; :meth:`from_dict` ignores keys it does not know.
        """
        d = asdict(self)
        d["schema_version"] = CONFIG_SCHEMA_VERSION
        return d

    def to_json(self, path: Path) -> None:
        """Write configuration to a JSON file.

        Parameters
        ----------
        path : Path
            Destination path (parent directories created if needed).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Configuration saved → %s", path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        """Reconstruct from a plain dictionary (e.g. loaded from JSON).

        JSON stores integer keys as strings; ``zernike_coeffs`` are
        automatically converted back to ``{int: float}``.

        Fields this build does not know are dropped with a warning rather
        than refused, so a file from a newer build still loads; a
        ``schema_version`` newer than :data:`CONFIG_SCHEMA_VERSION` is
        logged, since the values may then mean something else.
        """
        version = data.get("schema_version")
        if version is not None and int(version) > CONFIG_SCHEMA_VERSION:
            logger.warning(
                "configuration schema version %s is newer than this build's %s",
                version, CONFIG_SCHEMA_VERSION,
            )
        optics_data = _known_fields(OpticsConfig, dict(data.get("optics", {})))
        if "zernike_coeffs" in optics_data:
            optics_data["zernike_coeffs"] = {
                int(k): float(v)
                for k, v in optics_data["zernike_coeffs"].items()
            }
        optics = OpticsConfig(**optics_data)
        resist = ResistConfig(**_known_fields(ResistConfig, dict(data.get("resist", {}))))
        grid = GridConfig(**_known_fields(GridConfig, dict(data.get("grid", {}))))
        return cls(
            optics=optics,
            resist=resist,
            grid=grid,
            dose=float(data.get("dose", 1.0)),
            name=str(data.get("name", "sim")),
        )

    @classmethod
    def from_json(cls, path: Path) -> SimulationConfig:
        """Load configuration from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Configuration loaded ← %s", path)
        return cls.from_dict(data)

    @classmethod
    def from_tech_node(
        cls,
        node: str,
        resist_name: str = "Generic Positive",
        **kwargs: Any,
    ) -> SimulationConfig:
        """Construct from a named technology-node preset.

        Parameters
        ----------
        node : str
            Key from :data:`TECH_NODE_PRESETS` (e.g. ``"ArF"``).
        resist_name : str
            Key from :data:`RESIST_LIBRARY`.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`SimulationConfig` (e.g. ``name="my_run"``).

        Raises
        ------
        ValueError
            For unknown node or resist names.
        """
        if node not in TECH_NODE_PRESETS:
            raise ValueError(
                f"Unknown node '{node}'. Available: {list(TECH_NODE_PRESETS.keys())}"
            )
        if resist_name not in RESIST_LIBRARY:
            raise ValueError(
                f"Unknown resist '{resist_name}'. "
                f"Available: {list(RESIST_LIBRARY.keys())}"
            )

        valid_optics = {f.name for f in fields(OpticsConfig)}
        optics_data = {
            k: v
            for k, v in TECH_NODE_PRESETS[node].items()
            if k in valid_optics
        }
        optics = OpticsConfig(**optics_data)
        resist = ResistConfig(**RESIST_LIBRARY[resist_name])
        return cls(optics=optics, resist=resist, **kwargs)

