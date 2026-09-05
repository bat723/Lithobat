"""
The parameter model behind the app — declarative, and free of Qt.

Every knob the application offers is one :class:`ParamSpec`. The specs drive
three things at once: which widgets get built, what ranges they allow, and
how the values are assembled back into the engine's config dataclasses. Add a
row here and a control appears, correctly bounded, wired to the right field —
there is no second place to update.

Keeping this module Qt-free is deliberate. It means the half of the
application that decides *what to simulate* can be tested without a display,
which is most of the logic and all of the parts that are easy to get subtly
wrong.

The ranges are hand-written on purpose. The engine's dataclasses carry
defaults but no bounds — ``dataclasses.fields()`` cannot tell you that
``sigma_outer`` must not exceed 1, or that ``source_grid`` is the knob that
sets the cost. That knowledge lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig

#: Values a ``kind`` may take.
KINDS = ("float", "int", "choice", "bool")

#: Pipeline stages, in dependency order. A parameter belongs to the earliest
#: stage it can change, and changing it invalidates that stage and everything
#: after it. ``scale`` is special: dose and normalisation are a single
#: constant applied *after* the Abbe sum, so they never invalidate it.
STAGES = ("mask", "aerial", "scale", "resist", "profile3d", "view")

#: Which stages must be recomputed when a stage is invalidated.
#:
#: ``profile3d`` is deliberately *not* downstream of ``aerial``: the 3-D path
#: rebuilds the optics per depth plane with its own defocus and image index
#: and runs a fresh Abbe sum for each, so it is a parallel branch off the mask
#: rather than a continuation of the 2-D image. ``view`` invalidates nothing —
#: z-exaggeration and cutaway only redraw.
_DOWNSTREAM = {
    "mask": ("mask", "aerial", "scale", "resist", "profile3d"),
    "aerial": ("aerial", "scale", "resist"),
    "scale": ("scale", "resist"),
    "resist": ("resist",),
    "profile3d": ("profile3d",),
    "view": (),
}


@dataclass(frozen=True)
class ParamSpec:
    """One user-facing knob.

    Attributes
    ----------
    key : str
        Identifier, and the config field it targets unless *target* says
        otherwise.
    label : str
        What the user sees.
    kind : str
        One of :data:`KINDS`.
    default : Any
        Initial value, in *display* units.
    lo, hi, step : float
        Slider bounds and granularity, in display units.  Ignored for
        ``choice`` and ``bool``.
    unit : str
        Shown after the value.  Cosmetic.
    scale : float
        Display → SI multiplier.  A slider in nanometres carries
        ``scale=1e-9``; the engine only ever sees metres.
    choices : tuple
        Allowed values for ``choice``.
    group : str
        Which panel section the control belongs to.
    target : str
        Which config the value belongs to: ``"optics"``, ``"resist"``,
        ``"grid"`` or ``"mask"``.
    stage : str
        The earliest pipeline stage this parameter can change — one of
        :data:`STAGES`. This is what makes recomputation cheap: a control
        marked ``"resist"`` cannot invalidate the Abbe sum, so moving it
        costs 0.04 ms rather than 47 ms.
    help : str
        One-line explanation, shown as a tooltip.
    """

    key: str
    label: str
    kind: str
    default: Any
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.01
    unit: str = ""
    scale: float = 1.0
    choices: tuple[Any, ...] = ()
    group: str = "Optics"
    target: str = "optics"
    stage: str = "aerial"
    help: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got '{self.kind}'")
        if self.stage not in STAGES:
            raise ValueError(
                f"'{self.key}' has stage '{self.stage}'; must be one of {STAGES}"
            )
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"'{self.key}' is a choice but lists no choices")
        if self.kind in ("float", "int") and self.hi <= self.lo:
            raise ValueError(f"'{self.key}' has hi <= lo")

    def clamp(self, value: Any) -> Any:
        """Coerce *value* into this spec's domain."""
        if self.kind == "bool":
            return bool(value)
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(
                    f"'{self.key}' must be one of {list(self.choices)}, got {value!r}"
                )
            return value
        v = min(max(float(value), self.lo), self.hi)
        return int(round(v)) if self.kind == "int" else v


# ---------------------------------------------------------------------------
# The knobs
# ---------------------------------------------------------------------------

#: Mask patterns the imaging panel can build, by display name.
PATTERNS = ("lines and spaces", "contacts", "isolated line", "checkerboard")

SPECS: tuple[ParamSpec, ...] = (
    # -- mask ---------------------------------------------------------
    ParamSpec("pattern", "Pattern", "choice", "lines and spaces",
              choices=PATTERNS, group="Pattern", target="mask",
              stage="mask",
              help="Which test structure to image."),
    ParamSpec("pitch", "Pitch", "float", 200.0, 40.0, 800.0, 5.0, "nm", 1e-9,
              group="Pattern", target="mask",
              stage="mask",
              help="Centre-to-centre period of the drawn features."),
    ParamSpec("cd", "Drawn CD", "float", 100.0, 20.0, 400.0, 5.0, "nm", 1e-9,
              group="Pattern", target="mask",
              stage="mask",
              help="Width of the bright (transmitting) feature."),
    ParamSpec("mask_type", "Mask type", "choice", "binary",
              choices=("binary", "att-psm"),
              group="Pattern", target="mask", stage="mask",
              help="binary = chrome on glass. att-psm = attenuated phase "
                   "shift: dark regions leak 6% at 180°, whose destructive "
                   "interference steepens the image edge."),

    # -- mask 3-D -----------------------------------------------------
    # Everything here is stage="mask": it changes the diffraction spectrum, so
    # it invalidates the 3-D profile branch as well as the aerial image.
    ParamSpec("mask_model", "Mask model", "choice", "thin",
              choices=("thin", "multilayer", "fdtd"),
              group="Mask 3-D", target="optics", stage="mask",
              help="thin = Kirchhoff screen. multilayer = each order reflects "
                   "off the EUV mirror at its own angle (near-free, EUV only). "
                   "fdtd = Maxwell around the absorber; minutes on first use, "
                   "then cached."),
    ParamSpec("reduction", "Reduction", "float", 4.0, 1.0, 8.0, 1.0, "x",
              group="Mask 3-D", target="optics", stage="mask",
              help="Mask-to-wafer demagnification. Sets the reticle-side "
                   "incidence angle, which is the wafer angle divided by this."),
    ParamSpec("chief_ray_deg", "Chief ray", "float", 0.0, 0.0, 10.0, 0.5, "deg",
              group="Mask 3-D", target="optics", stage="mask",
              help="Illumination tilt at the reticle. 0 for transmissive DUV; "
                   "6 for EUV, where it is what makes the absorber shadow."),
    ParamSpec("absorber_thickness", "Absorber", "float", 60.0, 0.0, 120.0, 5.0,
              "nm", 1e-9,
              group="Mask 3-D", target="optics", stage="mask",
              help="Reticle-side absorber height. Not scaled by the reduction "
                   "— 60 nm is 60 nm, and its ratio to the mask-side feature "
                   "is what makes 3-D effects grow as features shrink."),
    ParamSpec("sidewall_deg", "Sidewall", "float", 90.0, 45.0, 90.0, 1.0, "deg",
              group="Mask 3-D", target="optics", stage="mask",
              help="Absorber wall angle; 90 is vertical. Used by the FDTD "
                   "topography only."),
    ParamSpec("m3d_angles", "Angle grid", "int", 3, 2, 7, 1,
              group="Mask 3-D", target="optics", stage="mask",
              help="Side of the illumination-angle grid the FDTD near-field "
                   "library is solved on. Cost goes as the square."),

    # -- optics -------------------------------------------------------
    ParamSpec("NA", "NA", "float", 0.93, 0.10, 1.50, 0.01,
              group="Optics", target="optics",
              stage="aerial",
              help="Numerical aperture. Above ~0.9 the vector effect matters."),
    ParamSpec("wavelength", "Wavelength", "float", 193.0, 13.5, 436.0, 0.5,
              "nm", 1e-9, group="Optics", target="optics",
              stage="aerial",
              help="Exposure wavelength. 193 = ArF, 13.5 = EUV."),
    ParamSpec("n_immersion", "n immersion", "float", 1.00, 1.00, 1.60, 0.01,
              group="Optics", target="optics",
              stage="aerial",
              help="Index of the fluid between lens and wafer. 1.44 = water."),
    ParamSpec("sigma_outer", "σ outer", "float", 0.80, 0.10, 1.00, 0.05,
              group="Optics", target="optics",
              stage="aerial",
              help="Outer partial-coherence factor."),
    ParamSpec("sigma_inner", "σ inner", "float", 0.00, 0.00, 0.95, 0.05,
              group="Optics", target="optics",
              stage="aerial",
              help="Inner factor; non-zero makes the source annular."),
    ParamSpec("source_type", "Illumination", "choice", "conventional",
              choices=("conventional", "annular", "monopole", "dipole",
                       "quadrupole", "quasar", "cquad"),
              group="Optics", target="optics",
              stage="aerial",
              help="Source shape."),
    ParamSpec("defocus", "Defocus", "float", 0.0, -400.0, 400.0, 10.0, "nm",
              1e-9, group="Optics", target="optics",
              stage="aerial",
              help="Distance off the focal plane."),
    ParamSpec("source_grid", "Source grid", "int", 21, 5, 41, 2,
              group="Optics", target="optics",
              stage="aerial",
              help="Illumination sampling. THE cost knob — one FFT per "
                   "source point, so ~200 of them at 21."),

    # -- vector -------------------------------------------------------
    ParamSpec("imaging_model", "Imaging", "choice", "scalar",
              choices=("scalar", "vector"), group="Vector", target="optics",
              stage="aerial",
              help="Vector tracks all three field components; costs 3x "
                   "(6x unpolarised)."),
    ParamSpec("polarisation", "Polarisation", "choice", "unpolarised",
              choices=("unpolarised", "x", "y", "te", "tm"),
              group="Vector", target="optics",
              stage="aerial",
              help="Only used by the vector model. TE is what hyper-NA "
                   "scanners use."),
    ParamSpec("n_image", "n image", "float", 1.00, 1.00, 2.00, 0.01,
              group="Vector", target="optics",
              stage="aerial",
              help="Index of the medium the image forms in. Set to the "
                   "resist index to evaluate the vector effect where it "
                   "physically happens. Never below the immersion index."),
    ParamSpec("exact_defocus", "Exact defocus", "bool", False,
              group="Vector", target="optics",
              stage="aerial",
              help="Non-paraxial defocus OPD; forced on by the vector model."),
    ParamSpec("normalisation", "Normalise", "choice", "peak",
              choices=("peak", "clear", "none"),
              group="Vector", target="optics",
              stage="scale",
              help="'clear' references the open-frame dose, which is what "
                   "makes threshold comparable across polarisations."),

    # -- the resist chain ---------------------------------------------
    # Grouped by the step that *reads* each knob, which is also the tab it
    # appears on: Film and Chemistry describe the coated resist, Exposure
    # what the scanner does to it, PEB and Reaction-diffusion the bake,
    # Develop the dissolution. The stage is a separate question from the
    # tab — every one of these is stage="resist" (or "scale" for dose)
    # because none can touch the Abbe sum.
    ParamSpec("resist_model", "Resist model", "choice", "threshold",
              choices=("threshold", "mack", "car"),
              group="Film", target="resist",
              stage="resist",
              help="threshold cuts the diffused aerial image at Threshold. "
                   "mack runs Dill exposure → PEB → Mack dissolution. car "
                   "runs the chemically amplified chain — acid generation, "
                   "quencher reaction–diffusion, catalytic deprotection — "
                   "and develops the protected fraction. mack and car read "
                   "Develop threshold and Develop time, not Threshold."),
    ParamSpec("threshold", "Threshold", "float", 0.30, 0.05, 0.95, 0.01,
              group="Develop", target="resist",
              stage="resist",
              help="Clearing threshold for the threshold model. THE CD "
                   "calibration knob — the default prints well under the "
                   "drawn CD at 1:1. The mack and car models cut on PAC "
                   "via Develop threshold instead."),
    ParamSpec("dose", "Dose", "float", 1.00, 0.20, 3.00, 0.05,
              group="Exposure", target="resist",
              stage="scale",
              help="Relative exposure dose."),
    ParamSpec("diffusion_sigma", "PEB diffusion", "float", 0.0, 0.0, 60.0,
              1.0, "nm", 1e-9, group="PEB", target="resist",
              stage="resist",
              help="Acid diffusion length during post-exposure bake, applied "
                   "to the aerial image before thresholding. Defaults to 0 so "
                   "the app reproduces the CLI exactly; raise it to soften the "
                   "image and shrink the printed feature."),
    ParamSpec("tone", "Tone", "choice", "positive",
              choices=("positive", "negative"),
              group="Film", target="resist",
              stage="resist",
              help="Which part of the resist survives development."),
    # Read by the 2-D mack/car models *and* the 3-D develop step, so they
    # carry stage="resist" (the same dual life `tone` has always led).
    ParamSpec("mack_Mth", "Develop threshold", "float", 0.50, 0.05, 0.95, 0.01,
              group="Develop", target="resist",
              stage="resist",
              help="Where development cuts in *chemistry* space — PAC after "
                   "the bake (mack), protected fraction (car), and the same "
                   "for the 3-D depth path. A different physical quantity "
                   "from the intensity Threshold above."),
    ParamSpec("develop_time", "Develop time", "float", 5.0, 0.5, 30.0, 0.5, "s",
              group="Develop", target="resist",
              stage="resist",
              help="How long the developer runs, for every finite-rate model "
                   "(mack, car, 3-D front). The scale that matters is "
                   "multiples of the just-clearing time — 1 s for a 100 nm "
                   "film at Rmax 100 nm/s — so the default is a 5x "
                   "over-develop."),

    # -- chemistry ----------------------------------------------------
    # The CAR formulation: read by the car model, the stochastic trials, and
    # (for the first two) the mack model. Bake kinetics sit under their own
    # Reaction-diffusion group on the Bake tab.
    ParamSpec("dose_nominal", "Dose to clear", "float", 30.0, 5.0, 100.0, 1.0,
              "mJ/cm²", group="Chemistry", target="resist",
              stage="resist",
              help="Real exposure dose at relative dose 1.0. The Dill "
                   "chemistry needs absolute units; the dimensionless Dose "
                   "slider multiplies this."),
    ParamSpec("dill_C", "Dill C", "float", 0.04, 0.005, 0.20, 0.005,
              "cm²/mJ", group="Chemistry", target="resist",
              stage="resist",
              help="Exposure rate constant — how fast PAC bleaches (mack) "
                   "or PAG converts to acid (car) per unit dose."),
    ParamSpec("pag_density", "PAG loading", "float", 0.20, 0.02, 1.00, 0.02,
              "nm⁻³", 1e27, group="Chemistry", target="resist",
              stage="resist",
              help="Photoacid-generator density. Sets the molecule count "
                   "per voxel — the knob that decides how loud the "
                   "Stochastics tab is. The deterministic car model never "
                   "reads it."),
    ParamSpec("quencher_ratio", "Quencher / PAG", "float", 0.10, 0.0, 0.5,
              0.01, group="Chemistry", target="resist",
              stage="resist",
              help="Base loading as a fraction of PAG. Annihilates "
                   "sub-threshold acid during the bake — where CAR contrast "
                   "comes from. car model only."),
    ParamSpec("bake_time", "PEB time", "float", 60.0, 5.0, 120.0, 5.0, "s",
              group="Reaction-diffusion", target="resist",
              stage="resist",
              help="Reaction–diffusion bake duration (car). The mack model "
                   "bakes with the PEB diffusion length instead."),
    ParamSpec("D_acid", "Acid D", "float", 4.0, 0.1, 20.0, 0.1, "nm²/s",
              1e-18, group="Reaction-diffusion", target="resist",
              stage="resist",
              help="Acid diffusivity during the car bake. √(2Dt) is the "
                   "diffusion length — 22 nm at the defaults."),
    ParamSpec("k_quench", "Quench rate", "float", 20.0, 0.0, 100.0, 1.0,
              "1/s", group="Reaction-diffusion", target="resist",
              stage="resist",
              help="Acid–base neutralisation rate, per unit concentration "
                   "in PAG₀ units. The term that turns the bake from a blur "
                   "into a threshold."),
    ParamSpec("k_amp", "Deprotection", "float", 0.05, 0.0, 0.50, 0.01, "1/s",
              group="Reaction-diffusion", target="resist",
              stage="resist",
              help="Catalytic deprotection rate per unit acid. k × acid × "
                   "bake time of a few is a well-amplified resist."),
    ParamSpec("electron_blur_sigma", "e⁻ blur", "float", 0.0, 0.0, 10.0, 0.5,
              "nm", 1e-9, group="Exposure", target="resist",
              stage="resist",
              help="Photoelectron cascade range — where EUV acid actually "
                   "appears, a few nm from the absorption site. 0 for DUV; "
                   "~4 nm at 13.5 nm."),

    # -- edge roughness (cosmetic) ------------------------------------
    ParamSpec("use_stochastic", "Edge roughness", "bool", False,
              group="Edge roughness", target="resist",
              stage="resist",
              help="Stamp correlated line-edge roughness onto the developed "
                   "image. Cosmetic — a statistical texture with the σ and ξ "
                   "below. Physical noise from photon and molecule counting "
                   "is the Stochastics tab."),
    ParamSpec("stochastic_sigma", "LER σ", "float", 1.0, 0.2, 10.0, 0.2,
              "nm", 1e-9, group="Edge roughness", target="resist",
              stage="resist",
              help="1-σ edge displacement of the cosmetic roughness."),
    ParamSpec("stochastic_corr_length", "LER ξ", "float", 25.0, 5.0, 100.0,
              5.0, "nm", 1e-9, group="Edge roughness", target="resist",
              stage="resist",
              help="Correlation length along the edge — what makes the "
                   "result look like a SEM image rather than static."),

    # -- grid ---------------------------------------------------------
    ParamSpec("n_pixels", "Grid", "int", 128, 32, 256, 32,
              group="Grid", target="grid",
              stage="mask",
              help="Mask array size. Cost scales as n² log n."),
    ParamSpec("pixel_size", "Pixel size", "float", 4.0, 1.0, 16.0, 0.5, "nm",
              1e-9, group="Grid", target="grid",
              stage="mask",
              help="Physical size of one pixel."),

    # -- the 3-D profile branch ---------------------------------------
    # None of these exist on the 2-D path, and all of them once fell back to
    # dataclass defaults with no control at all. Each sits on the tab of the
    # step that reads it — film discretisation with the Film, standing waves
    # with the Exposure, the depth develop model with Develop.
    ParamSpec("thickness", "Film thickness", "float", 100.0, 30.0, 300.0, 10.0,
              "nm", 1e-9, group="Film", target="resist",
              stage="profile3d",
              help="Resist thickness. With dz, this sets how many voxels deep "
                   "the developed solid is."),
    ParamSpec("n_z_slices", "Optical planes", "int", 11, 3, 41, 2,
              group="Film", target="grid",
              stage="profile3d",
              help="THE 3-D cost knob — one full Abbe sum per plane, then "
                   "interpolated onto the voxel grid."),
    ParamSpec("dz", "Voxel height", "float", 2.0, 1.0, 8.0, 0.5, "nm", 1e-9,
              group="Film", target="grid",
              stage="profile3d",
              help="Vertical voxel size. Must stay well under the "
                   "standing-wave period, about 57 nm at 193 nm."),
    ParamSpec("develop_model", "Develop model", "choice", "threshold",
              choices=("threshold", "mack", "front"),
              group="Depth develop", target="resist",
              stage="profile3d",
              help="'threshold' cuts at a PAC level and keeps whatever the "
                   "developer can reach from the top; 'mack' propagates a "
                   "finite-rate front for develop_time straight down each "
                   "column; 'front' advances that same front normal to "
                   "itself, so it can undercut — the only one of the three "
                   "that can produce a T-top or a foot, and the slowest."),
    ParamSpec("inhibition_depth", "Inhibition depth", "float", 0.0, 0.0, 40.0, 1.0, "nm",
              group="Depth develop", target="resist",
              stage="profile3d",
              help="Depth of the slow-dissolving surface layer. Solvent "
                   "entering unswollen glassy polymer is genuinely transport-"
                   "limited, unlike the rest of development; this is the "
                   "standard empirical stand-in. 0 disables it."),
    ParamSpec("inhibition_rate", "Surface rate", "float", 1.0, 0.02, 1.0, 0.02,
              group="Depth develop", target="resist",
              stage="profile3d",
              help="Dissolution rate at the very top surface as a fraction "
                   "of bulk. Needs develop_model='front' to show a T-top: a "
                   "slow cap alone just shifts a ray-marched profile down."),
    ParamSpec("standing_waves", "Standing waves", "bool", False,
              group="Through the film", target="resist",
              stage="profile3d",
              help="Interfere the downward wave with its substrate "
                   "reflection — the scalloped sidewall, and the reason PEB "
                   "and BARCs exist."),
    ParamSpec("substrate_reflectance", "Substrate R", "float", 0.35, 0.0, 0.6,
              0.05, group="Through the film", target="resist",
              stage="profile3d",
              help="Amplitude reflectance under the resist. Only bites with "
                   "standing waves on; 0 is a perfect BARC."),
    ParamSpec("focus_reference", "Focus at", "choice", "mid",
              choices=("top", "mid", "bottom"),
              group="Through the film", target="resist",
              stage="profile3d",
              help="Which plane in the film the defocus setting refers to."),

    # -- 3-D view (no physics) ----------------------------------------
    ParamSpec("z_exaggeration", "Z exaggeration", "float", 2.0, 1.0, 6.0, 0.5,
              group="3-D view", target="view",
              stage="view",
              help="Stretch the picture vertically for legibility. Scales "
                   "the drawing, never the mesh — live, and never recomputes."),
)

SPECS_BY_KEY: dict[str, ParamSpec] = {s.key: s for s in SPECS}

#: The step tabs, left to right in the order the physics runs, and the
#: control sections each one carries. A knob's ``group`` names its section;
#: this table is what puts the section on a tab. Moving a knob between tabs
#: is therefore one word in its spec — the placement of the resist chain is
#: a first pass and expected to be revised.
TAB_GROUPS: dict[str, tuple[str, ...]] = {
    "Mask": ("Pattern", "Mask 3-D", "Grid"),
    "Source": ("Optics", "Vector"),
    "Resist": ("Film", "Chemistry"),
    "Expose": ("Exposure", "Through the film"),
    "Bake": ("PEB", "Reaction-diffusion"),
    "Develop": ("Develop", "Depth develop", "Edge roughness", "3-D view"),
}

#: Tab order, as a tuple.
TABS: tuple[str, ...] = tuple(TAB_GROUPS)

#: Every section, in tab order — the flat view of :data:`TAB_GROUPS`.
GROUPS: tuple[str, ...] = tuple(g for gs in TAB_GROUPS.values() for g in gs)

#: Sections describing only the depth-resolved (3-D) profile. They are shown
#: everywhere — a 3-D setting is a setting, not a mode — but a panel may want
#: to label them, and the tests check they are all accounted for.
GROUPS_3D: tuple[str, ...] = (
    "Through the film", "Depth develop", "3-D view",
)


def tab_of(group: str) -> str:
    """Which step tab a control section lives on."""
    for tab, groups in TAB_GROUPS.items():
        if group in groups:
            return tab
    raise KeyError(f"section '{group}' is on no tab")


def mask_model_availability(
    wavelength: float, pattern: str, mask_type: str = "binary"
) -> dict[str, str | None]:
    """Which mask models can run here, and why the others cannot.

    Two of the three thick-mask models are conditional on settings that live in
    a *different* panel section, so a user changing "Mask model" alone has no
    way to see that the combination is impossible. Offering it anyway and then
    raising is the wrong trade: the panel should not present a choice that
    cannot work.

    Parameters
    ----------
    wavelength : float
        Exposure wavelength [m].
    pattern : str
        Key from :data:`PATTERNS`.

    Returns
    -------
    dict
        ``{model: None}`` when the model is available, ``{model: reason}``
        when it is not.  The reason is written to be shown to a user.
    """
    euv = abs(wavelength - 13.5e-9) < 1e-11
    one_dimensional = pattern in ("lines and spaces", "isolated line")
    if not one_dimensional:
        fdtd_reason = (
            f"'{pattern}' varies in both x and y. The solver is 2.5-D — it "
            f"handles a cross-section through line/space geometry. Use lines "
            f"and spaces, or an isolated line."
        )
    elif mask_type != "binary":
        # The FDTD topography is chrome/absorber on a blank — a partially
        # transmitting phase-shifting film is a different material stack the
        # solver does not model.
        fdtd_reason = (
            "Only for binary masks. The solved topography is an opaque "
            "absorber; an attenuated-PSM film transmits and phase-shifts, "
            "which the geometry builder does not represent."
        )
    else:
        fdtd_reason = None
    return {
        "thin": None,
        "multilayer": None if euv else (
            "Only at EUV (13.5 nm). It models the Bragg mirror an EUV mask "
            "reflects from; a transmissive DUV mask has no mirror."
        ),
        "fdtd": fdtd_reason,
    }

#: Parameters the 3-D *develop* step reads but the latent image does not.
#: Everything before ``develop_3d`` — the Abbe sums, the absorption march,
#: the bake — is 99.4 % of the 3-D cost, so caching the latent against
#: everything else turns a change here from ~730 ms into ~5 ms, which is what
#: lets these be live rather than waiting behind the Compute button.
DEVELOP3D_ONLY: tuple[str, ...] = (
    "develop_model", "mack_Mth", "develop_time", "tone",
    "inhibition_depth", "inhibition_rate",
)

#: Parameters that never enter the 3-D latent image. DEVELOP3D_ONLY plus
#: ``threshold``, which is the 2-D intensity calibration and has *no* effect
#: on the depth-resolved path at all — that one cuts on PAC via ``mack_Mth``.
#: Keeping the two sets apart matters: a knob in the first triggers a live
#: 3-D refresh, and re-rendering for a knob the 3-D path cannot read would be
#: 290 ms spent redrawing an identical picture.
LATENT3D_IGNORES: tuple[str, ...] = DEVELOP3D_ONLY + ("threshold",)


@dataclass
class ParameterModel:
    """Current value of every knob, and the configs they assemble into."""

    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for spec in SPECS:
            self.values.setdefault(spec.key, spec.default)

    # -- access -------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def set(self, key: str, value: Any) -> Any:
        """Set one knob, clamped to its spec. Returns the stored value."""
        if key not in SPECS_BY_KEY:
            raise KeyError(f"unknown parameter '{key}'")
        self.values[key] = SPECS_BY_KEY[key].clamp(value)
        return self.values[key]

    def si(self, key: str) -> Any:
        """Value converted out of display units."""
        spec = SPECS_BY_KEY[key]
        v = self.values[key]
        return v * spec.scale if spec.kind in ("float", "int") else v

    def in_group(self, group: str) -> list[ParamSpec]:
        return [s for s in SPECS if s.group == group]

    # -- config assembly ----------------------------------------------
    def _si_for(self, target: str) -> dict[str, Any]:
        return {
            s.key: self.si(s.key)
            for s in SPECS
            if s.target == target
        }

    def grid(self) -> GridConfig:
        d = self._si_for("grid")
        return GridConfig(
            n_pixels=int(d["n_pixels"]),
            pixel_size=d["pixel_size"],
            dz=d["dz"],
            n_z_slices=int(d["n_z_slices"]),
        )

    def optics(self) -> OpticsConfig:
        d = self._si_for("optics")
        # sigma_inner must stay below sigma_outer or the source builder
        # raises; clamping here keeps the UI from producing invalid states.
        sigma_inner = min(d["sigma_inner"], d["sigma_outer"] - 0.05)

        # NA = n·sinθ, so NA cannot exceed the index of the medium it works
        # in — dragging NA past 1.0 while the immersion fluid is still air
        # asks for sinθ > 1. The engine clips that internally and carries on
        # producing plausible-looking nonsense, which is worse than an error,
        # so the invalid state is prevented here instead.
        NA = min(d["NA"], d["n_immersion"])
        # The image forms in the resist, which is always at least as dense as
        # the fluid above it — a slider left at 1.00 under water would ask for
        # an image medium rarer than the immersion, and with it sin θ > 1
        # inside the film, which OpticsConfig refuses.
        n_image = max(d["n_image"], d["n_immersion"])

        kwargs: dict[str, Any] = {}
        if d["source_type"] == "dipole":
            kwargs["axis"] = "x"
        return OpticsConfig(
            wavelength=d["wavelength"],
            NA=NA,
            n_immersion=d["n_immersion"],
            defocus=d["defocus"],
            sigma_outer=d["sigma_outer"],
            sigma_inner=max(sigma_inner, 0.0),
            source_type=d["source_type"],
            source_grid=int(d["source_grid"]),
            source_kwargs=kwargs,
            imaging_model=d["imaging_model"],
            polarisation=d["polarisation"],
            n_image=n_image,
            exact_defocus=d["exact_defocus"],
            normalisation=d["normalisation"],
            mask_model=d["mask_model"],
            reduction=d["reduction"],
            chief_ray_deg=d["chief_ray_deg"],
            m3d_angles=int(d["m3d_angles"]),
            mask_stack=self.mask_stack(),
            mask_geometry=self.mask_geometry(),
        )

    def mask_stack(self) -> dict[str, Any]:
        """The mask film stack the thick-mask models solve.

        Assembled from the wavelength — 13.5 nm means a reflective EUV blank
        and anything else means chrome on quartz — with the two knobs the panel
        exposes laid over the preset.
        """
        from litho_sim.expose.m3d import MaskStack

        stack = MaskStack.for_wavelength(self.si("wavelength"))
        spec = stack.to_spec()
        spec["absorber_thickness"] = self.si("absorber_thickness")
        spec["sidewall_deg"] = self.si("sidewall_deg")
        return spec

    def mask_geometry(self) -> dict[str, Any] | None:
        """The drawn pattern, for models that need edges rather than pixels.

        Returns ``None`` for patterns that vary in both x and y — contacts and
        checkerboard genuinely need a 3-D solver, and guessing a cross-section
        through them would be worse than refusing.

        An isolated line *is* representable: the imaging path is periodic over
        the field, so one line in the field is a grating whose pitch is the
        field width. Saying so costs nothing and turns an error into an answer.
        """
        pattern = self["pattern"]
        if pattern == "lines and spaces":
            pitch = self.si("pitch")
        elif pattern == "isolated line":
            pitch = self.si("n_pixels") * self.si("pixel_size")
        else:
            return None

        cd = self.si("cd")
        if not 0.0 < cd < pitch:
            return None
        return {"pitch": pitch, "cd": cd, "orientation": "vertical"}

    def resist(self) -> ResistConfig:
        d = self._si_for("resist")
        return ResistConfig(
            tone=d["tone"],
            threshold=d["threshold"],
            diffusion_sigma=d["diffusion_sigma"],
            thickness=d["thickness"],
            focus_reference=d["focus_reference"],
            # Only consulted when standing waves are on; carrying it
            # regardless keeps the slider's value stable across the toggle.
            substrate_reflectance=d["substrate_reflectance"],
            # Read only by the 3-D develop step, but always carried: leaving
            # them out silently pinned the depth-resolved path to the
            # dataclass defaults, so its controls did nothing.
            mack_Mth=d["mack_Mth"],
            develop_time=d["develop_time"],
            inhibition_depth=d["inhibition_depth"],
            inhibition_rate=d["inhibition_rate"],
            # The chemistry block — read by the mack/car models and the
            # Stochastics tab. Carried always, same rule as above: leaving
            # a field out silently pins it to the dataclass default and the
            # control does nothing.
            dose_nominal=d["dose_nominal"],
            dill_C=d["dill_C"],
            pag_density=d["pag_density"],
            quencher_ratio=d["quencher_ratio"],
            bake_time=d["bake_time"],
            D_acid=d["D_acid"],
            k_quench=d["k_quench"],
            k_amp=d["k_amp"],
            electron_blur_sigma=d["electron_blur_sigma"],
            use_stochastic=d["use_stochastic"],
            stochastic_sigma=d["stochastic_sigma"],
            stochastic_corr_length=d["stochastic_corr_length"],
        )

    @property
    def standing_waves(self) -> bool:
        return bool(self.values["standing_waves"])

    @property
    def develop_model(self) -> str:
        return str(self.values["develop_model"])

    @property
    def resist_model(self) -> str:
        """The 2-D development model — threshold, mack, or car."""
        return str(self.values["resist_model"])

    @property
    def dose(self) -> float:
        return float(self.values["dose"])

    def signature(self) -> tuple:
        """Hashable snapshot, for deciding whether a recompute is needed."""
        return tuple(sorted((k, _hashable(v)) for k, v in self.values.items()))

    def stage_signature(self, stage: str) -> tuple:
        """Snapshot of only the parameters *stage* actually depends on.

        A stage depends on its own parameters and on every earlier stage's,
        because the pipeline is sequential. Keying a cache on this rather than
        on the whole model is what lets the threshold slider avoid rebuilding
        the aerial image.
        """
        if stage not in STAGES:
            raise ValueError(f"unknown stage '{stage}'; expected one of {STAGES}")
        upto = STAGES[: STAGES.index(stage) + 1]
        return tuple(sorted(
            (s.key, _hashable(self.values[s.key]))
            for s in SPECS
            if s.stage in upto
        ))

    def latent3d_signature(self) -> tuple:
        """Snapshot of everything the 3-D *latent* image depends on.

        Deliberately by exclusion: the latent is affected by essentially every
        physical knob, so listing what it ignores is both shorter and safer —
        a new parameter is conservatively assumed to matter, and the cache
        stays correct by default rather than by remembering to update it.
        """
        return tuple(sorted(
            (s.key, _hashable(self.values[s.key]))
            for s in SPECS
            if s.key not in LATENT3D_IGNORES and s.stage != "view"
        ))

    @staticmethod
    def stages_invalidated_by(key: str) -> tuple:
        """Which stages a change to *key* forces to recompute."""
        return _DOWNSTREAM[SPECS_BY_KEY[key].stage]


def _hashable(v: Any) -> Any:
    return tuple(v) if isinstance(v, (list, dict, set)) else v
