"""What a mask is made of, once it is allowed to have a thickness.

Everywhere else in this engine a mask is a ``(n, n)`` array of transmittance —
no material, no thickness, no cross-section. That is enough for a Kirchhoff
screen and is not enough for anything else, so this is where a mask acquires a
stack: an absorber of a given material and height standing on a substrate that
either transmits (DUV) or reflects (EUV).

Two regimes, and they are not variations on each other
------------------------------------------------------
``"duv_transmissive"``
    Light goes *through*: quartz substrate, chrome or MoSi absorber patterned on
    top, near field taken below the mask. The absorber is opaque or nearly so
    and the physics of interest is at its edges.

``"euv_reflective"``
    Nothing transmits at 13.5 nm. The substrate is a Mo/Si Bragg mirror, the
    absorber sits on it, and the near field is taken in *reflection*. The
    illumination arrives at 6° to stay clear of the reflected beam, which is
    why an EUV absorber shadows its own trench and a DUV one does not.

Thicknesses here are **reticle-side** physical dimensions and are not scaled by
the reduction. A 60 nm absorber is 60 nm whether the tool prints 4x or 8x — and
its ratio to the mask-side feature width is precisely what makes mask 3-D
effects grow as features shrink.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from litho_sim.coat.films import Film, FilmStack
from litho_sim.expose.m3d.materials import mask_material
from litho_sim.expose.m3d.multilayer import Multilayer

logger = logging.getLogger(__name__)

#: The two mask regimes.
REGIMES: tuple[str, ...] = ("euv_reflective", "duv_transmissive")


@dataclass(frozen=True)
class MaskStack:
    """An absorber on a substrate, with everything needed to solve it.

    Parameters
    ----------
    regime : str
        ``"euv_reflective"`` or ``"duv_transmissive"``.
    absorber : str
        Material name, resolved against
        :data:`~litho_sim.expose.m3d.materials.MASK_MATERIALS`.
    absorber_thickness : float
        Reticle-side absorber height [m].
    sidewall_deg : float
        Absorber sidewall angle from horizontal; 90° is vertical. Used by the
        FDTD topography builder, ignored by the thinner models.
    multilayer : Multilayer, optional
        The Bragg mirror. Required for ``"euv_reflective"``, meaningless
        otherwise.
    substrate : str
        Semi-infinite material below everything. Quartz for DUV.
    """

    regime: str = "euv_reflective"
    absorber: str = "TaBN"
    absorber_thickness: float = 60e-9
    sidewall_deg: float = 90.0
    multilayer: Multilayer | None = field(default_factory=Multilayer)
    substrate: str = "SiO2"

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(
                f"regime must be one of {list(REGIMES)}, got '{self.regime}'"
            )
        if self.absorber_thickness < 0.0:
            raise ValueError(
                f"absorber_thickness must be >= 0, got {self.absorber_thickness}"
            )
        if not 0.0 < self.sidewall_deg <= 90.0:
            raise ValueError(
                f"sidewall_deg must be in (0, 90], got {self.sidewall_deg}"
            )
        if self.is_reflective and self.multilayer is None:
            raise ValueError(
                "regime='euv_reflective' needs a multilayer — a reflective mask "
                "with no mirror reflects nothing."
            )

    @property
    def is_reflective(self) -> bool:
        return self.regime == "euv_reflective"

    # ------------------------------------------------------------------
    # The unpatterned blank
    # ------------------------------------------------------------------

    def blank_stack(self, wavelength: float) -> FilmStack:
        """The film stack seen where there is **no** absorber — the open trench.

        This is the reference the whole model is measured against: it is what an
        unpatterned mask would do, so it sets the clear-field amplitude, and for
        a reflective mask it is the mirror on its own.
        """
        if self.is_reflective:
            assert self.multilayer is not None  # guarded in __post_init__
            return self.multilayer.film_stack(wavelength)
        return FilmStack(
            [
                Film("vacuum", 0.0, mask_material("vacuum", wavelength)),
                # A nominal quartz layer so the stack has a "real" layer to
                # reflect from; the substrate below it is the same material, so
                # the interface is optically absent and only the top one counts.
                Film("quartz", 0.0, mask_material(self.substrate, wavelength)),
                Film(self.substrate, 0.0, mask_material(self.substrate, wavelength)),
            ],
            resist_index=1,
        )

    def blank_reflection(
        self, wavelength: float, theta: float = 0.0, polarisation: str = "s"
    ) -> complex:
        """Complex amplitude the open field returns, at one mask-side angle."""
        return self.blank_stack(wavelength).amplitude_reflection(
            wavelength, theta, polarisation
        )

    # ------------------------------------------------------------------
    # Serialisation — dicts, so SimulationConfig.to_json keeps working
    # ------------------------------------------------------------------

    def to_spec(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "absorber": self.absorber,
            "absorber_thickness": self.absorber_thickness,
            "sidewall_deg": self.sidewall_deg,
            "substrate": self.substrate,
            "multilayer": self.multilayer.to_spec() if self.multilayer else None,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | None) -> MaskStack:
        """Rebuild from a plain dict, or take the default for *None*."""
        if not spec:
            return cls()
        data = dict(spec)
        ml = data.pop("multilayer", None)
        return cls(multilayer=Multilayer.from_spec(ml) if ml else None, **data)

    @classmethod
    def for_wavelength(cls, wavelength: float) -> MaskStack:
        """The conventional stack for an exposure wavelength.

        13.5 nm gets the reflective EUV blank; anything else gets a binary
        chrome mask on quartz, which is what "a mask" means at DUV.
        """
        if abs(wavelength - 13.5e-9) < 1e-11:
            return cls(**MASK_STACK_PRESETS["EUV TaBN 60nm"])
        return cls(**MASK_STACK_PRESETS["DUV binary Cr"])

    def describe(self) -> str:
        head = (
            f"{self.absorber}({self.absorber_thickness * 1e9:.0f}nm"
            f"@{self.sidewall_deg:.0f}deg)"
        )
        if self.is_reflective and self.multilayer is not None:
            return f"{head} / {self.multilayer.describe()}"
        return f"{head} / {self.substrate}"


#: Stacks worth having by name.  Thicknesses are the ones these absorbers are
#: actually built at: 60 nm of TaBN gives ~3 % double-pass reflectivity against
#: a 75 % mirror, and 70 nm of chrome is OD > 3.
MASK_STACK_PRESETS: dict[str, dict[str, Any]] = {
    "EUV TaBN 60nm": {
        "regime": "euv_reflective",
        "absorber": "TaBN",
        "absorber_thickness": 60e-9,
        "substrate": "SiO2",
    },
    "EUV TaBN 45nm": {
        "regime": "euv_reflective",
        "absorber": "TaBN",
        "absorber_thickness": 45e-9,
        "substrate": "SiO2",
    },
    "DUV binary Cr": {
        "regime": "duv_transmissive",
        "absorber": "Cr",
        "absorber_thickness": 70e-9,
        "multilayer": None,
        "substrate": "quartz",
    },
    "DUV att-PSM 6%": {
        "regime": "duv_transmissive",
        "absorber": "MoSi",
        "absorber_thickness": 71.9e-9,
        "multilayer": None,
        "substrate": "quartz",
    },
}
