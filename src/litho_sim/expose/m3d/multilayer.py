"""The EUV mask blank: a Bragg mirror, and what it does to a diffraction pattern.

An EUV mask does not transmit. No material is usefully transparent at 13.5 nm,
so the mask is a **mirror** — 40 or so Mo/Si bilayers at a quarter-wave pitch,
reflecting by Bragg interference, with the pattern written in an absorber laid
on top.

That mirror is not a passive backdrop. A Bragg stack is tuned to one angle, and
its response falls away either side: the reflectance drops and, more
importantly, the reflection **phase** turns. A diffraction pattern is a fan of
plane waves leaving the mask at *different* angles, so each order sees a
different mirror.

Why that is the whole story
---------------------------
Fit the order phases of a dense grating to ``phi(m) = a + b*m + c*m^2``:

* ``b`` — a phase ramp across the pupil — is a **shift of the printed image**.
* ``c`` — a quadratic — is **defocus**, i.e. a best-focus shift.
* the amplitude spread across orders is **pupil apodisation**, i.e. contrast loss.

Three of the five classic mask 3-D artefacts, and none of them need Maxwell's
equations solved anywhere: the mirror is planar and unpatterned, so the
transfer-matrix method in :mod:`litho_sim.coat.films` gives each order's
reflection exactly. That is what this module does, and it is why the
``"multilayer"`` mask model exists as a step between ``"thin"`` and ``"fdtd"``.

What it does not capture is the absorber: shadowing, sidewall scattering, and
the phase of light that has been *through* 60 nm of TaBN. Those need the field
solved around the topography.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from litho_sim.coat.films import Film, FilmStack
from litho_sim.expose.m3d.materials import mask_material

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Multilayer:
    """A Mo/Si Bragg mirror with a capping layer.

    Parameters
    ----------
    n_bilayers : int
        Number of Mo/Si pairs.  Reflectance saturates: 40 gives 74.7 %, 30
        gives 72.8 %, and 60 adds only a further 0.6 point over 40 — the light
        simply does not reach the bottom of a deeper stack.
    period : float
        Bilayer thickness [m].  Sets the Bragg condition, so it is the one
        number that has to be right.
    gamma : float
        Fraction of the period that is molybdenum.  0.4 is the usual choice —
        it is near the optimum, because Mo supplies the contrast and Si is the
        one that does not absorb.
    cap_material, cap_thickness : str, float
        Oxidation barrier on top, typically 2.5 nm of ruthenium.
    substrate : str
        Semi-infinite material under the stack.
    """

    n_bilayers: int = 40
    period: float = 6.94e-9
    gamma: float = 0.4
    cap_material: str = "Ru"
    cap_thickness: float = 2.5e-9
    substrate: str = "Si"

    def __post_init__(self) -> None:
        if self.n_bilayers < 1:
            raise ValueError(f"n_bilayers must be >= 1, got {self.n_bilayers}")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if self.period <= 0.0:
            raise ValueError(f"period must be positive, got {self.period}")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def film_stack(self, wavelength: float) -> FilmStack:
        """Assemble the mirror as a :class:`~litho_sim.coat.films.FilmStack`.

        The ambient is vacuum and the cap is treated as the stack's first real
        layer, which is what makes ``resist_index=1`` the right handle for
        :meth:`~litho_sim.coat.films.FilmStack.amplitude_reflection` to reflect
        from.
        """
        t_mo = self.period * self.gamma
        t_si = self.period - t_mo
        films = [
            Film("vacuum", 0.0, mask_material("vacuum", wavelength)),
            Film(
                self.cap_material,
                self.cap_thickness,
                mask_material(self.cap_material, wavelength),
            ),
        ]
        n_si = mask_material("Si", wavelength)
        n_mo = mask_material("Mo", wavelength)
        for _ in range(self.n_bilayers):
            films.append(Film("Si", t_si, n_si))
            films.append(Film("Mo", t_mo, n_mo))
        films.append(Film(self.substrate, 0.0, mask_material(self.substrate, wavelength)))
        return FilmStack(films, resist_index=1)

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------

    def amplitude_reflection(
        self,
        wavelength: float,
        theta: float = 0.0,
        polarisation: str = "s",
    ) -> complex:
        """Complex reflection at one angle [rad]."""
        return self.film_stack(wavelength).amplitude_reflection(
            wavelength, theta, polarisation
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_spec(self) -> dict[str, Any]:
        return {
            "n_bilayers": self.n_bilayers,
            "period": self.period,
            "gamma": self.gamma,
            "cap_material": self.cap_material,
            "cap_thickness": self.cap_thickness,
            "substrate": self.substrate,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | None) -> Multilayer:
        return cls() if not spec else cls(**spec)

    def describe(self) -> str:
        t_mo = self.period * self.gamma
        return (
            f"{self.cap_material}({self.cap_thickness * 1e9:.1f}nm) / "
            f"{self.n_bilayers}x[Si({(self.period - t_mo) * 1e9:.2f}nm)"
            f"/Mo({t_mo * 1e9:.2f}nm)] / {self.substrate}"
        )
