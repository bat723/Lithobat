"""
Thin-film interference by the transfer-matrix method.

Light entering the resist does not simply reflect off "the substrate". It
reflects off *every* interface in the film stack, and those reflections
interfere. That is why a bottom anti-reflective coating works at all, and why
printed CD oscillates with resist thickness (the **swing curve**).

The previous model — :func:`litho_sim.develop.resist3d.add_standing_waves` — collapsed
all of that into a single scalar ``substrate_reflectance``. It gets the fringe
*period* right and nothing else: you cannot design a BARC with one number,
because a BARC works by having its own thickness and absorption tuned so its
two reflections cancel.

The method
----------
Each layer contributes a characteristic matrix

.. math::

    M_j = \\begin{pmatrix}
        \\cos\\delta_j & i\\sin\\delta_j / \\eta_j \\\\
        i\\eta_j \\sin\\delta_j & \\cos\\delta_j
    \\end{pmatrix},
    \\qquad \\delta_j = \\frac{2\\pi n_j \\cos\\theta_j d_j}{\\lambda}

with tilted admittance :math:`\\eta_j = n_j\\cos\\theta_j` (s-polarised) or
:math:`n_j/\\cos\\theta_j` (p-polarised). The stack matrix is their product,
and the reflection coefficient follows from the resulting surface admittance.

Division of labour with the Dill model
--------------------------------------
**This module supplies the interference envelope only.** Absorption is owned
by :func:`litho_sim.develop.resist3d.apply_absorption`, which solves the coupled Dill
equations. If the envelope also decayed with depth, the two would double-count
and the film bottom would receive roughly half the dose it should.

So :meth:`FilmStack.field_profile` propagates through the resist using the
**real part** of its index — pure phase, pure fringe — while every layer
*below* the resist keeps its full complex index, because absorption in a BARC
is exactly what makes it work. The returned envelope is normalised to unit
mean, matching the convention ``add_standing_waves`` already established.

This module is deliberately free of project imports so the physics can be
tested against textbook values in isolation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Films
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Film:
    """One layer in the stack.

    Parameters
    ----------
    name : str
        Label, for logging and plots.
    thickness : float
        Physical thickness [m].  Ignored for the ambient and the substrate,
        which are treated as semi-infinite.
    n : complex
        Complex refractive index as ``n + ik``, matching
        :data:`litho_sim.wafer.stack.MATERIAL_LIBRARY` (Si is ``0.883+2.778j``).
        A *positive* imaginary part means absorbing.
    """

    name: str = "film"
    thickness: float = 0.0
    n: complex = 1.0 + 0j


def _propagation_index(n: NDArray) -> NDArray:
    """Convert the ``n + ik`` storage convention to ``n − ik`` for propagation.

    The characteristic matrix uses ``cos δ`` and ``sin δ`` with
    ``δ = 2π n d cosθ / λ``.  With a *positive* imaginary index those become
    ``cosh``/``sinh`` of a positive argument — the layer amplifies instead of
    absorbing, and reflectance comes out greater than one.

    Both conventions are standard; they differ by the assumed sign of the
    time dependence. Storing ``n + ik`` and conjugating here keeps the
    material library readable while the physics stays right.
    """
    return np.conj(np.asarray(n, dtype=complex))


def _tilted_admittance(n: NDArray, cos_theta: NDArray, polarisation: str) -> NDArray:
    """Tilted optical admittance for s- or p-polarisation."""
    if polarisation == "s":
        return n * cos_theta
    if polarisation == "p":
        return n / cos_theta
    raise ValueError(f"polarisation must be 's' or 'p', got '{polarisation}'")


def _check_wavelength(wavelength: float) -> None:
    """Reject a non-positive (or NaN) wavelength before it becomes NaN output.

    ``δ = 2πnd/λ`` divides by the wavelength, so zero produces a silent wall
    of NaN reflectances rather than an error — worse than crashing, because a
    swing curve of NaNs plots as an empty axis and looks like a blank stack.
    """
    if not wavelength > 0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")


@dataclass
class FilmStack:
    """An ordered film stack: ambient first, substrate last.

    Parameters
    ----------
    films : list of Film
        ``films[0]`` is the ambient (immersion fluid or air) and ``films[-1]``
        the substrate; both are semi-infinite and their thickness is ignored.
        Everything between them is a real layer, ordered **top-down**.
    resist_index : int
        Which entry is the photoresist.  Defaults to 1, i.e. the first real
        layer under the ambient.
    """

    films: list[Film] = field(default_factory=list)
    resist_index: int = 1

    def __post_init__(self) -> None:
        if len(self.films) < 2:
            raise ValueError(
                "A stack needs at least an ambient and a substrate, got "
                f"{len(self.films)} film(s)"
            )
        if not 0 < self.resist_index < len(self.films) - 1:
            raise ValueError(
                f"resist_index {self.resist_index} is not a real layer "
                f"(must be between 1 and {len(self.films) - 2})"
            )
        for f in self.films[1:-1]:
            # A negative thickness flips the sign of δ, which turns an
            # absorbing layer into an amplifying one — silently unphysical.
            if not f.thickness >= 0:
                raise ValueError(
                    f"film '{f.name}' has negative thickness {f.thickness}"
                )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def resist(self) -> Film:
        return self.films[self.resist_index]

    def _cos_thetas(self, theta0: float) -> NDArray:
        """Propagation cosines in every layer, from Snell's law."""
        n = _propagation_index([f.n for f in self.films])
        sin0 = n[0] * np.sin(theta0)
        return np.sqrt(1.0 - (sin0 / n) ** 2)

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def _stack_matrix(
        self, wavelength: float, theta0: float, polarisation: str,
        start: int, end: int, cos_theta: NDArray,
    ) -> NDArray:
        """Characteristic matrix for layers ``start..end`` (exclusive of end)."""
        n = _propagation_index([f.n for f in self.films])
        M = np.eye(2, dtype=complex)
        for j in range(start, end):
            d = self.films[j].thickness
            delta = 2.0 * np.pi * n[j] * cos_theta[j] * d / wavelength
            eta = _tilted_admittance(n[j], cos_theta[j], polarisation)
            M = M @ np.array(
                [[np.cos(delta), 1j * np.sin(delta) / eta],
                 [1j * eta * np.sin(delta), np.cos(delta)]],
                dtype=complex,
            )
        return M

    def _reflection_from(
        self, layer: int, wavelength: float, theta0: float, polarisation: str
    ) -> complex:
        """Amplitude reflection seen looking *down* from the top of *layer*."""
        _check_wavelength(wavelength)
        cos_theta = self._cos_thetas(theta0)
        n = _propagation_index([f.n for f in self.films])
        eta = np.array(
            [_tilted_admittance(n[j], cos_theta[j], polarisation)
             for j in range(len(self.films))]
        )
        M = self._stack_matrix(
            wavelength, theta0, polarisation, layer, len(self.films) - 1, cos_theta
        )
        b, c = M @ np.array([1.0, eta[-1]], dtype=complex)
        Y = c / b
        eta_in = eta[layer - 1] if layer > 0 else eta[0]
        return complex((eta_in - Y) / (eta_in + Y))

    def amplitude_reflection(
        self, wavelength: float, theta0: float = 0.0, polarisation: str = "s"
    ) -> complex:
        """Complex amplitude reflection of the whole stack, from the ambient.

        :meth:`reflectance` throws the phase away, which is the right thing for
        a swing curve and the wrong thing for a mirror that has to preserve a
        wavefront. An EUV multilayer's reflection phase varies by tens of
        degrees across the angles a single diffraction pattern spans, and that
        variation *is* the image placement shift and the best-focus shift — so
        anything modelling a reflective mask needs this, not the modulus.
        """
        return self._reflection_from(1, wavelength, theta0, polarisation)

    def reflectance(
        self, wavelength: float, theta0: float = 0.0, polarisation: str = "s"
    ) -> float:
        """Intensity reflectance of the whole stack, seen from the ambient."""
        r = self.amplitude_reflection(wavelength, theta0, polarisation)
        return float(abs(r) ** 2)

    def substrate_reflection(
        self, wavelength: float, theta0: float = 0.0, polarisation: str = "s"
    ) -> complex:
        """Amplitude reflection at the **bottom of the resist**.

        This is the quantity the scalar ``substrate_reflectance`` was trying to
        be — except it is complex, wavelength-dependent, angle-dependent, and
        includes every layer beneath the resist.
        """
        return self._reflection_from(self.resist_index + 1, wavelength, theta0, polarisation)

    # ------------------------------------------------------------------
    # Field inside the resist
    # ------------------------------------------------------------------

    def field_profile(
        self,
        wavelength: float,
        z: NDArray[np.float64],
        theta0: float = 0.0,
        polarisation: str = "s",
        normalise: bool = True,
    ) -> NDArray[np.float64]:
        """Standing-wave intensity envelope through the resist.

        Parameters
        ----------
        wavelength : float
            Exposure wavelength in vacuum [m].
        z : NDArray
            Heights above the **resist bottom** [m], increasing upward, to
            match the bottom-up voxel convention.
        theta0 : float
            Incidence angle in the ambient [rad].
        polarisation : str
            ``"s"`` or ``"p"``.
        normalise : bool
            Scale to unit mean, so the envelope redistributes dose in depth
            without changing the total.  Leave True — absorption belongs to
            the Dill model.

        Returns
        -------
        NDArray
            ``|E(z)|²``, same shape as *z*.
        """
        _check_wavelength(wavelength)
        cos_theta = self._cos_thetas(theta0)
        j = self.resist_index
        # Real part only: this is the interference envelope. Attenuation is
        # applied separately by the Dill absorption march.
        n_r = float(np.real(self.films[j].n))
        cos_r = float(np.real(cos_theta[j]))
        k = 2.0 * np.pi * n_r * cos_r / wavelength
        T = self.films[j].thickness

        r_down = self.substrate_reflection(wavelength, theta0, polarisation)

        down = np.exp(-1j * k * (T - z))
        up = r_down * np.exp(-1j * k * (T + z))
        env = np.abs(down + up) ** 2

        if normalise and env.mean() > 0:
            env = env / env.mean()
        return env.astype(np.float64)

    def swing_curve(
        self,
        wavelength: float,
        thicknesses: Sequence[float],
        theta0: float = 0.0,
        polarisation: str = "s",
    ) -> NDArray[np.float64]:
        """Reflectance as a function of resist thickness — the swing curve.

        Oscillates with period ``λ / (2·n_resist·cosθ)``. Measuring that period
        is the cleanest single validation of the whole method.
        """
        import dataclasses

        out = np.empty(len(thicknesses), dtype=np.float64)
        for i, t in enumerate(thicknesses):
            films = list(self.films)
            films[self.resist_index] = dataclasses.replace(
                films[self.resist_index], thickness=float(t)
            )
            out[i] = FilmStack(films, self.resist_index).reflectance(
                wavelength, theta0, polarisation
            )
        return out

    # ------------------------------------------------------------------

    def describe(self) -> str:
        parts = []
        for i, f in enumerate(self.films):
            if i in (0, len(self.films) - 1):
                parts.append(f"{f.name}(∞)")
            else:
                parts.append(f"{f.name}({f.thickness * 1e9:.0f}nm)")
        return " / ".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FilmStack({self.describe()})"


# ---------------------------------------------------------------------------
# Building a stack from config records
# ---------------------------------------------------------------------------


def film_stack_from_records(
    records: Sequence[dict[str, Any]],
    resist_thickness: float,
    resist_n: complex,
    ambient_n: complex = 1.0 + 0j,
    substrate: str = "Si",
) -> FilmStack:
    """Assemble a :class:`FilmStack` from plain config dicts.

    Each record describes one layer **below the resist**, top-down, as either
    ``{"material": "SiARC", "thickness": 38e-9}`` — resolved against the
    material library — or ``{"n": 1.8, "k": 0.4, "thickness": 38e-9}``.

    Parameters
    ----------
    records : sequence of dict
        Layers between resist and substrate.  Empty means bare substrate.
    resist_thickness, resist_n : float, complex
        The resist film.
    ambient_n : complex
        Immersion medium (1.0 dry, 1.44 water).
    substrate : str
        Material name for the semi-infinite bottom.

    Returns
    -------
    FilmStack

    Raises
    ------
    ValueError
        If the resist or any record has a negative thickness.  A negative
        layer runs the characteristic matrix backwards, so an absorbing film
        *amplifies* and reflectance comes out greater than one.
    KeyError
        If a record names a material the library does not know.
    """
    # Imported lazily so the TMM physics stays independent of the voxel stack.
    from litho_sim.wafer import get_material

    if resist_thickness < 0:
        raise ValueError(f"resist thickness must be >= 0, got {resist_thickness}")

    films = [
        Film("ambient", 0.0, ambient_n),
        Film("resist", resist_thickness, resist_n),
    ]
    for rec in records:
        if "material" in rec:
            m = get_material(rec["material"])
            n = complex(m.n_index)
            name = m.name
        else:
            n = complex(float(rec.get("n", 1.5)), float(rec.get("k", 0.0)))
            name = rec.get("name", "layer")
        thickness = float(rec["thickness"])
        if thickness < 0:
            raise ValueError(
                f"film '{name}' has negative thickness {thickness}; "
                "layers must be >= 0 m thick"
            )
        films.append(Film(name, thickness, n))

    sub = get_material(substrate)
    films.append(Film(sub.name, 0.0, complex(sub.n_index)))
    return FilmStack(films, resist_index=1)


def resist_index_from_dill(
    n_real: float, dill_A: float, dill_B: float, wavelength: float
) -> complex:
    """Complex resist index, with ``k`` derived from the Dill coefficients.

    Dill ``A`` and ``B`` (µm⁻¹) and the extinction coefficient describe the
    same absorption, so deriving one from the other prevents setting both and
    silently halving the dose at the film bottom::

        α = A + B  [µm⁻¹]  →  k = α·λ / 4π

    At A=0.8, B=0.05 and 193 nm this gives k ≈ 0.013, matching the material
    library's photoresist entry.
    """
    alpha_per_m = (dill_A + dill_B) * 1e6
    k = alpha_per_m * wavelength / (4.0 * np.pi)
    return complex(n_real, k)
