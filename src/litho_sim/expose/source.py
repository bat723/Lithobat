"""
Illumination sources as first-class, authorable objects.

:class:`Source` is to illumination what :class:`~litho_sim.mask.layout.Layout` is to
masks: a small set of primitives you compose, save, and hand to the engine —
rather than a fixed menu of shapes selected by string.

The one primitive
-----------------
Every classical illumination shape is the same thing: an **annular sector**.

===================  ============================================
Shape                Pole parameters
===================  ============================================
conventional         one pole, σ_in = 0, opening = 360°
annular              one pole, σ_in > 0, opening = 360°
monopole             one pole, narrow opening, one angle
dipole               two poles, 180° apart
quadrupole / CQuad   four poles on the axes
quasar               four poles on the diagonals
===================  ============================================

Collapsing them into one primitive is what makes quasar possible at all. The
previous implementation built poles as *discs*, and a disc cannot represent the
arc-shaped poles a real quasar illuminator produces.

Weights and blur
----------------
Poles carry a ``weight``, and the whole source can be softened with ``blur``.
Both work because the Abbe sum in :mod:`litho_sim.expose.aerial_image` already treats
the source as a field of arbitrary float weights::

    weight = float(source[row_s, col_s])

so grey-scale sources need **no change to the imaging code at all**.

Cost
----
The Abbe sum costs one FFT per non-zero source point, so a blurred source is
directly more expensive — blurring a quasar takes it from 256 to 992 points.
:attr:`Source.prune` drops negligible weights after blurring; at the default
``1e-3`` the aerial image is unchanged to four decimal places while cutting the
point count substantially. :meth:`Source.n_points` reports the cost before you
pay it.

Conventions
-----------
Rendered on a normalised pupil grid spanning [-1, 1], with **rows = η and
columns = ξ**, matching :func:`litho_sim.expose.pupil.pupil_grid` and the
frequency mapping in the Abbe loop. Sources are normalised to unit sum.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pupil grid — single source of truth in expose.pupil, never reimplemented
# ---------------------------------------------------------------------------


from litho_sim.core.utils import JsonFileMixin  # noqa: E402
from litho_sim.expose.pupil import pupil_grid  # noqa: E402

# ---------------------------------------------------------------------------
# Pole
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pole:
    """One illumination pole, as an annular sector.

    Parameters
    ----------
    sigma_inner, sigma_outer : float
        Radial extent in pupil units (σ).  ``sigma_inner = 0`` gives a filled
        disc sector.
    angle_deg : float
        Azimuthal centre of the sector, measured counter-clockwise from +ξ.
    opening_deg : float
        Azimuthal width.  ``360`` gives a full ring, which is how conventional
        and annular illumination are expressed.
    weight : float
        Relative intensity.  Poles genuinely add where they overlap — the
        previous implementation clipped the sum to 1, which silently destroyed
        any weighting.
    """

    sigma_inner: float = 0.0
    sigma_outer: float = 0.9
    angle_deg: float = 0.0
    opening_deg: float = 360.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma_outer <= 0:
            raise ValueError(f"sigma_outer must be > 0, got {self.sigma_outer}")
        if self.sigma_inner < 0:
            raise ValueError(f"sigma_inner must be >= 0, got {self.sigma_inner}")
        if self.sigma_inner >= self.sigma_outer:
            raise ValueError(
                f"sigma_inner ({self.sigma_inner:.3f}) must be < "
                f"sigma_outer ({self.sigma_outer:.3f})"
            )
        if not 0.0 < self.opening_deg <= 360.0:
            raise ValueError(
                f"opening_deg must be in (0, 360], got {self.opening_deg}"
            )
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")

    def render(self, n: int) -> NDArray[np.float64]:
        """Rasterise this pole onto an ``(n, n)`` pupil grid."""
        rho, phi, _, _ = pupil_grid(n)
        radial = (rho >= self.sigma_inner) & (rho <= self.sigma_outer)
        if self.opening_deg >= 360.0:
            angular = np.ones_like(radial, dtype=bool)
        else:
            # Wrapped angular difference, so a sector spanning 0° works.
            d = np.angle(np.exp(1j * (phi - np.deg2rad(self.angle_deg))))
            angular = np.abs(d) <= np.deg2rad(self.opening_deg) / 2.0
        return self.weight * (radial & angular).astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


@dataclass
class Source(JsonFileMixin):
    """A composable illumination source.

    Parameters
    ----------
    poles : list of Pole
        Composed additively.
    pixels : NDArray, optional
        Free-form source map.  When present it **overrides** the poles and is
        resampled to the requested grid.  This is the pixelated-source route.
    blur : float
        Gaussian softening in pupil units (σ of the kernel relative to the
        pupil radius).  Real illuminators do not have infinitely sharp poles.
    prune : float
        After blurring, drop weights below ``prune × max``.  Purely a cost
        control — see the module docstring.
    name : str
        Identifier, used for saving.
    """

    poles: list[Pole] = field(default_factory=list)
    pixels: NDArray[np.float64] | None = None
    blur: float = 0.0
    prune: float = 1e-3
    name: str = "source"

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, n: int, normalise: bool = True) -> NDArray[np.float64]:
        """Rasterise to an ``(n, n)`` pupil map.

        Parameters
        ----------
        n : int
            Grid size.  This is ``OpticsConfig.source_grid`` — it sets both
            accuracy and cost.
        normalise : bool
            Scale to unit sum.  Leave True; the Abbe sum assumes it.

        Returns
        -------
        NDArray[np.float64]
            Shape ``(n, n)``, rows = η, columns = ξ.
        """
        if n < 3:
            raise ValueError(f"source grid must be >= 3, got {n}")

        if self.pixels is not None:
            src = self._render_pixels(n)
        else:
            src = np.zeros((n, n), dtype=np.float64)
            for pole in self.poles:
                src += pole.render(n)

        if self.blur > 0.0:
            from scipy.ndimage import gaussian_filter

            # blur is in pupil units; the grid spans 2 pupil units over n-1 cells.
            sigma_px = self.blur * (n - 1) / 2.0
            src = gaussian_filter(src, sigma=sigma_px, mode="constant")

        # Everything outside the pupil is unphysical — the illuminator cannot
        # deliver light the lens will not accept.
        rho, _, _, _ = pupil_grid(n)
        src = np.where(rho <= 1.0, src, 0.0)

        if self.prune > 0.0 and src.max() > 0:
            src = np.where(src >= self.prune * src.max(), src, 0.0)

        if normalise:
            total = src.sum()
            if total > 0:
                src = src / total
        return src

    def _render_pixels(self, n: int) -> NDArray[np.float64]:
        """Resample a free-form map onto the requested grid."""
        px = np.asarray(self.pixels, dtype=np.float64)
        if px.ndim != 2:
            raise ValueError(f"pixels must be 2-D, got shape {px.shape}")
        if px.shape == (n, n):
            return px.copy()
        from scipy.ndimage import zoom

        factors = (n / px.shape[0], n / px.shape[1])
        out = zoom(px, factors, order=1, mode="nearest")
        # zoom can be off by one on odd ratios
        if out.shape != (n, n):
            fixed = np.zeros((n, n), dtype=np.float64)
            k = min(n, out.shape[0]), min(n, out.shape[1])
            fixed[: k[0], : k[1]] = out[: k[0], : k[1]]
            out = fixed
        return np.clip(out, 0.0, None)

    # ------------------------------------------------------------------
    # Cost and metrics
    # ------------------------------------------------------------------

    def n_points(self, n: int) -> int:
        """Number of non-zero source points — i.e. FFTs per aerial image.

        This is the simulation cost. Check it before running a blurred or
        free-form source.
        """
        return int((self.render(n) > 0).sum())

    def pupil_fill(self, n: int = 101) -> float:
        """Fraction of the pupil area the source illuminates."""
        src = self.render(n)
        rho, _, _, _ = pupil_grid(n)
        inside = rho <= 1.0
        return float((src > 0).sum() / max(inside.sum(), 1))

    def describe(self) -> str:
        if self.pixels is not None:
            return f"free-form {self.pixels.shape[0]}x{self.pixels.shape[1]}"
        if not self.poles:
            return "empty"
        p = self.poles[0]
        return (
            f"{len(self.poles)} pole(s), σ {p.sigma_inner:.2f}–{p.sigma_outer:.2f}, "
            f"opening {p.opening_deg:.0f}°"
            + (f", blur {self.blur:.3f}" if self.blur else "")
        )

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def conventional(cls, sigma: float = 0.8, **kw) -> Source:
        """Uniform disc."""
        return cls(poles=[Pole(0.0, sigma, 0.0, 360.0)], name="conventional", **kw)

    @classmethod
    def annular(cls, sigma_outer: float = 0.9, sigma_inner: float = 0.6, **kw) -> Source:
        """Full ring — better depth of focus, orientation-neutral."""
        return cls(
            poles=[Pole(sigma_inner, sigma_outer, 0.0, 360.0)], name="annular", **kw
        )

    @classmethod
    def monopole(
        cls,
        sigma_outer: float = 0.9,
        sigma_inner: float = 0.6,
        angle_deg: float = 0.0,
        opening_deg: float = 60.0,
        **kw,
    ) -> Source:
        """A single off-axis pole. Maximum asymmetry; used for very specific pitches."""
        return cls(
            poles=[Pole(sigma_inner, sigma_outer, angle_deg, opening_deg)],
            name="monopole",
            **kw,
        )

    @classmethod
    def dipole(
        cls,
        sigma_outer: float = 0.9,
        sigma_inner: float = 0.6,
        opening_deg: float = 60.0,
        axis: str = "x",
        **kw,
    ) -> Source:
        """Two opposed poles — the highest-contrast choice for dense lines.

        ``axis="x"`` places poles on the ξ axis, which best images lines that
        run along y.
        """
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
        base = 0.0 if axis == "x" else 90.0
        return cls(
            poles=[
                Pole(sigma_inner, sigma_outer, base + a, opening_deg)
                for a in (0.0, 180.0)
            ],
            name=f"dipole-{axis}",
            **kw,
        )

    @classmethod
    def quadrupole(
        cls,
        sigma_outer: float = 0.9,
        sigma_inner: float = 0.6,
        opening_deg: float = 40.0,
        rotation_deg: float = 45.0,
        **kw,
    ) -> Source:
        """Four poles. ``rotation_deg=45`` is quasar; ``0`` is CQuad."""
        return cls(
            poles=[
                Pole(sigma_inner, sigma_outer, a + rotation_deg, opening_deg)
                for a in (0.0, 90.0, 180.0, 270.0)
            ],
            name="quadrupole",
            **kw,
        )

    @classmethod
    def quasar(
        cls,
        sigma_outer: float = 0.9,
        sigma_inner: float = 0.6,
        opening_deg: float = 40.0,
        rotation_deg: float = 45.0,
        **kw,
    ) -> Source:
        """Quasar — four arc poles on the **diagonals**.

        Distinct from :meth:`cquad` only by rotation, but that is the whole
        difference: quasar favours features at 45°, CQuad favours features on
        the axes.
        """
        s = cls.quadrupole(sigma_outer, sigma_inner, opening_deg, rotation_deg, **kw)
        s.name = "quasar"
        return s

    @classmethod
    def cquad(
        cls,
        sigma_outer: float = 0.9,
        sigma_inner: float = 0.6,
        opening_deg: float = 40.0,
        **kw,
    ) -> Source:
        """Cross-quadrupole — four arc poles on the **axes**."""
        s = cls.quadrupole(sigma_outer, sigma_inner, opening_deg, 0.0, **kw)
        s.name = "cquad"
        return s

    @classmethod
    def from_array(cls, pixels: NDArray, name: str = "freeform", **kw) -> Source:
        """Free-form source from a 2-D array of relative intensities."""
        return cls(pixels=np.asarray(pixels, dtype=np.float64), name=name, **kw)

    @classmethod
    def from_image(cls, path: str | Path, name: str | None = None, **kw) -> Source:
        """Free-form source from a greyscale image (PNG/TIFF)."""
        from PIL import Image

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Source image not found: {path}")
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        return cls.from_array(arr, name=name or path.stem, **kw)

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None, **kw) -> Source:
        """Free-form source from a CSV grid of intensities."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Source CSV not found: {path}")
        return cls.from_array(
            np.loadtxt(path, delimiter=","), name=name or path.stem, **kw
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-able dict.

        The pixel array becomes nested lists — ``OpticsConfig`` stores this dict
        so a free-form source survives ``SimulationConfig.to_json``, which would
        otherwise raise on an ndarray.
        """
        return {
            "name": self.name,
            "blur": self.blur,
            "prune": self.prune,
            "poles": [p.to_dict() for p in self.poles],
            "pixels": None if self.pixels is None else np.asarray(self.pixels).tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Source:
        px = data.get("pixels")
        return cls(
            poles=[Pole(**p) for p in data.get("poles", [])],
            pixels=None if px is None else np.asarray(px, dtype=np.float64),
            blur=float(data.get("blur", 0.0)),
            prune=float(data.get("prune", 1e-3)),
            name=str(data.get("name", "source")),
        )

    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Source({self.name!r}, {self.describe()})"


#: Named presets exposed to the UI, as zero-argument builders.
SOURCE_PRESETS: dict[str, Any] = {
    "Conventional": Source.conventional,
    "Annular": Source.annular,
    "Monopole": Source.monopole,
    "Dipole": Source.dipole,
    "Quasar": Source.quasar,
    "CQuad": Source.cquad,
}
