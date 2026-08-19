"""
Illumination source module.

Generates 2-D source intensity distributions (σ-maps) in the normalised
pupil plane for conventional, annular, dipole, and quadrupole illumination
modes.  All sources are returned normalised to unit sum.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

from litho_sim.expose.pupil import pupil_grid  # noqa: E402


def _normalise(src: NDArray) -> NDArray:
    s = src.sum()
    return src / s if s > 0 else src


def conventional(n_pixels: int, sigma: float) -> NDArray[np.float64]:
    """Uniform circular disk (conventional / low-sigma) illumination.

    Parameters
    ----------
    n_pixels : int
        Pupil array size.
    sigma : float
        Outer coherence factor (0 < σ ≤ 1).

    Returns
    -------
    NDArray
        Normalised source intensity, shape ``(n_pixels, n_pixels)``.
    """
    rho, *_ = pupil_grid(n_pixels)
    return _normalise((rho <= sigma).astype(np.float64))


def annular(
    n_pixels: int,
    sigma_outer: float,
    sigma_inner: float,
) -> NDArray[np.float64]:
    """Annular ring source.

    Parameters
    ----------
    n_pixels : int
        Pupil array size.
    sigma_outer, sigma_inner : float
        Outer and inner coherence factors.

    Raises
    ------
    ValueError
        If ``sigma_inner >= sigma_outer``.
    """
    if sigma_inner >= sigma_outer:
        raise ValueError(
            f"sigma_inner ({sigma_inner:.3f}) must be < sigma_outer ({sigma_outer:.3f})"
        )
    rho, *_ = pupil_grid(n_pixels)
    mask = (rho >= sigma_inner) & (rho <= sigma_outer)
    return _normalise(mask.astype(np.float64))


def dipole(
    n_pixels: int,
    sigma_outer: float,
    sigma_inner: float,
    axis: str = "x",
) -> NDArray[np.float64]:
    """Two-pole off-axis illumination.

    Parameters
    ----------
    n_pixels : int
        Pupil array size.
    sigma_outer, sigma_inner : float
        Pole annular extent.
    axis : str
        ``"x"`` places poles on the horizontal axis; ``"y"`` on vertical.
    """
    r_pole = (sigma_outer - sigma_inner) / 2.0
    c_mid = (sigma_outer + sigma_inner) / 2.0
    _, _, xi, eta = pupil_grid(n_pixels)
    src = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    centres = [(c_mid, 0.0), (-c_mid, 0.0)] if axis == "x" else [(0.0, c_mid), (0.0, -c_mid)]
    for cx, cy in centres:
        d = np.sqrt((xi - cx) ** 2 + (eta - cy) ** 2)
        src += (d <= r_pole).astype(np.float64)
    return _normalise(src)


def quadrupole(
    n_pixels: int,
    sigma_outer: float,
    sigma_inner: float,
    rotation_deg: float = 45.0,
) -> NDArray[np.float64]:
    """Four-pole off-axis illumination (cross-quad or c-quad).

    Parameters
    ----------
    n_pixels : int
        Pupil array size.
    sigma_outer, sigma_inner : float
        Pole annular extent.
    rotation_deg : float
        Rotation of the quad pattern in degrees.
        0° places poles on the x/y axes; 45° places them diagonally.
    """
    r_pole = (sigma_outer - sigma_inner) / 2.0
    c_mid = (sigma_outer + sigma_inner) / 2.0
    angles = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]) + rotation_deg)
    _, _, xi, eta = pupil_grid(n_pixels)
    src = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    for angle in angles:
        cx = c_mid * np.cos(angle)
        cy = c_mid * np.sin(angle)
        d = np.sqrt((xi - cx) ** 2 + (eta - cy) ** 2)
        src += (d <= r_pole).astype(np.float64)
    return _normalise(src)


def build_source(
    n_pixels: int,
    source_type: str,
    sigma_outer: float,
    sigma_inner: float = 0.0,
    spec: dict | None = None,
    **kwargs,
) -> NDArray[np.float64]:
    """Factory: build a normalised source distribution.

    Parameters
    ----------
    n_pixels : int
        Pupil grid size.
    source_type : str
        One of the keys listed in :data:`SOURCE_TYPES`.
    sigma_outer, sigma_inner : float
        Coherence factors.
    spec : dict, optional
        A serialised :class:`~litho_sim.expose.source.Source`.  When given it takes
        precedence over every other argument — this is the route for
        arbitrary pole sets, per-pole weights, blur, and free-form pixelated
        sources, none of which fit the ``σ_out``/``σ_in`` parametrisation.
    **kwargs
        Extra arguments for the named builder, e.g. ``axis="y"`` for dipole or
        ``rotation_deg=0.0`` for quadrupole.

    Returns
    -------
    NDArray
        Normalised source intensity array, rows = η, columns = ξ.

    Raises
    ------
    ValueError
        For an unrecognised *source_type*, or for kwargs the chosen builder
        does not accept.
    """
    if spec is not None:
        from litho_sim.expose.source import Source

        src = Source.from_dict(spec).render(n_pixels)
        logger.debug(
            "Source: from spec '%s', nonzero_pts=%d",
            spec.get("name", "?"), int((src > 0).sum()),
        )
        return src

    from litho_sim.expose.source import Source

    # Every named shape is a Source preset; the legacy disc-pole functions are
    # kept for the two shapes whose exact rasterisation older tests pin.
    builders = {
        "conventional": lambda: conventional(n_pixels, sigma_outer),
        "annular": lambda: annular(n_pixels, sigma_outer, sigma_inner),
        "dipole": lambda: dipole(n_pixels, sigma_outer, sigma_inner, **kwargs),
        "quadrupole": lambda: quadrupole(n_pixels, sigma_outer, sigma_inner, **kwargs),
        "monopole": lambda: Source.monopole(
            sigma_outer, sigma_inner, **kwargs
        ).render(n_pixels),
        "quasar": lambda: Source.quasar(
            sigma_outer, sigma_inner, **kwargs
        ).render(n_pixels),
        "cquad": lambda: Source.cquad(
            sigma_outer, sigma_inner, **kwargs
        ).render(n_pixels),
    }
    key = source_type.lower()
    if key not in builders:
        raise ValueError(
            f"Unknown source type '{source_type}'. "
            f"Available: {list(builders)}"
        )
    # conventional and annular take no extra arguments; silently swallowing
    # them hides typos in source_kwargs.
    if kwargs and key in ("conventional", "annular"):
        raise ValueError(
            f"Source type '{key}' accepts no extra arguments, got "
            f"{sorted(kwargs)}. Use 'monopole', 'dipole', 'quasar' or 'cquad' "
            f"for shapes with an angle or opening."
        )

    src = builders[key]()
    logger.debug(
        "Source: type=%s, σ_out=%.2f, σ_in=%.2f, nonzero_pts=%d",
        source_type, sigma_outer, sigma_inner, int((src > 0).sum()),
    )
    return src


#: Source shapes reachable by name through :func:`build_source`.
SOURCE_TYPES = (
    "conventional", "annular", "monopole", "dipole", "quadrupole", "quasar", "cquad",
)

