"""
Zernike decomposition of wafer surfaces.

Zernike polynomials are the natural basis for a circular aperture: they are
orthogonal on the unit disk, and their low orders map one-to-one onto the
shape terms metrology cares about -- piston (mean height), tilt (chucking),
power/defocus (bow), astigmatism and coma (warp lobes).  Fitting the surface
and subtracting the low orders leaves the residual warp and nanotopography
that actually threaten focus budget and die-level flatness.

Convention
----------
Terms are indexed by the ANSI/OSA pair ``(n, m)``: radial order ``n >= 0`` and
azimuthal frequency ``m`` with ``|m| <= n`` and ``n - |m|`` even.  Positive
``m`` is the cosine lobe, negative ``m`` the sine lobe.  The single ANSI index
is ``j = (n(n + 2) + m) / 2``.

Polynomials are **orthonormal**: ``mean(Z**2)`` over the unit disk is 1, so a
coefficient is directly the RMS height [m] that term contributes.  Coefficients
therefore add in quadrature, and the bar chart is an error budget.

The low orders correspond to the familiar Fringe indices named in surface
metrology practice:

===============  ==============  ================
ANSI ``(n, m)``  Fringe index    Name
===============  ==============  ================
``(0, 0)``       Z1              piston
``(1, 1)``       Z2              tilt x
``(1, -1)``      Z3              tilt y
``(2, 0)``       Z4              power / defocus
``(2, -2)``      Z5              astigmatism 45°
``(2, 2)``       Z6              astigmatism 0°
===============  ==============  ================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import factorial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .plotting import (
    INK_MUTED,
    PALETTE,
    annotate_stats,
    plot_wafer_panels,
)
from .synthesize import WaferSurface, apply_mask

logger = logging.getLogger(__name__)

Term = Tuple[int, int]

PISTON: Term = (0, 0)
TILT: Tuple[Term, Term] = ((1, 1), (1, -1))
POWER: Term = (2, 0)

TERM_NAMES: Dict[Term, str] = {
    (0, 0): "piston",
    (1, -1): "tilt y",
    (1, 1): "tilt x",
    (2, -2): "astig 45°",
    (2, 0): "power",
    (2, 2): "astig 0°",
    (3, -3): "trefoil y",
    (3, -1): "coma y",
    (3, 1): "coma x",
    (3, 3): "trefoil x",
    (4, 0): "spherical",
}
"""Human-readable names for the low-order terms (higher orders fall back to ``Z(n,m)``)."""


def term_label(term: Term) -> str:
    """Return a display label for *term*, e.g. ``"Z(2,0) power"``.

    Parameters
    ----------
    term : (int, int)
        ANSI ``(n, m)`` pair.

    Returns
    -------
    str
        Label combining the index and, where known, the classical name.
    """
    name = TERM_NAMES.get(term)
    n, m = term
    return f"Z({n},{m}) {name}" if name else f"Z({n},{m})"


def ansi_index(term: Term) -> int:
    """Return the single ANSI/OSA index ``j`` for ``(n, m)``.

    Parameters
    ----------
    term : (int, int)
        ANSI ``(n, m)`` pair.

    Returns
    -------
    int
        ``j = (n(n + 2) + m) / 2``.
    """
    n, m = term
    return (n * (n + 2) + m) // 2


# ---------------------------------------------------------------------------
# Polynomial evaluation
# ---------------------------------------------------------------------------


def zernike_terms(nmax: int = 6) -> List[Term]:
    """Enumerate ANSI ``(n, m)`` terms up to radial order *nmax*.

    Parameters
    ----------
    nmax : int
        Maximum radial order (inclusive).

    Returns
    -------
    list of (int, int)
        Terms in ascending ANSI index order.  Length is
        ``(nmax + 1)(nmax + 2) / 2``.

    Raises
    ------
    ValueError
        If *nmax* is negative.
    """
    if nmax < 0:
        raise ValueError(f"nmax must be >= 0, got {nmax}")
    terms = [(n, m) for n in range(nmax + 1) for m in range(-n, n + 1, 2)]
    return sorted(terms, key=ansi_index)


def zernike_radial(n: int, m: int, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the unnormalised radial polynomial ``R_n^|m|(rho)``.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal frequency (sign ignored).
    rho : NDArray
        Normalised radius in ``[0, 1]``.

    Returns
    -------
    NDArray
        ``R_n^|m|(rho)``, same shape as *rho*.

    Raises
    ------
    ValueError
        If ``n - |m|`` is odd or ``|m| > n``.
    """
    m = abs(m)
    if m > n or (n - m) % 2:
        raise ValueError(f"invalid Zernike order: n={n}, m={m}")

    out = np.zeros_like(rho, dtype=np.float64)
    for k in range((n - m) // 2 + 1):
        coeff = (
            (-1) ** k
            * factorial(n - k)
            / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
        )
        out += coeff * rho ** (n - 2 * k)
    return out


def zernike_norm(n: int, m: int) -> float:
    """Return the orthonormalisation factor for term ``(n, m)``.

    ``sqrt(n + 1)`` for ``m == 0`` and ``sqrt(2(n + 1))`` otherwise, which makes
    ``mean(Z**2) == 1`` over the unit disk.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal frequency.

    Returns
    -------
    float
        Normalisation factor.
    """
    return float(np.sqrt((n + 1) if m == 0 else 2 * (n + 1)))


def zernike(
    n: int, m: int, rho: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the orthonormal Zernike polynomial ``Z_n^m(rho, theta)``.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal frequency; positive selects the cosine lobe, negative the sine.
    rho : NDArray
        Normalised radius in ``[0, 1]``.
    theta : NDArray
        Azimuth [rad].

    Returns
    -------
    NDArray
        Polynomial values, same shape as *rho*.

    Examples
    --------
    >>> rho = np.array([0.0, 1.0])
    >>> theta = np.zeros(2)
    >>> np.allclose(zernike(0, 0, rho, theta), 1.0)
    True
    """
    radial = zernike_radial(n, m, rho) * zernike_norm(n, m)
    if m > 0:
        return radial * np.cos(m * theta)
    if m < 0:
        return radial * np.sin(abs(m) * theta)
    return radial


def zernike_design_matrix(
    rho: NDArray[np.float64],
    theta: NDArray[np.float64],
    terms: Sequence[Term],
) -> NDArray[np.float64]:
    """Build the least-squares design matrix for *terms*.

    Parameters
    ----------
    rho : NDArray
        Flat array of normalised radii for the sample points.
    theta : NDArray
        Flat array of azimuths [rad], same length as *rho*.
    terms : sequence of (int, int)
        ANSI terms, one per column.

    Returns
    -------
    NDArray
        Matrix of shape ``(len(rho), len(terms))``.
    """
    rho = np.asarray(rho, dtype=np.float64).ravel()
    theta = np.asarray(theta, dtype=np.float64).ravel()
    matrix = np.empty((rho.size, len(terms)), dtype=np.float64)
    for col, (n, m) in enumerate(terms):
        matrix[:, col] = zernike(n, m, rho, theta)
    return matrix


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class ZernikeFit:
    """Result of a least-squares Zernike fit to a wafer surface.

    Attributes
    ----------
    coefficients : dict
        ``{(n, m): value}`` in metres.  Because the basis is orthonormal each
        value is that term's RMS contribution.
    terms : list of (int, int)
        Fitted terms in ANSI order.
    nmax : int
        Maximum radial order fitted.
    radius : float
        Aperture radius used to normalise coordinates [m].
    model : NDArray
        Fitted surface [m], ``NaN`` outside the aperture.
    residual : NDArray
        ``surface - model`` [m], ``NaN`` outside the aperture.
    rms_residual : float
        RMS of the residual over the aperture [m].
    rms_surface : float
        RMS of the input surface about its mean [m].
    """

    coefficients: Dict[Term, float]
    terms: List[Term]
    nmax: int
    radius: float
    model: NDArray[np.float64]
    residual: NDArray[np.float64]
    rms_residual: float
    rms_surface: float
    _grid: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]] = field(
        repr=False, default=None  # type: ignore[assignment]
    )

    def __getitem__(self, term: Term) -> float:
        """Return the coefficient for *term* [m], or 0.0 if it was not fitted."""
        return self.coefficients.get(term, 0.0)

    @property
    def variance_explained(self) -> float:
        """Fraction of the surface variance captured by the fitted terms."""
        if self.rms_surface == 0.0:
            return 1.0
        return float(1.0 - (self.rms_residual / self.rms_surface) ** 2)

    def dominant(self, count: int = 5, skip_piston: bool = True) -> List[Tuple[Term, float]]:
        """Return the *count* largest terms by magnitude.

        Parameters
        ----------
        count : int
            Number of terms to return.
        skip_piston : bool
            Exclude piston, which carries no shape information.

        Returns
        -------
        list of ((int, int), float)
            ``(term, coefficient)`` pairs, largest magnitude first.
        """
        items = [
            (t, v) for t, v in self.coefficients.items() if not (skip_piston and t == PISTON)
        ]
        return sorted(items, key=lambda kv: abs(kv[1]), reverse=True)[:count]


def fit_zernikes(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    radius: float,
    nmax: int = 6,
    mask: Optional[NDArray[np.bool_]] = None,
) -> ZernikeFit:
    """Least-squares fit of Zernike terms to a wafer height map.

    Pixel coordinates are mapped to the unit disk via ``rho = r / radius``, and
    the fit is solved with :func:`numpy.linalg.lstsq` over the valid pixels
    only.

    Parameters
    ----------
    x, y : NDArray
        Coordinate grids [m], origin at the aperture centre.
    z : NDArray
        Height map [m]; ``NaN`` values are excluded from the fit.
    radius : float
        Aperture radius [m] used to normalise the radial coordinate.
    nmax : int
        Maximum radial order to fit.
    mask : NDArray of bool, optional
        Additional validity mask.  Defaults to "finite and inside the unit disk".

    Returns
    -------
    ZernikeFit
        Coefficients plus the reconstructed model and residual.

    Raises
    ------
    ValueError
        If fewer valid pixels remain than there are terms to fit.

    Examples
    --------
    >>> from .synthesize import make_wafer_grid
    >>> x, y, r, th, m = make_wafer_grid(96)
    >>> z = np.where(m, 3e-6 * zernike(2, 0, r / 0.15, th), np.nan)
    >>> fit = fit_zernikes(x, y, z, 0.15, nmax=4)
    >>> bool(abs(fit[(2, 0)] - 3e-6) < 1e-8)
    True
    """
    rr = np.hypot(x, y)
    rho = rr / radius
    theta = np.arctan2(y, x)

    valid = np.isfinite(z) & (rho <= 1.0)
    if mask is not None:
        valid &= mask

    terms = zernike_terms(nmax)
    n_valid = int(valid.sum())
    if n_valid < len(terms):
        raise ValueError(
            f"only {n_valid} valid pixels for {len(terms)} Zernike terms (nmax={nmax})"
        )

    design = zernike_design_matrix(rho[valid], theta[valid], terms)
    solution, *_ = np.linalg.lstsq(design, z[valid], rcond=None)
    coefficients = {t: float(v) for t, v in zip(terms, solution)}

    model_flat = design @ solution
    model = np.full(z.shape, np.nan, dtype=np.float64)
    model[valid] = model_flat
    residual = np.full(z.shape, np.nan, dtype=np.float64)
    residual[valid] = z[valid] - model_flat

    z_valid = z[valid]
    rms_surface = float(np.sqrt(np.mean((z_valid - z_valid.mean()) ** 2)))
    rms_residual = float(np.sqrt(np.mean(residual[valid] ** 2)))
    explained = 100.0 * (1.0 - (rms_residual / rms_surface) ** 2) if rms_surface else 100.0

    logger.info(
        "Zernike fit nmax=%d: %d terms, residual RMS %.3f nm (%.2f%% of variance explained)",
        nmax, len(terms), rms_residual * 1e9, explained,
    )
    return ZernikeFit(
        coefficients=coefficients,
        terms=terms,
        nmax=nmax,
        radius=radius,
        model=model,
        residual=residual,
        rms_residual=rms_residual,
        rms_surface=rms_surface,
        _grid=(x, y, valid),
    )


def fit_surface(surface: WaferSurface, nmax: int = 6) -> ZernikeFit:
    """Fit Zernike terms to a :class:`~wafer_metrology.synthesize.WaferSurface`.

    Parameters
    ----------
    surface : WaferSurface
        Surface to decompose.
    nmax : int
        Maximum radial order.

    Returns
    -------
    ZernikeFit
        The fit result.
    """
    return fit_zernikes(
        surface.x, surface.y, surface.z, surface.radius, nmax=nmax, mask=surface.mask
    )


def reconstruct(
    coefficients: Dict[Term, float],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    radius: float,
    mask: Optional[NDArray[np.bool_]] = None,
    terms: Optional[Iterable[Term]] = None,
) -> NDArray[np.float64]:
    """Rebuild a surface from Zernike coefficients.

    Parameters
    ----------
    coefficients : dict
        ``{(n, m): value}`` in metres.
    x, y : NDArray
        Coordinate grids [m].
    radius : float
        Aperture radius [m].
    mask : NDArray of bool, optional
        Where to evaluate; ``NaN`` is written elsewhere.  Defaults to the unit disk.
    terms : iterable of (int, int), optional
        Subset of terms to include.  Defaults to every key in *coefficients*.

    Returns
    -------
    NDArray
        Reconstructed height map [m].

    Examples
    --------
    >>> from .synthesize import make_wafer_grid
    >>> x, y, r, th, m = make_wafer_grid(64)
    >>> surf = reconstruct({(1, 1): 1e-6}, x, y, 0.15, m)
    >>> bool(np.nanmax(surf) > 0)
    True
    """
    rho = np.hypot(x, y) / radius
    theta = np.arctan2(y, x)
    if mask is None:
        mask = rho <= 1.0

    selected = list(terms) if terms is not None else list(coefficients.keys())
    out = np.zeros(x.shape, dtype=np.float64)
    for term in selected:
        value = coefficients.get(term, 0.0)
        if value == 0.0:
            continue
        n, m = term
        out += value * zernike(n, m, rho, theta)
    return apply_mask(out, mask)


def flatten(
    surface: WaferSurface,
    nmax: int = 6,
    remove_power: bool = False,
    extra_terms: Optional[Sequence[Term]] = None,
) -> Tuple[WaferSurface, ZernikeFit]:
    """Remove piston, tilt and optionally power from a surface.

    This is the metrology "flatten" operation: piston is an arbitrary datum,
    tilt is a chucking artefact, and power (defocus, Fringe Z4) is the
    parabolic bow that a scanner's focus control tracks out.  What remains is
    the residual warp and nanotopography that genuinely limit die-level
    flatness and fibre-to-chip alignment.

    Parameters
    ----------
    surface : WaferSurface
        Surface to flatten.
    nmax : int
        Maximum radial order used for the underlying fit.  Terms above the
        removed set are fitted but retained, so the flatten operation is
        insensitive to *nmax*.
    remove_power : bool
        Also remove power/defocus ``(2, 0)``.
    extra_terms : sequence of (int, int), optional
        Further terms to remove (e.g. astigmatism for a chuck-corrected tool).

    Returns
    -------
    (WaferSurface, ZernikeFit)
        The flattened surface and the underlying fit.

    Examples
    --------
    >>> from .synthesize import synthesize_wafer
    >>> surf = synthesize_wafer(n_pixels=128, seed=0)
    >>> flat, fit = flatten(surf, remove_power=True)
    >>> bool(flat.pv() < surf.pv())
    True
    """
    fit = fit_surface(surface, nmax=nmax)

    removed: List[Term] = [PISTON, *TILT]
    if remove_power:
        removed.append(POWER)
    if extra_terms:
        removed.extend(extra_terms)

    low_order = reconstruct(
        fit.coefficients, surface.x, surface.y, surface.radius, surface.mask, terms=removed
    )
    flattened = surface.with_z(apply_mask(surface.z - np.nan_to_num(low_order), surface.mask))
    logger.info(
        "Flattened surface: removed %s -> PV %.3f µm (was %.3f µm)",
        ", ".join(term_label(t) for t in removed), flattened.pv() * 1e6, surface.pv() * 1e6,
    )
    return flattened, fit


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_coefficient_spectrum(
    fit: ZernikeFit,
    *,
    skip_piston: bool = True,
    scale: float = 1e9,
    unit: str = "nm",
    title: str = "Zernike coefficient magnitudes",
    figsize: Tuple[float, float] = (9.0, 4.2),
) -> Tuple[Figure, Axes]:
    """Bar chart of Zernike coefficient magnitudes.

    Because the basis is orthonormal, each bar is that term's RMS contribution
    to the surface, so the chart reads as a shape error budget.

    Parameters
    ----------
    fit : ZernikeFit
        Fit to display.
    skip_piston : bool
        Drop the piston bar, which is only the arbitrary height datum.
    scale : float
        Multiplier applied to coefficients (``1e9`` -> nm).
    unit : str
        Unit label matching *scale*.
    title : str
        Axes title.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (Figure, Axes)
    """
    import matplotlib.pyplot as plt

    terms = [t for t in fit.terms if not (skip_piston and t == PISTON)]
    values = np.array([abs(fit[t]) * scale for t in terms])
    # Shape terms (removed by flattening) are drawn in a second hue so the
    # split between "correctable" and "residual" is visible at a glance.
    correctable = set(TILT) | {POWER}
    colors = [PALETTE[1] if t in correctable else PALETTE[0] for t in terms]

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(terms))
    ax.bar(positions, values, color=colors, width=0.72)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n},{m}" for n, m in terms], fontsize=7.5, rotation=90)
    ax.set_xlabel("Zernike term (n, m)")
    ax.set_ylabel(f"|coefficient| [{unit} RMS]")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE[1]),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE[0]),
    ]
    ax.legend(handles, ["tilt / power (removed by flatten)", "residual shape"], loc="upper right")

    # Direct-label only the few terms that matter, never every bar.
    for term, _ in fit.dominant(4, skip_piston=skip_piston):
        if term in terms:
            idx = terms.index(term)
            ax.annotate(
                term_label(term),
                (positions[idx], values[idx]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
                color=INK_MUTED,
            )
    fig.tight_layout()
    return fig, ax


def plot_flatten_summary(
    original: WaferSurface,
    flattened: WaferSurface,
    fit: ZernikeFit,
    *,
    title: str = "Zernike flattening",
) -> Tuple[Figure, NDArray]:
    """Three-panel figure: original surface, low-order model, flattened residual.

    Parameters
    ----------
    original : WaferSurface
        Input surface.
    flattened : WaferSurface
        Surface after :func:`flatten`.
    fit : ZernikeFit
        The fit used, supplying the low-order model panel.
    title : str
        Figure suptitle.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    removed = original.z - flattened.z
    fig, axes = plot_wafer_panels(
        original.x,
        original.y,
        [original.z, removed, flattened.z],
        [
            f"Measured surface\nPV {original.pv() * 1e6:.2f} µm",
            "Removed low-order terms\n(piston + tilt [+ power])",
            f"Flattened residual\nPV {flattened.pv() * 1e6:.3f} µm",
        ],
        scale=1e6,
        label="Height [µm]",
        signed=True,
        radius=original.radius,
        shared_scale=False,
        suptitle=title,
    )
    annotate_stats(
        axes[2],
        f"RMS {flattened.rms() * 1e9:.1f} nm\n"
        f"fit nmax {fit.nmax}\n"
        f"var expl {100 * fit.variance_explained:.2f}%",
        loc="lower left",
    )
    return fig, axes
