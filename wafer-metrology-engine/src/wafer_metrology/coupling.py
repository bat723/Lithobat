"""
Fibre-to-fibre Gaussian coupling models and the warp-to-loss capstone.

Leg C of the hardware plan measures insertion loss versus misalignment between
two SMF-28 fibres at 1550 nm; this module supplies the theory those scans are
fitted against, and the capstone analysis that convolves a *measured* wafer
height map with the loss model to price warpage in dB.

Models (identical single-mode fibres, fundamental Gaussian approximation)
-------------------------------------------------------------------------
Lateral offset ``d``:      ``L = 4.343 (d / w0)^2`` dB
Angular tilt ``theta``:    ``L = 4.343 (theta / theta0)^2`` dB,
                           ``theta0 = lambda / (pi w0)``
Longitudinal gap ``z``:    ``eta = 1 / (1 + (z / 2 z_R)^2)``,
                           ``z_R = pi w0^2 / lambda``
Lateral + gap combined:    ``eta = [4/(t^2+4)] exp(-4 (d/w0)^2 / (t^2+4))``
                           with ``t = z / z_R`` -- note the lateral tolerance
                           *widens* with gap, because the arriving beam does.

For SMF-28 at 1550 nm (``w0 = 5.2`` µm): 1 dB at 2.5 µm lateral, 3 dB at
4.3 µm, 1 dB at ~2.6 degrees of tilt, 1 dB at ~56 µm of gap.  The gap formula
is exact for Gaussian beams (derivable via the complex-q overlap); the
combined form is the standard identical-fibre result and is verified against
a numerical overlap integral in the tests.

Etalon ripple
-------------
Two flat UPC end faces form a weak Fabry-Perot cavity (R ~ 3.5 % per face).
:func:`etalon_transmission_db` gives the Airy transmission ripple versus gap:
0.61 dB peak-to-peak at R = 0.035, i.e. the plan's +/-0.3 dB Type B term.

Capstone
--------
:func:`attach_loss_budget` tiles a measured height map into die-sized sites,
fits each site's plane (via :func:`~wafer_metrology.flatness.site_planes`),
and converts site tilt and standoff error into a per-site coupling penalty:

* tilt -> angular misalignment of the facet (usually negligible),
* tilt x lever arm -> lateral offset at the attach point,
* site height deviation from the global reference plane -> gap error about
  the nominal working gap.

The result is the plan's headline sentence: a wafer of measured warp W costs
X dB at the worst site and Y dB at the median.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from .flatness import fit_reference_plane, site_planes
from .plotting import CMAP_MAGNITUDE, PALETTE, annotate_stats, wafer_imshow
from .synthesize import WaferSurface

logger = logging.getLogger(__name__)

WAVELENGTH_1550: float = 1550e-9
"""C-band test wavelength [m]."""

SMF28_W0_1550: float = 5.2e-6
"""SMF-28 mode-field *radius* at 1550 nm [m] (MFD ~ 10.4 µm)."""

FRESNEL_SILICA_AIR: float = 0.035
"""Power reflectance of a flat silica/air interface (~3.5 %)."""

_DB: float = 10.0 / np.log(10.0)
"""dB per neper of field-amplitude-squared: 4.3429..."""


# ---------------------------------------------------------------------------
# Loss models
# ---------------------------------------------------------------------------


def rayleigh_range(w0: float = SMF28_W0_1550, wavelength: float = WAVELENGTH_1550) -> float:
    """Rayleigh range ``z_R = pi w0^2 / lambda`` [m].

    Parameters
    ----------
    w0 : float
        Mode-field radius [m].
    wavelength : float
        Wavelength [m].

    Returns
    -------
    float
        Rayleigh range [m] (~55 µm for SMF-28 at 1550 nm).
    """
    return np.pi * w0**2 / wavelength


def divergence_half_angle(
    w0: float = SMF28_W0_1550, wavelength: float = WAVELENGTH_1550
) -> float:
    """Far-field divergence half-angle ``theta0 = lambda / (pi w0)`` [rad].

    Parameters
    ----------
    w0 : float
        Mode-field radius [m].
    wavelength : float
        Wavelength [m].

    Returns
    -------
    float
        Half-angle [rad] (~95 mrad for SMF-28 at 1550 nm).
    """
    return wavelength / (np.pi * w0)


def lateral_loss_db(offset, w0: float = SMF28_W0_1550):
    """Coupling loss from pure lateral offset between identical fibres.

    Parameters
    ----------
    offset : float or NDArray
        Lateral offset ``d`` [m].
    w0 : float
        Mode-field radius [m].

    Returns
    -------
    float or NDArray
        Loss [dB]: ``4.343 (d / w0)^2``.

    Examples
    --------
    >>> round(float(lateral_loss_db(2.5e-6)), 2)
    1.0
    """
    offset = np.asarray(offset, dtype=np.float64)
    return _DB * (offset / w0) ** 2


def angular_loss_db(
    tilt, w0: float = SMF28_W0_1550, wavelength: float = WAVELENGTH_1550
):
    """Coupling loss from pure angular tilt between identical fibres.

    Parameters
    ----------
    tilt : float or NDArray
        Tilt ``theta`` [rad].
    w0 : float
        Mode-field radius [m].
    wavelength : float
        Wavelength [m].

    Returns
    -------
    float or NDArray
        Loss [dB]: ``4.343 (theta / theta0)^2``.
    """
    tilt = np.asarray(tilt, dtype=np.float64)
    return _DB * (tilt / divergence_half_angle(w0, wavelength)) ** 2


def gap_loss_db(gap, w0: float = SMF28_W0_1550, wavelength: float = WAVELENGTH_1550):
    """Coupling loss from pure longitudinal gap between identical fibres.

    Exact for Gaussian beams: ``eta = 1 / (1 + (z / 2 z_R)^2)``.

    Parameters
    ----------
    gap : float or NDArray
        Face-to-face gap ``z`` [m].
    w0 : float
        Mode-field radius [m].
    wavelength : float
        Wavelength [m].

    Returns
    -------
    float or NDArray
        Loss [dB] (~1 dB at 56 µm for SMF-28 at 1550 nm).
    """
    gap = np.asarray(gap, dtype=np.float64)
    half = gap / (2.0 * rayleigh_range(w0, wavelength))
    return 10.0 * np.log10(1.0 + half**2)


def combined_loss_db(
    offset, gap, w0: float = SMF28_W0_1550, wavelength: float = WAVELENGTH_1550
):
    """Coupling loss for simultaneous lateral offset and gap.

    ``eta = [4 / (t^2 + 4)] exp(-4 (d/w0)^2 / (t^2 + 4))`` with
    ``t = z / z_R``.  Reduces to :func:`lateral_loss_db` at ``z = 0`` and to
    :func:`gap_loss_db` at ``d = 0``; verified against a numerical overlap
    integral in the tests.

    Parameters
    ----------
    offset : float or NDArray
        Lateral offset [m].
    gap : float or NDArray
        Longitudinal gap [m].
    w0 : float
        Mode-field radius [m].
    wavelength : float
        Wavelength [m].

    Returns
    -------
    float or NDArray
        Loss [dB].
    """
    offset = np.asarray(offset, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    t = gap / rayleigh_range(w0, wavelength)
    denom = t**2 + 4.0
    efficiency = (4.0 / denom) * np.exp(-4.0 * (offset / w0) ** 2 / denom)
    return -10.0 * np.log10(efficiency)


def etalon_transmission_db(
    gap,
    wavelength: float = WAVELENGTH_1550,
    reflectance: float = FRESNEL_SILICA_AIR,
):
    """Airy transmission of the weak Fabry-Perot formed by two flat end faces.

    ``T = 1 / (1 + F sin^2(2 pi z / lambda))`` with coefficient of finesse
    ``F = 4R / (1 - R)^2``.  At R = 3.5 % the ripple is 0.61 dB peak-to-peak
    (about +/-0.3 dB), with period lambda/2 in gap -- the plan's Type B term,
    and the reason to dither the gap and average.

    Parameters
    ----------
    gap : float or NDArray
        Face-to-face gap [m].
    wavelength : float
        Wavelength [m].
    reflectance : float
        Per-face power reflectance.

    Returns
    -------
    float or NDArray
        Transmission relative to the ripple maximum [dB] (always <= 0).
    """
    gap = np.asarray(gap, dtype=np.float64)
    finesse_coeff = 4.0 * reflectance / (1.0 - reflectance) ** 2
    transmission = 1.0 / (1.0 + finesse_coeff * np.sin(2.0 * np.pi * gap / wavelength) ** 2)
    return 10.0 * np.log10(transmission)


# ---------------------------------------------------------------------------
# Fitting a measured lateral scan
# ---------------------------------------------------------------------------


@dataclass
class LateralFit:
    """Result of fitting a measured lateral-offset loss scan.

    Attributes
    ----------
    w0 : float
        Fitted mode-field radius [m].
    w0_std : float
        1-sigma uncertainty of *w0* from the fit covariance [m].
    center : float
        Fitted scan-centre offset [m] (peak-up error).
    floor_db : float
        Fitted excess loss at the peak [dB] (connector + Fresnel losses).
    residual_rms_db : float
        RMS of the fit residuals [dB].
    """

    w0: float
    w0_std: float
    center: float
    floor_db: float
    residual_rms_db: float

    def model(self, offset) -> NDArray[np.float64]:
        """Evaluate the fitted parabola-in-dB at *offset* [m]."""
        return lateral_loss_db(np.asarray(offset) - self.center, self.w0) + self.floor_db


def fit_lateral_scan(
    offsets: NDArray[np.float64], loss_db: NDArray[np.float64]
) -> LateralFit:
    """Fit a measured loss-vs-lateral-offset scan to the Gaussian model.

    The model is ``L(d) = 4.343 ((d - d0) / w0)^2 + L0``: mode-field radius,
    centring error and excess-loss floor are all free, so the fitted ``w0``
    is insensitive to an imperfect peak-up or lossy connectors.

    Parameters
    ----------
    offsets : NDArray
        Commanded lateral offsets [m].
    loss_db : NDArray
        Measured loss at each offset [dB], relative to any fixed reference.

    Returns
    -------
    LateralFit
        Fitted parameters; compare ``w0`` against the nominal 5.2 µm.

    Raises
    ------
    ValueError
        If fewer than four points are given (three parameters to fit).
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    loss_db = np.asarray(loss_db, dtype=np.float64)
    if offsets.size < 4:
        raise ValueError("need at least 4 scan points to fit w0, centre and floor")

    def model(d, w0, d0, floor):
        return _DB * ((d - d0) / w0) ** 2 + floor

    p0 = (SMF28_W0_1550, 0.0, float(loss_db.min()))
    popt, pcov = curve_fit(model, offsets, loss_db, p0=p0)
    residuals = loss_db - model(offsets, *popt)
    fit = LateralFit(
        w0=abs(float(popt[0])),
        w0_std=float(np.sqrt(pcov[0, 0])),
        center=float(popt[1]),
        floor_db=float(popt[2]),
        residual_rms_db=float(np.sqrt(np.mean(residuals**2))),
    )
    logger.info(
        "Lateral-scan fit: w0 = %.2f ± %.2f µm (nominal %.2f), centre %.2f µm, "
        "floor %.2f dB, residual %.3f dB RMS",
        fit.w0 * 1e6, fit.w0_std * 1e6, SMF28_W0_1550 * 1e6,
        fit.center * 1e6, fit.floor_db, fit.residual_rms_db,
    )
    return fit


# ---------------------------------------------------------------------------
# Capstone: measured warp -> fibre-attach loss budget
# ---------------------------------------------------------------------------


@dataclass
class LossBudget:
    """Per-site fibre-attach coupling penalty derived from a measured wafer.

    Attributes
    ----------
    table : DataFrame
        One row per complete site: position, tilt, standoff error and the
        three penalty mechanisms plus their total [dB].
    penalty_map : NDArray
        Wafer-shaped map of the total penalty [dB], ``NaN`` off-site.
    worst_db, median_db, p95_db : float
        Summary statistics over complete sites [dB].
    die_size : float
        Site (die) edge length used [m].
    lever_arm : float
        Tilt-to-lateral lever arm assumed [m].
    nominal_gap : float
        Working gap the standoff error perturbs [m].
    w0 : float
        Mode-field radius used [m].
    wavelength : float
        Wavelength used [m].
    """

    table: pd.DataFrame
    penalty_map: NDArray[np.float64]
    worst_db: float
    median_db: float
    p95_db: float
    die_size: float
    lever_arm: float
    nominal_gap: float
    w0: float
    wavelength: float

    @property
    def headline(self) -> str:
        """The capstone sentence: warp priced in dB."""
        return (
            f"Across {len(self.table)} die sites ({self.die_size * 1e3:.0f} mm), the "
            f"measured warp costs up to {self.worst_db:.2f} dB of fibre-attach "
            f"penalty at the worst site, {self.median_db:.2f} dB median "
            f"(p95 {self.p95_db:.2f} dB; lever {self.lever_arm * 1e3:.0f} mm, "
            f"nominal gap {self.nominal_gap * 1e6:.0f} µm)."
        )

    def to_csv(self, path: str | Path) -> Path:
        """Write the per-site table to CSV.

        Parameters
        ----------
        path : str or Path
            Destination file.

        Returns
        -------
        Path
            The resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(out, index=False)
        logger.info("Wrote %s", out)
        return out


def attach_loss_budget(
    surface: WaferSurface,
    *,
    die_size: float = 10e-3,
    lever_arm: float = 2e-3,
    nominal_gap: float = 20e-6,
    w0: float = SMF28_W0_1550,
    wavelength: float = WAVELENGTH_1550,
    edge_exclusion: float = 3e-3,
) -> LossBudget:
    """Convolve a measured height map with the coupling model, per die site.

    The chain, mechanism by mechanism (all relative to the wafer's global
    least-squares reference plane, the proxy for the placement datum):

    * **Site tilt** ``theta`` misaligns the facet angularly -- through
      :func:`angular_loss_db`.  With theta0 ~ 95 mrad and warp-scale tilts of
      well under a milliradian this term is usually negligible, which is
      itself a result worth reporting.
    * **Tilt x lever arm** displaces the attach point laterally by
      ``theta x lever_arm`` -- through :func:`lateral_loss_db`.  The lever is
      the distance from the mechanical reference to the optical facet.
    * **Site standoff error** (the site's plane height at its centre, minus
      the global reference plane there) perturbs the working gap -- scored as
      ``gap_loss(nominal_gap + dz) - gap_loss(nominal_gap)``, clamped at zero
      gap.  A site *closer* than nominal shows a small negative penalty;
      that is real, not an error.

    Penalties are summed in dB -- the standard small-misalignment
    approximation, since the cross terms are second order in already-small
    quantities.

    Parameters
    ----------
    surface : WaferSurface
        Measured (or synthetic) wafer surface.
    die_size : float
        Die / coupler footprint edge length [m].
    lever_arm : float
        Tilt-to-lateral lever arm [m].
    nominal_gap : float
        Working gap of the attach process [m].
    w0 : float
        Mode-field radius [m].
    wavelength : float
        Wavelength [m].
    edge_exclusion : float
        Edge exclusion applied before tiling [m].

    Returns
    -------
    LossBudget
        Per-site penalties, painted map, and the headline numbers.

    Examples
    --------
    >>> from .synthesize import synthesize_wafer
    >>> s = synthesize_wafer(n_pixels=192, diameter=150e-3, seed=0)
    >>> budget = attach_loss_budget(s)
    >>> bool(budget.worst_db >= budget.median_db)
    True
    """
    x, y, z, mask = surface.x, surface.y, surface.z, surface.mask
    quality = mask & np.isfinite(z) & (np.hypot(x, y) <= surface.radius - edge_exclusion)
    _, (a0, b0, c0) = fit_reference_plane(x, y, z, quality)

    sites, site_id_map = site_planes(
        x, y, z, mask, surface.radius,
        site_size=die_size, edge_exclusion=edge_exclusion,
    )

    cx = sites["x_center_mm"].to_numpy() * 1e-3
    cy = sites["y_center_mm"].to_numpy() * 1e-3
    tilt_x = sites["tilt_x"].to_numpy() - a0
    tilt_y = sites["tilt_y"].to_numpy() - b0
    tilt = np.hypot(tilt_x, tilt_y)
    standoff = sites["height_center"].to_numpy() - (a0 * cx + b0 * cy + c0)

    loss_angular = angular_loss_db(tilt, w0, wavelength)
    loss_lateral = lateral_loss_db(tilt * lever_arm, w0)
    perturbed_gap = np.maximum(nominal_gap + standoff, 0.0)
    loss_gap = gap_loss_db(perturbed_gap, w0, wavelength) - float(
        gap_loss_db(nominal_gap, w0, wavelength)
    )
    total = loss_angular + loss_lateral + loss_gap

    table = pd.DataFrame(
        {
            "site_id": sites["site_id"],
            "x_center_mm": sites["x_center_mm"],
            "y_center_mm": sites["y_center_mm"],
            "n_points": sites["n_points"],
            "tilt_urad": tilt * 1e6,
            "standoff_um": standoff * 1e6,
            "loss_angular_db": loss_angular,
            "loss_lateral_db": loss_lateral,
            "loss_gap_db": loss_gap,
            "loss_total_db": total,
        }
    ).loc[sites["complete"].to_numpy()].reset_index(drop=True)

    if table.empty:
        raise ValueError(
            f"no complete {die_size * 1e3:.0f} mm sites on this wafer; "
            "reduce die_size or edge_exclusion"
        )

    complete_ids = set(table["site_id"].to_numpy().tolist())
    total_by_site = np.full(len(sites), np.nan)
    total_by_site[table["site_id"].to_numpy()] = table["loss_total_db"].to_numpy()
    penalty_map = np.full(x.shape, np.nan, dtype=np.float64)
    on_site = site_id_map >= 0
    penalty_map[on_site] = total_by_site[site_id_map[on_site]]

    worst = float(table["loss_total_db"].max())
    median = float(table["loss_total_db"].median())
    p95 = float(np.percentile(table["loss_total_db"], 95))

    budget = LossBudget(
        table=table, penalty_map=penalty_map,
        worst_db=worst, median_db=median, p95_db=p95,
        die_size=die_size, lever_arm=lever_arm, nominal_gap=nominal_gap,
        w0=w0, wavelength=wavelength,
    )
    logger.info("%s", budget.headline)
    return budget


def plot_loss_budget(
    surface: WaferSurface,
    budget: LossBudget,
    *,
    title: str = "Warp → fibre-attach loss budget",
    figsize: Tuple[float, float] = (12.5, 5.0),
) -> Tuple[Figure, NDArray]:
    """Per-site penalty map and its distribution, with the headline numbers.

    Parameters
    ----------
    surface : WaferSurface
        Wafer the budget was computed from (supplies the grid).
    budget : LossBudget
        Budget to display.
    title : str
        Figure suptitle.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    low = min(0.0, float(np.nanmin(budget.penalty_map)))
    high = float(np.nanmax(budget.penalty_map))
    wafer_imshow(
        axes[0], surface.x, surface.y, budget.penalty_map,
        scale=1.0, label="Attach penalty [dB]", signed=False, cmap=CMAP_MAGNITUDE,
        vlim=(low, high if high > low else low + 0.01),
        radius=surface.radius,
        title=f"Per-site penalty ({budget.die_size * 1e3:.0f} mm die)",
    )
    annotate_stats(
        axes[0],
        f"worst  {budget.worst_db:5.2f} dB\n"
        f"p95    {budget.p95_db:5.2f} dB\n"
        f"median {budget.median_db:5.2f} dB",
        loc="lower left",
    )

    values = budget.table["loss_total_db"].to_numpy()
    ax = axes[1]
    ax.hist(values, bins=min(25, max(6, values.size // 4)),
            color=PALETTE[0], edgecolor="white", linewidth=0.6)
    ax.axvline(budget.median_db, color=PALETTE[1], lw=2,
               label=f"median {budget.median_db:.2f} dB")
    ax.axvline(budget.worst_db, color=PALETTE[5], lw=2, ls="--",
               label=f"worst {budget.worst_db:.2f} dB")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Site attach penalty [dB]")
    ax.set_ylabel("Site count")
    ax.set_title("Distribution over complete sites")
    ax.legend(loc="upper right")

    fig.suptitle(
        f"{title} — λ = {budget.wavelength * 1e9:.0f} nm, "
        f"w₀ = {budget.w0 * 1e6:.1f} µm, lever {budget.lever_arm * 1e3:.0f} mm, "
        f"gap {budget.nominal_gap * 1e6:.0f} µm",
        fontsize=12,
    )
    fig.tight_layout()
    return fig, axes
