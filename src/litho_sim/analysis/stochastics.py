"""
Roughness and failure metrics: turning trials into numbers.

The stochastic pipeline (:func:`litho_sim.develop.stochastic.stochastic_trials`)
produces realisations; this module produces the quantities a litho engineer
would actually quote from them:

* **LER / LWR** — the σ of one edge's position, and of the width, measured
  row by row down a printed line with the same sub-pixel interpolation the
  CD measurement uses. Industry convention quotes 3σ; the dataclasses store
  1σ and expose 3σ properties so nobody has to remember which this is.
* **Correlation length** — where the edge's autocorrelation falls to 1/e.
  The number that distinguishes physical roughness from pixel noise, and the
  one a cosmetic model has to be *told* while the physical model produces it.
* **LCDU** — local CD uniformity: the spread of the same feature's CD across
  trials. This is the metric EUV lives and dies by, because at constant dose
  it shrinks as ``1/sqrt(photons per feature)`` and nothing else in the
  process can buy it back.
* **Stochastic failures** — bridges and breaks, counted by comparing the
  connected components of a trial against the deterministic reference.
  Failure *rates* are the tail statistics: a resist can have acceptable LER
  and still kill a die once per billion contacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from litho_sim.develop.resist import feature_edges, measure_cd_2d

logger = logging.getLogger(__name__)

__all__ = [
    "edge_positions",
    "measure_line_roughness",
    "LineRoughness",
    "measure_lcdu",
    "CDUniformity",
    "count_defects",
    "DefectCounts",
    "failure_rate",
    "FailureStats",
]


def edge_positions(
    image: NDArray[np.float64],
    pixel_size: float,
    axis: int = 1,
    threshold: float = 0.5,
    *,
    feature: str = "above",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Left and right edge positions of the centre feature, row by row [m].

    Each cutline perpendicular to the feature is measured independently with
    the same interpolated crossing logic as
    :func:`~litho_sim.develop.resist.measure_cd_1d`, so the positions are
    sub-pixel.  Rows where the feature is missing — a broken line — carry
    NaN, which downstream statistics must (and do) mask rather than average.

    Parameters
    ----------
    image : NDArray
        2-D resist image, binary or continuous.
    pixel_size : float
        Physical pixel size [m].
    axis : int
        1 → the feature runs vertically, measure each row (the default);
        0 → the feature runs horizontally, measure each column.
    threshold, feature :
        Forwarded to :func:`~litho_sim.develop.resist.feature_edges`.

    Returns
    -------
    (left, right) : tuple of NDArray
        Edge positions along the feature, each of length ``image.shape[axis
        == 1 ? 0 : 1]``, in metres from the array origin.
    """
    data = image if axis == 1 else image.T
    n_rows = data.shape[0]
    left = np.full(n_rows, np.nan)
    right = np.full(n_rows, np.nan)
    for i in range(n_rows):
        edges = feature_edges(data[i], threshold, feature=feature)
        if edges is not None:
            left[i], right[i] = edges
    return left * pixel_size, right * pixel_size


def _correlation_length(positions: NDArray[np.float64], step: float) -> float:
    """1/e decay length of a position series' autocorrelation [m].

    Linear interpolation between the bracketing lags; NaN when the series is
    too short or never decorrelates within half its length (beyond that the
    estimate is mostly noise).
    """
    x = positions - positions.mean()
    var = float(np.mean(x * x))
    n = x.size
    if var <= 0.0 or n < 8:
        return float("nan")
    target = 1.0 / np.e
    prev = 1.0
    for k in range(1, n // 2):
        acf = float(np.mean(x[:-k] * x[k:])) / var
        if acf < target:
            # Interpolate between lag k-1 (prev) and lag k (acf).
            frac = (prev - target) / (prev - acf) if prev != acf else 0.0
            return (k - 1 + frac) * step
        prev = acf
    return float("nan")


@dataclass
class LineRoughness:
    """Roughness of one printed line.  All σ values are 1σ, in metres.

    ``ler`` pools both edges; multiply by 3 (or use the ``*_3s`` properties)
    for the 3σ numbers papers and roadmaps quote.  ``corr_length`` is the 1/e
    autocorrelation length averaged over the edges.  ``n_rows`` counts the
    cutlines where the feature existed — fewer than the image height means
    the line broke somewhere.
    """

    ler: float
    lwr: float
    sigma_left: float
    sigma_right: float
    corr_length: float
    cd_mean: float
    n_rows: int

    @property
    def ler_3s(self) -> float:
        return 3.0 * self.ler

    @property
    def lwr_3s(self) -> float:
        return 3.0 * self.lwr


def measure_line_roughness(
    image: NDArray[np.float64],
    pixel_size: float,
    axis: int = 1,
    threshold: float = 0.5,
    *,
    feature: str = "above",
) -> LineRoughness:
    """Measure LER, LWR and correlation length of the centre feature.

    Statistics use ``ddof=1`` — these are estimates from a finite sample of
    rows, and at 128 rows the biased estimator would flatter the resist by
    half a percent for no reason.
    """
    left, right = edge_positions(
        image, pixel_size, axis=axis, threshold=threshold, feature=feature
    )
    ok = ~(np.isnan(left) | np.isnan(right))
    n_rows = int(ok.sum())
    if n_rows < 2:
        nan = float("nan")
        return LineRoughness(nan, nan, nan, nan, nan, nan, n_rows)

    xl, xr = left[ok], right[ok]
    width = xr - xl
    sigma_left = float(np.std(xl, ddof=1))
    sigma_right = float(np.std(xr, ddof=1))
    ler = float(np.sqrt(0.5 * (sigma_left**2 + sigma_right**2)))
    lwr = float(np.std(width, ddof=1))

    step = pixel_size  # consecutive rows are one pixel apart along the line
    xi = np.array([_correlation_length(xl, step), _correlation_length(xr, step)])
    xi = xi[~np.isnan(xi)]
    corr = float(xi.mean()) if xi.size else float("nan")

    return LineRoughness(
        ler=ler,
        lwr=lwr,
        sigma_left=sigma_left,
        sigma_right=sigma_right,
        corr_length=corr,
        cd_mean=float(width.mean()),
        n_rows=n_rows,
    )


@dataclass
class CDUniformity:
    """CD statistics of the same feature across stochastic trials.

    ``lcdu`` follows the industry convention: 3σ of the CD distribution.
    ``n_failed`` counts trials where no measurable feature printed at all —
    those cannot contribute a CD and are the extreme tail made visible.
    """

    cd_mean: float
    cd_sigma: float
    n_trials: int
    n_failed: int
    cds: NDArray[np.float64]

    @property
    def lcdu(self) -> float:
        return 3.0 * self.cd_sigma


def measure_lcdu(
    trials: NDArray[np.float64],
    pixel_size: float,
    axis: int = 1,
    threshold: float = 0.5,
) -> CDUniformity:
    """Local CD uniformity over a ``(trials, ny, nx)`` batch.

    Measures the centre-feature CD of every trial with
    :func:`~litho_sim.develop.resist.measure_cd_2d` and reduces.  A trial
    returning zero CD printed no feature and is counted in ``n_failed``
    rather than averaged in — a vanished contact is a failure, not a very
    small contact.
    """
    cds = np.array([
        measure_cd_2d(t, pixel_size, axis=axis, threshold=threshold)
        for t in trials
    ])
    printed = cds > 0.0
    n_failed = int((~printed).sum())
    good = cds[printed]
    if good.size >= 2:
        mean, sigma = float(good.mean()), float(np.std(good, ddof=1))
    elif good.size == 1:
        mean, sigma = float(good[0]), float("nan")
    else:
        mean = sigma = float("nan")
    return CDUniformity(
        cd_mean=mean,
        cd_sigma=sigma,
        n_trials=int(trials.shape[0]),
        n_failed=n_failed,
        cds=cds,
    )


@dataclass
class DefectCounts:
    """Topological defects of one trial relative to the reference print."""

    bridges: int
    breaks: int

    @property
    def any(self) -> bool:
        return (self.bridges + self.breaks) > 0


def count_defects(
    binary: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> DefectCounts:
    """Count bridges and breaks by comparing connected components.

    Topology, not geometry: a rough edge changes no counts, while a break
    splits a resist region (one more resist component than the reference)
    and a bridge splits a clear region (one more clear component).  Both are
    counted with 4-connectivity, the conservative choice — a single-pixel
    diagonal touch does not conduct, and should not count as connected.

    A feature that vanished outright *reduces* the resist count; that case
    is caught by :func:`measure_lcdu` as a failed trial rather than here.
    """
    def n_components(mask: NDArray[np.bool_]) -> int:
        return int(ndimage.label(mask)[1])

    resist_now = binary > 0.5
    resist_ref = reference > 0.5
    breaks = max(0, n_components(resist_now) - n_components(resist_ref))
    bridges = max(0, n_components(~resist_now) - n_components(~resist_ref))
    return DefectCounts(bridges=bridges, breaks=breaks)


@dataclass
class FailureStats:
    """Stochastic failure statistics over a batch of trials."""

    n_trials: int
    n_failed: int
    bridges_total: int
    breaks_total: int

    @property
    def rate(self) -> float:
        return self.n_failed / self.n_trials if self.n_trials else float("nan")


def failure_rate(
    trials: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> FailureStats:
    """Fraction of trials with at least one bridge or break vs *reference*.

    The reference is normally the deterministic ``model="car"`` print of the
    same aerial image — the design intent — so any topological difference is
    a stochastic failure.  With tens of trials this resolves rates down to
    a few percent; resolving parts-per-million tails honestly needs
    importance sampling this module does not pretend to do.
    """
    n_failed = bridges = breaks = 0
    for t in trials:
        d = count_defects(t, reference)
        bridges += d.bridges
        breaks += d.breaks
        if d.any:
            n_failed += 1
    return FailureStats(
        n_trials=int(trials.shape[0]),
        n_failed=n_failed,
        bridges_total=bridges,
        breaks_total=breaks,
    )
