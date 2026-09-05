"""
Roughness and failure metrics: turning trials into numbers.

The stochastic pipeline (:func:`litho_sim.develop.stochastic.stochastic_trials`)
produces realisations; this module produces the quantities a litho engineer
would actually quote from them:

* **LER / LWR** — the σ of one edge's position, and of the width, measured
  row by row down a printed line with the same sub-pixel interpolation the
  CD measurement uses. Industry convention quotes 3σ; the dataclasses store
  1σ and expose 3σ properties so nobody has to remember which this is.

  Measure these on the **continuous develop-depth field** a trial carries
  (:attr:`~litho_sim.develop.stochastic.StochasticResult.depth`), never on
  the binary image. A binary edge can only sit on a half-pixel, so its
  roughness is the grid's: at 4 nm pixels a 1.3 nm 3σ LER reads as 0.1 nm,
  at 2 nm it reads as the 1.7 nm quantisation floor whatever the physics
  did, and the correlation length collapses toward the pixel at every
  size. On the depth field the same trials give 1.3 / 1.6 / 1.6 nm at
  4 / 2 / 1 nm pixels and a 16–18 nm correlation length against a 19 nm
  acid diffusion length. :func:`trial_roughness` does the right thing.
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

from litho_sim.core.config import GridConfig
from litho_sim.develop.profile import top_loss
from litho_sim.develop.resist import feature_edges, measure_cd_2d

logger = logging.getLogger(__name__)

__all__ = [
    "edge_positions",
    "measure_line_roughness",
    "trial_roughness",
    "pool_roughness",
    "profile_roughness",
    "ProfileRoughness",
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
    *,
    feature: str = "above",
) -> CDUniformity:
    """Local CD uniformity over a ``(trials, ny, nx)`` batch.

    Measures the centre-feature CD of every trial with
    :func:`~litho_sim.develop.resist.measure_cd_2d` and reduces.  A trial
    returning zero CD printed no feature and is counted in ``n_failed``
    rather than averaged in — a vanished contact is a failure, not a very
    small contact.

    Pass the continuous depth fields with ``threshold=level`` and
    ``feature="below"`` for a sub-pixel CD; on binary images the CD is
    quantised to whole pixels and so is the uniformity.
    """
    cds = np.array([
        measure_cd_2d(t, pixel_size, axis=axis, threshold=threshold, feature=feature)
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
    """Topological defects of one trial relative to the reference print.

    ``bridges`` are reference spaces split in two by resist; ``breaks`` are
    reference lines split in two; ``residue`` counts resist components that
    overlap no reference line at all — specks left in the spaces — with
    ``residue_pixels`` their total area. Residue is real (it is what EUV
    scumming is) but it is not a break: counting a 4-pixel speck as a
    severed line put every EUV trial in the failed column.
    """

    bridges: int
    breaks: int
    residue: int = 0
    residue_pixels: int = 0

    @property
    def any(self) -> bool:
        return (self.bridges + self.breaks) > 0


def count_defects(
    binary: NDArray[np.float64],
    reference: NDArray[np.float64],
    threshold: float = 0.5,
    *,
    feature: str = "above",
) -> DefectCounts:
    """Count bridges and breaks by comparing connected components.

    Topology, not geometry: a rough edge changes no counts, while a break
    splits a resist region (one more resist component than the reference)
    and a bridge splits a clear region (one more clear component).  Both are
    counted with 4-connectivity, the conservative choice — a single-pixel
    diagonal touch does not conduct, and should not count as connected.

    A feature that vanished outright *reduces* the resist count; that case
    is caught by :func:`measure_lcdu` as a failed trial rather than here.

    *threshold* and *feature* say where the resist is: ``"above"`` 0.5 on a
    binary image (the default), ``"below"`` the film thickness on a depth
    field. Topology is the one statistic that is honestly binary, so either
    input serves.
    """
    def resist_of(field: NDArray) -> NDArray[np.bool_]:
        return field > threshold if feature == "above" else field < threshold

    resist_now = resist_of(np.asarray(binary))
    resist_ref = resist_of(np.asarray(reference))

    def splits_and_orphans(now: NDArray[np.bool_], ref: NDArray[np.bool_]) -> tuple[int, int, int]:
        """(reference regions split into several, orphan regions, orphan pixels).

        A reference region that overlaps k > 1 trial regions has been split
        k − 1 times; a trial region overlapping no reference region is an
        orphan — resist where the design has none, or a void where it has
        resist.
        """
        lab_now, n_now = ndimage.label(now)
        lab_ref, n_ref = ndimage.label(ref)
        if n_now == 0 or n_ref == 0:
            return 0, int(n_now), int(now.sum())
        pairs = np.unique(np.stack([lab_ref[now & ref], lab_now[now & ref]], axis=1), axis=0)
        per_ref = np.bincount(pairs[:, 0], minlength=n_ref + 1)[1:]
        splits = int(np.clip(per_ref - 1, 0, None).sum())
        touched = np.zeros(n_now + 1, dtype=bool)
        touched[pairs[:, 1]] = True
        touched[0] = True
        orphan_ids = np.nonzero(~touched)[0]
        orphan_px = int(np.isin(lab_now, orphan_ids).sum()) if orphan_ids.size else 0
        return splits, int(orphan_ids.size), orphan_px

    breaks, residue, residue_px = splits_and_orphans(resist_now, resist_ref)
    bridges, _voids, _void_px = splits_and_orphans(~resist_now, ~resist_ref)
    return DefectCounts(bridges=bridges, breaks=breaks, residue=residue,
                        residue_pixels=residue_px)


@dataclass
class FailureStats:
    """Stochastic failure statistics over a batch of trials.

    ``n_failed`` counts trials with a bridge or a break; residue is tallied
    separately in ``residue_total`` (specks) and ``residue_pixels``, and
    does not fail a trial on its own.
    """

    n_trials: int
    n_failed: int
    bridges_total: int
    breaks_total: int
    residue_total: int = 0
    residue_pixels: int = 0

    @property
    def rate(self) -> float:
        return self.n_failed / self.n_trials if self.n_trials else float("nan")


def failure_rate(
    trials: NDArray[np.float64],
    reference: NDArray[np.float64],
    threshold: float = 0.5,
    *,
    feature: str = "above",
) -> FailureStats:
    """Fraction of trials with at least one bridge or break vs *reference*.

    The reference is normally the deterministic ``model="car"`` print of the
    same aerial image — the design intent — so any topological difference is
    a stochastic failure.  With tens of trials this resolves rates down to
    a few percent; resolving parts-per-million tails honestly needs
    importance sampling this module does not pretend to do.
    """
    n_failed = bridges = breaks = residue = residue_px = 0
    for t in trials:
        d = count_defects(t, reference, threshold, feature=feature)
        bridges += d.bridges
        breaks += d.breaks
        residue += d.residue
        residue_px += d.residue_pixels
        if d.any:
            n_failed += 1
    return FailureStats(
        n_trials=int(trials.shape[0]),
        n_failed=n_failed,
        bridges_total=bridges,
        breaks_total=breaks,
        residue_total=residue,
        residue_pixels=residue_px,
    )


# ---------------------------------------------------------------------------
# Over a batch of trials
# ---------------------------------------------------------------------------


def pool_roughness(per_trial: list[LineRoughness]) -> LineRoughness:
    """The batch's roughness: means over the trials that have one.

    One trial's LER is itself uncertain by ±15 % at a 20 nm correlation
    length on a 128-row line, so a batch is reported as the mean. ``ξ`` can
    be NaN on a trial whose edge never decorrelates within half its length
    even when its σ is fine, so it pools over its own finite subset.
    """
    finite = [r for r in per_trial if np.isfinite(r.ler)]
    if not finite:
        nan = float("nan")
        return LineRoughness(nan, nan, nan, nan, nan, nan, 0)
    xi = [r.corr_length for r in finite if np.isfinite(r.corr_length)]
    return LineRoughness(
        ler=float(np.mean([r.ler for r in finite])),
        lwr=float(np.mean([r.lwr for r in finite])),
        sigma_left=float(np.mean([r.sigma_left for r in finite])),
        sigma_right=float(np.mean([r.sigma_right for r in finite])),
        corr_length=float(np.mean(xi)) if xi else float("nan"),
        cd_mean=float(np.mean([r.cd_mean for r in finite])),
        n_rows=int(np.mean([r.n_rows for r in finite])),
    )


def trial_roughness(result, pixel_size: float, axis: int = 1) -> LineRoughness:
    """Roughness of a stochastic batch, read off its continuous depth fields.

    The one call that does it right: sub-pixel crossings of each trial's
    cleared-depth field at the film thickness, resist on the ``"below"``
    side, pooled with :func:`pool_roughness`. *result* is a
    :class:`~litho_sim.develop.stochastic.StochasticResult`.
    """
    per_trial = [
        measure_line_roughness(d, pixel_size, axis=axis, threshold=result.level,
                               feature=result.feature)
        for d in result.depth
    ]
    return pool_roughness(per_trial)


# ---------------------------------------------------------------------------
# Through the film
# ---------------------------------------------------------------------------


@dataclass
class ProfileRoughness:
    """Roughness and failure statistics of a batch of stochastic 3-D prints.

    Everything per plane is pooled over the trials with
    :func:`pool_roughness`; everything per trial is an array over trials.
    All σ values are 1σ in metres; ``*_3s`` properties give the 3σ the
    roadmaps quote.

    Attributes
    ----------
    z_nm : NDArray
        Height of each plane above the substrate [nm].
    ler, lwr : NDArray
        Per plane, pooled over trials [m].
    cd : NDArray
        Mean printed width per plane [m] — the profile's own shape.
    corr_length : NDArray
        Per-plane correlation length [m].
    top_loss_nm : NDArray
        Per trial: resist lost from the top of the film [nm].
    footing_nm : NDArray
        Per trial: width at the substrate minus width at mid-film [nm].
        Positive is a foot, negative an undercut at the base.
    lcdu_mid : CDUniformity
        Local CD uniformity at mid-film, sub-voxel.
    bridges_base : int
        Bridges at the substrate plane — a space that failed to open between
        two lines — summed over trials against the deterministic print, when
        a reference volume was given; 0 otherwise.
    residue_base : float
        Mean fraction of the substrate area the deterministic print opens
        that a trial left covered: scumming, the base-plane failure counting
        statistics actually produce. Component counting reports each such
        speck as a "break", which is not what it is.
    fails_base : FailureStats or None
        The raw component-count statistics at the substrate plane, kept for
        the record; read ``bridges_base`` and ``residue_base`` instead.
    """

    z_nm: NDArray[np.float64]
    ler: NDArray[np.float64]
    lwr: NDArray[np.float64]
    cd: NDArray[np.float64]
    corr_length: NDArray[np.float64]
    top_loss_nm: NDArray[np.float64]
    footing_nm: NDArray[np.float64]
    lcdu_mid: CDUniformity
    bridges_base: int
    residue_base: float
    fails_base: FailureStats | None

    @property
    def ler_3s(self) -> NDArray[np.float64]:
        return 3.0 * self.ler

    @property
    def lwr_3s(self) -> NDArray[np.float64]:
        return 3.0 * self.lwr

    def summary(self) -> dict[str, float]:
        """The headline numbers, in nanometres: LWR at the top, middle and bottom."""
        n = self.z_nm.size
        pick = {"bottom": 0, "mid": n // 2, "top": n - 1}
        out = {f"lwr_3s_{k}_nm": float(self.lwr_3s[i] * 1e9) for k, i in pick.items()}
        out["top_loss_mean_nm"] = float(np.nanmean(self.top_loss_nm))
        out["top_loss_sigma_nm"] = (
            float(np.nanstd(self.top_loss_nm, ddof=1)) if self.top_loss_nm.size > 1
            else float("nan")
        )
        out["footing_mean_nm"] = float(np.nanmean(self.footing_nm))
        out["lcdu_mid_nm"] = float(self.lcdu_mid.lcdu * 1e9)
        out["residue_base_pct"] = 100.0 * self.residue_base
        out["bridges_base"] = float(self.bridges_base)
        return out


def profile_roughness(
    result,
    grid: GridConfig,
    reference: NDArray[np.float64] | None = None,
    axis: int = 1,
) -> ProfileRoughness:
    """Measure a :class:`~litho_sim.develop.stochastic.StochasticProfileResult`.

    Every plane of every trial's arrival-time field is read with sub-voxel
    crossings at the develop time; the per-plane statistics are pooled over
    the trials. *reference* is the deterministic print's arrival field (the
    same shape as one trial), against which bridges and breaks at the
    substrate are counted.
    """
    arrival, level, feature = result.arrival, result.level, result.feature
    n_trials, nz = arrival.shape[0], arrival.shape[1]
    px = grid.pixel_size

    ler = np.full(nz, np.nan)
    lwr = np.full(nz, np.nan)
    cd = np.full(nz, np.nan)
    xi = np.full(nz, np.nan)
    for iz in range(nz):
        pooled = pool_roughness([
            measure_line_roughness(arrival[t, iz], px, axis=axis, threshold=level,
                                   feature=feature)
            for t in range(n_trials)
        ])
        ler[iz], lwr[iz] = pooled.ler, pooled.lwr
        cd[iz], xi[iz] = pooled.cd_mean, pooled.corr_length

    mid = nz // 2
    top = np.array([top_loss(result.remaining[t], grid) for t in range(n_trials)])
    footing = np.full(n_trials, np.nan)
    for t in range(n_trials):
        w_bot = measure_cd_2d(arrival[t, 0], px, axis=axis, threshold=level, feature=feature)
        w_mid = measure_cd_2d(arrival[t, mid], px, axis=axis, threshold=level, feature=feature)
        if w_bot > 0.0 and w_mid > 0.0:
            footing[t] = (w_bot - w_mid) * 1e9
    lcdu_mid = measure_lcdu(arrival[:, mid], px, axis=axis, threshold=level, feature=feature)
    fails = None
    bridges = 0
    residue = float("nan")
    if reference is not None:
        fails = failure_rate(arrival[:, 0], reference[0], level, feature=feature)
        bridges = fails.bridges_total
        ref_open = reference[0] <= level if feature == "above" else reference[0] >= level
        if ref_open.any():
            covered = [
                float(((arrival[t, 0] > level) if feature == "above"
                       else (arrival[t, 0] < level))[ref_open].mean())
                for t in range(n_trials)
            ]
            residue = float(np.mean(covered))

    return ProfileRoughness(
        z_nm=np.asarray(result.z, dtype=np.float64) * 1e9,
        ler=ler, lwr=lwr, cd=cd, corr_length=xi,
        top_loss_nm=top, footing_nm=footing,
        lcdu_mid=lcdu_mid, bridges_base=int(bridges), residue_base=residue,
        fails_base=fails,
    )
