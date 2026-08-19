"""
Defect extraction from wafer surface maps.

Two independent paths, either of which consumes a measured (or reconstructed)
surface and returns the same :class:`DefectReport`:

**Classical** (:func:`detect_defects`) -- flatten the shape, high-pass away the
nanotopography, threshold the residual at a robust multiple of sigma, label the
connected components and measure them.  Fast, deterministic and explainable;
this is what a production tool does.

**Learned** (:func:`detect_defects_ml`) -- train a small convolutional
autoencoder on *clean* surfaces only, then flag regions the model reconstructs
badly.  Unsupervised, so it needs no defect labels and generalises to defect
types never seen in training.

The learned path needs PyTorch; the classical path needs nothing beyond
numpy/scipy.  Everything below the ``Optional ML path`` banner is skipped
cleanly when torch is absent, so importing this module always works.

scikit-image supplies ``label``/``regionprops`` when installed; a bundled
:func:`region_properties` built on :mod:`scipy.ndimage` produces the identical
table otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import ndimage as ndi

from .plotting import CMAP_MAGNITUDE, PALETTE, annotate_stats, wafer_imshow
from .synthesize import DefectTruth, WaferSurface, apply_mask
from .zernike import flatten

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whichever branch is installed
    from skimage.measure import label as _sk_label
    from skimage.measure import regionprops_table as _sk_regionprops_table

    HAVE_SKIMAGE = True
except ImportError:  # pragma: no cover
    _sk_label = None
    _sk_regionprops_table = None
    HAVE_SKIMAGE = False

try:  # pragma: no cover
    import torch
    from torch import nn

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Residual preparation
# ---------------------------------------------------------------------------


def local_plane_background(
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    mask: NDArray[np.bool_],
    sigma_px: float,
) -> NDArray[np.float64]:
    """Estimate a smooth background by Gaussian-weighted local plane fitting.

    At every pixel a plane is fitted to the neighbourhood under a Gaussian
    weight and evaluated at the centre.  Unlike a weighted *mean*, a local
    *plane* is unbiased wherever the window is truncated -- which is the whole
    outer annulus of a circular wafer.  A weighted mean there averages an
    asymmetric, one-sided neighbourhood of a sloped surface and reports a
    background biased by roughly ``slope x window offset``, printing a
    false ring of "defects" one window-width in from the edge.

    The nine weighted moments are separable Gaussian convolutions, and the
    per-pixel 3x3 systems solve as one batched call, so this costs about as
    much as a handful of smoothing passes.

    Parameters
    ----------
    z : NDArray
        Height map [m], ``NaN`` permitted outside *mask*.
    x, y : NDArray
        Coordinate grids [m].
    mask : NDArray of bool
        Valid-pixel mask.
    sigma_px : float
        Gaussian weight sigma in pixels.

    Returns
    -------
    NDArray
        Smooth background [m], ``NaN`` outside *mask*.
    """
    m = mask.astype(np.float64)
    zz = np.where(mask, np.nan_to_num(z, nan=0.0), 0.0)

    def blur(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return ndi.gaussian_filter(arr, sigma_px, mode="constant", cval=0.0)

    # Weighted moments in absolute coordinates.
    s00 = blur(m)
    s10, s01 = blur(m * x), blur(m * y)
    s20, s11, s02 = blur(m * x * x), blur(m * x * y), blur(m * y * y)
    t0 = blur(zz)
    t1, t2 = blur(zz * x), blur(zz * y)

    # Shift to coordinates local to each evaluation pixel.
    su, sv = s10 - x * s00, s01 - y * s00
    suu = s20 - 2.0 * x * s10 + x * x * s00
    svv = s02 - 2.0 * y * s01 + y * y * s00
    suv = s11 - x * s01 - y * s10 + x * y * s00
    suz, svz = t1 - x * t0, t2 - y * t0

    mat = np.stack(
        [
            np.stack([suu, suv, su], axis=-1),
            np.stack([suv, svv, sv], axis=-1),
            np.stack([su, sv, s00], axis=-1),
        ],
        axis=-2,
    )
    rhs = np.stack([suz, svz, t0], axis=-1)[..., None]

    out = np.full(z.shape, np.nan, dtype=np.float64)
    # A well-sampled window is needed for the plane to be identifiable.
    ok = mask & (s00 > 1e-3) & (np.abs(np.linalg.det(mat)) > 0.0)
    if ok.any():
        out[ok] = np.linalg.solve(mat[ok], rhs[ok])[..., 2, 0]
    return apply_mask(out, mask)


def residual_map(
    surface: WaferSurface,
    *,
    nmax: int = 6,
    highpass: float = 6e-3,
) -> NDArray[np.float64]:
    """Isolate discrete defects by removing shape and nanotopography.

    Zernike flattening removes the global shape; a Gaussian high-pass then
    removes the smooth mid-spatial-frequency nanotopography, leaving compact
    features -- particles and scratches -- standing above a near-white residual.

    Parameters
    ----------
    surface : WaferSurface
        Measured or reconstructed surface.
    nmax : int
        Maximum Zernike radial order removed as "shape".
    highpass : float
        Gaussian high-pass cut length [m].  Must exceed the defect size but
        stay below the nanotopography correlation scale times a few.

    Returns
    -------
    NDArray
        Residual height map [m], ``NaN`` outside the aperture.
    """
    flattened, _ = flatten(surface, nmax=nmax, remove_power=True)
    sigma_px = max(highpass / surface.pixel_size, 1.0)
    background = local_plane_background(
        flattened.z, surface.x, surface.y, surface.mask, sigma_px
    )
    return apply_mask(flattened.z - background, surface.mask)


def robust_sigma(values: NDArray[np.float64]) -> float:
    """Estimate the noise sigma from the median absolute deviation.

    MAD * 1.4826 matches the standard deviation for Gaussian data but is
    essentially unaffected by the defect outliers being searched for -- using a
    plain standard deviation here would let large defects raise the threshold
    that is supposed to find them.

    Parameters
    ----------
    values : NDArray
        Sample values (any shape); non-finite entries are ignored.

    Returns
    -------
    float
        Robust sigma estimate, in the units of *values*.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


# ---------------------------------------------------------------------------
# Connected-component measurement
# ---------------------------------------------------------------------------


def hysteresis_threshold(
    magnitude: NDArray[np.float64],
    high: float,
    low: float,
    valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Two-level threshold: keep low-level regions that contain a high-level seed.

    A single hard threshold breaks a shallow extended feature -- a scratch
    barely above the noise -- into a string of disconnected fragments wherever
    it dips below the cut, which then get counted and measured as separate
    defects.  Seeding at the confident level and growing at a permissive one
    recovers the whole feature without admitting noise blobs, since a
    low-level region survives only if it contains a high-level seed.

    Parameters
    ----------
    magnitude : NDArray
        Non-negative detection statistic (e.g. ``|residual|``).
    high : float
        Seed threshold; regions must exceed this somewhere to be kept.
    low : float
        Growth threshold; must be below *high*.
    valid : NDArray of bool
        Valid-pixel mask.

    Returns
    -------
    NDArray of bool
        Final foreground mask.

    Raises
    ------
    ValueError
        If *low* exceeds *high*.
    """
    if low > high:
        raise ValueError(f"low threshold {low} must not exceed high threshold {high}")

    finite = valid & np.isfinite(magnitude)
    seeds = finite & (magnitude >= high)
    grown = finite & (magnitude >= low)
    if not seeds.any():
        return seeds

    labels, _ = ndi.label(grown, structure=np.ones((3, 3), dtype=int))
    keep = np.unique(labels[seeds])
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def label_regions(binary: NDArray[np.bool_]) -> Tuple[NDArray[np.int64], int]:
    """Label connected components in a binary mask (8-connectivity).

    Uses :func:`skimage.measure.label` when available and
    :func:`scipy.ndimage.label` otherwise; both give identical labellings for
    this connectivity.

    Parameters
    ----------
    binary : NDArray of bool
        Foreground mask.

    Returns
    -------
    (NDArray, int)
        Integer label image (0 = background) and the number of labels.
    """
    if HAVE_SKIMAGE:
        labels = np.asarray(_sk_label(binary, connectivity=2), dtype=np.int64)
        return labels, int(labels.max())
    labels, count = ndi.label(binary, structure=np.ones((3, 3), dtype=int))
    return labels.astype(np.int64), int(count)


def region_properties(
    labels: NDArray[np.int64], intensity: NDArray[np.float64]
) -> pd.DataFrame:
    """Measure labelled regions.

    Parameters
    ----------
    labels : NDArray of int
        Label image, 0 = background.
    intensity : NDArray
        Signed residual height map [m] used for the intensity statistics.

    Returns
    -------
    DataFrame
        One row per label with columns ``label, area_px, row, col,
        peak_height, mean_height, bbox_rows, bbox_cols``.
    """
    n_labels = int(labels.max())
    if n_labels == 0:
        return pd.DataFrame(
            columns=["label", "area_px", "row", "col", "peak_height",
                     "mean_height", "bbox_rows", "bbox_cols"]
        )

    signed = np.nan_to_num(intensity, nan=0.0)
    index = np.arange(1, n_labels + 1)

    if HAVE_SKIMAGE:
        props = _sk_regionprops_table(
            labels,
            intensity_image=signed,
            properties=("label", "area", "centroid", "bbox", "intensity_mean"),
        )
        table = pd.DataFrame(props)
        # skimage reports the mean but not a signed peak; take it from ndimage.
        peak = _signed_peak(signed, labels, index)
        return pd.DataFrame(
            {
                "label": table["label"].to_numpy(),
                "area_px": table["area"].to_numpy(),
                "row": table["centroid-0"].to_numpy(),
                "col": table["centroid-1"].to_numpy(),
                "peak_height": peak,
                "mean_height": table["intensity_mean"].to_numpy(),
                "bbox_rows": table["bbox-2"].to_numpy() - table["bbox-0"].to_numpy(),
                "bbox_cols": table["bbox-3"].to_numpy() - table["bbox-1"].to_numpy(),
            }
        )

    # scipy fallback -- same quantities, same column names.
    area = np.asarray(ndi.sum_labels(np.ones_like(signed), labels, index))
    centroids = np.asarray(ndi.center_of_mass(np.ones_like(signed), labels, index))
    mean_height = np.asarray(ndi.mean(signed, labels, index))
    slices = ndi.find_objects(labels)
    bbox_rows = np.array([s[0].stop - s[0].start if s else 0 for s in slices])
    bbox_cols = np.array([s[1].stop - s[1].start if s else 0 for s in slices])
    return pd.DataFrame(
        {
            "label": index,
            "area_px": area,
            "row": centroids[:, 0],
            "col": centroids[:, 1],
            "peak_height": _signed_peak(signed, labels, index),
            "mean_height": mean_height,
            "bbox_rows": bbox_rows,
            "bbox_cols": bbox_cols,
        }
    )


def _signed_peak(
    signed: NDArray[np.float64], labels: NDArray[np.int64], index: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return the largest-magnitude signed value in each labelled region."""
    hi = np.asarray(ndi.maximum(signed, labels, index), dtype=np.float64)
    lo = np.asarray(ndi.minimum(signed, labels, index), dtype=np.float64)
    return np.where(np.abs(hi) >= np.abs(lo), hi, lo)


# ---------------------------------------------------------------------------
# Classical detection
# ---------------------------------------------------------------------------


@dataclass
class DefectReport:
    """Result of a defect extraction pass.

    Attributes
    ----------
    table : DataFrame
        One row per detected defect, with physical coordinates and sizes.
    labels : NDArray
        Label image over the wafer grid.
    residual : NDArray
        The residual map the detection ran on [m].
    threshold : float
        Absolute threshold applied, in the units of *residual* (metres for the
        classical path, anomaly score for the autoencoder).
    sigma : float
        Robust noise sigma the threshold was derived from -- metres for the
        classical path, log-anomaly-score units for the autoencoder.
    method : str
        ``"classical"`` or ``"autoencoder"``.
    """

    table: pd.DataFrame
    labels: NDArray[np.int64]
    residual: NDArray[np.float64]
    threshold: float
    sigma: float
    method: str

    @property
    def count(self) -> int:
        """Number of detected defects."""
        return int(len(self.table))

    def size_distribution(self, bins: Optional[Sequence[float]] = None) -> pd.DataFrame:
        """Bin the detections by equivalent diameter.

        Parameters
        ----------
        bins : sequence of float, optional
            Bin edges in millimetres.  Defaults to a decade-ish ladder from
            0.5 mm to 20 mm.

        Returns
        -------
        DataFrame
            Columns ``bin_mm, count, fraction``.
        """
        if bins is None:
            bins = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0]
        edges = np.asarray(bins, dtype=np.float64)
        diameters = self.table["equiv_diameter_mm"].to_numpy(dtype=np.float64) \
            if self.count else np.array([])
        counts, _ = np.histogram(diameters, bins=edges)
        total = max(int(counts.sum()), 1)
        return pd.DataFrame(
            {
                "bin_mm": [f"{edges[i]:g}–{edges[i + 1]:g}" for i in range(len(edges) - 1)],
                "count": counts,
                "fraction": counts / total,
            }
        )


def _empty_table() -> pd.DataFrame:
    """Return the defect table schema with no rows."""
    return pd.DataFrame(
        columns=["defect_id", "x_mm", "y_mm", "r_mm", "area_px", "area_mm2",
                 "equiv_diameter_mm", "peak_height_nm", "aspect_ratio", "kind"]
    )


def _build_table(
    props: pd.DataFrame,
    surface: WaferSurface,
    min_area_px: int,
) -> pd.DataFrame:
    """Convert raw region properties into a physical-units defect table."""
    if props.empty:
        return _empty_table()

    props = props.loc[props["area_px"] >= min_area_px].copy()
    if props.empty:
        return _empty_table()

    px = surface.pixel_size
    n = surface.n_pixels
    # Row/col -> physical coordinates on the same centred grid as make_wafer_grid.
    x_m = (props["col"].to_numpy() - (n - 1) / 2.0) * px
    y_m = (props["row"].to_numpy() - (n - 1) / 2.0) * px
    area_mm2 = props["area_px"].to_numpy() * (px * 1e3) ** 2
    equiv_d_mm = 2.0 * np.sqrt(area_mm2 / np.pi)

    rows = props["bbox_rows"].to_numpy(dtype=np.float64)
    cols = props["bbox_cols"].to_numpy(dtype=np.float64)
    aspect = np.maximum(rows, cols) / np.maximum(np.minimum(rows, cols), 1.0)

    # A long thin feature is a scratch; a compact one is a particle.  Sign
    # disambiguates the borderline cases: scratches cut down, particles sit up.
    peak = props["peak_height"].to_numpy()
    kind = np.where((aspect >= 2.5) | (peak < 0), "scratch", "particle")

    table = pd.DataFrame(
        {
            "defect_id": np.arange(1, len(props) + 1),
            "x_mm": x_m * 1e3,
            "y_mm": y_m * 1e3,
            "r_mm": np.hypot(x_m, y_m) * 1e3,
            "area_px": props["area_px"].to_numpy(),
            "area_mm2": area_mm2,
            "equiv_diameter_mm": equiv_d_mm,
            "peak_height_nm": peak * 1e9,
            "aspect_ratio": aspect,
            "kind": kind,
        }
    )
    return table.sort_values("peak_height_nm", key=np.abs, ascending=False, ignore_index=True)


def detect_defects(
    surface: WaferSurface,
    *,
    threshold_sigma: float = 5.0,
    grow_sigma: float = 2.5,
    min_area_px: int = 4,
    nmax: int = 6,
    highpass: float = 6e-3,
    residual: Optional[NDArray[np.float64]] = None,
) -> DefectReport:
    """Detect discrete defects by robust thresholding of the surface residual.

    Parameters
    ----------
    surface : WaferSurface
        Measured or reconstructed surface.
    threshold_sigma : float
        Seed threshold as a multiple of the robust residual sigma.
    grow_sigma : float
        Growth threshold for :func:`hysteresis_threshold`, as a multiple of
        sigma.  Set equal to *threshold_sigma* for a plain single threshold.
    min_area_px : int
        Reject components smaller than this, which are noise speckles.
    nmax : int
        Zernike order removed by :func:`residual_map`.
    highpass : float
        Gaussian high-pass cut length [m] for :func:`residual_map`.
    residual : NDArray, optional
        Pre-computed residual map [m]; skips :func:`residual_map` when given.

    Returns
    -------
    DefectReport
        Detections plus the intermediate maps.

    Examples
    --------
    >>> from .synthesize import synthesize_wafer
    >>> s = synthesize_wafer(n_pixels=256, seed=0)
    >>> report = detect_defects(s)
    >>> report.count > 0
    True
    """
    res = residual_map(surface, nmax=nmax, highpass=highpass) if residual is None else residual
    sigma = robust_sigma(res[surface.mask])
    threshold = threshold_sigma * sigma

    binary = hysteresis_threshold(
        np.abs(res), threshold, grow_sigma * sigma, surface.mask
    )
    labels, _ = label_regions(binary)
    props = region_properties(labels, res)
    table = _build_table(props, surface, min_area_px)

    logger.info(
        "Classical detection: sigma %.2f nm, threshold %.2f nm -> %d defects",
        sigma * 1e9, threshold * 1e9, len(table),
    )
    return DefectReport(
        table=table, labels=labels, residual=res,
        threshold=threshold, sigma=sigma, method="classical",
    )


def score_detections(
    report: DefectReport,
    truth: Sequence[DefectTruth],
    tolerance: float = 4e-3,
) -> pd.DataFrame:
    """Match detections against injected ground truth.

    Parameters
    ----------
    report : DefectReport
        Detections to score.
    truth : sequence of DefectTruth
        Injected defects from :func:`~wafer_metrology.synthesize.synthesize_wafer`.
    tolerance : float
        Maximum centre-to-centre distance for a match [m].  Scratches are
        matched with a tolerance widened to half their length, since a
        detection centroid legitimately lands anywhere along the line.

    Returns
    -------
    DataFrame
        One row per truth defect: ``kind, x_mm, y_mm, height_nm, detected,
        distance_mm``.
    """
    rows: List[dict] = []
    det_x = report.table["x_mm"].to_numpy() * 1e-3 if report.count else np.array([])
    det_y = report.table["y_mm"].to_numpy() * 1e-3 if report.count else np.array([])

    for item in truth:
        tol = tolerance + (0.5 * item.size if item.kind == "scratch" else 0.0)
        if det_x.size:
            dist = np.hypot(det_x - item.x, det_y - item.y)
            nearest = float(dist.min())
        else:
            nearest = float("inf")
        rows.append(
            {
                "kind": item.kind,
                "x_mm": item.x * 1e3,
                "y_mm": item.y * 1e3,
                "height_nm": item.height * 1e9,
                "detected": nearest <= tol,
                "distance_mm": nearest * 1e3 if np.isfinite(nearest) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        logger.info(
            "Detection score: %d/%d injected defects found",
            int(frame["detected"].sum()), len(frame),
        )
    return frame


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_defect_map(
    surface: WaferSurface,
    report: DefectReport,
    *,
    truth: Optional[Sequence[DefectTruth]] = None,
    title: str = "Defect extraction",
    figsize: Tuple[float, float] = (13.5, 4.6),
) -> Tuple[Figure, NDArray]:
    """Three-panel figure: residual, thresholded detections, size distribution.

    Parameters
    ----------
    surface : WaferSurface
        Surface analysed (supplies the coordinate grid).
    report : DefectReport
        Detections to display.
    truth : sequence of DefectTruth, optional
        Injected defects; drawn as open circles for comparison.
    title : str
        Figure suptitle.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # The classical path stores a height residual in metres; the autoencoder
    # path stores a dimensionless reconstruction error in the same field.
    height_like = report.method == "classical"
    scale_factor = 1e9 if height_like else 1.0
    unit = "nm" if height_like else ""
    signed_label = f"Residual [{unit}]" if height_like else "Anomaly score"
    magnitude_label = f"|Residual| [{unit}]" if height_like else "Anomaly score"
    first_title = (
        "Residual after shape + nanotopography removal"
        if height_like
        else "Autoencoder reconstruction error"
    )

    wafer_imshow(
        axes[0], surface.x, surface.y, report.residual,
        scale=scale_factor, label=signed_label, signed=height_like,
        cmap=None if height_like else CMAP_MAGNITUDE,
        radius=surface.radius, title=first_title,
    )
    annotate_stats(
        axes[0],
        f"σ (robust) {report.sigma * scale_factor:.3g} {unit}\n"
        f"threshold  {report.threshold * scale_factor:.3g} {unit}",
        loc="lower left",
    )

    # Detections over the detection statistic.  The colour scale is clipped to
    # twice the threshold: stretched to the defect peaks instead, the noise
    # floor collapses to one flat colour and the panel shows nothing.
    ceiling = 2.0 * report.threshold * scale_factor
    wafer_imshow(
        axes[1], surface.x, surface.y, np.abs(report.residual),
        scale=scale_factor, label=magnitude_label, signed=False, cmap=CMAP_MAGNITUDE,
        vlim=(0.0, ceiling) if np.isfinite(ceiling) and ceiling > 0 else None,
        radius=surface.radius, title=f"Detections ({report.count})",
    )
    if truth:
        axes[1].scatter(
            [t.x * 1e3 for t in truth], [t.y * 1e3 for t in truth],
            s=150, facecolors="none", edgecolors="white", linewidths=1.4,
            label="injected", zorder=3,
        )
    if report.count:
        for kind, colour, marker in (("particle", PALETTE[1], "o"), ("scratch", PALETTE[2], "s")):
            sel = report.table.loc[report.table["kind"] == kind]
            if len(sel):
                axes[1].scatter(
                    sel["x_mm"], sel["y_mm"], s=42, marker=marker,
                    facecolors="none", edgecolors=colour, linewidths=1.8,
                    label=f"{kind} ({len(sel)})", zorder=4,
                )
    axes[1].legend(loc="upper right", fontsize=8)

    # Size distribution.
    dist = report.size_distribution()
    ax = axes[2]
    ax.bar(dist["bin_mm"], dist["count"], color=PALETTE[0], edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Equivalent diameter [mm]")
    ax.set_ylabel("Defect count")
    ax.set_title("Size distribution")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # counts are integers
    for x_pos, count in zip(range(len(dist)), dist["count"]):
        if count:
            ax.annotate(str(int(count)), (x_pos, count), ha="center",
                        textcoords="offset points", xytext=(0, 3), fontsize=8)

    fig.suptitle(f"{title} — {report.method} path", fontsize=12)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Optional ML path (PyTorch)
# ---------------------------------------------------------------------------


def require_torch() -> None:
    """Raise a helpful error when the optional torch dependency is missing.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    if not HAVE_TORCH:
        raise ImportError(
            "the autoencoder defect path needs PyTorch: pip install 'wafer-metrology-engine[ml]'"
        )


if HAVE_TORCH:  # pragma: no cover - only defined when torch is installed

    class SurfaceAutoencoder(nn.Module):
        """Compact convolutional autoencoder for wafer-surface patches.

        Three stride-2 convolutions take a ``patch x patch`` tile down to 1/8
        resolution, a linear layer squeezes that to *latent* numbers, and the
        decoder mirrors the path back.

        The narrow linear bottleneck is the whole point.  Trained on clean
        surfaces the model learns the low-dimensional manifold of smooth
        nanotopography; a sharp local bump is not on that manifold and cannot be
        encoded in *latent* numbers, so it comes back smoothed and leaves a
        large reconstruction error exactly where the defect is.  A purely
        convolutional stack with no linear layer compresses a 32x32 tile only
        about two-fold -- easily enough capacity to reproduce defects faithfully,
        which destroys the anomaly signal it is supposed to produce.

        Parameters
        ----------
        patch : int
            Input tile edge length in pixels; must be a multiple of 8.
        base : int
            Channel width of the first convolution.
        latent : int
            Bottleneck width.  With the defaults this is a 64-fold compression.

        Raises
        ------
        ValueError
            If *patch* is not a positive multiple of 8.
        """

        def __init__(self, patch: int = 32, base: int = 8, latent: int = 16) -> None:
            super().__init__()
            if patch <= 0 or patch % 8:
                raise ValueError(f"patch must be a positive multiple of 8, got {patch}")
            self.patch = patch
            self.base = base
            self.latent = latent
            self._grid = patch // 8
            flat = base * 4 * self._grid * self._grid

            self.encoder = nn.Sequential(
                nn.Conv2d(1, base, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(flat, latent),
            )
            self.expand = nn.Linear(latent, flat)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base, 1, 4, stride=2, padding=1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Encode then decode *x*; returns the reconstruction."""
            code = self.encoder(x)
            grid = self.expand(code).view(-1, self.base * 4, self._grid, self._grid)
            return self.decoder(grid)

else:  # pragma: no cover
    SurfaceAutoencoder = None  # type: ignore[assignment]


def extract_patches(
    residual: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    patch: int = 32,
    stride: int = 16,
    coverage: float = 0.98,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Cut a residual map into overlapping square patches.

    Parameters
    ----------
    residual : NDArray
        Residual height map [m].
    mask : NDArray of bool
        Valid-pixel mask.
    patch : int
        Patch edge length in pixels; must be a multiple of 8 for the encoder.
    stride : int
        Step between patch origins in pixels.
    coverage : float
        Minimum fraction of valid pixels for a patch to be kept, so partial
        edge tiles never enter training.

    Returns
    -------
    (NDArray, NDArray)
        Patch stack of shape ``(n_patches, patch, patch)`` and the matching
        ``(n_patches, 2)`` array of ``(row, col)`` origins.
    """
    values = np.nan_to_num(residual, nan=0.0)
    rows: List[NDArray[np.float64]] = []
    origins: List[Tuple[int, int]] = []
    ny, nx = residual.shape
    for r0 in range(0, ny - patch + 1, stride):
        for c0 in range(0, nx - patch + 1, stride):
            tile_mask = mask[r0:r0 + patch, c0:c0 + patch]
            if tile_mask.mean() < coverage:
                continue
            rows.append(values[r0:r0 + patch, c0:c0 + patch])
            origins.append((r0, c0))
    if not rows:
        return np.empty((0, patch, patch)), np.empty((0, 2), dtype=np.int64)
    return np.stack(rows), np.asarray(origins, dtype=np.int64)


def train_autoencoder(
    clean_surfaces: Sequence[WaferSurface],
    *,
    patch: int = 32,
    stride: int = 16,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    base: int = 8,
    latent: int = 16,
    scale: Optional[float] = None,
    seed: int = 0,
) -> Tuple["SurfaceAutoencoder", float]:  # type: ignore[valid-type]
    """Train the autoencoder on defect-free surfaces.

    Parameters
    ----------
    clean_surfaces : sequence of WaferSurface
        Surfaces with no injected defects.  Generate them with
        ``synthesize_wafer(n_particles=0, n_scratches=0, seed=...)``.
    patch : int
        Patch edge length in pixels.
    stride : int
        Patch stride in pixels.
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Adam learning rate.
    base : int
        Encoder channel width.
    latent : int
        Bottleneck width; smaller forces the model to generalise harder and
        sharpens the anomaly signal.
    scale : float, optional
        Height normalisation [m].  Defaults to the training-set robust sigma,
        which puts the input in units of "noise sigma".
    seed : int
        Torch/NumPy seed for reproducible training.

    Returns
    -------
    (SurfaceAutoencoder, float)
        The trained model in eval mode and the *scale* used, which must be
        passed to :func:`anomaly_map` for consistent normalisation.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    ValueError
        If the surfaces yield no usable patches.
    """
    require_torch()
    torch.manual_seed(seed)

    stacks = []
    for surface in clean_surfaces:
        res = residual_map(surface)
        tiles, _ = extract_patches(res, surface.mask, patch=patch, stride=stride)
        if tiles.size:
            stacks.append(tiles)
    if not stacks:
        raise ValueError("no usable patches extracted from the clean surfaces")
    data = np.concatenate(stacks, axis=0)

    if scale is None:
        scale = robust_sigma(data)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(data)) or 1.0

    tensor = torch.from_numpy((data / scale).astype(np.float32)).unsqueeze(1)
    model = SurfaceAutoencoder(patch=patch, base=base, latent=latent)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    n = tensor.shape[0]
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch_size):
            batch = tensor[order[start:start + batch_size]]
            optimiser.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimiser.step()
            total += loss.detach().item() * batch.shape[0]
        logger.info("Autoencoder epoch %2d/%d: MSE %.4f", epoch + 1, epochs, total / n)

    model.eval()
    logger.info("Autoencoder trained on %d patches from %d clean wafers",
                n, len(clean_surfaces))
    return model, float(scale)


def anomaly_map(
    model: "SurfaceAutoencoder",  # type: ignore[valid-type]
    surface: WaferSurface,
    scale: float,
    *,
    patch: int = 32,
    stride: int = 8,
    pool_px: float = 1.0,
    residual: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Per-pixel reconstruction error of the autoencoder.

    Overlapping patches are averaged, so the map is smooth and free of tile
    seams, and the result is then pooled over a small neighbourhood.

    Parameters
    ----------
    model : SurfaceAutoencoder
        Trained model.
    surface : WaferSurface
        Surface to score.
    scale : float
        Normalisation returned by :func:`train_autoencoder`.
    patch : int
        Patch edge length in pixels; must match training.
    stride : int
        Patch stride; smaller than training stride gives a smoother map.
    pool_px : float
        Gaussian pooling sigma in pixels applied to the squared-error map.
        A raw per-pixel squared error is a one-degree-of-freedom chi-square:
        its *background* spread is enormous, which swamps the threshold even
        though a defect stands 400-fold above the noise floor.  Pooling over a
        couple of pixels averages many degrees of freedom together, collapsing
        the background spread while leaving a spatially coherent defect intact
        -- the robust sigma of the log score falls roughly four-fold, and the
        strongest defect goes from 3 sigma to over 10.  Set ``0`` to disable.
    residual : NDArray, optional
        Pre-computed residual map [m].

    Returns
    -------
    NDArray
        Squared reconstruction error per pixel (dimensionless, in units of
        normalised height squared), ``NaN`` outside the aperture.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    require_torch()
    res = residual_map(surface) if residual is None else residual
    tiles, origins = extract_patches(res, surface.mask, patch=patch, stride=stride)
    accum = np.zeros(res.shape, dtype=np.float64)
    weight = np.zeros(res.shape, dtype=np.float64)

    if tiles.size:
        tensor = torch.from_numpy((tiles / scale).astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            recon = model(tensor)
        error = ((recon - tensor) ** 2).squeeze(1).numpy()
        for (r0, c0), err in zip(origins, error):
            accum[r0:r0 + patch, c0:c0 + patch] += err
            weight[r0:r0 + patch, c0:c0 + patch] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(weight > 0, accum / weight, np.nan)

    if pool_px > 0.0:
        covered = np.isfinite(out) & surface.mask
        pooled = ndi.gaussian_filter(np.where(covered, out, 0.0), pool_px)
        norm = ndi.gaussian_filter(covered.astype(np.float64), pool_px)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(norm > 1e-9, pooled / norm, np.nan)

    return apply_mask(out, surface.mask)


def detect_defects_ml(
    model: "SurfaceAutoencoder",  # type: ignore[valid-type]
    surface: WaferSurface,
    scale: float,
    *,
    threshold_sigma: float = 5.0,
    min_area_px: int = 4,
    patch: int = 32,
    stride: int = 8,
) -> DefectReport:
    """Detect defects from autoencoder reconstruction error.

    The model must have been trained on clean surfaces that went through the
    *same* processing as *surface*.  Training on ideal surfaces and scoring a
    noisy measured one makes every pixel look anomalous, and the detector
    returns either everything or nothing.

    Parameters
    ----------
    model : SurfaceAutoencoder
        Model trained on clean surfaces.
    surface : WaferSurface
        Surface to inspect.
    scale : float
        Normalisation from :func:`train_autoencoder`.
    threshold_sigma : float
        Threshold as a multiple of the robust sigma of the **log** anomaly
        score, above its median.
    min_area_px : int
        Minimum component area in pixels.
    patch, stride : int
        Patch geometry; *patch* must match training.

    Returns
    -------
    DefectReport
        Detections, with ``method="autoencoder"``, *residual* holding the
        anomaly map rather than a height map, and *sigma* expressed in log-score
        units.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Notes
    -----
    The threshold is set on ``log(score)``, not on the score itself.  The
    anomaly map is a mean of squared errors, so it is chi-square-like: strictly
    positive, sharply peaked and very heavy tailed -- on a typical wafer its
    median and maximum differ by four orders of magnitude.  A ``median + k
    sigma`` cut on that raw distribution sits far inside the bulk and flags
    several percent of all pixels.  Taking the log makes the noise
    approximately symmetric, so the robust estimator means what it says.
    """
    require_torch()
    res = residual_map(surface)
    scores = anomaly_map(model, surface, scale, patch=patch, stride=stride, residual=res)

    inside = scores[surface.mask]
    inside = inside[np.isfinite(inside)]
    floor = 1e-12
    log_inside = np.log(np.maximum(inside, floor))
    sigma = robust_sigma(log_inside)
    log_threshold = float(np.median(log_inside) + threshold_sigma * sigma)
    threshold = float(np.exp(log_threshold))

    binary = surface.mask & np.isfinite(scores) & (scores >= threshold)
    labels, _ = label_regions(binary)
    # Measure the physical residual inside each anomalous region, so heights
    # stay in metres and the table matches the classical path.
    props = region_properties(labels, res)
    table = _build_table(props, surface, min_area_px)

    logger.info(
        "Autoencoder detection: threshold %.4g (log-median + %.1f sigma) -> %d defects",
        threshold, threshold_sigma, len(table),
    )
    return DefectReport(
        table=table, labels=labels, residual=scores,
        threshold=threshold, sigma=sigma, method="autoencoder",
    )
