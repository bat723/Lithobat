"""
Unit tests for defect extraction.

Covers the background estimator (including the edge behaviour that a naive
smoother gets wrong), hysteresis thresholding, region measurement in both
scikit-image and scipy back-ends, and end-to-end recall against ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafer_metrology.defects import (
    HAVE_TORCH,
    detect_defects,
    hysteresis_threshold,
    label_regions,
    local_plane_background,
    region_properties,
    residual_map,
    robust_sigma,
    score_detections,
)
from wafer_metrology.synthesize import apply_mask, make_wafer_grid, synthesize_wafer

from .conftest import make_surface


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------


def test_robust_sigma_matches_std_for_gaussian_data():
    """MAD-based sigma agrees with the standard deviation on clean noise."""
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 3.0, size=200_000)
    assert robust_sigma(sample) == pytest.approx(3.0, rel=0.02)


def test_robust_sigma_ignores_outliers():
    """Large outliers barely move the estimate -- the point of using MAD."""
    rng = np.random.default_rng(1)
    clean = rng.normal(0.0, 1.0, size=10_000)
    contaminated = clean.copy()
    contaminated[:200] = 500.0                 # 2 % gross outliers

    assert robust_sigma(contaminated) == pytest.approx(robust_sigma(clean), rel=0.05)
    assert np.std(contaminated) > 5 * np.std(clean)


def test_robust_sigma_handles_empty_input():
    """All-NaN input yields NaN rather than raising."""
    assert np.isnan(robust_sigma(np.array([np.nan, np.nan])))


# ---------------------------------------------------------------------------
# Background estimation
# ---------------------------------------------------------------------------


def test_local_plane_background_reproduces_a_plane_everywhere():
    """A plane is its own background, right up to the aperture edge.

    This is the property a Gaussian-weighted *mean* loses: where the window is
    truncated by the aperture it averages a one-sided neighbourhood and returns
    a biased value, printing a false ring of defects near the edge.
    """
    x, y, r, theta, mask = make_wafer_grid(192)
    z = apply_mask(3e-6 * x / 0.15 - 1.5e-6 * y / 0.15 + 2e-6, mask)

    background = local_plane_background(z, x, y, mask, sigma_px=8.0)

    assert np.abs((background - z)[mask]).max() < 1e-12


def test_local_plane_background_has_no_edge_bias():
    """On a curved surface the residual near the edge matches the interior."""
    x, y, r, theta, mask = make_wafer_grid(256)
    z = apply_mask(40e-6 * (r / 0.15) ** 2 + 8e-6 * (x / 0.15) ** 3, mask)

    residual = z - local_plane_background(z, x, y, mask, sigma_px=10.0)

    interior = mask & (r < 0.10)
    outer = mask & (r > 0.14)
    assert np.abs(residual[outer]).max() < 5.0 * np.abs(residual[interior]).max()


def test_residual_map_flattens_a_defect_free_wafer():
    """With no defects the residual is small and centred on zero."""
    clean = synthesize_wafer(n_pixels=256, seed=0, n_particles=0, n_scratches=0)

    residual = residual_map(clean)

    assert np.isnan(residual[~clean.mask]).all()
    assert abs(float(np.nanmean(residual[clean.mask]))) < 5e-9
    assert float(np.nanmax(np.abs(residual[clean.mask]))) < 200e-9


# ---------------------------------------------------------------------------
# Thresholding and labelling
# ---------------------------------------------------------------------------


def test_hysteresis_keeps_seeded_regions_and_drops_the_rest():
    """Only low-level regions containing a high-level seed survive."""
    magnitude = np.zeros((10, 20))
    magnitude[5, 2:8] = 3.0        # weak ridge...
    magnitude[5, 4] = 10.0         # ...with a strong seed
    magnitude[5, 12:18] = 3.0      # weak ridge with no seed
    valid = np.ones_like(magnitude, dtype=bool)

    kept = hysteresis_threshold(magnitude, high=8.0, low=2.0, valid=valid)

    assert kept[5, 2:8].all()      # the whole seeded ridge is recovered
    assert not kept[5, 12:18].any()  # the unseeded one is rejected


def test_hysteresis_reconnects_a_broken_feature():
    """A ridge dipping below the seed level stays one component, not three."""
    magnitude = np.zeros((10, 30))
    magnitude[5, 5:25] = 10.0
    magnitude[5, 11] = 3.0         # two dips that a hard threshold would cut at
    magnitude[5, 18] = 3.0
    valid = np.ones_like(magnitude, dtype=bool)

    hard = magnitude >= 8.0
    soft = hysteresis_threshold(magnitude, high=8.0, low=2.0, valid=valid)

    assert label_regions(hard)[1] == 3
    assert label_regions(soft)[1] == 1


def test_hysteresis_rejects_inverted_thresholds():
    """A growth level above the seed level is a programming error."""
    with pytest.raises(ValueError):
        hysteresis_threshold(np.zeros((4, 4)), high=1.0, low=2.0,
                             valid=np.ones((4, 4), dtype=bool))


def test_hysteresis_without_seeds_returns_nothing():
    """No seed means no detections, however much weak signal there is."""
    magnitude = np.full((8, 8), 3.0)
    kept = hysteresis_threshold(magnitude, high=8.0, low=1.0,
                                valid=np.ones((8, 8), dtype=bool))
    assert not kept.any()


def test_label_regions_uses_eight_connectivity():
    """Diagonally touching pixels form one region."""
    binary = np.zeros((5, 5), dtype=bool)
    binary[1, 1] = binary[2, 2] = True

    labels, count = label_regions(binary)

    assert count == 1
    assert labels[1, 1] == labels[2, 2] == 1


def test_region_properties_measures_a_known_block():
    """Area, centroid and signed peak are reported correctly."""
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[2:5, 3:7] = 1                       # 3 x 4 block
    intensity = np.zeros((10, 10))
    intensity[3, 4] = -7.5                     # a negative peak (scratch-like)
    intensity[2, 3] = 2.0

    props = region_properties(labels, intensity)

    assert len(props) == 1
    row = props.iloc[0]
    assert row["area_px"] == 12
    assert row["row"] == pytest.approx(3.0)
    assert row["col"] == pytest.approx(4.5)
    assert row["peak_height"] == pytest.approx(-7.5)   # largest magnitude, signed
    assert row["bbox_rows"] == 3
    assert row["bbox_cols"] == 4


def test_region_properties_on_empty_input():
    """No labels yields an empty table with the documented schema."""
    props = region_properties(np.zeros((5, 5), dtype=np.int64), np.zeros((5, 5)))
    assert props.empty
    assert "peak_height" in props.columns


# ---------------------------------------------------------------------------
# End-to-end detection
# ---------------------------------------------------------------------------


def test_detects_an_injected_particle_at_the_right_place():
    """A single synthetic bump is found within a pixel of where it was put."""
    x, y, r, theta, mask = make_wafer_grid(256)
    rng = np.random.default_rng(0)
    z = 5e-9 * rng.standard_normal(x.shape)
    cx, cy, height, sigma = 0.03, -0.02, 400e-9, 1.5e-3
    z = z + height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    surface = make_surface(x, y, z, mask)

    report = detect_defects(surface)

    assert report.count == 1
    row = report.table.iloc[0]
    assert row["x_mm"] == pytest.approx(cx * 1e3, abs=1.5)
    assert row["y_mm"] == pytest.approx(cy * 1e3, abs=1.5)
    assert row["peak_height_nm"] > 0
    assert row["kind"] == "particle"


def test_clean_wafer_yields_no_detections():
    """A defect-free wafer produces no false positives, edge included."""
    clean = synthesize_wafer(n_pixels=256, seed=11, n_particles=0, n_scratches=0)
    assert detect_defects(clean).count == 0


def test_recall_on_the_full_synthetic_wafer():
    """Every injected defect is found on a noise-free surface."""
    wafer = synthesize_wafer(n_pixels=512, seed=0)

    report = detect_defects(wafer)
    scores = score_detections(report, wafer.defects)

    assert bool(scores["detected"].all())
    assert report.count == len(wafer.defects)


def test_scratches_are_classified_by_elongation():
    """Long thin negative features come back labelled as scratches."""
    wafer = synthesize_wafer(n_pixels=512, seed=0)
    report = detect_defects(wafer)

    truth_scratches = sum(1 for d in wafer.defects if d.kind == "scratch")
    assert (report.table["kind"] == "scratch").sum() == truth_scratches


def test_threshold_scales_with_sigma():
    """A stricter sigma multiple raises the threshold and finds no more."""
    wafer = synthesize_wafer(n_pixels=256, seed=0)

    loose = detect_defects(wafer, threshold_sigma=4.0)
    strict = detect_defects(wafer, threshold_sigma=9.0)

    assert strict.threshold > loose.threshold
    assert strict.count <= loose.count


def test_size_distribution_totals_the_detections():
    """Histogram counts add up to the number of defects found."""
    wafer = synthesize_wafer(n_pixels=256, seed=0)
    report = detect_defects(wafer)

    distribution = report.size_distribution()

    assert int(distribution["count"].sum()) == report.count
    assert distribution["fraction"].sum() == pytest.approx(1.0)


def test_score_detections_reports_misses():
    """Ground truth with nothing detected is scored as a miss, not an error."""
    wafer = synthesize_wafer(n_pixels=192, seed=0)
    empty = detect_defects(wafer, threshold_sigma=1e6)

    scores = score_detections(empty, wafer.defects)

    assert empty.count == 0
    assert not scores["detected"].any()
    assert len(scores) == len(wafer.defects)


# ---------------------------------------------------------------------------
# Optional ML path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_TORCH, reason="PyTorch not installed")
def test_autoencoder_flags_the_injected_defects():
    """The unsupervised model, trained only on clean wafers, still finds defects."""
    from wafer_metrology.defects import anomaly_map, detect_defects_ml, train_autoencoder

    clean = [
        synthesize_wafer(n_pixels=192, seed=100 + i, n_particles=0, n_scratches=0)
        for i in range(2)
    ]
    model, scale = train_autoencoder(clean, epochs=3, seed=0)
    wafer = synthesize_wafer(n_pixels=192, seed=0)

    scores = anomaly_map(model, wafer, scale, stride=16)
    report = detect_defects_ml(model, wafer, scale, stride=16)

    assert np.isfinite(scores[wafer.mask]).any()
    assert report.method == "autoencoder"
    # The model must react more strongly to a defective wafer than a clean one.
    clean_scores = anomaly_map(model, clean[0], scale, stride=16)
    assert np.nanmax(scores) > np.nanmax(clean_scores)


@pytest.mark.skipif(HAVE_TORCH, reason="exercises the no-torch path")
def test_ml_path_raises_a_helpful_error_without_torch():
    """Without torch the ML entry points fail loudly and usefully."""
    from wafer_metrology.defects import require_torch

    with pytest.raises(ImportError, match="PyTorch"):
        require_torch()
