"""
API-surface tests for the illumination subfeature of ``litho_sim.expose``.

Everything here goes through the public package interface — the named-builder
route (:data:`SOURCE_TYPES` / :func:`build_source`), the parametric route
(:class:`Source` / :class:`Pole` / :data:`SOURCE_PRESETS`), and the config
route that carries a source into :func:`compute_aerial_image`.  Deep geometry
assertions live in ``test_illumination.py`` and ``test_source.py``; these
tests pin the *contract*: shapes, normalisation, non-negativity, the
spec-dict hand-off, and the errors a caller can rely on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core import GridConfig, OpticsConfig
from litho_sim.expose import (
    SOURCE_PRESETS,
    SOURCE_TYPES,
    Pole,
    Source,
    build_source,
    compute_aerial_image,
    pupil_grid,
)
from litho_sim.mask import lines_and_spaces

N = 21  # small pupil grid — every test here must stay fast


# ---------------------------------------------------------------------------
# Named builders: build_source over every SOURCE_TYPES entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_every_named_type_builds_a_sane_source(source_type):
    """Each name yields an (n, n) float map: non-negative, unit sum, non-empty."""
    src = build_source(N, source_type, sigma_outer=0.8, sigma_inner=0.5)
    assert src.shape == (N, N)
    assert src.dtype == np.float64
    assert (src >= 0).all()
    assert src.sum() == pytest.approx(1.0)
    assert (src > 0).sum() > 0


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_named_sources_stay_within_sigma_outer(source_type):
    """No illumination outside the requested outer coherence radius."""
    src = build_source(N, source_type, sigma_outer=0.8, sigma_inner=0.5)
    rho, *_ = pupil_grid(N)
    # half a grid cell of rasterisation slack
    assert rho[src > 0].max() <= 0.8 + 2.0 / (N - 1)


def test_source_type_is_case_insensitive():
    a = build_source(N, "Annular", sigma_outer=0.8, sigma_inner=0.5)
    b = build_source(N, "annular", sigma_outer=0.8, sigma_inner=0.5)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Parametric route: SOURCE_PRESETS, Source, Pole
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(SOURCE_PRESETS))
def test_every_preset_renders(preset_name):
    """Each UI preset is a zero-argument builder producing a normalised map."""
    src = SOURCE_PRESETS[preset_name]()
    assert isinstance(src, Source)
    assert isinstance(src.name, str) and src.name
    m = src.render(N)
    assert m.shape == (N, N)
    assert (m >= 0).all()
    assert m.sum() == pytest.approx(1.0)
    assert src.n_points(N) == (m > 0).sum() > 0
    assert 0.0 < src.pupil_fill(N) <= 1.0
    assert isinstance(src.describe(), str)


def test_preset_names_map_onto_named_builder_types():
    """The two routes agree on the vocabulary: every preset name, lowercased,
    is a valid ``build_source`` type."""
    for name in SOURCE_PRESETS:
        assert name.lower() in SOURCE_TYPES


def test_source_composed_from_poles_discretises():
    """Hand-built pole sets are first-class: a rotated custom dipole renders
    to a normalised map with both lobes present."""
    poles = [
        Pole(sigma_inner=0.5, sigma_outer=0.8, angle_deg=30.0, opening_deg=40.0),
        Pole(sigma_inner=0.5, sigma_outer=0.8, angle_deg=210.0, opening_deg=40.0),
    ]
    m = Source(poles=poles, name="custom-dipole").render(N)
    assert m.shape == (N, N)
    assert m.sum() == pytest.approx(1.0)
    # two distinct lobes, one per pole, carrying roughly equal weight
    from scipy.ndimage import label

    labels, n_lobes = label(m > 0)
    assert n_lobes == 2
    w0, w1 = (m[labels == k].sum() for k in (1, 2))
    assert w0 == pytest.approx(w1, rel=0.15)


def test_spec_dict_reaches_build_source_intact():
    """The config route: ``OpticsConfig.source_spec`` is ``Source.to_dict()``,
    and ``build_source(spec=...)`` must reproduce ``Source.render`` exactly,
    overriding the named-type arguments."""
    s = Source.quasar(sigma_outer=0.85, sigma_inner=0.55)
    via_spec = build_source(
        N, "conventional", sigma_outer=0.3, spec=s.to_dict()
    )
    np.testing.assert_allclose(via_spec, s.render(N))


# ---------------------------------------------------------------------------
# End-to-end: a named source in OpticsConfig drives the aerial image
# ---------------------------------------------------------------------------


def test_named_source_drives_aerial_image_through_config():
    grid = GridConfig(n_pixels=64, pixel_size=4e-9)
    optics = OpticsConfig(
        source_type="dipole",
        sigma_outer=0.8,
        sigma_inner=0.5,
        source_grid=9,
        source_kwargs={"axis": "x"},
    )
    mask = lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=160e-9, cd=80e-9)
    img = compute_aerial_image(mask, optics, grid, dose=1.0)
    assert img.shape == (64, 64)
    assert np.isfinite(img).all()
    assert (img >= 0).all()
    # default normalisation is "peak": brightest pixel equals the dose
    assert img.max() == pytest.approx(1.0)
    # the grating must actually modulate the image
    assert img.min() < 0.5 * img.max()


# ---------------------------------------------------------------------------
# Errors and edges
# ---------------------------------------------------------------------------


def test_unknown_source_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown source type"):
        build_source(N, "donut", sigma_outer=0.8)


def test_build_source_rejects_inverted_annulus():
    with pytest.raises(ValueError, match="sigma_inner"):
        build_source(N, "annular", sigma_outer=0.5, sigma_inner=0.8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_inner": 0.9, "sigma_outer": 0.5},  # inverted radii
        {"sigma_outer": 0.0},                      # degenerate pole
        {"opening_deg": 0.0},                      # empty sector
        {"weight": -1.0},                          # negative intensity
    ],
)
def test_pole_rejects_unphysical_parameters(kwargs):
    with pytest.raises(ValueError):
        Pole(**kwargs)


def test_render_rejects_degenerate_grid():
    with pytest.raises(ValueError, match="source grid"):
        Source.conventional().render(2)
