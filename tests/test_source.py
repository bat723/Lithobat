"""
Tests for the illumination source model.

The claims worth defending:

* **Quasar and CQuad are genuinely different sources.** Quasar puts its poles
  on the diagonals, CQuad on the axes. The previous disc-pole implementation
  could not express either, because a real quasar pole is an *arc*.
* **Blur and weights survive into the imaging.** The Abbe sum treats the source
  as arbitrary float weights, so soft poles need no imaging changes — but that
  only holds if rendering actually produces grey values.
* **Pruning is free accuracy-wise.** It exists purely to control the FFT count,
  and must not measurably move the aerial image.
* **A free-form source round-trips through JSON.** A numpy array in the config
  used to raise `TypeError` on save.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import GridConfig, OpticsConfig, SimulationConfig
from litho_sim.expose.aerial_image import compute_aerial_image
from litho_sim.expose.illumination import SOURCE_TYPES, build_source
from litho_sim.expose.pupil import pupil_grid
from litho_sim.expose.source import Pole, Source
from litho_sim.mask.patterns import lines_and_spaces

N = 41


@pytest.fixture(scope="module")
def grid() -> GridConfig:
    return GridConfig(n_pixels=96, pixel_size=4e-9)


@pytest.fixture(scope="module")
def dense_mask(grid) -> np.ndarray:
    """110 nm pitch — tight enough that illumination choice dominates."""
    return lines_and_spaces(grid.n_pixels, grid.pixel_size, pitch=110e-9, cd=55e-9)


def _contrast(mask, optics, grid) -> float:
    a = compute_aerial_image(mask, optics, grid)
    return float((a.max() - a.min()) / (a.max() + a.min()))


def _pole_angles(src: np.ndarray, n: int) -> list:
    """Azimuthal centroid of each connected lobe, in degrees."""
    from scipy.ndimage import center_of_mass, label

    lab, k = label(src > 0)
    out = []
    for i in range(1, k + 1):
        r, c = center_of_mass(src, lab, i)
        xi = (c - (n - 1) / 2) / ((n - 1) / 2)
        eta = (r - (n - 1) / 2) / ((n - 1) / 2)
        out.append(np.degrees(np.arctan2(eta, xi)) % 360.0)
    return sorted(out)


# ---------------------------------------------------------------------------
# Pole primitive
# ---------------------------------------------------------------------------


def test_pole_full_opening_is_a_ring():
    ring = Pole(0.6, 0.9, 0.0, 360.0).render(N)
    rho, _, _, _ = pupil_grid(N)
    assert ring[(rho >= 0.65) & (rho <= 0.85)].all()
    assert not ring[rho < 0.5].any()


def test_pole_zero_inner_is_a_disc():
    disc = Pole(0.0, 0.5, 0.0, 360.0).render(N)
    assert disc[N // 2, N // 2] > 0


def test_pole_sector_is_bounded_in_angle():
    p = Pole(0.6, 0.9, 0.0, 40.0).render(N)
    angles = _pole_angles(p, N)
    assert len(angles) == 1
    assert angles[0] == pytest.approx(0.0, abs=5.0) or angles[0] == pytest.approx(360.0, abs=5.0)


def test_pole_sector_wraps_across_zero():
    """A sector centred on 0 must not be split by the -pi/+pi branch cut."""
    p = Pole(0.6, 0.9, 0.0, 60.0).render(N)
    from scipy.ndimage import label

    _, k = label(p > 0)
    assert k == 1, f"sector centred at 0 fragmented into {k} pieces"


def test_pole_weight_is_honoured():
    a = Pole(0.6, 0.9, 0.0, 60.0, weight=1.0).render(N)
    b = Pole(0.6, 0.9, 0.0, 60.0, weight=2.5).render(N)
    assert b.max() == pytest.approx(2.5 * a.max())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sigma_inner=0.9, sigma_outer=0.6),   # inverted
        dict(sigma_outer=-1.0),                   # negative
        dict(opening_deg=400.0),                  # > 360
        dict(opening_deg=0.0),                    # zero
        dict(weight=-1.0),                        # negative weight
    ],
)
def test_pole_validates(kwargs):
    with pytest.raises(ValueError):
        Pole(**kwargs)


# ---------------------------------------------------------------------------
# Source presets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        Source.conventional(), Source.annular(), Source.monopole(),
        Source.dipole(), Source.quasar(), Source.cquad(),
    ],
)
def test_sources_are_normalised(src):
    r = src.render(N)
    assert r.sum() == pytest.approx(1.0)
    assert (r >= 0.0).all()


def _angular_distance(a: float, b: float) -> float:
    """Smallest separation between two azimuths [degrees]; 359° and 1° are 2° apart."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _assert_poles_at(src: np.ndarray, expected: tuple, tol: float = 6.0) -> None:
    angles = _pole_angles(src, N)
    assert len(angles) == len(expected), f"expected {len(expected)} poles, got {angles}"
    for e in expected:
        assert any(_angular_distance(a, e) < tol for a in angles), (
            f"no pole near {e}°; found {[round(a, 1) for a in angles]}"
        )


def test_quasar_poles_are_on_the_diagonals():
    """The defining property, and what a disc-pole implementation cannot do."""
    _assert_poles_at(Source.quasar().render(N), (45.0, 135.0, 225.0, 315.0))


def test_cquad_poles_are_on_the_axes():
    _assert_poles_at(Source.cquad().render(N), (0.0, 90.0, 180.0, 270.0))


def test_quasar_and_cquad_differ():
    assert not np.allclose(Source.quasar().render(N), Source.cquad().render(N))


def test_monopole_has_one_lobe():
    from scipy.ndimage import label

    _, k = label(Source.monopole().render(N) > 0)
    assert k == 1


def test_dipole_has_two_opposed_lobes():
    angles = _pole_angles(Source.dipole(axis="x").render(N), N)
    assert len(angles) == 2
    assert abs((angles[1] - angles[0]) - 180.0) < 6.0


def test_dipole_axis_transposes():
    """Same convention the legacy builder pins: rows = eta, cols = xi."""
    x = Source.dipole(axis="x").render(N)
    y = Source.dipole(axis="y").render(N)
    assert np.allclose(x, y.T, atol=1e-12)


def test_dipole_rejects_bad_axis():
    with pytest.raises(ValueError, match="axis must be"):
        Source.dipole(axis="z")


def test_sector_area_is_its_solid_angle():
    """A 90° sector must cover a quarter of the disc it is cut from.

    Pure geometry with no fudge factor available — if the angular test is
    wrong in any way, this number moves.
    """
    n = 201
    quarter = Source(poles=[Pole(0.0, 1.0, 0.0, 90.0)]).render(n)
    whole = Source(poles=[Pole(0.0, 1.0, 0.0, 360.0)]).render(n)
    ratio = (quarter > 0).sum() / (whole > 0).sum()
    assert ratio == pytest.approx(0.25, rel=0.02), f"90° sector covered {ratio:.3f}"


def test_sector_angle_convention_matches_the_engine():
    """Rotating a sector by 90° must equal transposing it.

    This pins rows = η / columns = ξ, the convention the Abbe loop assumes
    when it maps `col_s → fs_x` and `row_s → fs_y`. It fails if the azimuth
    is ever computed as arctan2(xi, eta).
    """
    a = Source(poles=[Pole(0.6, 0.9, 0.0, 50.0)]).render(N)
    b = Source(poles=[Pole(0.6, 0.9, 90.0, 50.0)]).render(N)
    assert np.allclose(a, b.T, atol=1e-12)


def test_blur_is_grid_independent():
    """The same blur must mean the same physical softening at any grid size.

    Catches an (n-1)/2 vs n/2 scaling error, which would otherwise only show
    up as non-monotonic convergence much later.
    """
    def second_moment(n: int) -> float:
        r = Source.monopole(blur=0.08, prune=0.0).render(n)
        rho, _, xi, eta = pupil_grid(n)
        w = r / r.sum()
        cx, cy = (w * xi).sum(), (w * eta).sum()
        return float((w * ((xi - cx) ** 2 + (eta - cy) ** 2)).sum())

    m21, m81 = second_moment(21), second_moment(81)
    assert m21 == pytest.approx(m81, rel=0.10), (
        f"blur width differs by grid: {m21:.5f} at n=21 vs {m81:.5f} at n=81"
    )


def test_source_never_illuminates_outside_the_pupil():
    """The illuminator cannot deliver light the lens will not accept."""
    s = Source(poles=[Pole(0.0, 1.5, 0.0, 360.0)])   # deliberately oversized
    rho, _, _, _ = pupil_grid(N)
    assert not (s.render(N)[rho > 1.0] > 0).any()


# ---------------------------------------------------------------------------
# Blur, weights, pruning
# ---------------------------------------------------------------------------


def test_blur_softens_and_spreads():
    sharp = Source.quasar()
    soft = Source.quasar(blur=0.06, prune=0.0)
    assert soft.n_points(N) > sharp.n_points(N)
    r = soft.render(N)
    distinct = len(np.unique(np.round(r[r > 0], 9)))
    assert distinct > 10, "blurred source should have graded weights, not binary"


def test_pruning_reduces_cost(dense_mask, grid):
    loose = Source.quasar(blur=0.06, prune=0.0)
    tight = Source.quasar(blur=0.06, prune=3e-2)
    assert tight.n_points(N) < loose.n_points(N)


def test_pruning_does_not_move_the_aerial_image(dense_mask, grid):
    """Pruning is a cost control, not an approximation you should notice.

    Measured: at 1e-3 the aerial image is unchanged to four decimal places.
    """
    unpruned = Source.quasar(blur=0.06, prune=0.0)
    pruned = Source.quasar(blur=0.06, prune=1e-3)
    a = compute_aerial_image(
        dense_mask, OpticsConfig(source_grid=N, source_spec=unpruned.to_dict()), grid
    )
    b = compute_aerial_image(
        dense_mask, OpticsConfig(source_grid=N, source_spec=pruned.to_dict()), grid
    )
    assert np.abs(a - b).max() < 1e-3


# ---------------------------------------------------------------------------
# Free-form
# ---------------------------------------------------------------------------


def test_freeform_resamples_to_the_requested_grid():
    rng = np.random.default_rng(0)
    s = Source.from_array(rng.random((15, 15)))
    assert s.render(N).shape == (N, N)
    assert s.render(25).shape == (25, 25)


def test_freeform_overrides_poles():
    s = Source(poles=[Pole(0.0, 0.3)], pixels=np.ones((11, 11)))
    r = s.render(N)
    rho, _, _, _ = pupil_grid(N)
    # A disc-only source would be dark near the pupil rim; the free-form is not.
    assert r[(rho > 0.8) & (rho <= 1.0)].sum() > 0


def test_freeform_rejects_non_2d():
    with pytest.raises(ValueError, match="must be 2-D"):
        Source.from_array(np.ones((4, 4, 4))).render(N)


def test_from_image_and_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Source.from_image(tmp_path / "nope.png")
    with pytest.raises(FileNotFoundError):
        Source.from_csv(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_source_json_round_trip(tmp_path):
    s = Source.quasar(0.9, 0.6, opening_deg=35.0, blur=0.04)
    p = tmp_path / "src.json"
    s.to_json(p)
    back = Source.from_json(p)
    assert back.poles == s.poles
    assert back.blur == s.blur
    assert np.allclose(back.render(N), s.render(N))


def test_freeform_survives_config_json(tmp_path):
    """A numpy array in OpticsConfig used to make to_json raise TypeError."""
    rng = np.random.default_rng(1)
    free = Source.from_array(rng.random((13, 13)))
    cfg = SimulationConfig(optics=OpticsConfig(source_spec=free.to_dict()))
    p = tmp_path / "cfg.json"
    cfg.to_json(p)
    back = SimulationConfig.from_json(p)
    assert np.allclose(np.asarray(back.optics.source_spec["pixels"]), free.pixels)


def test_source_from_json_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Source.from_json(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Integration with the imaging engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_every_named_type_images(source_type, dense_mask, grid):
    optics = OpticsConfig(
        sigma_outer=0.9, sigma_inner=0.6, source_type=source_type, source_grid=15
    )
    a = compute_aerial_image(dense_mask, optics, grid)
    assert np.isfinite(a).all() and a.max() == pytest.approx(1.0)


def test_source_spec_overrides_source_type(dense_mask, grid):
    """spec wins; otherwise a free-form source could be silently ignored."""
    spec = Source.dipole(axis="x").to_dict()
    with_spec = compute_aerial_image(
        dense_mask,
        OpticsConfig(source_type="conventional", source_grid=21, source_spec=spec),
        grid,
    )
    plain = compute_aerial_image(
        dense_mask, OpticsConfig(source_type="conventional", source_grid=21), grid
    )
    assert not np.allclose(with_spec, plain)


def test_dipole_beats_quad_on_dense_one_dimensional_lines(dense_mask, grid):
    """Dipole is optimal for a 1-D line array; a quad splits energy into poles
    that a single orientation cannot use.
    """
    def c(st, **kw):
        return _contrast(
            dense_mask,
            OpticsConfig(sigma_outer=0.9, sigma_inner=0.6, source_type=st,
                         source_grid=21, source_kwargs=kw),
            grid,
        )

    assert c("dipole", axis="x") > c("quasar")
    assert c("dipole", axis="x") > c("conventional")


def test_conventional_rejects_stray_kwargs():
    """Silently swallowing kwargs hid typos in source_kwargs."""
    with pytest.raises(ValueError, match="accepts no extra arguments"):
        build_source(21, "conventional", 0.8, source_kwargs_typo=1)


def test_unknown_source_type_lists_the_options():
    with pytest.raises(ValueError, match="Unknown source type"):
        build_source(21, "trefoil", 0.8)


def test_freeform_reproduces_an_analytic_source_exactly():
    """Feeding a rendered source back in as free-form pixels is a no-op.

    Proves the free-form path introduces no resampling or normalisation drift
    when the grids already match.
    """
    analytic = build_source(21, "dipole", 0.9, 0.6, axis="x")
    echoed = Source.from_array(analytic, prune=0.0).render(21)
    assert np.allclose(echoed, analytic, atol=1e-12)


def test_source_defaults_are_off():
    """The back-compatibility contract, in one assertion.

    `source_spec is None` means the engine takes the legacy path byte-for-byte.
    This fails loudly if someone helpfully flips a default.
    """
    assert OpticsConfig().source_spec is None
    assert OpticsConfig().source_type == "conventional"
    assert OpticsConfig().source_kwargs == {}
