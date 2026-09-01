"""Mask 3-D through the public interface, driven the way the application drives it.

The app builds an :class:`~litho_sim.core.OpticsConfig` with ``mask_model`` set
and hands it to :func:`litho_sim.expose.compute_aerial_image`; the m3d
subpackage is reached through that knob, never called directly. These tests
exercise exactly that route on tiny grids — the physics itself is pinned in
``test_m3d.py`` and ``test_fdtd.py`` and is not re-litigated here.

The FDTD model is deliberately not solved end to end: even the smallest library
build is marked ``slow`` in ``test_m3d.py`` and writes a disk cache as a side
effect. Its reachability from the public entry is covered instead by the
missing-geometry error path, which the factory raises before any solve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core import GridConfig, OpticsConfig
from litho_sim.expose import compute_aerial_image, m3d
from litho_sim.mask import lines_and_spaces

# ---------------------------------------------------------------------------
# Fixtures: the smallest EUV case the multilayer model has an opinion about
# ---------------------------------------------------------------------------


@pytest.fixture
def grid() -> GridConfig:
    """128 nm field at 2 nm — two pitches of a 32 nm half-pitch grating."""
    return GridConfig(n_pixels=64, pixel_size=2e-9)


@pytest.fixture
def mask(grid: GridConfig) -> np.ndarray:
    return lines_and_spaces(
        grid.n_pixels, grid.pixel_size, pitch=64e-9, cd=32e-9
    )


def _euv_optics(**kw) -> OpticsConfig:
    """The app's EUV configuration, small enough to run in milliseconds."""
    base = dict(
        wavelength=13.5e-9, NA=0.33, sigma_outer=0.6, source_grid=7,
        chief_ray_deg=6.0, reduction=4.0, normalisation="clear",
    )
    base.update(kw)
    return OpticsConfig(**base)


# ---------------------------------------------------------------------------
# The knob switches providers, through the same entry the app calls
# ---------------------------------------------------------------------------


def test_the_mask_model_knob_switches_the_image_through_the_public_entry(
    mask, grid
):
    """One config field is the whole coupling, and it must actually couple.

    Same mask, same optics, only ``mask_model`` differs. If the two images were
    equal the knob would be decoration — the provider seam reached from the
    public entry would be routing everything to the same spectrum.
    """
    thin = compute_aerial_image(mask, _euv_optics(mask_model="thin"), grid)
    ml = compute_aerial_image(mask, _euv_optics(mask_model="multilayer"), grid)

    for image in (thin, ml):
        assert image.shape == (grid.n_pixels, grid.n_pixels)
        assert np.isfinite(image).all()
        assert (image >= 0.0).all()
        assert image.max() > 0.0

    assert not np.allclose(thin, ml), (
        "mask_model='multilayer' produced the thin-mask image — the knob "
        "did not switch providers"
    )

    # Both are normalised against their own clear field, so they sit on the
    # same dose scale: a large ratio here means the blank bookkeeping is not
    # reaching the entry point, not that the physics changed.
    ratio = ml.max() / thin.max()
    assert 0.3 < ratio < 3.0, f"clear-field normalisation is off: ratio {ratio:.2f}"


def test_every_advertised_model_is_reachable_or_says_what_it_needs(mask, grid):
    """The models the config documents are the models the entry dispatches to.

    ``thin`` and ``multilayer`` must run outright. ``fdtd`` cannot run without
    geometry, and the requirement is that the *public entry* surfaces the
    factory's error rather than quietly falling back to a Kirchhoff screen —
    that proves the branch is wired without paying for a solve.
    """
    assert m3d.MASK_MODELS == ("thin", "multilayer", "fdtd")

    for model in ("thin", "multilayer"):
        image = compute_aerial_image(mask, _euv_optics(mask_model=model), grid)
        assert np.isfinite(image).all() and image.max() > 0.0

    with pytest.raises(ValueError, match="mask_geometry"):
        compute_aerial_image(mask, _euv_optics(mask_model="fdtd"), grid)


def test_repeated_calls_through_the_entry_are_deterministic(mask, grid):
    """A second identical call returns the same numbers, exactly.

    The provider is rebuilt per call and must carry no state between them; a
    drifting answer here would make every process-window sweep unrepeatable.
    """
    optics = _euv_optics(mask_model="multilayer")
    first = compute_aerial_image(mask, optics, grid)
    second = compute_aerial_image(mask, optics, grid)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# The provider seam, as exported by the subpackage
# ---------------------------------------------------------------------------


def test_the_provider_seam_reports_angle_dependence_honestly(mask, grid):
    """``is_angle_independent`` is the flag the Abbe loop hoists on.

    Thin says True and hands over one spectrum; multilayer says False and its
    per-source-point spectrum must actually differ from the unreflected base —
    otherwise the loop pays the per-point cost for the thin answer.
    """
    p_thin = m3d.make_spectrum_provider(mask, _euv_optics(mask_model="thin"), grid)
    assert p_thin.is_angle_independent
    base = p_thin.base_spectrum()
    assert base.shape == (grid.n_pixels, grid.n_pixels)
    assert base.dtype == np.complex128

    p_ml = m3d.make_spectrum_provider(
        mask, _euv_optics(mask_model="multilayer"), grid
    )
    assert not p_ml.is_angle_independent
    on_axis = p_ml.spectrum_for(0.0, 0.0)
    assert on_axis.shape == base.shape
    assert np.isfinite(on_axis).all()
    assert not np.allclose(on_axis, p_ml.base_spectrum()), (
        "the multilayer spectrum at the chief ray equals the unreflected "
        "base — the mirror is not being applied"
    )

    # The blank the clear-field normalisation divides by: an EUV mirror, so a
    # real reflectance strictly between nothing and everything.
    clear = p_ml.clear_intensity(0.0, 0.0)
    assert 0.0 < clear < 1.0


def test_the_subpackage_exports_resolve():
    """Everything ``litho_sim.expose.m3d`` promises in ``__all__`` exists.

    The top-level ``litho_sim.expose`` deliberately re-exports nothing from
    m3d — the subpackage is the documented public path — so its ``__all__``
    is the interface contract and every name in it must resolve.
    """
    for name in m3d.__all__:
        assert getattr(m3d, name, None) is not None, f"m3d.{name} is missing"


# ---------------------------------------------------------------------------
# Errors: bad configuration says what is wrong, at the public entry
# ---------------------------------------------------------------------------


def test_an_unknown_mask_model_is_rejected_at_the_entry(mask, grid):
    with pytest.raises(ValueError, match="mask_model must be one of"):
        compute_aerial_image(mask, _euv_optics(mask_model="rigorous"), grid)


def test_a_bad_stack_regime_is_rejected_with_the_valid_names(mask, grid):
    """A mistyped serialised stack fails loudly when the provider is built."""
    optics = _euv_optics(
        mask_model="multilayer",
        mask_stack={
            "regime": "euv",  # plausible truncation of "euv_reflective"
            "absorber": "TaBN",
            "absorber_thickness": 60e-9,
            "multilayer": None,
            "substrate": "SiO2",
        },
    )
    with pytest.raises(ValueError, match="regime must be one of"):
        compute_aerial_image(mask, optics, grid)


def test_an_unknown_mask_material_names_the_known_ones():
    with pytest.raises(KeyError, match="Unknown mask material"):
        m3d.mask_material("Unobtainium", 13.5e-9)
    with pytest.raises(KeyError, match="No mask-material table"):
        m3d.mask_material("TaBN", 100e-9)
