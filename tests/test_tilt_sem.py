"""
The tilt-stage SEM view: the physics on synthetic geometry, the geometry
pass against a volume whose surface is known, and the app view that puts
them together.

The physics tests need no renderer: :class:`GeometryBuffers` is plain
arrays, so a flat plane, a tilted plane, a step and a trench are built by
hand and each term of :func:`tilt_signal` is pinned on its own. The
geometry tests want VTK and skip without it — as CI does.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.core.config import GridConfig
from litho_sim.metrology.sem import (
    VACUUM_YIELD,
    GeometryBuffers,
    SEMConfig,
    screen_occlusion,
    tilt_sem,
    tilt_signal,
)

DOWN = np.array([0.0, 0.0, -1.0])      # the beam, looking straight down
UP = np.array([0.0, 1.0, 0.0])
RIGHT = np.array([1.0, 0.0, 0.0])


def _buffers(rows=32, cols=32, *, normal=(0.0, 0.0, 1.0), material=2, depth=0.0,
             pixel_nm=1.0, film_nm=100.0) -> GeometryBuffers:
    """A uniform patch of surface: one normal, one material, one depth."""
    n = np.broadcast_to(np.asarray(normal, dtype=float), (rows, cols, 3)).copy()
    d = np.full((rows, cols), float(depth))
    m = np.full((rows, cols), material, dtype=np.uint8)
    return GeometryBuffers(normals=n, depth=d, material=m, view=DOWN, up=UP, right=RIGHT,
                           focal=np.zeros(3), pixel_nm=pixel_nm, film_nm=film_nm)


def _plain() -> SEMConfig:
    """Every angular term off, no blur, no spread: the base yield alone."""
    return SEMConfig(beam_fwhm=0.0, escape_length=0.0, directionality=0.0,
                     shadowing=0.0, electrons_per_pixel=4000.0, frames=4)


# ---------------------------------------------------------------------------
# The terms, one at a time
# ---------------------------------------------------------------------------


def test_a_flat_resist_top_facing_the_beam_emits_one():
    sig = tilt_signal(_buffers(), _plain())
    assert np.allclose(sig, 1.0)


def test_the_substrate_reads_at_its_yield_and_the_vacuum_at_its_own():
    cfg = _plain()
    assert np.allclose(tilt_signal(_buffers(material=1), cfg), cfg.substrate_yield)
    b = _buffers(material=0)
    b.depth[:] = np.nan
    assert np.allclose(tilt_signal(b, cfg), VACUUM_YIELD)


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 60.0])
def test_a_surface_turned_from_the_beam_follows_the_secant_law(theta_deg):
    """Seiler: δ ∝ sec^n θ. A plane tilted by θ from the beam, seen
    everywhere at the same depth so no edge term enters."""
    t = np.radians(theta_deg)
    cfg = _plain()
    sig = tilt_signal(_buffers(normal=(np.sin(t), 0.0, np.cos(t))), cfg)
    assert np.allclose(sig, np.cos(t) ** (-cfg.secant_power), rtol=1e-6)


def test_the_secant_law_saturates_at_grazing_incidence():
    sig = tilt_signal(_buffers(normal=(1.0, 0.0, 0.0)), _plain())
    assert np.all(np.isfinite(sig))
    assert np.allclose(sig, 4.0), "capped, not divergent"


def test_a_full_height_silhouette_blooms_exactly_the_edge_yield():
    """A depth step of one film thickness between two columns: the two
    pixels either side each see half the wall, and together score
    ``edge_yield`` — the same bookkeeping as the top-down model."""
    cfg = _plain()
    b = _buffers(film_nm=100.0)
    b.depth[:, 16:] = 100.0                     # the right half is a film lower
    sig = tilt_signal(b, cfg)
    row = sig[8]
    assert np.allclose(row[:14], 1.0) and np.allclose(row[18:], 1.0)
    excess = row - 1.0
    assert excess[15] > 0 and excess[16] > 0
    assert np.isclose(excess.sum(), cfg.edge_yield, rtol=1e-6)


def test_a_half_height_step_blooms_half_as_much():
    cfg = _plain()
    full, half = _buffers(), _buffers()
    full.depth[:, 16:] = 100.0
    half.depth[:, 16:] = 50.0
    e_full = (tilt_signal(full, cfg)[8] - 1.0).sum()
    e_half = (tilt_signal(half, cfg)[8] - 1.0).sum()
    assert np.isclose(e_half, 0.5 * e_full, rtol=1e-6)


def test_the_escape_length_spreads_the_bloom_and_the_beam_blurs_everything():
    b = _buffers(material=2)
    b.depth[:, 16:] = 100.0
    b.material[:, 16:] = 1                     # a resist edge over the substrate
    sharp = tilt_signal(b, _plain())
    spread = tilt_signal(b, SEMConfig(beam_fwhm=0.0, escape_length=3e-9, directionality=0.0,
                                      shadowing=0.0))
    blurred = tilt_signal(b, SEMConfig(beam_fwhm=6e-9, escape_length=0.0, directionality=0.0,
                                       shadowing=0.0))
    # The spread keeps the bloom's integral and widens it.
    assert np.isclose((spread[8] - sharp[8]).sum(), 0.0, atol=1e-6)
    assert (spread[8, 12:20] > 1.0 + 1e-3).sum() > (sharp[8, 12:20] > 1.0 + 1e-3).sum()
    # The beam softens the material step too, which the escape length must not.
    assert np.isclose(spread[8, 4], 1.0, atol=1e-2) and np.isclose(spread[8, 28], 0.6, atol=1e-2)
    assert 0.6 < blurred[8, 15] < 1.0 + 3.0 and blurred[8, 17] > 0.6 + 1e-3


def test_a_face_turned_towards_the_detector_is_brighter():
    """The detector sits up the image and towards the column: a face whose
    normal leans that way collects more than one leaning away."""
    cfg = SEMConfig(beam_fwhm=0.0, escape_length=0.0, directionality=1.0, shadowing=0.0)
    t = np.radians(30.0)
    towards = tilt_signal(_buffers(normal=(0.0, np.sin(t), np.cos(t))), cfg).mean()
    away = tilt_signal(_buffers(normal=(0.0, -np.sin(t), np.cos(t))), cfg).mean()
    alike = tilt_signal(_buffers(normal=(0.0, np.sin(t), np.cos(t))),
                        SEMConfig(beam_fwhm=0.0, escape_length=0.0, directionality=0.0,
                                  shadowing=0.0)).mean()
    # Collection is a fraction of the ideal, never more: the face towards
    # the detector keeps nearly all of its electrons, the one away loses.
    assert away < towards <= alike
    assert away < 0.7 * alike


def test_a_trench_floor_is_occluded_and_an_open_plane_is_not():
    depth = np.full((40, 60), 100.0)
    depth[:, 20:40] = 160.0                    # a 20-px-wide trench, 60 nm deep
    occ = screen_occlusion(depth, pixel_nm=1.0, coarsen=1)
    assert np.allclose(occ[:, 5], 0.0), "the open top sees the whole sky"
    assert np.allclose(occ[:, 19], 0.0), "even at the trench's lip: the walls are below it"
    assert occ[20, 30] > 0.4, "the floor is walled in"
    assert np.isclose(occ[20, 21], occ[20, 38], atol=0.02), "symmetric, as the trench is"
    shallow = depth.copy()
    shallow[:, 20:40] = 110.0
    shallow_occ = screen_occlusion(shallow, pixel_nm=1.0, coarsen=1)
    assert shallow_occ[20, 30] < occ[20, 30], "a shallower trench is less so"
    # The coarse evaluation the view uses agrees away from the lip, where
    # its interpolation smears the step over a pixel or two.
    coarse = screen_occlusion(depth, pixel_nm=1.0)
    assert coarse.shape == depth.shape
    assert np.isclose(coarse[20, 30], occ[20, 30], atol=0.03)
    assert np.allclose(coarse[:, 5], 0.0)


def test_shadowing_darkens_the_floor_by_its_knob():
    b = _buffers(rows=40, cols=60, material=1)
    b.depth[:, 20:40] = 60.0
    b.material[:, :20] = 2
    b.material[:, 40:] = 2
    lit = tilt_signal(b, _plain())
    shaded = tilt_signal(b, SEMConfig(beam_fwhm=0.0, escape_length=0.0, directionality=0.0,
                                      shadowing=1.0))
    assert shaded[20, 30] < lit[20, 30]
    assert np.isclose(shaded[20, 5], lit[20, 5]), "the open top is untouched"


def test_shot_noise_falls_with_electrons():
    b = _buffers()
    quiet = tilt_sem(b, SEMConfig(electrons_per_pixel=4000.0, frames=4, seed=1))
    loud = tilt_sem(b, SEMConfig(electrons_per_pixel=20.0, frames=1, seed=1))
    assert quiet.mode == "tilt"
    assert np.std(quiet.image - quiet.signal) < np.std(loud.image - loud.signal)
    assert np.isclose(quiet.pixel_size, 1e-9)


def test_the_new_instrument_knobs_are_validated():
    with pytest.raises(ValueError):
        SEMConfig(directionality=1.5)
    with pytest.raises(ValueError):
        SEMConfig(shadowing=-0.1)
    with pytest.raises(ValueError):
        SEMConfig(secant_power=-1.0)


def test_buffers_disagreeing_on_shape_are_refused():
    b = _buffers()
    b.material = b.material[:-1]
    with pytest.raises(ValueError):
        tilt_signal(b, _plain())


def test_the_projection_is_parallel_and_in_pixels():
    b = _buffers(rows=40, cols=60, pixel_nm=0.5)
    centre = b.project(np.zeros(3))[0]
    assert np.allclose(centre, (30.0, 20.0))
    along = b.project(np.array([[10.0, 0.0, 0.0]]))[0]
    assert np.allclose(along, (30.0 + 20.0, 20.0)), "10 nm along `right` is 20 px"
    up = b.project(np.array([[0.0, 10.0, 0.0]]))[0]
    assert np.allclose(up, (30.0, 20.0 - 20.0)), "up the sample is up the image"


# ---------------------------------------------------------------------------
# The geometry pass (VTK)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def render():
    pytest.importorskip("pyvista")
    from litho_sim.viz import render as mod

    if not mod.available():
        pytest.skip("VTK not importable")
    try:
        # A first render also proves an off-screen GL context exists here.
        vol = np.zeros((4, 8, 8), dtype=bool)
        vol[:, 2:6, 2:6] = True
        surf = mod.profile_surface(vol, spacing_nm=(2.0, 2.0, 2.0))
        mod.render_buffers(surf, (16.0, 16.0), 8.0, mod.Stage(pixel_nm=1.0), voxel_nm=(2.0, 2.0))
    except Exception as exc:                   # noqa: BLE001 - no GL here
        pytest.skip(f"off-screen rendering unavailable: {exc}")
    return mod


def _line_volume(nz=20, ny=32, nx=48, x0=16, x1=32):
    vol = np.zeros((nz, ny, nx), dtype=bool)
    vol[:, :, x0:x1] = True
    return vol


def test_the_surface_of_a_box_is_closed_and_where_the_box_is(render):
    vol = _line_volume()
    surf = render.profile_surface(vol, spacing_nm=(2.5, 3.0, 3.0))
    assert surf.n_open_edges == 0, "closed: the field boundary and the substrate are faces"
    b = surf.bounds
    assert abs(b[0] - 16 * 3.0) < 3.0 and abs(b[1] - 32 * 3.0) < 3.0
    assert b[4] > -2.5 and abs(b[5] - 20 * 2.5) < 2.5


def test_a_field_and_its_level_give_the_same_surface_as_the_boolean(render):
    vol = _line_volume()
    field = np.where(vol, 5.0, 0.2)
    from_field = render.profile_surface(field=field, level=1.0, spacing_nm=(2.5, 3.0, 3.0))
    from_bool = render.profile_surface(vol, spacing_nm=(2.5, 3.0, 3.0))
    assert np.allclose(from_field.bounds, from_bool.bounds, atol=2.0)


def test_stage_settings_are_validated(render):
    with pytest.raises(ValueError):
        render.Stage(tilt=95.0)
    with pytest.raises(ValueError):
        render.Stage(pixel_nm=0.0)
    with pytest.raises(ValueError):
        render.Stage(frame_px=10)


def test_an_unset_pitch_frames_the_field_at_the_asked_size(render):
    vol = _line_volume()
    surf = render.profile_surface(vol, spacing_nm=(2.5, 3.0, 3.0))
    buf = render.render_buffers(surf, (144.0, 96.0), 50.0,
                                render.Stage(frame_px=600, margin_px=10), voxel_nm=(3.0, 3.0))
    assert max(buf.shape) == 600
    assert 0.2 <= buf.pixel_nm < 1.0


def test_the_buffers_measure_the_geometry(render):
    vol = _line_volume()
    surf = render.profile_surface(vol, spacing_nm=(2.5, 3.0, 3.0))
    stage = render.Stage(tilt=40.0, azimuth=15.0, pixel_nm=1.0, substrate_nm=30.0)
    buf = render.render_buffers(surf, (48 * 3.0, 32 * 3.0), 50.0, stage, voxel_nm=(3.0, 3.0))
    seen = buf.seen
    assert set(np.unique(buf.material)) == {0, 1, 2}
    assert np.isfinite(buf.depth[seen]).all() and np.isnan(buf.depth[~seen]).all()
    assert np.allclose(np.linalg.norm(buf.normals[seen], axis=-1), 1.0, atol=0.02)
    # A point on the line's top faces +z and is resist; the floor beside it
    # is substrate; the cleaved face of the substrate faces the viewer (−y).
    c, r = buf.project(np.array([[24 * 3.0, 20 * 3.0, 50.0]]))[0].round().astype(int)
    assert buf.material[r, c] == 2 and buf.normals[r, c, 2] > 0.95
    c, r = buf.project(np.array([[6 * 3.0, 20 * 3.0, 0.0]]))[0].round().astype(int)
    assert buf.material[r, c] == 1 and buf.normals[r, c, 2] > 0.95
    c, r = buf.project(np.array([[24 * 3.0, -3.0, -15.0]]))[0].round().astype(int)
    assert buf.material[r, c] == 1 and buf.normals[r, c, 1] < -0.95
    # And the camera's own report of its axes is what the pixels obey: a
    # 100 nm bar along x is 100 × |x̂ projected into the image plane| px
    # at 1 nm/px — no perspective, wherever it lies.
    for y in (0.0, 90.0):
        ends = buf.project(np.array([[0.0, y, 0.0], [100.0, y, 0.0]]))
        assert np.isclose(np.linalg.norm(ends[1] - ends[0]),
                          100.0 * np.hypot(buf.right[0], buf.up[0]), atol=1e-6)


def test_the_silhouette_lands_where_the_projection_says(render):
    vol = _line_volume()
    surf = render.profile_surface(vol, spacing_nm=(2.5, 3.0, 3.0))
    stage = render.Stage(tilt=40.0, azimuth=15.0, pixel_nm=1.0, substrate_nm=30.0, margin_px=10)
    buf = render.render_buffers(surf, (144.0, 96.0), 50.0, stage, voxel_nm=(3.0, 3.0))
    corners = np.array([[x, y, z] for x in (-3.0, 147.0) for y in (-3.0, 99.0)
                        for z in (-30.0, 0.0)])
    pc = buf.project(corners)
    cols = np.nonzero(buf.seen.any(axis=0))[0]
    rows = np.nonzero(buf.seen.any(axis=1))[0]
    assert abs(cols.min() - pc[:, 0].min()) <= 2 and abs(cols.max() - pc[:, 0].max()) <= 2
    assert abs(rows.max() - pc[:, 1].max()) <= 2


def test_the_whole_path_makes_a_micrograph_and_a_figure(render):
    from litho_sim.viz.plots import plot_tilt_sem

    grid = GridConfig(n_pixels=48, pixel_size=3e-9, dz=2.5e-9, n_z_slices=5)
    vol = _line_volume()
    sem, buf = render.tilt_sem_of_profile(vol, grid, film_nm=50.0,
                                          stage=render.Stage(pixel_nm=1.0))
    assert sem.mode == "tilt" and sem.image.shape == buf.shape
    assert np.isfinite(sem.image).all()
    # The line's top is brighter than the floor beside it: material contrast
    # survives the whole chain.
    c, r = buf.project(np.array([[24 * 3.0, 20 * 3.0, 50.0]]))[0].round().astype(int)
    c2, r2 = buf.project(np.array([[6 * 3.0, 20 * 3.0, 0.0]]))[0].round().astype(int)
    assert sem.signal[r, c] > sem.signal[r2, c2]
    from matplotlib.figure import Figure

    # A bare Figure, not pyplot's: this module must not start an interactive
    # backend, or the Qt view below cannot start its own.
    fig = plot_tilt_sem(sem, buf, title="a line", caption="test", fig=Figure())
    texts = [t.get_text() for t in fig.texts]
    assert any("a line" in t for t in texts) and any("scale bar" in t for t in texts)
    assert fig.axes and fig.axes[0].images, "the micrograph is on the axes"


def test_the_boolean_path_and_the_field_path_agree_on_the_picture(render):
    grid = GridConfig(n_pixels=48, pixel_size=3e-9, dz=2.5e-9, n_z_slices=5)
    vol = _line_volume()
    field = np.where(vol, 5.0, 0.2)
    cfg = SEMConfig(electrons_per_pixel=4000.0, frames=8)
    a, _ = render.tilt_sem_of_profile(vol, grid, film_nm=50.0, cfg=cfg,
                                      stage=render.Stage(pixel_nm=1.0))
    b, _ = render.tilt_sem_of_profile(None, grid, field=field, level=1.0, film_nm=50.0,
                                      cfg=cfg, stage=render.Stage(pixel_nm=1.0))
    assert a.signal.shape == b.signal.shape
    assert np.corrcoef(a.signal.ravel(), b.signal.ravel())[0, 1] > 0.97


# ---------------------------------------------------------------------------
# The app's view
# ---------------------------------------------------------------------------


def test_the_profile_view_images_the_solid_and_caches_the_geometry(render, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from litho_sim.app.compute import Profile3DResult
    from litho_sim.app.qt import QtWidgets
    from litho_sim.app.views import Profile3DView

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    grid = GridConfig(n_pixels=48, pixel_size=3e-9, dz=2.5e-9, n_z_slices=5)
    vol = _line_volume()
    r = Profile3DResult(
        remaining=vol, cut_remaining=vol[:, 16, :], cut_latent=np.zeros((20, 48)),
        x_nm=np.arange(48) * 3.0, height_nm=50.0, row=16, sidewall_deg=88.0,
        film_remaining_pct=33.0, top_loss_nm=0.0, sealed=False, cleared=False,
        label="test", elapsed_ms=1.0, signature=("t",),
    )
    view = Profile3DView()
    try:
        view.sem._stage = render.Stage(pixel_nm=1.0)
        view.show_profile(r, grid, SEMConfig(), tilt=40.0, azimuth=15.0)
        assert view.sem.last is not None and view.sem.last.mode == "tilt"
        geometry = view.sem._buffers
        view.set_instrument(SEMConfig(beam_fwhm=5e-9))
        assert view.sem._buffers is geometry, "a knob re-forms the image, not the geometry"
        view.set_stage(50.0, 15.0)
        assert view.sem._buffers is not geometry, "the stage moved: re-rendered"
        view.show_placeholder("gone")
        assert view.sem.last is None
    finally:
        view.close()
