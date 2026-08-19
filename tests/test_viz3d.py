"""
Tests for 3-D visualisation.

Plotly is an optional dependency, so the Plotly-specific tests skip when it
is absent. The geometry extraction underneath is pure numpy and is always
tested — that is where the real logic lives; the Plotly calls on top are a
thin wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

from litho_sim.core.config import GridConfig
from litho_sim.viz.viz3d import (
    _NICE_NM,
    HAS_PLOTLY,
    PROCESS_FAMILIES,
    classify_step,
    crop,
    cross_section_figure,
    device_figure_mpl,
    payload_size,
    process_flow_panel,
    profile_figure,
    stack_figure,
    stack_figure_mpl,
    surface_payload,
    voxel_mesh,
)
from litho_sim.wafer import Stack

needs_plotly = pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")


@pytest.fixture
def stack() -> Stack:
    grid = GridConfig(n_pixels=32, pixel_size=4e-9, dz=4e-9)
    st = Stack.blank(grid, dz=4e-9, substrate_thickness=20e-9, headroom=120e-9)
    st.deposit_blanket("poly-Si", 20e-9)
    st.deposit_blanket("photoresist", 40e-9)
    return st


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------


def test_surface_payload_shapes(stack):
    p = surface_payload(stack, "photoresist")
    assert p is not None
    ny, nx = stack.shape_xy
    assert p["x"].shape == (nx,)
    assert p["y"].shape == (ny,)
    assert p["z_top"].shape == (ny, nx)
    assert p["z_bottom"].shape == (ny, nx)


def test_surface_payload_is_in_nanometres(stack):
    p = surface_payload(stack, "photoresist")
    # Resist sits on 20 nm substrate + 20 nm poly, and is 40 nm thick.
    assert np.nanmax(p["z_top"]) == pytest.approx(80.0, abs=4.0)
    assert np.nanmin(p["z_bottom"]) == pytest.approx(40.0, abs=4.0)


def test_surface_payload_returns_none_when_absent(stack):
    assert surface_payload(stack, "spacer-oxide") is None


def test_surface_payload_marks_missing_columns(stack):
    """Where a material is absent the height must be NaN, not zero.

    Zero would render as a sheet lying on the substrate.
    """
    ny, nx = stack.shape_xy
    stack.clear((stack.mat == 8) & (np.arange(nx)[None, None, :] < nx // 2))
    p = surface_payload(stack, "photoresist")
    assert np.isnan(p["z_top"][:, : nx // 2]).all()
    assert np.isfinite(p["z_top"][:, nx // 2:]).all()


def test_payload_size_shows_the_surface_saving(stack):
    """Height fields must be dramatically lighter than the raw volume.

    This is the reason go.Volume / go.Isosurface are not used: a 128x128x40
    volume is 655k points, which is 8-12 MB of JSON per render.
    """
    sizes = payload_size(stack)
    assert sizes["surface_points"] < sizes["volume_points"]
    assert sizes["ratio"] > 3


# ---------------------------------------------------------------------------
# Matplotlib fallbacks (always available)
# ---------------------------------------------------------------------------


def test_cross_section_figure_renders(stack):
    ax = cross_section_figure(stack)
    assert ax.images, "no image drawn"
    assert ax.get_ylabel() == "z [nm]"


def test_cross_section_axes_are_physical(stack):
    ax = cross_section_figure(stack)
    x0, x1, y0, y1 = ax.images[0].get_extent()
    assert x1 == pytest.approx(stack.shape_xy[1] * stack.pixel_size * 1e9)
    assert y1 == pytest.approx(stack.nz * stack.dz * 1e9)


def test_cross_section_rejects_bad_axis(stack):
    with pytest.raises(ValueError):
        cross_section_figure(stack, axis="z")


def test_stack_figure_mpl_renders(stack):
    # An exaggerated z-axis must say so on the label, or the plot lies about
    # aspect ratio; unexaggerated axes keep the plain label.
    ax = stack_figure_mpl(stack, z_exaggeration=2.0)
    assert ax.get_zlabel() == "z [nm] ×2"
    ax = stack_figure_mpl(stack, z_exaggeration=1.0)
    assert ax.get_zlabel() == "z [nm]"


def test_crop_keeps_the_requested_fraction(stack):
    ny, nx = stack.shape_xy
    c = crop(stack, y=(0.5, 1.0))
    assert c.shape_xy == (ny - ny // 2, nx)
    assert stack.shape_xy == (ny, nx), "crop must not mutate its input"


def test_crop_trims_substrate_and_headroom(stack):
    c = crop(stack, z_substrate=8e-9)
    assert c.nz < stack.nz
    # Everything that was built is still there; only bulk and vacuum went.
    for m in stack.present_materials():
        assert (c.mat == m.id).any(), f"crop lost {m.name}"


def test_crop_caps_the_cut_face(stack):
    """The reason cropping beats discarding triangles behind a plane.

    `voxel_mesh` emits a face only where a solid voxel borders something that
    is not itself, so a plane drawn through the middle of a film is not a face
    and filtering triangles leaves the solid open. Cropping makes the cut the
    edge of the array, which `voxel_mesh` pads as exposed — so the section
    comes out closed and you can see the layer structure on it.
    """
    c = crop(stack, y=(0.5, 1.0))
    V, F = voxel_mesh(c, "poly-Si")
    tris = V[F]
    on_cut = (tris[:, :, 1] == 0.0).all(axis=1)
    assert on_cut.any(), "no capping faces were emitted on the cut plane"


def test_device_figure_uses_one_collection_per_figure(stack):
    """The whole point of it, versus `stack_figure_mpl`.

    Matplotlib depth-sorts each Poly3DCollection independently, so one
    collection per material lets a buried film paint over the film burying it.
    Merging every material into a single collection makes all faces sort
    against each other, which is what a device with five or six films needs.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def n_poly(ax):
        return sum(isinstance(c, Poly3DCollection) for c in ax.collections)

    assert n_poly(device_figure_mpl(stack)) == 1
    assert n_poly(stack_figure_mpl(stack)) > 1


def test_device_figure_labels_z_exaggeration(stack):
    assert device_figure_mpl(stack, z_exaggeration=2.0).get_zlabel() == "z [nm] ×2"
    assert device_figure_mpl(stack, z_exaggeration=1.0).get_zlabel() == "z [nm]"


def test_device_figure_can_restrict_materials(stack):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def faces(**kw):
        ax = device_figure_mpl(stack, shade=False, **kw)
        coll = [c for c in ax.collections if isinstance(c, Poly3DCollection)][0]
        return coll.get_facecolor()

    one = faces(materials=["photoresist"])
    # Shading off, so one material is exactly one colour across every face.
    assert len(np.unique(one, axis=0)) == 1
    # And restricting really dropped geometry rather than just recolouring it.
    assert len(one) < len(faces())


def test_chrome_axes_keeps_matplotlib_decoration(stack):
    # Axes3D tracks its own decoration in `_axis3don`; the inherited 2-D
    # `axison` flag reads False even on a fresh 3-D axes, so asserting on that
    # one passes and fails for the wrong reasons.
    assert device_figure_mpl(stack, chrome="axes")._axis3don
    assert not device_figure_mpl(stack, chrome="none")._axis3don


@pytest.mark.parametrize("style, n_edges", [("none", 0), ("corner", 6), ("box", 12)])
def test_chrome_draws_the_frame_it_promises(stack, style, n_edges):
    ax = device_figure_mpl(stack, chrome=style)
    assert len(ax.lines) == n_edges


def test_scale_bar_is_drawn_and_labelled(stack):
    ax = device_figure_mpl(stack, chrome="none", scale_bar=True, scale_bar_nm=50)
    assert len(ax.lines) == 1
    labels = [t.get_text() for t in ax.texts]
    assert "50 nm" in labels, labels


def test_scale_bar_length_is_a_round_number(stack):
    # Auto length must land on the printable list, not on 38.4 % of whatever
    # the extent happens to be — a bar reads as a measurement.
    ax = device_figure_mpl(stack, chrome="none", scale_bar=True)
    value = float(ax.texts[0].get_text().split()[0])
    assert value in _NICE_NM
    extent_nm = stack.shape_xy[1] * stack.pixel_size * 1e9
    assert value <= extent_nm


@pytest.mark.parametrize("entry, family, verb", [
    ("blank: Si 40 nm", "substrate", "Substrate"),
    ("deposit SiGe 10 nm", "deposit", "Deposition"),
    ("deposit SiO2 174 nm (planarized)", "deposit", "Deposition"),
    ("deposit conformal SiN 5 nm", "ald", "ALD / conformal"),
    ("deposit conformal TiN 4 nm on Si, poly-Si", "ald", "ALD / conformal"),
    ("deposit photoresist 45 nm (planarized)", "litho", "Resist coat"),
    ("expose gate dose=1.80 overlay=(0.0, 0.0) nm", "litho", "Expose"),
    ("develop photoresist: 29% retained", "litho", "Develop"),
    ("etch poly-Si 194 nm (aniso=1.00)", "etch", "Etch"),
    ("etch SiGe 8 nm (lateral front)", "etch", "Lateral etch"),
    ("etch_back SiN 5 nm (overetch=20%)", "etch", "Spacer etch-back"),
    ("CMP stopping on Si at 94 nm", "cmp", "CMP"),
    ("strip photoresist", "strip", "Strip"),
])
def test_every_process_verb_is_recognised(entry, family, verb):
    """A conformal deposit is ALD, not "deposition" — the panel is read by
    someone who cares about the difference, and the two are one word apart in
    the engine's own history line."""
    got_family, got_verb, _ = classify_step(entry)
    assert (got_family, got_verb) == (family, verb)
    assert got_family in PROCESS_FAMILIES


def test_conformal_is_matched_before_plain_deposit():
    """Rule order, not rule content, is what makes this one work.

    "deposit conformal SiN" contains "deposit". With the general rule first
    every ALD step in every flow would silently read as a blanket deposition —
    correct-looking, and wrong about the process.
    """
    assert classify_step("deposit conformal SiN 5 nm")[1] == "ALD / conformal"
    assert classify_step("deposit SiN 5 nm")[1] == "Deposition"


def test_display_only_history_is_not_process(stack):
    """`crop` records a viewer framing a picture. It is not a process step."""
    from litho_sim.viz.viz3d import crop as crop_stack

    _, before = process_flow_panel(stack)
    _, after = process_flow_panel(crop_stack(stack, y=(0.5, 1.0)))
    assert "(display)" in crop_stack(stack, y=(0.5, 1.0)).history[-1]
    assert after == before, "a display crop was counted as a process step"


def test_repeated_steps_collapse_with_a_multiplier(stack):
    """A superlattice is the same two deposits three times over."""
    s = stack.copy()
    s.history.clear()
    for _ in range(3):
        s.history.append("deposit SiGe 10 nm")
    _, rows = process_flow_panel(s)
    assert rows == 1, f"three identical deposits drew {rows} rows"


def test_profile_figure_renders():
    grid = GridConfig(n_pixels=16, pixel_size=4e-9, dz=2e-9)
    remaining = np.zeros((20, 16, 16), dtype=bool)
    remaining[:, :, 4:8] = True
    ax = profile_figure(remaining, grid)
    assert ax.images


def test_resist_profile_3d_figure_renders():
    """The composite figure draws the solid, the profile, and the latent."""
    import matplotlib.pyplot as plt

    from litho_sim.viz.viz3d import resist_profile_3d_figure

    grid = GridConfig(n_pixels=16, pixel_size=4e-9, dz=2e-9)
    remaining = np.zeros((20, 16, 16), dtype=bool)
    remaining[:, :, 4:8] = True
    remaining[:16, :, 10:14] = True          # a second line with top loss
    latent = np.linspace(0.2, 0.9, 20)[:, None, None] * np.ones((20, 16, 16))

    fig = resist_profile_3d_figure({"remaining": remaining, "latent": latent}, grid)

    assert len(fig.axes) >= 3                # 3-D + profile + latent (+ colorbar)
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "remaining" in texts and "sidewall" in texts
    plt.close(fig)


def test_resist_profile_3d_figure_flags_a_sealed_film():
    """An undeveloped film gets a diagnosis on the figure, not a blank axis."""
    import matplotlib.pyplot as plt

    from litho_sim.viz.viz3d import resist_profile_3d_figure

    grid = GridConfig(n_pixels=12, pixel_size=4e-9, dz=2e-9)
    sealed = np.ones((10, 12, 12), dtype=bool)
    latent = np.full((10, 12, 12), 0.8)

    fig = resist_profile_3d_figure({"remaining": sealed, "latent": latent}, grid)

    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "did not develop" in texts
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------


@needs_plotly
def test_stack_figure_has_a_trace_per_material(stack):
    fig = stack_figure(stack)
    # Two surfaces (top + base) per present material.
    assert len(fig.data) == 2 * len(stack.present_materials())


@needs_plotly
def test_stack_figure_cutaway_reduces_extent(stack):
    full = stack_figure(stack)
    half = stack_figure(stack, cutaway=0.5)
    assert half.data[0].y.shape[0] < full.data[0].y.shape[0]


@needs_plotly
def test_stack_figure_can_hide_the_substrate(stack):
    fig = stack_figure(stack, show_substrate=False)
    names = {t.name for t in fig.data}
    assert "Si" not in names


def test_stack_figure_without_plotly_explains_itself(stack, monkeypatch):
    """The error must name the fix, not just fail on an import."""
    import litho_sim.viz.viz3d as v

    monkeypatch.setattr(v, "HAS_PLOTLY", False)
    with pytest.raises(ImportError, match="pip install plotly"):
        v.stack_figure(stack)


class _FakeFigure:
    def __init__(self):
        self.data = []
        self.layout = {}

    def add_trace(self, trace):
        self.data.append(trace)

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)


class _FakePlotly:
    """Just enough of plotly.graph_objects to let stack_figure run."""

    Figure = _FakeFigure

    @staticmethod
    def Mesh3d(**kwargs):
        return SimpleNamespace(**kwargs)


@pytest.fixture
def stubbed_plotly(monkeypatch):
    import litho_sim.viz.viz3d as v

    monkeypatch.setattr(v, "HAS_PLOTLY", True)
    monkeypatch.setattr(v, "go", _FakePlotly)
    return v


def test_stack_figure_aspect_ratio_follows_the_cutaway(stack, stubbed_plotly):
    """Guards the whole Plotly layout path against name and unit errors.

    The @needs_plotly tests above skip whenever Plotly is absent, which is the
    normal state of this venv — so `stack_figure` could and did ship with an
    undefined name in its aspectratio for as long as nobody installed Plotly.
    Stubbing the module runs the body regardless.
    """
    ny, nx = stack.shape_xy

    full = stubbed_plotly.stack_figure(stack)
    assert full.layout["scene"]["aspectratio"]["y"] == pytest.approx(ny / nx)

    half = stubbed_plotly.stack_figure(stack, cutaway=0.5)
    assert half.layout["scene"]["aspectratio"]["y"] == pytest.approx(ny / nx / 2)


def test_resist_surface_draws_a_closed_solid_on_a_substrate():
    """The height field must read as a block, not a sheet hanging in space.

    Regression guard: the first version drew a bare `plot_surface` coloured by
    height, which lost the material identity, the sides and the ground plane
    all at once.
    """
    from matplotlib.figure import Figure

    from litho_sim.viz.viz3d import resist_surface_mpl

    remaining = np.zeros((10, 8, 8), dtype=bool)
    remaining[:6, :, :4] = True          # a step: 6 voxels tall on one half
    grid = GridConfig(n_pixels=8, pixel_size=10e-9, dz=5e-9)

    ax = Figure().add_subplot(projection="3d")
    resist_surface_mpl(remaining, grid, ax=ax, stride=1)

    lo, hi = ax.get_zlim()
    assert lo < 0.0, "no substrate slab beneath the film"
    assert hi == pytest.approx(30.0), "film top should be 6 x 5 nm"
    # top surface + 2 slab faces + 4 slab walls
    assert len(ax.collections) == 7

    # The mesh carries the resist colour, and the cleared half the substrate's
    surface = ax.collections[-1]
    colours = {tuple(np.round(c[:3], 3)) for c in surface._facecolor3d}
    assert len(colours) > 1, "cleared trenches should not be resist-coloured"


def test_resist_surface_still_accepts_a_colormap():
    """The height-mapped look stays available, just not as the default."""
    from matplotlib.figure import Figure

    from litho_sim.viz.viz3d import resist_surface_mpl

    remaining = np.zeros((6, 8, 8), dtype=bool)
    remaining[:3] = True
    grid = GridConfig(n_pixels=8, pixel_size=10e-9, dz=5e-9)
    ax = Figure().add_subplot(projection="3d")
    resist_surface_mpl(remaining, grid, ax=ax, cmap="viridis", substrate=False)
    assert ax.get_zlim()[0] == 0.0
