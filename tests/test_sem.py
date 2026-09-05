"""
The SEM image model: what an inspection tool sees of a printed profile.

Headless, numpy only. The physics claims each test pins are the ones the
Metrology note makes: material contrast, edge bloom that scales with wall
height and widens with the beam and the escape length, Poisson noise that
falls with electrons and frames, and a CD read off the image by its edge
peaks that agrees with the profile it was formed from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.metrology import (
    SEMConfig,
    material_topdown_signal,
    material_xsection_sem,
    material_xsection_signal,
    material_yields,
    measure_cd_sem,
    stack_topdown_sem,
    stack_xsection_sem,
    top_surface,
    topdown_sem,
    topdown_signal,
    xsection_sem,
    xsection_signal,
)
from litho_sim.metrology.sem import VACUUM_YIELD
from litho_sim.wafer.materials import MATERIAL_LIBRARY, VACUUM, get_material

PX = 4e-9
FILM = 100e-9


def _line(width_px: int, n: int = 64, height: float = FILM, start: int = 22):
    """One resist line of *width_px* columns on a cleared substrate."""
    h = np.zeros((n, n))
    h[:, start:start + width_px] = height
    return h


def _quiet() -> SEMConfig:
    """Noiseless enough to read the signal off the image."""
    return SEMConfig(electrons_per_pixel=2000.0, frames=8)


# ---------------------------------------------------------------------------
# The three contrast mechanisms
# ---------------------------------------------------------------------------


def test_a_flat_film_has_no_edges_and_no_cd():
    cfg = SEMConfig()
    signal = topdown_signal(np.full((32, 32), FILM), PX, cfg, film_thickness=FILM)
    assert np.allclose(signal, 1.0, atol=1e-9), "the flat resist top emits 1"
    m = measure_cd_sem(signal, PX)
    assert not m.found


def test_material_contrast_reads_the_substrate_darker():
    cfg = _quiet()
    h = _line(20)
    sem = topdown_sem(h, PX, cfg, film_thickness=FILM)
    inside = sem.signal[:, 30]        # well inside the line
    outside = sem.signal[:, 5]        # well clear of it
    assert inside.mean() == pytest.approx(1.0, abs=0.02)
    assert outside.mean() == pytest.approx(cfg.substrate_yield, abs=0.02)


def test_edges_bloom_and_the_bloom_scales_with_wall_height():
    cfg = _quiet()
    full = topdown_signal(_line(20, height=FILM), PX, cfg, film_thickness=FILM)
    half = topdown_signal(_line(20, height=0.5 * FILM), PX, cfg, film_thickness=FILM)
    row_full, row_half = full[16], half[16]
    # A bright line at each wall, above both the resist top and the floor.
    assert row_full.max() > 1.5, "no edge bloom"
    peaks = np.argsort(row_full)[-2:]
    assert sorted(peaks) == pytest.approx([21.5, 41.5], abs=1.0)
    # Half the wall, roughly half the bloom above the material level.
    assert (row_half.max() - 1.0) == pytest.approx(0.5 * (row_full.max() - 1.0), rel=0.25)


def _edge_width(row: np.ndarray, base: float, half_window: int = 8) -> float:
    """Equivalent width of the brightest peak above *base*, in pixels.

    Area over height — continuous, unlike a half-maximum crossing counted
    in whole pixels, which reads 3 px for a 2 nm beam and 3 px for an 8 nm
    one at 4 nm pixels and cannot order them.
    """
    i = int(np.argmax(row))
    lo, hi = max(i - half_window, 0), min(i + half_window + 1, row.size)
    excess = np.clip(row[lo:hi] - base, 0.0, None)
    return float(excess.sum() / max(row[i] - base, 1e-12))


def test_the_beam_widens_everything_and_the_escape_length_only_the_edges():
    h = _line(20)
    narrow = topdown_signal(h, PX, SEMConfig(beam_fwhm=2e-9, escape_length=0.0), FILM)
    wide = topdown_signal(h, PX, SEMConfig(beam_fwhm=8e-9, escape_length=0.0), FILM)
    assert _edge_width(wide[16], 1.0) > _edge_width(narrow[16], 1.0)
    # The beam reaches further from the wall: 6 nm out on the floor the
    # wide beam has lifted the signal, the narrow one has not — and far
    # away neither has.
    assert wide[16, 20] > narrow[16, 20] + 0.05
    assert wide[16, 5] == pytest.approx(narrow[16, 5], abs=1e-6)

    tight = topdown_signal(h, PX, SEMConfig(beam_fwhm=2e-9, escape_length=0.0), FILM)
    loose = topdown_signal(h, PX, SEMConfig(beam_fwhm=2e-9, escape_length=6e-9), FILM)
    assert _edge_width(loose[16], 1.0) > _edge_width(tight[16], 1.0)
    # Far from any edge the escape length changes nothing.
    assert loose[16, 5] == pytest.approx(tight[16, 5], abs=1e-6)
    assert loose[16, 31] == pytest.approx(tight[16, 31], abs=1e-6)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def test_shot_noise_falls_with_electrons_and_frames():
    flat = np.full((64, 64), FILM)
    for n_e, frames in ((100.0, 1), (100.0, 4), (400.0, 4)):
        cfg = SEMConfig(electrons_per_pixel=n_e, frames=frames, seed=3)
        sem = topdown_sem(flat, PX, cfg, film_thickness=FILM)
        # Signal is exactly 1 everywhere, so the count is Poisson(N) / N and
        # its relative spread is 1/sqrt(N).
        assert sem.image.std() == pytest.approx(1.0 / np.sqrt(n_e * frames), rel=0.12)
        assert sem.snr == pytest.approx(np.sqrt(n_e * frames), rel=0.12)


def test_the_seed_reproduces_the_grain():
    h = _line(20)
    a = topdown_sem(h, PX, SEMConfig(seed=11), FILM)
    b = topdown_sem(h, PX, SEMConfig(seed=11), FILM)
    c = topdown_sem(h, PX, SEMConfig(seed=12), FILM)
    assert np.array_equal(a.image, b.image)
    assert not np.array_equal(a.image, c.image)
    assert np.array_equal(a.signal, c.signal), "noise must not touch the signal"


def test_bad_instrument_settings_are_rejected():
    with pytest.raises(ValueError):
        SEMConfig(electrons_per_pixel=0.0)
    with pytest.raises(ValueError):
        SEMConfig(frames=0)
    with pytest.raises(ValueError):
        SEMConfig(beam_fwhm=-1e-9)


# ---------------------------------------------------------------------------
# Measuring the image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width_px", [8, 16, 25])
def test_peak_to_peak_cd_recovers_the_line_width(width_px):
    sem = topdown_sem(_line(width_px), PX, _quiet(), FILM)
    m = measure_cd_sem(sem.image, PX, feature="line")
    assert m.found
    assert m.cd == pytest.approx(width_px * PX, abs=PX)
    assert m.feature == "line"
    assert m.edges.size == 2


def test_the_space_is_the_complement_of_the_line():
    n = 64
    sem = topdown_sem(_line(20, n=n), PX, _quiet(), FILM)
    line = measure_cd_sem(sem.image, PX, feature="line")
    space = measure_cd_sem(sem.image, PX, feature="space")
    assert line.found and space.found
    # One line in a periodic field: the space wraps round to fill the rest.
    assert line.cd + space.cd == pytest.approx(n * PX, abs=2 * PX)


def test_a_line_on_the_wrap_seam_is_still_found():
    h = np.zeros((48, 48))
    h[:, :6] = FILM
    h[:, -6:] = FILM                  # 12 px wide, straddling the seam
    sem = topdown_sem(h, PX, _quiet(), FILM)
    m = measure_cd_sem(sem.image, PX, feature="line")
    assert m.found
    assert m.cd == pytest.approx(12 * PX, abs=PX)


def test_a_noisy_frame_measures_within_a_pixel():
    sem = topdown_sem(_line(20), PX, SEMConfig(electrons_per_pixel=20.0, frames=1), FILM)
    m = measure_cd_sem(sem.image, PX)
    assert m.found
    assert m.cd == pytest.approx(20 * PX, abs=PX)


def test_measurement_refuses_what_it_cannot_tell_apart():
    with pytest.raises(ValueError, match="line"):
        measure_cd_sem(np.zeros((8, 8)), PX, feature="edge")
    with pytest.raises(ValueError):
        measure_cd_sem(np.zeros(8), PX)


# ---------------------------------------------------------------------------
# Volumes and cross-sections
# ---------------------------------------------------------------------------


def test_top_surface_sees_the_cap_over_an_undercut():
    nz, ny, nx = 10, 4, 6
    vol = np.zeros((nz, ny, nx), dtype=bool)
    vol[:, :, 1] = True                   # a full column: 10 voxels
    vol[6:, :, 2] = True                  # a cap with nothing under it
    vol[:3, :, 3] = True                  # a stump
    h = top_surface(vol, dz=2e-9)
    assert h[0, 0] == 0.0
    assert h[0, 1] == pytest.approx(20e-9)
    assert h[0, 2] == pytest.approx(20e-9), "an undercut column presents its top"
    assert h[0, 3] == pytest.approx(6e-9)


def test_a_cross_section_outlines_every_boundary():
    nz, nx = 30, 40
    slab = np.zeros((nz, nx), dtype=bool)
    slab[:, 10:22] = True                 # one resist line, full height
    cfg = _quiet()
    sem = xsection_sem(slab, PX, 2e-9, cfg)
    sig = sem.signal
    sub = int(round(-sem.row_origin / 2e-9))
    assert sub >= 2, "substrate rows drawn under the film"
    assert sig.shape[0] > sub + nz, "vacuum drawn above the film"
    x0, x1, z0, z1 = sem.extent_nm
    assert z0 < 0.0 < z1
    assert z1 > nz * 2e-9 * 1e9, "the extent reaches above the film top"

    film = sig[sub:sub + nz]
    resist_bulk = film[nz // 2, 16]
    vacuum = film[nz // 2, 32]
    wall = film[nz // 2, 9:12].max()
    top_edge = film[nz - 1, 16]
    assert vacuum < resist_bulk, "vacuum reads dark"
    assert wall > resist_bulk, "the resist/vacuum wall blooms"
    assert top_edge > resist_bulk, "the resist top edge blooms"
    assert sem.mode == "xsection"


def test_xsection_signal_validates_its_input():
    with pytest.raises(ValueError):
        xsection_signal(np.zeros(5, dtype=bool), PX, 2e-9, SEMConfig())


# ---------------------------------------------------------------------------
# Wafer stacks: material contrast
# ---------------------------------------------------------------------------


def _sharp() -> SEMConfig:
    """No probe and no escape length, so a signal is exactly its terms."""
    return SEMConfig(beam_fwhm=0.0, escape_length=0.0)


SI, SIO2, TIN = (get_material(n).id for n in ("Si", "SiO2", "TiN"))


def test_material_yields_follow_the_library():
    lut = material_yields()
    assert lut.shape == (256,)
    assert lut[VACUUM] == VACUUM_YIELD
    for m in MATERIAL_LIBRARY.values():
        if m.id != VACUUM:
            assert lut[m.id] == m.se_yield
    assert lut[200] == 1.0, "an ID nothing registers reads as resist, not as a hole"
    # The one number the resist path and the stack path share: silicon's
    # yield is the substrate-yield default, so a floor images the same grey
    # whichever way it arrived.
    assert get_material("Si").se_yield == SEMConfig().substrate_yield
    over = material_yields(overrides={"Si": 0.3})
    assert over[SI] == 0.3 and over[SIO2] == lut[SIO2]


def _layered_section(nz: int = 30, nx: int = 40) -> np.ndarray:
    """Si under SiO2, a TiN block on top, vacuum above — row 0 at the bottom."""
    ids = np.full((nz, nx), VACUUM, dtype=np.uint8)
    ids[:10] = SI
    ids[10:20] = SIO2
    ids[20:26, 10:20] = TIN
    return ids


def test_a_cleaved_stack_shows_material_contrast_and_blooms_only_at_the_outline():
    ids = _layered_section()
    sig = material_xsection_signal(ids, PX, 2e-9, _sharp())
    # Trimmed to the highest solid row plus a little vacuum: 26 rows of
    # solid, and a tenth of that above.
    assert sig.shape == (29, 40)
    lut = material_yields()
    # The bulk of each material is exactly its yield.
    assert sig[5, 30] == pytest.approx(lut[SI])
    assert sig[15, 30] == pytest.approx(lut[SIO2])
    assert sig[22, 15] == pytest.approx(lut[TIN])
    assert sig[28, 30] == pytest.approx(VACUUM_YIELD)
    # A buried interface is a step in grey and nothing more.
    assert sig[9, 30] == pytest.approx(lut[SI])
    assert sig[10, 30] == pytest.approx(lut[SIO2])
    # The outline of the solid blooms: the oxide's top under vacuum, the
    # metal block's wall and top.
    assert sig[19, 30] > lut[SIO2]
    assert sig[22, 10] > lut[TIN] and sig[22, 9] > VACUUM_YIELD
    assert sig[25, 15] > lut[TIN]


def test_a_section_with_no_headroom_is_given_some():
    ids = _layered_section(nz=26)          # solid to the very top row
    sig = material_xsection_signal(ids, PX, 2e-9, _sharp())
    assert sig.shape[0] > 26, "vacuum padded above so the top surface is an edge"
    assert sig[25, 15] > material_yields()[TIN], "the top of the block blooms"
    sem = material_xsection_sem(ids, PX, 2e-9, _sharp(), headroom=5)
    assert sem.signal.shape == (31, 40)
    assert sem.row_origin == 0.0 and sem.mode == "xsection"
    with pytest.raises(ValueError):
        material_xsection_signal(ids[0], PX, 2e-9, _sharp())


def test_the_top_down_of_a_wafer_reads_the_top_material_and_its_steps():
    ny, nx = 32, 48
    ids = np.full((ny, nx), SIO2, dtype=np.uint8)
    ids[:, 10:20] = TIN
    flat = np.full((ny, nx), 100e-9)
    lut = material_yields()
    # Planarised: material contrast alone, no walls anywhere.
    sig = material_topdown_signal(flat, ids, PX, _sharp())
    assert np.allclose(sig[:, 5], lut[SIO2]) and np.allclose(sig[:, 15], lut[TIN])
    # Raise the metal 20 nm and its walls bloom; the reference height is the
    # tallest step present, so a full wall scores edge_yield split over the
    # two pixels either side of it.
    stepped = flat.copy()
    stepped[:, 10:20] = 120e-9
    cfg = _sharp()
    sig = material_topdown_signal(stepped, ids, PX, cfg)
    assert sig[16, 5] == pytest.approx(lut[SIO2])
    assert sig[16, 15] == pytest.approx(lut[TIN])
    assert sig[16, 10] == pytest.approx(lut[TIN] + 0.5 * cfg.edge_yield)
    assert sig[16, 9] == pytest.approx(lut[SIO2] + 0.5 * cfg.edge_yield)
    with pytest.raises(ValueError):
        material_topdown_signal(flat, ids[:, :10], PX, cfg)


def test_the_gaa_nanosheet_images_in_both_planes():
    """The device the tab was built to look at.

    Across the fin through the gate (a y–z cut) the sheets are silicon
    wrapped in metal; along the channel (x–z) the interlayer oxide stands
    either side of the gate trench. Every voxel images at its own material's
    yield, whichever way the preset arrived — built, or loaded from a cache
    written before materials carried a yield.
    """
    from litho_sim.tech.devices import build

    stack, _label = build("gaa")
    for m in stack.materials.values():
        assert m.se_yield == get_material(m.id).se_yield
    lut = material_yields(stack.materials.values())
    cfg = _sharp()

    across = stack.cross_section("x")
    sem = stack_xsection_sem(stack, "x", None, cfg)
    assert sem.mode == "xsection" and sem.row_origin == 0.0
    nz = sem.signal.shape[0]
    assert sem.signal.shape[1] == across.shape[1]
    assert nz <= across.shape[0], "the stack's spare headroom is not imaged"
    for mid in (SI, TIN, SIO2):
        rows, cols = np.nonzero(across[:nz] == mid)
        assert rows.size, f"the cut through the gate holds material {mid}"
        # Deep inside the material, not on its outline, the signal is the
        # yield itself. Take the interior voxels: every neighbour the same.
        interior = [
            (r, c) for r, c in zip(rows, cols)
            if 0 < r < nz - 1 and 0 < c < across.shape[1] - 1
            and (across[r - 1:r + 2, c - 1:c + 2] == mid).all()
        ]
        assert interior, f"material {mid} has an interior on the cut"
        r, c = interior[len(interior) // 2]
        assert sem.signal[r, c] == pytest.approx(lut[mid])

    along = stack_xsection_sem(stack, "y", None, cfg)
    assert along.signal.shape[1] == stack.cross_section("y").shape[1]
    x0, _x1, z0, _z1 = along.extent_nm
    assert x0 == 0.0 and z0 == 0.0

    top = stack_topdown_sem(stack, cfg)
    assert top.mode == "topdown" and top.signal.shape == stack.shape_xy
    assert top.signal.min() >= min(lut[stack.top_material()].min(), 1.0) - 1e-9


# ---------------------------------------------------------------------------
# Against the engine's own print
# ---------------------------------------------------------------------------


def test_the_sem_cd_agrees_with_the_printed_profile_at_half_height():
    """Form an SEM of a real 2-D print and measure it back.

    The number to agree with is the height map's own width at half height,
    measured by the engine's sub-pixel kernel — the same surface the beam
    scans. The threshold-model CD the print reports is a different width on
    a sloped profile and is not the comparison.
    """
    from litho_sim.app.compute import compute_imaging
    from litho_sim.app.params import ParameterModel
    from litho_sim.develop import measure_cd_2d

    m = ParameterModel()
    m.set("n_pixels", 96)
    m.set("source_grid", 7)
    m.set("pitch", 256.0)
    m.set("cd", 128.0)
    r = compute_imaging(m)
    height = r.thickness_nm * 1e-9
    film = r.film_nm * 1e-9
    px = m.grid().pixel_size

    sem = topdown_sem(height, px, _quiet(), film_thickness=film)
    meas = measure_cd_sem(sem.image, px, feature="line")
    assert meas.found
    reference = measure_cd_2d(height, px, threshold=0.5 * film, feature="above")
    assert reference > 0
    assert meas.cd == pytest.approx(reference, abs=2 * px)
