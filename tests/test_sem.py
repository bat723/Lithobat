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
    measure_cd_sem,
    top_surface,
    topdown_sem,
    topdown_signal,
    xsection_sem,
    xsection_signal,
)

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
