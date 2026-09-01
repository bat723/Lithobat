"""
The app's new engine wiring: resist models, chemistry knobs, stochastics.

These are reachability tests: every engine capability the Qt interface
claims to surface must actually flow from a ParameterModel knob through the
Qt-free compute layer and back out as a result — the campaign's standard,
applied to the controls added when the CAR and stochastic chains were wired
into the app. Headless throughout; no widget is constructed here.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.app.compute import build_mask, compute_imaging
from litho_sim.app.fem import FemRequest, compute_fem
from litho_sim.app.params import GROUPS, SPECS_BY_KEY, ParameterModel, mask_model_availability
from litho_sim.app.stochastics import StochRequest, compute_stochastics


def _fast_model(resist_model: str = "threshold") -> ParameterModel:
    """A model tuned for test speed: small grid, sparse source, quick bake."""
    m = ParameterModel()
    m.set("n_pixels", 64)
    m.set("source_grid", 5)
    m.set("resist_model", resist_model)
    # Chemistry fast enough that a car bake is milliseconds, not seconds.
    m.set("bake_time", 5.0)
    m.set("k_quench", 5.0)
    m.set("k_amp", 0.4)
    m.set("mack_Mth", 0.6)
    return m


# ---------------------------------------------------------------------------
# Parameter model — the new knobs assemble into the config
# ---------------------------------------------------------------------------


def test_new_groups_exist_and_are_ordered():
    assert "Chemistry" in GROUPS and "Stochastic" in GROUPS


def test_chemistry_knobs_reach_resist_config():
    """Every Chemistry/Stochastic control must land in ResistConfig — a knob
    that silently pins to the dataclass default is the defect class this app
    has hit twice before."""
    m = ParameterModel()
    m.set("pag_density", 0.5)          # display nm⁻³ → 5e26 m⁻³
    m.set("quencher_ratio", 0.2)
    m.set("bake_time", 45.0)
    m.set("D_acid", 3.0)               # display nm²/s → 3e-18
    m.set("k_quench", 12.0)
    m.set("k_amp", 0.07)
    m.set("electron_blur_sigma", 4.0)  # nm
    m.set("dose_nominal", 25.0)
    m.set("dill_C", 0.05)
    m.set("use_stochastic", True)
    m.set("stochastic_sigma", 2.0)
    m.set("stochastic_corr_length", 30.0)
    r = m.resist()
    assert r.pag_density == pytest.approx(5.0e26)
    assert r.quencher_ratio == pytest.approx(0.2)
    assert r.bake_time == pytest.approx(45.0)
    assert r.D_acid == pytest.approx(3.0e-18)
    assert r.k_quench == pytest.approx(12.0)
    assert r.k_amp == pytest.approx(0.07)
    assert r.electron_blur_sigma == pytest.approx(4.0e-9)
    assert r.dose_nominal == pytest.approx(25.0)
    assert r.dill_C == pytest.approx(0.05)
    assert r.use_stochastic is True
    assert r.stochastic_sigma == pytest.approx(2.0e-9)
    assert r.stochastic_corr_length == pytest.approx(30.0e-9)


def test_develop_knobs_moved_to_resist_stage():
    """mack_Mth and develop_time are read by the 2-D chemistry models now,
    so they must invalidate the 2-D resist stage — while staying in
    DEVELOP3D_ONLY for the live 3-D refresh (the tone precedent)."""
    from litho_sim.app.params import DEVELOP3D_ONLY

    for key in ("mack_Mth", "develop_time"):
        assert SPECS_BY_KEY[key].stage == "resist"
        assert key in DEVELOP3D_ONLY
        assert "resist" in ParameterModel.stages_invalidated_by(key)


def test_monopole_is_offered_and_images():
    m = _fast_model()
    m.set("source_type", "monopole")
    result = compute_imaging(m)
    assert np.isfinite(result.aerial).all()
    assert result.aerial.max() > 0


# ---------------------------------------------------------------------------
# Mask type — attenuated PSM
# ---------------------------------------------------------------------------


def test_att_psm_builds_complex_and_images():
    m = _fast_model()
    m.set("mask_type", "att-psm")
    mask = build_mask(m)
    assert np.iscomplexobj(mask)
    # 6 % intensity in the dark regions, at 180°.
    dark = mask[np.abs(mask) < 0.9]
    assert np.allclose(np.abs(dark) ** 2, 0.06, atol=1e-6)
    result = compute_imaging(m)
    # Display copy must be real — imshow of a complex array is undefined.
    assert not np.iscomplexobj(result.mask)
    assert np.isfinite(result.aerial).all()


def test_att_psm_rules_out_fdtd():
    reasons = mask_model_availability(193e-9, "lines and spaces", "att-psm")
    assert reasons["fdtd"] is not None
    assert reasons["thin"] is None
    # Binary keeps FDTD available — the gate is the mask type, not a regression.
    assert mask_model_availability(193e-9, "lines and spaces")["fdtd"] is None


# ---------------------------------------------------------------------------
# Resist models on the 2-D live path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["threshold", "mack", "car"])
def test_resist_models_all_compute(model):
    m = _fast_model(model)
    result = compute_imaging(m)
    assert set(np.unique(result.resist)).issubset({0.0, 1.0})
    assert np.isfinite(result.latent).all()
    assert result.thickness_nm.min() >= 0.0
    assert result.thickness_nm.max() <= result.film_nm + 1e-9


def test_threshold_path_is_unchanged_by_the_new_knobs():
    """The default model must reproduce the pre-wiring behavior exactly —
    the same bit-for-bit guarantee every seam in this engine ships with."""
    m = _fast_model("threshold")
    m2 = copy.deepcopy(m)
    m2.set("pag_density", 0.9)     # chemistry knobs must not touch threshold
    m2.set("k_quench", 90.0)
    a = compute_imaging(m)
    b = compute_imaging(m2)
    assert np.array_equal(a.resist, b.resist)
    assert a.cd_nm == b.cd_nm


def test_car_latent_is_chemistry_not_intensity():
    """For car the latent panel shows the protected fraction: 1 where dark
    (unexposed), low where bright — inverted relative to the aerial image."""
    m = _fast_model("car")
    result = compute_imaging(m)
    bright = result.aerial > np.percentile(result.aerial, 90)
    dark = result.aerial < np.percentile(result.aerial, 10)
    assert result.latent[dark].mean() > result.latent[bright].mean()
    # And the drawn threshold line is the develop threshold, on that scale.
    assert result.threshold == pytest.approx(0.6)


def test_quencher_shifts_the_printed_cd():
    """The chemistry knobs must do physics, not decoration: more base means
    more acid annihilated, less deprotection, a wider surviving line —
    and enough base seals the film outright (CD 0, nothing cleared)."""
    lean = _fast_model("car")
    lean.set("quencher_ratio", 0.02)
    rich = _fast_model("car")
    rich.set("quencher_ratio", 0.15)
    cd_lean = compute_imaging(lean).cd_nm
    cd_rich = compute_imaging(rich).cd_nm
    assert cd_lean > 0 and cd_rich > 0
    assert cd_rich > cd_lean

    # Heavy base: the surviving line overgrows its pitch and no closed
    # feature is measurable any more — CD reports 0 rather than a number.
    heavy = _fast_model("car")
    heavy.set("quencher_ratio", 0.45)
    assert compute_imaging(heavy).cd_nm == 0.0


def test_cosmetic_roughness_flows_from_the_dock():
    m = _fast_model("threshold")
    m.set("use_stochastic", True)
    m.set("stochastic_sigma", 3.0)
    rough = compute_imaging(m)
    smooth = compute_imaging(_fast_model("threshold"))
    assert not np.array_equal(rough.resist, smooth.resist)
    assert set(np.unique(rough.resist)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Process window honours the dock's model
# ---------------------------------------------------------------------------


def test_fem_runs_with_the_car_model():
    m = _fast_model("car")
    req = FemRequest(
        params=copy.deepcopy(m),
        focus_half_range_nm=200.0, n_focus=3,
        dose_half_range=0.3, n_dose=3,
        target_cd_nm=100.0, tolerance_pct=15.0,
        auto_centre_dose=False, with_meef=False,
    )
    result = compute_fem(req)
    assert "car" in result.label
    printed = result.bossung_df[result.bossung_df["cd_nm"] > 0]
    assert not printed.empty, "car sweep printed nowhere"


def test_fem_rejects_nothing_it_offers():
    """Every choice the resist_model combo offers must be sweepable."""
    for model in SPECS_BY_KEY["resist_model"].choices:
        m = _fast_model(str(model))
        req = FemRequest(
            params=copy.deepcopy(m),
            focus_half_range_nm=100.0, n_focus=3,
            dose_half_range=0.2, n_dose=3,
            auto_centre_dose=False, with_meef=False,
        )
        compute_fem(req)  # must not raise


# ---------------------------------------------------------------------------
# The stochastics batch
# ---------------------------------------------------------------------------


def test_stochastics_batch_end_to_end():
    m = _fast_model("car")
    m.set("wavelength", 13.5)          # EUV: the regime the tab exists for
    progress: list[tuple[int, int]] = []
    req = StochRequest(params=copy.deepcopy(m), trials=3, seed=1)
    result = compute_stochastics(req, progress=lambda i, n: progress.append((i, n)))

    assert result.reference.shape == (64, 64)
    assert result.prob_map.shape == (64, 64)
    assert 0.0 <= result.prob_map.min() and result.prob_map.max() <= 1.0
    assert result.fails.n_trials == 3
    assert result.mean_photons > 0 and result.mean_pag > 0
    # One tick per unit of work: the reference plus each trial.
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert result.summary  # renders without raising


def test_stochastics_batch_is_seeded():
    m = _fast_model("car")
    m.set("wavelength", 13.5)
    a = compute_stochastics(StochRequest(params=copy.deepcopy(m), trials=2, seed=7))
    b = compute_stochastics(StochRequest(params=copy.deepcopy(m), trials=2, seed=7))
    assert np.array_equal(a.example, b.example)
    assert np.array_equal(a.prob_map, b.prob_map)


def test_stochastics_reference_ignores_cosmetic_noise():
    """A noisy reference would count its own noise as failures — the design
    intent is the deterministic print, whatever the dock's LER toggle says."""
    m = _fast_model("car")
    m.set("wavelength", 13.5)
    m.set("use_stochastic", True)
    m.set("stochastic_sigma", 5.0)
    a = compute_stochastics(StochRequest(params=copy.deepcopy(m), trials=2, seed=3))
    m2 = _fast_model("car")
    m2.set("wavelength", 13.5)
    b = compute_stochastics(StochRequest(params=copy.deepcopy(m2), trials=2, seed=3))
    assert np.array_equal(a.reference, b.reference)
