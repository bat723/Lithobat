"""A planar bulk nFET from two printed masks.

The companion piece to ``demo_gaa.py``, and deliberately the same discipline:
every patterned feature is *printed* on the device wafer — resist spun on the
real stack, exposed through a drawn layout with the full Abbe path, developed
in place — and every etch finds its mask by the engine's emergent-mask rule.
No step here takes a hand-drawn mask array.

Two lithographic levels are the only geometry inputs:

* **active** — the island that survives the shallow-trench recess;
* **gate**   — the stripe across it that becomes the poly gate.

Everything after the gate etch is self-aligned, which is the whole reason the
planar flow looked like this for twenty years: the spacer forms on the gate's
own sidewalls, the gate-oxide clear opens only what the gate and spacer do not
cover, the raised source/drain nucleates on the silicon that clear exposed, and
the silicide caps whatever silicon is still bare. None of them needs a mask.

The optics differ from the GAA on purpose. A 96 nm gate is comfortable for
193 nm immersion, which is the tool this architecture actually shipped on;
``demo_gaa.py`` moves to EUV because its gate pitch is out of ArF-i's reach.
Same engine, same printed-mask rule, different node.

Run ``python scripts/demo_nfet.py`` for the flow and the structural checks.
Use ``scripts/render_devices_3d.py`` to see the finished device in 3-D.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask.geometry import Rect
from litho_sim.mask.layout import Layout
from litho_sim.patterning.steps import ProcessContext
from litho_sim.tech.flows import FlowRun, contact_count, run_calibration
from litho_sim.wafer import Stack, get_material

# 193 nm immersion: the tool this architecture shipped on. mask_model stays
# "thin" — this script is about the flow, not about mask 3-D.
ARF_I = dict(wavelength=193e-9, NA=1.35, sigma_outer=0.9, sigma_inner=0.6,
             source_type="annular", source_grid=11, normalisation="clear")

RESIST_NM = 90.0

#: Dose-to-size, calibrated with ``--calibrate``: the dose whose printed resist
#: CD lands nearest the drawn CD for each level, per level because the two
#: features have different duty cycles on this field.
ACTIVE_DOSE = 1.80
GATE_DOSE = 1.50

#: A rate of zero is a hard stop — the same mechanism selectivity always uses.
STOPS = {"Si": 0.0, "SiO2": 0.0, "SiN": 0.0, "poly-Si": 0.0}


def build_nfet(n=128, px_nm=3.0,
               active_nm=210.0, gate_nm=96.0,
               sti_nm=50.0, gox_nm=4.0, poly_nm=90.0,
               spacer_nm=8.0, epi_nm=20.0, silicide_nm=4.0,
               active_dose=ACTIVE_DOSE, gate_dose=GATE_DOSE,
               verbose=True, stop_after: str | None = None,
               on_step: Callable | None = None,
               on_context: Callable | None = None):
    """Build the device. Two printed masks; no step takes a mask array.

    Every process operation is applied through a
    :class:`~litho_sim.tech.flows.FlowRun`, which carries the step parameters
    as data so the app's recipe panel can show what actually ran.

    Parameters
    ----------
    stop_after : {"active_litho", "gate_litho"}, optional
        Return early with the metrics gathered so far — used by dose
        calibration, which needs the printed CD without paying for the rest.
    on_step, on_context : callable, optional
        Recipe-recording hooks — see :class:`~litho_sim.tech.flows.FlowRun`.
    """
    px = px_nm * 1e-9
    grid = GridConfig(n_pixels=n, pixel_size=px, dz=2e-9, n_z_slices=5)
    W = n * px

    # Substrate deep enough to hold the trench and still leave bulk under it.
    s = Stack.blank(grid, dz=2e-9, substrate_thickness=(sti_nm + 40) * 1e-9,
                    headroom=(poly_nm + 260) * 1e-9)
    ctx = ProcessContext(
        grid=grid,
        optics=OpticsConfig(**ARF_I),
        resist=ResistConfig(thickness=RESIST_NM * 1e-9, threshold=0.32),
        layouts={
            "active": Layout([Rect(cx=0, cy=0, w=3 * W, h=active_nm * 1e-9)],
                             name="active"),
            "gate":   Layout([Rect(cx=0, cy=0, w=gate_nm * 1e-9, h=3 * W)],
                             name="gate"),
        },
    )
    run = FlowRun(s, ctx, verbose=verbose, on_step=on_step, on_context=on_context)

    # 1. ACTIVE LEVEL, printed. The resist island survives over the active; the
    #    trench etch below never sees a mask argument — resist on top of a
    #    column is what closes it, exactly as on a real wafer.
    run.print_level("active", "active", active_dose, measure_along="y")
    run.say(f" 1 active litho        drawn {active_nm:.0f} nm -> printed "
            f"{run.metrics['active_printed_nm']:.0f} nm resist")
    if stop_after == "active_litho":
        return s, run.metrics

    run.do("etch", targets="Si", depth=sti_nm * 1e-9,
           selectivity={"Si": 1.0})
    run.say(f" 2 STI etch            emergent mask: the resist is the mask "
            f"({sti_nm:.0f} nm)")
    run.do("strip")

    # 2. Trench fill and CMP back to the active. The polish stop is material
    #    selectivity — no mask — and it leaves the field oxide level with the
    #    silicon it isolates.
    run.do("deposit", material="SiO2", thickness=(sti_nm + 80) * 1e-9,
           conformal=False, planarize=True)
    run.do("cmp", stop_on="Si")
    run.say(" 3 STI fill + CMP      stops on the active silicon")

    # 3. Gate oxide, then poly planarized flat. Litho on a planarized film is
    #    not a convenience: it is why real flows planarize before every level.
    run.do("deposit", material="SiO2", thickness=gox_nm * 1e-9,
           conformal=False)
    run.say(f" 4 gate oxide          SiO2 {gox_nm:.0f} nm")
    run.do("deposit", material="poly-Si", thickness=poly_nm * 1e-9,
           conformal=False, planarize=True)
    poly_top = float(s.top_height().max())
    run.do("cmp", height=poly_top)
    run.say(f" 5 poly deposit        {poly_nm:.0f} nm, CMP flat")

    # 4. GATE LEVEL, printed on that flat surface.
    run.print_level("gate", "gate", gate_dose, measure_along="x")
    run.say(f" 6 gate litho          drawn {gate_nm:.0f} nm -> printed "
            f"{run.metrics['gate_printed_nm']:.0f} nm resist")
    if stop_after == "gate_litho":
        return s, run.metrics

    run.do("etch", targets="poly-Si", depth=(poly_nm + 40) * 1e-9,
           selectivity={"poly-Si": 1.0, **{k: 0.0 for k in
                                           ("Si", "SiO2", "SiN")}})
    run.say(" 7 gate etch           resist masks; stops on the gate oxide")
    run.do("strip")

    # 5. Spacer: conformal nitride, directional etch-back. What survives is the
    #    collar on the gate's own sidewalls — the self-alignment every later
    #    step keys off.
    run.do("deposit", material="SiN", thickness=spacer_nm * 1e-9,
           conformal=True)
    run.do("etchback", material="SiN", thickness=spacer_nm * 1e-9)
    run.say(f" 8 gate spacer         SiN {spacer_nm:.0f} nm, etched back to collars")

    # 6. Gate-oxide clear — and note what is NOT here: a mask. The etch attacks
    #    oxide wherever oxide is the exposed surface, which after the gate and
    #    spacer is exactly the source/drain and the field, and nowhere else.
    run.do("etch", targets="SiO2", depth=(gox_nm + 2) * 1e-9,
           selectivity={"SiO2": 1.0, **{k: 0.0 for k in
                                        ("Si", "SiN", "poly-Si")}})
    run.say(" 9 gate-ox clear       no mask: self-aligned to gate + spacer")

    # 7. Raised source/drain: selective epi, so it nucleates on silicon only.
    run.do("deposit", material="Si", thickness=5e-9, conformal=True, on="Si")
    run.do("deposit", material="Si", thickness=epi_nm * 1e-9,
           conformal=False, on="Si")
    run.say(f"10 raised S/D epi      selective on Si; {epi_nm:.0f} nm")

    # 8. Salicide: caps every bare silicon surface at once — source, drain and
    #    the top of the poly — and nothing else, because nothing else is
    #    silicon. The spacer is what keeps that from shorting gate to S/D.
    run.do("deposit", material="TiN", thickness=silicide_nm * 1e-9,
           conformal=True, on=["Si", "poly-Si"])
    run.say(f"11 salicide            TiN {silicide_nm:.0f} nm on exposed silicon")

    return s, run.metrics


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def checks(s, n):
    """The claims that make it an nFET, as numbers."""
    mid = n // 2
    si = get_material("Si").id
    oxide = get_material("SiO2").id
    nitride = get_material("SiN").id
    poly = get_material("poly-Si").id

    # Gate length: poly width along x at the channel row.
    poly_row = (s.mat[:, mid, :] == poly).any(axis=0)
    gate_len = float(poly_row.sum()) * s.pixel_size * 1e9

    # The channel column: silicon, then gate oxide, then poly, in that order.
    col = s.mat[:, mid, mid]
    poly_runs = s.column_runs(poly, mid, mid)
    gate_on_oxide = False
    if poly_runs:
        z0 = poly_runs[0][0]
        gate_on_oxide = bool(z0 >= 2 and col[z0 - 1] == oxide and
                             (col[:z0] == si).any())

    # Gate oxide unbroken: no poly voxel anywhere touches silicon directly.
    gate_shorts = contact_count(s, "poly-Si", "Si")

    # Spacer collars either side of the gate, on the channel row.
    nit_row = (s.mat[:, mid, :] == nitride).any(axis=0)
    collars, in_collar = 0, False
    for v in nit_row:
        if v and not in_collar:
            collars, in_collar = collars + 1, True
        elif not v:
            in_collar = False

    # Raised epi on both sides: silicon above the original active surface,
    # left and right of the gate.
    poly_x = np.flatnonzero(poly_row)
    epi_sides = 0
    if poly_x.size:
        top = s.top_height() * 1e9
        base = float(np.median(top[mid, :]))
        for sl in (slice(0, poly_x[0]), slice(poly_x[-1] + 1, n)):
            band = s.mat[:, mid, sl]
            if band.size and (band == si).any():
                # epi is silicon lying above the field-oxide top
                zi = np.argwhere(band == si)[:, 0].max() * s.dz * 1e9
                if zi > base * 0.35:
                    epi_sides += 1

    # STI: field oxide reaching below the active silicon surface.
    ox_runs = s.column_runs(oxide, 2, 2)
    sti_isolated = bool(ox_runs and (ox_runs[0][1] - ox_runs[0][0]) * s.dz * 1e9 > 20)

    return dict(
        gate_len_nm=round(gate_len, 1),
        gate_on_oxide=gate_on_oxide,
        gate_shorts=gate_shorts,
        spacer_collars=collars,
        sd_epi_sides=epi_sides,
        sti_isolated=sti_isolated,
    )


def report(s, n):
    c = checks(s, n)
    print(f"\nchecks at column ({n // 2},{n // 2}):")
    print("   " + "  ".join(f"{k}={v}" for k, v in c.items()))
    return c


# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate-nm", type=float, default=96.0)
    ap.add_argument("--section", action="store_true",
                    help="print an ASCII cross-section along the channel")
    ap.add_argument("--calibrate", action="store_true",
                    help="sweep dose for each level and report dose-to-size")
    a = ap.parse_args()

    if a.calibrate:
        run_calibration(
            build_nfet,
            levels=(("active_litho", 210.0), ("gate_litho", a.gate_nm)),
            doses=(0.9, 1.05, 1.2, 1.3, 1.45, 1.6),
            dose_keys=("active_dose", "gate_dose"),
            gate_nm=a.gate_nm,
        )
        return

    s, metrics = build_nfet(gate_nm=a.gate_nm)
    c = report(s, s.shape_xy[1])
    if a.section:
        print(s.ascii_section("y"))
    ok = (c["gate_on_oxide"] and c["gate_shorts"] == 0
          and c["spacer_collars"] == 2 and c["sd_epi_sides"] == 2
          and c["sti_isolated"])
    print(f"\n{'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}   "
          f"printed: active {metrics['active_printed_nm']:.0f} nm, "
          f"gate {metrics['gate_printed_nm']:.0f} nm")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
