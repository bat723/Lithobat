"""A gate-all-around nanosheet transistor from two printed masks.

Every patterned feature on this wafer is *printed*, on the wafer itself: resist
is spun onto the device stack, exposed through a drawn layout with the full
Abbe imaging path, developed in place, and the etch sees the resist because the
engine's emergent-mask rule protects whatever is on top of a column. No etch in
this flow takes a hand-drawn mask array.

Two lithographic levels are the only geometry inputs:

* **active/fin** — a stripe that becomes the superlattice fin;
* **gate** — a stripe across it that becomes the dummy gate.

Everything else is self-aligned, which is the actual beauty of the
replacement-metal-gate flow: the S/D recess is masked by the gate and its own
spacer, the STI recess stops on the fin, the CMP stops on the dummy gate, the
release enters through the trench the dummy gate leaves behind.

Run ``python scripts/demo_gaa.py --figure results/gaa_steps.png`` to get one
panel per step with depositions in green and etched material in red.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask.geometry import Rect
from litho_sim.mask.layout import Layout
from litho_sim.patterning.steps import ProcessContext
from litho_sim.tech.flows import FlowRun, contact_count, run_calibration
from litho_sim.wafer import Stack, get_material

# EUV, because the through-pitch study shows this gate-pitch regime is out of
# reach of single-exposure ArF-i (k1 < 0.25) while EUV holds to 40 nm pitch.
# mask_model stays "thin" here: this script is about the flow.
EUV = dict(wavelength=13.5e-9, NA=0.33, sigma_outer=0.8, sigma_inner=0.4,
           source_type="annular", source_grid=11, normalisation="clear")

RESIST_NM = 45.0

#: Dose-to-size, calibrated with ``--calibrate``: the dose whose printed
#: resist CD lands nearest the drawn CD for each level. Per-level, because the
#: two features have different duty cycles on this field.
FIN_DOSE = 1.8
GATE_DOSE = 1.8

#: Etch-stop shorthand: a rate of zero is a hard stop, the same mechanism
#: selectivity always uses.
STOPS = {"Si": 0.0, "SiGe": 0.0, "SiN": 0.0, "SiO2": 0.0}


# ---------------------------------------------------------------------------
# Step recording, for the per-step figure
# ---------------------------------------------------------------------------


@dataclass
class Snap:
    label: str
    axis: str          # "y" = along the channel, "x" = across the fin
    mat: np.ndarray
    view_only: bool = False   # re-view of the same state; no delta drawn


@dataclass
class Recorder:
    snaps: list[Snap] = field(default_factory=list)

    def __call__(self, s: Stack, label: str, axis: str, view_only=False) -> None:
        self.snaps.append(Snap(label, axis, s.mat.copy(), view_only))


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def build_gaa(n=112, px_nm=2.0, sheets=3, sac_nm=10.0, ch_nm=8.0,
              fin_nm=96.0, gate_nm=64.0,
              fin_dose=FIN_DOSE, gate_dose=GATE_DOSE,
              verbose=True, record: Callable | None = None,
              stop_after: str | None = None,
              on_step: Callable | None = None,
              on_context: Callable | None = None):
    """Build the device. Two printed masks; no step takes a mask array.

    Every process operation is applied through a
    :class:`~litho_sim.tech.flows.FlowRun`, which carries the step parameters
    as data so the app's recipe panel can show what actually ran.

    Parameters
    ----------
    stop_after : {"fin_litho", "gate_litho"}, optional
        Return early with the metrics gathered so far — used by dose
        calibration, which needs the printed CD without paying for the rest.
    record : callable, optional
        The per-step figure's hook, fired per *stage* of the narrative rather
        than per step; ``on_step`` is the sibling that fires per step.
    on_step, on_context : callable, optional
        Recipe-recording hooks — see :class:`~litho_sim.tech.flows.FlowRun`.
    """
    px = px_nm * 1e-9
    grid = GridConfig(n_pixels=n, pixel_size=px, dz=2e-9, n_z_slices=5)
    W = n * px
    stack_nm = sheets * (sac_nm + ch_nm)
    rec = record or (lambda *a, **k: None)

    s = Stack.blank(grid, dz=2e-9, headroom=(200 + stack_nm * 3) * 1e-9)
    ctx = ProcessContext(
        grid=grid,
        optics=OpticsConfig(**EUV),
        resist=ResistConfig(thickness=RESIST_NM * 1e-9, threshold=0.32),
        layouts={
            "fin":  Layout([Rect(cx=0, cy=0, w=3 * W, h=fin_nm * 1e-9)], name="fin"),
            "gate": Layout([Rect(cx=0, cy=0, w=gate_nm * 1e-9, h=3 * W)], name="gate"),
        },
    )
    rec(s, "bare substrate", "x")
    run = FlowRun(s, ctx, verbose=verbose, on_step=on_step, on_context=on_context)

    # 1. Superlattice: alternating sacrificial SiGe and channel Si.
    for _ in range(sheets):
        run.do("deposit", material="SiGe", thickness=sac_nm * 1e-9,
               conformal=False)
        run.do("deposit", material="Si", thickness=ch_nm * 1e-9,
               conformal=False)
    run.say(f" 1 superlattice        {sheets} x (SiGe {sac_nm:.0f} / Si {ch_nm:.0f} nm)")
    rec(s, f"superlattice — {sheets}x(SiGe {sac_nm:.0f} + Si {ch_nm:.0f} nm)", "x")

    # 2. ACTIVE LEVEL, printed. The resist stripe survives over the fin; the
    #    fin etch below never sees a mask argument — resist on top of a column
    #    is what closes it, exactly as on a real wafer.
    run.print_level("fin", "fin", fin_dose, measure_along="y")
    run.say(f" 2 fin litho           drawn {fin_nm:.0f} nm -> printed "
            f"{run.metrics['fin_printed_nm']:.0f} nm resist")
    rec(s, f"FIN LITHO — resist {run.metrics['fin_printed_nm']:.0f} nm "
           f"(drawn {fin_nm:.0f})", "x")
    if stop_after == "fin_litho":
        return s, run.metrics

    run.do("etch", targets=["Si", "SiGe"], depth=(stack_nm + 8) * 1e-9,
           selectivity={"Si": 1.0, "SiGe": 1.0})
    run.say(" 3 fin etch            emergent mask: the resist is the mask")
    rec(s, "fin etch — resist protects the fin", "x")
    run.do("strip")
    rec(s, "resist strip", "x")

    # 3. Shallow-trench isolation: fill, CMP to the fin, recess to re-expose
    #    the superlattice. The CMP stop and the recess stop are both material
    #    selectivity — no mask.
    run.do("deposit", material="SiO2", thickness=(stack_nm + 120) * 1e-9,
           conformal=False, planarize=True)
    run.do("cmp", stop_on="Si")
    rec(s, "STI fill + CMP stops on fin", "x")
    run.do("etch", targets="SiO2", depth=(stack_nm + 2) * 1e-9,
           selectivity={"SiO2": 1.0,
                        **{k: 0.0 for k in ("Si", "SiGe", "SiN")}})
    run.say(" 4 STI                 fill, CMP on fin, selective recess")
    rec(s, "STI recess — stops on silicon", "x")

    # 4. Dummy gate: poly, planarized flat (a conformal blanket over the fin
    #    leaves a step the ILD later seals — the bug the first build hit),
    #    then the GATE LEVEL is printed on that flat surface. Litho happening
    #    only on planarized films is not a convenience here; it is why real
    #    flows planarize before every masking level.
    poly_top = (40 + stack_nm + 60) * 1e-9
    run.do("deposit", material="poly-Si", thickness=poly_top,
           conformal=False, planarize=True)
    run.do("cmp", height=poly_top)
    rec(s, "dummy poly, CMP flat", "y")

    run.print_level("gate", "gate", gate_dose, measure_along="x")
    run.say(f" 5 gate litho          drawn {gate_nm:.0f} nm -> printed "
            f"{run.metrics['gate_printed_nm']:.0f} nm resist")
    rec(s, f"GATE LITHO — resist {run.metrics['gate_printed_nm']:.0f} nm "
           f"(drawn {gate_nm:.0f})", "y")
    if stop_after == "gate_litho":
        return s, run.metrics

    run.do("etch", targets="poly-Si", depth=poly_top + 40e-9,
           selectivity={"poly-Si": 1.0, **STOPS})
    run.say(" 6 gate etch           resist masks; stops on fin and STI")
    rec(s, "gate etch — dummy gate stands", "y")
    run.do("strip")
    rec(s, "resist strip", "y")

    # 5. Gate spacer: conformal nitride, directional etch-back. What survives
    #    is the collar on the dummy-gate sidewalls — the self-alignment that
    #    keys every later step.
    run.do("deposit", material="SiN", thickness=5e-9, conformal=True)
    run.do("etchback", material="SiN", thickness=5e-9)
    run.say(" 7 gate spacer         SiN 5 nm, etched back to collars")
    rec(s, "gate spacer — SiN collars", "y")

    # 6. S/D recess — and note what is NOT here: a mask. The etch attacks
    #    Si/SiGe wherever they are the exposed surface, which after the gate
    #    and spacer is exactly the source/drain fin and nowhere else: poly
    #    caps the channel, SiN caps the spacer ring, SiO2 caps the field.
    run.do("etch", targets=["Si", "SiGe"], depth=(stack_nm + 6) * 1e-9,
           selectivity={"Si": 1.0, "SiGe": 1.0})
    run.say(" 8 S/D recess          no mask: self-aligned to gate + spacer")
    rec(s, "S/D recess — emergent, self-aligned", "y")

    # 7. Inner spacer: lateral recess of each sheet end, conformal fill,
    #    etch-back, and a lateral pull-back to re-open the sheet ends the
    #    directional etch-back cannot reach.
    run.do("etch", targets="SiGe", depth=8e-9, exposure="any")
    run.do("deposit", material="SiN", thickness=4e-9, conformal=True)
    run.do("etchback", material="SiN", thickness=4e-9)
    run.do("etch", targets="SiN", depth=5e-9, exposure="any")
    run.say(" 9 inner spacer        8 nm recess, fill, etch-back, 5 nm pull-back")
    rec(s, "inner spacer — pockets at sheet ends", "y")

    # 8. Source/drain epi: selective, so it nucleates on silicon only.
    #    Conformal first to seed the vertical sheet ends (a bottom-up deposit
    #    alone leaves every sheet electrically stranded), then raised past the
    #    top sheet.
    run.do("deposit", material="Si", thickness=6e-9, conformal=True, on="Si")
    run.do("deposit", material="Si", thickness=(stack_nm + 14) * 1e-9,
           conformal=False, on="Si")
    run.say("10 S/D epi             selective on Si; seeds the sheet ends")
    rec(s, "S/D epi — grows on silicon only", "y")

    # 9. ILD, CMP stopping on the dummy gate — which is what makes the gate
    #    removable as its own self-aligned window.
    run.do("deposit", material="SiO2", thickness=(stack_nm + 200) * 1e-9,
           conformal=False, planarize=True)
    run.do("cmp", stop_on="poly-Si")
    run.say("11 ILD + CMP           stops on the dummy gate")
    rec(s, "ILD fill + CMP stops on dummy gate", "y")

    # 10. Replacement gate: pull the dummy, release the channel through the
    #     trench it leaves, wrap the real gate.
    run.do("strip", material="poly-Si")
    rec(s, "dummy gate pulled — trench open", "y")
    sige = get_material("SiGe").id
    before = int((s.mat == sige).sum())
    run.do("etch", targets="SiGe", depth=80e-9, exposure="any")
    run.say(f"12 channel release     lateral front, SiGe {before} -> "
            f"{int((s.mat == sige).sum())} voxels")
    rec(s, "channel release — sheets suspended", "y")
    run.do("deposit", material="SiO2", thickness=3e-9, conformal=True)
    run.do("deposit", material="TiN", thickness=8e-9, conformal=True)
    run.say("13 gate stack          SiO2 3 nm + TiN 8 nm, wraps every sheet")
    rec(s, "gate oxide + metal — wrapped", "y")
    rec(s, "final device, across the fin", "x", view_only=True)

    return s, run.metrics


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def sheet_runs(s, x, y):
    """Contiguous runs of channel silicon in one column, bottom-up."""
    return s.column_runs("Si", x, y)


def checks(s, n):
    """The five claims that make it a GAA transistor, as numbers."""
    from scipy.ndimage import label

    mid = n // 2
    runs = sheet_runs(s, mid, mid)
    oxide = get_material("SiO2").id
    lab, _ = label(s.mat == get_material("Si").id)
    return dict(
        sheets=len(runs) - 1,
        wrapped=sum(1 for lo, hi in runs[1:]
                    if s.mat[lo - 1, mid, mid] == oxide
                    and s.mat[hi, mid, mid] == oxide),
        sacrificial_left=int((s.mat == get_material("SiGe").id).sum()),
        gate_shorts=contact_count(s, "TiN", "Si"),
        sd_tied=bool(all(lab[(lo + hi) // 2, mid, mid] == lab[2, mid, mid]
                         for lo, hi in runs[1:])),
    )


def report(s, n):
    c = checks(s, n)
    mid = n // 2
    oxide = get_material("SiO2").id
    print(f"\nchecks at column ({mid},{mid}):")
    for lo, hi in sheet_runs(s, mid, mid)[1:]:
        u, o = s.mat[lo - 1, mid, mid], s.mat[hi, mid, mid]
        print(f"   sheet z {lo:3d}-{hi:3d}  under={u} over={o}"
              f"   {'GATE ALL AROUND' if u == oxide and o == oxide else 'NOT WRAPPED'}")
    print("   " + "  ".join(f"{k}={v}" for k, v in c.items()))
    return c


# ---------------------------------------------------------------------------
# The step-by-step figure
# ---------------------------------------------------------------------------


def render_steps(recorder: Recorder, s: Stack, path: str) -> None:
    """One panel per step: green = deposited this step, red = etched away."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from litho_sim.viz.viz3d import material_colormap

    # The palette must cover every material that appears in ANY snapshot, not
    # just the final stack: the sacrificial SiGe, the dummy poly and the resist
    # are all gone by the end — which is exactly why they must be drawable.
    ids = sorted({int(v) for sn in recorder.snaps for v in np.unique(sn.mat)} - {0})
    lut, cmap, norm, mat_objs = material_colormap(ids, vacuum="#ffffff")

    snaps = recorder.snaps
    # The stack grows its z extent when a deposit outruns the headroom, so
    # snapshots from different points in the flow can have different heights.
    # Pad them all to the tallest with vacuum before any differencing.
    nz_max = max(sn.mat.shape[0] for sn in snaps)
    for sn in snaps:
        if sn.mat.shape[0] < nz_max:
            pad = np.zeros((nz_max - sn.mat.shape[0],) + sn.mat.shape[1:],
                           dtype=sn.mat.dtype)
            sn.mat = np.concatenate([sn.mat, pad], axis=0)
    panels = snaps[1:]
    ncol = 4
    nrow = int(np.ceil((len(panels) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 2.55 * nrow))
    fig.suptitle(
        "GAA nanosheet, step by step — two printed masks, everything else "
        "self-aligned   (green = deposited, red = etched)",
        fontsize=15, weight="bold")

    mid = s.shape_xy[1] // 2
    px_nm, dz_nm = s.pixel_size * 1e9, s.dz * 1e9

    for k, snap in enumerate(panels):
        ax = axes.flat[k]
        prev = snaps[k]                       # snaps[k] precedes panels[k]
        sec = snap.mat[:, mid, :] if snap.axis == "y" else snap.mat[:, :, mid]
        psec = prev.mat[:, mid, :] if snap.axis == "y" else prev.mat[:, :, mid]

        top = max(int(np.max(np.nonzero(sec)[0], initial=0)),
                  int(np.max(np.nonzero(psec)[0], initial=0)))
        w = sec.shape[1] * px_nm
        ax.imshow(lut[sec], origin="lower", cmap=cmap, norm=norm, aspect="auto",
                  extent=(0, w, 0, sec.shape[0] * dz_nm), interpolation="nearest")

        note = ""
        if not snap.view_only:
            added3 = (prev.mat == 0) & (snap.mat != 0)
            removed3 = (prev.mat != 0) & (snap.mat == 0)
            swapped3 = (prev.mat != snap.mat) & (prev.mat != 0) & (snap.mat != 0)
            a2 = (psec == 0) & (sec != 0)
            r2 = ((psec != 0) & (sec == 0)) | \
                 ((psec != sec) & (psec != 0) & (sec != 0))
            for m2, colr in ((a2, "#00c853"), (r2, "#e53935")):
                if m2.any():
                    ax.imshow(np.ma.masked_where(~m2, np.ones(sec.shape)),
                              origin="lower", cmap=ListedColormap([colr]),
                              alpha=0.55, aspect="auto",
                              extent=(0, w, 0, sec.shape[0] * dz_nm),
                              interpolation="nearest")
            note = (f"  +{int(added3.sum()):,} / "
                    f"−{int((removed3 | swapped3).sum()):,} vox")

        ax.set(ylim=(18, (top + 6) * dz_nm), xticks=[], yticks=[])
        ax.set_title(f"{k + 1}. {snap.label}{note}", fontsize=8.3)

    ax = axes.flat[len(panels)]
    ax.axis("off")
    c = checks(s, s.shape_xy[1])
    ax.text(0.02, 0.97,
            "verification\n" + "\n".join(f"  {k} = {v}" for k, v in c.items()),
            va="top", family="monospace", fontsize=10)
    ax.legend(handles=[Patch(facecolor=m.color, label=m.name) for m in mat_objs],
              loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    for ax in axes.flat[len(panels) + 1:]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=115, facecolor="white")
    print(f"saved {path}")


# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheets", type=int, default=3)
    ap.add_argument("--gate-nm", type=float, default=64.0)
    ap.add_argument("--figure", type=str, default=None,
                    help="write the per-step figure here")
    ap.add_argument("--calibrate", action="store_true",
                    help="sweep dose for each level and report dose-to-size")
    a = ap.parse_args()

    if a.calibrate:
        run_calibration(
            build_gaa,
            levels=(("fin_litho", 96.0), ("gate_litho", a.gate_nm)),
            doses=(0.9, 1.05, 1.2, 1.35, 1.5),
            dose_keys=("fin_dose", "gate_dose"),
            sheets=a.sheets, gate_nm=a.gate_nm,
        )
        return

    rec = Recorder() if a.figure else None
    s, metrics = build_gaa(sheets=a.sheets, gate_nm=a.gate_nm, record=rec)
    c = report(s, s.shape_xy[1])
    if a.figure:
        render_steps(rec, s, a.figure)
    ok = (c["wrapped"] == c["sheets"] and c["sacrificial_left"] == 0
          and c["gate_shorts"] == 0 and c["sd_tied"])
    print(f"\n{'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}   "
          f"printed: fin {metrics['fin_printed_nm']:.0f} nm, "
          f"gate {metrics['gate_printed_nm']:.0f} nm")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
