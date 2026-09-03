"""
Regenerate the derived notes in the Obsidian vault at ``docs/``.

Run from the repository root::

    python scripts/build_docs.py            # regenerate everything
    python scripts/build_docs.py --fast     # skip the benchmarks

What this touches
-----------------
Only files it generates, each stamped with a banner:

* ``docs/reference/<area>/*.md`` and ``docs/reference/Reference.md`` — API
  surface, from introspection, grouped by feature area
* ``docs/patterning/recipes/*.md`` — step lists, from actually constructing each Flow
* ``docs/reference/Measured Numbers.md`` — benchmarks, from actually running them
* ``docs/assets/*.png`` — figures copied from ``results/``

Hand-written notes (the feature folders' concept notes and MOCs, ``docs/log/``,
``LithoPy``, ``Engine``, ``Roadmap``, ``Publishing and Privacy``,
``Mack Reading Map``) are **never** written to.

This exists because those four categories go stale the moment the code moves, and stale
API docs are worse than none. Everything conceptual stays hand-written, because that is
where the value is and generators cannot produce it.

Local only — writes nothing outside ``docs/``. See ``docs/Publishing and Privacy.md``.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import shutil
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

BANNER = (
    "> [!abstract] Generated\n"
    "> Written by `scripts/build_docs.py` — **do not hand-edit**, your changes will be\n"
    "> overwritten. Re-run the script instead.\n"
)

#: Feature areas, in pipeline order. Each maps to its hand-written MOC note
#: (None = no MOC yet) — the [[Reference]] index heading links to it.
#: Modules inside an area are auto-discovered with pkgutil, so a new module
#: gets a page without touching this file; only its concept links are curated.
AREAS: dict[str, str | None] = {
    "core": None,
    "mask": "Patterning",
    "expose": "Exposure",
    "coat": "Coating",
    "bake": "Bake",
    "develop": "Development",
    "wafer": "Wafer",
    "patterning": "Patterning",
    "analysis": "Analysis",
    "metrology": "Metrology",
    "viz": "Visualisation",
    "ml": None,
    "tech": "Patterning",
}

#: "area/module" → the concept notes it implements, so the graph connects
#: code to physics. Modules not listed here still get a page, just without
#: an Implements line.
MODULE_CONCEPTS: dict[str, list[str]] = {
    "mask/geometry": ["Layout Decomposition", "Overlay and Pitch Walking"],
    "mask/layout": ["Layout Decomposition"],
    "mask/patterns": ["Layout Decomposition"],
    "expose/pupil": ["Aerial Image Formation"],
    "expose/illumination": ["Partial Coherence", "Illumination Sources", "Source Sampling"],
    "expose/source": ["Illumination Sources", "Source Sampling"],
    "expose/aerial_image": ["Aerial Image Formation", "Source Sampling", "Defocus and Depth"],
    "coat/films": ["Thin-Film Interference", "Standing Waves"],
    "bake/peb": ["Standing Waves", "Development Models"],
    "develop/resist": ["Dill Exposure Model", "Development Models"],
    "develop/resist3d": ["Defocus and Depth", "Dill Exposure Model", "Standing Waves",
                         "Development Models"],
    "wafer/stack": ["Self-Aligned Patterning", "Engine"],
    "wafer/materials": ["Engine"],
    "patterning/steps": ["Emergent Behaviour", "Self-Aligned Patterning"],
    "patterning/flow": ["Emergent Behaviour"],
    "patterning/multipatterning": ["Self-Aligned Patterning", "Overlay and Pitch Walking",
                                   "Layout Decomposition"],
    "analysis/process_window": ["Defocus and Depth", "Process Window"],
    "analysis/stochastics": ["Roughness Metrics", "Stochastic Printing"],
    "metrology/sem": ["SEM Imaging", "Desktop App"],
    "viz/viz3d": ["Engine", "Interactive 3D Viewers"],
    "expose/m3d": ["Mask 3-D Effects"],
    "expose/m3d.multilayer": ["Mask 3-D Effects"],
    "expose/m3d.nearfield": ["Mask 3-D Effects", "FDTD Near-Field Solver"],
    "expose/m3d.provider": ["Mask 3-D Effects"],
    "expose/m3d.yee": ["FDTD Near-Field Solver"],
    "tech/devices": ["Printing a GAA Transistor", "Printing a Planar nFET"],
}


def _first_paragraph(doc: str | None) -> str:
    """First paragraph of a docstring, whitespace-normalised."""
    if not doc:
        return ""
    out: list[str] = []
    for line in inspect.cleandoc(doc).splitlines():
        if not line.strip():
            break
        out.append(line.strip())
    return " ".join(out)


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return "(...)"


def _area_modules(area: str) -> list[str]:
    """Public module names inside ``litho_sim.<area>``, sorted.

    Subpackages are walked, so ``expose/m3d/yee.py`` comes back as ``m3d.yee``
    and the subpackage itself as ``m3d``. The name stays dotted rather than
    becoming a nested folder because Obsidian resolves ``[[wikilinks]]`` by
    shortest path, and both ``wafer/materials`` and ``expose/m3d/materials``
    exist — a bare ``[[materials]]`` would be ambiguous.
    """
    pkg = importlib.import_module(f"litho_sim.{area}")
    prefix = pkg.__name__ + "."
    names = [
        m.name[len(prefix):]
        for m in pkgutil.walk_packages(pkg.__path__, prefix=prefix)
    ]
    return sorted(
        n for n in names
        if not any(part.startswith("_") for part in n.split("."))
    )


def _module_lines(mod: Any) -> tuple[str, int]:
    """(repo-relative source path, line count) for a module object."""
    src = Path(inspect.getsourcefile(mod))
    rel = src.relative_to(ROOT)
    return str(rel), len(src.read_text(encoding="utf-8").splitlines())


# ---------------------------------------------------------------------------
# Reference notes (per-module pages + the index)
# ---------------------------------------------------------------------------


def build_reference() -> list[str]:
    written: list[str] = []
    index_rows: dict[str, list[str]] = {}

    for area, moc in AREAS.items():
        (DOCS / "reference" / area).mkdir(parents=True, exist_ok=True)
        index_rows[area] = []

        for name in _area_modules(area):
            mod = importlib.import_module(f"litho_sim.{area}.{name}")
            rel, n_lines = _module_lines(mod)

            functions, classes = [], []
            for attr, obj in sorted(vars(mod).items()):
                if attr.startswith("_"):
                    continue
                if getattr(obj, "__module__", None) != mod.__name__:
                    continue  # imported, not defined here
                if inspect.isclass(obj):
                    classes.append((attr, obj))
                elif inspect.isfunction(obj):
                    functions.append((attr, obj))

            lines = [
                "---", "tags: [module, generated]", "---", "",
                f"# `litho_sim.{area}.{name}`", "", BANNER, "",
                f"**Source:** `{rel}` · {n_lines} lines", "",
            ]

            concepts = MODULE_CONCEPTS.get(f"{area}/{name}", [])
            if concepts:
                lines += ["**Implements:** " + " · ".join(f"[[{c}]]" for c in concepts), ""]

            summary = _first_paragraph(mod.__doc__)
            if summary:
                lines += ["## Purpose", "", summary, ""]

            if classes:
                lines += ["## Classes", ""]
                for attr, obj in classes:
                    lines.append(f"### `{attr}`")
                    desc = _first_paragraph(obj.__doc__)
                    if desc:
                        lines += ["", desc]
                    methods = [
                        (m, f) for m, f in sorted(vars(obj).items())
                        if not m.startswith("_")
                        and (inspect.isfunction(f) or isinstance(f, (classmethod, staticmethod)))
                    ]
                    if methods:
                        lines += ["", "```python"]
                        for m, f in methods:
                            fn = f.__func__ if isinstance(f, (classmethod, staticmethod)) else f
                            lines.append(f"{m}{_signature(fn)}")
                        lines.append("```")
                    lines.append("")

            if functions:
                lines += ["## Functions", ""]
                for attr, obj in functions:
                    lines.append(f"### `{attr}{_signature(obj)}`")
                    desc = _first_paragraph(obj.__doc__)
                    if desc:
                        lines += ["", desc]
                    lines.append("")

            path = DOCS / "reference" / area / f"{name}.md"
            path.write_text("\n".join(lines), encoding="utf-8")
            written.append(f"{area}/{name}.md")

            summary_short = summary
            if len(summary_short) > 90:
                summary_short = summary_short[:87].rsplit(" ", 1)[0] + "…"
            index_rows[area].append(f"| [[{name}]] | {n_lines} | {summary_short} |")

    # The index — one table per area, in pipeline order.
    idx = [
        "---", "tags: [moc, generated]", "---", "",
        "# Reference", "", BANNER, "",
        "Per-module API surface, grouped by feature area. The *reasoning* lives in",
        "[[Engine]] and the concept notes linked from each page; this is the",
        "reference layer. Benchmarks: [[Measured Numbers]].", "",
    ]
    for area, moc in AREAS.items():
        heading = f"## `{area}/`"
        if moc:
            heading += f" — [[{moc}]]"
        idx += [heading, "", "| Module | Lines | Purpose |", "|---|---|---|"]
        idx += index_rows[area]
        idx.append("")
    (DOCS / "reference" / "Reference.md").write_text("\n".join(idx), encoding="utf-8")
    written.append("Reference.md")
    return written


# ---------------------------------------------------------------------------
# Recipe notes
# ---------------------------------------------------------------------------

RECIPE_NOTES = {
    "LELE": (
        "Litho-Etch-Litho-Etch. The layout is split by [[Layout Decomposition|graph "
        "colouring]] so no two features closer than the minimum spacing share an "
        "exposure; each colour is then printed and etched into the same hardmask.\n\n"
        "The second exposure carries the overlay offset, which is where "
        "[[Overlay and Pitch Walking|pitch walking]] comes from. A second source of "
        "walk is a dose difference between colours (`dose_b`), since the two exposures "
        "are independent events."
    ),
    "LE3": (
        "Triple patterning — [[LELE]] with three colours. Needed when the layout "
        "contains an odd conflict cycle, which cannot be two-coloured no matter how "
        "good the solver is. See [[Layout Decomposition]]."
    ),
    "SADP": (
        "Self-Aligned Double Patterning. The mandrel prints at a relaxed pitch; the "
        "final features are derived from its *edges*, so the count doubles and the CD "
        "is set by deposition rather than by the optics.\n\n"
        "See [[Self-Aligned Patterning]]. Note SADP has its own pitch-walking mechanism "
        "independent of overlay: if the mandrel CD differs from the gap between "
        "mandrels, the two resulting spaces alternate."
    ),
    "SAQP": (
        "Self-Aligned Quadruple Patterning — the [[SADP]] block run twice, with the "
        "first spacer serving as the second mandrel. It needs **no new step types**, "
        "which is the strongest evidence the step abstraction is right.\n\n"
        "At 4 nm pixels a 12–16 nm second spacer is only 3–4 pixels wide, so SAQP CDs "
        "are quantisation-limited. Use a finer grid for quantitative work."
    ),
    "Cut Masks": (
        "A cut or block mask applied on top of any other scheme — the standard way one "
        "drawn line becomes several gates. It composes with everything because it is "
        "just another litho + etch pair, and because "
        "[[Emergent Behaviour|the etch mask is emergent]]. See [[Line-End Formation]] "
        "for the cut / block / keep semantics — the cut prints with `tone=\"clear\"` "
        "(drawn shapes open) and lands after the spacer strip, so the lines are "
        "actually exposed when the etch runs."
    ),
}


def build_recipe_notes() -> list[str]:
    from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
    from litho_sim.mask.layout import Layout, cut_bar, line_array
    from litho_sim.patterning import lele, sadp, saqp

    (DOCS / "patterning" / "recipes").mkdir(parents=True, exist_ok=True)

    grid = GridConfig(n_pixels=128, pixel_size=4e-9, dz=4e-9, n_z_slices=9)
    optics = OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=15)
    resist = ResistConfig(dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9)
    dense = Layout(line_array(6, pitch=80e-9, cd=40e-9, length=600e-9), name="gates")
    mandrel = Layout(line_array(4, pitch=160e-9, cd=80e-9, length=700e-9), name="mandrel")
    cut = Layout([cut_bar(0.0, 0.0, 200e-9, 60e-9)], name="cut")

    flows = {
        "LELE": lele(dense, grid, optics, resist, min_spacing=70e-9, overlay=(6e-9, 0)),
        "LE3": lele(dense, grid, optics, resist, min_spacing=70e-9, n_colors=3),
        "SADP": sadp(mandrel, grid, optics, resist, spacer_thickness=20e-9),
        "SAQP": saqp(mandrel, grid, optics, resist),
        "Cut Masks": sadp(
            mandrel, grid, optics, resist, spacer_thickness=20e-9, cut_layout=cut
        ),
    }

    written = []
    for key, blurb in RECIPE_NOTES.items():
        lines = ["---", "tags: [recipe, generated]", "---", "",
                 f"# {key}", "", BANNER, "", blurb, ""]
        flow = flows[key]
        source_note = (
            "Built by `litho_sim.patterning` — this listing is the real flow, "
            "not a description of it."
        )
        if key == "Cut Masks":
            source_note = (
                "The real SADP flow with `add_cut_mask()` appended — this listing is "
                "constructed, not transcribed."
            )
        lines += [
            f"## Steps ({len(flow)})", "",
            source_note, "",
            "```", *flow.describe(), "```", "",
        ]
        if flow.layouts:
            lines += [f"**Masks:** {', '.join(sorted(flow.layouts))}", ""]
        path = DOCS / "patterning" / "recipes" / f"{key}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        written.append(path.name)
    return written


# ---------------------------------------------------------------------------
# Measured numbers
# ---------------------------------------------------------------------------


def build_measured_numbers(fast: bool = False) -> str:
    import numpy as np

    from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
    from litho_sim.expose.aerial_image import compute_aerial_image
    from litho_sim.mask.layout import Layout, line_array
    from litho_sim.mask.patterns import lines_and_spaces

    out_path = DOCS / "reference" / "Measured Numbers.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "---", "tags: [reference, generated]", "---", "",
        "# Measured Numbers", "", BANNER, "",
        "Every figure here is produced by running the code, not transcribed. Re-run",
        "`python scripts/build_docs.py` to refresh after a change.", "",
    ]

    if fast:
        lines += ["> [!warning] Skipped", "> Run without `--fast` to populate.", ""]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return "Measured Numbers.md"

    # --- source sampling ---
    grid = GridConfig(n_pixels=128, pixel_size=4e-9)
    mask = lines_and_spaces(128, 4e-9, pitch=200e-9, cd=100e-9)
    t0 = time.perf_counter()
    dense = compute_aerial_image(mask, OpticsConfig(sigma_outer=0.8, source_grid=128), grid)
    t_dense = time.perf_counter() - t0

    lines += [
        "## Source sampling", "",
        "See [[Source Sampling]]. Dense = source sampled on the mask grid, i.e. the",
        "original behaviour.", "",
        "| source grid | time | speedup | max ΔI vs dense |", "|---|---|---|---|",
        f"| 128 (dense) | {t_dense:.3f} s | 1× | — |",
    ]
    for sg in (41, 31, 21, 15, 11):
        t0 = time.perf_counter()
        a = compute_aerial_image(mask, OpticsConfig(sigma_outer=0.8, source_grid=sg), grid)
        dt = time.perf_counter() - t0
        lines.append(
            f"| {sg} | {dt:.3f} s | {t_dense/dt:.0f}× | {np.abs(a - dense).max():.4f} |"
        )
    lines.append("")

    # --- aerial contrast vs pitch ---
    optics = OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=21)
    lines += [
        "## Aerial contrast vs pitch", "",
        "Why an 80 nm pitch needs [[LELE]]: a single exposure has no contrast to work",
        "with. See [[Development Models]] on why this forces per-pitch dose calibration.", "",
        "Measured on a **6-line `Layout`** — the same construction the recipes use. A",
        "full-field periodic array reads lower (0.15 at 80 nm) because it has no line",
        "ends; both are correct, they are different masks.", "",
        "| pitch | CD | contrast | single-exposure verdict |", "|---|---|---|---|",
    ]
    g_c = GridConfig(n_pixels=128, pixel_size=4e-9)
    for pitch in (80e-9, 120e-9, 160e-9, 200e-9, 240e-9):
        lay = Layout(line_array(6, pitch=pitch, cd=pitch / 2, length=600e-9))
        a = compute_aerial_image(lay.rasterize(g_c), optics, g_c)
        c = (a.max() - a.min()) / (a.max() + a.min())
        verdict = "unprintable" if c < 0.5 else ("marginal" if c < 0.75 else "prints")
        lines.append(f"| {pitch*1e9:.0f} nm | {pitch/2*1e9:.0f} nm | {c:.2f} | {verdict} |")
    lines.append("")

    # --- 3D cost + stack memory ---
    lines += [
        "## 3-D exposure cost", "",
        "| grid | z slices | source grid | time |", "|---|---|---|---|",
    ]
    from litho_sim.develop.resist3d import exposure_volume

    for n, nz in ((64, 11), (96, 15), (128, 15)):
        g = GridConfig(n_pixels=n, pixel_size=4e-9, dz=4e-9, n_z_slices=nz)
        r = ResistConfig(thickness=100e-9)
        m = lines_and_spaces(n, 4e-9, pitch=160e-9, cd=80e-9)
        t0 = time.perf_counter()
        exposure_volume(m, optics, g, r)
        lines.append(f"| {n}×{n} | {nz} | 21 | {time.perf_counter()-t0:.2f} s |")
    lines.append("")

    lines += [
        "## Stack memory and render payload", "",
        "Why voxels are affordable and volume rendering is not — see [[Engine]].", "",
        "| stack | uint8 memory |", "|---|---|",
    ]
    for n, nz in ((128, 64), (256, 128), (512, 256)):
        lines.append(f"| {n}×{n}×{nz} | {n*n*nz/1e6:.2f} MB |")
    lines.append("")

    from litho_sim.viz.viz3d import payload_size
    from litho_sim.wafer import Stack

    st = Stack.blank(GridConfig(n_pixels=128, pixel_size=4e-9), dz=4e-9,
                     substrate_thickness=20e-9, headroom=160e-9)
    st.deposit_blanket("poly-Si", 40e-9)
    st.deposit_blanket("photoresist", 80e-9)
    p = payload_size(st)
    lines += [
        f"Surface rendering sends **{p['surface_points']:,}** points where a volume would "
        f"send **{p['volume_points']:,}** — **{p['ratio']}× lighter**.", "",
    ]

    # --- pitch walking ---
    from litho_sim.patterning import lele

    g = GridConfig(n_pixels=128, pixel_size=4e-9, dz=4e-9, n_z_slices=9)
    r = ResistConfig(dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9)
    target = Layout(line_array(6, pitch=80e-9, cd=40e-9, length=600e-9), name="gates")
    lines += [
        "## Pitch walking vs overlay", "",
        "See [[Overlay and Pitch Walking]] and [[2026-08-05 Pitch Walk Sweep]].", "",
        "| overlay | printed CDs (nm) | walk | 2× overlay |", "|---|---|---|---|",
    ]
    for ov in (0.0, 3.0, 6.0, 9.0, 12.0):
        flow = lele(target, g, optics, r, min_spacing=70e-9, overlay=(ov * 1e-9, 0))
        m = flow.run(snapshot=False).measurements.iloc[-1]
        cds = ", ".join(f"{v:.0f}" for v in m["lines_nm"])
        lines.append(
            f"| {ov:.0f} nm | {cds} | {m['pitch_walk_nm']:.1f} nm | {2*ov:.0f} nm |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return "Measured Numbers.md"


# ---------------------------------------------------------------------------
# Assets + one-time migration guard
# ---------------------------------------------------------------------------


def copy_assets() -> list[str]:
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted((ROOT / "results").glob("*.png")):
        shutil.copy2(src, DOCS / "assets" / src.name)
        copied.append(src.name)
    return copied


def remove_stale_generated() -> list[str]:
    """Delete pre-reorg generated files (old locations), banner-checked."""
    removed = []
    candidates = [DOCS / "Modules Index.md", DOCS / "Measured Numbers.md"]
    for folder in (DOCS / "modules", DOCS / "recipes"):
        if folder.is_dir():
            candidates.extend(folder.glob("*.md"))
    for path in candidates:
        if path.exists() and "[!abstract] Generated" in path.read_text(encoding="utf-8"):
            path.unlink()
            removed.append(str(path.relative_to(DOCS)))
    for folder in (DOCS / "modules", DOCS / "recipes"):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    return removed


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="skip the benchmark run")
    args = ap.parse_args()

    if not DOCS.exists():
        raise SystemExit(f"No vault at {DOCS}. Expected docs/ to exist.")

    print("Regenerating derived notes in docs/ …\n")
    stale = remove_stale_generated()
    if stale:
        print(f"  migrated  : removed {len(stale)} stale generated file(s)")
    refs = build_reference()
    print(f"  reference : {len(refs)} notes")
    recs = build_recipe_notes()
    print(f"  recipes   : {len(recs)} notes")
    nums = build_measured_numbers(fast=args.fast)
    print(f"  numbers   : {nums}")
    assets = copy_assets()
    print(f"  assets    : {len(assets)} figures")

    print("\nHand-written notes untouched: feature folders' concepts + MOCs, log/, "
          "LithoPy, Engine, Roadmap, Publishing and Privacy, Mack Reading Map")
    print(f"Open the vault: {DOCS}")


if __name__ == "__main__":
    main()
