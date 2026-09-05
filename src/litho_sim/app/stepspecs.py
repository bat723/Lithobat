"""
What a process step's parameters look like to a user.

``dataclasses.fields(Etch)`` gives names, types and defaults. It does not give
that ``depth`` is in metres and wants a nanometre slider, that ``anisotropy``
runs 0 to 1, or that ``material`` should be a dropdown of the eleven things the
library actually knows about. That is the same argument
:mod:`litho_sim.app.params` already makes for ``ParamSpec`` — the engine's
dataclasses carry defaults but no bounds — so this reuses ``ParamSpec`` rather
than inventing a parallel vocabulary, and the same widget builder draws both.

``stage`` is ``"view"`` throughout, which is true rather than a convenience:
none of these touch the imaging pipeline, so none of them may invalidate its
cache.

Two fields do not map to a single widget
----------------------------------------
``Etch`` carries the only two awkward fields in the vocabulary, and both are
translated here rather than exposed raw:

* ``selectivity: Dict[str, float]`` — a whole table, and in practice a user
  wants one row of it: *stop on this material*. Presented as a material choice
  plus a rate, and reassembled into the dict by :func:`build_step`.
* ``targets: Any`` — a string or a sequence of strings, the only polymorphic
  field. Presented as a single material, which covers every recipe in the
  codebase today; a multi-target etch still round-trips through
  :meth:`~litho_sim.patterning.steps.ProcessStep.from_dict` untouched.

Construct by keyword, always
----------------------------
``kind`` is permanently field 0 on every step (dataclass inheritance orders
fields by first appearance, and ``ProcessStep`` declares ``kind`` first), so
``Strip("photoresist")`` silently yields a step whose *kind* is
``"photoresist"``. :func:`build_step` goes through ``from_dict``, which is
keyword-only by construction and raises a useful error on an unknown kind.
"""

from __future__ import annotations

import dataclasses
from dataclasses import MISSING, fields
from typing import Any, cast

from litho_sim.app.params import ParamSpec
from litho_sim.patterning import STEP_REGISTRY, Etch, ProcessStep
from litho_sim.wafer import MATERIAL_LIBRARY

__all__ = ["STEP_SPECS", "PALETTE", "build_step", "values_for", "readout_for",
           "apply_edit", "specs_for_step", "uneditable_keys", "format_field",
           "material_choices", "defaults_for", "top_material_name",
           "open_floor_material_name"]


def material_choices() -> tuple[str, ...]:
    """Every material a step may name, vacuum excluded."""
    return tuple(sorted(n for n in MATERIAL_LIBRARY if n != "vacuum"))


_MATERIALS = material_choices()

#: Sentinel for "this etch stops on nothing in particular".
NO_STOP = "—"

#: Sentinel for "this film grows off every exposed surface", as against
#: selective epitaxy that nucleates on one material only. Same glyph as
#: :data:`NO_STOP` and deliberately a different name: they mean different
#: things, and a later change to one should not silently move the other.
NO_SEED = "—"

#: Spec keys that are *not* fields on the step, per kind — they are translated
#: by :func:`build_step` into whatever the dataclass actually wants. Kind-aware
#: because `stop_on` is synthetic for an etch (it folds into `selectivity`) and
#: a genuine field on a CMP.
_SYNTHETIC: dict[str, tuple[str, ...]] = {
    "etch": ("stop_on", "stop_rate"),
    "cmp": ("cmp_mode", "depth", "height", "stop_on"),
}


def _material(key: str, label: str, default: str, help: str) -> ParamSpec:
    return ParamSpec(
        key, label, "choice", default, choices=_MATERIALS,
        group="Step", target="view", stage="view", help=help,
    )


def _nm(key: str, label: str, default: float, lo: float, hi: float,
        step: float, help: str) -> ParamSpec:
    return ParamSpec(
        key, label, "float", default, lo, hi, step, "nm", 1e-9,
        group="Step", target="view", stage="view", help=help,
    )


#: One tuple of specs per step kind. Only the kinds the palette offers — the
#: rest of the registry still serialises fine, it just has no editor yet.
STEP_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "spincoat": (
        _material("material", "Material", "photoresist",
                  "What is being spun on. Usually resist, but the same step "
                  "coats a spin-on carbon hardmask."),
        _nm("thickness", "Thickness", 90.0, 10.0, 500.0, 5.0,
            "As-coated film thickness."),
        ParamSpec("planarize", "Planarising", "bool", True,
                  group="Step", target="view", stage="view",
                  help="A spun film fills topography and comes out flat. Turn "
                       "this off for a film that follows the surface."),
    ),
    "deposit": (
        _material("material", "Material", "spacer-oxide",
                  "The film being deposited."),
        _nm("thickness", "Thickness", 20.0, 1.0, 200.0, 1.0,
            "Deposited thickness — and, for a conformal film over a mandrel, "
            "the spacer CD that survives the etch-back."),
        ParamSpec("conformal", "Conformal", "bool", True,
                  group="Step", target="view", stage="view",
                  help="Conformal grows the same thickness in every direction, "
                       "including sideways off a vertical sidewall — which is "
                       "the entire basis of self-aligned pitch division. "
                       "Blanket follows the topography downward only."),
        ParamSpec("planarize", "Planarising", "bool", False,
                  group="Step", target="view", stage="view",
                  help="Fill the topography and come out flat, the way a thick "
                       "oxide fill does before a CMP. Blanket films only — a "
                       "conformal film that planarized would not be conformal."),
        ParamSpec("on", "Nucleates on", "choice", NO_SEED,
                  choices=(NO_SEED,) + _MATERIALS,
                  group="Step", target="view", stage="view",
                  help="Selective epitaxy: grow only where this material is "
                       "exposed, and nowhere else. Source/drain silicon "
                       "nucleates on silicon, which is what connects the sheet "
                       "ends. '—' grows off every exposed surface."),
    ),
    "pattern": (
        _material("material", "Material", "photoresist",
                  "The masking film being printed. Resist for an ordinary "
                  "litho step; a hardmask if the mask has to survive a long "
                  "etch."),
        _nm("thickness", "Thickness", 90.0, 10.0, 400.0, 5.0,
            "As-coated film thickness, before it is opened up."),
        ParamSpec("shape", "Shape", "choice", "lines",
                  choices=("lines", "contacts"),
                  group="Step", target="view", stage="view",
                  help="Lines and spaces, or a square array of holes."),
        _nm("pitch", "Pitch", 128.0, 20.0, 400.0, 4.0,
            "Centre-to-centre spacing. The array is sized to the field at run "
            "time, so this stays right at any grid."),
        _nm("cd", "CD", 64.0, 4.0, 200.0, 2.0,
            "Width of the opening — of the trench for lines, of the hole for "
            "contacts. At or above the pitch the film comes out unbroken and "
            "there is no mask edge at all."),
        ParamSpec("orientation", "Orientation", "choice", "vertical",
                  choices=("vertical", "horizontal"),
                  group="Step", target="view", stage="view",
                  help="Which way the lines run. Ignored for contacts."),
        ParamSpec("planarize", "Planarising", "bool", True,
                  group="Step", target="view", stage="view",
                  help="A spun film fills topography and comes out flat. Turn "
                       "this off for a film that follows the surface."),
    ),
    "etch": (
        _material("targets", "Etch", "SOC",
                  "The material that is open to this etch. A column etches "
                  "only where THIS is the exposed surface — anything else "
                  "sitting on top masks what is below it, which is why "
                  "patterned resist, a hardmask and a freestanding spacer all "
                  "work as masks without anything being told they are one."),
        _nm("depth", "Depth", 60.0, 1.0, 400.0, 1.0,
            "Etch budget, in nanometres of a unit-rate material. A slower "
            "material burns more of it per nanometre."),
        ParamSpec("anisotropy", "Anisotropy", "float", 1.0, 0.0, 1.0, 0.05,
                  group="Step", target="view", stage="view",
                  help="1.0 is perfectly directional — straight down, no "
                       "undercut. Below 1 the etch also spreads sideways, "
                       "which is how you get undercut and, with a slow "
                       "surface layer, a T-top."),
        ParamSpec("stop_on", "Stop on", "choice", NO_STOP,
                  choices=(NO_STOP,) + _MATERIALS,
                  group="Step", target="view", stage="view",
                  help="Give one material a low etch rate so the etch arrests "
                       "when it reaches it. This is the whole of selectivity "
                       "as far as the interface is concerned."),
        # Bounds chosen so `lo + i*step` lands exactly on 0.0, 0.005, 0.01,
        # 0.05 … — the numbers a user reaches for. The old grid started at
        # 0.001 and stepped by 0.005, so it could not express 0.01 at all, and
        # the default sat on a value the slider could never return to.
        ParamSpec("stop_rate", "Stop rate", "float", 0.005, 0.0, 1.0, 0.005,
                  group="Step", target="view", stage="view",
                  help="How fast the stop layer etches, relative to nominal. "
                       "Zero is a perfect stop; 1.0 is no selectivity at all."),
        ParamSpec("blanket", "Blanket", "bool", False,
                  group="Step", target="view", stage="view",
                  help="Recess the whole surface regardless of which material "
                       "each column exposes. The one thing the emergent mask "
                       "cannot express: normally something on top protects "
                       "what is below, and here nothing does."),
        ParamSpec("line_of_sight", "Line of sight", "bool", False,
                  group="Step", target="view", stage="view",
                  help="Ion flux travels in straight lines and cannot turn a "
                       "corner, so a voxel etches only if the material above "
                       "it in its own column is already gone. Everything past "
                       "the first obstruction is shadowed."),
        ParamSpec("contact", "Point of contact", "bool", False,
                  group="Step", target="view", stage="view",
                  help="The etchant has to physically reach it. Permits eating "
                       "under an overhang; forbids opening a sealed void. Both "
                       "this and line of sight on is a realistic RIE; contact "
                       "alone is a wet etch."),
        ParamSpec("exposure", "Reaches", "choice", "surface",
                  choices=("surface", "any"),
                  group="Step", target="view", stage="view",
                  help="Which surfaces this etch can attack. 'surface' opens a "
                       "column only where the target is the material on top — "
                       "right for every top-down patterning etch. 'any' attacks "
                       "the target wherever the etchant reaches it, sidewalls "
                       "included: a channel release, an inner spacer or a "
                       "source/drain undercut is buried under a cap and open "
                       "only on the wall of a trench, which a top-down etch "
                       "cannot turn a corner to reach. Slower — it solves a "
                       "front through the wafer rather than a column integral."),
        _nm("bias", "Bias", 0.0, -30.0, 30.0, 1.0,
            "Lateral offset at the top surface. Positive widens the opening — "
            "the etch undercuts the mask; negative pulls it in."),
        ParamSpec("sidewall_deg", "Sidewall angle", "float", 90.0, 60.0, 120.0, 1.0, "°",
                  group="Step", target="view", stage="view",
                  help="Wall angle from horizontal, 90° vertical — the same "
                       "convention the resist profile uses. Below 90° the hole "
                       "narrows with depth; above 90° it flares, which is "
                       "re-entrant and makes a later fill pinch off."),
        _nm("top_radius", "Top corner r", 0.0, 0.0, 50.0, 1.0,
            "Rounding where the wall meets the top surface, from mask erosion."),
        _nm("bottom_radius", "Bottom corner r", 0.0, 0.0, 50.0, 1.0,
            "Fillet where the wall meets the floor."),
        _nm("footer_height", "Footer height", 0.0, 0.0, 60.0, 2.0,
            "How tall the foot of unetched material at the base is."),
        _nm("footer_extent", "Footer extent", 0.0, 0.0, 40.0, 1.0,
            "How far that foot juts into the opening, at the floor. Tapers to "
            "nothing at the footer height."),
    ),
    "etchback": (
        _material("material", "Material", "spacer-oxide",
                  "The conformal film being etched back to leave sidewalls."),
        ParamSpec("overetch", "Over-etch", "float", 0.2, 0.0, 1.0, 0.05,
                  group="Step", target="view", stage="view",
                  help="Fraction beyond the flat-field thickness, to make sure "
                       "the horizontal film clears everywhere."),
    ),
    "cmp": (
        ParamSpec("cmp_mode", "Mode", "choice", "depth",
                  choices=("depth", "height", "stop"),
                  group="Step", target="view", stage="view",
                  help="'depth' removes a set amount measured down from the "
                       "current high point — what you use when the incoming "
                       "topography is whatever the last step left. 'height' is "
                       "an absolute level. 'stop' polishes until the pad "
                       "reaches a layer and stops there."),
        _nm("depth", "CMP depth", 30.0, 1.0, 300.0, 1.0,
            "How much to take off, measured down from the highest point on the "
            "wafer. Used when Mode is 'depth'."),
        _nm("height", "Polish to", 100.0, 5.0, 500.0, 5.0,
            "Absolute level to flatten at. Used when Mode is 'height'."),
        ParamSpec("stop_on", "Stop on", "choice", "SiN", choices=_MATERIALS,
                  group="Step", target="view", stage="view",
                  help="Polish until the pad reaches this material. The plane "
                       "lands at its HIGHEST point anywhere on the wafer, "
                       "because that is where the pad touches it first — so a "
                       "polish stop leaves material behind wherever the stop "
                       "layer is low. Used when Mode is 'stop'."),
    ),
    "strip": (
        _material("material", "Material", "photoresist",
                  "Removed everywhere it appears — a wet strip, an ash, or a "
                  "mandrel pull, which needs no special step of its own."),
    ),
    # The litho triplet. These are not in `PALETTE` — `pattern` is still how
    # you *add* lithography to a flow you are writing — but a printed device's
    # recipe contains them, and dose is the first thing anyone reaches for when
    # a printed CD misses the drawn one. Without these the two exposures were
    # the only steps in a GAA with nothing to adjust.
    "expose": (
        ParamSpec("layout", "Layout", "choice", "main", choices=("main",),
                  group="Step", target="view", stage="view",
                  help="Which drawn layout this exposure prints. The choices "
                       "are the layouts the recipe carries — a device brings "
                       "its own, so a GAA offers 'fin' and 'gate'."),
        ParamSpec("dose", "Dose", "float", 1.0, 0.2, 4.0, 0.05,
                  group="Step", target="view", stage="view",
                  help="Relative exposure dose. Dose-to-size is where the "
                       "printed CD meets the drawn one; above it a dark-tone "
                       "line thins, below it the line fattens and eventually "
                       "bridges to its neighbour."),
        _nm("focus", "Defocus", 0.0, -200.0, 200.0, 5.0,
            "Focal offset. Away from best focus the aerial image contrast "
            "falls and the printed CD drifts — the other half of the "
            "process window dose defines."),
        _nm("overlay_dx", "Overlay x", 0.0, -20.0, 20.0, 0.5,
            "Misalignment of this exposure across the design grid. The only "
            "place overlay enters, which is why pitch walking emerges on its "
            "own rather than being modelled."),
        _nm("overlay_dy", "Overlay y", 0.0, -20.0, 20.0, 0.5,
            "Misalignment of this exposure along the design grid."),
        ParamSpec("tone", "Tone", "choice", "clear",
                  choices=("clear", "dark"),
                  group="Step", target="view", stage="view",
                  help="Mask polarity. 'dark' prints the drawn shape as "
                       "surviving resist; 'clear' prints it as an opening."),
        ParamSpec("standing_waves", "Standing waves", "bool", False,
                  group="Step", target="view", stage="view",
                  help="Include the vertical interference from substrate "
                       "reflection, which ripples the sidewall. A post-"
                       "exposure bake is what washes it out again."),
    ),
    "peb": (
        _nm("diffusion_sigma", "Diffusion σ", 20.0, 0.0, 100.0, 1.0,
            "How far the latent image diffuses during the bake. Enough of it "
            "smooths away standing waves; too much erases the image."),
    ),
    "develop": (
        ParamSpec("model", "Model", "choice", "threshold",
                  choices=("threshold", "front", "mack"),
                  group="Step", target="view", stage="view",
                  help="'threshold' dissolves wherever the latent falls below "
                       "the clearing dose — fast, and enough when only the "
                       "footprint matters. 'front' advances the same rate law "
                       "as a moving front, and is the only one that can "
                       "undercut, so the only one that makes a T-top or a "
                       "foot. 'mack' develops each column downward at the "
                       "local rate until the time budget runs out."),
        _material("material", "Material", "photoresist",
                  "The film the developed profile is written into."),
    ),
}

#: The order the palette offers them in, roughly the order a flow uses them.
#: ``pattern`` stands in for the coat/expose/develop triplet, so it sits where
#: litho sits: after the films that prepare the surface, immediately before the
#: etch that consumes the mask it prints.
PALETTE: tuple[str, ...] = (
    "spincoat", "deposit", "pattern", "etch", "etchback", "cmp", "strip",
)


#: Fields naming a material the step must find *already on the wafer*, and how
#: to pick it. A static default is a coin flip for these: the palette's etch
#: defaulted to SOC, which is not on a bare wafer, so adding an etch and
#: running it did nothing at all — silently, because a no-op etch is not an
#: error.
#:
#: Two different questions, so two different rules. A strip or an etch-back
#: names the film it is there to remove, which is whatever covers most of the
#: surface. An etch is aimed *down through an opening*, so it names what is at
#: the bottom of one — after a ``pattern`` the mask is usually the larger share
#: of the surface, and "most of the surface" would propose etching the mask
#: away instead of etching through it.
_WANTS_MATERIAL: dict[str, tuple[str, str]] = {
    "etch": ("targets", "floor"),
    "strip": ("material", "mode"),
    "etchback": ("material", "mode"),
}


def _default_of(kind: str, key: str) -> Any:
    """One spec's default, so a fallback elsewhere cannot drift away from it.

    ``stop_rate`` in particular has to stay below :data:`~litho_sim.wafer.stack.
    _STOP_RATE`; a second copy of the number in :func:`build_step` is exactly
    how that guarantee gets quietly lost.
    """
    return next(s for s in STEP_SPECS[kind] if s.key == key).default


def defaults_for(kind: str, stack=None) -> dict[str, Any]:
    """Display-unit defaults for every editable field of *kind*.

    With a *stack*, the defaults are chosen for **that wafer** rather than in
    the abstract: a step that consumes an existing material starts on whatever
    is currently exposed, so adding it and pressing Run does something visible.
    """
    if kind not in STEP_SPECS:
        raise KeyError(f"No editor for step '{kind}'. Known: {sorted(STEP_SPECS)}")
    values = {s.key: s.default for s in STEP_SPECS[kind]}

    entry = _WANTS_MATERIAL.get(kind)
    if stack is not None and entry is not None:
        field, rule = entry
        chosen = (open_floor_material_name(stack) if rule == "floor"
                  else top_material_name(stack))
        if chosen is not None:
            values[field] = chosen
    return values


def top_material_name(stack) -> str | None:
    """The material covering most of the wafer's surface, or None if bare.

    The mode rather than any single column, so a mostly-oxide surface with a
    few resist lines on it still proposes the oxide.
    """
    import numpy as np

    from litho_sim.wafer import VACUUM, get_material

    top = stack.top_material()
    ids, counts = np.unique(top[top != VACUUM], return_counts=True)
    if ids.size == 0:
        return None
    return get_material(int(ids[counts.argmax()])).name


def open_floor_material_name(stack) -> str | None:
    """The exposed material at the bottom of the openings, or None if bare.

    What an etch is aimed at. On an unpatterned wafer there is only one exposed
    material and this agrees with :func:`top_material_name`; once a mask is on
    the wafer they disagree, and this is the one that means "etch *through* the
    opening" rather than "etch the mask away".

    Lowest **mean** surface height per material, not lowest single column: one
    stray pinhole in a mask should not decide what the etch is for.
    """
    import numpy as np

    from litho_sim.wafer import VACUUM, get_material

    top = stack.top_material()
    height = stack.top_height()
    ids = np.unique(top[top != VACUUM])
    if ids.size == 0:
        return None
    lowest = min(ids, key=lambda i: float(height[top == i].mean()))
    return get_material(int(lowest)).name


def build_step(kind: str, values: dict[str, Any]) -> ProcessStep:
    """Assemble a step from display-unit values.

    Applies each spec's ``scale`` (nanometre sliders, metre engine) and folds
    the etch's stop-material pair back into a ``selectivity`` dict.
    """
    if kind not in STEP_SPECS:
        raise KeyError(f"No editor for step '{kind}'. Known: {sorted(STEP_SPECS)}")

    payload: dict[str, Any] = {"kind": kind}
    for spec in STEP_SPECS[kind]:
        if spec.key in _SYNTHETIC.get(kind, ()):
            continue                       # handled below; not a step field
        value = values.get(spec.key, spec.default)
        payload[spec.key] = spec.clamp(value) if spec.kind != "choice" else value
        if spec.kind in ("float", "int"):
            payload[spec.key] = payload[spec.key] * spec.scale

    if kind == "deposit":
        # "—" is the widget's way of saying `None`: grow off everything. The
        # engine wants the absence, not the glyph.
        if payload.get("on") == NO_SEED:
            payload["on"] = None
    elif kind == "etch":
        stop = values.get("stop_on", NO_STOP)
        payload["selectivity"] = (
            {} if stop == NO_STOP
            else {stop: float(values.get("stop_rate", _default_of("etch", "stop_rate")))}
        )
    elif kind == "cmp":
        # Exactly one of the three, chosen by the mode; the others must be None
        # or `planarize` refuses them as ambiguous.
        mode = values.get("cmp_mode", "depth")
        payload["depth"] = payload["height"] = payload["stop_on"] = None
        if mode == "depth":
            payload["depth"] = float(values.get("depth", 30.0)) * 1e-9
        elif mode == "height":
            payload["height"] = float(values.get("height", 100.0)) * 1e-9
        else:
            payload["stop_on"] = values.get("stop_on", _MATERIALS[0])

    return ProcessStep.from_dict(payload)


def values_for(step: ProcessStep) -> dict[str, Any]:
    """The inverse of :func:`build_step` — display-unit values from a step.

    Round-tripping matters because the form is repopulated from the selected
    step every time the selection moves; if this disagreed with
    :func:`build_step`, merely clicking a step would silently edit it.
    """
    kind = step.kind
    if kind not in STEP_SPECS:
        raise KeyError(f"No editor for step '{kind}'")

    out: dict[str, Any] = {}
    for spec in STEP_SPECS[kind]:
        if spec.key in _SYNTHETIC.get(kind, ()):
            continue
        raw = getattr(step, spec.key)
        if spec.kind in ("float", "int") and raw is not None:
            raw = raw / spec.scale
        out[spec.key] = spec.default if raw is None else raw

    if kind == "etch":
        sel = getattr(step, "selectivity", {}) or {}
        if sel:
            name, rate = next(iter(sel.items()))
            out["stop_on"], out["stop_rate"] = name, rate
        else:
            out["stop_on"] = NO_STOP
            out["stop_rate"] = _default_of("etch", "stop_rate")
    elif kind == "cmp":
        stop_on = getattr(step, "stop_on", None)
        height = getattr(step, "height", None)
        depth = getattr(step, "depth", None)
        if stop_on is not None:
            out.update(cmp_mode="stop", stop_on=stop_on, depth=30.0, height=100.0)
        elif height is not None:
            out.update(cmp_mode="height", height=height * 1e9,
                       depth=30.0, stop_on=_MATERIALS[0])
        else:
            out.update(cmp_mode="depth", stop_on=_MATERIALS[0], height=100.0,
                       depth=(depth or 30e-9) * 1e9)

    return out


def unspecced_kinds() -> tuple[str, ...]:
    """Registered steps with no editor — informational, not an error."""
    return tuple(sorted(set(STEP_REGISTRY) - set(STEP_SPECS)))


# ---------------------------------------------------------------------------
# Editing a step that already exists
# ---------------------------------------------------------------------------
#
# `build_step` assembles a step from a form's whole value dict, which is right
# when the form *is* the step's only source — adding one from the palette. It
# is wrong for editing a recipe that came from somewhere else, because it
# rebuilds every field from what the widgets can say, and the widgets cannot
# say everything. Measured against the two device presets, rebuilding damaged
# 10 of 57 steps: a five-row selectivity table collapsed to one row, an
# etch-back's explicit 5 nm became "measure it", and a 254 nm ILD was clamped
# to the slider's 200 nm ceiling.
#
# So an edit is applied as a *replacement of one field*. Whatever the form
# never showed is never written, and therefore cannot be lost.


def uneditable_keys(step: ProcessStep) -> tuple[str, ...]:
    """Spec keys whose widget cannot faithfully hold *this* step's value.

    A dropdown of one material cannot express ``["Si", "SiGe"]``, and a single
    stop-material-plus-rate pair cannot express a five-row selectivity table.
    Rather than show a control that would quietly narrow the value the moment
    it is touched, those render read-only — visible, honest, and inert.
    """
    kind = step.kind
    out: list = []
    if kind == "etch":
        if not isinstance(getattr(step, "targets", ""), str):
            out.append("targets")
        if len(getattr(step, "selectivity", None) or {}) > 1:
            out += ["stop_on", "stop_rate"]
    elif kind == "deposit":
        on = getattr(step, "on", None)
        if on is not None and not isinstance(on, str):
            out.append("on")
    return tuple(out)


def specs_for_step(step: ProcessStep,
                   layouts: dict[str, Any] | None = None
                   ) -> tuple[ParamSpec, ...]:
    """This kind's specs, fitted to *this* step.

    Two adjustments, both because a static spec table cannot know what it will
    be asked to show:

    * **Bounds are widened** where the step's own value falls outside them. A
      slider that cannot reach the value it is displaying is not a control, it
      is a trap — the GAA's 254 nm ILD sits past ``deposit.thickness``'s 200 nm
      ceiling, so touching that step used to thin the film by 54 nm. Stretched
      per step rather than raised globally, because the default range is what
      makes the common case easy to aim.
    * **The layout choice comes from the recipe**, since the valid names are
      whatever the flow carries. A device brings its own: a GAA offers ``fin``
      and ``gate``, and offering ``main`` instead would name a layout that
      would raise on the next run.
    """
    specs = STEP_SPECS.get(step.kind, ())
    out = []
    for spec in specs:
        if spec.key == "layout" and layouts:
            names = tuple(sorted(layouts))
            current = getattr(step, "layout", None)
            if current is not None and current not in names:
                names = names + (current,)
            spec = dataclasses.replace(spec, choices=names)
        elif (spec.kind in ("float", "int")
                and spec.key not in _SYNTHETIC.get(step.kind, ())):
            raw = getattr(step, spec.key, None)
            if raw is not None:
                shown = raw / spec.scale
                if shown < spec.lo or shown > spec.hi:
                    spec = dataclasses.replace(
                        spec, lo=min(spec.lo, shown), hi=max(spec.hi, shown)
                    )
        out.append(spec)
    return tuple(out)


def apply_edit(step: ProcessStep, key: str, values: dict[str, Any]) -> ProcessStep:
    """*step* with the one field the user just changed applied to it.

    The inverse of nothing — deliberately. It writes exactly one field (or one
    synthetic group), leaving every other field of the step as it was, so the
    parts of a recipe the editor cannot express survive being edited around.
    """
    kind = step.kind
    spec = next((s for s in specs_for_step(step) if s.key == key), None)
    if spec is None:
        raise KeyError(f"'{key}' is not an editable field of a '{kind}' step")
    if key in uneditable_keys(step):
        raise ValueError(
            f"'{key}' on this step holds a value its editor cannot represent; "
            f"it is shown read-only"
        )

    if key in _SYNTHETIC.get(kind, ()):
        return _apply_synthetic(step, key, values)

    value = values.get(key, spec.default)
    if spec.kind != "choice":
        value = spec.clamp(value)
    if spec.kind in ("float", "int"):
        value = value * spec.scale
    elif kind == "deposit" and key == "on" and value == NO_SEED:
        value = None            # the widget's way of saying "off everything"
    return dataclasses.replace(step, **{key: value})


def _apply_synthetic(step: ProcessStep, key: str, values: dict[str, Any]) -> ProcessStep:
    """The two fields that are a group of widgets rather than one.

    Reached only when the group *can* express the step — a multi-row
    selectivity table is read-only, so replacing the whole dict here is not
    lossy, it is the entire content.
    """
    kind = step.kind
    if kind == "etch":
        stop = values.get("stop_on", NO_STOP)
        if stop == NO_STOP:
            return dataclasses.replace(cast(Etch, step), selectivity={})
        rate = float(values.get("stop_rate", _default_of("etch", "stop_rate")))
        return dataclasses.replace(cast(Etch, step), selectivity={stop: rate})
    if kind == "cmp":
        # Exactly one of the three, or `planarize` refuses them as ambiguous.
        mode = values.get("cmp_mode", "depth")
        out: dict[str, Any] = {"depth": None, "height": None, "stop_on": None}
        if mode == "depth":
            out["depth"] = float(values.get("depth", 30.0)) * 1e-9
        elif mode == "height":
            out["height"] = float(values.get("height", 100.0)) * 1e-9
        else:
            out["stop_on"] = values.get("stop_on", _MATERIALS[0])
        return dataclasses.replace(step, **out)
    raise KeyError(f"no synthetic group '{key}' on a '{kind}' step")


# ---------------------------------------------------------------------------
# Reading a step back out, in full
# ---------------------------------------------------------------------------
#
# `values_for` answers "what should the sliders show", and is bounded by what
# the sliders can express: one etch target, one row of the selectivity table,
# nothing at all for a kind with no editor. That is the right answer for a
# form and the wrong one for the question "what settings built this device",
# where a value the widgets cannot show is exactly the value worth showing.
#
# So this reads the *dataclass*, via `to_dict`, and formats whatever it finds.
# No field can go missing, because none is enumerated by hand.

#: Fields carrying a length in metres. The engine is SI throughout and the
#: field name is the only signal — `depth` is metres, `z_frac` is a fraction,
#: and nothing in the type says so. Cross-checked by a test against every
#: ``scale=1e-9`` spec in :data:`STEP_SPECS`, so the two cannot drift.
_METRE_FIELDS = frozenset({
    "thickness", "depth", "height", "pitch", "cd", "bias",
    "top_radius", "bottom_radius", "footer_height", "footer_extent",
    "diffusion_sigma", "overlay_dx", "overlay_dy", "focus",
})

#: Names worth spelling out, for fields with no spec to borrow a label from.
_READOUT_LABELS = {
    "cd": "CD",
    "diffusion_sigma": "Diffusion σ",
    "focus": "Defocus",
    "line_of_sight": "Line of sight",
    "on": "Nucleates on",
    "open_layout": "Through layout",
    "open_tone": "Opening tone",
    "optics_overrides": "Optics overrides",
    "overlay_dx": "Overlay x",
    "overlay_dy": "Overlay y",
    "sidewall_deg": "Sidewall angle",
    "standing_waves": "Standing waves",
    "z_frac": "Height fraction",
}


def _readout_label(kind: str, key: str) -> str:
    for spec in STEP_SPECS.get(kind, ()):
        if spec.key == key:
            return spec.label
    if key in _READOUT_LABELS:
        return _READOUT_LABELS[key]
    return key.replace("_", " ").capitalize()


def _format_number(value: float) -> str:
    """Trim a float to something readable without lying about it."""
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{value:.4g}"


def format_field(key: str, value: Any) -> str:
    """One field, formatted the way the readout formats it.

    Public so the editor can render a read-only row the same way the as-built
    readout does — two spellings of the same number is how a panel starts
    disagreeing with itself.
    """
    return _format_value(key, value)


def _format_value(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        if not value:
            return "—"
        # A selectivity table, almost always: material and its relative rate.
        return ", ".join(f"{k} {_format_number(float(v))}"
                         if isinstance(v, (int, float)) and not isinstance(v, bool)
                         else f"{k} {v}"
                         for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "—"
    if isinstance(value, (int, float)):
        if key in _METRE_FIELDS:
            return f"{_format_number(value * 1e9)} nm"
        if key == "sidewall_deg":
            return f"{_format_number(float(value))}°"
        return _format_number(float(value))
    return str(value)


def readout_for(step: ProcessStep) -> list[tuple[str, str, bool]]:
    """Every setting on *step*, as ``(label, value, is_default)`` rows.

    Built from :meth:`~litho_sim.patterning.steps.ProcessStep.to_dict`, so it
    covers kinds the palette has no editor for — ``expose`` and ``develop``
    among them, which is most of what makes a printed device printed.

    The third element says whether the field is still at its dataclass
    default, so a display can mute those. They are reported either way: a
    default is a setting the step *ran with*, and leaving it out would turn
    "these are the settings" into "these are the interesting settings".
    """
    values = step.to_dict()
    values.pop("kind", None)
    defaults = {
        f.name: (f.default_factory() if f.default_factory is not MISSING  # type: ignore[misc]
                 else f.default)
        for f in fields(step)
    }
    return [
        (_readout_label(step.kind, key),
         _format_value(key, value),
         value == defaults.get(key))
        for key, value in values.items()
    ]
