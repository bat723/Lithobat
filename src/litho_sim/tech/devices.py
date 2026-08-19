"""Named device presets — build one, or load it from the cache.

The two device flows live in ``scripts/`` (``demo_gaa.py``, ``demo_nfet.py``)
because each is a runnable, documented artefact in its own right. This module
is the thin adapter that lets the rest of the package ask for one *by name* —
so the app's File ▸ Load device menu, ``scripts/show_device.py`` and anything
later all go through one definition of "the GAA preset" rather than three.

Building a device costs 2–7 s, so the result is cached to ``presets/`` as a
compressed ``.npz``. The cache is keyed on the preset name; delete the file to
force a rebuild after changing a flow.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from litho_sim.patterning.flow import Flow
from litho_sim.patterning.steps import ProcessStep
from litho_sim.wafer import Stack

logger = logging.getLogger(__name__)

#: Bumped when the flow cache's meaning changes. Version 1 stored steps only,
#: which was enough to show a recipe and not to re-run one — a cache at that
#: version is treated as absent so the device rebuilds with its context.
FLOW_CACHE_VERSION = 2

#: Repo root, from ``src/litho_sim/tech/devices.py``.
_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = _ROOT / "scripts"
PRESET_DIR = _ROOT / "presets"


class DeviceSpec:
    """One named device: how to build it, and how to section it for viewing."""

    def __init__(self, key: str, title: str, module: str, builder: str,
                 section: dict, blurb: str):
        self.key = key
        self.title = title
        self.module = module
        self.builder = builder
        self.section = section
        self.blurb = blurb


DEVICES: dict[str, DeviceSpec] = {
    "gaa": DeviceSpec(
        "gaa", "GAA nanosheet", "demo_gaa", "build_gaa",
        dict(y=(0.45, 1.0), x=(0.15, 0.85), z_substrate=6e-9),
        "Gate-all-around nanosheet FET — EUV, two printed masks",
    ),
    "nfet": DeviceSpec(
        "nfet", "Planar nFET", "demo_nfet", "build_nfet",
        dict(y=(0.42, 1.0), z_substrate=10e-9),
        "Planar bulk nFET — 193 nm immersion, two printed masks",
    ),
}


class DeviceFlow:
    """The recipe that built a device, and the wafer after each step of it.

    A :class:`~litho_sim.wafer.stack.Stack` remembers what happened to it as
    prose — ``etch Si, SiGe 62 nm (aniso=1.00)`` — which is enough to caption a
    picture and not enough to say what actually ran: the selectivity table, the
    exposure mode and the etch profile are all gone by the time the sentence is
    written. The steps are kept here instead, as the
    :class:`~litho_sim.patterning.steps.ProcessStep` objects the flow applied,
    so the recipe on screen is the recipe that ran rather than a paraphrase.

    The steps alone are enough to *show* the recipe and not enough to **re-run**
    it. A step names its layout — ``Expose(layout="fin")`` — and looks it up in
    a :class:`~litho_sim.patterning.steps.ProcessContext` that also carries the
    optics and the resist. Without those, editing a device step and replaying
    would expose the GAA at the library's default 193 nm instead of 13.5 nm
    EUV, and the layout lookup would raise. So the context travels with the
    recipe, as a :class:`~litho_sim.patterning.flow.Flow` — which already
    serialises exactly these four things.

    Attributes
    ----------
    flow : Flow
        The recipe *and* its context: steps, layouts, grid, optics, resist.
    snapshots : list of Stack
        ``snapshots[i]`` is the wafer after ``flow.steps[i]``. One per step.
    base : Stack
        The bare wafer the flow started from — the scrubber's first position.
    changed : list of bool
        Did step *i* alter the wafer? A step is allowed to do nothing, but it
        should not do nothing silently.
    label : str
        The device's printed-CD label, e.g. ``GAA nanosheet — printed fin 98 nm``.
    """

    def __init__(self, flow, snapshots, base, changed, label: str):
        self.flow: Flow = flow
        self.snapshots: list[Stack] = list(snapshots)
        self.base: Stack = base
        self.changed: list[bool] = list(changed)
        self.label = label

    @property
    def steps(self) -> list[ProcessStep]:
        return self.flow.steps

    @property
    def layouts(self):
        return self.flow.layouts

    @property
    def optics(self):
        return self.flow.optics

    @property
    def resist(self):
        return self.flow.resist

    @property
    def grid(self):
        return self.flow.grid

    def __len__(self) -> int:
        return len(self.flow.steps)


def flow_path(name: str, preset_dir: Path | None = None) -> Path:
    """Where the recipe sits — beside the stack, not inside it.

    A sibling file rather than more keys in ``<name>.npz`` so
    :meth:`~litho_sim.wafer.stack.Stack.save` keeps its format: an old preset
    still loads, and a missing sibling costs the recipe panel and nothing else.
    """
    return cache_path(name, preset_dir).with_suffix(".flow.npz")


def _save_flow(path: Path, flow: DeviceFlow) -> None:
    """Write the recipe and its snapshots to a compressed ``.npz``.

    The snapshots differ in height — the stack grows as deposition needs
    headroom — so each ``mat`` is its own array rather than one stacked block.
    Layered uint8 compresses hard: 36 snapshots of a GAA come to ~0.3 MB.

    The recipe goes in as ``Flow.to_dict()`` rather than a bare step list,
    because that already carries the layouts, grid, optics and resist an
    ``Expose`` needs to run again — and already re-casts the Zernike integer
    keys JSON would otherwise hand back as strings.
    """
    meta = {
        "version": FLOW_CACHE_VERSION,
        "label": flow.label,
        "flow": flow.flow.to_dict(),
        "changed": flow.changed,
        "history": [list(s.history) for s in flow.snapshots],
        "base_history": list(flow.base.history),
        "dz": flow.base.dz,
        "n_pixels": flow.base.grid.n_pixels,
        "pixel_size": flow.base.grid.pixel_size,
    }
    arrays = {f"mat_{i:03d}": s.mat for i, s in enumerate(flow.snapshots)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), mat_base=flow.base.mat,
                        **arrays)
    logger.info("device flow saved → %s (%.1f KB, %d steps)",
                path, path.stat().st_size / 1e3, len(flow))


def _load_flow(path: Path) -> DeviceFlow:
    """Read back what :func:`_save_flow` wrote.

    Raises ``ValueError`` on a cache written before the context was stored.
    A recipe that cannot be re-run is worse than none: it would look editable
    and then expose at the wrong wavelength.
    """
    from litho_sim.core.config import GridConfig

    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    if meta.get("version") != FLOW_CACHE_VERSION:
        raise ValueError(
            f"flow cache is version {meta.get('version')!r}, this build wants "
            f"{FLOW_CACHE_VERSION!r}"
        )
    grid = GridConfig(n_pixels=meta["n_pixels"], pixel_size=meta["pixel_size"])

    def stack_of(mat, history) -> Stack:
        return Stack(mat=mat, grid=grid, dz=meta["dz"], history=list(history))

    snapshots = [stack_of(data[f"mat_{i:03d}"], h)
                 for i, h in enumerate(meta["history"])]
    return DeviceFlow(
        flow=Flow.from_dict(meta["flow"]),
        snapshots=snapshots,
        base=stack_of(data["mat_base"], meta["base_history"]),
        changed=meta["changed"],
        label=meta["label"],
    )


def flow_for(name: str, preset_dir: Path | None = None) -> DeviceFlow | None:
    """The recipe for a cached device, or ``None`` if there isn't one.

    ``None`` rather than an exception: a preset written before recipes were
    recorded, or one whose sibling file was deleted, should still open and
    draw. The recipe panel is the thing that goes missing, not the device.
    """
    path = flow_path(name, preset_dir)
    if not path.is_file():
        return None
    try:
        return _load_flow(path)
    except (OSError, KeyError, ValueError):
        logger.warning("device %s: unreadable flow cache %s — ignoring",
                       name, path, exc_info=True)
        return None


def _load_builder(spec: DeviceSpec) -> Callable:
    """Import the flow module out of ``scripts/``.

    Not importable as a package — it is a directory of runnable scripts — so
    the path goes on ``sys.path`` once, the same way ``show_device.py`` does it.
    """
    if not SCRIPTS_DIR.is_dir():
        raise FileNotFoundError(
            f"device flows live in {SCRIPTS_DIR}, which is not there. The "
            f"presets need the repository checkout, not just the installed "
            f"package."
        )
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    module = __import__(spec.module)
    return getattr(module, spec.builder)


def cache_path(name: str, preset_dir: Path | None = None) -> Path:
    return (Path(preset_dir) if preset_dir else PRESET_DIR) / f"{name}.npz"


def build(name: str, use_cache: bool = True,
          preset_dir: Path | None = None) -> tuple[Stack, str]:
    """Return ``(stack, label)`` for a named device, building it if needed.

    The stack is the **whole** device, uncropped — sectioning is a viewing
    decision and belongs to the caller, which is why :attr:`DeviceSpec.section`
    is data rather than something applied here.
    """
    if name not in DEVICES:
        raise KeyError(f"unknown device {name!r}. Known: {sorted(DEVICES)}")
    spec = DEVICES[name]
    path = cache_path(name, preset_dir)

    if use_cache and path.is_file():
        # A cache hit needs *both* files current. The wafer alone would open a
        # device whose recipe panel is empty or, worse, whose recipe cannot be
        # re-run; rebuilding costs 2–7 s once and `presets/` is gitignored.
        #
        # The label lives in the flow cache too, so a cached load reports the
        # same printed CDs a fresh build does — the two paths used to disagree.
        flow = flow_for(name, preset_dir)
        if flow is not None:
            logger.info("device %s: loading cached preset %s", name, path)
            return Stack.load(path), flow.label
        logger.info("device %s: recipe cache missing or outdated — rebuilding",
                    name)

    builder = _load_builder(spec)
    # One snapshot per step, collected as the flow runs. `Stack.copy` because
    # steps mutate in place and return the same object: without the copy every
    # snapshot would be the finished device.
    steps: list[ProcessStep] = []
    snapshots: list[Stack] = []
    changed: list[bool] = []
    base: list[Stack] = []
    context: list = []

    def collect(step: ProcessStep | None, stack: Stack) -> None:
        if step is None:            # the bare wafer, before anything ran
            base.append(stack.copy())
            return
        previous = snapshots[-1] if snapshots else base[0]
        steps.append(step)
        changed.append(not np.array_equal(previous.mat, stack.mat))
        snapshots.append(stack.copy())

    stack, metrics = builder(verbose=False, on_step=collect,
                             on_context=context.append)
    printed = ", ".join(f"{k.replace('_printed_nm', '')} {v:.0f} nm"
                        for k, v in metrics.items() if k.endswith("_printed_nm"))
    label = f"{spec.title} — printed {printed}" if printed else spec.title
    try:
        stack.save(path)
        logger.info("device %s: cached to %s", name, path)
        if base and context:
            # Both, or neither: a recipe without its context would load looking
            # editable and then expose at the wrong wavelength.
            ctx = context[0]
            recipe = Flow(steps=steps, name=name, layouts=dict(ctx.layouts),
                          grid=ctx.grid, optics=ctx.optics, resist=ctx.resist)
            _save_flow(flow_path(name, preset_dir),
                       DeviceFlow(recipe, snapshots, base[0], changed, label))
    except OSError:  # pragma: no cover - read-only checkout, still usable
        logger.warning("device %s: could not write cache %s", name, path)
    return stack, label


def sectioned(name: str, stack: Stack) -> Stack:
    """Apply the preset's viewing section — the cut the still renders use."""
    from litho_sim.viz.viz3d import crop

    return crop(stack, **DEVICES[name].section)
