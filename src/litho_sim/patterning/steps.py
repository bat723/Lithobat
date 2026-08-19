"""
Process steps — the operations a recipe is built from.

Every step takes a :class:`~litho_sim.wafer.stack.Stack` and mutates it. The
design rule that makes multi-patterning fall out of a small set of
primitives is this:

    **The etch mask is emergent, never a parameter.**

:class:`Etch` looks at whatever material is currently on top of each
column. Patterned resist masks a column by being there; so does a
hardmask, so does a spacer. Nothing has to be told "this is the mask".
That single decision is why LELE, SADP, SAQP, and cut masks are all just
different orderings of the same nine steps rather than four bespoke
implementations.

Overlay error enters in exactly one place — :class:`Expose` — and it is
applied to the analytic geometry before rasterisation, so a 1 nm
misalignment on a 4 nm grid is represented rather than rounded away.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from litho_sim.bake.peb import apply_peb_3d
from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.develop.resist3d import (
    apply_absorption,
    apply_vertical_interference,
    develop_3d,
    exposure_volume,
    write_resist_to_stack,
)
from litho_sim.mask.layout import Layout, contact_grid, line_array
from litho_sim.wafer import Stack, get_material

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ProcessContext:
    """Shared state threaded through a flow.

    Attributes
    ----------
    grid : GridConfig
        Lateral and vertical simulation grid.
    optics : OpticsConfig
        Default optical settings; individual :class:`Expose` steps may
        override fields.
    resist : ResistConfig
        Default resist chemistry.
    layouts : dict
        ``{name: Layout}`` that :class:`Expose` steps refer to by name.
    state : dict
        Scratch space carrying the in-flight latent image between
        :class:`Expose`, :class:`PostExposureBake`, and :class:`Develop`.
    measurements : list
        Rows appended by :class:`Measure`.
    """

    grid: GridConfig = field(default_factory=GridConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    resist: ResistConfig = field(default_factory=ResistConfig)
    layouts: dict[str, Layout] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    measurements: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base class and registry
# ---------------------------------------------------------------------------

STEP_REGISTRY: dict[str, type[ProcessStep]] = {}


def register_step(kind: str) -> Callable[[type[ProcessStep]], type[ProcessStep]]:
    """Class decorator registering a step under a name for (de)serialisation."""

    def wrap(cls: type[ProcessStep]) -> type[ProcessStep]:
        cls.kind = kind
        STEP_REGISTRY[kind] = cls
        return cls

    return wrap


@dataclass
class ProcessStep:
    """Base class for a single wafer operation."""

    kind: str = "step"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        raise NotImplementedError

    def describe(self) -> str:
        return self.kind

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        d = asdict(self)
        d["kind"] = self.kind
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ProcessStep:
        data = dict(d)
        kind = data.pop("kind")
        if kind not in STEP_REGISTRY:
            raise ValueError(
                f"Unknown process step '{kind}'. Known: {sorted(STEP_REGISTRY)}"
            )
        cls = STEP_REGISTRY[kind]
        data.pop("kind", None)
        return cls(**data)


# ---------------------------------------------------------------------------
# Film steps
# ---------------------------------------------------------------------------


@register_step("spincoat")
@dataclass
class SpinCoat(ProcessStep):
    """Spin on a film.

    Spin-on films planarize by default, which matters: on the second
    exposure of a LELE flow the resist is spun over the topography left by
    the first etch, so it is thicker in the trenches than over the lines.
    """

    material: str = "photoresist"
    thickness: float = 90e-9
    planarize: bool = True
    kind: str = "spincoat"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        stack.deposit_blanket(self.material, self.thickness, planarize=self.planarize)
        return stack

    def describe(self) -> str:
        return f"Coat {self.material} {self.thickness*1e9:.0f} nm"


@register_step("deposit")
@dataclass
class Deposit(ProcessStep):
    """Deposit a film, conformally or as a blanket.

    Conformal deposition is what makes self-aligned patterning possible: the
    film grows off vertical sidewalls as well as horizontal surfaces, so
    removing the mandrel later leaves the sidewall film standing alone.
    """

    material: str = "spacer-oxide"
    thickness: float = 20e-9
    conformal: bool = True
    #: Fill the topography and come out flat. Blanket films only — a conformal
    #: film that planarized would not be conformal.
    planarize: bool = False
    #: Grow only off exposed surfaces of these materials — selective epitaxy,
    #: where source/drain silicon nucleates on silicon and nowhere else. A
    #: material *name*, or a list of them, rather than a `Material`, so the
    #: step still survives ``to_dict`` the way ``Expose.layout`` names its
    #: layout. Polymorphic for the same reason ``Etch.targets`` is: a salicide
    #: caps exposed silicon *and* poly in one step.
    on: str | Sequence[str] | None = None
    kind: str = "deposit"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        if self.conformal:
            stack.deposit_conformal(self.material, self.thickness, on=self.on)
        else:
            stack.deposit_blanket(
                self.material, self.thickness,
                planarize=self.planarize, on=self.on,
            )
        return stack

    def describe(self) -> str:
        how = ["conformal" if self.conformal else "blanket"]
        if self.planarize and not self.conformal:
            how.append("planarised")
        if self.on:
            seeds = self.on if isinstance(self.on, str) else ", ".join(self.on)
            how.append(f"on {seeds}")
        return (
            f"Deposit {self.material} {self.thickness*1e9:.0f} nm "
            f"({', '.join(how)})"
        )


@register_step("pattern")
@dataclass
class Pattern(ProcessStep):
    """Print an ideal resist mask — litho without the optics.

    Stands in for the coat/expose/PEB/develop triplet when the point is the
    *mask*, not the printing. Everything downstream of a mask edge — the etch
    profile's bias, wall angle and corner radii, line-of-sight shadowing,
    point-of-contact access, anisotropic undercut — needs one to act on, and on
    a blanket film there is nothing to act on: a blanket etch has no sidewall.

    A step and not an import, unlike a developed profile: that is a volume and
    could not survive :meth:`to_dict`, whereas this geometry is four scalars.

    The line and contact *count* is deliberately not a field. It is derived
    from ``ctx.grid`` at apply time so the array tiles whatever field the
    recipe is run on, and the recipe itself stays resolution-independent.
    """

    material: str = "photoresist"
    thickness: float = 90e-9
    shape: str = "lines"              # "lines" | "contacts"
    pitch: float = 128e-9
    cd: float = 64e-9
    orientation: str = "vertical"     # lines only
    planarize: bool = True
    kind: str = "pattern"

    def mask(self, grid: GridConfig) -> NDArray[np.float64]:
        """Open-area coverage on *grid*: 1 where the film is to be removed."""
        field = grid.n_pixels * grid.pixel_size
        # Span (n-1)*pitch >= field, so the array overhangs both edges and no
        # margin is left unpatterned at a pitch the field is not a multiple of.
        n = max(int(np.ceil(field / self.pitch)) + 1, 1)
        if self.shape == "lines":
            # Twice the field, so the line *ends* fall outside the window. An
            # end inside it is a mask edge like any other, and the etch's
            # signed-distance field would dutifully carve a wall there.
            shapes = line_array(n, self.pitch, self.cd, field * 2.0,
                                orientation=self.orientation)
        elif self.shape == "contacts":
            shapes = contact_grid(n, n, self.pitch, self.pitch, self.cd)
        else:
            raise ValueError(
                f"shape must be 'lines' or 'contacts', got '{self.shape}'"
            )
        return Layout(shapes, name="pattern").rasterize(grid, tone="clear")

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        mid = get_material(self.material).id
        surf_before = stack.surface_index()
        stack.deposit_blanket(self.material, self.thickness,
                              planarize=self.planarize)

        # Which voxels are *this* film. Both halves are load-bearing. Not "the
        # top N voxels": a planarising coat is thicker in a trench than over a
        # line, so there is no single N. And not "every voxel of this
        # material": the wafer may already carry a buried layer of it, and
        # carving by material alone would destroy it. What was vacuum before is
        # exact. `zz` is built after the deposit because `ensure_headroom` may
        # have grown `nz`; indices are bottom-referenced, so `surf_before`
        # survives that, and an empty column's -1 marks the whole column fresh.
        zz = np.arange(stack.nz)[:, None, None]
        fresh = (stack.mat == mid) & (zz > surf_before[None])

        open_col = self.mask(ctx.grid) > 0.5
        stack.clear(fresh & open_col[None])
        stack.history.append(self.describe())
        if not open_col.any() or open_col.all():
            stack.history.append(
                "pattern: the film came out unbroken, so there is no mask edge "
                "for a later etch to shape — check the CD against the pitch"
            )
        return stack

    def describe(self) -> str:
        what = "lines" if self.shape == "lines" else "contacts"
        return (f"Pattern {self.material} {what} {self.cd*1e9:.0f}/"
                f"{self.pitch*1e9:.0f} nm ({self.thickness*1e9:.0f} nm film)")


# ---------------------------------------------------------------------------
# Litho steps
# ---------------------------------------------------------------------------


@register_step("expose")
@dataclass
class Expose(ProcessStep):
    """Project a layout into the resist film.

    Parameters
    ----------
    layout : str
        Key into :attr:`ProcessContext.layouts`.
    dose : float
        Relative exposure dose.
    focus : float
        Defocus [m].
    overlay_dx, overlay_dy : float
        Misalignment of this exposure relative to the design grid [m].
        Applied analytically to the geometry before rasterisation.

        This is the only place overlay enters, and it is what makes pitch
        walking emerge on its own: in a LELE flow the second exposure lands
        displaced while the first did not, so the spaces between them
        alternate wide/narrow by twice the offset.
    tone : str
        ``"clear"`` or ``"dark"`` mask polarity.
    standing_waves : bool
        Include substrate-reflection standing waves.
    optics_overrides : dict
        Partial :class:`~litho_sim.core.config.OpticsConfig` fields for this
        exposure only — e.g. a different illumination for the mandrel layer.
    """

    layout: str = "main"
    dose: float = 1.0
    focus: float = 0.0
    overlay_dx: float = 0.0
    overlay_dy: float = 0.0
    tone: str = "clear"
    standing_waves: bool = False
    optics_overrides: dict[str, Any] = field(default_factory=dict)
    kind: str = "expose"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        if self.layout not in ctx.layouts:
            raise KeyError(
                f"Expose references layout '{self.layout}', which is not in the "
                f"recipe. Available: {sorted(ctx.layouts)}"
            )
        film = _resist_film_thickness(stack, ctx)
        resist_cfg = replace(ctx.resist, thickness=film)
        optics = replace(ctx.optics, defocus=self.focus, **self.optics_overrides)

        mask = ctx.layouts[self.layout].rasterize(
            ctx.grid, tone=self.tone, dx=self.overlay_dx, dy=self.overlay_dy
        )
        intensity, _ = exposure_volume(mask, optics, ctx.grid, resist_cfg, dose=self.dose)
        if self.standing_waves:
            intensity = apply_vertical_interference(
                intensity, resist_cfg, optics, ctx.grid
            )
        # exposure_volume already applied self.dose to the intensity; applying
        # it to the absorption as well would double-count (dose² sweeps).
        pac, _ = apply_absorption(intensity, resist_cfg, ctx.grid.dz, dose=1.0)

        ctx.state["pac"] = pac
        ctx.state["resist_cfg"] = resist_cfg
        ctx.state["mask"] = mask
        stack.history.append(
            f"expose {self.layout} dose={self.dose:.2f} "
            f"overlay=({self.overlay_dx*1e9:.1f}, {self.overlay_dy*1e9:.1f}) nm"
        )
        return stack

    def describe(self) -> str:
        ov = ""
        if self.overlay_dx or self.overlay_dy:
            ov = f", overlay ({self.overlay_dx*1e9:+.1f}, {self.overlay_dy*1e9:+.1f}) nm"
        return f"Expose {self.layout} @ dose {self.dose:.2f}{ov}"


@register_step("peb")
@dataclass
class PostExposureBake(ProcessStep):
    """Diffuse the latent image. Also what washes out standing waves."""

    diffusion_sigma: float | None = None
    kind: str = "peb"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        pac = _require_latent(ctx, "PostExposureBake")
        cfg = ctx.state["resist_cfg"]
        if self.diffusion_sigma is not None:
            cfg = replace(cfg, diffusion_sigma=self.diffusion_sigma)
            ctx.state["resist_cfg"] = cfg
        ctx.state["pac"] = apply_peb_3d(pac, cfg, ctx.grid)
        stack.history.append("PEB")
        return stack

    def describe(self) -> str:
        if self.diffusion_sigma is None:
            return "Post-exposure bake"
        return f"Post-exposure bake (σ={self.diffusion_sigma*1e9:.0f} nm)"


@register_step("develop")
@dataclass
class Develop(ProcessStep):
    """Dissolve the exposed resist and write the profile into the stack."""

    model: str = "threshold"
    material: str = "photoresist"
    kind: str = "develop"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        pac = _require_latent(ctx, "Develop")
        cfg = ctx.state["resist_cfg"]
        remaining = develop_3d(pac, cfg, ctx.grid, model=self.model)
        write_resist_to_stack(stack, remaining, material=self.material)
        ctx.state["remaining"] = remaining
        ctx.state.pop("pac", None)
        return stack

    def describe(self) -> str:
        return f"Develop ({self.model})"


# ---------------------------------------------------------------------------
# Etch steps
# ---------------------------------------------------------------------------


@register_step("etch")
@dataclass
class Etch(ProcessStep):
    """Etch downward wherever *targets* is the exposed surface material.

    No mask argument by default: whatever is on top of a column masks it
    (the emergent-mask principle). *open_layout* adds an optional extra
    gate for ideal (litho-free) cuts and blocks — the etch then acts only
    where the drawn geometry allows, ANDed with the emergent mask.

    Parameters
    ----------
    targets : str or sequence of str
        Material(s) that may be attacked where they are the exposed surface.
    depth : float
        Etch budget [m] in unit-rate-material terms (each material advances
        at its ``etch_rate``).
    anisotropy : float
        1.0 = perfectly directional; below 1 adds isotropic undercut.
    selectivity : dict
        Per-material rate overrides for this etch.
    open_layout : str, optional
        Key into :attr:`ProcessContext.layouts`. When set, the layout is
        rasterised at apply time and passed to :meth:`Stack.etch` as its
        ``open_mask`` — a JSON-safe reference, never an array on the step.
    open_tone : str
        ``"clear"`` — etch only *inside* the drawn shapes (an ideal cut);
        ``"dark"`` — etch only *outside* them (an ideal block/keep).
    """

    targets: Any = "SOC"
    depth: float = 60e-9
    anisotropy: float = 1.0
    selectivity: dict[str, float] = field(default_factory=dict)
    open_layout: str | None = None
    open_tone: str = "clear"
    # Where the etchant can reach. Both off is the column-wise approximation
    # the engine has always used.
    line_of_sight: bool = False
    contact: bool = False
    #: ``"surface"`` opens a column only where *targets* is the material on
    #: top; ``"any"`` attacks the target on sidewalls too, which is what a
    #: channel release, an inner spacer or a source/drain undercut needs and
    #: what a top-down model cannot express at all.
    exposure: str = "surface"
    #: Recess the whole surface regardless of what each column exposes — the
    #: one thing the emergent-mask rule cannot express.
    blanket: bool = False
    # Profile shape. All zero with a 90 degree wall is a vertical cut on the
    # mask edge, which is what this did before the fields existed.
    bias: float = 0.0
    sidewall_deg: float = 90.0
    top_radius: float = 0.0
    bottom_radius: float = 0.0
    footer_height: float = 0.0
    footer_extent: float = 0.0
    kind: str = "etch"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        open_mask = None
        if self.open_layout is not None:
            if self.open_layout not in ctx.layouts:
                raise KeyError(
                    f"Etch references layout '{self.open_layout}', which is not in "
                    f"the recipe. Available: {sorted(ctx.layouts)}"
                )
            open_mask = (
                ctx.layouts[self.open_layout].rasterize(ctx.grid, tone=self.open_tone)
                > 0.5
            )
        from litho_sim.wafer.etch_profile import EtchProfile

        profile = EtchProfile(
            bias=self.bias, sidewall_deg=self.sidewall_deg,
            top_radius=self.top_radius, bottom_radius=self.bottom_radius,
            footer_height=self.footer_height, footer_extent=self.footer_extent,
        )
        stack.etch(
            self.targets, depth=self.depth, anisotropy=self.anisotropy,
            selectivity=self.selectivity or None, open_mask=open_mask,
            profile=None if profile.is_identity else profile,
            line_of_sight=self.line_of_sight, contact=self.contact,
            blanket=self.blanket, exposure=self.exposure,
        )
        return stack

    def describe(self) -> str:
        from litho_sim.wafer.etch_profile import EtchProfile

        t = "everything" if self.blanket else (
            self.targets if isinstance(self.targets, str) else ", ".join(self.targets)
        )
        through = f" through {self.open_layout}" if self.open_layout else ""
        # Only the *slow* entries are stops. A selectivity table names the
        # target at full rate as often as it names what arrests the etch —
        # `{"Si": 1.0, "SiGe": 1.0}` stops on nothing — and reporting the
        # first key regardless turned "etch silicon" into "stop on silicon".
        stops = [n for n, rate in (self.selectivity or {}).items() if rate < 1.0]
        if not stops:
            stop = ""
        elif len(stops) <= 3:
            stop = f", stop on {', '.join(stops)}"
        else:
            stop = f", stop on {', '.join(stops[:3])} +{len(stops) - 3} more"
        access = [n for n, on in (("LOS", self.line_of_sight),
                                  ("contact", self.contact),
                                  ("lateral", self.exposure == "any")) if on]
        acc = f" [{'+'.join(access)}]" if access else ""
        shape = EtchProfile(
            bias=self.bias, sidewall_deg=self.sidewall_deg,
            top_radius=self.top_radius, bottom_radius=self.bottom_radius,
            footer_height=self.footer_height, footer_extent=self.footer_extent,
        ).describe()
        shape = f" ({shape})" if shape else ""
        verb = "Blanket etch" if self.blanket else "Etch"
        return f"{verb} {t} {self.depth*1e9:.0f} nm{through}{stop}{acc}{shape}"


@register_step("etchback")
@dataclass
class SpacerEtchback(ProcessStep):
    """Directional etch that clears horizontal film but spares sidewalls."""

    material: str = "spacer-oxide"
    thickness: float | None = None
    overetch: float = 0.2
    kind: str = "etchback"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        stack.etch_back(self.material, thickness=self.thickness, overetch=self.overetch)
        return stack

    def describe(self) -> str:
        return f"Spacer etch-back {self.material}"


@register_step("strip")
@dataclass
class Strip(ProcessStep):
    """Remove a material entirely — a wet strip or ash.

    Under the name "mandrel pull" this is the step that leaves self-aligned
    spacers freestanding and doubles the line count.
    """

    material: str = "photoresist"
    kind: str = "strip"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        stack.strip(self.material)
        return stack

    def describe(self) -> str:
        return f"Strip {self.material}"


@register_step("cmp")
@dataclass
class CMP(ProcessStep):
    """Chemical-mechanical planarisation to a flat plane.

    Three ways to say where the plane goes, at most one at a time — see
    :meth:`~litho_sim.wafer.stack.Stack.planarize`. With none of them it takes
    the surface down to the lowest peak, the least it can remove and still come
    out flat.
    """

    height: float | None = None
    #: Remove this much, measured down from the current high point.
    depth: float | None = None
    #: Polish until the pad reaches this material and stop.
    stop_on: str | None = None
    kind: str = "cmp"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        stack.planarize(
            height=self.height, depth=self.depth, stop_on=self.stop_on
        )
        return stack

    def describe(self) -> str:
        if self.stop_on is not None:
            return f"CMP stopping on {self.stop_on}"
        if self.depth is not None:
            return f"CMP {self.depth*1e9:.0f} nm down"
        if self.height is not None:
            return f"CMP to {self.height*1e9:.0f} nm"
        return "CMP to the lowest peak"


# ---------------------------------------------------------------------------
# Metrology
# ---------------------------------------------------------------------------


@register_step("measure")
@dataclass
class Measure(ProcessStep):
    """Record line and space widths without changing the wafer.

    Space widths are usually the interesting output: alternating spaces are
    the signature of overlay-induced pitch walking.
    """

    name: str = "CD"
    material: str | None = None
    z_frac: float = 0.5
    kind: str = "measure"

    def apply(self, stack: Stack, ctx: ProcessContext) -> Stack:
        feats = stack.measure_features(self.material, self.z_frac)
        lines = [w * 1e9 for w in feats["lines"]]
        spaces = [w * 1e9 for w in feats["spaces"]]
        walk_lines = _pitch_walk(lines)
        walk_spaces = _pitch_walk(spaces)
        row = {
            "name": self.name,
            "material": self.material or "any",
            "n_lines": len(lines),
            "lines_nm": lines,
            "spaces_nm": spaces,
            "mean_line_nm": float(np.mean(lines)) if lines else float("nan"),
            "mean_space_nm": float(np.mean(spaces)) if spaces else float("nan"),
            "line_walk_nm": walk_lines,
            "space_walk_nm": walk_spaces,
            # Whichever alternates more is the signature. Which one it lands
            # in depends on mask tone: displacing a trench keeps the trench CD
            # and alternates the lines between trenches, and vice versa.
            "pitch_walk_nm": max(walk_lines, walk_spaces),
        }
        ctx.measurements.append(row)
        logger.info(
            "Measure '%s': %d lines, mean CD %.1f nm, pitch walk %.1f nm",
            self.name, len(lines), row["mean_line_nm"], row["pitch_walk_nm"],
        )
        return stack

    def describe(self) -> str:
        return f"Measure '{self.name}'"


def _pitch_walk(spaces: Sequence[float]) -> float:
    """Alternation amplitude of a space sequence [nm].

    Zero for uniform spaces; grows as adjacent spaces diverge. This is the
    number that reveals overlay error in a LELE flow or mandrel CD error in
    SADP, and it needs at least two spaces to mean anything.
    """
    if len(spaces) < 2:
        return 0.0
    even = np.mean(spaces[0::2])
    odd = np.mean(spaces[1::2])
    return float(abs(even - odd))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_latent(ctx: ProcessContext, who: str) -> NDArray[np.float64]:
    if "pac" not in ctx.state:
        raise RuntimeError(
            f"{who} needs a latent image, but no Expose step has run since the "
            "last Develop. Check the step order in your recipe."
        )
    return ctx.state["pac"]


def _resist_film_thickness(stack: Stack, ctx: ProcessContext) -> float:
    """Thickness of the thickest resist column currently on the wafer [m].

    Using the maximum matters when resist is spun over topography: the film
    is deeper in the trenches, and the exposure volume has to span the
    deepest column or the bottom of the trench never gets simulated.
    """
    t = stack.thickness_of("photoresist")
    thick = float(t.max()) if t.size else 0.0
    if thick <= 0.0:
        raise RuntimeError(
            "Expose ran with no photoresist on the wafer. Add a SpinCoat step first."
        )
    return thick
