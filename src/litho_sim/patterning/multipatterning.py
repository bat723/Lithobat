"""
Prebuilt multi-patterning recipes.

Every scheme here is an ordering of the same primitives from
:mod:`litho_sim.patterning.steps`. None of them needs special-case code, and
the two effects people actually care about are *emergent* rather than
modelled:

**Pitch walking in LELE** comes from :class:`~litho_sim.patterning.steps.Expose`
carrying an overlay offset. The second exposure lands displaced while the
first did not, so the A–B and B–A spaces differ by twice the offset.

**Pitch division in SADP** comes from conformal deposition plus a
directional etch-back plus a strip. Two sidewalls survive per mandrel, so N
mandrels become 2N lines. SAQP is that block run twice, with the first
spacer serving as the second mandrel — which is the clearest evidence that
the abstraction is right.
"""

from __future__ import annotations

import logging

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask.layout import Layout, decompose, split_by_color
from litho_sim.patterning.flow import Flow
from litho_sim.patterning.steps import (
    Deposit,
    Develop,
    Etch,
    Expose,
    Measure,
    PostExposureBake,
    SpacerEtchback,
    SpinCoat,
    Strip,
)

logger = logging.getLogger(__name__)

from litho_sim.patterning.cuts import cut_block, ideal_cut  # noqa: E402
from litho_sim.patterning.stackspec import (  # noqa: E402
    HARDMASK,
    MANDREL,
    RESIST,
    SPACER1,
    SPACER2,
    TARGET,
)


def _cut_steps(style: str, dose: float) -> list:
    """Dispatch a cut insertion for the spacer recipes."""
    if style == "litho":
        return cut_block("cut", dose=dose)
    if style == "ideal":
        return ideal_cut("cut")
    raise ValueError(f"cut_style must be 'litho' or 'ideal', got '{style}'")


def _litho_block(
    layout: str,
    dose: float = 1.0,
    focus: float = 0.0,
    overlay: tuple[float, float] = (0.0, 0.0),
    resist_thickness: float = 90e-9,
    tone: str = "clear",
    develop_model: str = "threshold",
) -> list:
    """Coat → expose → bake → develop. The unit every litho step repeats."""
    return [
        SpinCoat(material=RESIST, thickness=resist_thickness, planarize=True),
        Expose(
            layout=layout, dose=dose, focus=focus,
            overlay_dx=overlay[0], overlay_dy=overlay[1], tone=tone,
        ),
        PostExposureBake(),
        Develop(model=develop_model),
    ]


def _target_film(thickness: float = 40e-9) -> list:
    """Lay down the film the whole flow is trying to pattern.

    Without this there is nothing under the hardmask to transfer into, and
    the final measurement finds an empty wafer.
    """
    return [Deposit(material=TARGET, thickness=thickness, conformal=False)]


def _pattern_transfer(
    hardmask_thickness: float, etch_target: bool = False, target_depth: float = 40e-9
) -> list:
    """Transfer the developed resist pattern into the hardmask, then strip."""
    steps = [
        Etch(targets=HARDMASK, depth=hardmask_thickness,
             selectivity={RESIST: 0.25, TARGET: 0.02}),
        Strip(material=RESIST),
    ]
    if etch_target:
        steps.append(
            Etch(targets=TARGET, depth=target_depth, selectivity={HARDMASK: 0.05})
        )
    return steps


def _flow_layouts(mandrel_layout: Layout, cut_layout: Layout | None) -> dict:
    """The layouts a spacer flow needs: the mandrel, and the cut if there is one."""
    layouts = {"mandrel": mandrel_layout}
    if cut_layout is not None:
        layouts["cut"] = cut_layout
    return layouts


def _mandrel_steps(mandrel_height: float, dose: float) -> tuple:
    """Print and etch the mandrel the spacers will form on."""
    return (
        *_target_film(),
        SpinCoat(material=MANDREL, thickness=mandrel_height, planarize=True),
        *_litho_block("mandrel", dose=dose),
        Etch(targets=MANDREL, depth=mandrel_height,
             selectivity={RESIST: 0.3, TARGET: 0.01}),
        Strip(material=RESIST),
        Measure(name="mandrel", material=MANDREL),
    )


def _spacer_generation(spacer: str, thickness: float, mandrel: str,
                       measure_name: str) -> tuple:
    """One pitch division: coat the mandrel, etch back, pull the mandrel.

    The three steps that move pitch control from lithography to deposition.
    In SAQP the first generation's spacer is the second generation's mandrel.
    """
    return (
        Deposit(material=spacer, thickness=thickness, conformal=True),
        SpacerEtchback(material=spacer, thickness=thickness, overetch=0.25),
        Strip(material=mandrel),
        Measure(name=measure_name, material=spacer),
    )


def _transfer_and_cut(flow: Flow, final_spacer: str,
                      cut_layout: Layout | None, cut_style: str,
                      dose: float) -> None:
    """Etch the pattern into the target, strip the spacer, cut, measure."""
    flow.add(
        Etch(targets=TARGET, depth=40e-9, selectivity={final_spacer: 0.02}),
        Strip(material=final_spacer),
    )
    # The cut must land AFTER the spacer strip: while the spacer still caps
    # the lines, an Etch(targets=TARGET) finds no exposed target and is a
    # silent no-op — which is exactly the bug this ordering fixes.
    if cut_layout is not None:
        flow.add(*_cut_steps(cut_style, dose))
    flow.add(
        Measure(name="final", material=TARGET),
    )


# ---------------------------------------------------------------------------
# Single exposure
# ---------------------------------------------------------------------------


def single_exposure(
    layout: Layout,
    grid: GridConfig,
    optics: OpticsConfig,
    resist: ResistConfig,
    dose: float = 1.0,
    focus: float = 0.0,
    target_thickness: float = 40e-9,
) -> Flow:
    """The baseline: one mask, one exposure, one etch."""
    flow = Flow(name="Single exposure", grid=grid, optics=optics, resist=resist,
                layouts={"main": layout})
    flow.add(
        *_target_film(target_thickness),
        SpinCoat(material=HARDMASK, thickness=60e-9),
        *_litho_block("main", dose=dose, focus=focus),
        *_pattern_transfer(60e-9, etch_target=True),
        Strip(material=HARDMASK),
        Measure(name="final", material=TARGET),
    )
    return flow


# ---------------------------------------------------------------------------
# Litho-Etch-Litho-Etch
# ---------------------------------------------------------------------------


def lele(
    layout: Layout,
    grid: GridConfig,
    optics: OpticsConfig,
    resist: ResistConfig,
    min_spacing: float = 90e-9,
    overlay: tuple[float, float] = (0.0, 0.0),
    dose: float = 1.0,
    dose_b: float | None = None,
    n_colors: int = 2,
) -> Flow:
    """Litho-Etch-Litho-Etch (LE², or LE³ with ``n_colors=3``).

    The layout is split by graph colouring so that no two features closer
    than *min_spacing* share an exposure.  Each colour is then printed and
    etched into the same hardmask in turn.

    Parameters
    ----------
    layout : Layout
        The full target layout, too dense for one exposure.
    min_spacing : float
        Single-exposure minimum space [m]; drives the decomposition.
    overlay : tuple
        Misalignment applied to every exposure after the first [m].
        **This is the pitch-walking knob.**
    dose : float
        Dose for the first exposure.
    dose_b : float, optional
        Dose for later exposures.  Differing from *dose* produces CD
        mismatch between colours — the second, independent source of pitch
        walking in a real LELE flow.
    n_colors : int
        2 for LELE, 3 for LE³.

    Raises
    ------
    ValueError
        If the layout cannot be coloured, with the offending feature pairs
        named.  An odd conflict cycle is a genuine design-rule violation,
        not a solver failure.
    """
    dec = decompose(layout.shapes, min_spacing=min_spacing, n_colors=n_colors)
    if not dec.ok:
        raise ValueError(
            f"Layout '{layout.name}' is not {n_colors}-colourable at "
            f"{min_spacing*1e9:.0f} nm spacing: {len(dec.conflicts)} conflicting "
            f"pair(s) {dec.conflicts[:5]}. Relax the spacing rule, move the "
            f"features, or use n_colors={n_colors + 1}."
        )
    colors = split_by_color(layout, dec)

    name = {2: "LELE (LE²)", 3: "LELELE (LE³)"}.get(n_colors, f"LE^{n_colors}")
    flow = Flow(name=name, grid=grid, optics=optics, resist=resist, layouts=colors)
    flow.add(
        *_target_film(),
        SpinCoat(material=HARDMASK, thickness=60e-9, planarize=True),
    )

    for i, key in enumerate(sorted(colors)):
        # The first exposure defines the grid; every later one carries the
        # misalignment relative to it.
        ov = (0.0, 0.0) if i == 0 else overlay
        d = dose if (i == 0 or dose_b is None) else dose_b
        flow.add(*_litho_block(key, dose=d, overlay=ov))
        flow.add(
            Etch(targets=HARDMASK, depth=60e-9,
                 selectivity={RESIST: 0.25, TARGET: 0.02}),
            Strip(material=RESIST),
        )

    flow.add(
        Etch(targets=TARGET, depth=40e-9, selectivity={HARDMASK: 0.05}),
        Strip(material=HARDMASK),
        Measure(name="final", material=TARGET),
    )
    return flow


# ---------------------------------------------------------------------------
# Self-aligned double / quadruple patterning
# ---------------------------------------------------------------------------


def sadp(
    mandrel_layout: Layout,
    grid: GridConfig,
    optics: OpticsConfig,
    resist: ResistConfig,
    spacer_thickness: float = 20e-9,
    mandrel_height: float = 80e-9,
    dose: float = 1.0,
    cut_layout: Layout | None = None,
    cut_style: str = "litho",
) -> Flow:
    """Self-Aligned Double Patterning: N mandrels become 2N lines.

    The mandrel is printed at a relaxed pitch the scanner can actually
    resolve; the final pitch is half of it, and is set by deposition
    thickness rather than by the optics.  That is the entire point — pitch
    control moves from lithography to deposition, which is far more precise.

    Parameters
    ----------
    mandrel_layout : Layout
        Mandrel lines, at twice the target pitch.
    spacer_thickness : float
        Conformal spacer thickness [m]; becomes the final line width.
    mandrel_height : float
        Mandrel film thickness [m].
    cut_layout : Layout, optional
        A cut mask applied after pattern transfer.

    Notes
    -----
    SADP has its own pitch-walking mechanism, independent of overlay: if the
    mandrel CD does not equal the space between mandrels, the two resulting
    spaces differ and alternate.  Sweep the mandrel dose to see it.
    """
    flow = Flow(name="SADP", grid=grid, optics=optics, resist=resist,
                layouts=_flow_layouts(mandrel_layout, cut_layout))
    flow.add(
        *_mandrel_steps(mandrel_height, dose),
        # The three steps that do the pitch division.
        *_spacer_generation(SPACER1, spacer_thickness, MANDREL, "spacers"),
    )
    _transfer_and_cut(flow, SPACER1, cut_layout, cut_style, dose)
    return flow


def saqp(
    mandrel_layout: Layout,
    grid: GridConfig,
    optics: OpticsConfig,
    resist: ResistConfig,
    spacer1_thickness: float = 24e-9,
    spacer2_thickness: float = 12e-9,
    mandrel_height: float = 80e-9,
    dose: float = 1.0,
    cut_layout: Layout | None = None,
    cut_style: str = "litho",
) -> Flow:
    """Self-Aligned Quadruple Patterning: N mandrels become 4N lines.

    Literally the SADP block twice.  The first spacer becomes the second
    mandrel, so the pitch halves again.  That this needs no new step types
    is the strongest evidence the step abstraction is the right one.
    """
    flow = Flow(name="SAQP", grid=grid, optics=optics, resist=resist,
                layouts=_flow_layouts(mandrel_layout, cut_layout))
    flow.add(
        *_mandrel_steps(mandrel_height, dose),
        # First pitch division: mandrel → spacer 1.
        *_spacer_generation(SPACER1, spacer1_thickness, MANDREL, "spacer1"),
        # Second pitch division: spacer 1 is now the mandrel.
        *_spacer_generation(SPACER2, spacer2_thickness, SPACER1, "spacer2"),
    )
    _transfer_and_cut(flow, SPACER2, cut_layout, cut_style, dose)
    return flow


# ---------------------------------------------------------------------------
# Cut / block masks
# ---------------------------------------------------------------------------


# The cut/block/keep builders and add_cut_mask live in
# litho_sim.patterning.cuts — one module per recipe family.
