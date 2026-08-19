"""
Line-end formation: cut, block, and keep.

Three semantics, all orderings of the existing steps (no new step kinds —
the emergent-mask principle does the work):

**Cut** — drawn shapes mark segments to *remove* from existing lines.
Litho route: ``tone="clear"`` (drawn = bright → exposed → develops away),
so the resist opens exactly at the drawn shapes; the following
``Etch(targets=<line material>)`` severs the lines there. Each fully
crossed line becomes two line-ends; pitch is untouched.

**Block** — drawn shapes are *protected* during a subsequent etch.
Litho route: ``tone="dark"`` (drawn = opaque → resist survives on the
shapes). A block carries no etch of its own — the resist bars simply join
the emergent mask of whatever etch the caller runs next, and the caller
strips afterwards.

**Keep** — only drawn regions survive: block litho + blanket etch + strip.
(This is what the pre-slice ``_cut_block`` actually did while its docstring
claimed to cut — tone="dark" protects the drawn bars, it does not open
them.)

Every builder also has an *ideal* (litho-free) counterpart via
``Etch(open_layout=..., open_tone=...)``, which gates the etch with the
drawn geometry directly — no resist, no dose, no diffusion. Use it to
separate "does the flow work" from "does the cut print".

Tone table
----------
============  ==============  =====================================
semantics     Expose tone     effect on the drawn shapes
============  ==============  =====================================
cut           ``"clear"``     opened → etched away
block / keep  ``"dark"``      covered → protected from the etch
============  ==============  =====================================
"""

from __future__ import annotations

import logging

from litho_sim.mask.layout import Layout
from litho_sim.patterning.flow import Flow
from litho_sim.patterning.stackspec import RESIST, TARGET
from litho_sim.patterning.steps import (
    Develop,
    Etch,
    Expose,
    Measure,
    PostExposureBake,
    SpinCoat,
    Strip,
)

logger = logging.getLogger(__name__)


def cut_block(
    layout_key: str,
    *,
    dose: float = 1.0,
    target: str = TARGET,
    depth: float = 60e-9,
) -> list:
    """Steps that sever *target* lines where the cut layout is drawn.

    Coat → expose (``tone="clear"``) → bake → develop opens the resist at
    the drawn shapes; the etch then attacks only columns whose exposed top
    is *target* — line segments inside the windows. Field areas inside a
    window whose top is not *target* are untouched (emergent mask).

    Parameters
    ----------
    layout_key : str
        Key into the flow's layouts naming the cut shapes.
    dose : float
        Relative exposure dose for the cut print.
    target : str
        The line material being cut.
    depth : float
        Etch budget [m]; the default clears a 40 nm poly-Si film with
        margin (rate 0.8 → 48 nm of removal).
    """
    return [
        SpinCoat(material=RESIST, thickness=90e-9, planarize=True),
        Expose(layout=layout_key, dose=dose, tone="clear"),
        PostExposureBake(),
        Develop(),
        Etch(targets=target, depth=depth, selectivity={RESIST: 0.1}),
        Strip(material=RESIST),
    ]


def block_resist(layout_key: str, *, dose: float = 1.0) -> list:
    """Steps that leave resist standing on the drawn shapes, nothing more.

    Composes with whatever etch the caller runs next — the resist joins the
    emergent mask — and the caller strips the resist afterwards.
    """
    return [
        SpinCoat(material=RESIST, thickness=90e-9, planarize=True),
        Expose(layout=layout_key, dose=dose, tone="dark"),
        PostExposureBake(),
        Develop(),
    ]


def keep_block(
    layout_key: str,
    *,
    dose: float = 1.0,
    target: str = TARGET,
    depth: float = 60e-9,
) -> list:
    """Steps after which only the drawn regions of *target* survive.

    Block litho + blanket etch + strip: the resist bars protect the drawn
    shapes while the etch clears every exposed *target* column.
    """
    return [
        *block_resist(layout_key, dose=dose),
        Etch(targets=target, depth=depth, selectivity={RESIST: 0.1}),
        Strip(material=RESIST),
    ]


def ideal_cut(
    layout_key: str,
    *,
    target: str = TARGET,
    depth: float = 60e-9,
) -> list:
    """A single geometry-gated etch: the cut without the lithography."""
    return [
        Etch(targets=target, depth=depth, open_layout=layout_key, open_tone="clear"),
    ]


def add_cut_mask(
    flow: Flow,
    cut_layout: Layout,
    *,
    dose: float = 1.0,
    style: str = "litho",
    target: str = TARGET,
    depth: float = 60e-9,
) -> Flow:
    """Append a cut to any finished flow, before its trailing measurements.

    Cut masks are how one drawn line becomes several gates. They compose
    with every scheme here because they are just another litho + etch pair
    — provided the lines are actually exposed when the cut runs, which is
    why this inserts at the very end of the flow (after any spacer strip),
    displacing only the trailing ``Measure`` steps.

    Parameters
    ----------
    style : str
        ``"litho"`` — full coat/expose/develop/etch/strip cut print;
        ``"ideal"`` — one geometry-gated etch (no lithography).
    """
    if style not in ("litho", "ideal"):
        raise ValueError(f"style must be 'litho' or 'ideal', got '{style}'")
    flow.layouts["cut"] = cut_layout
    # Insert before the final Measure so the measurement sees the cut result.
    tail = []
    while flow.steps and isinstance(flow.steps[-1], Measure):
        tail.insert(0, flow.steps.pop())
    if style == "litho":
        flow.add(*cut_block("cut", dose=dose, target=target, depth=depth))
    else:
        flow.add(*ideal_cut("cut", target=target, depth=depth))
    flow.add(*tail)
    return flow
