"""Patterning: single prints compose into technologies.

A flow is an ordered list of steps applied to a
:class:`~litho_sim.wafer.stack.Stack`.  Multi-patterning schemes are orderings
of the same primitives, not separate implementations:

>>> from litho_sim.patterning import lele, sadp
>>> flow = sadp(mandrel_layout, grid, optics, resist, spacer_thickness=20e-9)
>>> result = flow.run()
>>> result.measurements

Recipe families each get a module: ``multipatterning`` (single exposure,
LELE/LE³, SADP/SAQP) today; ``cuts`` (line-end formation), ``damascene`` and
``selfaligned`` as the roadmap lands them.
"""

from litho_sim.patterning.cuts import (
    add_cut_mask,
    block_resist,
    cut_block,
    ideal_cut,
    keep_block,
)
from litho_sim.patterning.flow import Flow, RunResult
from litho_sim.patterning.multipatterning import (
    lele,
    sadp,
    saqp,
    single_exposure,
)
from litho_sim.patterning.steps import (
    CMP,
    STEP_REGISTRY,
    Deposit,
    Develop,
    Etch,
    Expose,
    Measure,
    Pattern,
    PostExposureBake,
    ProcessContext,
    ProcessStep,
    SpacerEtchback,
    SpinCoat,
    Strip,
    register_step,
)

__all__ = [
    "Flow", "RunResult",
    "ProcessStep", "ProcessContext", "STEP_REGISTRY", "register_step",
    "SpinCoat", "Deposit", "Pattern", "Expose", "PostExposureBake", "Develop",
    "Etch", "SpacerEtchback", "Strip", "CMP", "Measure",
    "single_exposure", "lele", "sadp", "saqp",
    "add_cut_mask", "block_resist", "cut_block", "ideal_cut", "keep_block",
]
