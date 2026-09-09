"""Optical proximity correction: moving mask edges until the wafer prints the design.

``fragments`` is the geometry — polygon edges cut into movable pieces and
rebuilt from their offsets; ``model`` is the print model (the engine's own
imaging and resist path) and the edge placement error measured against it;
``correct`` is the model-based iteration and the one-shot ``verify``;
``assist`` places sub-resolution scattering bars by rule. A corrected
layout is an ordinary :class:`~litho_sim.mask.layout.Layout`, so it goes
wherever a drawn one does.

``ilt`` is the other kind of correction: the mask as a free field rather
than a moved polygon, solved by gradient descent through the adjoint of
the imaging. It answers with curves and with assist features nobody
placed, and it answers with an *array* — the one thing here that is not a
``Layout``, because it never was one.
"""

from litho_sim.opc.assist import add_scattering_bars, assist_features_printed
from litho_sim.opc.correct import OPCResult, default_fragment_length, run_opc, verify
from litho_sim.opc.fragments import (
    Fragment,
    FragmentedShape,
    bias_layout,
    fragment_layout,
    fragment_shape,
    rebuild_layout,
)
from litho_sim.opc.ilt import (
    ILTFrame,
    ILTResult,
    dose_for_target,
    print_mask,
    run_ilt,
)
from litho_sim.opc.model import EPE, PrintedImage, PrintModel, measure_epe

__all__ = [
    "Fragment", "FragmentedShape", "bias_layout", "fragment_layout", "fragment_shape",
    "rebuild_layout",
    "EPE", "PrintedImage", "PrintModel", "measure_epe",
    "OPCResult", "default_fragment_length", "run_opc", "verify",
    "add_scattering_bars", "assist_features_printed",
    "ILTFrame", "ILTResult", "dose_for_target", "print_mask", "run_ilt",
]
