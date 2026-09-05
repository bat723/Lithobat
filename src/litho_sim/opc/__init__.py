"""Optical proximity correction: moving mask edges until the wafer prints the design.

``fragments`` is the geometry — polygon edges cut into movable pieces and
rebuilt from their offsets; ``model`` is the print model (the engine's own
imaging and resist path) and the edge placement error measured against it;
``correct`` is the model-based iteration and the one-shot ``verify``;
``assist`` places sub-resolution scattering bars by rule. A corrected
layout is an ordinary :class:`~litho_sim.mask.layout.Layout`, so it goes
wherever a drawn one does.
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
from litho_sim.opc.model import EPE, PrintedImage, PrintModel, measure_epe

__all__ = [
    "Fragment", "FragmentedShape", "bias_layout", "fragment_layout", "fragment_shape",
    "rebuild_layout",
    "EPE", "PrintedImage", "PrintModel", "measure_epe",
    "OPCResult", "default_fragment_length", "run_opc", "verify",
    "add_scattering_bars", "assist_features_printed",
]
