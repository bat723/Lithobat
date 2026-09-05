"""
LithoPy command-line interface — ``litho-sim`` / ``python -m litho_sim``.

One parser, one subcommand per thing the engine can do from a terminal:

=================  ==========================================================
``demo``           one simulation, straight to plots and CSV
``bossung``        CD-through-focus curves across dose
``window``         the full process window: EL, DOF, best focus and dose
``opc``            model-based OPC on a layout with every classic problem
``multipatterning``  LELE pitch walking, SADP, and SADP with a cut mask
``vector``         the polarisation effect at hyper-NA, law and images
``device``         build a printed device (``gaa`` or ``nfet``) and verify it
``stochastic``     print the same exposure many times: LER, LWR, LCDU, failures,
                   and with ``--profile`` the roughness through the film
=================  ==========================================================

Every command lives in its own module as an ``add_parser``/``run`` pair and
is imported only when the parser is built, so ``--help`` stays instant. The
``scripts/demo_*.py`` files are these same commands, kept as shims so the
documented invocations still work.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """The complete parser, with every command registered."""
    from litho_sim.cli import basic, device, multipatterning, opc, stochastic, vector

    parser = argparse.ArgumentParser(
        prog="litho-sim",
        description="LithoPy – Photolithography Simulation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    subs = parser.add_subparsers(dest="command", metavar="COMMAND")
    subs.required = True
    for module in (basic, opc, multipatterning, vector, device, stochastic):
        module.add_parser(subs)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse *argv* (``sys.argv[1:]`` by default), run the command, exit on failure."""
    from litho_sim.core.utils import setup_logging

    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    from litho_sim.viz.plots import apply_style

    apply_style()
    status = args.func(args)
    if status:
        sys.exit(status)


__all__ = ["build_parser", "main"]
