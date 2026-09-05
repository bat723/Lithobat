"""Flags shared between commands, so every command spells them the same way."""

from __future__ import annotations

import argparse
from pathlib import Path

from litho_sim.core.config import RESIST_LIBRARY, TECH_NODE_PRESETS

NODES = tuple(TECH_NODE_PRESETS)
RESISTS = tuple(RESIST_LIBRARY)

#: The resist a node runs by default: a conventional resist at i-line, the
#: chemically amplified formulation at DUV, the thin EUV formulation at EUV.
#: The demo used to print every node in the i-line/DUV "Generic Positive"
#: film, which at EUV is a 100 nm bleaching resist that no EUV tool exposes.
RESIST_FOR_NODE = {
    "i-line": "Generic Positive",
    "KrF": "CAR (Positive)",
    "ArF": "CAR (Positive)",
    "ArF_immersion": "CAR (Positive)",
    "EUV": "EUV CAR (Positive)",
}


def resist_for(args: argparse.Namespace) -> str:
    """The resist preset a command should use: ``--resist`` if given, else the node's."""
    chosen = getattr(args, "resist", None)
    return chosen if chosen else RESIST_FOR_NODE[args.node]


def add_node_arg(p: argparse.ArgumentParser, default: str = "ArF") -> None:
    p.add_argument("--node", default=default, choices=NODES,
                   help="technology-node preset")
    p.add_argument("--resist", default=None, choices=RESISTS,
                   help="resist preset (default: the node's usual one — a conventional "
                        "resist at i-line, the CAR at DUV, the EUV CAR at EUV)")


def add_output_args(p: argparse.ArgumentParser, default: str | Path = "results") -> None:
    p.add_argument("--output", type=Path, default=Path(default),
                   help="output directory (or file, for single-figure commands)")
    p.add_argument("--no-show", action="store_true", help="don't open a plot window")


def add_litho_args(p: argparse.ArgumentParser) -> None:
    """Node, pattern, threshold and output: the flags every sweep command takes."""
    add_node_arg(p)
    p.add_argument("--pitch", type=float, default=200.0, help="Pattern pitch [nm]")
    p.add_argument("--cd", type=float, default=100.0, help="Target CD [nm]")
    p.add_argument("--threshold", type=float, default=None, metavar="T",
                   help="Resist clearing threshold, 0–1 (default: the resist "
                        "preset's own value). This is the CD calibration knob: "
                        "a higher threshold prints a wider feature. The demo "
                        "image is peak-normalised; the bossung/window sweeps "
                        "normalise to the clear field instead, so a dose means "
                        "the same energy at every focus — the same threshold "
                        "therefore prints differently in the two commands.")
    add_output_args(p)


def add_sweep_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-doses", type=int, default=5, help="Number of dose levels")
    p.add_argument("--n-defoci", type=int, default=11, help="Number of defocus steps")
    p.add_argument("--defocus-range", type=float, default=300.0,
                   help="Half-range of defocus sweep [nm]")
