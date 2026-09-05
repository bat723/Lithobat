"""Flags shared between commands, so every command spells them the same way."""

from __future__ import annotations

import argparse
from pathlib import Path

from litho_sim.core.config import TECH_NODE_PRESETS

NODES = tuple(TECH_NODE_PRESETS)


def add_node_arg(p: argparse.ArgumentParser, default: str = "ArF") -> None:
    p.add_argument("--node", default=default, choices=NODES,
                   help="technology-node preset")


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
