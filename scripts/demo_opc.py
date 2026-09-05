"""Optical proximity correction on a layout that has every classic problem.

The command lives in ``litho_sim.cli.opc``; this script is the same as
``litho-sim opc``, kept so the documented invocation still works.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.cli import main

if __name__ == "__main__":
    main(["opc", *sys.argv[1:]])
