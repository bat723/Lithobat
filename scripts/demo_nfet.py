"""A planar bulk nFET from two printed masks.

The flow lives in ``litho_sim.tech.nfet``; this script is the same command
as ``litho-sim device nfet``, kept so the documented invocation still works.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.cli import main

if __name__ == "__main__":
    main(["device", "nfet", *sys.argv[1:]])
