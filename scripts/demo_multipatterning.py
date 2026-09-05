"""3-D printing, overlay-driven pitch walking, and SADP.

The command lives in ``litho_sim.cli.multipatterning``; this script is the same as
``litho-sim multipatterning``, kept so the documented invocation still works.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.cli import main

if __name__ == "__main__":
    main(["multipatterning", *sys.argv[1:]])
