#!/usr/bin/env python3
"""
wafer-metrology-engine -- pipeline entry point.

Runs synthesise -> Zernike -> interferometry -> flatness -> defects -> DOE,
writes every figure and table to ``results/`` and prints the metric tables.

Usage
-----
    python run.py                      # full run, 512 px grid
    python run.py --quick              # fast smoke run
    python run.py --n-pixels 768 -v    # finer grid, debug logging
    python run.py --no-doe --no-ml     # figures and flatness only
    python run.py --help

Works without installing the package: ``src`` is put on the path below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wafer_metrology.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
