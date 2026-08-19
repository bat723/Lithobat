"""
Utility helpers for the LithoPy simulation engine.

Provides:
* Logging setup with a consistent format.
* The 2-D spatial-frequency grid builder the imaging path is built on.
* JSON file persistence for anything that speaks ``to_dict``/``from_dict``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to write formatted output to stdout.

    Parameters
    ----------
    level : int
        Verbosity level, e.g. ``logging.DEBUG`` or ``logging.INFO``.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


def make_freq_grid(
    n_pixels: int,
    pixel_size: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a 2-D spatial-frequency meshgrid in FFT ordering.

    Parameters
    ----------
    n_pixels : int
        Number of pixels per axis.
    pixel_size : float
        Physical pixel size [m].

    Returns
    -------
    fx, fy : NDArray
        Frequency arrays in m⁻¹, shape ``(n_pixels, n_pixels)``, in
        standard NumPy FFT ordering (DC at [0, 0]).
    """
    freqs = np.fft.fftfreq(n_pixels, d=pixel_size)
    fx, fy = np.meshgrid(freqs, freqs)
    return fx.astype(np.float64), fy.astype(np.float64)


class JsonFileMixin:
    """JSON file persistence for classes that speak ``to_dict``/``from_dict``.

    The file format *is* the dict representation, pretty-printed; the mixin
    adds only the path handling, so every serialisable object — a layout, a
    flow, a source — reads and writes its files the same way.
    """

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("%s saved → %s", type(self).__name__, path)

    @classmethod
    def from_json(cls, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{cls.__name__} file not found: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

