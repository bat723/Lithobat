#!/usr/bin/env python3
"""
Script 1: Generate synthetic aerial-image dataset (clean + defective).

Usage
-----
    python scripts/generate_data.py [--n-clean 200] [--n-defect 200]
                                    [--node ArF] [--output data/raw_synthetic]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.core.config import SimulationConfig
from litho_sim.core.utils import setup_logging
from litho_sim.expose.aerial_image import compute_aerial_image
from litho_sim.mask.patterns import lines_and_spaces
from litho_sim.ml.defects import add_bridge_defect, add_line_roughness, add_particle_defect

try:
    from skimage.io import imsave
    _HAS_SKIMAGE = True
except ImportError:
    import matplotlib.pyplot as plt
    _HAS_SKIMAGE = False


def _save_image(arr: np.ndarray, path: Path) -> None:
    """Save a float [0,1] image as 8-bit PNG."""
    img_8bit = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if _HAS_SKIMAGE:
        imsave(str(path), img_8bit, check_contrast=False)
    else:
        plt.imsave(str(path), img_8bit, cmap="gray", vmin=0, vmax=255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic wafer image dataset.")
    p.add_argument("--n-clean", type=int, default=200, help="Number of clean images")
    p.add_argument("--n-defect", type=int, default=200, help="Number of defective images")
    p.add_argument("--node", default="ArF", help="Technology node preset")
    p.add_argument("--pitch", type=float, default=200.0, help="Pattern pitch [nm]")
    p.add_argument("--cd", type=float, default=100.0, help="Feature CD [nm]")
    p.add_argument("--output", type=Path, default=Path("data/raw_synthetic"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    rng = np.random.default_rng(args.seed)
    clean_dir = args.output / "clean"
    defect_dir = args.output / "defect"
    clean_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimulationConfig.from_tech_node(args.node, name="data_gen")
    mask = lines_and_spaces(
        cfg.grid.n_pixels,
        cfg.grid.pixel_size,
        pitch=args.pitch * 1e-9,
        cd=args.cd * 1e-9,
    )

    # ------------------------------------------------------------------
    # Clean images (slight focus/dose variation for realistic diversity)
    # ------------------------------------------------------------------
    logger.info("Generating %d clean images…", args.n_clean)
    for i in tqdm(range(args.n_clean), desc="Clean"):
        # Small random focus / dose jitter
        import copy
        local_optics = copy.replace(
            cfg.optics,
            defocus=float(rng.uniform(-20e-9, 20e-9)),
        )
        dose = float(rng.uniform(0.95, 1.05))
        aerial = compute_aerial_image(mask, local_optics, cfg.grid, dose=dose)
        _save_image(aerial, clean_dir / f"{i:04d}.png")

    # ------------------------------------------------------------------
    # Defective images
    # ------------------------------------------------------------------
    logger.info("Generating %d defective images…", args.n_defect)
    defect_funcs = [
        lambda img, r=rng: add_particle_defect(
            img, particle_radius_px=float(r.integers(4, 10)), seed=int(r.integers(1e6))),
        lambda img, r=rng: add_line_roughness(
            img, roughness_amplitude=float(r.uniform(0.08, 0.18)), seed=int(r.integers(1e6))),
        lambda img, r=rng: add_bridge_defect(img, seed=int(r.integers(1e6))),
    ]

    for i in tqdm(range(args.n_defect), desc="Defect"):
        aerial = compute_aerial_image(mask, cfg.optics, cfg.grid, dose=1.0)
        defect_fn = defect_funcs[i % len(defect_funcs)]
        aerial_defect = defect_fn(aerial)
        _save_image(aerial_defect, defect_dir / f"{i:04d}.png")

    logger.info(
        "Done. Clean: %d images → %s | Defect: %d images → %s",
        args.n_clean, clean_dir,
        args.n_defect, defect_dir,
    )


if __name__ == "__main__":
    main()

