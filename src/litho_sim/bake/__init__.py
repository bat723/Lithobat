"""Post-exposure bake: acid diffusion smoothing the latent image.

The step between exposure and development — and the reason standing waves
don't print. ``peb`` holds the 2-D and 3-D Gaussian diffusion models;
reaction–diffusion (acid/quencher, CAR kinetics) lands here when the
chemically-amplified-resist work does.
"""

from litho_sim.bake.peb import apply_peb, apply_peb_3d

__all__ = ["apply_peb", "apply_peb_3d"]
