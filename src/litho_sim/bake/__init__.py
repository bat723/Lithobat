"""Post-exposure bake: what happens to the latent image between exposure and
development.

Two models of the same step:

* ``peb`` — a single Gaussian blur, the linear single-species approximation.
  What washes out standing waves, and all a conventional resist needs.
* ``reaction`` — coupled acid/quencher reaction–diffusion with catalytic
  deprotection, the chemically-amplified-resist kinetics. The quencher is
  what turns the bake from a blur into a threshold; the develop model
  ``"car"`` runs on this.
"""

from litho_sim.bake.peb import apply_peb, apply_peb_3d
from litho_sim.bake.reaction import bake_reaction_diffusion, diffusion_length

__all__ = [
    "apply_peb",
    "apply_peb_3d",
    "bake_reaction_diffusion",
    "diffusion_length",
]
