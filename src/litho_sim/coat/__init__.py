"""Spin-coating: the film stack the light lands on.

``films`` is the transfer-matrix optics of the coated stack — reflectance,
swing curves, BARC design, and the standing-wave envelope inside the resist.
Spin/planarisation physics of its own (``coat/spin.py``) is reserved for
later; the ``SpinCoat`` process step lives in ``litho_sim.patterning``.
"""

from litho_sim.coat.films import (
    Film,
    FilmStack,
    film_stack_from_records,
    resist_index_from_dill,
)

__all__ = ["Film", "FilmStack", "film_stack_from_records", "resist_index_from_dill"]
