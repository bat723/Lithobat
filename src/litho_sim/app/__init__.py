"""The desktop application — a native Qt front end for the engine.

Split so that most of it needs no display:

* ``params``    — every knob, declared once, driving both widgets and configs
* ``compute``   — parameters in, results out; plain functions over plain data
* ``scheduler`` — when to recompute, so dragging a slider feels immediate
* ``main``      — the Qt widgets, and the only module that imports PySide6

Run it with ``python -m litho_sim.app`` (needs the ``app`` extra installed).
"""

__all__ = ["params", "compute", "scheduler"]
