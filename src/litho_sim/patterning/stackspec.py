"""
The conventional tri-layer stack the recipes assume.

One module owns these names so ``multipatterning``, ``cuts``, and the future
``damascene``/``selfaligned`` recipe families agree on what they are
patterning without importing each other. Phase 3's material registry will
let recipes override these; until then they are the shared default.
"""

#: Carbon-rich underlayer the resist pattern is first transferred into.
HARDMASK = "SOC"
#: The imaging film.
RESIST = "photoresist"
#: The film every flow is ultimately trying to pattern.
TARGET = "poly-Si"
#: Sacrificial mandrel material for spacer patterning.
MANDREL = "a-C"
#: First and second spacer materials (SADP / SAQP).
SPACER1 = "spacer-oxide"
SPACER2 = "spacer-nitride"
