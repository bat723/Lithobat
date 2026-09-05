"""Technology pipelines: printed devices, and the flows that build them.

``gaa`` and ``nfet`` are the two device flows — a gate-all-around nanosheet
transistor and a planar bulk nFET, each built from two printed masks and
self-aligned processing; ``flows`` is what they share; ``devices`` names them
as presets and caches the built wafers. Run one with ``litho-sim device gaa``.

DRAM cell architectures (8F², 6F², 4F²), 2-D NAND and 3-D NAND flows land here
too, each a function returning a Flow plus its layout set; the device-physics
ladder (doping, extraction, electrical) follows. See the vault Roadmap.
"""
