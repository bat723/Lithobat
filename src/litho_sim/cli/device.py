"""``litho-sim device gaa`` / ``litho-sim device nfet`` — build a printed device.

Each device's flags are its own module's (:mod:`litho_sim.tech.gaa`,
:mod:`litho_sim.tech.nfet`); this only puts them behind one word.
"""

from __future__ import annotations

import argparse

from litho_sim.tech import gaa, nfet

FLOWS = {"gaa": gaa, "nfet": nfet}


def add_parser(subs) -> None:
    ap = subs.add_parser("device", help="build a printed device and verify it structurally")
    which = ap.add_subparsers(dest="device", metavar="DEVICE")
    which.required = True
    for name, module in FLOWS.items():
        sp = which.add_parser(name, help=(module.__doc__ or "").splitlines()[0],
                              description=(module.__doc__ or "").split("\n\n")[0])
        module.add_arguments(sp)
        sp.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    return FLOWS[args.device].run(args)
