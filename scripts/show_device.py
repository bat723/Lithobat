"""Open the app with a printed device already loaded, in 3-D, ready to rotate.

``render_devices_3d.py`` writes a picture; this opens the thing itself. It runs
a device flow, hands the finished wafer to the app's Wafer Stack tab as the
state you inherit, and switches that tab to the solid view — so the window
comes up on the device rather than on an empty recipe.

    python scripts/show_device.py                 # the GAA nanosheet
    python scripts/show_device.py --device nfet   # the planar nFET
    python scripts/show_device.py --whole         # uncropped, no section
    python scripts/show_device.py --stack results/mine.npz

Because it arrives as the *starting* wafer, the recipe list opens empty and
anything you add there runs on top of the device — deposit a film over the
finished transistor, etch into it, and scrub back and forth through the result.

The view is sectioned by default. That is not decoration: by the last step of
either flow every interesting feature is interior — the nFET's gate is clad on
all four sides and the GAA's sheets are sealed under the ILD — so an uncut
device is a rectangular block from every angle. ``--whole`` gives you that
block if you want to see the outside.

Dragging inside the 3-D view rotates it. The mesh coarsens while the button is
down and snaps back to full detail on release, which is what keeps a
quarter-million-triangle wafer usable under the mouse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from litho_sim.tech.devices import (
    DEVICES,
    flow_for,
    sectioned,
)
from litho_sim.tech.devices import (
    build as build_device,
)
from litho_sim.viz.viz3d import crop
from litho_sim.wafer import Stack

#: One definition of "the GAA preset", shared with the app's File > Load device
#: menu. Duplicating the section here is how the two quietly drift apart.
SECTIONS = {k: v.section for k, v in DEVICES.items()}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--device", choices=sorted(SECTIONS), default="gaa",
                    help="which flow to run (default: gaa)")
    ap.add_argument("--stack", type=str, default=None,
                    help="load a saved Stack (.npz) instead of running a flow")
    ap.add_argument("--whole", action="store_true",
                    help="do not section it — show the device as a solid block")
    a = ap.parse_args()

    flow = None
    section = None
    if a.stack:
        stack = Stack.load(a.stack)
        label = Path(a.stack).name
        print(f"loaded {a.stack}  {stack.mat.shape}")
        if not a.whole:
            # A saved stack has no section recipe of its own; halve it in y so
            # the cut face shows the layer structure rather than the outside.
            section = lambda s: crop(s, y=(0.45, 1.0), z_substrate=10e-9)  # noqa: E731
    else:
        print(f"building {a.device} (cached after the first run) ...")
        stack, label = build_device(a.device)
        print(f"   {label}")
        # The recipe that built it, so the app's Wafer Stack panel lists the
        # steps instead of standing empty. `build_device` writes this cache on
        # a cold run, so it is here by the time we ask.
        flow = flow_for(a.device)
        if flow is not None:
            print(f"   {len(flow)} steps recorded")
        if not a.whole:
            section = lambda s, _n=a.device: sectioned(_n, s)  # noqa: E731

    print(f"opening the app — {stack.mat.shape[0]}x{stack.mat.shape[1]}"
          f"x{stack.mat.shape[2]} voxels; drag inside the view to rotate")
    from litho_sim.app.main import main as app_main

    return app_main(stack, label=label, flow=flow, section=section)


if __name__ == "__main__":
    raise SystemExit(main())
