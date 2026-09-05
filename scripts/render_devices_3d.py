"""Render the finished devices in 3-D — the final product, nothing else.

Builds both device flows and draws each one as a solid: the planar bulk nFET
from ``litho_sim.tech.nfet`` and the gate-all-around nanosheet from
``litho_sim.tech.gaa``.
No per-step panels, no cross-sections — just what the wafer ends up holding.

Both are sectioned with :func:`~litho_sim.viz.viz3d.crop`, which cuts the voxel
array rather than filtering triangles, so the cut face comes out capped and you
look into a real cross-section. Sectioning is not decoration here: by the last
step every interesting feature is interior. The nFET's poly gate is clad on all
four sides — spacer on the sidewalls, silicide on top — and the GAA's wrapped
sheets are sealed under the ILD. Viewed from outside, both are blocks.

Run ``python scripts/render_devices_3d.py --outdir results``.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho_sim.tech.devices import sectioned
from litho_sim.viz.viz3d import device_figure_mpl
from litho_sim.wafer import Stack

#: Text tokens. The material swatches carry identity; the words stay neutral ink
#: so nothing competes with the object.
INK, MUTED = "#1c1f24", "#5b6169"


def render(stack: Stack, title: str, subtitle: str, path: Path,
           figsize=(9.5, 7.6), zoom: float = 1.25,
           z_exaggeration: float = 1.0, elev: float = 15.0, azim: float = -76.0,
           downsample: int = 1,
           materials: Sequence[str] | None = None) -> None:
    """One device, one 3-D solid, one file.

    Presentation is deliberately bare: no panes, no grid, no ticks, no axis
    labels. A solid object already shows its own extent, and Matplotlib's
    default 3-D decoration draws three lots of scale furniture around it that
    the eye has to look past. What replaces it is a single labelled scale bar —
    valid on every axis here because ``z_exaggeration`` is 1 and the projection
    is orthographic.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.82), projection="3d")

    device_figure_mpl(
        stack, ax=ax, z_exaggeration=z_exaggeration, downsample=downsample,
        elev=elev, azim=azim, materials=materials,
        chrome="none", scale_bar=True, projection="ortho", zoom=zoom,
        # A still is drawn once and looked at for a long time, so it pays the
        # exact per-voxel mesh: no merged-quad sort artefacts at any cost in
        # frame rate, because there are no frames. The interactive view makes
        # the opposite trade.
        merge=False,
    )

    present = [m for m in stack.present_materials()
               if materials is None or m.name in set(materials)]
    fig.legend(
        handles=[Patch(facecolor=m.color, label=m.name) for m in present],
        loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=len(present),
        fontsize=10, frameon=False, labelcolor=INK,
        handlelength=1.0, handleheight=1.0, columnspacing=1.6,
        handletextpad=0.5,
    )
    fig.text(0.5, 0.983, title, ha="center", va="top", fontsize=16,
             weight="bold", color=INK)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10,
             color=MUTED)

    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--downsample", type=int, default=1)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    from litho_sim.tech.gaa import build_gaa
    from litho_sim.tech.gaa import checks as gaa_checks
    from litho_sim.tech.nfet import build_nfet
    from litho_sim.tech.nfet import checks as nfet_checks

    print("building planar nFET ...")
    nfet, nm = build_nfet(verbose=False)
    nc = nfet_checks(nfet, nfet.shape_xy[1])
    print(f"   {nc}")
    render(
        sectioned("nfet", nfet),
        "Planar bulk nFET  —  printed, not drawn",
        f"193 nm immersion · two printed masks · gate drawn 96 nm, printed "
        f"{nm['gate_printed_nm']:.0f} nm at dose-to-size · {nc['spacer_collars']} spacer "
        f"collars, {nc['gate_shorts']} gate shorts · sectioned at the channel",
        out / "device_nfet_3d.png",
        figsize=(10.5, 5.5), zoom=1.45,
        elev=16.0, azim=-72.0, downsample=a.downsample,
    )

    print("building GAA nanosheet ...")
    gaa, gm = build_gaa(verbose=False)
    gc = gaa_checks(gaa, gaa.shape_xy[1])
    print(f"   {gc}")
    render(
        sectioned("gaa", gaa),
        "Gate-all-around nanosheet FET  —  printed, not drawn",
        f"EUV · two printed masks · gate drawn 64 nm, printed "
        f"{gm['gate_printed_nm']:.0f} nm at dose-to-size · {gc['sheets']} sheets, "
        f"{gc['wrapped']} wrapped, {gc['gate_shorts']} gate shorts · sectioned at the channel",
        out / "device_gaa_3d.png",
        figsize=(9.0, 7.8), zoom=1.24,
        elev=14.0, azim=-78.0, downsample=a.downsample,
    )


if __name__ == "__main__":
    main()
