"""One visual system for every figure LithoPy draws.

This is the only module that knows a colour by its hex. Everything else —
the desktop app's canvases, the CLI's saved figures, the scripts — asks for
a *role*: the ink a label is written in, the hue the aerial image is drawn
in, the ramp a dose series steps through. Change a value here and every
figure moves together; nothing else needs to know.

Three rules the helpers enforce, because they are the three things that
made the earlier figures read as noise:

* **The title says what; the caption says how much.** A title is a short
  noun phrase. Numbers — contrast, NILS, CD, elapsed time — go in the
  caption row beneath it, in the secondary ink, so the eye finds the picture
  first and the readout second.
* **Every image has physical axes** and a colourbar with units when its
  colour means a quantity. A binary picture (a mask, a resist footprint) is
  drawn in two flat colours and gets a swatch, not a colourbar.
* **Nothing is dashed and no legend sits on the data.** Gridlines are solid
  hairlines, opt-in, and only on the axis that carries the reading. Legends
  go above or beside the axes and only exist when there are two or more
  series — one series is named by the title.

Call :func:`apply` once per process, deliberately. Importing this module
changes nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import SimpleNamespace

import matplotlib
import numpy as np
from cycler import cycler
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

# ---------------------------------------------------------------------------
# Ink and surface
# ---------------------------------------------------------------------------

#: The paper every figure is drawn on. Off-white rather than white so a pale
#: ramp step still shows against it.
SURFACE = "#fcfcfb"
#: Primary text: titles, values that matter.
INK = "#0b0b0b"
#: Secondary text: captions, axis labels, legend entries, tick labels.
INK2 = "#52514e"
#: Reference furniture: a target line, a "NILS 2" rule, the as-coated film.
MUTED = "#898781"
#: Gridlines — one step off the surface, hairline, solid.
GRID = "#e1e0d9"
#: Spines and tick marks.
AXIS = "#c3c2b7"
#: A neutral wash: the tolerance band, the diverging midpoint.
WASH = "#f0efec"

# ---------------------------------------------------------------------------
# Categorical palette — eight hues in a fixed order
# ---------------------------------------------------------------------------

#: Assigned in this order, never cycled past the eighth: adjacent slots are
#: separated under simulated colour-vision deficiency, and that separation
#: is a property of the *order*.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CATEGORICAL

#: What each recurring series is drawn in, so a quantity keeps its colour
#: from the Expose tab to the Develop tab to the Simulate page. A line and
#: the image of the same quantity share a hue: the aerial cut is blue under
#: a blue aerial image, the latent cut orange under an orange latent image.
SERIES = SimpleNamespace(
    aerial=BLUE,
    latent=ORANGE,
    resist=AQUA,
    threshold=INK2,
    reference=MUTED,
    warn=RED,
    signal=BLUE,
)

#: Process families for the wafer's flow strip. Names are the engine's; the
#: colours are the categorical slots, so the strip reads in the same voice as
#: every other figure.
FAMILY: dict[str, str] = {
    "substrate": MUTED,
    "litho": AQUA,
    "deposit": BLUE,
    "ald": VIOLET,
    "etch": RED,
    "cmp": ORANGE,
    "strip": YELLOW,
}

# ---------------------------------------------------------------------------
# Colour maps — one hue per quantity
# ---------------------------------------------------------------------------

#: The blue sequential ramp, light to dark. Steps 100–700 of the palette's
#: blue scale; the light end is used by images and the dark half by ordinal
#: series such as a dose ladder.
_BLUE_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b",
)
_ORANGE_RAMP = (
    "#fde3d6", "#fbd0bb", "#f8bb9d", "#f5a47f", "#f28c60", "#ee7647",
    "#eb6834", "#d95926", "#c04d20", "#a4411b", "#873516", "#6b2a11",
    "#52200d",
)


def _ramp(colors: Sequence[str], name: str, *, reverse: bool = False):
    cols = list(colors)[::-1] if reverse else list(colors)
    return LinearSegmentedColormap.from_list(name, cols, N=256)


#: Colour by the *job* the colour does, never by the panel it sits in.
#:
#: ``intensity`` and ``latent`` are image ramps: dark where there is little,
#: pale where there is much, the way a micrograph reads. ``probability`` is
#: the same blue job in a different panel. ``cd`` is diverging — thin in
#: blue, on-target in the neutral wash, fat in red — because a CD map has a
#: real midpoint. Micrographs are grey.
CMAP = SimpleNamespace(
    intensity=_ramp(("#f4f8fe",) + _BLUE_RAMP, "litho.intensity", reverse=True),
    latent=_ramp(("#fef5f0",) + _ORANGE_RAMP, "litho.latent", reverse=True),
    probability=_ramp(("#f4f8fe",) + _BLUE_RAMP, "litho.probability", reverse=True),
    cd=_ramp(("#0d366b", "#1c5cab", "#2a78d6", "#86b6ef", WASH,
              "#f0a3a2", "#e34948", "#a92f2f", "#6e1d1d"), "litho.cd"),
    # Ink on paper: the illumination's unlit points are the page, the lit
    # ones blue — a source is a drawing of settings, not a micrograph.
    source=_ramp((SURFACE,) + _BLUE_RAMP[:9], "litho.source"),
    micrograph="gray",
)


def binary_cmap(color: str, background: str = SURFACE):
    """Two flat colours for a one-bit picture: *background* where 0, *color* where 1.

    Returns ``(cmap, norm)`` for ``imshow``. A binary image drawn through a
    continuous ramp lies twice — it invents intermediate shades that are not
    in the data, and it earns a colourbar for a quantity with two values.
    """
    return ListedColormap([background, color]), BoundaryNorm([-0.5, 0.5, 1.5], 2)


def ordinal_colors(n: int, hue: str = "blue") -> list[str]:
    """*n* steps of one hue, light to dark, for an ordered series.

    A dose ladder or a set of pitches is ordinal: swapping two entries would
    change the meaning. Colour them along one ramp so the order is visible
    in the colour, and keep the light end far enough from the surface to
    read against it.
    """
    ramp = {"blue": _BLUE_RAMP, "orange": _ORANGE_RAMP}[hue]
    cmap = _ramp(ramp[3:], f"litho.ordinal.{hue}")
    if n <= 1:
        return [to_hex(cmap(0.85))]
    return [to_hex(cmap(t)) for t in np.linspace(0.0, 1.0, n)]


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

_RC: dict = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica Neue", "Helvetica", "SF Pro Text", "Segoe UI",
        "DejaVu Sans", "Arial", "sans-serif",
    ],
    "font.size": 9.5,
    "text.color": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "medium",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.titlepad": 6.0,
    "axes.labelsize": 9.0,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.markersize": 6.0,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK2,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "image.cmap": "gray",
    "image.interpolation": "nearest",
    "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
}


def apply() -> None:
    """Install the theme into matplotlib's rcParams and register the ramps.

    Idempotent. Called once by the app's Qt bootstrap and by
    :func:`litho_sim.viz.plots.apply_style`; scripts that build their own
    figures call it at the top.
    """
    matplotlib.rcParams.update(_RC)
    for cm in (CMAP.intensity, CMAP.latent, CMAP.probability, CMAP.cd, CMAP.source):
        if cm.name not in matplotlib.colormaps:
            matplotlib.colormaps.register(cm)


# ---------------------------------------------------------------------------
# Helpers — what the views call instead of hand-rolling
# ---------------------------------------------------------------------------

#: The caption sits this far above the axes; the title is pushed up to make
#: room for it. Points.
_CAPTION_PAD = 3.0
_TITLE_PAD_WITH_CAPTION = 17.0


def title(ax, what: str, *, caption_text: str | None = None) -> None:
    """Name the picture — a short noun phrase, no numbers.

    With *caption_text* the numbers row is set at the same time; otherwise
    an existing caption is kept and the title just moves to make room for it.
    """
    has_caption = caption_text is not None or getattr(ax, "_litho_caption", None) is not None
    pad = _TITLE_PAD_WITH_CAPTION if has_caption else matplotlib.rcParams["axes.titlepad"]
    ax.set_title(what, loc="left", pad=pad)
    if caption_text is not None:
        caption(ax, caption_text)


def caption(target, text: str) -> None:
    """The numbers row: small, secondary ink, beneath the title.

    On an axes the caption is an annotation pinned to the top-left corner of
    the frame; calling again replaces the text rather than stacking a second
    one. On a figure it is the suptitle, left-aligned, so a four-panel page
    still has one line that says what settings it was made with.
    """
    if isinstance(target, Figure):
        target.suptitle(text, x=0.01, ha="left", fontsize=8.5, color=INK2,
                        fontweight="normal")
        return
    ax = target
    art = getattr(ax, "_litho_caption", None)
    if art is not None and art.axes is ax:
        art.set_text(text)
    else:
        art = ax.annotate(
            text, xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(0.0, _CAPTION_PAD), textcoords="offset points",
            ha="left", va="bottom", fontsize=8.5, color=INK2,
            annotation_clip=False,
        )
        ax._litho_caption = art
    # The title has to sit above the caption; re-pad it in place.
    t = ax.title
    if t.get_text():
        ax.set_title(t.get_text(), loc="left", pad=_TITLE_PAD_WITH_CAPTION)
    # A legend already on the top edge would now sit on the caption; lift it
    # onto the title row.
    leg = ax.get_legend()
    if leg is not None and getattr(leg, "_litho_where", None) == "top":
        leg.set_bbox_to_anchor((1.0, 1.0), transform=_top_row_transform(ax))


def physical_image(
    ax, data, extent, cmap, *, vmin=None, vmax=None, cbar_label: str | None = None,
    norm=None, aspect: str = "equal", xlabel: str = "x [nm]", ylabel: str = "y [nm]",
):
    """Draw an image in real units, with a colourbar only when the colour is a quantity.

    Parameters
    ----------
    extent : (x0, x1, y0, y1)
        In the axis units — nanometres for every litho picture.
    cbar_label : str, optional
        Given, a colourbar is added and labelled with it (put the units in).
        Omitted, no colourbar: the picture is binary or a micrograph and its
        colour carries no reading.

    Returns
    -------
    AxesImage
        With ``.colorbar`` set when one was drawn. Mutate it with
        ``set_data``/``set_clim`` on later updates; the colourbar follows.
    """
    im = ax.imshow(
        data, origin="lower", extent=extent, cmap=cmap, norm=norm,
        vmin=vmin, vmax=vmax, aspect=aspect, interpolation="nearest",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if cbar_label is not None:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label, color=INK2, fontsize=8.5)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, color=AXIS, labelcolor=INK2, length=2.5)
    return im


def legend(ax, *, where: str = "top", ncol: int | None = None, **kw):
    """A legend beside the data, never on it — and none for a single series.

    Parameters
    ----------
    where : {"top", "right", "below"}
        ``"top"`` runs the entries along the top edge, right-aligned, on the
        same row as the caption (wide panels). ``"right"`` stacks them in a
        column outside the right spine (tall panels). ``"below"`` puts a row
        under the x label.
    """
    handles, labels = ax.get_legend_handles_labels()
    old = ax.get_legend()
    if len(labels) < 2:
        if old is not None:
            old.remove()
        return None
    n = ncol or len(labels)
    if where == "right":
        opts = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0),
                "borderaxespad": 0.0, "ncol": 1}
    elif where == "below":
        opts = {"loc": "upper center", "bbox_to_anchor": (0.5, -0.22),
                "borderaxespad": 0.0, "ncol": n}
    else:
        opts = {"loc": "lower right", "bbox_to_anchor": (1.0, 1.0),
                "borderaxespad": 0.2, "ncol": n, "handlelength": 1.2,
                "columnspacing": 1.0}
        if getattr(ax, "_litho_caption", None) is not None:
            # The caption owns the row just above the axes; the legend
            # takes the title row instead, right-aligned against the title.
            opts["bbox_transform"] = _top_row_transform(ax)
    opts.update(kw)
    leg = ax.legend(handles, labels, **opts)
    leg._litho_where = where
    return leg


def _top_row_transform(ax):
    """Axes fraction, shifted up by the caption row's height."""
    lift = (_CAPTION_PAD + 8.5 + 2.0) / 72.0
    return ax.transAxes + ScaledTranslation(0.0, lift, ax.figure.dpi_scale_trans)


def end_label(ax, x: float, y: float, text: str, *, color: str = INK2,
              dx: float = 4.0, dy: float = 0.0, fontsize: float = 8.0, **kw):
    """Label a series where it ends instead of in a legend box."""
    return ax.annotate(
        text, (x, y), xytext=(dx, dy), textcoords="offset points",
        ha="left", va="center", fontsize=fontsize, color=color,
        annotation_clip=False, **kw,
    )


def rule(ax, *, y: float | None = None, x: float | None = None,
         text: str | None = None, color: str = MUTED, ls: str = "-",
         lw: float = 1.0, label: str | None = None):
    """A reference line — a target, a threshold, the NILS 2 rule — with its name.

    The name is written at the line's far end in the muted ink, so the line
    needs no legend entry; pass *label* to give it one anyway.
    """
    ann = None
    if y is not None:
        line = ax.axhline(y, color=color, lw=lw, ls=ls, label=label, zorder=1.5)
        if text:
            ann = ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"),
                              xytext=(-2, 3), textcoords="offset points",
                              ha="right", va="bottom", fontsize=8, color=color)
    else:
        line = ax.axvline(x, color=color, lw=lw, ls=ls, label=label, zorder=1.5)
        if text:
            ann = ax.annotate(text, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                              xytext=(3, -2), textcoords="offset points",
                              ha="left", va="top", fontsize=8, color=color)
    # Kept on the line so a view that moves the rule can move its name.
    line._litho_text = ann
    return line


def material_colour(name: str) -> str:
    """The library colour of a material — the one colour the theme does not own."""
    from litho_sim.wafer import get_material

    return get_material(name).color


def grid(ax, axis: str = "y") -> None:
    """Opt-in hairline grid, on the axis that carries the reading."""
    ax.grid(True, axis=axis, color=GRID, lw=0.8, ls="-")
    ax.set_axisbelow(True)


def material_swatches(materials: Iterable) -> list[Patch]:
    """Legend handles for a material section — one flat swatch per material."""
    return [Patch(facecolor=m.color, edgecolor="none", label=m.name)
            for m in materials]


def placeholder(ax_or_fig, message: str):
    """Say why there is no picture yet, in the space the picture will use.

    Returns the text artist, so a view can take it away again.
    """
    if isinstance(ax_or_fig, Figure):
        return ax_or_fig.text(0.5, 0.5, message, ha="center", va="center",
                              color=INK2, fontsize=10, wrap=True)
    return ax_or_fig.text(0.5, 0.5, message, transform=ax_or_fig.transAxes,
                          ha="center", va="center", color=INK2, fontsize=10,
                          wrap=True)


__all__ = [
    "SURFACE", "INK", "INK2", "MUTED", "GRID", "AXIS", "WASH",
    "CATEGORICAL", "BLUE", "ORANGE", "AQUA", "YELLOW", "MAGENTA", "GREEN",
    "VIOLET", "RED", "SERIES", "FAMILY", "CMAP",
    "apply", "binary_cmap", "ordinal_colors", "material_colour", "title", "caption",
    "physical_image", "legend", "end_label", "rule", "grid",
    "material_swatches", "placeholder",
]
