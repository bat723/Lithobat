"""Layout Editor — author mask geometry as a table, see it rasterised live."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from app_lib import (
    TABLE_COLUMNS,
    aerial_for,
    apply_dark_axes,
    get_grid,
    get_layout,
    get_optics,
    layout_to_rows,
    page_header,
    rows_to_layout,
    set_layout,
)

from litho_sim.mask.layout import (
    Layout,
    contact_grid,
    cut_bar,
    decompose,
    line_array,
    split_by_color,
)

st.set_page_config(page_title="Layout Editor", layout="wide")
page_header(
    "Layout Editor",
    "Author mask geometry numerically. Coordinates are nanometres from the "
    "centre of the field.",
)

grid = get_grid()
optics = get_optics()
layout = get_layout()
field_nm = grid.grid_size * 1e9

# ---------------------------------------------------------------------------
# Sidebar: parametric generators and grid
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Field")
    n_px = st.select_slider("Grid (pixels)", [64, 96, 128, 160, 192], value=grid.n_pixels)
    px_nm = st.slider("Pixel size [nm]", 1.0, 8.0, grid.pixel_size * 1e9, step=0.5)
    if n_px != grid.n_pixels or abs(px_nm - grid.pixel_size * 1e9) > 1e-9:
        grid.n_pixels = int(n_px)
        grid.pixel_size = px_nm * 1e-9
        st.rerun()
    st.caption(f"Field: {grid.grid_size * 1e9:.0f} × {grid.grid_size * 1e9:.0f} nm")

    st.header("Add features")
    gen = st.selectbox("Generator", ["Line array", "Contact grid", "Cut bar"])
    if gen == "Line array":
        n = st.number_input("Lines", 1, 40, 6)
        pitch = st.number_input("Pitch [nm]", 20.0, 800.0, 80.0, step=5.0)
        cd = st.number_input("CD [nm]", 5.0, 400.0, 40.0, step=5.0)
        length = st.number_input("Length [nm]", 20.0, 2000.0, 600.0, step=20.0)
        orient = st.radio("Orientation", ["vertical", "horizontal"], horizontal=True)
        layer = st.text_input("Layer", "gate")
        if st.button("Add line array", width="stretch", type="primary"):
            set_layout(Layout(
                layout.shapes + line_array(
                    int(n), pitch * 1e-9, cd * 1e-9, length * 1e-9,
                    layer=layer, orientation=orient,
                ),
                name=layout.name,
            ))
            st.rerun()
    elif gen == "Contact grid":
        nx = st.number_input("Columns", 1, 20, 4)
        ny = st.number_input("Rows", 1, 20, 4)
        pitch = st.number_input("Pitch [nm]", 20.0, 500.0, 100.0, step=5.0)
        d = st.number_input("Diameter [nm]", 5.0, 300.0, 45.0, step=5.0)
        layer = st.text_input("Layer", "contact")
        if st.button("Add contact grid", width="stretch", type="primary"):
            set_layout(Layout(
                layout.shapes + contact_grid(
                    int(nx), int(ny), pitch * 1e-9, pitch * 1e-9, d * 1e-9, layer=layer
                ),
                name=layout.name,
            ))
            st.rerun()
    else:
        cx = st.number_input("Centre x [nm]", -1000.0, 1000.0, 0.0, step=5.0)
        cy = st.number_input("Centre y [nm]", -1000.0, 1000.0, 0.0, step=5.0)
        w = st.number_input("Width [nm]", 5.0, 2000.0, 600.0, step=10.0)
        h = st.number_input("Height [nm]", 5.0, 2000.0, 40.0, step=5.0)
        if st.button("Add cut bar", width="stretch", type="primary"):
            set_layout(Layout(
                layout.shapes + [cut_bar(cx * 1e-9, cy * 1e-9, w * 1e-9, h * 1e-9)],
                name=layout.name,
            ))
            st.rerun()

    st.divider()
    if st.button("Clear layout", width="stretch"):
        set_layout(Layout([], name=layout.name))
        st.rerun()

# ---------------------------------------------------------------------------
# Main: table + preview
# ---------------------------------------------------------------------------

col_table, col_view = st.columns([1.05, 1.35], gap="large")

with col_table:
    st.subheader("Shapes")
    st.caption(
        "Edit any cell. Add rows with the ＋ at the bottom; delete with the "
        "checkbox on the left. All units nanometres."
    )
    rows = layout_to_rows(layout)
    df = pd.DataFrame(rows, columns=TABLE_COLUMNS) if rows else pd.DataFrame(columns=TABLE_COLUMNS)
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        height=380,
        column_config={
            "kind": st.column_config.SelectboxColumn(
                "kind", options=["Rect", "Contact"], default="Rect", width="small"
            ),
            "layer": st.column_config.TextColumn("layer", default="gate", width="small"),
            "cx_nm": st.column_config.NumberColumn("x", format="%.1f", default=0.0),
            "cy_nm": st.column_config.NumberColumn("y", format="%.1f", default=0.0),
            "w_nm": st.column_config.NumberColumn("w", format="%.1f", default=40.0),
            "h_nm": st.column_config.NumberColumn("h", format="%.1f", default=200.0),
            "rot_deg": st.column_config.NumberColumn("rot°", format="%.1f", default=0.0),
        },
        key="shape_table",
    )
    if st.button("Apply table edits", type="primary", width="stretch"):
        set_layout(rows_to_layout(edited.to_dict("records"), name=layout.name))
        st.rerun()

    st.metric("Shapes", len(layout))
    st.caption(f"Layers: {', '.join(layout.layers()) or '—'}")

    st.divider()
    st.subheader("Save / load")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download JSON",
            data=json.dumps(layout.to_dict(), indent=2),
            file_name=f"{layout.name}.json",
            mime="application/json",
            width="stretch",
        )
    with c2:
        up = st.file_uploader("Load JSON", type="json", label_visibility="collapsed")
        if up is not None:
            set_layout(Layout.from_dict(json.loads(up.read().decode("utf-8"))))
            st.rerun()

with col_view:
    st.subheader("Preview")
    tone = st.radio(
        "Mask tone", ["clear", "dark"], horizontal=True,
        help="'clear' = drawn shapes transmit light (they print as trenches "
             "in positive resist). 'dark' = drawn shapes are opaque.",
    )
    show_aerial = st.checkbox("Show aerial image", value=True)

    mask = layout.rasterize(grid, tone=tone)
    ncols = 2 if show_aerial else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 5.0))
    axes = np.atleast_1d(axes)
    extent = (-field_nm / 2, field_nm / 2, -field_nm / 2, field_nm / 2)

    axes[0].imshow(mask, cmap="gray", origin="lower", extent=extent, vmin=0, vmax=1)
    axes[0].set_title(f"Mask ({tone} tone)")
    axes[0].set_xlabel("x [nm]")
    axes[0].set_ylabel("y [nm]")

    if show_aerial:
        aerial = aerial_for(mask, optics, grid)
        im = axes[1].imshow(aerial, cmap="inferno", origin="lower", extent=extent)
        contrast = (aerial.max() - aerial.min()) / (aerial.max() + aerial.min() + 1e-12)
        axes[1].set_title(f"Aerial image — contrast {contrast:.2f}")
        axes[1].set_xlabel("x [nm]")
        fig.colorbar(im, ax=axes[1], fraction=0.046)

    for ax in axes:
        apply_dark_axes(ax)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    if show_aerial:
        if contrast < 0.5:
            st.warning(
                f"Aerial contrast is only {contrast:.2f}. This layout is too "
                "dense for a single exposure — decompose it below and print it "
                "with LELE."
            )
        else:
            st.success(f"Aerial contrast {contrast:.2f} — printable in one exposure.")

# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Multi-patterning decomposition")
st.caption(
    "Features closer than the minimum single-exposure spacing must be split "
    "across exposures. An odd cycle of conflicts cannot be two-coloured — that "
    "is a real design-rule violation, not a solver limitation."
)

d1, d2, d3 = st.columns([1, 1, 2])
with d1:
    min_sp = st.number_input("Min spacing [nm]", 5.0, 400.0, 70.0, step=5.0)
with d2:
    n_colors = st.selectbox("Exposures", [2, 3], index=0,
                            format_func=lambda n: f"{n}  ({'LELE' if n == 2 else 'LE³'})")

if len(layout) >= 2:
    dec = decompose(layout.shapes, min_spacing=min_sp * 1e-9, n_colors=int(n_colors))
    groups = split_by_color(layout, dec)

    with d3:
        if dec.ok:
            st.success(
                f"Decomposed cleanly into {n_colors} exposures: "
                + ", ".join(f"{k}={len(v)}" for k, v in sorted(groups.items()))
            )
        else:
            st.error(
                f"{len(dec.conflicts)} pair(s) cannot be separated with "
                f"{n_colors} exposures — e.g. shapes {dec.conflicts[0]}. "
                f"Try {int(n_colors) + 1} exposures or relax the spacing."
            )

    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (key, sub) in zip(axes, sorted(groups.items())):
        ax.imshow(sub.rasterize(grid), cmap="gray", origin="lower", extent=extent)
        ax.set_title(f"{key} — {len(sub)} shapes")
        ax.set_xlabel("x [nm]")
        apply_dark_axes(ax)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
else:
    st.info("Add at least two shapes to decompose.")
