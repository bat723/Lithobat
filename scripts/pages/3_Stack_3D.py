"""Stack 3D — rotate, slice, and inspect the printed wafer stack."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from app_lib import get_grid, page_header

from litho_sim.viz.viz3d import (
    HAS_PLOTLY,
    cross_section_figure,
    payload_size,
    stack_figure,
    stack_figure_mpl,
)

st.set_page_config(page_title="Stack 3D", layout="wide")
page_header(
    "Printed Stack — 3D",
    "The physical result: photoresist and films as a solid you can turn around.",
)

result = st.session_state.get("flow_result")
if result is None:
    st.info(
        "No run yet. Go to the **Process Flow** page, pick a scheme, and press "
        "**Run flow** — then come back here."
    )
    st.stop()

grid = get_grid()
snapshots = result.snapshots
flow_name = st.session_state.get("flow_name", "flow")

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("View")
    step = st.slider("Process step", 1, len(snapshots), len(snapshots)) - 1
    stack = snapshots[step]

    present = stack.present_materials()
    names = [m.name for m in present]
    visible = st.multiselect("Materials", names, default=names)

    z_exag = st.slider(
        "Z exaggeration", 1.0, 6.0, 2.0, 0.5,
        help="100 nm of resist over a 500 nm field is a pancake. Every real "
             "litho tool exaggerates the vertical axis.",
    )
    cut = st.slider(
        "Cutaway (fraction of y kept)", 0.2, 1.0, 1.0, 0.05,
        help="Slice the stack open to see inside it.",
    )
    renderer = st.radio(
        "Renderer",
        ["Interactive (Plotly)", "Static (Matplotlib)"],
        index=0 if HAS_PLOTLY else 1,
        help="Plotly rotates in the browser; Matplotlib is a static fallback.",
    )

st.caption(f"**{flow_name}** — step {step + 1}/{len(snapshots)}")

# ---------------------------------------------------------------------------
# 3D view
# ---------------------------------------------------------------------------

if renderer.startswith("Interactive") and not HAS_PLOTLY:
    st.warning(
        "Plotly is not installed, so the interactive view is unavailable. "
        "Install it with `pip install plotly`, or use the static renderer."
    )
    renderer = "Static (Matplotlib)"

if renderer.startswith("Interactive"):
    fig = stack_figure(
        stack, materials=visible or None, z_exaggeration=z_exag,
        cutaway=None if cut >= 0.999 else cut,
        title="",
    )
    fig.update_layout(height=620, paper_bgcolor="#0e1117", font_color="#c8c8c8")
    st.plotly_chart(fig, width="stretch")
else:
    fig = plt.figure(figsize=(10, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    stack_figure_mpl(stack, materials=visible or None, z_exaggeration=z_exag, ax=ax)
    ax.view_init(elev=26, azim=-58)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Cross-section and metrology
# ---------------------------------------------------------------------------

st.divider()
c_left, c_right = st.columns([1.6, 1], gap="large")

with c_left:
    st.subheader("Cross-section")
    axis = st.radio("Cut plane", ["y", "x"], horizontal=True,
                    format_func=lambda a: f"{a}-cut ({'x–z' if a == 'y' else 'y–z'} plane)")
    n = stack.shape_xy[0 if axis == "y" else 1]
    idx = st.slider("Cut position [px]", 0, n - 1, n // 2)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    cross_section_figure(stack, axis=axis, index=idx, ax=ax, title="")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with c_right:
    st.subheader("Metrology")
    mat = st.selectbox(
        "Measure material", [m.name for m in stack.present_materials()],
        index=len(stack.present_materials()) - 1,
    )
    z_frac = st.slider("Height for CD [fraction]", 0.05, 0.95, 0.5, 0.05)
    feats = stack.measure_features(mat, z_frac)
    lines = [w * 1e9 for w in feats["lines"]]
    spaces = [w * 1e9 for w in feats["spaces"]]

    m1, m2 = st.columns(2)
    m1.metric("Features", len(lines))
    m2.metric("Mean CD", f"{np.mean(lines):.1f} nm" if lines else "—")
    st.write(f"**Widths:** {', '.join(f'{w:.0f}' for w in lines) or '—'} nm")
    st.write(f"**Spaces:** {', '.join(f'{w:.0f}' for w in spaces) or '—'} nm")
    st.metric("Max thickness", f"{stack.thickness_of(mat).max() * 1e9:.0f} nm")

    sizes = payload_size(stack)
    st.caption(
        f"Rendered as {sizes['surface_points']:,} surface points instead of "
        f"{sizes['volume_points']:,} voxels ({sizes['ratio']}× lighter) — which "
        "is why this stays interactive."
    )

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

st.divider()
with st.expander("Process history and export"):
    st.code("\n".join(stack.history), language=None)
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, mat=stack.mat)
    st.download_button(
        "Download stack (.npz)",
        data=buf.getvalue(),
        file_name=f"{flow_name.replace(' ', '_')}_step{step + 1}.npz",
        mime="application/octet-stream",
    )
