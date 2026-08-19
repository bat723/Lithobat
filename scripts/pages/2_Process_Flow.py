"""Process Flow — assemble a recipe, run it, and watch the stack evolve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import streamlit as st
from app_lib import get_grid, get_layout, get_optics, get_resist, page_header, set_result

from litho_sim.patterning import Flow, lele, sadp, saqp, single_exposure
from litho_sim.viz.viz3d import cross_section_figure

st.set_page_config(page_title="Process Flow", layout="wide")
page_header(
    "Process Flow",
    "Build a multi-patterning recipe and run it. Overlay error and pitch "
    "division are not modelled — they emerge from the steps.",
)

grid = get_grid()
optics = get_optics()
resist = get_resist()
layout = get_layout()

# ---------------------------------------------------------------------------
# Sidebar — recipe and process parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Recipe")
    scheme = st.selectbox(
        "Scheme", ["Single exposure", "LELE (LE²)", "LELELE (LE³)", "SADP", "SAQP"],
        index=1,
    )

    st.subheader("Exposure")
    dose = st.slider("Relative dose", 0.5, 2.0, 1.0, 0.05)
    focus_nm = st.slider("Focus [nm]", -200, 200, 0, 10)
    dose_nominal = st.slider(
        "Dose to clear [mJ/cm²]", 8.0, 60.0, resist.dose_nominal, 1.0,
        help="Sets where the develop threshold falls. Around 21 reproduces "
             "the drawn CD for these optics.",
    )
    resist.dose_nominal = dose_nominal

    overlay_nm = 0.0
    min_sp = 70.0
    spacer_nm = 20.0
    spacer2_nm = 16.0

    if scheme.startswith("LELE") or scheme.startswith("LELELE"):
        st.subheader("Decomposition & overlay")
        min_sp = st.number_input("Min spacing [nm]", 5.0, 400.0, 70.0, step=5.0)
        overlay_nm = st.slider(
            "Overlay error [nm]", 0.0, 20.0, 6.0, 0.5,
            help="Misalignment of every exposure after the first. Printed "
                 "features should alternate by about twice this.",
        )
    elif scheme in ("SADP", "SAQP"):
        st.subheader("Spacer")
        spacer_nm = st.slider("Spacer 1 thickness [nm]", 6.0, 40.0, 20.0, 1.0)
        if scheme == "SAQP":
            spacer2_nm = st.slider("Spacer 2 thickness [nm]", 6.0, 40.0, 16.0, 1.0)

    st.subheader("Resist")
    resist.diffusion_sigma = st.slider(
        "PEB diffusion [nm]", 0.0, 40.0, resist.diffusion_sigma * 1e9, 1.0
    ) * 1e-9
    resist.mack_Mth = st.slider("Develop threshold (PAC)", 0.1, 0.9, resist.mack_Mth, 0.05)

    run = st.button("Run flow", type="primary", width="stretch")

# ---------------------------------------------------------------------------
# Build the flow
# ---------------------------------------------------------------------------

optics.defocus = focus_nm * 1e-9


def build_flow() -> Flow:
    if scheme == "Single exposure":
        return single_exposure(layout, grid, optics, resist, dose=dose)
    if scheme.startswith("LELE (") or scheme.startswith("LELELE"):
        n_colors = 3 if scheme.startswith("LELELE") else 2
        return lele(
            layout, grid, optics, resist, min_spacing=min_sp * 1e-9,
            overlay=(overlay_nm * 1e-9, 0.0), dose=dose, n_colors=n_colors,
        )
    mandrel = layout
    if scheme == "SADP":
        return sadp(mandrel, grid, optics, resist,
                    spacer_thickness=spacer_nm * 1e-9, dose=dose)
    return saqp(mandrel, grid, optics, resist,
                spacer1_thickness=spacer_nm * 1e-9,
                spacer2_thickness=spacer2_nm * 1e-9, dose=dose)


try:
    flow = build_flow()
    build_error = None
except ValueError as exc:
    flow, build_error = None, str(exc)

if build_error:
    st.error(build_error)
    st.info(
        "Open the **Layout Editor** page to relax the spacing rule, move the "
        "conflicting features, or switch to LE³."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Recipe listing
# ---------------------------------------------------------------------------

col_steps, col_out = st.columns([1, 1.7], gap="large")

with col_steps:
    st.subheader(f"{flow.name} — {len(flow)} steps")
    st.code("\n".join(flow.describe()), language=None)
    st.download_button(
        "Download recipe JSON",
        data=json.dumps(flow.to_dict(), indent=2),
        file_name=f"{flow.name.replace(' ', '_')}.json",
        mime="application/json",
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if run:
    prog = st.progress(0.0, text="Running…")

    def _tick(i, step, stack):
        prog.progress((i + 1) / len(flow), text=f"{i + 1}/{len(flow)} · {step.describe()}")

    try:
        result = flow.run(snapshot=True, on_step=_tick)
        st.session_state["flow_result"] = result
        st.session_state["flow_name"] = flow.name
        set_result(result)
    except RuntimeError as exc:
        prog.empty()
        st.error(str(exc))
        st.stop()
    prog.empty()

result = st.session_state.get("flow_result")

with col_out:
    if result is None:
        st.info("Press **Run flow** in the sidebar.")
        st.stop()

    st.subheader("Measurements")
    if len(result.measurements):
        show = result.measurements.copy()
        for c in ("lines_nm", "spaces_nm"):
            show[c] = show[c].apply(lambda v: ", ".join(f"{x:.0f}" for x in v))
        st.dataframe(
            show[["name", "n_lines", "mean_line_nm", "lines_nm",
                  "mean_space_nm", "pitch_walk_nm"]],
            width="stretch", hide_index=True,
            column_config={
                "mean_line_nm": st.column_config.NumberColumn("mean CD", format="%.1f nm"),
                "mean_space_nm": st.column_config.NumberColumn("mean space", format="%.1f nm"),
                "pitch_walk_nm": st.column_config.NumberColumn("pitch walk", format="%.1f nm"),
                "lines_nm": st.column_config.TextColumn("widths [nm]"),
            },
        )
        final = result.measurements.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Features printed", int(final["n_lines"]))
        m2.metric("Mean CD", f"{final['mean_line_nm']:.1f} nm")
        m3.metric(
            "Pitch walk", f"{final['pitch_walk_nm']:.1f} nm",
            delta=f"{final['pitch_walk_nm'] - 2 * overlay_nm:+.1f} vs 2×overlay"
            if overlay_nm else None,
            delta_color="inverse",
        )
    else:
        st.info("This recipe recorded no measurements.")

# ---------------------------------------------------------------------------
# Step scrubber
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Step through the flow")
st.caption("Scrub to any point in the recipe and see the wafer cross-section there.")

idx = st.slider(
    "Step", 1, len(result.snapshots), len(result.snapshots),
    format="%d",
) - 1
st.markdown(f"**{idx + 1}. {flow.steps[idx].describe()}**")

snap = result.snapshots[idx]
fig, ax = plt.subplots(figsize=(11, 3.4))
cross_section_figure(snap, ax=ax, title="")
fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

cols = st.columns(max(len(snap.present_materials()), 1))
for c, m in zip(cols, snap.present_materials()):
    c.metric(m.name, f"{snap.thickness_of(m).max() * 1e9:.0f} nm")

st.caption(
    "Open the **Stack 3D** page to rotate the final result."
)
