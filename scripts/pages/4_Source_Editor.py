"""Source Editor — author illumination and see what it does to the image."""

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
    aerial_for,
    apply_dark_axes,
    get_grid,
    get_layout,
    get_optics,
    get_source,
    page_header,
    set_source,
)

from litho_sim.analysis import compute_nils
from litho_sim.expose.source import SOURCE_PRESETS, Pole, Source

st.set_page_config(page_title="Source Editor", layout="wide")
page_header(
    "Source Editor",
    "Illumination as a composable object. Every classical shape — conventional, "
    "annular, dipole, quasar — is the same annular-sector primitive with "
    "different parameters.",
)

grid = get_grid()
optics = get_optics()
layout = get_layout()
source = get_source()

POLE_COLUMNS = ["sigma_inner", "sigma_outer", "angle_deg", "opening_deg", "weight"]

# ---------------------------------------------------------------------------
# Sidebar — presets and global settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Preset")
    preset = st.selectbox("Start from", list(SOURCE_PRESETS), index=0)
    c1, c2 = st.columns(2)
    with c1:
        s_out = st.number_input("σ outer", 0.05, 1.0, 0.90, step=0.05)
    with c2:
        s_in = st.number_input("σ inner", 0.0, 0.95, 0.60, step=0.05)
    opening = st.slider("Pole opening [°]", 5, 360, 40, 5)
    if st.button("Load preset", type="primary", width="stretch"):
        ctor = SOURCE_PRESETS[preset]
        try:
            if preset == "Conventional":
                new = ctor(s_out)
            elif preset == "Annular":
                new = ctor(s_out, s_in)
            elif preset == "Dipole":
                new = ctor(s_out, s_in, opening_deg=opening)
            else:
                new = ctor(s_out, s_in, opening_deg=opening)
            set_source(new)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.divider()
    st.header("Global")
    blur = st.slider(
        "Pole blur", 0.0, 0.20, float(source.blur), 0.01,
        help="Gaussian softening in pupil units. Real illuminators do not have "
             "infinitely sharp poles — but blur costs FFTs.",
    )
    prune = st.select_slider(
        "Prune threshold", [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2],
        value=float(source.prune),
        format_func=lambda v: "off" if v == 0 else f"{v:.0e}",
        help="Drop negligible weights after blurring. Pure cost control — at "
             "1e-3 the aerial image is unchanged to four decimal places.",
    )
    src_grid = st.select_slider(
        "Source grid", [11, 15, 21, 31, 41, 61], value=int(optics.source_grid),
        help="Sampling of the illuminator. This sets simulation cost: the Abbe "
             "sum costs one FFT per non-zero source point.",
    )
    if blur != source.blur or prune != source.prune:
        source.blur, source.prune = blur, prune
        set_source(source)
        st.rerun()
    if src_grid != optics.source_grid:
        optics.source_grid = int(src_grid)
        st.rerun()

# ---------------------------------------------------------------------------
# Main — pole table and pupil
# ---------------------------------------------------------------------------

col_table, col_pupil = st.columns([1.05, 1.2], gap="large")

with col_table:
    st.subheader("Poles")
    st.caption(
        "Each row is an annular sector. `opening_deg = 360` gives a full ring; "
        "`sigma_inner = 0` gives a filled disc."
    )
    if source.pixels is not None:
        st.info(
            "This is a free-form source — the pixel map overrides any poles. "
            "Clear it below to go back to pole editing."
        )

    rows = [
        {k: getattr(p, k) for k in POLE_COLUMNS} for p in source.poles
    ] or [{k: v for k, v in zip(POLE_COLUMNS, [0.6, 0.9, 0.0, 40.0, 1.0])}]
    edited = st.data_editor(
        pd.DataFrame(rows, columns=POLE_COLUMNS),
        num_rows="dynamic", width="stretch", height=260,
        column_config={
            "sigma_inner": st.column_config.NumberColumn("σ in", format="%.2f", default=0.6),
            "sigma_outer": st.column_config.NumberColumn("σ out", format="%.2f", default=0.9),
            "angle_deg": st.column_config.NumberColumn("angle°", format="%.0f", default=0.0),
            "opening_deg": st.column_config.NumberColumn("open°", format="%.0f", default=40.0),
            "weight": st.column_config.NumberColumn("weight", format="%.2f", default=1.0),
        },
        key="pole_table",
    )
    if st.button("Apply pole edits", type="primary", width="stretch"):
        poles, errors = [], []
        for i, r in enumerate(edited.to_dict("records")):
            try:
                poles.append(Pole(**{k: float(r[k]) for k in POLE_COLUMNS}))
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f"row {i + 1}: {exc}")
        if errors:
            for e in errors:
                st.error(e)
        else:
            set_source(Source(poles=poles, blur=blur, prune=prune, name="custom"))
            st.rerun()

    st.divider()
    st.subheader("Free-form")
    up = st.file_uploader("Import pixel map (PNG or CSV)", type=["png", "csv"])
    if up is not None:
        tmp = Path(st.session_state.get("_tmpdir", ".")) / up.name
        try:
            if up.name.lower().endswith(".csv"):
                arr = np.loadtxt(up, delimiter=",")
            else:
                from PIL import Image

                arr = np.asarray(Image.open(up).convert("L"), dtype=float) / 255.0
            set_source(Source.from_array(arr, name=up.name, blur=blur, prune=prune))
            st.rerun()
        except Exception as exc:
            st.error(f"Could not read {up.name}: {exc}")
    if source.pixels is not None and st.button("Clear free-form map", width="stretch"):
        source.pixels = None
        set_source(source)
        st.rerun()

    st.divider()
    st.download_button(
        "Download source JSON",
        data=json.dumps(source.to_dict(), indent=2),
        file_name=f"{source.name}.json",
        mime="application/json",
        width="stretch",
    )

with col_pupil:
    st.subheader("Pupil")
    n = int(optics.source_grid)
    rendered = source.render(n)
    n_pts = int((rendered > 0).sum())

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.imshow(rendered, cmap="hot", origin="lower", extent=(-1, 1, -1, 1),
              interpolation="nearest")
    ring = plt.Circle((0, 0), 1.0, fill=False, color="#5aa9e6", lw=1.4, ls="--")
    ax.add_patch(ring)
    ax.set(xlabel="ξ [pupil units]", ylabel="η [pupil units]",
           title=f"{source.name} — {source.describe()}")
    apply_dark_axes(ax)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Source points", n_pts, help="One FFT each — this is the cost.")
    m2.metric("Pupil fill", f"{source.pupil_fill(61) * 100:.0f}%")
    m3.metric("Poles", len(source.poles) if source.pixels is None else "free-form")

    if n_pts > 600:
        st.warning(
            f"{n_pts} source points means {n_pts} FFTs per aerial image. "
            "Raise the prune threshold or lower the source grid if this gets slow."
        )

# ---------------------------------------------------------------------------
# What it does to the image
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Effect on the current layout")
st.caption(
    f"Imaging **{layout.name}** ({len(layout)} shapes) at NA {optics.NA:.2f}, "
    f"λ {optics.wavelength * 1e9:.0f} nm. Off-axis illumination wins at tight "
    "pitch; at relaxed pitch it can lose."
)

mask = layout.rasterize(grid)
compare = st.multiselect(
    "Compare against",
    ["Conventional", "Annular", "Dipole", "Quasar", "CQuad"],
    default=["Conventional", "Dipole"],
)

specs = {"current": source}
for name in compare:
    specs[name] = SOURCE_PRESETS[name]()

cols = st.columns(len(specs))
extent = (-grid.grid_size / 2 * 1e9, grid.grid_size / 2 * 1e9) * 2
for (name, s), col in zip(specs.items(), cols):
    import dataclasses

    o = dataclasses.replace(optics, source_spec=s.to_dict())
    aerial = aerial_for(mask, o, grid)
    contrast = float((aerial.max() - aerial.min()) / (aerial.max() + aerial.min() + 1e-12))
    mid = aerial[aerial.shape[0] // 2, :]
    nils = compute_nils(mid, grid.pixel_size, threshold=0.3,
                        nominal_cd=grid.pixel_size * 10)
    with col:
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax.imshow(aerial, cmap="inferno", origin="lower", extent=extent)
        ax.set_title(f"{name}\ncontrast {contrast:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        apply_dark_axes(ax)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption(f"NILS {nils:.2f} · {s.n_points(int(optics.source_grid))} pts")

st.caption(
    "The active source is written to `OpticsConfig.source_spec`, so the Layout "
    "Editor preview and every Process Flow exposure use it too."
)
