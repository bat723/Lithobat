"""
Shared helpers for the Streamlit pages.

Session-state rules worth stating once:

* A :class:`~litho_sim.wafer.stack.Stack` lives in ``st.session_state``, never in
  ``st.cache_data``.  It is megabytes of mutable array; caching it would
  deep-copy on every access.
* ``st.cache_data`` is used only on pure functions keyed by small,
  hashable arguments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.mask.geometry import Contact, Rect, Shape
from litho_sim.mask.layout import Layout, line_array

# Session-state keys
K_LAYOUT = "ls_layout"
K_FLOW = "ls_flow"
K_RESULT = "ls_result"
K_GRID = "ls_grid"
K_OPTICS = "ls_optics"
K_RESIST = "ls_resist"
K_SOURCE = "ls_source"

WAVELENGTHS = {
    "i-line": 365e-9, "KrF": 248e-9, "ArF": 193e-9,
    "ArF_immersion": 193e-9, "EUV": 13.5e-9,
}
NODE_DEFAULTS = {
    "i-line": dict(NA=0.57, n_immersion=1.0),
    "KrF": dict(NA=0.75, n_immersion=1.0),
    "ArF": dict(NA=0.93, n_immersion=1.0),
    "ArF_immersion": dict(NA=1.35, n_immersion=1.44),
    "EUV": dict(NA=0.33, n_immersion=1.0),
}


# ---------------------------------------------------------------------------
# Defaults and session state
# ---------------------------------------------------------------------------


def default_layout() -> Layout:
    """Six lines at 80 nm pitch — too dense for one ArF exposure.

    Chosen so the Process Flow page has something interesting to do
    immediately: a single exposure of this washes out, LELE prints it.
    """
    return Layout(
        line_array(6, pitch=80e-9, cd=40e-9, length=600e-9, layer="gate"),
        name="gates",
    )


def get_grid() -> GridConfig:
    if K_GRID not in st.session_state:
        st.session_state[K_GRID] = GridConfig(
            n_pixels=128, pixel_size=4e-9, dz=4e-9, n_z_slices=9
        )
    return st.session_state[K_GRID]


def get_optics() -> OpticsConfig:
    if K_OPTICS not in st.session_state:
        st.session_state[K_OPTICS] = OpticsConfig(
            wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=15
        )
    return st.session_state[K_OPTICS]


def get_resist() -> ResistConfig:
    if K_RESIST not in st.session_state:
        st.session_state[K_RESIST] = ResistConfig(
            n_resist=1.7, dill_A=0.8, dill_B=0.05, dill_C=0.04,
            dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9,
        )
    return st.session_state[K_RESIST]


def get_source():
    """The active illumination source, shared across pages."""
    from litho_sim.expose.source import Source

    if K_SOURCE not in st.session_state:
        st.session_state[K_SOURCE] = Source.conventional(0.8)
    return st.session_state[K_SOURCE]


def set_source(source) -> None:
    """Store the source and push it onto the shared optics.

    Writing ``source_spec`` is what makes every other page — the layout
    preview, the process flow — actually use it.
    """
    st.session_state[K_SOURCE] = source
    get_optics().source_spec = source.to_dict()
    st.session_state.pop(K_RESULT, None)


def get_layout() -> Layout:
    if K_LAYOUT not in st.session_state:
        st.session_state[K_LAYOUT] = default_layout()
    return st.session_state[K_LAYOUT]


def set_layout(layout: Layout) -> None:
    st.session_state[K_LAYOUT] = layout
    # A new layout invalidates any run built from the old one.
    st.session_state.pop(K_RESULT, None)


def get_result():
    return st.session_state.get(K_RESULT)


def set_result(result) -> None:
    st.session_state[K_RESULT] = result


# ---------------------------------------------------------------------------
# Shape table <-> Layout
# ---------------------------------------------------------------------------

TABLE_COLUMNS = ["kind", "layer", "cx_nm", "cy_nm", "w_nm", "h_nm", "rot_deg"]


def layout_to_rows(layout: Layout) -> list[dict[str, Any]]:
    """Flatten a layout to editable table rows, in nanometres.

    Only ``Rect`` and ``Contact`` round-trip through the table; polygons and
    paths are shown read-only, since a flat table cannot express a vertex
    list without becoming unusable.
    """
    rows = []
    for s in layout.shapes:
        x0, y0, x1, y1 = s.bounds()
        rows.append({
            "kind": type(s).__name__,
            "layer": s.layer,
            "cx_nm": round((x0 + x1) / 2 * 1e9, 2),
            "cy_nm": round((y0 + y1) / 2 * 1e9, 2),
            "w_nm": round((x1 - x0) * 1e9, 2),
            "h_nm": round((y1 - y0) * 1e9, 2),
            "rot_deg": round(getattr(s, "rotation", 0.0), 2),
        })
    return rows


def rows_to_layout(rows, name: str = "layout") -> Layout:
    """Rebuild a layout from edited table rows.

    Rows with a missing or non-positive size are skipped rather than raising,
    because a data editor produces half-filled rows while the user types.
    """
    shapes: list[Shape] = []
    for r in rows:
        try:
            kind = str(r.get("kind") or "Rect")
            layer = str(r.get("layer") or "gate")
            cx = float(r.get("cx_nm") or 0.0) * 1e-9
            cy = float(r.get("cy_nm") or 0.0) * 1e-9
            w = float(r.get("w_nm") or 0.0) * 1e-9
            h = float(r.get("h_nm") or 0.0) * 1e-9
            rot = float(r.get("rot_deg") or 0.0)
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        if kind == "Contact":
            shapes.append(Contact(layer, cx, cy, w, "square"))
        else:
            shapes.append(Rect(layer, cx, cy, w, h, rot))
    return Layout(shapes=shapes, name=name)


# ---------------------------------------------------------------------------
# Cached compute
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_aerial(
    mask_bytes: bytes, shape: tuple, wavelength: float, NA: float,
    sigma_outer: float, sigma_inner: float, source_type: str,
    source_grid: int, defocus: float, n_immersion: float,
    n_pixels: int, pixel_size: float,
    source_kwargs_json: str = "{}", source_spec_json: str = "null",
) -> np.ndarray:
    """Aerial image, keyed on primitives so Streamlit hashes cheaply.

    The source arguments arrive as JSON strings because ``st.cache_data``
    needs hashable keys and a dict is not one. ``sort_keys=True`` at the call
    site keeps the key stable across dict insertion order.
    """
    from litho_sim.expose.aerial_image import compute_aerial_image

    mask = np.frombuffer(mask_bytes, dtype=np.float64).reshape(shape)
    optics = OpticsConfig(
        wavelength=wavelength, NA=NA, sigma_outer=sigma_outer,
        sigma_inner=sigma_inner, source_type=source_type,
        source_grid=source_grid, defocus=defocus, n_immersion=n_immersion,
        source_kwargs=json.loads(source_kwargs_json),
        source_spec=json.loads(source_spec_json),
    )
    grid = GridConfig(n_pixels=n_pixels, pixel_size=pixel_size)
    return compute_aerial_image(mask, optics, grid)


def aerial_for(mask: np.ndarray, optics: OpticsConfig, grid: GridConfig) -> np.ndarray:
    """Convenience wrapper around :func:`cached_aerial`.

    This used to drop ``source_kwargs`` and had no notion of ``source_spec``,
    so dipole axis, quad rotation, and every custom source were silently
    ignored on the pages that call it — the aerial preview quietly showed
    conventional illumination no matter what you selected.
    """
    m = np.ascontiguousarray(mask, dtype=np.float64)
    return cached_aerial(
        m.tobytes(), m.shape, optics.wavelength, optics.NA, optics.sigma_outer,
        optics.sigma_inner, optics.source_type, optics.source_grid,
        optics.defocus, optics.n_immersion, grid.n_pixels, grid.pixel_size,
        json.dumps(optics.source_kwargs or {}, sort_keys=True),
        json.dumps(optics.source_spec, sort_keys=True),
    )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def apply_dark_axes(ax) -> None:
    """Match the app's dark terminal theme."""
    ax.set_facecolor("#0e1117")
    ax.figure.patch.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_color("#3a3f4b")
    ax.tick_params(colors="#c8c8c8", labelsize=8)
    ax.xaxis.label.set_color("#c8c8c8")
    ax.yaxis.label.set_color("#c8c8c8")
    ax.title.set_color("#e8e8e8")


def stack_summary(stack) -> dict[str, Any]:
    """Compact per-material summary for a metrics row."""
    return {
        m.name: f"{stack.thickness_of(m).max() * 1e9:.0f} nm"
        for m in stack.present_materials()
    }
