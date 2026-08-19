"""
LithoPy Interactive Simulator
==============================
Run with:
    streamlit run scripts/interactive_app.py

"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from litho_sim.analysis import compute_nils
from litho_sim.core.config import (
    TECH_NODE_PRESETS,
    GridConfig,
    OpticsConfig,
    ResistConfig,
    SimulationConfig,
)
from litho_sim.develop.resist import apply_peb, threshold_development
from litho_sim.expose.aerial_image import compute_aerial_image
from litho_sim.expose.illumination import SOURCE_TYPES
from litho_sim.expose.pupil import zernike_noll
from litho_sim.mask.patterns import (
    checkerboard,
    contact_array,
    isolated_line,
    lines_and_spaces,
)


def _develop(aerial, threshold, diffusion_nm, pixel_size):
    """Post-exposure bake, then threshold — via the library, not inline.

    This used to be a bare ``np.where(aerial >= threshold, 0., 1.)``, which
    meant the PEB Diffusion slider changed nothing but the cache key. Acid
    diffusion is applied here as a Gaussian on the aerial image, the standard
    cheap proxy for the intensity-space threshold model.
    """
    latent = apply_peb(aerial, diffusion_nm * 1e-9, pixel_size)
    return threshold_development(latent, threshold, tone="positive")


# =============================================================================
# CONSTANTS
# =============================================================================

# Wavelength, NA and immersion index come from the library's node presets —
# this table used to be a hand-copy and had drifted (i-line NA 0.63 vs 0.57,
# KrF 0.80 vs 0.75, immersion n 1.436 vs 1.44). One source of truth now.
WAVELENGTHS_M = {n: p["wavelength"] for n, p in TECH_NODE_PRESETS.items()}
NODE_N_FLUID = {n: p.get("n_immersion", 1.0) for n, p in TECH_NODE_PRESETS.items()}

#: Default partial-coherence settings per node — a UI choice of this app,
#: not a library preset.
_NODE_SIGMAS = {
    "i-line":        (0.70, 0.0),
    "KrF":           (0.75, 0.0),
    "ArF":           (0.85, 0.0),
    "ArF_immersion": (0.90, 0.55),
    "EUV":           (0.80, 0.40),
}

NODE_DEFAULTS = {
    n: dict(na=p["NA"], sigma_outer=_NODE_SIGMAS[n][0],
            sigma_inner=_NODE_SIGMAS[n][1])
    for n, p in TECH_NODE_PRESETS.items()
}

# Fixed: both lists now use "Horizontal Line" and "Vertical Line" (capital L)
MASK_TYPES = [
    "Wordlines",
    "Contact Array",
    "Horizontal Line",
    "Vertical Line",
    "SRAM",
]

DEFECT_TYPES = [
    "None",
    "Pinhole",
    "Blob",
    "Bridge",
    "Break",
    "Line Edge Roughness",
]

# Labels follow the library's Noll table (litho_sim.expose.pupil), which this
# app's own copy used to contradict: its sine/cosine assignments were swapped
# for every m != 0 pair, so Z5 here and Z5 in the engine were different
# aberrations. The display now evaluates the same polynomials the engine does.
ZERNIKE_NAMES = {
    4:  "Z4  Defocus",
    5:  "Z5  Astigmatism 45 deg",
    6:  "Z6  Astigmatism 0 deg",
    7:  "Z7  Coma Y",
    8:  "Z8  Coma X",
    9:  "Z9  Trefoil Y",
    10: "Z10 Trefoil X",
    11: "Z11 Primary Spherical",
    12: "Z12 2nd Astigmatism 0 deg",
    13: "Z13 2nd Astigmatism 45 deg",
    14: "Z14 Tetrafoil 0 deg",
    15: "Z15 Tetrafoil 45 deg",
    16: "Z16 2nd Coma X",
    17: "Z17 2nd Coma Y",
    18: "Z18 2nd Trefoil X",
    19: "Z19 2nd Trefoil Y",
    20: "Z20 Pentafoil X",
    21: "Z21 Pentafoil Y",
}


# =============================================================================
# PAGE CONFIG AND CSS
# =============================================================================

st.set_page_config(
    page_title="LithoPy Simulator",
    layout="wide",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #1a1a1a;
    color: #c8c8c8;
    font-family: 'Courier New', Courier, monospace;
}
[data-testid="stSidebar"] {
    background-color: #222222;
    border-right: none;
}
[data-testid="stSidebar"] * {
    font-family: 'Courier New', Courier, monospace;
    color: #c8c8c8;
}
[data-testid="stHeader"] {
    background-color: #1a1a1a;
    border-bottom: none;
}
h1 {
    font-family: 'Courier New', Courier, monospace;
    color: #c8913a;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
h2, h3 {
    font-family: 'Courier New', Courier, monospace;
    color: #888888;
    font-size: 0.75rem;
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
[data-testid="stCaptionContainer"] p {
    color: #666666;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
}
[data-testid="stMetric"] {
    background-color: #222222;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] p {
    color: #888888;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #c8913a;
    font-family: 'Courier New', Courier, monospace;
    font-size: 1.4rem;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem; }
[data-testid="stSlider"] label p {
    color: #888888;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
[data-baseweb="slider"] [data-testid="stTickBarMin"],
[data-baseweb="slider"] [data-testid="stTickBarMax"] { color: #555555; }
[data-baseweb="select"] {
    background-color: #2a2a2a;
    border: 1px solid #333333;
    border-radius: 3px;
}
[data-baseweb="select"] * {
    color: #c8c8c8;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.85rem;
}
[data-testid="stFormSubmitButton"] button {
    background-color: #1a1a1a;
    color: #c8913a;
    border: 1px solid #c8913a;
    border-radius: 3px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: background-color 0.15s ease, color 0.15s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #c8913a;
    color: #1a1a1a;
}
[data-testid="stDownloadButton"] button {
    background-color: #222222;
    color: #c8c8c8;
    border: 1px solid #333333;
    border-radius: 3px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #c8913a;
    color: #c8913a;
}
[data-testid="stButton"] button {
    background-color: #222222;
    color: #c8c8c8;
    border: 1px solid #444444;
    border-radius: 3px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
}
[data-testid="stButton"] button:hover {
    border-color: #c8913a;
    color: #c8913a;
}
hr { border-color: #2e2e2e; }
[data-testid="stSpinner"] p {
    color: #888888;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
}
[data-testid="stAlert"] {
    background-color: #2a1a1a;
    border: 1px solid #5a2a2a;
    border-radius: 3px;
    color: #cc6666;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.82rem;
}
[data-testid="stSidebar"] h3 {
    color: #555555;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 1.4rem;
    margin-bottom: 0.3rem;
}
button[data-baseweb="tab"] {
    color: #888888;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    background-color: transparent;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #c8913a;
    border-bottom-color: #c8913a;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>Optical Lithography Simulation</h1>
<p style="color:#666666; font-family:'Courier New',monospace;
          font-size:0.85rem; margin-top:-0.5rem;">
    193nm aerial image and resist model -- adjust parameters and run simulation.
</p>
""", unsafe_allow_html=True)


# =============================================================================
# ZERNIKE POLYNOMIAL HELPERS
# =============================================================================

def build_pupil_display(
    N: int,
    zernike_coeffs_waves: dict,
    defocus_nm: float,
    NA: float,
    wavelength_nm: float,
    n_fluid: float = 1.0,
) -> tuple:
    """
    Build a complex pupil function on an N x N normalised pupil coordinate grid
    for display purposes (independent of the simulation pixel size).

    Wavefront:
        W(rho, theta) = sum_j c_j * Z_j(rho, theta)
                      + n * dz / lambda * (sqrt(1 - (rho*NA/n)^2) - 1)

    Complex pupil:
        P = A(rho) * exp(2*pi*i * W)

    where A(rho) = 1 for rho <= 1, 0 otherwise.

    Parameters
    ----------
    N                    : grid size in pixels (N x N output)
    zernike_coeffs_waves : {Noll_index: coefficient_in_waves}
    defocus_nm           : additional defocus in nm (separate from Z4)
    NA                   : numerical aperture
    wavelength_nm        : wavelength in nm
    n_fluid              : immersion medium refractive index

    Returns
    -------
    pupil    : complex ndarray (N, N)
    W_masked : wavefront in waves (N, N), NaN outside the pupil
    inside   : bool ndarray (N, N), True inside NA circle
    rms      : float, RMS wavefront error inside pupil in waves
    """
    coords = np.linspace(-1.0, 1.0, N)
    XX, YY = np.meshgrid(coords, coords)
    rho    = np.sqrt(XX ** 2 + YY ** 2)
    theta  = np.arctan2(YY, XX)

    inside      = rho <= 1.0
    rho_safe    = np.where(inside, rho, 0.0)   # avoid out-of-range values

    W = np.zeros((N, N), dtype=float)

    # Accumulate Zernike terms — the engine's own polynomials, so the display
    # and the simulation agree on what every slider means.
    for j, coeff in zernike_coeffs_waves.items():
        if abs(coeff) > 1e-12:
            W += coeff * zernike_noll(j, rho_safe, theta)

    # Defocus OPD -- exact non-paraxial formula in waves
    if abs(defocus_nm) > 1e-6:
        arg     = np.clip((rho_safe * NA / n_fluid) ** 2, 0.0, 1.0)
        opd_nm  = n_fluid * defocus_nm * (np.sqrt(1.0 - arg) - 1.0)
        W      += opd_nm / wavelength_nm

    amplitude = np.where(inside, 1.0, 0.0)
    pupil     = amplitude * np.exp(2j * np.pi * W)

    W_masked  = np.where(inside, W, np.nan)
    W_inside  = W[inside]
    rms       = float(np.std(W_inside)) if W_inside.size > 0 else 0.0

    return pupil, W_masked, inside, rms


# =============================================================================
# MEASUREMENT HELPERS
# =============================================================================

def _measure_cd_nm(profile: np.ndarray, pixel_size_m: float) -> float:
    """
    Measure the CD of the widest cleared feature in a 1D resist profile.

    Convention: cleared = 0 (exposed and developed), unexposed = 1.
    Finds all contiguous runs of cleared pixels and returns the width
    of the widest one in nm.

    Parameters
    ----------
    profile      : 1D float array, resist state (0 = cleared, 1 = standing)
    pixel_size_m : pixel pitch in metres
    """
    cleared = profile < 0.5
    if not np.any(cleared):
        return 0.0
    padded      = np.concatenate([[False], cleared, [False]])
    transitions = np.diff(padded.astype(int))
    starts      = np.where(transitions ==  1)[0]
    ends        = np.where(transitions == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return 0.0
    widths    = ends - starts
    cd_pixels = float(np.max(widths))
    return cd_pixels * pixel_size_m * 1e9


# =============================================================================
# DEFECT INJECTION AND DETECTION
# =============================================================================

def inject_defect(
    mask: np.ndarray,
    defect_type: str,
    defect_size_nm: float,
    pixel_size_m: float,
    seed: int = 42,
) -> tuple:
    """
    Inject a physical mask-level defect into a clean binary mask.

    Parameters
    ----------
    mask           : 2D float array, binary mask (0 = opaque, 1 = clear)
    defect_type    : one of DEFECT_TYPES
    defect_size_nm : defect radius in nm
    pixel_size_m   : pixel pitch in metres
    seed           : random seed (controls LER noise pattern)

    Returns
    -------
    (defective_mask, cx, cy, r_px)
        defective_mask : ndarray with defect applied
        cx, cy         : defect centre in pixel coordinates
        r_px           : defect radius in pixels
    """
    n   = mask.shape[0]
    mid = n // 2

    if defect_type == "None":
        return mask.copy(), mid, mid, 0

    rng = np.random.default_rng(int(seed))
    m   = mask.copy().astype(float)
    r   = max(1, int(round(defect_size_nm * 1e-9 / pixel_size_m)))
    cx, cy = mid, mid

    yy, xx = np.ogrid[:n, :n]

    if defect_type == "Pinhole":
        # Clear spot inserted into an opaque region
        row         = m[mid, :]
        opaque_cols = np.where(row < 0.5)[0]
        cx          = int(opaque_cols[len(opaque_cols) // 2]) if len(opaque_cols) else mid
        cy          = mid
        m[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 1.0

    elif defect_type == "Blob":
        # Opaque particle placed in a clear region
        row        = m[mid, :]
        clear_cols = np.where(row > 0.5)[0]
        cx         = int(clear_cols[len(clear_cols) // 2]) if len(clear_cols) else mid
        cy         = mid
        m[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 0.0

    elif defect_type == "Bridge":
        # Horizontal band connecting two clear regions
        half_w = min(r * 4, n // 3)
        half_h = max(1, r // 3)
        cx, cy = mid, mid
        m[mid - half_h: mid + half_h, mid - half_w: mid + half_w] = 1.0

    elif defect_type == "Break":
        # Vertical cut through a clear feature
        row        = m[mid, :]
        clear_cols = np.where(row > 0.5)[0]
        cx         = int(clear_cols[len(clear_cols) // 2]) if len(clear_cols) else mid
        cy         = mid
        half_h     = min(r * 3, n // 4)
        half_w     = max(1, r // 2)
        m[mid - half_h: mid + half_h, cx - half_w: cx + half_w] = 0.0

    elif defect_type == "Line Edge Roughness":
        from scipy.ndimage import binary_dilation, binary_erosion
        binary  = m > 0.5
        dilated = binary_dilation(binary, iterations=max(1, r))
        eroded  = binary_erosion(binary,  iterations=max(1, r))
        edge    = dilated ^ eroded
        noise   = rng.random(m.shape) < 0.5
        m[edge &  noise] = 1.0
        m[edge & ~noise] = 0.0
        cx, cy = mid, mid

    return np.clip(m, 0.0, 1.0), cx, cy, r


def detect_defects(
    clean_resist: np.ndarray,
    defect_resist: np.ndarray,
    pixel_size_m: float,
) -> dict:
    """
    Detect printed defects by differencing defective vs. clean resist images.

    A pixel is flagged as a printed defect if its resist state differs
    by more than 0.5 between the clean and defect simulations.

    Parameters
    ----------
    clean_resist  : 2D resist pattern without defect
    defect_resist : 2D resist pattern with defect
    pixel_size_m  : pixel pitch in metres

    Returns
    -------
    dict with keys: defect_map, n_defects, defect_areas_nm2,
                    total_area_nm2, printable
    """
    from scipy import ndimage

    diff_map       = np.abs(defect_resist.astype(float) - clean_resist.astype(float)) > 0.5
    labeled, count = ndimage.label(diff_map)
    px_nm          = pixel_size_m * 1e9

    if count == 0:
        return {
            "defect_map":       diff_map,
            "n_defects":        0,
            "defect_areas_nm2": [],
            "total_area_nm2":   0.0,
            "printable":        False,
        }

    areas_px  = ndimage.sum(diff_map, labeled, range(1, count + 1))
    areas_nm2 = [float(a) * px_nm ** 2 for a in areas_px]

    return {
        "defect_map":       diff_map,
        "labeled":          labeled,
        "n_defects":        count,
        "defect_areas_nm2": areas_nm2,
        "total_area_nm2":   sum(areas_nm2),
        "printable":        count > 0,
    }


# =============================================================================
# MASK BUILDER (shared between run_simulation and run_bossung_sweep)
# =============================================================================

def _build_mask(
    mask_type: str,
    n_pixels: int,
    pixel_size: float,
    pitch_m: float,
    cd_m: float,
) -> np.ndarray:
    """
    Return a 2D binary mask array for the requested pattern type.

    Extracted as a standalone function so that both run_simulation and
    run_bossung_sweep can call it without code duplication.
    """
    builders = {
        "Wordlines":       lambda: lines_and_spaces(n_pixels, pixel_size,
                                                     pitch=pitch_m, cd=cd_m),
        "Contact Array":   lambda: contact_array(n_pixels, pixel_size,
                                                   pitch_m, pitch_m, cd_m, cd_m),
        "Horizontal Line": lambda: isolated_line(n_pixels, pixel_size,
                                                  cd=cd_m, orientation="horizontal"),
        "Vertical Line":   lambda: isolated_line(n_pixels, pixel_size,
                                                  cd=cd_m, orientation="vertical"),
        "SRAM":            lambda: checkerboard(n_pixels, pixel_size,
                                                 pitch=pitch_m, cd=cd_m),
    }
    if mask_type not in builders:
        raise ValueError(f"Unknown mask type: {mask_type!r}")
    return builders[mask_type]()


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

#: The one imaging grid this app simulates on.
PIXEL_SIZE_M = 4e-9


def _simulation_config(node, na, sigma_outer, sigma_inner, source_type,
                       defocus_nm, threshold, diffusion_nm, n_pixels,
                       zernike_coeffs, dose) -> SimulationConfig:
    """The optics/resist/grid/dose quartet one simulated condition needs.

    ``run_simulation`` and ``run_bossung_sweep`` used to assemble these four
    dataclasses by hand, in two drifting copies — the sweep's inside its
    dose × focus double loop.
    """
    optics = OpticsConfig(
        wavelength     = WAVELENGTHS_M[node],
        NA             = na,
        n_immersion    = NODE_N_FLUID[node],
        sigma_outer    = sigma_outer,
        sigma_inner    = sigma_inner,
        source_type    = source_type,
        defocus        = defocus_nm * 1e-9,
        zernike_coeffs = zernike_coeffs,
    )
    resist_cfg = ResistConfig(
        threshold       = threshold,
        diffusion_sigma = diffusion_nm * 1e-9,
    )
    grid = GridConfig(n_pixels=n_pixels, pixel_size=PIXEL_SIZE_M)
    return SimulationConfig(optics=optics, resist=resist_cfg, grid=grid,
                            dose=dose, name=node)


@st.cache_data(show_spinner=False)
def run_simulation(
    node: str,
    mask_type: str,
    pitch_nm: float,
    cd_nm: float,
    na: float,
    sigma_outer: float,
    sigma_inner: float,
    dose: float,
    defocus_nm: float,
    threshold: float,
    diffusion_nm: float,
    n_pixels: int,
    zernike_tuple: tuple,
    source_type: str      = "conventional",
    defect_type: str      = "None",
    defect_size_nm: float = 80.0,
    defect_seed: int      = 42,
) -> dict:
    """
    Full simulation pipeline: aerial image -> resist model -> CD/NILS.

    zernike_tuple is a tuple of (Noll_index, coefficient_waves) pairs.
    Using a tuple instead of a dict makes the argument hashable so that
    @st.cache_data can key the cache correctly.
    """
    pixel_size = PIXEL_SIZE_M
    cfg = _simulation_config(
        node, na, sigma_outer, sigma_inner, source_type,
        defocus_nm, threshold, diffusion_nm, n_pixels,
        dict(zernike_tuple), dose,
    )

    pitch_m = pitch_nm * 1e-9
    cd_m    = cd_nm    * 1e-9

    mask_clean   = _build_mask(mask_type, n_pixels, pixel_size, pitch_m, cd_m)
    aerial_clean = compute_aerial_image(mask_clean, cfg.optics, cfg.grid, dose=cfg.dose)
    resist_clean = _develop(aerial_clean, threshold, diffusion_nm, pixel_size)

    defect_info     = None
    defect_location = None

    if defect_type != "None":
        mask_defect, dcx, dcy, dr_px = inject_defect(
            mask_clean, defect_type, defect_size_nm, pixel_size, seed=defect_seed
        )
        aerial_defect = compute_aerial_image(mask_defect, cfg.optics, cfg.grid, dose=cfg.dose)
        resist_defect = _develop(aerial_defect, threshold, diffusion_nm, pixel_size)
        defect_info   = detect_defects(resist_clean, resist_defect, pixel_size)
        defect_location = (dcx, dcy, dr_px)

        mask       = mask_defect
        aerial     = aerial_defect
        resist_img = resist_defect
    else:
        mask       = mask_clean
        aerial     = aerial_clean
        resist_img = resist_clean

    # Use the axis with more intensity variation for the cross-section profile
    mid   = n_pixels // 2
    var_x = np.var(aerial[mid, :])
    var_y = np.var(aerial[:, mid])

    if var_y > var_x:
        profile_aerial = aerial[:, mid]
        profile_resist = resist_img[:, mid].astype(float)
        profile_axis   = "y"
    else:
        profile_aerial = aerial[mid, :]
        profile_resist = resist_img[mid, :].astype(float)
        profile_axis   = "x"

    cd_measured_nm = _measure_cd_nm(profile_resist, pixel_size)
    nils_val       = compute_nils(profile_aerial, pixel_size,
                                   threshold=threshold, nominal_cd=cd_m)
    contrast       = float(
        (aerial.max() - aerial.min()) / (aerial.max() + aerial.min() + 1e-9)
    )

    return {
        "mask":            mask,
        "aerial":          aerial,
        "resist":          resist_img,
        "defect_info":     defect_info,
        "defect_type":     defect_type,
        "defect_location": defect_location,
        "cd_measured_nm":  cd_measured_nm,
        "nils":            nils_val,
        "contrast":        contrast,
        "profile_aerial":  profile_aerial,
        "profile_resist":  profile_resist,
        "profile_axis":    profile_axis,
        "pixel_size":      pixel_size,
        "n_pixels":        n_pixels,
        # Config objects, so the 3-D profile tab can re-run the same exposure
        # depth-resolved without reconstructing the sidebar state.
        "optics":          cfg.optics,
        "resist_cfg":      cfg.resist,
        "grid":            cfg.grid,
        "dose":            dose,
    }


@st.cache_data(show_spinner=False)
def run_bossung_sweep(
    node: str,
    mask_type: str,
    pitch_nm: float,
    cd_nm: float,
    na: float,
    sigma_outer: float,
    sigma_inner: float,
    threshold: float,
    diffusion_nm: float,
    n_pixels: int,
    zernike_tuple: tuple,
    focus_range_nm_tuple: tuple,
    dose_range_tuple: tuple,
    source_type: str = "conventional",
) -> dict:
    """
    Sweep defocus and dose to compute Bossung curves.

    The mask is built once and reused across all (focus, dose) combinations
    since it does not depend on those parameters.

    Parameters
    ----------
    focus_range_nm_tuple : tuple of defocus values in nm
    dose_range_tuple     : tuple of dose multipliers (1.0 = nominal)

    Returns
    -------
    dict with cd_matrix (n_dose x n_focus), focus_nm, dose_range
    """
    zernike_coeffs = dict(zernike_tuple)
    focus_range_nm = list(focus_range_nm_tuple)
    dose_range     = list(dose_range_tuple)

    pixel_size = PIXEL_SIZE_M
    pitch_m    = pitch_nm * 1e-9
    cd_m       = cd_nm    * 1e-9

    mask = _build_mask(mask_type, n_pixels, pixel_size, pitch_m, cd_m)
    mid  = n_pixels // 2

    n_d       = len(dose_range)
    n_f       = len(focus_range_nm)
    cd_matrix = np.zeros((n_d, n_f))

    for i_d, dose in enumerate(dose_range):
        for i_f, f_nm in enumerate(focus_range_nm):
            cfg = _simulation_config(
                node, na, sigma_outer, sigma_inner, source_type,
                f_nm, threshold, diffusion_nm, n_pixels,
                zernike_coeffs, dose,
            )
            aerial    = compute_aerial_image(mask, cfg.optics, cfg.grid, dose=cfg.dose)
            resist    = _develop(aerial, threshold, diffusion_nm, pixel_size)

            var_x     = np.var(aerial[mid, :])
            var_y     = np.var(aerial[:, mid])
            profile_r = resist[:, mid] if var_y > var_x else resist[mid, :]

            cd_matrix[i_d, i_f] = _measure_cd_nm(profile_r, pixel_size)

    return {
        "cd_matrix": cd_matrix,
        "focus_nm":  focus_range_nm,
        "dose_range": dose_range,
    }


def _longest_true_run(flags: np.ndarray) -> tuple[int, int] | None:
    """
    First and last index of the longest contiguous True run, or None if there
    is none.

    A process window has to be a single connected band: two in-spec islands
    either side of an out-of-spec gap are not a window you can centre on.
    """
    best     = None
    best_len = 0
    start    = None

    for i, flag in enumerate(flags):
        if flag:
            if start is None:
                start = i
        elif start is not None:
            if i - start > best_len:
                best_len, best = i - start, (start, i - 1)
            start = None

    if start is not None and len(flags) - start > best_len:
        best = (start, len(flags) - 1)

    return best


def compute_process_window(
    bossung: dict,
    cd_target_nm: float,
    tolerance: float = 0.10,
) -> dict:
    """
    Rectangular process window: the exposure-latitude / depth-of-focus tradeoff.

    A (dose, focus) point is in spec when its printed CD lands within
    +-tolerance of the target.  For every contiguous focus window we take the
    doses that stay in spec right across that window, keep the widest contiguous
    dose band among them, and express it as an exposure latitude

        EL% = (dose_max - dose_min) / dose_centre * 100

    Sweeping the window width traces the familiar falling EL-vs-DOF curve: a
    wider focus window can only ever be held with an equal or narrower dose
    band, so depth of focus is bought with exposure latitude.  EL is a width,
    so it is never negative.

    Parameters
    ----------
    bossung      : dict from run_bossung_sweep
    cd_target_nm : nominal target CD in nm
    tolerance    : fractional CD tolerance (0.10 = +-10%)

    Returns
    -------
    dict with the EL-DOF curve (dof_nm / el_pct), the per-dose focus range
    (dose_dof_nm), and the summary points of the curve
    """
    cd_matrix  = np.asarray(bossung["cd_matrix"],  dtype=float)
    focus_nm   = np.asarray(bossung["focus_nm"],   dtype=float)
    dose_range = np.asarray(bossung["dose_range"], dtype=float)

    # The window logic reads neighbouring indices as neighbouring conditions,
    # so sort rather than trust the caller to have swept in ascending order.
    f_order    = np.argsort(focus_nm)
    d_order    = np.argsort(dose_range)
    focus_nm   = focus_nm[f_order]
    dose_range = dose_range[d_order]
    cd_matrix  = cd_matrix[np.ix_(d_order, f_order)]

    cd_lo = cd_target_nm * (1.0 - tolerance)
    cd_hi = cd_target_nm * (1.0 + tolerance)

    in_spec         = (cd_matrix >= cd_lo) & (cd_matrix <= cd_hi) & (cd_matrix > 0.0)
    n_dose, n_focus = in_spec.shape

    # Widest dose band that survives each contiguous focus window.  Several
    # windows share a width, so keep the best latitude found at each one.
    el_at_dof: dict[float, float] = {}

    for i0 in range(n_focus):
        for i1 in range(i0, n_focus):
            held = in_spec[:, i0:i1 + 1].all(axis=1)
            band = _longest_true_run(held)
            if band is None:
                continue

            d_lo, d_hi = dose_range[band[0]], dose_range[band[1]]
            centre     = 0.5 * (d_lo + d_hi)
            el         = float((d_hi - d_lo) / centre * 100.0) if centre > 0.0 else 0.0
            dof        = float(focus_nm[i1] - focus_nm[i0])

            if el > el_at_dof.get(dof, -1.0):
                el_at_dof[dof] = el

    dof_list = sorted(el_at_dof)
    el_list  = [el_at_dof[d] for d in dof_list]

    # Focus range each individual dose holds on its own -- the per-row view
    # that pairs with the Bossung spec band.
    dose_dof = []
    for i_d in range(n_dose):
        run = _longest_true_run(in_spec[i_d])
        dose_dof.append(float(focus_nm[run[1]] - focus_nm[run[0]]) if run else 0.0)

    return {
        "dof_nm":       dof_list,
        "el_pct":       el_list,
        "dose_dof_nm":  dose_dof,
        "dose_range":   [float(d) for d in dose_range],
        "best_dof_nm":  dof_list[-1] if dof_list else 0.0,
        "best_el_pct":  el_list[-1]  if el_list  else 0.0,
        "max_el_pct":   max(el_list) if el_list  else 0.0,
        "cd_target":    cd_target_nm,
        "cd_lo":        cd_lo,
        "cd_hi":        cd_hi,
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def _dark_axes(axes):
    """Apply consistent dark-theme styling to a collection of Axes."""
    bg = "#0e1117"
    for ax in np.array(axes).flat:
        ax.set_facecolor(bg)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")


def make_figure(r: dict, threshold: float) -> plt.Figure:
    """Render the 2x3 main results figure (mask, aerial, resist + profiles)."""
    n    = r["n_pixels"]
    px   = r["pixel_size"]
    ext  = [0, n * px * 1e9, 0, n * px * 1e9]
    x    = np.arange(n) * px * 1e9
    axis = r["profile_axis"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("#0e1117")
    _dark_axes(axes)

    imshow_kw = dict(origin="lower", extent=ext, aspect="auto")

    # Row 0 -- 2D images -------------------------------------------------------

    axes[0, 0].imshow(r["mask"], cmap="gray", vmin=0, vmax=1, **imshow_kw)
    axes[0, 0].set_title("Mask Pattern")
    axes[0, 0].set_xlabel("x [nm]")
    axes[0, 0].set_ylabel("y [nm]")

    im1 = axes[0, 1].imshow(r["aerial"], cmap="inferno", vmin=0, vmax=1, **imshow_kw)
    cb1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="white")
    axes[0, 1].set_xlabel("x [nm]")

    aerial_title = "Aerial Image - Intensity"
    dloc = r.get("defect_location")
    if dloc is not None:
        dcx, dcy, dr_px = dloc
        if dr_px > 0:
            px_nm  = r["pixel_size"] * 1e9
            cx_nm  = dcx * px_nm
            cy_nm  = dcy * px_nm
            dr_nm  = dr_px * px_nm
            circ1  = plt.Circle(
                (cx_nm, cy_nm), dr_nm,
                fill=False, color="yellow", lw=1.5, ls="--", alpha=0.9,
            )
            axes[0, 1].add_patch(circ1)
            aerial_title = "Aerial Image  (yellow = defect site)"
    axes[0, 1].set_title(aerial_title)

    im2 = axes[0, 2].imshow(r["resist"], cmap="RdYlGn", vmin=0, vmax=1, **imshow_kw)
    cb2 = fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="white")
    axes[0, 2].set_xlabel("x [nm]")

    di = r.get("defect_info")
    if di and di["n_defects"] > 0:
        d_rgba = np.zeros((*di["defect_map"].shape, 4), dtype=float)
        d_rgba[..., 0] = 1.0
        d_rgba[..., 3] = di["defect_map"].astype(float) * 0.75
        axes[0, 2].imshow(d_rgba, origin="lower", extent=ext, aspect="auto")
        axes[0, 2].set_title(f"Resist Pattern  ({di['n_defects']} defect(s) -- red)")
    else:
        axes[0, 2].set_title("Resist Pattern  (green = unexposed)")

    if dloc is not None and dloc[2] > 0:
        px_nm  = r["pixel_size"] * 1e9
        circ2  = plt.Circle(
            (dloc[0] * px_nm, dloc[1] * px_nm), dloc[2] * px_nm,
            fill=False, color="yellow", lw=1.5, ls="--", alpha=0.9,
        )
        axes[0, 2].add_patch(circ2)

    # Row 1 -- 1D cross-section profiles ---------------------------------------

    axes[1, 0].plot(x, r["profile_aerial"], color="#ff6b35", lw=2, label="Intensity")
    axes[1, 0].axhline(
        threshold, color="cyan", lw=1.5, ls="--",
        label=f"Threshold = {threshold:.2f}",
    )
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].set_xlabel(f"{axis} [nm]")
    axes[1, 0].set_ylabel("Intensity")
    axes[1, 0].set_title(f"Aerial Image - Cross-Section ({axis} = centre)")
    axes[1, 0].legend(facecolor="#1e1e2e", labelcolor="white", framealpha=0.8)
    axes[1, 0].grid(True, alpha=0.2)

    axes[1, 1].fill_between(x, r["profile_resist"], alpha=0.4, color="#4ecdc4")
    axes[1, 1].plot(x, r["profile_resist"], color="#4ecdc4", lw=2)
    axes[1, 1].set_ylim(-0.05, 1.15)
    axes[1, 1].set_xlabel(f"{axis} [nm]")
    axes[1, 1].set_ylabel("Unexposed (1) / Cleared (0)")
    axes[1, 1].set_title(f"Resist Profile - Cross-Section ({axis} = centre)")
    axes[1, 1].grid(True, alpha=0.2)

    axes[1, 2].axis("off")
    nils_ok     = r["nils"] >= 2.0
    nils_icon   = "[PASS]" if nils_ok else "[FAIL]"
    metrics_str = (
        f"SIMULATION METRICS\n"
        f"{'-' * 30}\n\n"
        f"  Printed CD      {r['cd_measured_nm']:.1f} nm\n\n"
        f"  NILS            {r['nils']:.2f}  {nils_icon}\n"
        f"  (>= 2.0 = good printability)\n\n"
        f"  Image Contrast  {r['contrast']:.3f}\n\n"
        f"  Max Intensity   {r['aerial'].max():.3f}\n"
        f"  Min Intensity   {r['aerial'].min():.3f}"
    )
    axes[1, 2].text(
        0.05, 0.95, metrics_str,
        transform=axes[1, 2].transAxes,
        fontsize=11, va="top", fontfamily="monospace", color="white",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e1e2e",
                  edgecolor="#555555", alpha=0.95),
    )

    fig.tight_layout(pad=2.0)
    return fig


def make_bossung_figure(
    bossung: dict,
    cd_target_nm: float,
    tolerance: float,
) -> plt.Figure:
    """
    Plot Bossung curves: printed CD vs. defocus for each dose level.

    The cyan horizontal band marks the +-tolerance CD specification window.
    The width of this band at each dose level corresponds to the DOF.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0e1117")
    _dark_axes([ax])

    focus_nm  = np.array(bossung["focus_nm"])
    cd_matrix = bossung["cd_matrix"]
    doses     = bossung["dose_range"]

    colors = plt.cm.plasma(np.linspace(0.15, 0.90, len(doses)))

    for i_d, dose in enumerate(doses):
        ax.plot(
            focus_nm, cd_matrix[i_d, :],
            "o-", color=colors[i_d], lw=2, markersize=5,
            label=f"Dose {dose:.2f}x",
        )

    cd_lo = cd_target_nm * (1.0 - tolerance)
    cd_hi = cd_target_nm * (1.0 + tolerance)

    ax.axhline(cd_target_nm, color="cyan", ls="-",  lw=1.5, alpha=0.8,
               label=f"Target {cd_target_nm:.0f} nm")
    ax.axhspan(cd_lo, cd_hi, alpha=0.10, color="cyan",
               label=f"+-{tolerance * 100:.0f}% band")

    ax.set_xlabel("Defocus [nm]")
    ax.set_ylabel("Printed CD [nm]")
    ax.set_title("Bossung Curves")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", framealpha=0.8, fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def make_process_window_figure(pw: dict) -> plt.Figure:
    """
    Plot the process window: Exposure Latitude vs. Depth of Focus.

    Each point is one focus-window width, carrying the widest dose band that
    holds CD in spec right across it.  The curve falls: focus is bought with
    latitude.  Both end points are labelled.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0e1117")
    _dark_axes([ax])

    dof = np.array(pw["dof_nm"])
    el  = np.array(pw["el_pct"])

    if dof.size:
        ax.plot(dof, el, "-", color="#c8913a", lw=1.5, alpha=0.6)
        ax.scatter(dof, el, color="#c8913a", s=80, zorder=5)

        # The curve only ever falls, so the upper-right corner is free whatever
        # shape it takes -- put the end points there rather than chase the line
        # with leader lines that end up crossing it.
        ax.text(
            0.97, 0.96,
            f"Max EL : {el[0]:5.1f}%  @ DOF {dof[0]:.0f} nm\n"
            f"Max DOF: {dof[-1]:5.0f} nm @ EL  {el[-1]:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            color="white", fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e2e",
                      edgecolor="#555555", alpha=0.85),
        )
        ax.margins(x=0.08, y=0.14)
    else:
        ax.text(
            0.5, 0.5,
            "No in-spec points found.\nRelax CD tolerance or adjust parameters.",
            ha="center", va="center", transform=ax.transAxes,
            color="#888888", fontsize=9, fontfamily="monospace",
        )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("DOF [nm]")
    ax.set_ylabel("Exposure Latitude [%]")
    ax.set_title("Process Window  (EL vs. DOF)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def make_pupil_figure(
    pupil: np.ndarray,
    W_masked: np.ndarray,
    rms: float,
) -> plt.Figure:
    """
    Plot pupil amplitude and wavefront phase maps side by side.

    Left  : |P(rho, theta)| -- amplitude showing the NA aperture stop
    Right : W(rho, theta) in waves -- wavefront showing aberration shape

    The dashed circle marks the NA boundary (rho = 1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#0e1117")
    _dark_axes(axes)

    extent = [-1, 1, -1, 1]

    # Amplitude map
    im0 = axes[0].imshow(
        np.abs(pupil), cmap="gray", vmin=0, vmax=1,
        origin="lower", extent=extent,
    )
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(colors="white")
    cb0.set_label("Amplitude", color="white")
    axes[0].set_title("Pupil Amplitude")
    axes[0].set_xlabel("Normalised fx")
    axes[0].set_ylabel("Normalised fy")
    axes[0].add_patch(
        plt.Circle((0, 0), 1.0, fill=False, color="#c8913a", lw=1.5, ls="--", alpha=0.9)
    )

    # Wavefront map (NaN outside pupil renders as background colour)
    vmax = float(np.nanmax(np.abs(W_masked)))
    vmax = max(vmax, 1e-4)

    im1 = axes[1].imshow(
        W_masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        origin="lower", extent=extent,
    )
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="white")
    cb1.set_label("Wavefront [waves]", color="white")
    axes[1].set_title(f"Wavefront Map  (RMS = {rms:.4f} waves)")
    axes[1].set_xlabel("Normalised fx")
    axes[1].set_ylabel("Normalised fy")
    axes[1].add_patch(
        plt.Circle((0, 0), 1.0, fill=False, color="white", lw=1.0, alpha=0.5)
    )

    fig.tight_layout()
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Parameters")

    with st.form("sim_form"):

        st.subheader("Technology Node")
        node = st.selectbox(
            "Node", list(WAVELENGTHS_M.keys()), index=2,
            help="Selects default wavelength and NA for the chosen process node.",
        )

        st.subheader("Mask")
        mask_type = st.selectbox("Pattern Type", MASK_TYPES)
        pitch_nm  = st.slider("Pitch [nm]",      50,  800, 400, step=10)
        cd_nm     = st.slider("Feature CD [nm]", 10,  400, 150, step=5)

        st.subheader("Optics")
        defaults    = NODE_DEFAULTS[node]
        na          = st.slider("NA",           0.10, 1.35,
                                 float(defaults["na"]),          step=0.01)
        source_type = st.selectbox(
            "Illumination", SOURCE_TYPES, index=0,
            help="Off-axis shapes (dipole, quasar, cquad) trade orientation "
                 "coverage for contrast at tight pitch.",
        )
        sigma_outer = st.slider("Sigma Outer",  0.10, 1.00,
                                 float(defaults["sigma_outer"]), step=0.05)
        sigma_inner = st.slider("Sigma Inner",  0.00, 0.90,
                                 float(defaults["sigma_inner"]), step=0.05,
                                 help="Inner coherence factor. Only used by the "
                                      "shapes with a hole in the middle — "
                                      "annular, dipole, quasar, cquad.")
        dose        = st.slider("Relative Dose", 0.50, 2.00, 1.00, step=0.05)
        defocus_nm  = st.slider("Defocus [nm]", -300, 300, 0, step=10)

        # -- Zernike aberration controls ---------------------------------------
        with st.expander("Zernike Aberrations"):
            st.caption(
                "Coefficients in waves (1 wave = lambda). "
                "Values above 0.05 waves are considered large. "
                "Z4 here adds to the Defocus slider above."
            )
            zernike_vals = {}
            for j in range(4, 22):
                label = ZERNIKE_NAMES.get(j, f"Z{j}")
                zernike_vals[j] = st.slider(
                    label,
                    min_value=-0.15,
                    max_value=0.15,
                    value=0.0,
                    step=0.005,
                    format="%.3f",
                    key=f"zernike_{j}",
                )

        st.subheader("Resist")
        threshold    = st.slider(
            "Exposure Threshold", 0.10, 0.90, 0.35, step=0.01,
            help="Intensity at which the resist clears.",
        )
        diffusion_nm = st.slider(
            "PEB Diffusion [nm]", 0, 60, 10, step=1,
            help="Gaussian diffusion length during Post-Exposure Bake.",
        )

        st.subheader("Grid")
        n_pixels = st.select_slider(
            "Resolution (pixels)",
            options=[64, 128, 256],
            value=128,
            help="64 = fast (~2 s)  |  128 = balanced (~10 s)  |  256 = detailed (~60 s)",
        )

        st.subheader("Defect Injection")
        defect_type    = st.selectbox("Defect Type", DEFECT_TYPES, index=0)
        defect_size_nm = st.slider(
            "Defect Radius [nm]", 10, 200, 80, step=5,
            help="Defect radius in nm.",
        )
        defect_seed = st.number_input(
            "Random Seed", value=42, step=1,
            help="Controls defect placement. Change to move the defect.",
        )

        submitted = st.form_submit_button(
            "Run Simulation", use_container_width=True, type="primary",
        )


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

param_errors = []

if cd_nm >= pitch_nm:
    param_errors.append(
        f"CD ({cd_nm} nm) must be smaller than Pitch ({pitch_nm} nm). "
        f"Try CD <= {int(pitch_nm * 0.45)} nm."
    )

wl_nm     = WAVELENGTHS_M[node] * 1e9
min_pitch = wl_nm / na

if pitch_nm < min_pitch:
    param_errors.append(
        f"Pitch ({pitch_nm} nm) is below the resolution limit. "
        f"For {node} (lambda={wl_nm:.0f} nm, NA={na:.2f}), "
        f"minimum resolvable pitch is approx. {min_pitch:.0f} nm."
    )

if sigma_inner >= sigma_outer and source_type != "conventional":
    param_errors.append(
        f"Sigma inner ({sigma_inner}) must be less than sigma outer ({sigma_outer})."
    )

if param_errors:
    for msg in param_errors:
        st.error(msg)
    st.stop()

if defect_type != "None":
    res_limit_nm   = 0.5 * wl_nm / na
    defect_diam_nm = defect_size_nm * 2
    if defect_diam_nm < res_limit_nm:
        st.warning(
            f"Defect diameter ({defect_diam_nm:.0f} nm) is below the estimated "
            f"resolution limit ({res_limit_nm:.0f} nm for {node}). "
            f"The defect may not print. "
            f"Try radius >= {int(res_limit_nm / 2) + 5} nm."
        )


# =============================================================================
# RUN MAIN SIMULATION
# =============================================================================

# Convert zernike_vals dict to a sorted tuple so @st.cache_data can hash it
zernike_tuple = tuple(sorted(zernike_vals.items()))

if submitted or "results" not in st.session_state:
    with st.spinner("Computing... (64px ~2 s  |  128px ~10 s  |  256px ~60 s)"):
        try:
            results = run_simulation(
                node           = node,
                mask_type      = mask_type,
                pitch_nm       = pitch_nm,
                cd_nm          = cd_nm,
                na             = na,
                sigma_outer    = sigma_outer,
                sigma_inner    = sigma_inner,
                source_type    = source_type,
                dose           = dose,
                defocus_nm     = float(defocus_nm),
                threshold      = threshold,
                diffusion_nm   = diffusion_nm,
                n_pixels       = n_pixels,
                zernike_tuple  = zernike_tuple,
                defect_type    = defect_type,
                defect_size_nm = float(defect_size_nm),
                defect_seed    = int(defect_seed),
            )
            st.session_state["results"]   = results
            st.session_state["threshold"] = threshold

            # Store last submitted parameters for Bossung sweep
            st.session_state["last_params"] = dict(
                node         = node,
                mask_type    = mask_type,
                pitch_nm     = pitch_nm,
                cd_nm        = cd_nm,
                na           = na,
                sigma_outer  = sigma_outer,
                sigma_inner  = sigma_inner,
                source_type  = source_type,
                threshold    = threshold,
                diffusion_nm = diffusion_nm,
                n_pixels     = n_pixels,
                zernike_tuple = zernike_tuple,
            )

            # Clear stale Bossung data when simulation parameters change
            st.session_state.pop("bossung_results", None)
            st.session_state.pop("pw_results", None)

        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.stop()


# =============================================================================
# MAIN AREA -- THREE TABS
# =============================================================================

tab_results, tab_bossung, tab_pupil, tab_3d = st.tabs([
    "Results",
    "Bossung and Process Window",
    "Pupil Viewer",
    "3-D Resist Profile",
])


# ---------------------------------------------------------------------------
# TAB 1 -- RESULTS
# ---------------------------------------------------------------------------

with tab_results:
    if "results" in st.session_state:
        r   = st.session_state["results"]
        thr = st.session_state.get("threshold", 0.35)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Printed CD",     f"{r['cd_measured_nm']:.1f} nm")
        c2.metric(
            "NILS", f"{r['nils']:.2f}",
            delta="Good" if r["nils"] >= 2.0 else "Poor",
            delta_color="normal" if r["nils"] >= 2.0 else "inverse",
        )
        c3.metric("Image Contrast", f"{r['contrast']:.3f}")
        c4.metric("Max Intensity",  f"{r['aerial'].max():.3f}")

        st.divider()

        fig = make_figure(r, thr)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.divider()
        st.subheader("Export")

        n_px = r["n_pixels"]
        px   = r["pixel_size"]
        df   = pd.DataFrame({
            "position_nm":      np.arange(n_px) * px * 1e9,
            "aerial_intensity": r["profile_aerial"],
            "resist_state":     r["profile_resist"],
        })

        col_a, col_b = st.columns(2)
        col_a.download_button(
            "Download Cross-Section CSV",
            data=df.to_csv(index=False),
            file_name="litho_cross_section.csv",
            mime="text/csv",
            use_container_width=True,
        )
        col_b.download_button(
            "Download Aerial Image (NPY)",
            data=r["aerial"].tobytes(),
            file_name="aerial_image.npy",
            mime="application/octet-stream",
            use_container_width=True,
        )

        # Defect analysis report
        di = r.get("defect_info")
        if di is not None:
            st.divider()
            st.subheader("Defect Analysis Report")

            if not di["printable"]:
                st.success(
                    f"Defect type '{r['defect_type']}' does NOT print under current "
                    f"process conditions. The mask defect is sub-resolution."
                )
            else:
                st.error(f"{di['n_defects']} defect region(s) printed into the resist.")
                d1, d2, d3 = st.columns(3)
                d1.metric("Printed Defect Regions", str(di["n_defects"]))
                d2.metric("Total Defect Area",      f"{di['total_area_nm2']:.0f} nm sq")
                d3.metric("Largest Defect",
                           f"{max(di['defect_areas_nm2']):.0f} nm sq")

                df_d = pd.DataFrame({
                    "Defect Region":  range(1, di["n_defects"] + 1),
                    "Area (nm sq)":   [f"{a:.0f}" for a in di["defect_areas_nm2"]],
                    "Est. Size (nm)": [f"{a ** 0.5:.1f}" for a in di["defect_areas_nm2"]],
                })
                st.dataframe(df_d, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 2 -- BOSSUNG AND PROCESS WINDOW
# ---------------------------------------------------------------------------

with tab_bossung:
    st.markdown("""
<h3>Bossung Curve Analysis</h3>
<p style="color:#666666; font-family:'Courier New',monospace; font-size:0.82rem;">
    Sweep defocus and dose to map the process window.
    Each curve shows printed CD vs. defocus at a fixed dose level.
    The shaded band marks the CD specification (target +/- tolerance).
    Uses the parameter values from the last Run Simulation.
</p>
""", unsafe_allow_html=True)

    with st.expander("Sweep Settings", expanded=True):
        bc1, bc2, bc3 = st.columns(3)

        with bc1:
            focus_min   = st.number_input("Focus Min [nm]", -800,    0, -400, step=50)
            focus_max   = st.number_input("Focus Max [nm]",    0,  800,  400, step=50)
            focus_steps = st.slider("Focus Steps", 3, 15, 9, step=2)

        with bc2:
            dose_min   = st.number_input("Dose Min (x nominal)", 0.50, 1.00, 0.80, step=0.05)
            dose_max   = st.number_input("Dose Max (x nominal)", 1.00, 2.00, 1.20, step=0.05)
            dose_steps = st.slider("Dose Steps", 3, 9, 5, step=2)

        with bc3:
            cd_target_bossung = st.number_input(
                "Target CD [nm]",
                min_value=1.0, max_value=1000.0,
                value=float(cd_nm), step=1.0,
                help="Nominal CD used to draw the spec band and compute DOF.",
            )
            pw_tolerance = st.slider(
                "CD Tolerance [%]", 5, 25, 10, step=1,
                help="Acceptable CD variation (10 = +-10%).",
            ) / 100.0

        lp        = st.session_state.get("last_params", {})
        n_px_b    = lp.get("n_pixels", n_pixels)
        n_total   = focus_steps * dose_steps
        t_per_run = 0.5 if n_px_b == 64 else 5 if n_px_b == 128 else 30
        st.caption(
            f"Total simulation runs: {n_total}  x  {n_px_b}px grid.  "
            f"Estimated time: {n_total * t_per_run:.0f} s"
        )

    run_bossung_btn = st.button(
        "Run Bossung Analysis",
        use_container_width=True,
        help="Sweeps the focus and dose ranges using the last submitted parameters.",
    )

    if run_bossung_btn:
        lp = st.session_state.get("last_params", {})
        if not lp:
            st.error("Run the main simulation first (click Run Simulation in the sidebar).")
        else:
            focus_range_tup = tuple(
                float(v) for v in np.linspace(focus_min, focus_max, focus_steps)
            )
            dose_range_tup = tuple(
                float(v) for v in np.linspace(dose_min, dose_max, dose_steps)
            )

            with st.spinner(f"Running {len(focus_range_tup) * len(dose_range_tup)} simulations..."):
                try:
                    bossung = run_bossung_sweep(
                        node                 = lp["node"],
                        mask_type            = lp["mask_type"],
                        pitch_nm             = lp["pitch_nm"],
                        cd_nm                = lp["cd_nm"],
                        na                   = lp["na"],
                        sigma_outer          = lp["sigma_outer"],
                        sigma_inner          = lp["sigma_inner"],
                        threshold            = lp["threshold"],
                        diffusion_nm         = lp["diffusion_nm"],
                        n_pixels             = lp["n_pixels"],
                        zernike_tuple        = lp["zernike_tuple"],
                        focus_range_nm_tuple = focus_range_tup,
                        dose_range_tuple     = dose_range_tup,
                        source_type          = lp.get("source_type", "conventional"),
                    )
                    pw = compute_process_window(bossung, cd_target_bossung, pw_tolerance)
                    st.session_state["bossung_results"]   = bossung
                    st.session_state["pw_results"]        = pw
                    st.session_state["bossung_cd_target"] = cd_target_bossung
                    st.session_state["bossung_tolerance"] = pw_tolerance
                except Exception as exc:
                    st.error(f"Bossung sweep failed: {exc}")

    if "bossung_results" in st.session_state:
        bossung = st.session_state["bossung_results"]
        pw      = st.session_state["pw_results"]
        cd_t    = st.session_state.get("bossung_cd_target", float(cd_nm))
        tol     = st.session_state.get("bossung_tolerance", 0.10)

        col_b1, col_b2 = st.columns([3, 2])

        with col_b1:
            fig_b = make_bossung_figure(bossung, cd_t, tol)
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)

        with col_b2:
            fig_pw = make_process_window_figure(pw)
            st.pyplot(fig_pw, use_container_width=True)
            plt.close(fig_pw)

        st.divider()
        st.subheader("Process Window Summary")

        rayleigh_dof = 0.5 * wl_nm / (na ** 2)
        m1, m2, m3 = st.columns(3)
        m1.metric("Max EL",       f"{pw['max_el_pct']:.1f}%",
                  help="Widest dose band at best focus -- the left end of the curve.")
        m2.metric("Best DOF",     f"{pw['best_dof_nm']:.0f} nm",
                  help="Widest focus window that still holds CD in spec at some dose.")
        m3.metric("Rayleigh DOF", f"+/-{rayleigh_dof:.0f} nm")

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.caption("Exposure latitude vs. depth of focus")
            st.dataframe(
                pd.DataFrame({
                    "DOF [nm]":            [f"{d:.0f}" for d in pw["dof_nm"]],
                    "Exposure Latitude %": [f"{e:.1f}" for e in pw["el_pct"]],
                }),
                hide_index=True, use_container_width=True,
            )

        with col_t2:
            st.caption("Focus range held by each dose on its own")
            st.dataframe(
                pd.DataFrame({
                    "Dose (x nominal)": [f"{d:.2f}" for d in pw["dose_range"]],
                    "DOF [nm]":         [f"{d:.0f}" for d in pw["dose_dof_nm"]],
                }),
                hide_index=True, use_container_width=True,
            )

        # Export Bossung data
        rows = [
            {
                "dose_x_nominal": dose,
                "defocus_nm":     f_nm,
                "cd_nm":          bossung["cd_matrix"][i_d, i_f],
            }
            for i_d, dose in enumerate(bossung["dose_range"])
            for i_f, f_nm in enumerate(bossung["focus_nm"])
        ]
        st.download_button(
            "Download Bossung Data CSV",
            data=pd.DataFrame(rows).to_csv(index=False),
            file_name="bossung_curves.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Configure sweep settings above and click Run Bossung Analysis.")


# ---------------------------------------------------------------------------
# TAB 3 -- PUPIL VIEWER
# ---------------------------------------------------------------------------

with tab_pupil:
    st.markdown("""
<h3>Pupil Function Viewer</h3>
<p style="color:#666666; font-family:'Courier New',monospace; font-size:0.82rem;">
    Visualises the complex pupil P(rho, theta) = A(rho) * exp(2*pi*i*W).
    Left: amplitude (shows the NA aperture).
    Right: wavefront W in waves (shows Zernike aberrations plus defocus OPD).
    The dashed circle is the NA boundary at rho = 1.
    Updates automatically from the current sidebar slider values --
    no need to click Run Simulation.
</p>
""", unsafe_allow_html=True)

    n_fluid     = NODE_N_FLUID.get(node, 1.0)
    wl_nm_local = WAVELENGTHS_M[node] * 1e9

    try:
        pupil, W_masked, inside, rms = build_pupil_display(
            N                    = 256,
            zernike_coeffs_waves = zernike_vals,
            defocus_nm           = float(defocus_nm),
            NA                   = float(na),
            wavelength_nm        = wl_nm_local,
            n_fluid              = n_fluid,
        )

        # Strehl ratio via Marechal approximation: S = exp(-(2*pi*sigma)^2)
        # Valid for small aberrations (sigma < 0.1 waves)
        strehl       = float(np.exp(-(2.0 * math.pi * rms) ** 2))
        active_terms = sum(1 for v in zernike_vals.values() if abs(v) > 1e-4)

        p1, p2, p3 = st.columns(3)
        p1.metric(
            "RMS Wavefront Error",
            f"{rms:.4f} waves",
            help="Marechal criterion: below 0.07 waves for diffraction-limited imaging.",
        )
        p2.metric(
            "Strehl Ratio (Marechal approx.)",
            f"{strehl:.4f}",
            delta="Diffraction-limited" if strehl >= 0.80 else "Aberrated",
            delta_color="normal" if strehl >= 0.80 else "inverse",
            help="S = exp(-(2*pi*sigma)^2). Valid for sigma below ~0.1 waves.",
        )
        p3.metric(
            "Active Zernike Terms",
            str(active_terms),
            help="Number of terms with |coeff| > 0.0001 waves.",
        )

        st.divider()

        fig_pup = make_pupil_figure(pupil, W_masked, rms)
        st.pyplot(fig_pup, use_container_width=True)
        plt.close(fig_pup)

        st.divider()
        st.subheader("Zernike Coefficient Summary")

        rows_z = [
            {
                "Noll Index":    j,
                "Name":          ZERNIKE_NAMES.get(j, f"Z{j}"),
                "Coeff (waves)": f"{zernike_vals.get(j, 0.0):+.4f}",
                "OPD (nm)":      f"{zernike_vals.get(j, 0.0) * wl_nm_local:+.2f}",
                "Active":        "Yes" if abs(zernike_vals.get(j, 0.0)) > 1e-4 else "-",
            }
            for j in range(4, 22)
        ]
        st.dataframe(pd.DataFrame(rows_z), hide_index=True, use_container_width=True)

    except Exception as exc:
        st.error(f"Pupil viewer error: {exc}")


# ---------------------------------------------------------------------------
# TAB 4 -- 3-D RESIST PROFILE
# ---------------------------------------------------------------------------

with tab_3d:
    if "results" not in st.session_state:
        st.info("Run a simulation first (button in the sidebar) — the 3-D "
                "profile develops the same exposure through the film depth.")
    else:
        r = st.session_state["results"]

        st.markdown(
            "The depth-resolved pipeline: aerial image at every depth → Dill "
            "absorption and bleaching → optional substrate standing waves → "
            "3-D PEB → development. The 2-D Results tab shows the footprint; "
            "this shows the **solid**, with its sidewall angle and top loss."
        )

        cc1, cc2, cc3, cc4 = st.columns(4)
        thickness_nm = cc1.slider("Film thickness [nm]", 50, 300, 100, 10)
        develop_model = cc2.radio(
            "Develop model", ["threshold", "mack"],
            help="threshold: binary cut at the PAC threshold, plus a "
                 "reachability rule. mack: finite-rate ray develop — the one "
                 "that eats through slow standing-wave layers instead of "
                 "stopping at them.",
        )
        sw_on = cc3.checkbox(
            "Standing waves", value=False,
            help="Interfere the downward wave with its substrate reflection "
                 "(period λ/2n ≈ 57 nm for ArF). Needs a reflectance below.",
        )
        r_sub = cc4.slider(
            "Substrate reflectance", 0.0, 0.6, 0.35, 0.05,
            disabled=not sw_on,
            help="0 is a perfect BARC (no standing waves at all).",
        )

        if st.button("Compute 3-D profile", type="primary"):
            import dataclasses

            from litho_sim.develop.resist3d import print_resist_3d
            from litho_sim.viz.viz3d import resist_profile_3d_figure

            resist3d_cfg = dataclasses.replace(
                r["resist_cfg"],
                thickness=thickness_nm * 1e-9,
                substrate_reflectance=r_sub if sw_on else 0.0,
            )
            with st.spinner("Developing the film in 3-D…"):
                res3 = print_resist_3d(
                    r["mask"], r["optics"], r["grid"], resist3d_cfg,
                    dose=r["dose"],
                    standing_waves=sw_on,
                    develop_model=develop_model,
                )
            st.session_state["res3d"] = res3
            st.session_state["res3d_grid"] = r["grid"]
            st.session_state["res3d_label"] = (
                f"{develop_model} develop, "
                + (f"standing waves r={r_sub:.2f}" if sw_on else "no standing waves")
            )

        if "res3d" in st.session_state:
            from litho_sim.develop.resist3d import sidewall_angle
            from litho_sim.viz.viz3d import resist_profile_3d_figure

            res3 = st.session_state["res3d"]
            rem = res3["remaining"]
            g3 = st.session_state["res3d_grid"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Film remaining", f"{100 * float(rem.mean()):.1f}%")
            angle = sidewall_angle(rem, g3)
            m2.metric("Sidewall angle", "—" if not np.isfinite(angle) else f"{angle:.1f}°")
            occ = np.nonzero(rem.any(axis=(1, 2)))[0]
            top_loss = (
                (rem.shape[0] - 1 - int(occ.max())) * g3.dz * 1e9 if occ.size else float("nan")
            )
            m3.metric("Resist top loss", "—" if not np.isfinite(top_loss) else f"{top_loss:.1f} nm")

            fig3d = resist_profile_3d_figure(
                res3, g3, title=f"3-D resist profile — {st.session_state['res3d_label']}"
            )
            st.pyplot(fig3d, use_container_width=True)
            plt.close(fig3d)

            if float(rem.mean()) > 0.995:
                st.warning(
                    "The film did not develop: a standing-wave node at the top "
                    "of the film is insoluble under the threshold model, and "
                    "the developer cannot reach past it. This is the physical "
                    "reason BARCs and PEB exist — try the **mack** develop "
                    "model (finite rate), raise PEB diffusion, or set the "
                    "reflectance to 0."
                )

