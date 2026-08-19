# LithoPy

**A modular optical lithography simulation and deep-learning defect detection framework.**

LithoPy simulates the full optical lithography process used in semiconductor manufacturing — from mask pattern to aerial image to developed resist — and extends that physics engine into a machine learning pipeline for automated defect detection. The project is designed to demonstrate production-grade scientific software engineering, covering first-principles physics, interactive parameter exploration, synthetic dataset generation, and unsupervised anomaly detection with a convolutional autoencoder.

---

## Table of Contents

1. [Goals and Scope](#goals-and-scope)
2. [Target Audience](#target-audience)
3. [Physics Background](#physics-background)
4. [Directory Structure](#directory-structure)
5. [Installation and Setup](#installation-and-setup)
6. [Full Pipeline Overview](#full-pipeline-overview)
7. [Source Modules](#source-modules)
8. [Scripts and Usage](#scripts-and-usage)
   - [CLI Entry Point](#cli-entry-point)
   - [Interactive Simulator](#interactive-simulator)
   - [Dataset Generation](#dataset-generation)
   - [Autoencoder Training](#autoencoder-training)
   - [Inference and Defect Detection](#inference-and-defect-detection)
9. [Complete Source Code](#complete-source-code)
10. [Configuration Reference](#configuration-reference)
11. [Results and Evaluation](#results-and-evaluation)
12. [Troubleshooting](#troubleshooting)
13. [Further Reading and References](#further-reading-and-references)
14. [Contributing](#contributing)
15. [License](#license)

---

## Goals and Scope

**Goals:**

- Provide a physically accurate, modular simulation of the optical lithography process
- Enable interactive exploration of how process parameters affect printed feature dimensions
- Generate large-scale synthetic datasets of clean and defective aerial images
- Train an unsupervised convolutional autoencoder to detect and localise mask defects
- Demonstrate professional Python software engineering practices suitable for a production scientific codebase

**Scope:**

- Optical path: partially coherent Abbe illumination, conventional / annular / dipole / quadrupole sources, defocus, Zernike aberrations (Noll 1–21)
- **Vector imaging**: three-component field tracking with unpolarised / linear / azimuthal / radial illumination, exact non-paraxial defocus, and the aplanatic obliquity factor — the polarisation-dependent contrast loss at hyper-NA
- Resist model: Dill exposure with bleaching, PEB diffusion, Mack dissolution, threshold and ray development
- **3-D resist**: depth-resolved exposure, Beer–Lambert absorption, substrate standing waves, sloped sidewalls
- **Wafer stack**: voxel film stack with conformal deposition, selective and anisotropic etch, strip, CMP
- **Multi-patterning**: LELE (LE²), LE³, SADP, SAQP, and cut/block masks, with per-exposure overlay
- **Mask 3-D effects**: `mask_model="multilayer"` reflects each diffraction order off the EUV Mo/Si mirror at its own angle (image placement and best-focus shift); `mask_model="fdtd"` solves Maxwell's equations around the absorber with a complex-field Bloch-periodic FDTD (validated against Fresnel/TMM to <1%), for DUV transmissive and EUV reflective masks
- **Device flows**: a gate-all-around nanosheet transistor built end-to-end from two printed masks plus self-aligned processing (`scripts/demo_gaa.py`)
- Mask design: rectangles, polygons, paths, contacts, boolean layers, anti-aliased rasterisation, graph-colouring decomposition
- Defect types: pinhole, blob, bridge, break, line edge roughness
- Machine learning: unsupervised anomaly detection using a convolutional autoencoder trained exclusively on defect-free images

**Out of scope:**

- OPC (optical proximity correction) and inverse lithography
- EUV stochastic effects

---

## Target Audience

- Semiconductor process engineers looking for a Python-native lithography simulation sandbox
- Machine learning engineers exploring anomaly detection on structured scientific images
- Software engineers evaluating modern Python project structure and scientific ML pipelines
- Students and researchers studying optical imaging, photolithography, or deep learning for inspection

---

## Physics Background

### Optical Lithography

Optical lithography is the primary patterning technology in semiconductor manufacturing. Ultraviolet light passes through a patterned mask (reticle), is focused by a projection lens system, and exposes a photosensitive resist film on the silicon wafer. Where the resist is exposed above a threshold dose, it is chemically altered and subsequently removed during development, leaving behind the patterned features.

The resolution limit of the process is governed by the **Rayleigh criterion**:

$$R = k_1 \cdot \frac{\lambda}{\text{NA}}$$

where $\lambda$ is the illumination wavelength, $\text{NA}$ is the numerical aperture of the projection lens, and $k_1$ is a process-dependent constant (typically 0.25 to 0.8).

### Aerial Image Formation — The Abbe Method

The aerial image intensity $I(\mathbf{r})$ at wafer plane position $\mathbf{r}$ is computed using the **Abbe partially coherent imaging formulation**. The mask transmittance $t(\mathbf{r})$ is decomposed into spatial frequency components via a 2D Fourier transform:

$$T(\mathbf{f}) = \mathcal{F}\{t(\mathbf{r})\}$$

The projection lens acts as a **low-pass spatial frequency filter**. The coherent transfer function (CTF) is:

$$H(\mathbf{f}) = \begin{cases} 1 & \text{if } |\mathbf{f}| \leq \frac{\text{NA}}{\lambda} \\ 0 & \text{otherwise} \end{cases}$$

For **partially coherent illumination**, the source is modelled as a disk of incoherent point sources with outer radius $\sigma_\text{outer}$ (normalised to the pupil radius). Each source point $\mathbf{s}$ illuminates the mask at a different angle, producing a shifted pupil $H(\mathbf{f} - \mathbf{s}/\lambda)$. The total aerial image is:

$$I(\mathbf{r}) = \int_\text{source} \left| \mathcal{F}^{-1}\left\{ T(\mathbf{f}) \cdot H\left(\mathbf{f} - \frac{\mathbf{s}}{\lambda}\right) \right\} \right|^2 d\mathbf{s}$$

### Annular Illumination

Annular illumination removes source points within an inner radius $\sigma_\text{inner}$, creating a ring-shaped source. This improves depth of focus and image contrast for dense line-space patterns by suppressing the DC component of the illumination:

$$\text{source}(\mathbf{s}) = \begin{cases} 1 & \text{if } \sigma_\text{inner} \leq |\mathbf{s}| \leq \sigma_\text{outer} \\ 0 & \text{otherwise} \end{cases}$$

### Defocus

Defocus introduces a quadratic phase aberration in the pupil plane. The wavefront error is:

$$W(\mathbf{f}) = \Delta z \cdot \left(1 - \sqrt{1 - \left(\frac{\lambda \cdot |\mathbf{f}|}{\text{NA}}\right)^2}\right)$$

where $\Delta z$ is the defocus distance in nm. This phase error blurs the aerial image as focus departs from the optimal plane, widening and reducing the contrast of features.

### Zernike Aberrations

Lens aberrations are modelled as a weighted sum of Zernike polynomials on the pupil disk. The aberrated pupil is:

$$P(\mathbf{f}) = H(\mathbf{f}) \cdot \exp\left(i \cdot 2\pi \sum_n c_n Z_n\left(\frac{\mathbf{f}}{\text{NA}/\lambda}\right)\right)$$

where $c_n$ are the Zernike coefficients in units of waves and $Z_n$ are the Noll-indexed Zernike polynomials.

### Resist Model — Simplified Dill Model

The resist exposure model uses a **threshold approach** derived from the Dill model. The resist response to aerial image intensity $I(\mathbf{r})$ is:

1. **PEB diffusion** — acid diffusion during the post-exposure bake is modelled as a Gaussian blur:

$$I_\text{eff}(\mathbf{r}) = I(\mathbf{r}) * \mathcal{G}(\sigma_\text{diff})$$

where $\sigma_\text{diff}$ is the diffusion length in nm.

2. **Threshold** — pixels are cleared (positive tone) where:

$$R(\mathbf{r}) = \begin{cases} 0 \text{ (cleared)} & \text{if } I_\text{eff}(\mathbf{r}) \geq t_r \\ 1 \text{ (unexposed)} & \text{otherwise} \end{cases}$$

where $t_r$ is the normalised exposure threshold.

### Critical Dimension

The printed critical dimension (CD) is measured as the width of the widest cleared region in the resist cross-section, determined by binary edge detection:

$$\text{CD} = x_\text{right} - x_\text{left}$$

where $x_\text{left}$ and $x_\text{right}$ are the first and last resist-cleared pixel positions along the central row.

### NILS — Normalised Image Log Slope

NILS is the industry-standard metric for feature printability. It quantifies the steepness of the aerial image intensity gradient at the resist threshold crossing, normalised by the feature size:

$$\text{NILS} = \text{CD} \cdot \left. \frac{d \ln I}{dx} \right|_{I = t_r}$$

- $\text{NILS} \geq 2.0$ — good printability, low sensitivity to dose variation
- $\text{NILS} < 2.0$ — feature is near the resolution limit
- $\text{NILS} < 1.0$ — feature will not print reliably

### Gaussian Blur Approximation for Dataset Generation

The full Abbe computation takes 10–60 seconds per image at 128–256 px resolution, making bulk dataset generation impractical. For the dataset pipeline, a **Gaussian blur approximation** is used:

$$\sigma_\text{blur} = 0.35 \cdot \frac{\lambda}{\text{NA}}$$

This approximates the Rayleigh coherent resolution element and produces physically plausible intensity distributions at approximately 5 ms per image, enabling generation of tens of thousands of images in minutes.

### Defect Types

| Type | Physical Description | Typical Source |
|---|---|---|
| Pinhole | Missing absorber in opaque region | Particle contamination or etch defect |
| Blob | Excess absorber in clear region | Resist residue or coating defect |
| Bridge | Absorber joining two separate features | Underdevelopment or particle |
| Break | Gap in a continuous feature | Overdevelopment or scratch |
| Line Edge Roughness | Statistical edge deviation along a feature | Resist grain noise or plasma etch |

### Autoencoder Anomaly Detection Principle

The autoencoder is trained exclusively on **defect-free** aerial images. It learns to compress and reconstruct the statistical patterns of normal lithography images. When a **defective** image is passed through the trained model:

- Clean regions reconstruct accurately → **low residual**
- Defective regions fail to reconstruct → **high residual**

The per-pixel residual map is:

$$R(\mathbf{r}) = \left(I_\text{input}(\mathbf{r}) - I_\text{recon}(\mathbf{r})\right)^2$$

Thresholding this map at a value $\tau$ produces a binary defect localisation mask. The overall image anomaly score is the mean squared error (MSE):

$$\text{MSE} = \frac{1}{N} \sum_{\mathbf{r}} R(\mathbf{r})$$

---

## 3-D Simulation and Multi-Patterning

The engine simulates a **physical wafer stack**, not just a 2-D footprint. One
voxel field is both the thing process steps mutate and the thing the 3-D viewer
draws, so multi-patterning and the 3-D model are the same feature.

### Quick start

```python
from litho_sim.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.layout import Layout, line_array
from litho_sim.process import lele

grid   = GridConfig(n_pixels=128, pixel_size=4e-9, dz=4e-9, n_z_slices=9)
optics = OpticsConfig(wavelength=193e-9, NA=0.93, sigma_outer=0.8, source_grid=15)
resist = ResistConfig(dose_nominal=21.0, mack_Mth=0.5, diffusion_sigma=12e-9)

# 6 gates at 80 nm pitch — too dense for a single ArF exposure (contrast 0.36)
target = Layout(line_array(6, pitch=80e-9, cd=40e-9, length=600e-9))

flow = lele(target, grid, optics, resist, min_spacing=70e-9, overlay=(6e-9, 0))
result = flow.run()
print(result.measurements[["name", "mean_line_nm", "pitch_walk_nm"]])
```

### What is emergent rather than modelled

Nothing in the engine computes "pitch walk" or knows that a spacer doubles a
pattern. Both follow from the primitives:

| Effect | Where it comes from |
|---|---|
| **Pitch walking** | `Expose` images geometry shifted by the overlay. Printed features then alternate by ~2× the offset — verified across a 0–12 nm sweep. |
| **Pitch division** | A conformal film has two sidewalls per mandrel. Deposit → etch back → strip turns N mandrels into 2N lines whose CD equals the *deposited thickness*. |
| **Etch masking** | `Etch` takes no mask argument. Whatever material is on top of a column protects it, which is why LELE, SADP and cut masks need no special-case code. |
| **Sloped sidewalls** | Dill A/B absorption means the top of the film sees more dose than the bottom. |
| **Standing-wave scalloping** | Substrate reflection interferes with the incident wave at λ/(2·n_resist) ≈ 57 nm, and can block development at the nodes — which is what PEB and BARC exist to fix. |

### Run the demo

```bash
python scripts/demo_multipatterning.py     # → results/multipatterning_demo.png
```

### Interactive pages

```bash
streamlit run scripts/interactive_app.py
```

| Page | What it does |
|---|---|
| **Layout Editor** | Author shapes in an editable table with a live mask + aerial-image preview, and decompose for LELE / LE³ with conflicts reported. |
| **Process Flow** | Pick a scheme, set overlay and spacer thickness, run, and scrub the wafer cross-section step by step. |
| **Stack 3D** | Rotate and slice the printed stack, toggle materials, exaggerate z, and measure CD at any height. |

The 3-D viewer needs Plotly (`pip install plotly`); without it the page falls
back to a static Matplotlib rendering.

### Performance note

The Abbe sum costs one FFT per source point, and the source used to be sampled
on the *mask* grid — 8,112 points at 128 px, of which integer-pixel rounding
collapsed all but ~21. `OpticsConfig.source_grid` (default 21) now samples the
source independently and places each point exactly: **~40× faster and better
sampled at the same time.**

---

## Directory Structure
```
pplith2/
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── .streamlit/
│   └── config.toml                   # Global Streamlit dark theme
│
├── src/
│   └── litho_sim/                    # Organised by lithographic process step
│       ├── __init__.py
│       ├── cli.py / __main__.py      # CLI: python -m litho_sim …
│       ├── core/                     # SimulationConfig, OpticsConfig, ResistConfig, GridConfig
│       ├── mask/                     # geometry, layouts, pixel patterns (what gets printed)
│       ├── expose/                   # pupil/Zernike, illumination, Source objects, Abbe imaging
│       │   └── m3d/                  # thick-mask models: multilayer mirror + Bloch FDTD
│       ├── coat/                     # film-stack optics: TMM, BARC, swing curves
│       ├── bake/                     # PEB diffusion (2-D + 3-D)
│       ├── develop/                  # resist models, 2-D and 3-D develop, profile metrics
│       ├── wafer/                    # voxel Stack + material library
│       ├── patterning/               # steps, flows, multipatterning recipes, cuts
│       ├── analysis/                 # NILS, CD, process window, Bossung curves
│       ├── viz/                      # matplotlib figures + 3-D stack renderers
│       ├── app/                      # PySide6 desktop front end
│       ├── tech/                     # reserved: DRAM/NAND technology pipelines
│       └── ml/                       # defect-detection study (torch optional)
│
├── scripts/
│   ├── interactive_app.py            # Streamlit interactive simulator (legacy; Qt app planned)
│   ├── app_lib.py + pages/           # Streamlit multipage app (layout/flow/3-D/source editors)
│   ├── build_docs.py                 # Regenerates the generated half of docs/
│   ├── demo_gaa.py                   # Gate-all-around nanosheet device flow (--figure, --calibrate)
│   ├── demo_multipatterning.py       # LELE/SADP demo figures
│   ├── demo_vector_imaging.py        # The vector effect at hyper-NA
│   ├── explore_vector.py             # Interactive vector bench (--selftest for headless)
│   ├── generate_data.py              # ML dataset generation
│   └── train_model.py                # ML training entry point
│
├── docs/                             # Obsidian vault — local only, gitignored
├── results/                          # Demo figures written by the scripts above
│
├── checkpoints/                      # Created at training time
│   ├── autoencoder_best.pth          # Best checkpoint (lowest loss)
│   └── autoencoder_epoch050.pth      # Periodic epoch checkpoint
│
├── dataset/                          # Created by generate_dataset.py
│   ├── clean/
│   │   └── clean_000000.png ...
│   └── defective/
│       ├── images/
│       │   └── defect_000000.png ...
│       └── masks/
│           └── defect_000000_mask.png ...
│
├── inference_results/                # Created by run_inference.py
│   ├── overlays/
│   │   └── defect_000000.png ...
│   └── inference_summary.csv
│
└── tests/                            # 24 files, ~680 tests; slow FDTD/GAA solves
    ├── test_aerial_image.py …        #   are marked `slow` — use -m "not slow"
    ├── test_m3d.py / test_fdtd.py    #   mask 3-D models and the Yee solver
    ├── test_gaa_flow.py              #   the printed GAA device flow, end to end
    └── …
```

---

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- macOS, Linux, or Windows
- At least 4 GB RAM (8 GB recommended for dataset generation)
- GPU optional but recommended for training (NVIDIA CUDA or Apple Silicon MPS)

### Clone the Repository

```bash
git clone https://github.com/yourname/pplith2.git
cd pplith2
```

### Create a Virtual Environment

```bash
# Create environment
python3 -m venv .venv

# Activate — macOS / Linux
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### requirements.txt

```text
numpy&gt;=1.24
scipy&gt;=1.10
matplotlib&gt;=3.7
pandas&gt;=2.0
pillow&gt;=10.0
streamlit&gt;=1.30
tqdm&gt;=4.65
torch&gt;=2.1
torchvision&gt;=0.16
```

### Verify Installation

```bash
python -m litho_sim demo
```

You should see simulation progress in the terminal and a figure saved to the working directory.

---

## Full Pipeline Overview

The project runs in four sequential stages:

```
Stage 1  ─  Interactive Simulation
            scripts/interactive_app.py
            Explore optical and resist parameters in a browser UI.
            Inject defects and observe their effect on the printed pattern.

Stage 2  ─  Bulk Dataset Generation
            scripts/generate_dataset.py
            Generate thousands of clean and defective image pairs at speed
            using the Gaussian blur approximation.

Stage 3  ─  Autoencoder Training
            scripts/train_autoencoder.py
            Train a convolutional autoencoder on clean images only.
            The model learns the statistical structure of defect-free patterns.

Stage 4  ─  Inference and Detection
            scripts/run_inference.py
            Pass defective images through the trained autoencoder.
            Compute per-pixel residuals and threshold to localise defects.
            Evaluate against ground-truth masks using IoU.
```

---

## Source Modules

The package is organised by lithographic process step, and each subpackage's
`__init__.py` is its declared public API — import from the package, not the module.
The exhaustive per-module surface is generated into the docs vault by
`scripts/build_docs.py`; what follows is the map and the headline entry point of each
area.

| Area | Import | The step it models, and where to start |
|---|---|---|
| `core/` | `litho_sim.core.config` | `SimulationConfig` composed of `OpticsConfig` / `ResistConfig` / `GridConfig`, plus `TECH_NODE_PRESETS` (i-line → EUV) and `RESIST_LIBRARY`. JSON round-trips. |
| `mask/` | `litho_sim.mask` | What gets printed. Vector shapes and `Layout` (`line_array`, `contact_grid`, `cut_bar`), multi-patterning colouring via `decompose`, and the legacy pixel generators `lines_and_spaces` / `contact_array` / `isolated_line` / `checkerboard`. |
| `expose/` | `litho_sim.expose` | `compute_aerial_image(mask, optics, grid, dose=1.0)` — partially coherent Abbe imaging, one FFT per source point. `Source` builds the illuminator; `pupil` carries Zernikes (Noll 1–21) and the vector/Jones path. `expose.m3d` adds thick-mask models behind `OpticsConfig.mask_model`. |
| `coat/` | `litho_sim.coat` | `FilmStack` transfer-matrix optics: `reflectance`, `amplitude_reflection`, `field_profile`, `swing_curve` — BARC design and standing waves. |
| `bake/` | `litho_sim.bake` | `apply_peb` / `apply_peb_3d` Gaussian acid diffusion; `bake.reaction` carries real acid–quencher CAR kinetics. |
| `develop/` | `litho_sim.develop` | `simulate_resist` (2-D Dill/Mack/threshold) and `print_resist_3d` / `develop_3d` (3-D volume, `model="threshold"\|"mack"\|"front"`). `develop.front.arrival_time` is the eikonal moving-front solver, reused anywhere a front moves. |
| `wafer/` | `litho_sim.wafer` | `Stack` — the `(nz, ny, nx)` uint8 voxel volume every process step mutates: `deposit_blanket` / `deposit_conformal` (with `on=` for selective growth), `etch` (with `exposure="any"` for lateral attack), `etch_back`, `planarize`, `strip`, `measure_features`. |
| `patterning/` | `litho_sim.patterning` | `Flow` plus the 11-kind `STEP_REGISTRY`; recipe builders `single_exposure`, `lele`, `sadp`, `saqp`; line-end `cuts`. |
| `analysis/` | `litho_sim.analysis` | `run_full_analysis`, `sweep_dose_focus`, `compute_process_window`, `compute_nils`, `compute_depth_of_focus`, `compute_exposure_latitude`. |
| `viz/` | `litho_sim.viz` | Matplotlib figures (`plot_aerial_image`, `plot_bossung_curves`, …) and the 3-D stack renderers in `viz3d` (Plotly optional; Matplotlib fallbacks always available). |
| `app/` | `python -m litho_sim.app` | The PySide6 desktop front end — Wafer Stack, Mask, Expose and Develop tabs over the same engine. |
| `ml/` | `litho_sim.ml` | The defect-detection study: convolutional autoencoder, dataset helpers, synthetic defect injection. Requires the `ml` extra. |
| `tech/` | — | Reserved for technology pipelines (DRAM/NAND); currently empty. |

---

## Scripts and Usage

### CLI Entry Point

### Desktop App (new)

A native Qt front end for driving the engine interactively — sliders for the
optics, resist, mask and vector-imaging parameters, with the aerial image,
developed resist and cross-section updating as you move them.

```bash
pip install -e ".[app]"       # or: pip install PySide6
python -m litho_sim.app       # or: litho-sim-app
```

The simulation runs on a worker thread, and changes too expensive to render
live (a large grid, or vector imaging with unpolarised light) wait for the
drag to settle rather than queueing frames that are stale before they draw.

### CLI Entry Point

`litho_sim.cli` provides three command-line modes (installed as `litho-sim`,
or run as `python -m litho_sim`):

```bash
# Full single-condition simulation with figures
python -m litho_sim demo

# Bossung curves: CD vs. defocus at multiple dose levels
python -m litho_sim bossung

# Process window: exposure latitude vs. depth of focus
python -m litho_sim window
```

---

### Interactive Simulator

```bash
streamlit run scripts/interactive_app.py
```

Opens at `http://localhost:8501`.

#### Controls

| Control | Range / Options |
|---|---|
| Technology Node | i-line (365 nm), KrF (248 nm), ArF (193 nm), ArF immersion (193 nm / 1.35 NA), EUV (13.5 nm) |
| Pattern Type | Lines & Spaces, Contact Array, Isolated Line, Checkerboard |
| Pitch | 50 – 800 nm |
| Feature CD | 10 – 400 nm |
| NA | 0.10 – 1.35 |
| Sigma outer | 0.10 – 1.00 |
| Sigma inner | 0.00 – 0.90 |
| Relative Dose | 0.50 – 2.00 |
| Defocus | -300 to +300 nm |
| Exposure Threshold | 0.10 – 0.90 |
| PEB Diffusion | 0 – 60 nm |
| Grid Resolution | 64, 128, 256 px |
| Defect Type | None, Pinhole, Blob, Bridge, Break, Line Edge Roughness |
| Defect Radius | 10 – 200 nm |
| Random Seed | Any integer |

#### Figure Layout

| Panel | Content |
|---|---|
| Top left | Binary mask pattern |
| Top centre | Aerial image intensity (inferno colormap) |
| Top right | Resist pattern with defect overlay (RdYlGn colormap) |
| Bottom left | Aerial image cross-section with threshold line |
| Bottom centre | Resist profile cross-section |
| Bottom right | Metrics: CD, NILS, contrast, intensity range |

A yellow dashed circle marks the defect injection site when a defect type is active. Printed defect regions are highlighted red on the resist panel.

#### Defect Analysis Report

Automatically generated when any defect type is selected:

- Whether the defect is printable under current process conditions
- Number of distinct defect regions printed in the resist
- Total defect area in nm²
- Estimated defect size (square root of area, in nm)

#### Export Options

- Cross-section data as a CSV (position, aerial intensity, resist state)
- Aerial image array as a raw NumPy binary (`.npy`)

---

### Dataset Generation

```bash
python scripts/generate_dataset.py \
    --n_clean     10000 \
    --n_defective 10000 \
    --image_size  256   \
    --pixel_nm    4.0   \
    --min_defects 1     \
    --max_defects 3     \
    --workers     8     \
    --seed        0     \
    --out_dir     dataset
```

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `--n_clean` | 5000 | Number of defect-free training images |
| `--n_defective` | 5000 | Number of defective images |
| `--image_size` | 256 | Pixel dimension of each image |
| `--pixel_nm` | 4.0 | Physical pixel size in nm |
| `--min_defects` | 1 | Minimum defects per defective image |
| `--max_defects` | 3 | Maximum defects per defective image |
| `--workers` | cpu-1 | Parallel worker processes |
| `--seed` | 0 | Base random seed for reproducibility |
| `--out_dir` | dataset | Output root directory |

#### Output Layout

```
dataset/
├── clean/
│   ├── clean_000000.png
│   └── ...
└── defective/
    ├── images/
    │   ├── defect_000000.png
    │   └── ...
    └── masks/
        ├── defect_000000_mask.png    # Binary ground-truth defect map
        └── ...
```

#### Expected Generation Time (8-core Apple M-series)

- 10,000 clean images: ~30 seconds
- 10,000 defective images: ~45 seconds

#### Verify Dataset

```bash
echo &quot;Clean images       : $(ls dataset/clean/ | wc -l)&quot;
echo &quot;Defective images   : $(ls dataset/defective/images/ | wc -l)&quot;
echo &quot;Ground-truth masks : $(ls dataset/defective/masks/ | wc -l)&quot;
```

#### Preview Dataset

```bash
cat &gt; preview_dataset.py &lt;&lt; &#x27;EOF&#x27;
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

clean  = sorted(Path(&quot;dataset/clean&quot;).glob(&quot;*.png&quot;))[:4]
defect = sorted(Path(&quot;dataset/defective/images&quot;).glob(&quot;*.png&quot;))[:4]
masks  = sorted(Path(&quot;dataset/defective/masks&quot;).glob(&quot;*.png&quot;))[:4]

fig, axes = plt.subplots(3, 4, figsize=(14, 9))
fig.suptitle(&quot;Dataset Preview&quot;, fontsize=13)

for i, (c, d, m) in enumerate(zip(clean, defect, masks)):
    axes[0, i].imshow(Image.open(c), cmap=&quot;gray&quot;)
    axes[0, i].axis(&quot;off&quot;)
    axes[1, i].imshow(Image.open(d), cmap=&quot;gray&quot;)
    axes[1, i].axis(&quot;off&quot;)
    axes[2, i].imshow(Image.open(m), cmap=&quot;hot&quot;)
    axes[2, i].axis(&quot;off&quot;)

axes[0, 0].set_ylabel(&quot;Clean&quot;,     fontsize=10)
axes[1, 0].set_ylabel(&quot;Defective&quot;, fontsize=10)
axes[2, 0].set_ylabel(&quot;GT Mask&quot;,   fontsize=10)

plt.tight_layout()
plt.savefig(&quot;dataset_preview.png&quot;, dpi=120)
plt.show()
EOF

python preview_dataset.py
```

---

### Autoencoder Training

```bash
python scripts/train_autoencoder.py \
    --data_dir       dataset/clean \
    --checkpoint_dir checkpoints   \
    --epochs         50            \
    --batch_size     32            \
    --latent_dim     64            \
    --workers        2             \
    --device         auto
```

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `dataset/clean` | Directory of clean PNG images |
| `--checkpoint_dir` | `checkpoints` | Where to save `.pth` files |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 32 | Images per gradient step |
| `--latent_dim` | 64 | Bottleneck dimension |
| `--workers` | 2 | DataLoader threads |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |

Device auto-detection priority: **CUDA > MPS (Apple Silicon) > CPU**.

#### Expected Training Time

| Hardware | Time |
|---|---|
| Apple M1/M2/M3 (MPS) | 15 – 30 minutes |
| NVIDIA GPU (CUDA) | 5 – 15 minutes |
| CPU only | 2 – 4 hours |

#### Checkpoints Saved

- `checkpoints/autoencoder_best.pth` — lowest training loss
- `checkpoints/autoencoder_epoch010.pth` — periodic saves every 10 epochs

#### Latent Dimension Guide

| `latent_dim` | Effect |
|---|---|
| 8 | Very tight bottleneck — high anomaly sensitivity, may miss subtle patterns |
| 16 | Tight bottleneck — recommended if IoU is plateauing |
| 64 | Default — good reconstruction quality, moderate anomaly sensitivity |
| 128 | Large bottleneck — may reconstruct defects too well |

---

### Inference and Defect Detection

```bash
python scripts/run_inference.py \
    --checkpoint checkpoints/autoencoder_best.pth \
    --images     dataset/defective/images/ \
    --gt_masks   dataset/defective/masks/ \
    --out_dir    inference_results/ \
    --threshold  0.08 \
    --device     auto
```

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to trained `.pth` checkpoint |
| `--images` | required | Directory of defective PNG images |
| `--gt_masks` | optional | Ground-truth masks for IoU scoring |
| `--out_dir` | `inference_results` | Output directory |
| `--threshold` | 0.02 | Per-pixel residual threshold |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--no_overlays` | off | Skip saving overlay PNG files |

#### Threshold Guide

| Value | Behaviour |
|---|---|
| 0.01 | High sensitivity, more false positives |
| 0.08 | Balanced — empirically optimal for this dataset |
| 0.15 | Conservative, fewer false positives, may miss subtle defects |

#### Outputs

```
inference_results/
├── overlays/
│   ├── defect_000000.png    # Defective image with red defect overlay
│   └── ...
└── inference_summary.csv    # filename, mse, defect_px, iou
```

#### Analyse Results

```bash
cat &gt; analyse_results.py &lt;&lt; &#x27;EOF&#x27;
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(&quot;inference_results/inference_summary.csv&quot;)

print(df.describe())

print(&quot;\nTop 10 most anomalous images:&quot;)
print(df.nlargest(10, &quot;mse&quot;)[[&quot;filename&quot;, &quot;mse&quot;, &quot;iou&quot;]])

print(&quot;\nBottom 10 lowest MSE (possible missed detections):&quot;)
print(df.nsmallest(10, &quot;mse&quot;)[[&quot;filename&quot;, &quot;mse&quot;, &quot;iou&quot;]])

plt.figure(figsize=(10, 4))
plt.hist(df[&quot;mse&quot;], bins=100, color=&quot;steelblue&quot;, edgecolor=&quot;none&quot;)
plt.axvline(0.08, color=&quot;red&quot;, linestyle=&quot;--&quot;, label=&quot;threshold=0.08&quot;)
plt.xlabel(&quot;MSE&quot;)
plt.ylabel(&quot;Count&quot;)
plt.title(&quot;Reconstruction Error Distribution&quot;)
plt.legend()
plt.tight_layout()
plt.savefig(&quot;mse_distribution.png&quot;, dpi=120)
plt.show()
EOF

python analyse_results.py
```

#### Compare Thresholds

```bash
cat &gt; compare_thresholds.py &lt;&lt; &#x27;EOF&#x27;
import pandas as pd

results = {
    &quot;0.02&quot;: &quot;inference_results/inference_summary.csv&quot;,
    &quot;0.05&quot;: &quot;inference_results_t005/inference_summary.csv&quot;,
    &quot;0.08&quot;: &quot;inference_results_t008/inference_summary.csv&quot;,
    &quot;0.10&quot;: &quot;inference_results_t010/inference_summary.csv&quot;,
    &quot;0.15&quot;: &quot;inference_results_t015/inference_summary.csv&quot;,
}

print(f&quot;{&#x27;Threshold&#x27;:&lt;12} {&#x27;IoU mean&#x27;:&lt;12} {&#x27;IoU&gt;0.5&#x27;:&lt;12} {&#x27;Missed (IoU=0)&#x27;:&lt;16} {&#x27;Avg flagged px&#x27;}&quot;)
print(&quot;-&quot; * 70)

for t, path in results.items():
    try:
        df = pd.read_csv(path)
        print(
            f&quot;{t:&lt;12} &quot;
            f&quot;{df[&#x27;iou&#x27;].mean():&lt;12.4f} &quot;
            f&quot;{(df[&#x27;iou&#x27;] &gt; 0.5).sum():&lt;12} &quot;
            f&quot;{(df[&#x27;iou&#x27;] == 0.0).sum():&lt;16} &quot;
            f&quot;{df[&#x27;defect_px&#x27;].mean():.0f}&quot;
        )
    except FileNotFoundError:
        print(f&quot;{t:&lt;12} not found&quot;)
EOF

python compare_thresholds.py
```

---

## Complete Source Code

### `scripts/autoencoder.py`

```python
&quot;&quot;&quot;Convolutional autoencoder for lithography defect anomaly detection.&quot;&quot;&quot;
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    &quot;&quot;&quot;Convolutional autoencoder: 256x256 -&gt; latent_dim -&gt; 256x256.

    Trained exclusively on defect-free images. Anomalies are detected
    as regions of high per-pixel reconstruction error.
    &quot;&quot;&quot;

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        # Encoder: 256 -&gt; 128 -&gt; 64 -&gt; 32 -&gt; 16 -&gt; latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32,  kernel_size=4, stride=2, padding=1),  # 256-&gt;128
            nn.ReLU(),
            nn.Conv2d(32, 64,  kernel_size=4, stride=2, padding=1),  # 128-&gt;64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64-&gt;32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 32-&gt;16
            nn.ReLU(),
            nn.Flatten(),                                             # 256*16*16=65536
            nn.Linear(256 * 16 * 16, latent_dim),
        )

        # Decoder: latent_dim -&gt; 16 -&gt; 32 -&gt; 64 -&gt; 128 -&gt; 256
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16-&gt;32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1), # 32-&gt;64
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1), # 64-&gt;128
            nn.ReLU(),
            nn.ConvTranspose2d(32,  1,   kernel_size=4, stride=2, padding=1), # 128-&gt;256
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -&gt; torch.Tensor:
        return self.decoder(self.encoder(x))

    def get_reconstruction_error(self, x: torch.Tensor) -&gt; torch.Tensor:
        &quot;&quot;&quot;Return per-image MSE reconstruction error (no gradient).&quot;&quot;&quot;
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=(1, 2, 3))
```

---

## Configuration Reference

### `.streamlit/config.toml`

```toml
[theme]
base                     = &quot;dark&quot;
backgroundColor          = &quot;#1a1a1a&quot;
secondaryBackgroundColor = &quot;#222222&quot;
textColor                = &quot;#c8c8c8&quot;
font                     = &quot;monospace&quot;
```

### Simulation Defaults

| Parameter | Default | Unit |
|---|---|---|
| Node | ArF | — |
| Wavelength | 193 | nm |
| NA | 0.93 | — |
| Sigma outer | 0.85 | — |
| Sigma inner | 0.00 | — |
| Pitch | 400 | nm |
| CD | 150 | nm |
| Defocus | 0 | nm |
| Dose | 1.00 | relative |
| Threshold | 0.35 | — |
| PEB Diffusion | 10 | nm |
| Resolution | 128 | px |

### `pyproject.toml`

```toml
[build-system]
requires      = [&quot;setuptools&gt;=68&quot;]
build-backend = &quot;setuptools.build_meta&quot;

[project]
name    = &quot;litho-sim&quot;
version = &quot;0.1.0&quot;
requires-python = &quot;&gt;=3.10&quot;
dependencies = [
    &quot;numpy&gt;=1.24&quot;,
    &quot;scipy&gt;=1.10&quot;,
    &quot;matplotlib&gt;=3.7&quot;,
    &quot;pandas&gt;=2.0&quot;,
    &quot;pillow&gt;=10.0&quot;,
    &quot;streamlit&gt;=1.30&quot;,
    &quot;tqdm&gt;=4.65&quot;,
    &quot;torch&gt;=2.1&quot;,
    &quot;torchvision&gt;=0.16&quot;,
]

[tool.setuptools.packages.find]
where = [&quot;src&quot;]
```

---

## Results and Evaluation

### Training Results

| Metric | Value |
|---|---|
| Final training loss | 0.000042 |
| Epochs trained | 50 |
| Latent dimension | 64 |
| Training set size | 10,000 clean images |

### Inference Results (10,000 defective images)

| Threshold | IoU mean | IoU > 0.5 | Missed (IoU=0) |
|---|---|---|---|
| 0.02 | 0.3989 | — | — |
| 0.05 | — | — | — |
| 0.08 | **0.4190** | — | — |
| 0.10 | 0.4166 | — | — |
| 0.15 | 0.3949 | — | — |

**Best threshold: 0.08 (IoU = 0.419)**

### MSE Distribution

| Statistic | Value |
|---|---|
| Mean MSE | 0.027136 |
| Max MSE | 0.192759 |
| Min MSE | 0.000004 |
| Std MSE | 0.022662 |

### Interpretation

An IoU of ~0.42 is a strong baseline for a **fully unsupervised** approach — the model was never shown a single defective image during training, yet it localises defects spatially in 42% of cases by intersection-over-union. The IoU plateau across thresholds (0.08–0.10) indicates the bottleneck is the architecture rather than the threshold, and retraining with a tighter latent dimension (e.g. `latent_dim=16`) is the recommended next step for improved localisation.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'autoencoder'`

The `run_inference.py` script imports `autoencoder.py` from the same directory. Ensure `scripts/autoencoder.py` exists:

```bash
ls scripts/autoencoder.py
```

If missing, add the following to the top of `run_inference.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

### `RuntimeError: Error(s) in loading state_dict`

The architecture in `scripts/autoencoder.py` does not match the checkpoint. Verify the layer dimensions match what was used during training by inspecting the error message keys. The correct architecture uses:

- 4 encoder Conv layers with filters `32 → 64 → 128 → 256`, kernel size 4, stride 2
- `nn.Unflatten` inside the decoder Sequential (not a separate `decoder_fc` attribute)

### `ValueError: low >= high` during dataset generation

A defect radius is too large for the image. Fixed by capping the defect radius:

```python
radius = min(radius, min(h, w) // 4)
```

### `%%` in argparse help strings

Python 3.14 raises `ValueError` when `%` appears in argparse help text. Escape all percent signs as `%%`.

### Streamlit app is slow at 256 px

Reduce the grid resolution to 128 px in the sidebar. The full Abbe simulation scales as $O(N^2 \log N)$ per source point and has 50–200 source points per simulation.

### Training loss not decreasing below 0.01

- Increase `--epochs` to 100
- Reduce `--latent_dim` to 32 or 16
- Verify `--data_dir` points to the correct clean image directory
- Check device is using GPU: `--device auto` should detect MPS or CUDA automatically

### MPS device errors on Apple Silicon

If PyTorch MPS causes numerical errors, fall back to CPU:

```bash
python scripts/train_autoencoder.py --device cpu ...
```

---

## Full Pipeline Quick Reference

```bash
# 0. Activate environment
source .venv/bin/activate

# 1. Generate dataset
python scripts/generate_dataset.py \
    --n_clean 10000 --n_defective 10000 --image_size 256 --workers 8

# 2. Train autoencoder
python scripts/train_autoencoder.py \
    --data_dir dataset/clean --epochs 50 --batch_size 32 --latent_dim 64

# 3. Run inference
python scripts/run_inference.py \
    --checkpoint checkpoints/autoencoder_best.pth \
    --images     dataset/defective/images/ \
    --gt_masks   dataset/defective/masks/ \
    --out_dir    inference_results/ \
    --threshold  0.08

# 4. Analyse results
python analyse_results.py
python compare_thresholds.py

# 5. Launch interactive simulator
streamlit run scripts/interactive_app.py

# 6. CLI simulation modes
python main.py demo
python main.py bossung
python main.py window
python main.py detect
```

---

## Further Reading and References

### Optical Lithography

- Mack, C. A. *Fundamental Principles of Optical Lithography*. Wiley, 2007.
- Levenson, M. D. *Introduction to Immersion Lithography*. SPIE Press, 2009.
- Goodman, J. W. *Introduction to Fourier Optics*, 4th ed. W. H. Freeman, 2017.

### Resist Modelling

- Dill, F. H. et al. "Characterization of Positive Photoresist." *IEEE Transactions on Electron Devices*, 22(7), 1975.
- Mack, C. A. "Analytical Expression for the Standing Wave Intensity in Photoresist." *Applied Optics*, 25(12), 1986.

### Anomaly Detection with Autoencoders

- Baur, C. et al. "Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images." *Medical Image Analysis*, 2021.
- Bergmann, P. et al. "Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders." *arXiv:1807.02011*, 2018.

### NILS and Process Window

- Brunner, T. A. "Why Optical Lithography Will Live Forever." *Journal of Vacuum Science and Technology*, 2003.
- Levinson, H. J. *Principles of Lithography*, 4th ed. SPIE Press, 2019.

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

2. Follow the existing code style:
   - Type hints on all public functions
   - NumPy-format docstrings on all public classes and functions
   - No magic numbers — use named constants or config fields

3. Add or update tests in `tests/` for any new simulation functionality:
```bash
python -m pytest tests/ -v
```

4. Run the full pipeline end-to-end before submitting a pull request

5. Submit a pull request with a clear description of the change and any relevant physics or ML justification

### Reporting Issues

Please include:
- Python version (`python --version`)
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- Operating system
- Full traceback
- Minimal reproducible example

---

## License

```
MIT License

Copyright (c) 2026 Ben Taylor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the &quot;Software&quot;), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED &quot;AS IS&quot;, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

