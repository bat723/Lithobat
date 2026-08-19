import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
import pandas as pd

# ── 1. Simulation thumbnail: mask → aerial pair ─────────────────────
n = 256
pitch_px = 30
mask = np.zeros((n, n))
for x in range(0, n, pitch_px):
    mask[:, x:x + pitch_px // 2] = 1.0

aerial = gaussian_filter(mask, sigma=5.0)

fig_sim, axes_sim = plt.subplots(1, 2, figsize=(3, 1.5))
axes_sim[0].imshow(mask, cmap="gray")
axes_sim[0].set_title("Mask", fontsize=7)
axes_sim[0].axis("off")
axes_sim[1].imshow(aerial, cmap="gray")
axes_sim[1].set_title("Aerial", fontsize=7)
axes_sim[1].axis("off")
fig_sim.tight_layout(pad=0.3)
fig_sim.savefig("slides/thumb_simulation.png", dpi=150, bbox_inches="tight")
plt.close(fig_sim)
print("1/4  thumb_simulation.png")

# ── 2. Dataset thumbnail: 2x2 grid of clean + defective ─────────────
clean_dir = Path("dataset/clean")
defect_dir = Path("dataset/defective/images")

clean_imgs = sorted(clean_dir.glob("*.png"))[:2]
defect_imgs = sorted(defect_dir.glob("*.png"))[:2]

fig_ds, axes_ds = plt.subplots(2, 2, figsize=(3, 3))
for i, c in enumerate(clean_imgs):
    axes_ds[0, i].imshow(Image.open(c), cmap="gray")
    axes_ds[0, i].axis("off")
for i, d in enumerate(defect_imgs):
    axes_ds[1, i].imshow(Image.open(d), cmap="gray")
    axes_ds[1, i].axis("off")
axes_ds[0, 0].set_ylabel("Clean", fontsize=7)
axes_ds[1, 0].set_ylabel("Defective", fontsize=7)
fig_ds.tight_layout(pad=0.3)
fig_ds.savefig("slides/thumb_dataset.png", dpi=150, bbox_inches="tight")
plt.close(fig_ds)
print("2/4  thumb_dataset.png")

# ── 3. Autoencoder thumbnail: block diagram ──────────────────────────
fig_ae, ax_ae = plt.subplots(1, 1, figsize=(3.5, 1.5))
ax_ae.set_xlim(0, 10)
ax_ae.set_ylim(0, 3)
ax_ae.axis("off")

widths =  [1.0, 0.8, 0.6, 0.4]
heights = [2.4, 2.0, 1.6, 1.0]
x = 0.3
colors_enc = ["#4A90D9", "#3B7DD8", "#2C6AD7", "#1D57D6"]
for w, h, c in zip(widths, heights, colors_enc):
    y = (3 - h) / 2
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", linewidth=0.5)
    ax_ae.add_patch(rect)
    x += w + 0.15

lat = mpatches.FancyBboxPatch((x, 1.0), 0.3, 1.0, boxstyle="round,pad=0.05",
                               facecolor="#E74C3C", edgecolor="white", linewidth=0.5)
ax_ae.add_patch(lat)
ax_ae.text(x + 0.15, 0.65, "z", fontsize=7, ha="center", color="white", fontweight="bold")
x += 0.3 + 0.15

colors_dec = ["#27AE60", "#2ECC71", "#58D68D", "#82E0AA"]
for w, h, c in zip(reversed(widths), reversed(heights), colors_dec):
    y = (3 - h) / 2
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", linewidth=0.5)
    ax_ae.add_patch(rect)
    x += w + 0.15

ax_ae.text(2.0, 2.85, "Encoder", fontsize=7, ha="center", fontweight="bold", color="#2C3E50")
ax_ae.text(4.35, 2.85, "Latent", fontsize=6, ha="center", fontweight="bold", color="#E74C3C")
ax_ae.text(7.0, 2.85, "Decoder", fontsize=7, ha="center", fontweight="bold", color="#2C3E50")

ax_ae.annotate("", xy=(0.3, 1.5), xytext=(-0.1, 1.5),
               arrowprops=dict(arrowstyle="->", color="#555"))
ax_ae.text(-0.15, 1.8, "Input", fontsize=5, ha="center", color="#555")
ax_ae.annotate("", xy=(x + 0.1, 1.5), xytext=(x - 0.2, 1.5),
               arrowprops=dict(arrowstyle="->", color="#555"))
ax_ae.text(x + 0.3, 1.8, "Output", fontsize=5, ha="center", color="#555")

fig_ae.tight_layout(pad=0.1)
fig_ae.savefig("slides/thumb_autoencoder.png", dpi=150, bbox_inches="tight",
               facecolor="white")
plt.close(fig_ae)
print("3/4  thumb_autoencoder.png")

# ── 4. Defect map thumbnail: one overlay result ─────────────────────
overlay_dir = Path("inference_results_t008/overlays")
df = pd.read_csv("inference_results_t008/inference_summary.csv")
good_row = df[(df["iou"] > 0.5) & (df["iou"] < 0.8)].iloc[0]
overlay_path = overlay_dir / good_row["filename"]

fig_dm, ax_dm = plt.subplots(1, 1, figsize=(1.8, 1.8))
ax_dm.imshow(Image.open(overlay_path))
ax_dm.axis("off")
fig_dm.tight_layout(pad=0.1)
fig_dm.savefig("slides/thumb_defectmap.png", dpi=150, bbox_inches="tight")
plt.close(fig_dm)
print("4/4  thumb_defectmap.png")

# ── 5. Assemble the pipeline figure (NO ARROWS) ─────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5),
                          gridspec_kw={"width_ratios": [1, 1, 1.2, 0.8]})

thumbs = [
    ("slides/thumb_simulation.png",  "Simulation Engine"),
    ("slides/thumb_dataset.png",     "Synthetic Dataset\n20K images"),
    ("slides/thumb_autoencoder.png", "Convolutional\nAutoencoder"),
    ("slides/thumb_defectmap.png",   "Defect Detection"),
]

for ax, (path, title) in zip(axes, thumbs):
    img = Image.open(path)
    ax.imshow(img)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

fig.tight_layout(pad=1.0)
fig.savefig("slides/slide3_pipeline.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("\nSaved: slides/slide3_pipeline.png")
plt.show()
