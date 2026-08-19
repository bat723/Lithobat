import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

# ── Auto-detect dataset paths ────────────────────────────────────────
candidates = [
    {
        "clean":  Path("dataset/clean"),
        "defect": Path("dataset/defective/images"),
        "masks":  Path("dataset/defective/masks"),
    },
    {
        "clean":  Path("data/raw_synthetic/clean"),
        "defect": Path("data/raw_synthetic/defect"),
        "masks":  None,
    },
]

paths = None
for c in candidates:
    if c["clean"].exists() and c["defect"].exists():
        paths = c
        break

if paths is None:
    print("ERROR: Could not find dataset. Checked:")
    for c in candidates:
        print(f"  {c['clean']}  exists={c['clean'].exists()}")
        print(f"  {c['defect']}  exists={c['defect'].exists()}")
    exit(1)

print(f"Using:  clean  = {paths['clean']}")
print(f"        defect = {paths['defect']}")

has_masks = paths["masks"] is not None and paths["masks"].exists()
if has_masks:
    print(f"        masks  = {paths['masks']}")
else:
    print("        masks  = (none found — will show 2 rows instead of 3)")

# ── Collect images ───────────────────────────────────────────────────
n_cols = 5

clean_paths  = sorted(paths["clean"].glob("*.png"))[:n_cols]
defect_paths = sorted(paths["defect"].glob("*.png"))[:n_cols]

if has_masks:
    mask_paths = sorted(paths["masks"].glob("*.png"))[:n_cols]

# ── Build figure ─────────────────────────────────────────────────────
n_rows = 3 if has_masks else 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.2))

# Row 0: clean
for i, p in enumerate(clean_paths):
    axes[0, i].imshow(Image.open(p), cmap="gray")
    axes[0, i].axis("off")

# Row 1: defective
for i, p in enumerate(defect_paths):
    axes[1, i].imshow(Image.open(p), cmap="gray")
    axes[1, i].axis("off")

# Row 2: masks (if available)
if has_masks:
    for i, p in enumerate(mask_paths):
        axes[2, i].imshow(Image.open(p), cmap="hot")
        axes[2, i].axis("off")

# Row labels
row_labels = ["Clean", "Defective", "Ground Truth\nMask"] if has_masks else ["Clean", "Defective"]
for row, label in enumerate(row_labels):
    axes[row, 0].text(
        -0.15, 0.5, label,
        transform=axes[row, 0].transAxes,
        fontsize=14, fontweight="bold",
        va="center", ha="right",
        color="#2C3E50"
    )

# Column count labels
total_clean = len(list(paths["clean"].glob("*.png")))
total_defect = len(list(paths["defect"].glob("*.png")))

fig.suptitle(
    f"Synthetic Dataset — {total_clean:,} clean  +  {total_defect:,} defective images",
    fontsize=16, fontweight="bold", y=1.02
)

plt.tight_layout()
Path("slides").mkdir(exist_ok=True)
fig.savefig("slides/slide5_dataset.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved: slides/slide5_dataset.png")
print(f"       {total_clean} clean, {total_defect} defective images shown")
plt.show()
