import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

clean  = sorted(Path("dataset/clean").glob("*.png"))[:4]
defect = sorted(Path("dataset/defective/images").glob("*.png"))[:4]
masks  = sorted(Path("dataset/defective/masks").glob("*.png"))[:4]

fig, axes = plt.subplots(3, 4, figsize=(14, 9))

row_labels = ["Clean", "Defective", "Ground Truth Mask"]
for i, (c, d, m) in enumerate(zip(clean, defect, masks)):
    axes[0, i].imshow(Image.open(c), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(Image.open(d), cmap="gray")
    axes[1, i].axis("off")
    axes[2, i].imshow(Image.open(m), cmap="hot")
    axes[2, i].axis("off")

for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("slides/slide4_dataset_preview.png", dpi=150, bbox_inches="tight")
print("Saved: slides/slide4_dataset_preview.png")
plt.show()
