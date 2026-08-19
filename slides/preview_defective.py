import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

defect = sorted(Path("dataset/defective/images").glob("*.png"))[:5]
masks  = sorted(Path("dataset/defective/masks").glob("*.png"))[:5]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))

for i, (d, m) in enumerate(zip(defect, masks)):
    axes[0, i].imshow(Image.open(d), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(Image.open(m), cmap="hot")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Defective Image", fontsize=13, fontweight="bold")
axes[1, 0].set_ylabel("Ground Truth Mask", fontsize=13, fontweight="bold")

fig.suptitle("Defective Images with Injected Defects (10,000 total)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("slides/slide4b_defective_preview.png", dpi=150, bbox_inches="tight")
print("Saved: slides/slide4b_defective_preview.png")
plt.show()
