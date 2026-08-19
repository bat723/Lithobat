import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

clean = sorted(Path("dataset/clean").glob("*.png"))[:6]

fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
fig.suptitle("Clean Training Images (10,000 total)", fontsize=14, fontweight="bold")

for ax, c in zip(axes, clean):
    ax.imshow(Image.open(c), cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.savefig("slides/slide4a_clean_preview.png", dpi=150, bbox_inches="tight")
print("Saved: slides/slide4a_clean_preview.png")
plt.show()
