import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pandas as pd

overlay_dir = Path("inference_results_t008/overlays")

df = pd.read_csv("inference_results_t008/inference_summary.csv")

perfect = df[df["iou"] == 1.0].iloc[0]["filename"]
good    = df[(df["iou"] > 0.68) & (df["iou"] < 0.75)].iloc[0]["filename"]
partial = df[(df["iou"] > 0.28) & (df["iou"] < 0.32)].iloc[0]["filename"]
missed  = df[df["iou"] == 0.0].iloc[0]["filename"]

examples = {
    "Perfect (IoU = 1.0)": perfect,
    "Good (IoU ~ 0.7)":    good,
    "Partial (IoU ~ 0.3)": partial,
    "Missed (IoU = 0.0)":  missed,
}

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

for ax, (label, fname) in zip(axes, examples.items()):
    img = Image.open(overlay_dir / fname)
    ax.imshow(img)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("slides/slide6_overlay_examples.png", dpi=150, bbox_inches="tight")
print("Saved: slides/slide6_overlay_examples.png")
plt.show()
