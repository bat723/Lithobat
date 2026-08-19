import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.insert(0, "scripts")
from autoencoder import ConvAutoencoder

device = torch.device("cpu")
model = ConvAutoencoder(latent_dim=64)
ckpt = torch.load("checkpoints/autoencoder_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# CHANGE THIS FILENAME if find_good_example.py gives a different one
img_path  = "dataset/defective/images/defect_001235.png"
mask_path = "dataset/defective/masks/defect_001235_mask.png"

img  = np.array(Image.open(img_path).convert("L")).astype(np.float32) / 255.0
mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0

tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    recon = model(tensor).squeeze().numpy()

residual = (img - recon) ** 2
binary   = (residual > 0.08).astype(np.float32)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
panels = [
    (img,      "Input (defective)", "gray"),
    (recon,    "Reconstruction",    "gray"),
    (residual, "Residual map",      "hot"),
    (binary,   "Detected defect",   "Reds"),
]

for ax, (data, title, cmap) in zip(axes, panels):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("slides/slide5_detection_pipeline.png", dpi=150, bbox_inches="tight")
print("Saved: slides/slide5_detection_pipeline.png")
plt.show()
