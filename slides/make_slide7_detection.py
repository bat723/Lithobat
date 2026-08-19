import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms

Path("slides").mkdir(exist_ok=True)

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*16*16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*16*16),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('checkpoints/autoencoder_best.pth', map_location=device)
model = ConvAutoencoder(latent_dim=ckpt.get('latent_dim', 64)).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"Model loaded on {device}")

img_dir  = Path("dataset/defective/images")
mask_dir = Path("dataset/defective/masks")

transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((256, 256)), transforms.ToTensor()
])

THRESHOLD = 0.08

image_files = sorted(img_dir.glob("*.png"))
print(f"Found {len(image_files)} defective images, scanning first 200...")

results = []
for img_path in image_files[:200]:
    img = Image.open(img_path)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(x)

    x_np = x.squeeze().cpu().numpy()
    r_np = recon.squeeze().cpu().numpy()
    residual = np.abs(x_np - r_np)
    detection = (residual > THRESHOLD).astype(float)

    gt_mask = None
    iou = 0.0
    mask_path = mask_dir / f"{img_path.stem}_mask.png"

    if mask_path.exists():
        gt_raw = np.array(Image.open(mask_path).convert("L")).astype(float) / 255.0
        if gt_raw.shape != detection.shape:
            gt_raw = np.array(
                Image.fromarray((gt_raw * 255).astype(np.uint8)).resize((256, 256))
            ) / 255.0
        gt_mask = (gt_raw > 0.5).astype(float)
        intersection = (detection * gt_mask).sum()
        union = ((detection + gt_mask) > 0).sum()
        iou = intersection / union if union > 0 else 0.0

    results.append({
        "path": img_path,
        "input": x_np,
        "recon": r_np,
        "residual": residual,
        "detection": detection,
        "gt_mask": gt_mask,
        "iou": iou,
    })

results_valid = [r for r in results if r["iou"] > 0.01]
results_valid.sort(key=lambda r: r["iou"], reverse=True)
print(f"Found {len(results_valid)} images with IoU > 0.01")

if len(results_valid) == 0:
    print("ERROR: No valid detections.")
    exit(1)

best = results_valid[0]
print(f"Best IoU: {best['iou']:.3f} ({best['path'].name})")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Step-by-step pipeline (no arrows)
# ═══════════════════════════════════════════════════════════════════
col_data = [
    ("Input (defective)",        best["input"],     "gray"),
    ("Reconstruction",           best["recon"],     "gray"),
    ("Residual Map",             best["residual"],  "hot"),
    ("Detection Mask",           best["detection"], "Reds"),
]
if best["gt_mask"] is not None:
    col_data.append(("Ground Truth", best["gt_mask"], "Greens"))

n_cols = len(col_data)
fig1, axes1 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))

for ax, (title, data, cmap) in zip(axes1, col_data):
    vmax = None if cmap == "hot" else 1
    ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")

if best["gt_mask"] is not None:
    axes1[-1].set_xlabel(f"IoU = {best['iou']:.3f}", fontsize=11,
                          fontweight="bold", color="#27AE60", labelpad=10)

fig1.suptitle("Autoencoder Defect Detection", fontsize=15, fontweight="bold", y=1.0)
fig1.tight_layout(pad=1.0)
fig1.savefig("slides/slide7_detection_pipeline.png", dpi=180,
             bbox_inches="tight", facecolor="white")
print("\nSaved: slides/slide7_detection_pipeline.png")
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Range of detection quality
# ═══════════════════════════════════════════════════════════════════
n = len(results_valid)
if n >= 4:
    picks = [results_valid[0], results_valid[n//3],
             results_valid[2*n//3], results_valid[-1]]
else:
    picks = results_valid[:4]

labels = ["Strong", "Good", "Partial", "Weak"]
n_rows = len(picks)

fig2, axes2 = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
if n_rows == 1:
    axes2 = axes2.reshape(1, -1)

col_titles = ["Input", "Residual Map", "Detection Mask", "Ground Truth"]

for row in range(n_rows):
    ex = picks[row]
    axes2[row, 0].imshow(ex["input"], cmap="gray")
    axes2[row, 1].imshow(ex["residual"], cmap="hot")
    axes2[row, 2].imshow(ex["detection"], cmap="Reds", vmin=0, vmax=1)
    if ex["gt_mask"] is not None:
        axes2[row, 3].imshow(ex["gt_mask"], cmap="Greens", vmin=0, vmax=1)
        axes2[row, 3].set_xlabel(f"IoU = {ex['iou']:.3f}", fontsize=9)
    else:
        axes2[row, 3].text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=axes2[row, 3].transAxes)
    for col in range(4):
        axes2[row, col].axis("off")
    axes2[row, 0].set_ylabel(labels[row], fontsize=12, fontweight="bold",
                               rotation=90, labelpad=15, color="#2C3E50")

for col, title in enumerate(col_titles):
    axes2[0, col].set_title(title, fontsize=12, fontweight="bold", pad=10)

fig2.suptitle("Detection Quality Across Examples",
              fontsize=14, fontweight="bold", y=1.01)
fig2.tight_layout(pad=0.5)
fig2.savefig("slides/slide7_detection_range.png", dpi=180,
             bbox_inches="tight", facecolor="white")
print("Saved: slides/slide7_detection_range.png")
plt.close(fig2)

print("\nDone!")
print("  1. slides/slide7_detection_pipeline.png")
print("  2. slides/slide7_detection_range.png")
