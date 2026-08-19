import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path
from torchvision import transforms

Path("slides").mkdir(exist_ok=True)

# ── Model ────────────────────────────────────────────────────────────
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),     # 0,1
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),    # 2,3
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),   # 4,5
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),  # 6,7
            nn.Flatten(),                                              # 8
            nn.Linear(256*16*16, latent_dim),                         # 9
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*16*16),                         # 0
            nn.Unflatten(1, (256, 16, 16)),                           # 1
            nn.ReLU(),                                                 # 2
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),  # 3,4
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 5,6
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),    # 7,8
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),  # 9,10
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('checkpoints/autoencoder_best.pth', map_location=device)
model = ConvAutoencoder(latent_dim=ckpt.get('latent_dim', 64)).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"Model loaded on {device}")

# ── Register hooks to capture intermediate activations ───────────────
activations = {}

def hook(name):
    def fn(module, inp, out):
        activations[name] = out.detach().cpu()
    return fn

model.encoder[1].register_forward_hook(hook('enc1_relu'))    # 32×128×128
model.encoder[3].register_forward_hook(hook('enc2_relu'))    # 64×64×64
model.encoder[5].register_forward_hook(hook('enc3_relu'))    # 128×32×32
model.encoder[7].register_forward_hook(hook('enc4_relu'))    # 256×16×16
model.encoder[9].register_forward_hook(hook('latent'))       # 64-dim
model.decoder[2].register_forward_hook(hook('dec_reshape'))  # 256×16×16
model.decoder[4].register_forward_hook(hook('dec1_relu'))    # 128×32×32
model.decoder[6].register_forward_hook(hook('dec2_relu'))    # 64×64×64
model.decoder[8].register_forward_hook(hook('dec3_relu'))    # 32×128×128
model.decoder[10].register_forward_hook(hook('output'))      # 1×256×256

# ── Load a good defective example ────────────────────────────────────
img_dir  = Path("dataset/defective/images")
mask_dir = Path("dataset/defective/masks")

transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((256, 256)), transforms.ToTensor()
])

# Scan a few images and pick one with clear defects
image_files = sorted(img_dir.glob("*.png"))[:50]
best_path = None
best_residual_sum = 0

for img_path in image_files:
    x = transform(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        r = model(x)
    res = np.abs(x.squeeze().cpu().numpy() - r.squeeze().cpu().numpy()).sum()
    if res > best_residual_sum:
        best_residual_sum = res
        best_path = img_path

print(f"Using: {best_path.name} (residual sum = {best_residual_sum:.0f})")

x = transform(Image.open(best_path)).unsqueeze(0).to(device)
with torch.no_grad():
    recon = model(x)

input_np = x.squeeze().cpu().numpy()
output_np = recon.squeeze().cpu().numpy()

# ── Build visualization stages ───────────────────────────────────────
# Each entry: (title, subtitle, image_2d, colormap, title_color)
BLUE = "#2980B9"
RED  = "#E74C3C"
GREEN = "#27AE60"

def mean_channels(act_name):
    """Get mean across channels for a multi-channel activation."""
    a = activations[act_name].squeeze(0)  # remove batch dim
    if a.dim() == 3:
        return a.mean(dim=0).numpy()
    return a.numpy()

def latent_as_image(act_name):
    """Reshape 64-dim latent vector into 8x8 for visualization."""
    a = activations[act_name].squeeze(0).numpy()
    return a.reshape(8, 8)

stages = [
    ("Input",              "1 × 256 × 256",    input_np,                        "gray",    BLUE),
    ("Conv2d + ReLU",      "32 × 128 × 128",   mean_channels('enc1_relu'),      "Blues",   BLUE),
    ("Conv2d + ReLU",      "64 × 64 × 64",     mean_channels('enc2_relu'),      "Blues",   BLUE),
    ("Conv2d + ReLU",      "128 × 32 × 32",    mean_channels('enc3_relu'),      "Blues",   BLUE),
    ("Conv2d + ReLU",      "256 × 16 × 16",    mean_channels('enc4_relu'),      "Blues",   BLUE),
    ("Latent Space",       "z = 64-dim",        latent_as_image('latent'),       "RdYlBu_r", RED),
    ("Linear + Reshape",   "256 × 16 × 16",    mean_channels('dec_reshape'),    "Greens",  GREEN),
    ("ConvT2d + ReLU",     "128 × 32 × 32",    mean_channels('dec1_relu'),      "Greens",  GREEN),
    ("ConvT2d + ReLU",     "64 × 64 × 64",     mean_channels('dec2_relu'),      "Greens",  GREEN),
    ("ConvT2d + ReLU",     "32 × 128 × 128",   mean_channels('dec3_relu'),      "Greens",  GREEN),
    ("Output",             "1 × 256 × 256",     output_np,                       "gray",    GREEN),
]

# ── Create figure ────────────────────────────────────────────────────
n = len(stages)
fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 4.5))

for ax, (title, subtitle, img, cmap, color) in zip(axes, stages):
    ax.imshow(img, cmap=cmap, aspect='equal')
    ax.set_title(title, fontsize=9, fontweight="bold", color=color, pad=8)
    ax.set_xlabel(subtitle, fontsize=7, color="#555", labelpad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)

# Add section brackets
fig.text(0.27, -0.02, "ENCODER", fontsize=11, fontweight="bold",
         color=BLUE, ha="center", transform=fig.transFigure)
fig.text(0.50, -0.02, "LATENT", fontsize=11, fontweight="bold",
         color=RED, ha="center", transform=fig.transFigure)
fig.text(0.77, -0.02, "DECODER", fontsize=11, fontweight="bold",
         color=GREEN, ha="center", transform=fig.transFigure)

fig.suptitle("Feature Maps Through the Autoencoder Pipeline",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout(pad=0.8)
fig.savefig("slides/slide7_activations.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("\nSaved: slides/slide7_activations.png")
plt.close(fig)

# ── Also save a version with the residual/detection appended ─────────
THRESHOLD = 0.08
residual = np.abs(input_np - output_np)
detection = (residual > THRESHOLD).astype(float)

gt_mask = None
mask_path = mask_dir / f"{best_path.stem}_mask.png"
if mask_path.exists():
    gt_raw = np.array(Image.open(mask_path).convert("L")).astype(float) / 255.0
    if gt_raw.shape != detection.shape:
        gt_raw = np.array(
            Image.fromarray((gt_raw * 255).astype(np.uint8)).resize((256, 256))
        ) / 255.0
    gt_mask = (gt_raw > 0.5).astype(float)

extra_cols = [
    ("Input",          input_np,   "gray"),
    ("Reconstruction", output_np,  "gray"),
    ("Residual Map",   residual,   "hot"),
    ("Detection",      detection,  "Reds"),
]
if gt_mask is not None:
    extra_cols.append(("Ground Truth", gt_mask, "Greens"))

n2 = len(extra_cols)
fig2, axes2 = plt.subplots(1, n2, figsize=(4.2 * n2, 4.5))

for ax, (title, data, cmap) in zip(axes2, extra_cols):
    vmax = None if cmap == "hot" else 1
    ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")

if gt_mask is not None:
    inter = (detection * gt_mask).sum()
    union = ((detection + gt_mask) > 0).sum()
    iou = inter / union if union > 0 else 0
    axes2[-1].set_xlabel(f"IoU = {iou:.3f}", fontsize=11,
                          fontweight="bold", color="#27AE60", labelpad=10)

fig2.suptitle("Defect Detection Result", fontsize=15, fontweight="bold", y=1.0)
fig2.tight_layout(pad=1.0)
fig2.savefig("slides/slide7_detection_pipeline.png", dpi=180,
             bbox_inches="tight", facecolor="white")
print("Saved: slides/slide7_detection_pipeline.png")
plt.close(fig2)

print("\nDone!")
print("  1. slides/slide7_activations.png       — feature maps at every layer")
print("  2. slides/slide7_detection_pipeline.png — final detection result")
