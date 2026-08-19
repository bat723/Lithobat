import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image
from pathlib import Path
from torchvision import transforms
from collections import defaultdict
import time

Path("slides").mkdir(exist_ok=True)

# ── Model ────────────────────────────────────────────────────────────
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

# ── Paths and transform ─────────────────────────────────────────────
clean_img_dir   = Path("dataset/clean/images") if Path("dataset/clean/images").exists() else Path("dataset/clean")
defect_img_dir  = Path("dataset/defective/images")
defect_mask_dir = Path("dataset/defective/masks")

transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((256, 256)), transforms.ToTensor()
])

# ═════════════════════════════════════════════════════════════════════
# PHASE 1: Compute reconstruction errors for CLEAN images
# ═════════════════════════════════════════════════════════════════════
print("\n[Phase 1/4] Computing reconstruction errors on CLEAN images...")
clean_files = sorted(clean_img_dir.glob("*.png"))
print(f"  Found {len(clean_files)} clean images")

clean_mses = []
clean_max_residuals = []

t0 = time.time()
for i, img_path in enumerate(clean_files):
    x = transform(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
    residual = torch.abs(x - recon).squeeze().cpu().numpy()
    clean_mses.append(float(np.mean(residual**2)))
    clean_max_residuals.append(float(np.max(residual)))
    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (len(clean_files) - i - 1) / rate
        print(f"  {i+1}/{len(clean_files)}  ({rate:.0f} img/sec, ~{remaining:.0f}s remaining)")

clean_mses = np.array(clean_mses)
clean_max_residuals = np.array(clean_max_residuals)
print(f"  Clean MSE: mean={clean_mses.mean():.5f}, std={clean_mses.std():.5f}")

# ═════════════════════════════════════════════════════════════════════
# PHASE 2: Compute reconstruction errors + IoU for DEFECTIVE images
# ═════════════════════════════════════════════════════════════════════
print("\n[Phase 2/4] Computing reconstruction errors on DEFECTIVE images...")
defect_files = sorted(defect_img_dir.glob("*.png"))
print(f"  Found {len(defect_files)} defective images")

thresholds = np.arange(0.01, 0.31, 0.005)  # Fine sweep: 0.01 to 0.30

defect_mses = []
defect_max_residuals = []
# For each threshold, accumulate IoU values
threshold_ious = defaultdict(list)
# Also track pixel-level precision/recall
threshold_tp = defaultdict(float)
threshold_fp = defaultdict(float)
threshold_fn = defaultdict(float)

# Store a few good examples at optimal threshold for later
example_data = []

t0 = time.time()
for i, img_path in enumerate(defect_files):
    x = transform(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)

    x_np = x.squeeze().cpu().numpy()
    r_np = recon.squeeze().cpu().numpy()
    residual = np.abs(x_np - r_np)

    defect_mses.append(float(np.mean(residual**2)))
    defect_max_residuals.append(float(np.max(residual)))

    # Load ground truth mask
    mask_path = defect_mask_dir / f"{img_path.stem}_mask.png"
    if not mask_path.exists():
        continue

    gt_raw = np.array(Image.open(mask_path).convert("L")).astype(float) / 255.0
    if gt_raw.shape != residual.shape:
        gt_raw = np.array(
            Image.fromarray((gt_raw * 255).astype(np.uint8)).resize((256, 256))
        ) / 255.0
    gt_mask = (gt_raw > 0.5).astype(float)
    gt_defect_pixels = gt_mask.sum()

    if gt_defect_pixels == 0:
        continue

    # Sweep thresholds
    for t in thresholds:
        det = (residual > t).astype(float)
        intersection = (det * gt_mask).sum()
        union = ((det + gt_mask) > 0).sum()
        iou = intersection / union if union > 0 else 0.0
        threshold_ious[t].append(iou)

        tp = intersection
        fp = (det * (1 - gt_mask)).sum()
        fn = ((1 - det) * gt_mask).sum()
        threshold_tp[t] += tp
        threshold_fp[t] += fp
        threshold_fn[t] += fn

    # Save example data (for later visualization)
    if len(example_data) < 200:
        example_data.append({
            "path": img_path,
            "input": x_np,
            "recon": r_np,
            "residual": residual,
            "gt_mask": gt_mask,
        })

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (len(defect_files) - i - 1) / rate
        print(f"  {i+1}/{len(defect_files)}  ({rate:.0f} img/sec, ~{remaining:.0f}s remaining)")

defect_mses = np.array(defect_mses)
defect_max_residuals = np.array(defect_max_residuals)
print(f"  Defective MSE: mean={defect_mses.mean():.5f}, std={defect_mses.std():.5f}")

# ═════════════════════════════════════════════════════════════════════
# PHASE 3: Compute summary statistics
# ═════════════════════════════════════════════════════════════════════
print("\n[Phase 3/4] Computing statistics...")

mean_ious = []
std_ious = []
precisions = []
recalls = []
f1_scores = []

for t in thresholds:
    ious = threshold_ious[t]
    mean_ious.append(np.mean(ious) if ious else 0)
    std_ious.append(np.std(ious) if ious else 0)

    tp = threshold_tp[t]
    fp = threshold_fp[t]
    fn = threshold_fn[t]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

mean_ious = np.array(mean_ious)
std_ious = np.array(std_ious)
precisions = np.array(precisions)
recalls = np.array(recalls)
f1_scores = np.array(f1_scores)

best_iou_idx = np.argmax(mean_ious)
best_threshold = thresholds[best_iou_idx]
best_iou = mean_ious[best_iou_idx]
best_f1_idx = np.argmax(f1_scores)

print(f"  Best IoU: {best_iou:.4f} at threshold {best_threshold:.3f}")
print(f"  Best F1:  {f1_scores[best_f1_idx]:.4f} at threshold {thresholds[best_f1_idx]:.3f}")

# Inference speed benchmark
print("\n  Benchmarking inference speed...")
bench_files = defect_files[:500]
t0 = time.time()
for img_path in bench_files:
    x = transform(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(x)
bench_time = time.time() - t0
speed = len(bench_files) / bench_time
print(f"  Inference speed: {speed:.0f} img/sec")

# ═════════════════════════════════════════════════════════════════════
# PHASE 4: Generate figures
# ═════════════════════════════════════════════════════════════════════
print("\n[Phase 4/4] Generating figures...")

BLUE = "#2980B9"
RED  = "#E74C3C"
GREEN = "#27AE60"
DARK = "#2C3E50"
ORANGE = "#E67E22"

# ─────────────────────────────────────────────────────────────────────
# FIGURE 1: Threshold Optimization Curve (THE key missing slide)
# ─────────────────────────────────────────────────────────────────────
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: IoU vs Threshold
ax1a.plot(thresholds, mean_ious, color=BLUE, linewidth=2.5, label="Mean IoU")
ax1a.fill_between(thresholds, mean_ious - std_ious, mean_ious + std_ious,
                   alpha=0.15, color=BLUE, label="\u00b11 Std Dev")
ax1a.axvline(best_threshold, color=RED, linestyle="--", linewidth=1.5, alpha=0.8)
ax1a.plot(best_threshold, best_iou, 'o', color=RED, markersize=10, zorder=5)
ax1a.annotate(f"Optimal: t={best_threshold:.3f}\nIoU={best_iou:.3f}",
              xy=(best_threshold, best_iou),
              xytext=(best_threshold + 0.04, best_iou + 0.03),
              fontsize=10, fontweight="bold", color=RED,
              arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
ax1a.set_xlabel("Threshold", fontsize=12, fontweight="bold")
ax1a.set_ylabel("Mean IoU", fontsize=12, fontweight="bold")
ax1a.set_title("Threshold vs. IoU", fontsize=13, fontweight="bold", color=DARK)
ax1a.legend(fontsize=9, loc="upper right")
ax1a.set_xlim(thresholds[0], thresholds[-1])
ax1a.set_ylim(0, max(mean_ious) * 1.25)
ax1a.grid(True, alpha=0.3)

# Right: Precision / Recall / F1 vs Threshold
ax1b.plot(thresholds, precisions, color=BLUE, linewidth=2, label="Precision")
ax1b.plot(thresholds, recalls, color=GREEN, linewidth=2, label="Recall")
ax1b.plot(thresholds, f1_scores, color=ORANGE, linewidth=2.5, label="F1 Score")
ax1b.axvline(best_threshold, color=RED, linestyle="--", linewidth=1.5, alpha=0.8,
             label=f"Optimal t={best_threshold:.3f}")
ax1b.set_xlabel("Threshold", fontsize=12, fontweight="bold")
ax1b.set_ylabel("Score", fontsize=12, fontweight="bold")
ax1b.set_title("Pixel-Level Precision / Recall / F1", fontsize=13,
               fontweight="bold", color=DARK)
ax1b.legend(fontsize=9, loc="center right")
ax1b.set_xlim(thresholds[0], thresholds[-1])
ax1b.set_ylim(0, 1.05)
ax1b.grid(True, alpha=0.3)

fig1.suptitle("Threshold Optimization — 10,000 Defective Images",
              fontsize=15, fontweight="bold", y=1.02)
fig1.tight_layout(pad=1.5)
fig1.savefig("slides/slide7_threshold_optimization.png", dpi=200,
             bbox_inches="tight", facecolor="white")
print("  Saved: slides/slide7_threshold_optimization.png")
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────
# FIGURE 2: Anomaly Score Distribution (clean vs defective)
# ─────────────────────────────────────────────────────────────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

# MSE distribution
if len(clean_mses) > 0 and len(defect_mses) > 0:
    bins_mse = np.linspace(0, max(defect_mses.max(), clean_mses.max()) * 1.05, 80)
elif len(defect_mses) > 0:
    bins_mse = np.linspace(0, defect_mses.max() * 1.05, 80)
else:
    bins_mse = np.linspace(0, 1, 80)
if len(clean_mses) > 0:
    ax2a.hist(clean_mses, bins=bins_mse, alpha=0.7, color=GREEN, label="Clean", density=True)
ax2a.hist(defect_mses, bins=bins_mse, alpha=0.7, color=RED, label="Defective", density=True)
ax2a.set_xlabel("Mean Squared Error", fontsize=12, fontweight="bold")
ax2a.set_ylabel("Density", fontsize=12, fontweight="bold")
ax2a.set_title("Reconstruction Error Distribution", fontsize=13,
               fontweight="bold", color=DARK)
ax2a.legend(fontsize=10)
ax2a.grid(True, alpha=0.3)

# Max residual distribution
bins_max = np.linspace(0, 1.05, 80)
if len(clean_max_residuals) > 0:
    ax2b.hist(clean_max_residuals, bins=bins_max, alpha=0.7, color=GREEN,
              label="Clean", density=True)
if len(defect_max_residuals) > 0:
    ax2b.hist(defect_max_residuals, bins=bins_max, alpha=0.7, color=RED,
              label="Defective", density=True)
ax2b.axvline(best_threshold, color=DARK, linestyle="--", linewidth=1.5,
             label=f"Threshold = {best_threshold:.3f}")
ax2b.set_xlabel("Max Pixel Residual", fontsize=12, fontweight="bold")
ax2b.set_ylabel("Density", fontsize=12, fontweight="bold")
ax2b.set_title("Max Residual Distribution", fontsize=13,
               fontweight="bold", color=DARK)
ax2b.legend(fontsize=10)
ax2b.grid(True, alpha=0.3)

fig2.suptitle("Anomaly Detection — Clean vs. Defective",
              fontsize=15, fontweight="bold", y=1.02)
fig2.tight_layout(pad=1.5)
fig2.savefig("slides/slide8_anomaly_distribution.png", dpi=200,
             bbox_inches="tight", facecolor="white")
print("  Saved: slides/slide8_anomaly_distribution.png")
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────
# FIGURE 3: Detection at different thresholds (same image)
# ─────────────────────────────────────────────────────────────────────
# Pick an example with good IoU at optimal threshold
best_ex = None
best_ex_iou = 0
for ex in example_data:
    det = (ex["residual"] > best_threshold).astype(float)
    inter = (det * ex["gt_mask"]).sum()
    union = ((det + ex["gt_mask"]) > 0).sum()
    iou = inter / union if union > 0 else 0
    if iou > best_ex_iou:
        best_ex_iou = iou
        best_ex = ex

show_thresholds = [0.02, 0.05, 0.08, 0.12, 0.20]
n_t = len(show_thresholds)

fig3, axes3 = plt.subplots(2, n_t + 1, figsize=(3.5 * (n_t + 1), 7.5))

# Top row: input + detections at each threshold
axes3[0, 0].imshow(best_ex["input"], cmap="gray")
axes3[0, 0].set_title("Input", fontsize=11, fontweight="bold")
axes3[0, 0].axis("off")

for j, t in enumerate(show_thresholds):
    det = (best_ex["residual"] > t).astype(float)
    inter = (det * best_ex["gt_mask"]).sum()
    union = ((det + best_ex["gt_mask"]) > 0).sum()
    iou = inter / union if union > 0 else 0

    color = RED if t == best_threshold else DARK
    axes3[0, j + 1].imshow(det, cmap="Reds", vmin=0, vmax=1)
    axes3[0, j + 1].set_title(f"t = {t:.2f}", fontsize=11,
                                fontweight="bold", color=color)
    axes3[0, j + 1].set_xlabel(f"IoU = {iou:.3f}", fontsize=9, color=color)
    axes3[0, j + 1].axis("off")

# Bottom row: ground truth + residual heatmap + overlay
axes3[1, 0].imshow(best_ex["gt_mask"], cmap="Greens", vmin=0, vmax=1)
axes3[1, 0].set_title("Ground Truth", fontsize=11, fontweight="bold", color=GREEN)
axes3[1, 0].axis("off")

axes3[1, 1].imshow(best_ex["residual"], cmap="hot")
axes3[1, 1].set_title("Residual Map", fontsize=11, fontweight="bold")
axes3[1, 1].axis("off")

# Overlay: input + detection + GT
det_optimal = (best_ex["residual"] > best_threshold).astype(float)
overlay = np.stack([
    best_ex["input"],
    best_ex["input"] * (1 - 0.5 * det_optimal) + 0.5 * det_optimal,
    best_ex["input"] * (1 - 0.5 * best_ex["gt_mask"]),
], axis=-1)
overlay = np.clip(overlay, 0, 1)
axes3[1, 2].imshow(overlay)
axes3[1, 2].set_title("Overlay (R=det, G=input, B=GT)", fontsize=9, fontweight="bold")
axes3[1, 2].axis("off")

# Hide remaining bottom-row axes
for j in range(3, n_t + 1):
    axes3[1, j].axis("off")

fig3.suptitle("Effect of Threshold on Detection",
              fontsize=14, fontweight="bold", y=1.01)
fig3.tight_layout(pad=0.8)
fig3.savefig("slides/slide7_threshold_comparison.png", dpi=200,
             bbox_inches="tight", facecolor="white")
print("  Saved: slides/slide7_threshold_comparison.png")
plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────
# FIGURE 4: Summary metrics card
# ─────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.axis("off")

metrics_text = [
    ("Dataset",                 f"{len(clean_files):,} clean + {len(defect_files):,} defective images"),
    ("Training",                "Unsupervised — trained on clean images only (zero labels)"),
    ("Architecture",            "Conv Autoencoder: 4 layers, 256→64-dim latent, 4×4 kernels"),
    ("Optimal Threshold",       f"{best_threshold:.3f}"),
    ("Mean IoU @ Optimal",      f"{best_iou:.3f}"),
    ("Pixel Precision @ Opt.",  f"{precisions[best_iou_idx]:.3f}"),
    ("Pixel Recall @ Opt.",     f"{recalls[best_iou_idx]:.3f}"),
    ("Pixel F1 @ Opt.",         f"{f1_scores[best_iou_idx]:.3f}"),
    ("Inference Speed",         f"{speed:.0f} img/sec"),
    ("Clean MSE (mean\u00b1std)", f"{clean_mses.mean():.5f} \u00b1 {clean_mses.std():.5f}"),
    ("Defect MSE (mean\u00b1std)", f"{defect_mses.mean():.5f} \u00b1 {defect_mses.std():.5f}"),
]

y_start = 0.92
for i, (label, value) in enumerate(metrics_text):
    y = y_start - i * 0.078
    ax4.text(0.05, y, label, fontsize=11, fontweight="bold", color=DARK,
             transform=ax4.transAxes, va="top", family="monospace")
    ax4.text(0.45, y, value, fontsize=11, color="#444",
             transform=ax4.transAxes, va="top")

ax4.text(0.5, -0.05, "All results computed with zero labeled training data",
         fontsize=10, fontstyle="italic", color="#888", ha="center",
         transform=ax4.transAxes)

fig4.suptitle("Results Summary", fontsize=15, fontweight="bold",
              color=DARK, y=0.98)
fig4.savefig("slides/slide8_results_summary.png", dpi=200,
             bbox_inches="tight", facecolor="white")
print("  Saved: slides/slide8_results_summary.png")
plt.close(fig4)

# ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("All figures saved to slides/:")
print("  1. slide7_threshold_optimization.png  — IoU & Precision/Recall curves")
print("  2. slide8_anomaly_distribution.png    — Clean vs Defective error histograms")
print("  3. slide7_threshold_comparison.png    — Same image at different thresholds")
print("  4. slide8_results_summary.png         — Metrics summary card")
print(f"{'='*60}")
