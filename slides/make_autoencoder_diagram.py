import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path

Path("slides").mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Architecture Diagram (no info box)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
ax.set_xlim(-1, 18)
ax.set_ylim(-2, 8)
ax.axis("off")
fig.patch.set_facecolor("white")

layers = [
    # (x,   width, height, color,     label_top,              label_bot)
    (0.0,  0.8,   6.0,  "#5DADE2",  "Input\n1 x 256 x 256",      "65,536 px"),
    (1.5,  0.9,   5.0,  "#3498DB",  "Conv2d\n3x3, s2",       "8 x 128 x 128"),
    (3.2,  1.0,   4.0,  "#2E86C1",  "Conv2d\n3x3, s2",       "16 x 64 x 64"),
    (5.0,  1.0,   3.0,  "#2874A6",  "Conv2d\n3x3, s2",       "32 x 32 x 32"),
    (6.8,  0.6,   2.2,  "#1B4F72",  "Flatten\n+ Linear",     "32,768 -> 64"),
    (8.0,  0.5,   1.2,  "#E74C3C",  "Latent\nz = 64-dim",    "64-dim"),
    (9.2,  0.6,   2.2,  "#1D8348",  "Linear\n+ Reshape",     "64 -> 32,768"),
    (10.8, 1.0,   3.0,  "#239B56",  "ConvT2d\n4x4, s2",      "16 x 64 x 64"),
    (12.6, 1.0,   4.0,  "#2ECC71",  "ConvT2d\n4x4, s2",      "8 x 128 x 128"),
    (14.4, 0.9,   5.0,  "#58D68D",  "ConvT2d\n4x4, s2",      "1 x 256 x 256"),
    (16.0, 0.8,   6.0,  "#ABEBC6",  "Output\n1 x 256 x 256", "65,536 px"),
]

centers_x = []
for (x, w, h, color, label_top, label_bot) in layers:
    y = (6 - h) / 2
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white", linewidth=1.5,
        alpha=0.92
    )
    ax.add_patch(rect)
    cx = x + w / 2
    centers_x.append(cx)

    ax.text(cx, y + h + 0.25, label_top, fontsize=8, ha="center", va="bottom",
            fontweight="bold", color="#2C3E50",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.text(cx, y - 0.25, label_bot, fontsize=7, ha="center", va="top",
            color="#555", fontstyle="italic")

for i in range(len(layers) - 1):
    x1 = layers[i][0] + layers[i][1]
    x2 = layers[i + 1][0]
    mid_y = 3.0
    ax.annotate("",
                xy=(x2 - 0.05, mid_y), xytext=(x1 + 0.05, mid_y),
                arrowprops=dict(arrowstyle="-|>", color="#888",
                                lw=1.5, mutation_scale=12))

activations = [
    (1, "ReLU"),
    (2, "ReLU"),
    (3, "ReLU"),
    (7, "ReLU"),
    (8, "ReLU"),
    (9, "Sigmoid"),
]

for (idx, act_name) in activations:
    x1 = layers[idx][0] + layers[idx][1]
    x2 = layers[idx + 1][0]
    mid_x = (x1 + x2) / 2
    color = "#E74C3C" if act_name == "Sigmoid" else "#8E44AD"
    ax.text(mid_x, 3.55, act_name, fontsize=6, ha="center", va="bottom",
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=color, linewidth=0.8, alpha=0.9))

sections = [
    (0.0, 7.4,  "ENCODER", "#2874A6"),
    (7.8, 8.7,  "LATENT",  "#E74C3C"),
    (9.0, 16.8, "DECODER", "#239B56"),
]

for (x1, x2, label, color) in sections:
    y = -1.3
    ax.annotate("", xy=(x1, y), xytext=(x2, y),
                arrowprops=dict(arrowstyle="|-|", color=color, lw=2))
    ax.text((x1 + x2) / 2, y - 0.4, label, fontsize=11, ha="center",
            fontweight="bold", color=color)

ax.annotate("1,024x compression",
            xy=(8.25, 1.2), xytext=(8.25, -0.8),
            fontsize=9, ha="center", fontweight="bold", color="#E74C3C",
            arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.5))

plt.tight_layout()
fig.savefig("slides/slide6_autoencoder_detailed.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("Saved: slides/slide6_autoencoder_detailed.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Training Info Box (separate, no Optimizer)
# ═══════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 2.5))
ax2.axis("off")
fig2.patch.set_facecolor("white")

info_lines = [
    ("Loss:",              r"MSE   $\mathcal{L} = \frac{1}{N}\sum(x_i - \hat{x}_i)^2$"),
    ("Training Data:",     "10,000 clean images"),
    ("Total Parameters:",  "4.24M  with 99.6% in Linear layers"),
]

y_pos = 0.85
for label, value in info_lines:
    ax2.text(0.05, y_pos, label, fontsize=12, fontweight="bold",
             va="center", ha="left", transform=ax2.transAxes, color="#2C3E50")
    ax2.text(0.38, y_pos, value, fontsize=12,
             va="center", ha="left", transform=ax2.transAxes, color="#555")
    y_pos -= 0.30

fig2.tight_layout()
fig2.savefig("slides/slide6_training_info.png", dpi=200, bbox_inches="tight",
             facecolor="white")
print("Saved: slides/slide6_training_info.png")
plt.close(fig2)

print("\nDone. Two files generated:")
print("  1. slides/slide6_autoencoder_detailed.png  (architecture diagram)")
print("  2. slides/slide6_training_info.png          (training specs box)")
