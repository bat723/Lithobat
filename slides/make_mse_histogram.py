import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("inference_results_t008/inference_summary.csv")

plt.figure(figsize=(8, 4))
plt.hist(df["mse"], bins=100, color="steelblue", edgecolor="none", alpha=0.9)
plt.axvline(0.08, color="red", linestyle="--", linewidth=2, label="threshold = 0.08")
plt.xlabel("Reconstruction Error (MSE)", fontsize=12)
plt.ylabel("Image Count", fontsize=12)
plt.title("MSE Distribution Across 10,000 Defective Images", fontsize=13)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("slides/slide6_mse_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: slides/slide6_mse_distribution.png")
plt.show()
