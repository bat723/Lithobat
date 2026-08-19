import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("inference_results/inference_summary.csv")

# Overall summary
print(df.describe())

# Images the model is most confident contain defects
print("\nTop 10 most anomalous:")
print(df.nlargest(10, "mse")[["filename", "mse", "iou"]])

# Images that look clean (possible missed detections)
print("\nBottom 10 (lowest MSE - possibly missed defects):")
print(df.nsmallest(10, "mse")[["filename", "mse", "iou"]])

# Plot MSE distribution
plt.figure(figsize=(10, 4))
plt.hist(df["mse"], bins=100, color="steelblue", edgecolor="none")
plt.axvline(0.02, color="red", linestyle="--", label="threshold=0.02")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("mse_distribution.png", dpi=120)
plt.show()
