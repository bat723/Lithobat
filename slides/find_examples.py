import pandas as pd

df = pd.read_csv("inference_results_t008/inference_summary.csv")

print("PERFECT (IoU = 1.0):")
print(df[df["iou"] == 1.0].head(1)[["filename", "mse", "iou"]])

print("\nGOOD (IoU ~ 0.7):")
print(df[(df["iou"] > 0.68) & (df["iou"] < 0.75)].head(1)[["filename", "mse", "iou"]])

print("\nPARTIAL (IoU ~ 0.3):")
print(df[(df["iou"] > 0.28) & (df["iou"] < 0.32)].head(1)[["filename", "mse", "iou"]])

print("\nMISSED (IoU = 0.0):")
print(df[df["iou"] == 0.0].head(1)[["filename", "mse", "iou"]])
