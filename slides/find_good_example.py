import pandas as pd
df = pd.read_csv("inference_results_t008/inference_summary.csv")
good = df[(df["iou"] > 0.65) & (df["iou"] < 0.80)].nlargest(5, "mse")
print("Good detection examples:")
print(good[["filename", "mse", "iou"]])
