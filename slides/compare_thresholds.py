import pandas as pd

results = {
    "0.02": "inference_results/inference_summary.csv",
    "0.05": "inference_results_t005/inference_summary.csv",
    "0.08": "inference_results_t008/inference_summary.csv",
    "0.10": "inference_results_t010/inference_summary.csv",
    "0.15": "inference_results_t015/inference_summary.csv",
}

print(f"{'Threshold':<12} {'IoU mean':<12} {'IoU>0.5':<12} {'Missed (IoU=0)':<16} {'Avg flagged px'}")
print("-" * 70)

for t, path in results.items():
    try:
        df = pd.read_csv(path)
        print(
            f"{t:<12} "
            f"{df['iou'].mean():<12.4f} "
            f"{(df['iou'] > 0.5).sum():<12} "
            f"{(df['iou'] == 0.0).sum():<16} "
            f"{df['defect_px'].mean():.0f}"
        )
    except FileNotFoundError:
        print(f"{t:<12} not found")
