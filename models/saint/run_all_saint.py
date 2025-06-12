import os
import pandas as pd
from train_saint import train_and_evaluate

ROOT = r"C:\Users\meiyoudg\Downloads\TabMini\plotting\data"
RESULTS = []

for group in os.listdir(ROOT):
    group_path = os.path.join(ROOT, group)
    if not os.path.isdir(group_path):
        continue

    for dataset in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset)
        print(f"Running: {dataset}")
        try:
            metrics = train_and_evaluate(dataset_path)
            metrics["group"] = group
            RESULTS.append(metrics)
        except Exception as e:
            print(f"Error with {dataset}: {e}")

# Lưu kết quả
if RESULTS:
    df = pd.DataFrame(RESULTS)
    os.makedirs("../../results", exist_ok=True)
    df.to_csv("../../results/saint_results.csv", index=False)
    print("Saved to saint_results.csv")
else:
    print("No results were collected.")
