import os
import pandas as pd
from train_tabnet import train_and_evaluate

results = []
ROOT = r"C:\Users\meiyoudg\Downloads\TabMini\plotting\data"

for group in os.listdir(ROOT):
    group_path = os.path.join(ROOT, group)
    if not os.path.isdir(group_path): continue

    for dataset in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset)
        try:
            print(f"Running: {dataset}")
            metrics = train_and_evaluate(dataset_path)
            metrics["dataset"] = dataset
            metrics["group"] = group
            results.append(metrics)
        except Exception as e:
            print(f"Error with {dataset}: {e}")

# Lưu kết quả ra file
df = pd.DataFrame(results)
df.to_csv("tabnet_results.csv", index=False)
print("Saved to tabnet_results.csv")
