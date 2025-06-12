import os
import pandas as pd
from train_tabnet import train_and_evaluate

ROOT = r"C:\Users\meiyoudg\Downloads\TabMini\plotting\data"
configs = ["config_1", "config_2", "config_3"]

results = []

for group in os.listdir(ROOT):
    group_path = os.path.join(ROOT, group)
    if not os.path.isdir(group_path): continue

    for dataset in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset)

        for cfg in configs:
            try:
                print(f"Dataset: {dataset} | Config: {cfg}")
                metrics = train_and_evaluate(dataset_path, config_name=cfg)
                metrics["group"] = group
                results.append(metrics)
            except Exception as e:
                print(f"[ERROR] {dataset} ({cfg}): {e}")

df = pd.DataFrame(results)
df.to_csv("C:\Users\meiyoudg\Downloads\tabmini_project\results\tabnet_tuning_results.csv", index=False)
print("Done tuning. Kết quả lưu vào tabnet_tuning_results.csv")
