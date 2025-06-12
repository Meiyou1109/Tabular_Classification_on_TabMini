import os
import warnings
import pandas as pd
from train_mlpplr import train_and_eval

warnings.filterwarnings("ignore")

ROOT_DATA_DIR = "C:/Users/meiyoudg/Downloads/TabMini/plotting/data"
SAVE_PATH = "C:/Users/meiyoudg/Downloads/tabmini_project/results/mlpplr_results.csv"

# Ghi header mới
with open(SAVE_PATH, "w") as f:
    f.write("dataset,accuracy,precision,recall,f1,auc\n")

# Duyệt toàn bộ tập dữ liệu
for group in os.listdir(ROOT_DATA_DIR):
    group_path = os.path.join(ROOT_DATA_DIR, group)
    if not os.path.isdir(group_path):
        continue

    for dataset in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        print(f"=== Running MLP-PLR on: {dataset} ===")
        try:
            metrics = train_and_eval(dataset_path)
            with open(SAVE_PATH, "a") as f:
                f.write(f"{metrics['dataset']},{metrics['accuracy']:.6f},{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f},{metrics['auc']:.6f}\n")
        except Exception as e:
            print(f" Failed on {dataset}: {e}")

print("\n Done running all datasets with MLP-PLR!")
