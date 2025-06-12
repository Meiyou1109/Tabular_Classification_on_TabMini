import os
import warnings
from train_tabr import train_and_evaluate

warnings.filterwarnings("ignore")

ROOT_DATA_DIR = "C:/Users/meiyoudg/Downloads/TabMini/plotting/data"
SAVE_PATH = "C:/Users/meiyoudg/Downloads/tabmini_project/results/tabr_results.csv"

# Ghi header
with open(SAVE_PATH, "w") as f:
    f.write("dataset,accuracy,precision,recall,f1,auc\n")

# Duyệt tất cả dataset
for group in os.listdir(ROOT_DATA_DIR):
    group_path = os.path.join(ROOT_DATA_DIR, group)
    if not os.path.isdir(group_path):
        continue

    for dataset in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        print(f"=== Running TabR on: {dataset} ===")
        try:
            metrics = train_and_evaluate(dataset_path)
            with open(SAVE_PATH, "a") as f:
                f.write(f"{metrics['dataset']},{metrics['accuracy']:.6f},{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f},{metrics['auc']:.6f}\n")
        except Exception as e:
            print(f" Failed on {dataset}: {e}")

print("\n Done running all datasets with TabR!")
