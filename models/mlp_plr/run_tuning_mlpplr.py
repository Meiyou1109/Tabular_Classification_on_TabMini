import os
import warnings
import pandas as pd
from train_mlpplr import train_and_eval

warnings.filterwarnings("ignore")

# === Hyperparameter configurations ===
configs = [
    {"hidden_dim": 32, "emb_dim": 4},
    {"hidden_dim": 64, "emb_dim": 8},
    {"hidden_dim": 128, "emb_dim": 16},
]

ROOT_DATA_DIR = "C:/Users/meiyoudg/Downloads/TabMini/plotting/data"
SAVE_PATH = "C:/Users/meiyoudg/Downloads/tabmini_project/results/mlpplr_tuning_results.csv"

# Ghi header mới
with open(SAVE_PATH, "w") as f:
    f.write("dataset,group,accuracy,precision,recall,f1,auc,hidden_dim,emb_dim\n")

# Duyệt toàn bộ tập dữ liệu
for group in os.listdir(ROOT_DATA_DIR):
    group_path = os.path.join(ROOT_DATA_DIR, group)
    if not os.path.isdir(group_path):
        continue

    for dataset in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for cfg in configs:
            print(f"=== Tuning MLP-PLR on: {dataset} | config={cfg} ===")
            try:
                metrics = train_and_eval(dataset_path, hidden_dim=cfg["hidden_dim"], emb_dim=cfg["emb_dim"])
                with open(SAVE_PATH, "a") as f:
                    f.write(f"{metrics['dataset']},{group},{metrics['accuracy']:.6f},{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f},{metrics['auc']:.6f},{cfg['hidden_dim']},{cfg['emb_dim']}\n")
            except Exception as e:
                print(f" Failed on {dataset} with config {cfg}: {e}")

print("\n Done tuning all datasets with MLP-PLR!")
