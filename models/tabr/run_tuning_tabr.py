import os
import warnings
from train_tabr import train_and_evaluate

warnings.filterwarnings("ignore")

# Các cấu hình
configs = [
    {"hidden_dim": 32, "rnn_type": "gru", "use_attention": True},
    {"hidden_dim": 64, "rnn_type": "gru", "use_attention": False},
    {"hidden_dim": 128, "rnn_type": "lstm", "use_attention": True},
]

ROOT_DATA_DIR = "C:/Users/meiyoudg/Downloads/TabMini/plotting/data"
SAVE_PATH = "C:/Users/meiyoudg/Downloads/tabmini_project/results/tabr_tuning_results.csv"

# Ghi header
with open(SAVE_PATH, "w") as f:
    f.write("dataset,group,accuracy,precision,recall,f1,auc,hidden_dim,rnn_type,use_attention\n")

# Duyệt toàn bộ dataset
for group in os.listdir(ROOT_DATA_DIR):
    group_path = os.path.join(ROOT_DATA_DIR, group)
    if not os.path.isdir(group_path):
        continue

    for dataset in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for cfg in configs:
            print(f"=== Tuning TabR on: {dataset} | config={cfg} ===")
            try:
                metrics = train_and_evaluate(dataset_path, **cfg)
                with open(SAVE_PATH, "a") as f:
                    f.write(f"{metrics['dataset']},{group},{metrics['accuracy']:.6f},{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f},{metrics['auc']:.6f},{cfg['hidden_dim']},{cfg['rnn_type']},{cfg['use_attention']}\n")
            except Exception as e:
                print(f" Failed on {dataset} with config {cfg}: {e}")

print("\n Done tuning TabR on all datasets!")
