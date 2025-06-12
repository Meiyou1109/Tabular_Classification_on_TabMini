import os
import subprocess

ROOT_DATA_DIR = "C:/Users/meiyoudg/Downloads/TabMini/plotting/data"
SAVE_PATH = "C:/Users/meiyoudg/Downloads/tabmini_project/results/tabtransformer_results.csv"
PYTHON_EXE = "C:/Users/meiyoudg/Downloads/tabmini_project/models/tabtransformer/tabmini_env/Scripts/python.exe"

# Duyệt qua tất cả thư mục chứa dataset 
group_map = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5
}

if os.path.exists(SAVE_PATH):
    os.remove(SAVE_PATH)

for group_folder in os.listdir(ROOT_DATA_DIR):
    group_path = os.path.join(ROOT_DATA_DIR, group_folder)
    if not os.path.isdir(group_path):
        continue

    group_num = group_map.get(group_folder, 1)

    for dataset_name in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"\n=== Running on dataset: {dataset_name} (Group {group_num}) ===")

        # Gọi script train_tabtransformer.py qua subprocess
        subprocess.run([
            PYTHON_EXE, "train_tabtransformer.py",
            "--data_dir", dataset_path,
            "--save_path", SAVE_PATH,
            "--group", str(group_num)
        ])

print("\n Hoàn tất chạy toàn bộ dataset!")
