import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Đường dẫn file kết quả tuning
df = pd.read_csv("C:/Users/meiyoudg/Downloads/tabmini_project/results/mlpplr_tuning_results.csv")
sns.set(style="whitegrid")

# Các metric cần vẽ
metrics = ["accuracy", "precision", "recall", "f1", "auc"]
plot_dir = "C:/Users/meiyoudg/Downloads/tabmini_project/plots"
os.makedirs(plot_dir, exist_ok=True)

# Vẽ boxplot cho từng metric
for metric in metrics:
    plt.figure(figsize=(8, 4))
    sns.boxplot(y=metric, data=df)
    plt.title(f"Phân bố {metric} của MLP-PLR trên TabMini")
    plt.savefig(f"{plot_dir}/mlpplr_{metric}.png")
    plt.close()

print(" Đã lưu các biểu đồ đánh giá vào thư mục plots/")
