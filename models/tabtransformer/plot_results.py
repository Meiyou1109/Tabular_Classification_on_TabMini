import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Đường dẫn
RESULT_CSV = "C:/Users/meiyoudg/Downloads/tabmini_project/results/tabtransformer_results.csv"
PLOT_DIR = "C:/Users/meiyoudg/Downloads/tabmini_project/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load kết quả
df = pd.read_csv(RESULT_CSV)
sns.set(style="whitegrid")

# Vẽ boxplot cho từng metric
metrics = ["accuracy", "precision", "recall", "f1", "auc"]
for metric in metrics:
    plt.figure(figsize=(8, 4))
    sns.boxplot(y=metric, data=df)
    plt.title(f"Phân bố {metric} của TabTransformer trên TabMini")
    plt.savefig(os.path.join(PLOT_DIR, f"tabtransformer_{metric}.png"))
    plt.close()
