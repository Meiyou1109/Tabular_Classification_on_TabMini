import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load kết quả
df = pd.read_csv("C:/Users/meiyoudg/Downloads/tabmini_project/results/saint_results.csv")
sns.set(style="whitegrid")


# Vẽ boxplot cho từng metric
metrics = ["accuracy", "precision", "recall", "f1", "auc"]
for metric in metrics:
    plt.figure(figsize=(8, 4))
    sns.boxplot(y=metric, data=df)
    plt.title(f"Phân bố {metric} của SAINT trên TabMini")
    plt.savefig(f"C:/Users/meiyoudg/Downloads/tabmini_project/plots/saint_{metric}.png")
    plt.close()
