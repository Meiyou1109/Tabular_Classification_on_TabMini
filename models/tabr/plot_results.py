import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("C:/Users/meiyoudg/Downloads/tabmini_project/results/tabr_results.csv")
sns.set(style="whitegrid")

# Metrics to plot
metrics = ["accuracy", "precision", "recall", "f1", "auc"]

# Create output directory
plot_dir = "C:/Users/meiyoudg/Downloads/tabmini_project/plots"
os.makedirs(plot_dir, exist_ok=True)

# Plot each metric
for metric in metrics:
    plt.figure(figsize=(8, 4))
    sns.boxplot(y=metric, data=df)
    plt.title(f"Phân bố {metric} của TabR trên TabMini")
    plt.savefig(f"{plot_dir}/tabr_{metric}.png")
    plt.close()
