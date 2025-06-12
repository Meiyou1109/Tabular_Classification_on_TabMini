import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:\Users\meiyoudg\Downloads\tabmini_project\results\tabnet_results.csv")

sns.set(style="whitegrid")

# Vẽ boxplot cho các metric
metrics = ["accuracy", "precision", "recall", "f1", "auc"]

for metric in metrics:
    plt.figure(figsize=(8, 4))
    sns.boxplot(y=metric, data=df)
    plt.title(f"Phân bố {metric} của TabNet trên TabMini")
    plt.savefig(f"C:\Users\meiyoudg\Downloads\tabmini_project\plots\tabnet_{metric}.png")
    plt.show()
