import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from saint_model import SAINTModel

# === Hyperparameter configurations ===
configs = [
    {"hidden_dim": 16, "num_layers": 1, "dropout": 0.1, "lr": 0.01},
    {"hidden_dim": 32, "num_layers": 2, "dropout": 0.2, "lr": 0.005},
    {"hidden_dim": 64, "num_layers": 3, "dropout": 0.3, "lr": 0.001},
]

# === Dataset root path ===
ROOT = r"C:\Users\meiyoudg\Downloads\TabMini\plotting\data"
results = []

def load_data(dataset_path):
    X = pd.read_csv(os.path.join(dataset_path, "X.csv"))
    y = pd.read_csv(os.path.join(dataset_path, "y.csv"))
    y.columns = ["target"]
    X.columns = [str(col) for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y["target"], test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    return X_train, X_test, y_train, y_test


def run_training(config, X_train, X_test, y_train, y_test):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAINTModel(
        input_dim=X_train.shape[1],
        hidden_dim=config["hidden_dim"],
        num_heads=2,
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    model.train()
    for _ in range(10):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
        "lr": config["lr"],
        "accuracy": accuracy_score(y_test, all_preds),
        "precision": precision_score(y_test, all_preds, zero_division=0),
        "recall": recall_score(y_test, all_preds, zero_division=0),
        "f1": f1_score(y_test, all_preds, zero_division=0),
        "auc": roc_auc_score(y_test, all_probs),
    }

# === Main loop ===
for group in os.listdir(ROOT):
    group_path = os.path.join(ROOT, group)
    if not os.path.isdir(group_path):
        continue
    for dataset in os.listdir(group_path):
        dataset_path = os.path.join(group_path, dataset)
        try:
            print(f"Tuning {dataset}...")
            X_train, X_test, y_train, y_test = load_data(dataset_path)
            for cfg in configs:
                result = run_training(cfg, X_train, X_test, y_train, y_test)
                result["dataset"] = dataset
                result["group"] = group
                results.append(result)
        except Exception as e:
            print(f"Error with {dataset}: {e}")

# === Save results ===
df = pd.DataFrame(results)
os.makedirs("../../results", exist_ok=True)
df.to_csv("../../results/saint_tuning_results.csv", index=False)
print("Done tuning. Saved to saint_tuning_results.csv")
