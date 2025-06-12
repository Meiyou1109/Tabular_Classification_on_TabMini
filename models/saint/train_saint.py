import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from saint_model import SAINTModel

def train_and_evaluate(dataset_path):
    X = pd.read_csv(os.path.join(dataset_path, "X.csv"))
    y = pd.read_csv(os.path.join(dataset_path, "y.csv"))
    y.columns = ["target"]

    # Chuyển dữ liệu sang tensor
    X.columns = [str(col) for col in X.columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y["target"], test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAINTModel(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train
    model.train()
    for _ in range(10):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Đánh giá
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
        "dataset": os.path.basename(dataset_path),
        "accuracy": accuracy_score(y_test, all_preds),
        "precision": precision_score(y_test, all_preds, zero_division=0),
        "recall": recall_score(y_test, all_preds, zero_division=0),
        "f1": f1_score(y_test, all_preds, zero_division=0),
        "auc": roc_auc_score(y_test, all_probs),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()

    metrics = train_and_evaluate(args.dataset_path)
    df = pd.DataFrame([metrics])
    os.makedirs("../../results", exist_ok=True)
    df.to_csv("../../results/saint_results.csv", index=False)
    print(metrics)
