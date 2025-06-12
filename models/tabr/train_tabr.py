import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tabr_model import TabR
import warnings

warnings.filterwarnings("ignore")

def load_dataset(dataset_path):
    X = pd.read_csv(os.path.join(dataset_path, "X.csv"))
    y = pd.read_csv(os.path.join(dataset_path, "y.csv"))
    y.columns = ["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y["target"], test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    return X_train, X_test, y_train, y_test

def train_and_evaluate(dataset_path, hidden_dim=64, rnn_type="gru", use_attention=True):
    X_train, X_test, y_train, y_test = load_dataset(dataset_path)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabR(input_dim=X_train.shape[1], hidden_dim=hidden_dim, rnn_type=rnn_type, use_attention=use_attention).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(10):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)

    return {
        "dataset": os.path.basename(dataset_path),
        "accuracy": accuracy_score(y_test, all_preds),
        "precision": precision_score(y_test, all_preds, zero_division=0),
        "recall": recall_score(y_test, all_preds, zero_division=0),
        "f1": f1_score(y_test, all_preds, zero_division=0),
        "auc": roc_auc_score(y_test, all_probs),
    }

if __name__ == "__main__":
    dataset_path = "C:/Users/meiyoudg/Downloads/TabMini/plotting/data/1/analcatdata_aids"
    result = train_and_evaluate(dataset_path)
    print(result)
