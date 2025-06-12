import os
import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

warnings.filterwarnings("ignore")

# === MLP-PLR Model ===
class MLPPLR(nn.Module):
    def __init__(self, num_features, cat_dims, emb_dim=8, hidden_dim=64, n_bins=10):
        super().__init__()
        self.n_bins = n_bins
        self.emb_dim = emb_dim

        self.num_embeds = nn.ModuleList([
            nn.Embedding(n_bins, emb_dim) for _ in range(num_features)
        ])

        self.cat_embeds = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) for cardinality in cat_dims
        ])

        input_dim = emb_dim * (num_features + len(cat_dims))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_num_idx, x_cat_idx):
        num_vecs = [emb(x_idx) for emb, x_idx in zip(self.num_embeds, x_num_idx.T)]
        cat_vecs = [emb(x_idx) for emb, x_idx in zip(self.cat_embeds, x_cat_idx.T)]
        x = torch.cat(num_vecs + cat_vecs, dim=1)
        return self.mlp(x).squeeze(1)


def load_dataset(data_dir, n_bins=10):
    X = pd.read_csv(os.path.join(data_dir, "X.csv"))
    y = pd.read_csv(os.path.join(data_dir, "y.csv"))
    y = y.values.ravel()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X[num_cols] = discretizer.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_num_train = torch.tensor(X_train[num_cols].values, dtype=torch.long)
    x_cat_train = torch.tensor(X_train[cat_cols].values, dtype=torch.long) if cat_cols else torch.empty((len(X_train), 0), dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    x_num_test = torch.tensor(X_test[num_cols].values, dtype=torch.long)
    x_cat_test = torch.tensor(X_test[cat_cols].values, dtype=torch.long) if cat_cols else torch.empty((len(X_test), 0), dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    cat_dims = [int(X[col].nunique()) for col in cat_cols]

    return (x_num_train, x_cat_train, y_train), (x_num_test, x_cat_test, y_test), len(num_cols), cat_dims


def train_and_eval(data_dir, n_bins=10, hidden_dim=64, emb_dim=8):
    (x_num_train, x_cat_train, y_train), (x_num_test, x_cat_test, y_test), n_num, cat_dims = load_dataset(data_dir, n_bins)

    train_loader = DataLoader(TensorDataset(x_num_train, x_cat_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_num_test, x_cat_test, y_test), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPLR(n_num, cat_dims, emb_dim=emb_dim, hidden_dim=hidden_dim, n_bins=n_bins).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(10):
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            logits = model(x_num, x_cat)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x_num, x_cat, y in test_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            logits = model(x_num, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    metrics = {
        "dataset": os.path.basename(data_dir),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs)
    }

    return metrics

if __name__ == '__main__':
    DATA_DIR = "C:/Users/meiyoudg/Downloads/TabMini/plotting/data/1/analcatdata_aids"
    result = train_and_eval(DATA_DIR)
    print(result)
