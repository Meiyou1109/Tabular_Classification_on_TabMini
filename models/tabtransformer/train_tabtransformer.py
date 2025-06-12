import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from tab_transformer_pytorch import TabTransformer
from torch import nn

def load_data(data_dir):
    X = pd.read_csv(os.path.join(data_dir, "X.csv"))
    y = pd.read_csv(os.path.join(data_dir, "y.csv"))

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test, cat_cols, num_cols

def build_model(cat_dims, num_cont, dim, depth, heads, attn_dropout):
    model = TabTransformer(
        categories=tuple(cat_dims),
        num_continuous=num_cont,
        dim=dim,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=nn.ReLU()
    )
    return model

def train_model(model, dataloader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)

    for epoch in range(10):
        model.train()
        for x_cat, x_cont, y in dataloader:
            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)
            logits = model(x_cat, x_cont).squeeze()
            loss = loss_fn(logits, y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for x_cat, x_cont, y in val_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            logits = model(x_cat, x_cont).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, prec, rec, f1, auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--group', type=int, default=1)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test, cat_cols, num_cols = load_data(args.data_dir)
    cat_dims = [X_train[col].nunique() for col in cat_cols]

    x_cat_train = torch.tensor(X_train[cat_cols].values, dtype=torch.long)
    x_cont_train = torch.tensor(X_train[num_cols].values, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)

    x_cat_test = torch.tensor(X_test[cat_cols].values, dtype=torch.long)
    x_cont_test = torch.tensor(X_test[num_cols].values, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    train_dataset = TensorDataset(x_cat_train, x_cont_train, y_train)
    test_dataset = TensorDataset(x_cat_test, x_cont_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = build_model(cat_dims, len(num_cols), args.dim, args.depth, args.heads, args.attn_dropout)
    model = train_model(model, train_loader, test_loader, device)
    acc, prec, rec, f1, auc = evaluate_model(model, test_loader, device)

    dataset_name = os.path.basename(args.data_dir)
    print(f"{dataset_name}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, auc={auc:.4f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    header = not os.path.exists(args.save_path)
    with open(args.save_path, "a") as f:
        if header:
            f.write("dataset,accuracy,precision,recall,f1,auc,group,dim,depth,heads,attn_dropout\n")
        f.write(f"{dataset_name},{acc:.6f},{prec:.6f},{rec:.6f},{f1:.6f},{auc:.6f},{args.group},{args.dim},{args.depth},{args.heads},{args.attn_dropout}\n")

if __name__ == '__main__':
    main()
