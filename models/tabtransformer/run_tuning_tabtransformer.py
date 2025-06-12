import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tab_transformer_pytorch import TabTransformer

# === 3 cấu hình tiêu biểu ===
configs = [
    {"dim": 16, "depth": 2, "heads": 2, "attn_dropout": 0.1},
    {"dim": 32, "depth": 4, "heads": 4, "attn_dropout": 0.2},
    {"dim": 64, "depth": 3, "heads": 4, "attn_dropout": 0.1},
]

ROOT = r"C:/Users/meiyoudg/Downloads/TabMini/plotting/data"
SAVE_PATH = "C:/Users/meiyoudg/Downloads/tabmini_project/results/tabtransformer_tuning_results_fast.csv"

results = []

def load_data(dataset_path):
    X = pd.read_csv(os.path.join(dataset_path, "X.csv"))
    y = pd.read_csv(os.path.join(dataset_path, "y.csv"))
    y = y.values.ravel()

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, cat_cols, num_cols

def run_training(config, X_train, X_test, y_train, y_test, cat_cols, num_cols):
    x_cat_train = torch.tensor(X_train[cat_cols].values, dtype=torch.long)
    x_cont_train = torch.tensor(X_train[num_cols].values, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)

    x_cat_test = torch.tensor(X_test[cat_cols].values, dtype=torch.long)
    x_cont_test = torch.tensor(X_test[num_cols].values, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    train_loader = DataLoader(TensorDataset(x_cat_train, x_cont_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_cat_test, x_cont_test, y_test), batch_size=64)

    model = TabTransformer(
        categories=[X_train[col].nunique() for col in cat_cols],
        num_continuous=len(num_cols),
        dim=config["dim"],
        depth=config["depth"],
        heads=config["heads"],
        attn_dropout=config["attn_dropout"],
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=nn.ReLU()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(10):
        for x_cat, x_cont, y in train_loader:
            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)
            logits = model(x_cat, x_cont).squeeze()
            loss = loss_fn(logits, y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for x_cat, x_cont, y in test_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            logits = model(x_cat, x_cont).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)

    return {
        "dim": config["dim"],
        "depth": config["depth"],
        "heads": config["heads"],
        "attn_dropout": config["attn_dropout"],
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
            X_train, X_test, y_train, y_test, cat_cols, num_cols = load_data(dataset_path)
            for cfg in configs:
                result = run_training(cfg, X_train, X_test, y_train, y_test, cat_cols, num_cols)
                result["dataset"] = dataset
                result["group"] = group
                results.append(result)
        except Exception as e:
            print(f"Error with {dataset}: {e}")

# === Save
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
pd.DataFrame(results).to_csv(SAVE_PATH, index=False)
print(" Done. Saved to", SAVE_PATH)
