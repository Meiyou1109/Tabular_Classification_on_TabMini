import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam

def train_and_evaluate(DATASET_PATH, config_name="config_1"):
    # Đọc dữ liệu
    X = pd.read_csv(f"{DATASET_PATH}/X.csv")
    y = pd.read_csv(f"{DATASET_PATH}/y.csv").values.ravel()

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

    configs = {
        "config_1": dict(n_d=8, n_a=8, lr=0.02, max_epochs=100),
        "config_2": dict(n_d=16, n_a=16, lr=0.01, max_epochs=200),
        "config_3": dict(n_d=24, n_a=24, lr=0.005, max_epochs=300)
    }

    params = configs.get(config_name, configs["config_1"])
    lr = params.pop("lr")
    max_epochs = params.pop("max_epochs")

    model = TabNetClassifier(
        **params,
        optimizer_fn=Adam,
        optimizer_params=dict(lr=lr),
        verbose=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=max_epochs,
        patience=10
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "dataset": DATASET_PATH.split("\\")[-1],
        "config": config_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob)
    }
