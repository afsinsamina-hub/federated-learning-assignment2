"""Shared utilities: stratified split, Dirichlet non-IID partition, metrics (incl. AUC)."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

def split_data(X, y, seed=0, val=0.15, test=0.15):
    """Stratified train/val/test split; the test set is never used for training or selection."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=val/(1-test), stratify=y_tr, random_state=seed)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)

def dirichlet_partition(y, n_clients=20, alpha=0.5, seed=0, min_size=2):
    """Label-skew non-IID split. Returns a list of index arrays, one per client."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    while True:
        idx = [[] for _ in range(n_clients)]
        for c in classes:
            ids = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                idx[k] += part.tolist()
        if min(len(i) for i in idx) >= min_size:
            return [np.array(i) for i in idx]

def evaluate(y_true, y_prob, thr=0.5):
    """Accuracy, Precision, Recall, F1, AUC and confusion matrix for binary classification."""
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_prob),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}


# ---------------------------------------------------------------- datasets
import os, pandas as pd
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_raw(name):
    if name == "heart":            # UCI Cleveland (303 patients), copy of kb22/Heart-Disease-Prediction; label flipped -> 1 = disease
        df = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"), encoding="utf-8-sig")
        return df.drop(columns="target").values.astype(float), (1 - df["target"].values).astype(int)   # file coding is reversed (see README): flipped so 1 = heart disease present
    if name == "breast_cancer":    # Wisconsin Diagnostic (569) via scikit-learn; label 1 = benign -> flip so 1 = malignant
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True); return X.astype(float), (1 - y).astype(int)
    if name == "diabetes":         # Pima Indians Diabetes (768); impossible zeros -> NaN -> train median
        df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
        X = df.drop(columns="Outcome").copy()
        for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]: X[c] = X[c].replace(0, np.nan)
        return X.values.astype(float), df["Outcome"].values.astype(int)
    raise ValueError(name)

def prepare(name, seed, n_clients=10, alpha=0.5):
    X, y = load_raw(name)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_data(X, y, seed=seed)
    med = np.nanmedian(Xtr, axis=0)                       # fit on train only
    for A in (Xtr, Xva, Xte):
        r, c = np.where(np.isnan(A)); A[r, c] = med[c]
    lo, hi = Xtr.min(0), Xtr.max(0); rng_ = np.where(hi - lo == 0, 1, hi - lo)
    sc = lambda A: np.clip((A - lo) / rng_, 0, 1)         # Min-Max, train statistics only
    parts = dirichlet_partition(ytr, n_clients, alpha, seed=seed, min_size=5)
    return dict(name=name, Xtr=sc(Xtr), ytr=ytr, Xva=sc(Xva), yva=yva, Xte=sc(Xte), yte=yte, parts=parts)
