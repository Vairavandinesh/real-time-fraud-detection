import os
import json
import shutil
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

DATA_FILE = "dataset.csv"
MODEL_FILE = "fraud_model.json"
CANDIDATE_FILE = "fraud_model_candidate.json"
BACKUP_FILE = "fraud_model_backup.json"
REPORT_FILE = "retraining_report.json"

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

NUMERIC = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_counts = np.histogram(current, bins=edges)[0] / len(current)

    ref_counts = np.clip(ref_counts, 0.0001, None)
    cur_counts = np.clip(cur_counts, 0.0001, None)

    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))

print("Loading feature dataset...")
data = pd.read_parquet("transaction_features.parquet")

data["event_timestamp"] = pd.to_datetime(data["event_timestamp"], utc=True)
data = data.sort_values("event_timestamp")

split_point = int(len(data) * 0.8)
reference = data.iloc[:split_point]
current = data.iloc[split_point:]

drift_scores = {
    col: psi(reference[col].dropna().to_numpy(),
             current[col].dropna().to_numpy())
    for col in NUMERIC
}

max_psi = max(drift_scores.values())
drift_detected = max_psi >= 0.20

print("\nDrift scores:")
for feature, score in drift_scores.items():
    print(f"{feature}: {score:.4f}")

print("Drift detected:", drift_detected)

report = {
    "drift_scores": drift_scores,
    "max_psi": max_psi,
    "drift_detected": drift_detected,
    "retraining_attempted": False,
    "model_replaced": False
}

if not drift_detected:
    print("\nNo significant drift. Existing model retained.")
else:
    print("\nDrift detected. Preparing candidate model...")

    df = pd.read_csv(DATA_FILE)

    X = df[
        ["type", "amount", "oldbalanceOrg",
         "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    ].copy()

    y = df["isFraud"]

    X = pd.get_dummies(X, columns=["type"], dtype=int)
    X = X.reindex(columns=FEATURES, fill_value=0)

    fraud = y == 1
    normal = df.loc[~fraud].sample(n=100000, random_state=42)

    selected = list(normal.index) + list(df.index[fraud])
    X = X.loc[selected]
    y = y.loc[selected]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train, y_train
    )

    candidate = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    candidate.fit(X_train_balanced, y_train_balanced)

    candidate_score = average_precision_score(
        y_test, candidate.predict_proba(X_test)[:, 1]
    )

    existing = XGBClassifier()
    existing.load_model(MODEL_FILE)

    existing_score = average_precision_score(
        y_test, existing.predict_proba(X_test)[:, 1]
    )

    report["retraining_attempted"] = True
    report["existing_pr_auc"] = float(existing_score)
    report["candidate_pr_auc"] = float(candidate_score)

    print(f"\nExisting model PR-AUC: {existing_score:.6f}")
    print(f"Candidate model PR-AUC: {candidate_score:.6f}")

    if candidate_score > existing_score:
        candidate.save_model(CANDIDATE_FILE)
        shutil.copy2(MODEL_FILE, BACKUP_FILE)
        os.replace(CANDIDATE_FILE, MODEL_FILE)
        report["model_replaced"] = True
        print("\nCandidate performed better. Model updated.")
        print("Previous model backup:", BACKUP_FILE)
    else:
        if os.path.exists(CANDIDATE_FILE):
            os.remove(CANDIDATE_FILE)
        print("\nCandidate did not outperform existing model. Keeping existing model.")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\nRetraining workflow completed.")
print("Report saved to:", REPORT_FILE)
