import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = "dataset.csv"
MODEL_FILE = "fraud_gnn.pt"
OUTPUT_FILE = "gnn_account_predictions.csv"
METRICS_FILE = "gnn_metrics.json"
SAMPLE_SIZE = 50000
EPOCHS = 40

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

print("Loading transaction sample...")
df = pd.read_csv(
    DATA_FILE,
    usecols=["nameOrig", "nameDest", "amount", "isFraud"],
    nrows=SAMPLE_SIZE
).dropna()

df["isFraud"] = df["isFraud"].astype(int)

# Account labels: 1 if the account originated a fraud transaction.
fraud_accounts = set(
    df.loc[df["isFraud"] == 1, "nameOrig"].unique()
)

# Use only normal transactions to construct graph and node features.
normal = df[df["isFraud"] == 0].copy()

accounts = sorted(
    set(normal["nameOrig"]) |
    set(normal["nameDest"]) |
    fraud_accounts
)

account_to_id = {account: i for i, account in enumerate(accounts)}
num_nodes = len(accounts)

print("Accounts:", num_nodes)
print("Fraud-labelled accounts:", len(fraud_accounts))

# Build account-level features from normal transactions only.
out_stats = normal.groupby("nameOrig")["amount"].agg(
    out_count="count",
    out_sum="sum",
    out_mean="mean"
)

in_stats = normal.groupby("nameDest")["amount"].agg(
    in_count="count",
    in_sum="sum",
    in_mean="mean"
)

features = pd.DataFrame(index=accounts)
features = features.join(out_stats).join(in_stats).fillna(0)

feature_columns = [
    "out_count", "in_count",
    "out_sum", "in_sum",
    "out_mean", "in_mean"
]

X = np.log1p(features[feature_columns].to_numpy(dtype=np.float32))

y = np.array(
    [1 if account in fraud_accounts else 0 for account in accounts],
    dtype=np.float32
)

# Build bidirectional edges from normal transactions only.
source = normal["nameOrig"].map(account_to_id).to_numpy()
destination = normal["nameDest"].map(account_to_id).to_numpy()

edge_index = torch.tensor(
    np.vstack([
        np.concatenate([source, destination]),
        np.concatenate([destination, source])
    ]),
    dtype=torch.long
)

# Stratified account-level train/test split.
indices = np.arange(num_nodes)

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X[train_idx] = scaler.fit_transform(X[train_idx])
X[test_idx] = scaler.transform(X[test_idx])

x = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
edge_index = edge_index.to(device)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

train_mask = train_mask.to(device)
test_mask = test_mask.to(device)

class FraudGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.classifier(x).squeeze(-1)

model = FraudGraphSAGE(len(feature_columns)).to(device)

positive_count = float(y[train_idx].sum())
negative_count = float(len(train_idx) - positive_count)

if positive_count == 0:
    raise ValueError("No positive training accounts found.")

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

print("\nTraining GraphSAGE on CPU...")
print("Training accounts:", len(train_idx))
print("Testing accounts:", len(test_idx))
print("Positive training accounts:", int(positive_count))

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(x, edge_index)
    loss = criterion(logits[train_mask], y_tensor[train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    probabilities = torch.sigmoid(model(x, edge_index)).cpu().numpy()

test_y = y[test_idx].astype(int)
test_prob = probabilities[test_idx]
test_pred = (test_prob >= 0.5).astype(int)

pr_auc = average_precision_score(test_y, test_prob)

print("\nGNN Evaluation")
print("Test PR-AUC:", round(float(pr_auc), 6))
print("\nConfusion matrix:")
print(confusion_matrix(test_y, test_pred))
print("\nClassification report:")
print(classification_report(test_y, test_pred, zero_division=0))

torch.save({
    "model_state_dict": model.state_dict(),
    "input_features": feature_columns,
    "hidden_dim": 32,
    "account_ids": accounts,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist()
}, MODEL_FILE)

pd.DataFrame({
    "account_id": accounts,
    "actual_fraud_account": y.astype(int),
    "gnn_risk_score": probabilities
}).sort_values(
    "gnn_risk_score", ascending=False
).to_csv(OUTPUT_FILE, index=False)

metrics = {
    "model": "GraphSAGE",
    "sample_transactions": int(len(df)),
    "graph_accounts": int(num_nodes),
    "fraud_labelled_accounts": int(y.sum()),
    "test_accounts": int(len(test_idx)),
    "test_fraud_accounts": int(test_y.sum()),
    "test_pr_auc": float(pr_auc),
    "threshold": 0.5,
    "device": "CPU",
    "graph_edges_source": "normal transactions only",
    "label_definition": "account originated at least one fraud transaction"
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nSaved:", MODEL_FILE)
print("Saved:", OUTPUT_FILE)
print("Saved:", METRICS_FILE)
print("\nGraphSAGE training completed.")
