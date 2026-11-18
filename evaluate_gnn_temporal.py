import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, confusion_matrix, classification_report

DATA_FILE = "dataset.csv"
SAMPLE_SIZE = 50000
TRAIN_END_STEP = 6
EPOCHS = 40

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

print("Loading transaction sample...")
df = pd.read_csv(
    DATA_FILE,
    usecols=["step", "nameOrig", "nameDest", "amount", "isFraud"],
    nrows=SAMPLE_SIZE
).dropna()

df["isFraud"] = df["isFraud"].astype(int)

train_df = df[df["step"] <= TRAIN_END_STEP].copy()
test_df = df[df["step"] > TRAIN_END_STEP].copy()

# Training-period fraud labels.
train_fraud_accounts = set(
    train_df.loc[train_df["isFraud"] == 1, "nameOrig"]
)

# Test-period accounts and labels.
test_accounts = sorted(test_df["nameOrig"].unique())
test_fraud_accounts = set(
    test_df.loc[test_df["isFraud"] == 1, "nameOrig"]
)
test_y = np.array(
    [int(account in test_fraud_accounts) for account in test_accounts]
)

# Only historical normal transactions are used for graph/features.
normal_train = train_df[train_df["isFraud"] == 0].copy()

train_accounts = (
    set(normal_train["nameOrig"]) |
    set(normal_train["nameDest"]) |
    train_fraud_accounts
)

# Include test source accounts, but do not use test-period edges/features.
accounts = sorted(train_accounts | set(test_accounts))
account_to_id = {account: i for i, account in enumerate(accounts)}

train_account_ids = np.array(
    [account_to_id[a] for a in sorted(train_accounts)]
)
test_idx = np.array([account_to_id[a] for a in test_accounts])

print("Training-period transactions:", len(train_df))
print("Testing-period transactions:", len(test_df))
print("Training accounts:", len(train_accounts))
print("Testing source accounts:", len(test_accounts))
print("Training fraud accounts:", len(train_fraud_accounts))
print("Testing fraud accounts:", int(test_y.sum()))

# Historical features: training-period normal transactions only.
out_stats = normal_train.groupby("nameOrig")["amount"].agg(
    out_count="count", out_sum="sum", out_mean="mean"
)
in_stats = normal_train.groupby("nameDest")["amount"].agg(
    in_count="count", in_sum="sum", in_mean="mean"
)

features = pd.DataFrame(index=accounts)
features = features.join(out_stats).join(in_stats).fillna(0)

feature_columns = [
    "out_count", "in_count",
    "out_sum", "in_sum",
    "out_mean", "in_mean"
]

X = np.log1p(features[feature_columns].to_numpy(dtype=np.float32))

# Fit scaler only on training-period accounts.
scaler = StandardScaler()
X[train_account_ids] = scaler.fit_transform(X[train_account_ids])

x = torch.tensor(X, dtype=torch.float32, device=device)

# Training labels are based only on training-period fraud.
train_y = np.array(
    [int(account in train_fraud_accounts) for account in accounts],
    dtype=np.float32
)
y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)

# Graph edges from training-period normal transactions only.
source = normal_train["nameOrig"].map(account_to_id).to_numpy()
destination = normal_train["nameDest"].map(account_to_id).to_numpy()

edge_index = torch.tensor(
    np.vstack([
        np.concatenate([source, destination]),
        np.concatenate([destination, source])
    ]),
    dtype=torch.long,
    device=device
)

train_mask = torch.zeros(len(accounts), dtype=torch.bool, device=device)
train_mask[train_account_ids] = True

class FraudGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x).squeeze(-1)

model = FraudGraphSAGE(len(feature_columns)).to(device)

positive_count = float(train_y[train_account_ids].sum())
negative_count = float(len(train_account_ids) - positive_count)

if positive_count == 0 or negative_count == 0:
    raise ValueError("Training data must contain both fraud and normal accounts.")

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([negative_count / positive_count])
)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.005, weight_decay=1e-4
)

print("\nTraining temporal GraphSAGE on CPU...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(x, edge_index)
    loss = criterion(
        logits[train_mask],
        y_tensor[train_mask]
    )

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss.item():.4f}")

# Evaluate only on accounts originating transactions in later steps.
model.eval()
with torch.no_grad():
    probabilities = torch.sigmoid(model(x, edge_index)).cpu().numpy()

test_prob = probabilities[test_idx]
test_pred = (test_prob >= 0.5).astype(int)

if len(np.unique(test_y)) == 2:
    pr_auc = float(average_precision_score(test_y, test_prob))
else:
    pr_auc = None
    print("Warning: Test period has only one class; PR-AUC is unavailable.")

print("\nTemporal GNN Evaluation")
print("Test PR-AUC:", pr_auc)
print("\nConfusion matrix:")
print(confusion_matrix(test_y, test_pred, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    test_y, test_pred, labels=[0, 1],
    target_names=["normal", "fraud"], zero_division=0
))

pd.DataFrame({
    "account_id": test_accounts,
    "actual_fraud_in_test_period": test_y,
    "gnn_risk_score": test_prob
}).to_csv("gnn_temporal_predictions.csv", index=False)

metrics = {
    "model": "GraphSAGE temporal evaluation",
    "sample_transactions": int(len(df)),
    "training_steps": f"1-{TRAIN_END_STEP}",
    "testing_steps": f"{TRAIN_END_STEP + 1}-{int(df['step'].max())}",
    "training_accounts": int(len(train_account_ids)),
    "testing_source_accounts": int(len(test_accounts)),
    "training_fraud_accounts": int(positive_count),
    "testing_fraud_accounts": int(test_y.sum()),
    "test_pr_auc": pr_auc,
    "threshold": 0.5,
    "device": "CPU",
    "features_source": "training-period normal transactions only",
    "graph_edges_source": "training-period normal transactions only",
    "test_definition": "account originated at least one fraud transaction during testing period"
}

with open("gnn_temporal_metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

torch.save({
    "model_state_dict": model.state_dict(),
    "input_features": feature_columns,
    "hidden_dim": 32,
    "account_ids": accounts,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist()
}, "fraud_gnn_temporal.pt")

print("\nSaved: gnn_temporal_predictions.csv")
print("Saved: gnn_temporal_metrics.json")
print("Saved: fraud_gnn_temporal.pt")
print("Temporal evaluation completed.")
