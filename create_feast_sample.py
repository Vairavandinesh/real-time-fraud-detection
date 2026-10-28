import pandas as pd

source = "transaction_features.parquet"
output = "feast_repo/feature_repo/data/fraud_features.parquet"

columns = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "event_timestamp",
]

df = pd.read_parquet(source, columns=columns).head(5000).copy()
df.insert(0, "transaction_id", range(1, len(df) + 1))
df["transaction_id"] = df["transaction_id"].astype("int64")
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

df.to_parquet(output, index=False)

print("Saved:", output)
print("Rows:", len(df))
print("Types:", df["type"].value_counts().to_dict())
print("Columns:", df.columns.tolist())
