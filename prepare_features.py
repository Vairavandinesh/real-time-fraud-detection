
import pandas as pd

df = pd.read_csv("dataset.csv")

df["event_timestamp"] = (
    pd.Timestamp("2026-01-01", tz="UTC")
    + pd.to_timedelta(df["step"], unit="h")
)

# Keep transaction type so Feast can supply the model's categorical features.
df = df[
    [
        "nameOrig",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "event_timestamp",
    ]
].copy()

df.to_parquet("transaction_features.parquet", index=False)

print("Feature data created:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
