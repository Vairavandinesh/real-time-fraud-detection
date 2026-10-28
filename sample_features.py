import pandas as pd

df = pd.read_parquet("transaction_features.parquet")

sample = df.sample(n=min(50000, len(df)), random_state=42)
sample.to_parquet("transaction_features_sample.parquet", index=False)

print("Sample saved:", sample.shape)