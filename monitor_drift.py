
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Dataset containing historical transaction features
REFERENCE_FILE = "transaction_features_sample.parquet"

# 1. Load only the model's numerical input features
feature_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

reference = pd.read_parquet(REFERENCE_FILE)[feature_columns]

# 2. Create a sample of incoming transactions
current = reference.sample(
    n=min(5000, len(reference)),
    random_state=7
).copy()

# 3. Simulate data drift by increasing transaction amounts
current["amount"] = current["amount"] * 1.5

# 4. Generate the data drift report
report = Report(
    metrics=[DataDriftPreset()]
)

result = report.run(
    reference_data=reference.head(5000),
    current_data=current
)

# 5. Save the report as HTML
REPORT_FILE = "drift_report.html"
result.save_html(REPORT_FILE)

print("Drift monitoring completed successfully!")
print(f"Reference transactions: {min(5000, len(reference))}")
print(f"Current transactions: {len(current)}")
print("Simulated change: amount multiplied by 1.5")
print(f"Report saved to: {REPORT_FILE}")
