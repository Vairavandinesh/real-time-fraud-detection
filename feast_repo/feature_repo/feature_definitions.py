from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String

transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
)

data_path = str(
    Path(__file__).parent / "data" / "fraud_features.parquet"
)

transaction_source = FileSource(
    name="fraud_transactions_source",
    path=data_path,
    timestamp_field="event_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="type", dtype=String),
        Field(name="amount", dtype=Float32),
        Field(name="oldbalanceOrg", dtype=Float32),
        Field(name="newbalanceOrig", dtype=Float32),
        Field(name="oldbalanceDest", dtype=Float32),
        Field(name="newbalanceDest", dtype=Float32),
    ],
    source=transaction_source,
    online=True,
)
