from feast import FeatureStore

store = FeatureStore(repo_path=".")
result = store.get_online_features(
    features=[
        "transaction_features:amount",
        "transaction_features:oldbalanceOrg",
        "transaction_features:newbalanceOrig",
        "transaction_features:oldbalanceDest",
        "transaction_features:newbalanceDest",
    ],
    entity_rows=[{"transaction_id": 1}],
).to_dict()

print(result)
