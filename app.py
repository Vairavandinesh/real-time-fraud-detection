
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from xgboost import XGBClassifier
from pathlib import Path
from feast import FeatureStore
from pathlib import Path
app = FastAPI(title="Fraud Detection API")

# Load trained model
model = XGBClassifier()
model.load_model("fraud_model.json")


class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([{
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "newbalanceOrig": transaction.newbalanceOrig,
        "oldbalanceDest": transaction.oldbalanceDest,
        "newbalanceDest": transaction.newbalanceDest,
        "type_CASH_IN": int(transaction.type == "CASH_IN"),
        "type_CASH_OUT": int(transaction.type == "CASH_OUT"),
        "type_DEBIT": int(transaction.type == "DEBIT"),
        "type_PAYMENT": int(transaction.type == "PAYMENT"),
        "type_TRANSFER": int(transaction.type == "TRANSFER"),
    }])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "prediction": "Fraud" if prediction == 1 else "Normal",
        "fraud_probability": round(probability, 4)
    }

@app.get("/features/{transaction_id}")
def get_transaction_features(transaction_id: int):
    from feast import FeatureStore

    repo_path = Path(__file__).parent / "feast_repo" / "feature_repo"
    store = FeatureStore(repo_path=str(repo_path))

    result = store.get_online_features(
        features=[
            "transaction_features:amount",
            "transaction_features:oldbalanceOrg",
            "transaction_features:newbalanceOrig",
            "transaction_features:oldbalanceDest",
            "transaction_features:newbalanceDest",
            "transaction_features:type",
        ],
        entity_rows=[{"transaction_id": transaction_id}],
    ).to_dict()

    if result["amount"][0] is None:
        return {
            "transaction_id": transaction_id,
            "message": "No features found in Feast online store",
        }

    return {
        "transaction_id": transaction_id,
        "features": {
            "amount": result["amount"][0],
            "oldbalanceOrg": result["oldbalanceOrg"][0],
            "newbalanceOrig": result["newbalanceOrig"][0],
            "oldbalanceDest": result["oldbalanceDest"][0],
            "newbalanceDest": result["newbalanceDest"][0],
            "type": result["type"][0],
        },
    }

@app.get("/predict-from-feast/{transaction_id}")
def predict_from_feast(transaction_id: int):
    

    repo_path = Path(__file__).parent / "feast_repo" / "feature_repo"
    store = FeatureStore(repo_path=str(repo_path))

    result = store.get_online_features(
        features=[
            "transaction_features:type",
            "transaction_features:amount",
            "transaction_features:oldbalanceOrg",
            "transaction_features:newbalanceOrig",
            "transaction_features:oldbalanceDest",
            "transaction_features:newbalanceDest",
        ],
        entity_rows=[{"transaction_id": transaction_id}],
    ).to_dict()

    if result["amount"][0] is None or result["type"][0] is None:
        return {
            "transaction_id": transaction_id,
            "message": "Transaction features not found in Feast",
        }

    transaction_type = result["type"][0]

    data = pd.DataFrame([{
        "amount": result["amount"][0],
        "oldbalanceOrg": result["oldbalanceOrg"][0],
        "newbalanceOrig": result["newbalanceOrig"][0],
        "oldbalanceDest": result["oldbalanceDest"][0],
        "newbalanceDest": result["newbalanceDest"][0],
        "type_CASH_IN": int(transaction_type == "CASH_IN"),
        "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
        "type_DEBIT": int(transaction_type == "DEBIT"),
        "type_PAYMENT": int(transaction_type == "PAYMENT"),
        "type_TRANSFER": int(transaction_type == "TRANSFER"),
    }])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "transaction_id": transaction_id,
        "type": transaction_type,
        "prediction": "Fraud" if prediction == 1 else "Normal",
        "fraud_probability": round(probability, 4),
    }

