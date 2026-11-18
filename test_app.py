from fastapi.testclient import TestClient
from app import app
import unittest
import pandas as pd
from pathlib import Path


class TestFraudAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_openapi_endpoint(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/predict", response.json()["paths"])
        self.assertIn("/gnn-risk/{account_id}", response.json()["paths"])

    def test_gnn_risk_endpoint(self):
        file_path = Path(__file__).parent / "gnn_account_predictions.csv"
        predictions = pd.read_csv(file_path, dtype={"account_id": str})
        account_id = predictions.iloc[0]["account_id"]

        response = self.client.get(f"/gnn-risk/{account_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "GraphSAGE")
        self.assertIn("gnn_risk_score", response.json())
        self.assertNotIn("actual_fraud_account", response.json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
