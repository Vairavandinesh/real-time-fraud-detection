from fastapi.testclient import TestClient
from app import app
import unittest


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
