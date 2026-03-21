"""Integration tests for the Flask API validation endpoints.

Run with:  python -m unittest tests.test_api -v
Slow tests (live yfinance) are marked; skip them with:
  python -m unittest tests.test_api.TestAPIFast -v
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the Flask app (IRIS_System init may fail in CI — that's fine)
from app import app as flask_app


class TestAPIFast(unittest.TestCase):
    """Tests that don't require live network access."""

    def setUp(self):
        flask_app.config["TESTING"] = True
        self.client = flask_app.test_client()

    def test_health_endpoint(self):
        """GET /api/health should return 200 with ticker_db_loaded=true."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data.get("ticker_db_loaded"), "Ticker DB should be loaded")
        self.assertGreater(data.get("ticker_count", 0), 0)

    def test_validate_missing_body(self):
        """POST /api/validate-ticker with no body should return valid=false."""
        resp = self.client.post("/api/validate-ticker",
                                content_type="application/json",
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertFalse(data.get("valid"))

    def test_validate_invalid_format(self):
        """POST /api/validate-ticker with bad format should return valid=false immediately."""
        resp = self.client.post("/api/validate-ticker",
                                content_type="application/json",
                                data=json.dumps({"ticker": "123ABC"}))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertFalse(data.get("valid"))
        self.assertIn("error", data)

    def test_analyze_rejects_invalid_ticker(self):
        """GET /api/analyze with a clearly invalid ticker should return 422."""
        resp = self.client.get("/api/analyze?ticker=XYZQW")
        self.assertEqual(resp.status_code, 422)
        data = json.loads(resp.data)
        self.assertFalse(data.get("valid"))
        self.assertIn("error", data)


class TestAPISlow(unittest.TestCase):
    """Tests that hit live yfinance — skip in CI with -k 'not slow'."""

    def setUp(self):
        flask_app.config["TESTING"] = True
        self.client = flask_app.test_client()

    def test_validate_valid_ticker(self):
        """POST /api/validate-ticker with AAPL should return valid=true."""
        resp = self.client.post("/api/validate-ticker",
                                content_type="application/json",
                                data=json.dumps({"ticker": "AAPL"}))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("valid"))
        self.assertEqual(data.get("ticker"), "AAPL")
        self.assertIn("company_name", data)

    def test_validate_invalid_ticker(self):
        """POST /api/validate-ticker with XYZQW should return valid=false with error."""
        resp = self.client.post("/api/validate-ticker",
                                content_type="application/json",
                                data=json.dumps({"ticker": "XYZQW"}))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertFalse(data.get("valid"))
        self.assertIn("error", data)

    def test_analyze_accepts_valid_ticker(self):
        """GET /api/analyze with AAPL should NOT return 422 (validation must pass)."""
        resp = self.client.get("/api/analyze?ticker=AAPL")
        self.assertNotEqual(resp.status_code, 422,
                            "Validation gate should not reject AAPL")


if __name__ == "__main__":
    unittest.main()
