"""Tests for ticker DB refresh, startup checks, and health endpoint DB-age fields."""

import json
import os
import sys
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ticker_db


def _make_sec_response(tickers: dict[str, str]) -> MagicMock:
    """Build a mock requests.Response for the SEC tickers endpoint.

    *tickers* maps ticker symbol → company title.
    """
    payload = {
        str(i): {"ticker": sym, "title": name}
        for i, (sym, name) in enumerate(tickers.items())
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = payload
    return mock_resp


class TestStartupCreatesDbIfMissing(unittest.TestCase):

    def test_startup_creates_db_if_missing(self):
        """If valid_tickers.json doesn't exist, run_startup_checks() must create it."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, "valid_tickers.json")
            name_file = os.path.join(tmp_dir, "ticker_names.json")
            lock_file = os.path.join(tmp_dir, "valid_tickers.lock")

            sec_data = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp", "NVDA": "NVIDIA"}

            with (
                patch.object(ticker_db, "_DATA_FILE", data_file),
                patch.object(ticker_db, "_NAME_FILE", name_file),
                patch.object(ticker_db, "_LOCK_FILE", lock_file),
                patch.object(ticker_db, "_ticker_cache", None),
                patch.object(ticker_db, "_name_cache", None),
                patch("ticker_db.requests.get", return_value=_make_sec_response(sec_data)),
            ):
                self.assertFalse(os.path.exists(data_file), "Pre-condition: file must not exist")
                ticker_db.run_startup_checks()
                self.assertTrue(os.path.exists(data_file), "DB file should be created after startup check")

                with open(data_file, encoding="utf-8") as f:
                    loaded = set(json.load(f))
                self.assertEqual(loaded, {"AAPL", "MSFT", "NVDA"})


class TestHealthReportsDbAge(unittest.TestCase):

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

    def test_health_reports_db_age(self):
        """GET /api/health must include ticker_db_age_hours and ticker_db_stale."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertIn("ticker_db_age_hours", data)
        self.assertIn("ticker_db_stale", data)
        self.assertIn("ticker_db_loaded", data)
        self.assertIn("ticker_count", data)

        # age_hours must be a non-negative number (or None if file missing)
        age = data["ticker_db_age_hours"]
        if age is not None:
            self.assertIsInstance(age, (int, float))
            self.assertGreaterEqual(age, 0)

        self.assertIsInstance(data["ticker_db_stale"], bool)

    def test_health_stale_flag_false_for_fresh_file(self):
        """ticker_db_stale must be False when the file was just written."""
        # Touch the data file to make it appear fresh
        with patch(
            "ticker_db.get_db_file_age_hours", return_value=1.0
        ), patch(
            "ticker_db.is_db_stale", return_value=False
        ):
            resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["ticker_db_stale"])


class TestRefreshUpdatesData(unittest.TestCase):

    def test_refresh_updates_data(self):
        """refresh_ticker_db() must detect added/removed tickers and update caches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, "valid_tickers.json")
            name_file = os.path.join(tmp_dir, "ticker_names.json")
            lock_file = os.path.join(tmp_dir, "valid_tickers.lock")

            # Seed an initial DB with 3 tickers
            initial = ["AAPL", "MSFT", "GOOG"]
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(initial, f)

            # New SEC data: removed GOOG, added NVDA and AMZN
            new_sec_data = {
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corp",
                "NVDA": "NVIDIA Corp",
                "AMZN": "Amazon.com Inc.",
            }

            with (
                patch.object(ticker_db, "_DATA_FILE", data_file),
                patch.object(ticker_db, "_NAME_FILE", name_file),
                patch.object(ticker_db, "_LOCK_FILE", lock_file),
                patch.object(ticker_db, "_ticker_cache", set(initial)),
                patch.object(ticker_db, "_name_cache", None),
                patch("ticker_db.requests.get", return_value=_make_sec_response(new_sec_data)),
            ):
                result = ticker_db.refresh_ticker_db()

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["ticker_count"], 4)
            self.assertEqual(result["added"], 2,    f"Expected 2 added, got {result['added']}")
            self.assertEqual(result["removed"], 1,  f"Expected 1 removed, got {result['removed']}")

            # Verify the file on disk reflects the new set
            with open(data_file, encoding="utf-8") as f:
                on_disk = set(json.load(f))
            self.assertEqual(on_disk, {"AAPL", "MSFT", "NVDA", "AMZN"})


if __name__ == "__main__":
    unittest.main()
