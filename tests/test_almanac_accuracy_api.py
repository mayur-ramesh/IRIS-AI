"""Tests for the Almanac historic accuracy API routes."""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as app_module


ACCURACY_FIXTURE = {
    "meta": {
        "last_updated": "2026-04-05T12:00:00Z",
        "total_days_scored": 3,
        "data_range": {"from": "2026-01-02", "to": "2026-01-06"},
        "source": "Historic CSV backtest via scripts/seed_accuracy.py",
    },
    "daily": {
        "2026-01-02": {
            "actual": {"dji": 101.0, "sp500": 199.0, "nasdaq": 300.0},
            "prev_close": {"dji": 100.0, "sp500": 200.0, "nasdaq": 300.0},
            "pct_change": {"dji": 0.01, "sp500": -0.005, "nasdaq": 0.0},
            "almanac_scores": {"d": 60.0, "s": 40.0, "n": 50.0},
            "results": {
                "d": {"verdict": "HIT", "predicted": "UP", "actual": "UP"},
                "s": {"verdict": "HIT", "predicted": "DOWN", "actual": "DOWN"},
                "n": {"verdict": None, "predicted": None, "actual": "FLAT"},
            },
            "hits": 2,
            "total_calls": 2,
            "context": "Opening session",
        },
        "2026-01-05": {
            "actual": {"dji": 100.0, "sp500": 200.0, "nasdaq": 303.0},
            "prev_close": {"dji": 101.0, "sp500": 199.0, "nasdaq": 300.0},
            "pct_change": {"dji": -0.009901, "sp500": 0.005025, "nasdaq": 0.01},
            "almanac_scores": {"d": 45.0, "s": 55.0, "n": 70.0},
            "results": {
                "d": {"verdict": "HIT", "predicted": "DOWN", "actual": "DOWN"},
                "s": {"verdict": "HIT", "predicted": "UP", "actual": "UP"},
                "n": {"verdict": "HIT", "predicted": "UP", "actual": "UP"},
            },
            "hits": 3,
            "total_calls": 3,
            "context": "",
        },
        "2026-01-06": {
            "actual": {"dji": 102.0, "sp500": 198.0, "nasdaq": 300.0},
            "prev_close": {"dji": 100.0, "sp500": 200.0, "nasdaq": 303.0},
            "pct_change": {"dji": 0.02, "sp500": -0.01, "nasdaq": -0.009901},
            "almanac_scores": {"d": 80.0, "s": 20.0, "n": 60.0},
            "results": {
                "d": {"verdict": "HIT", "predicted": "UP", "actual": "UP"},
                "s": {"verdict": "HIT", "predicted": "DOWN", "actual": "DOWN"},
                "n": {"verdict": "MISS", "predicted": "UP", "actual": "DOWN"},
            },
            "hits": 2,
            "total_calls": 3,
            "context": "Momentum test",
        },
    },
    "weekly": {
        "2025-12-29": {
            "dates": ["2026-01-02"],
            "hits": 2,
            "total_calls": 2,
            "accuracy": 100.0,
            "dow": {"hits": 1, "total": 1, "pct": 100.0},
            "sp500": {"hits": 1, "total": 1, "pct": 100.0},
            "nasdaq": {"hits": 0, "total": 0, "pct": 0.0},
        },
        "2026-01-05": {
            "dates": ["2026-01-05", "2026-01-06"],
            "hits": 5,
            "total_calls": 6,
            "accuracy": 83.3,
            "dow": {"hits": 2, "total": 2, "pct": 100.0},
            "sp500": {"hits": 2, "total": 2, "pct": 100.0},
            "nasdaq": {"hits": 1, "total": 2, "pct": 50.0},
        },
    },
    "monthly": {
        "2026-01": {
            "hits": 7,
            "total_calls": 8,
            "accuracy": 87.5,
            "dow": {"hits": 3, "total": 3, "pct": 100.0},
            "sp500": {"hits": 3, "total": 3, "pct": 100.0},
            "nasdaq": {"hits": 1, "total": 2, "pct": 50.0},
            "trading_days": 3,
        }
    },
}

CROSS_YEAR_ACCURACY_FIXTURE = {
    "meta": {
        "last_updated": "2026-04-05T12:00:00Z",
        "total_days_scored": 3,
        "data_range": {"from": "2025-12-30", "to": "2026-01-02"},
        "source": "Historic CSV backtest via scripts/seed_accuracy.py",
    },
    "daily": {
        "2025-12-30": {
            "actual": {"dji": 99.0, "sp500": 199.0, "nasdaq": 299.0},
            "prev_close": {"dji": 100.0, "sp500": 200.0, "nasdaq": 300.0},
            "pct_change": {"dji": -0.01, "sp500": -0.005, "nasdaq": -0.003333},
            "almanac_scores": {"d": 42.9, "s": 42.9, "n": 38.1},
            "results": {
                "d": {"verdict": "HIT", "predicted": "DOWN", "actual": "DOWN"},
                "s": {"verdict": "HIT", "predicted": "DOWN", "actual": "DOWN"},
                "n": {"verdict": "HIT", "predicted": "DOWN", "actual": "DOWN"},
            },
            "hits": 3,
            "total_calls": 3,
            "context": "",
        },
        "2025-12-31": {
            "actual": {"dji": 98.0, "sp500": 198.0, "nasdaq": 298.0},
            "prev_close": {"dji": 99.0, "sp500": 199.0, "nasdaq": 299.0},
            "pct_change": {"dji": -0.010101, "sp500": -0.005025, "nasdaq": -0.003344},
            "almanac_scores": {"d": 33.3, "s": 28.6, "n": 28.6},
            "results": {
                "d": {"verdict": "HIT", "predicted": "DOWN", "actual": "DOWN"},
                "s": {"verdict": "HIT", "predicted": "DOWN", "actual": "DOWN"},
                "n": {"verdict": "HIT", "predicted": "DOWN", "actual": "DOWN"},
            },
            "hits": 3,
            "total_calls": 3,
            "context": "Last Trading Day of the Year",
        },
        "2026-01-02": {
            "actual": {"dji": 99.0, "sp500": 199.0, "nasdaq": 297.0},
            "prev_close": {"dji": 98.0, "sp500": 198.0, "nasdaq": 298.0},
            "pct_change": {"dji": 0.010204, "sp500": 0.005051, "nasdaq": -0.003356},
            "almanac_scores": {"d": 66.7, "s": 52.4, "n": 61.9},
            "results": {
                "d": {"verdict": "HIT", "predicted": "UP", "actual": "UP"},
                "s": {"verdict": "HIT", "predicted": "UP", "actual": "UP"},
                "n": {"verdict": "MISS", "predicted": "UP", "actual": "DOWN"},
            },
            "hits": 2,
            "total_calls": 3,
            "context": "First Trading Day of Year",
        },
    },
    "weekly": {
        "2025-12-29": {
            "dates": ["2025-12-30", "2025-12-31", "2026-01-02"],
            "hits": 8,
            "total_calls": 9,
            "accuracy": 88.9,
            "dow": {"hits": 3, "total": 3, "pct": 100.0},
            "sp500": {"hits": 3, "total": 3, "pct": 100.0},
            "nasdaq": {"hits": 2, "total": 3, "pct": 66.7},
        }
    },
    "monthly": {
        "2025-12": {
            "hits": 6,
            "total_calls": 6,
            "accuracy": 100.0,
            "dow": {"hits": 2, "total": 2, "pct": 100.0},
            "sp500": {"hits": 2, "total": 2, "pct": 100.0},
            "nasdaq": {"hits": 2, "total": 2, "pct": 100.0},
            "trading_days": 2,
        },
        "2026-01": {
            "hits": 2,
            "total_calls": 3,
            "accuracy": 66.7,
            "dow": {"hits": 1, "total": 1, "pct": 100.0},
            "sp500": {"hits": 1, "total": 1, "pct": 100.0},
            "nasdaq": {"hits": 0, "total": 1, "pct": 0.0},
            "trading_days": 1,
        },
    },
}


class TestAlmanacAccuracyAPI(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()
        app_module._accuracy_data = None
        app_module._accuracy_mtime = 0.0

    def tearDown(self):
        app_module._accuracy_data = None
        app_module._accuracy_mtime = 0.0

    def make_root(self, name: str) -> Path:
        root = Path(__file__).resolve().parent.parent / "tmp_feedback_test_main" / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def write_accuracy_fixture(self, root: Path) -> None:
        self.write_accuracy_payload(root, ACCURACY_FIXTURE)

    def write_accuracy_payload(self, root: Path, payload: dict) -> None:
        path = root / "data" / "almanac_2026" / "accuracy_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_accuracy_endpoint_reports_unavailable_when_seed_file_is_missing(self):
        temp_root = self.make_root("accuracy_api_unavailable")
        try:
            with patch.object(app_module, "PROJECT_ROOT", temp_root):
                resp = self.client.get("/api/almanac/accuracy")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["available"], False)
        self.assertIn("seed_accuracy.py", data["message"])

    def test_accuracy_endpoint_supports_all_single_date_and_range_queries(self):
        temp_root = self.make_root("accuracy_api_daily")
        self.write_accuracy_fixture(temp_root)
        try:
            with patch.object(app_module, "PROJECT_ROOT", temp_root):
                all_resp = self.client.get("/api/almanac/accuracy")
                day_resp = self.client.get("/api/almanac/accuracy?date=2026-01-06")
                range_resp = self.client.get("/api/almanac/accuracy?from=2026-01-05&to=2026-01-06")
                missing_resp = self.client.get("/api/almanac/accuracy?date=2026-01-07")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(all_resp.status_code, 200)
        self.assertIn("2026-01-02", all_resp.get_json()["daily"])
        self.assertEqual(day_resp.status_code, 200)
        self.assertEqual(day_resp.get_json()["results"]["n"]["verdict"], "MISS")
        self.assertEqual(range_resp.status_code, 200)
        self.assertEqual(sorted(range_resp.get_json()["daily"].keys()), ["2026-01-05", "2026-01-06"])
        self.assertEqual(missing_resp.status_code, 404)

    def test_accuracy_week_and_month_routes_return_expected_records(self):
        temp_root = self.make_root("accuracy_api_periods")
        self.write_accuracy_fixture(temp_root)
        try:
            with patch.object(app_module, "PROJECT_ROOT", temp_root):
                week_resp = self.client.get("/api/almanac/accuracy/week?start=2026-01-05")
                month_resp = self.client.get("/api/almanac/accuracy/month?month=2026-01")
                invalid_week_resp = self.client.get("/api/almanac/accuracy/week?start=not-a-date")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(week_resp.status_code, 200)
        self.assertEqual(week_resp.get_json()["hits"], 5)
        self.assertEqual(week_resp.get_json()["dates"], ["2026-01-05", "2026-01-06"])
        self.assertEqual(week_resp.get_json()["dow"]["pct"], 100.0)
        self.assertEqual(week_resp.get_json()["nasdaq"]["pct"], 50.0)
        self.assertEqual(week_resp.get_json()["week_start"], "2026-01-05")
        self.assertEqual(week_resp.get_json()["week_end"], "2026-01-09")
        self.assertEqual(month_resp.status_code, 200)
        self.assertEqual(month_resp.get_json()["trading_days"], 3)
        self.assertEqual(invalid_week_resp.status_code, 400)

    def test_accuracy_week_route_supports_cross_year_monday_week_starts(self):
        temp_root = self.make_root("accuracy_api_cross_year_week")
        self.write_accuracy_payload(temp_root, CROSS_YEAR_ACCURACY_FIXTURE)
        try:
            with patch.object(app_module, "PROJECT_ROOT", temp_root):
                week_resp = self.client.get("/api/almanac/accuracy/week?start=2025-12-29")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(week_resp.status_code, 200)
        data = week_resp.get_json()
        self.assertEqual(data["week_start"], "2025-12-29")
        self.assertEqual(data["week_end"], "2026-01-02")
        self.assertEqual(data["dates"], ["2025-12-30", "2025-12-31", "2026-01-02"])
        self.assertEqual(data["hits"], 8)
        self.assertEqual(data["total_calls"], 9)
        self.assertEqual(data["dow"]["pct"], 100.0)
        self.assertEqual(data["nasdaq"]["pct"], 66.7)

    def test_accuracy_summary_aggregates_monthly_and_per_index_totals(self):
        temp_root = self.make_root("accuracy_api_summary")
        self.write_accuracy_fixture(temp_root)
        try:
            with patch.object(app_module, "PROJECT_ROOT", temp_root):
                resp = self.client.get("/api/almanac/accuracy/summary")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["overall"]["hits"], 7)
        self.assertEqual(data["overall"]["total_calls"], 8)
        self.assertEqual(data["overall"]["accuracy"], 87.5)
        self.assertEqual(data["per_index"]["dow"]["pct"], 100.0)
        self.assertEqual(data["per_index"]["nasdaq"]["pct"], 50.0)
        self.assertEqual(data["last_scored_date"], "2026-01-06")
        self.assertEqual(data["total_days"], 3)


if __name__ == "__main__":
    unittest.main()
