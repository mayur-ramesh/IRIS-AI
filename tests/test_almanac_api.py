"""Tests for the almanac comparison page and read-only almanac API routes."""

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as app_module
from data.almanac_2026.build_almanac_json import build_payload, build_structured_db_dump


class TestAlmanacAPI(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()
        app_module._almanac_data = None

    def tearDown(self):
        app_module._almanac_data = None

    def test_almanac_page_route(self):
        resp = self.client.get("/almanac")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Seasonality Comparison", resp.data)
        self.assertIn(b"IRIS predictions vs Stock Trader's Almanac 2026", resp.data)
        self.assertIn(b"Current Week", resp.data)

    def test_homepage_contains_almanac_nav_link(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'href="/almanac"', resp.data)

    def test_almanac_daily_specific_date(self):
        resp = self.client.get("/api/almanac/daily?date=2026-04-09")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["date"], "2026-04-09")
        self.assertEqual(data["s"], 61.9)
        self.assertEqual(data["icon"], "bull")

    def test_almanac_week_endpoint(self):
        resp = self.client.get("/api/almanac/week?start=2026-04-06")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["week_start"], "2026-04-06")
        self.assertEqual(data["week_end"], "2026-04-10")
        self.assertEqual(len(data["daily"]), 5)
        self.assertEqual(len(data["weekdays"]), 5)
        self.assertTrue(all(day["market_open"] for day in data["weekdays"]))
        self.assertTrue(all(day["almanac_available"] for day in data["weekdays"]))
        self.assertEqual(data["month_overview"]["name"], "April")

    def test_almanac_week_endpoint_uses_monday_to_friday_calendar_range(self):
        resp = self.client.get("/api/almanac/week?start=2025-12-29")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["week_start"], "2025-12-29")
        self.assertEqual(data["week_end"], "2026-01-02")
        self.assertEqual(list(data["daily"].keys()), ["2025-12-29", "2025-12-30", "2025-12-31", "2026-01-02"])
        self.assertEqual([day["date"] for day in data["weekdays"]], [
            "2025-12-29",
            "2025-12-30",
            "2025-12-31",
            "2026-01-01",
            "2026-01-02",
        ])
        self.assertTrue(data["weekdays"][0]["almanac_available"])
        self.assertEqual(data["weekdays"][0]["source_month"], "2026-01")
        self.assertTrue(data["weekdays"][1]["almanac_available"])
        self.assertTrue(data["weekdays"][2]["almanac_available"])
        self.assertEqual(data["weekdays"][3]["status"], "closed")
        self.assertFalse(data["weekdays"][3]["market_open"])
        self.assertIn("New Year's Day", data["weekdays"][3]["status_reason"])
        self.assertTrue(data["weekdays"][4]["almanac_available"])
        self.assertEqual(data["month_overview"]["name"], "January")

    def test_build_payload_includes_cross_year_january_lead_in_dates(self):
        payload = build_payload()
        daily = payload["daily"]
        self.assertIn("2025-12-29", daily)
        self.assertIn("2025-12-30", daily)
        self.assertIn("2025-12-31", daily)
        self.assertEqual(daily["2025-12-31"]["icon"], "bear")
        self.assertEqual(daily["2025-12-31"]["source_month"], "2026-01")
        self.assertIn("Last Trading Day of the Year", daily["2025-12-31"]["notes"])

    def test_almanac_month_endpoint(self):
        resp = self.client.get("/api/almanac/month/2026-04")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["month"]["name"], "April")
        self.assertEqual(data["month"]["vital_stats"]["sp500"]["rank"], 2)
        self.assertIn("2026-04-09", data["daily"])
        self.assertEqual(len(data["calendar_days"]), 30)
        good_friday = next(day for day in data["calendar_days"] if day["date"] == "2026-04-03")
        self.assertEqual(good_friday["status"], "closed")
        self.assertFalse(good_friday["market_open"])
        self.assertIn("Good Friday", good_friday["status_reason"])
        april_ninth = next(day for day in data["calendar_days"] if day["date"] == "2026-04-09")
        self.assertTrue(april_ninth["almanac_available"])
        self.assertEqual(april_ninth["s"], 61.9)

    def test_almanac_seasonal_endpoint(self):
        resp = self.client.get("/api/almanac/seasonal")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("heatmap", data)
        self.assertIn("signals", data)
        self.assertIn("months", data)
        self.assertEqual(data["heatmap"]["2026-10"]["sp500_midterm_rank"], 1)
        self.assertEqual(data["months"]["2026-08"]["vital_stats"]["sp500"]["midterm_avg"], -0.4)

    def test_build_payload_includes_november_and_december_seasonal_signals(self):
        payload = build_payload()
        seasonal_signals = payload["seasonal_signals"]
        november_signals = [signal for signal in seasonal_signals if signal["source_month"] == "2026-11"]
        december_signals = [signal for signal in seasonal_signals if signal["source_month"] == "2026-12"]

        self.assertEqual(len(november_signals), 6)
        self.assertEqual(len(december_signals), 6)
        self.assertEqual(
            {signal["label"] for signal in november_signals},
            {
                "Best Six Months Begins",
                "November First Trading Day Strong",
                "Midterm Election Bullish Window",
                "November OpEx Week Strong",
                "Thanksgiving Trade",
                "Pre-Thanksgiving Week Weakness",
            },
        )
        self.assertEqual(
            {signal["label"] for signal in december_signals},
            {
                "Santa Claus Rally",
                "Q4 Triple Witching Most Bullish",
                "January Effect Begins Mid-December",
                "Free Lunch Strategy",
                "December First Trading Day Weak",
                "Year-End Bearish Final Days",
            },
        )

    def test_almanac_daily_missing_date_returns_404(self):
        resp = self.client.get("/api/almanac/daily?date=2026-04-05")
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_almanac_month_missing_returns_404(self):
        resp = self.client.get("/api/almanac/month/2027-01")
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_almanac_missing_json_returns_404(self):
        with patch.object(app_module, "_load_almanac_data", return_value={"error": "almanac_2026.json not found"}):
            resp = self.client.get("/api/almanac/seasonal")
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertEqual(data["error"], "almanac_2026.json not found")

    def test_almanac_structured_json_db_fallback(self):
        fallback_payload = build_structured_db_dump(build_payload())
        temp_root = Path(__file__).resolve().parent / "tmp_almanac_fallback"
        if temp_root.exists():
            shutil.rmtree(temp_root)
        try:
            almanac_dir = temp_root / "data" / "almanac_2026"
            almanac_dir.mkdir(parents=True, exist_ok=True)
            (almanac_dir / "almanac_2026_db_dump.json").write_text(
                json.dumps(fallback_payload, indent=2),
                encoding="utf-8",
            )

            with patch.object(app_module, "PROJECT_ROOT", temp_root):
                app_module._almanac_data = None
                resp = self.client.get("/api/almanac/daily?date=2026-04-09")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["date"], "2026-04-09")
        self.assertEqual(data["s"], 61.9)
        self.assertEqual(data["icon"], "bull")


if __name__ == "__main__":
    unittest.main()
