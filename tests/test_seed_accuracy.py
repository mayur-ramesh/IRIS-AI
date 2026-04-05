"""CLI tests for scripts/seed_accuracy.py."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "seed_accuracy.py"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Date", "Open", "High", "Low", "Close"])
        writer.writeheader()
        writer.writerows(rows)


def write_primary_almanac(path: Path) -> None:
    payload = {
        "meta": {"source": "fixture", "year": 2026, "generated_at": "2026-04-05T00:00:00Z"},
        "months": {},
        "daily": {
            "2026-01-02": {"d": 60.0, "s": 40.0, "n": 50.0, "notes": "Opening session"},
            "2026-01-05": {"d": 45.0, "s": 55.0, "n": 70.0, "notes": ""},
            "2026-01-06": {"d": 80.0, "s": 20.0, "n": 60.0, "notes": "Momentum test"},
        },
        "seasonal_signals": [],
        "seasonal_heatmap": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_cross_year_primary_almanac(path: Path) -> None:
    payload = {
        "meta": {"source": "fixture", "year": 2026, "generated_at": "2026-04-05T00:00:00Z"},
        "months": {},
        "daily": {
            "2025-12-29": {"d": 42.9, "s": 47.6, "n": 38.1, "notes": ""},
            "2025-12-30": {"d": 42.9, "s": 42.9, "n": 38.1, "notes": ""},
            "2025-12-31": {
                "d": 33.3,
                "s": 28.6,
                "n": 28.6,
                "notes": "Last Trading Day of the Year",
            },
            "2026-01-02": {
                "d": 66.7,
                "s": 52.4,
                "n": 61.9,
                "notes": "First Trading Day of Year",
            },
        },
        "seasonal_signals": [],
        "seasonal_heatmap": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_fallback_almanac(path: Path) -> None:
    payload = {
        "daily_probabilities": {
            "rows": [
                {"date": "2026-01-02", "dow_prob": 60.0, "sp500_prob": 40.0, "nasdaq_prob": 50.0, "notes": "Opening session"},
                {"date": "2026-01-05", "dow_prob": 45.0, "sp500_prob": 55.0, "nasdaq_prob": 70.0, "notes": ""},
                {"date": "2026-01-06", "dow_prob": 80.0, "sp500_prob": 20.0, "nasdaq_prob": 60.0, "notes": "Momentum test"},
            ]
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def seed_fixture_history(root: Path) -> None:
    historical_dir = root / "data" / "historical"
    write_csv(
        historical_dir / "DJI_daily.csv",
        [
            {"Date": "12/31/2025", "Open": "0", "High": "0", "Low": "0", "Close": "100"},
            {"Date": "01/02/2026", "Open": "0", "High": "0", "Low": "0", "Close": "101"},
            {"Date": "01/05/2026", "Open": "0", "High": "0", "Low": "0", "Close": "100"},
            {"Date": "01/06/2026", "Open": "0", "High": "0", "Low": "0", "Close": "102"},
        ],
    )
    write_csv(
        historical_dir / "GSPC_daily.csv",
        [
            {"Date": "12/31/2025", "Open": "0", "High": "0", "Low": "0", "Close": "200"},
            {"Date": "01/02/2026", "Open": "0", "High": "0", "Low": "0", "Close": "199"},
            {"Date": "01/05/2026", "Open": "0", "High": "0", "Low": "0", "Close": "200"},
            {"Date": "01/06/2026", "Open": "0", "High": "0", "Low": "0", "Close": "198"},
        ],
    )
    write_csv(
        historical_dir / "IXIC_daily.csv",
        [
            {"Date": "12/31/2025", "Open": "0", "High": "0", "Low": "0", "Close": "300"},
            {"Date": "01/02/2026", "Open": "0", "High": "0", "Low": "0", "Close": "300"},
            {"Date": "01/05/2026", "Open": "0", "High": "0", "Low": "0", "Close": "303"},
            {"Date": "01/06/2026", "Open": "0", "High": "0", "Low": "0", "Close": "300"},
        ],
    )


def seed_cross_year_history(root: Path) -> None:
    historical_dir = root / "data" / "historical"
    write_csv(
        historical_dir / "DJI_daily.csv",
        [
            {"Date": "12/29/2025", "Open": "0", "High": "0", "Low": "0", "Close": "100"},
            {"Date": "12/30/2025", "Open": "0", "High": "0", "Low": "0", "Close": "99"},
            {"Date": "12/31/2025", "Open": "0", "High": "0", "Low": "0", "Close": "98"},
            {"Date": "01/02/2026", "Open": "0", "High": "0", "Low": "0", "Close": "99"},
        ],
    )
    write_csv(
        historical_dir / "GSPC_daily.csv",
        [
            {"Date": "12/29/2025", "Open": "0", "High": "0", "Low": "0", "Close": "200"},
            {"Date": "12/30/2025", "Open": "0", "High": "0", "Low": "0", "Close": "199"},
            {"Date": "12/31/2025", "Open": "0", "High": "0", "Low": "0", "Close": "198"},
            {"Date": "01/02/2026", "Open": "0", "High": "0", "Low": "0", "Close": "199"},
        ],
    )
    write_csv(
        historical_dir / "IXIC_daily.csv",
        [
            {"Date": "12/29/2025", "Open": "0", "High": "0", "Low": "0", "Close": "300"},
            {"Date": "12/30/2025", "Open": "0", "High": "0", "Low": "0", "Close": "299"},
            {"Date": "12/31/2025", "Open": "0", "High": "0", "Low": "0", "Close": "298"},
            {"Date": "01/02/2026", "Open": "0", "High": "0", "Low": "0", "Close": "297"},
        ],
    )


class TestSeedAccuracyScript(unittest.TestCase):
    def make_project_root(self, name: str) -> Path:
        root = REPO_ROOT / "tmp_feedback_test_main" / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def run_script(self, project_root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_seed_accuracy_generates_output_from_primary_almanac_json(self):
        project_root = self.make_project_root("seed_accuracy_primary")
        try:
            write_primary_almanac(project_root / "data" / "almanac_2026" / "almanac_2026.json")
            seed_fixture_history(project_root)

            result = self.run_script(project_root)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("January", result.stdout)
            self.assertIn("Q1 Total", result.stdout)

            output_path = project_root / "data" / "almanac_2026" / "accuracy_results.json"
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["meta"]["total_days_scored"], 3)
            self.assertEqual(payload["meta"]["data_range"]["from"], "2026-01-02")
            self.assertEqual(payload["meta"]["data_range"]["to"], "2026-01-06")
            self.assertEqual(payload["daily"]["2026-01-02"]["hits"], 2)
            self.assertEqual(payload["daily"]["2026-01-02"]["total_calls"], 2)
            self.assertEqual(payload["daily"]["2026-01-02"]["results"]["n"]["verdict"], None)
            self.assertEqual(payload["daily"]["2026-01-06"]["results"]["n"]["verdict"], "MISS")
            self.assertEqual(payload["weekly"]["2025-12-29"]["hits"], 2)
            self.assertEqual(payload["weekly"]["2026-01-05"]["nasdaq"]["pct"], 50.0)
            self.assertEqual(payload["monthly"]["2026-01"]["hits"], 7)
            self.assertEqual(payload["monthly"]["2026-01"]["total_calls"], 8)
            self.assertEqual(payload["monthly"]["2026-01"]["accuracy"], 87.5)
            self.assertEqual(payload["monthly"]["2026-01"]["trading_days"], 3)
        finally:
            shutil.rmtree(project_root, ignore_errors=True)

    def test_seed_accuracy_falls_back_to_structured_dump(self):
        project_root = self.make_project_root("seed_accuracy_fallback")
        try:
            write_fallback_almanac(project_root / "data" / "almanac_2026" / "almanac_2026_db_dump.json")
            seed_fixture_history(project_root)

            result = self.run_script(project_root)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output_path = project_root / "data" / "almanac_2026" / "accuracy_results.json"
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["daily"]["2026-01-05"]["hits"], 3)
            self.assertEqual(payload["daily"]["2026-01-06"]["context"], "Momentum test")
        finally:
            shutil.rmtree(project_root, ignore_errors=True)

    def test_seed_accuracy_scores_cross_year_january_lead_in_dates(self):
        project_root = self.make_project_root("seed_accuracy_cross_year")
        try:
            write_cross_year_primary_almanac(project_root / "data" / "almanac_2026" / "almanac_2026.json")
            seed_cross_year_history(project_root)

            result = self.run_script(project_root)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output_path = project_root / "data" / "almanac_2026" / "accuracy_results.json"
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["meta"]["total_days_scored"], 3)
            self.assertEqual(payload["meta"]["data_range"]["from"], "2025-12-30")
            self.assertEqual(payload["meta"]["data_range"]["to"], "2026-01-02")
            self.assertNotIn("2025-12-29", payload["daily"])
            self.assertIn("2025-12-30", payload["daily"])
            self.assertIn("2025-12-31", payload["daily"])
            self.assertEqual(payload["daily"]["2025-12-30"]["hits"], 3)
            self.assertEqual(payload["daily"]["2025-12-31"]["hits"], 3)
            self.assertEqual(payload["daily"]["2026-01-02"]["hits"], 2)
            self.assertEqual(payload["weekly"]["2025-12-29"]["dates"], ["2025-12-30", "2025-12-31", "2026-01-02"])
            self.assertEqual(payload["weekly"]["2025-12-29"]["hits"], 8)
            self.assertEqual(payload["weekly"]["2025-12-29"]["total_calls"], 9)
            self.assertEqual(payload["monthly"]["2025-12"]["hits"], 6)
            self.assertEqual(payload["monthly"]["2025-12"]["total_calls"], 6)
            self.assertEqual(payload["monthly"]["2026-01"]["hits"], 2)
            self.assertEqual(payload["monthly"]["2026-01"]["total_calls"], 3)
        finally:
            shutil.rmtree(project_root, ignore_errors=True)

    def test_seed_accuracy_returns_error_when_history_is_missing(self):
        project_root = self.make_project_root("seed_accuracy_missing_history")
        try:
            write_primary_almanac(project_root / "data" / "almanac_2026" / "almanac_2026.json")

            result = self.run_script(project_root)

            self.assertEqual(result.returncode, 1)
            self.assertIn("Missing historical CSV", result.stderr)
        finally:
            shutil.rmtree(project_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
