"""Tests for the Almanac IRIS snapshot and refresh API routes."""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as app_module
import iris_mvp


def build_report(
    symbol: str,
    *,
    current_price: float,
    predicted_price: float,
    session_date: str,
    generated_at: str,
    trend_label: str = "STRONG UPTREND",
    investment_signal: str = "BUY",
    check_engine_light: str = " GREEN (Safe to Proceed)",
    pct_change: float = 1.25,
    direction: str = "upward",
):
    return {
        "meta": {
            "symbol": symbol,
            "source_symbol": symbol,
            "generated_at": generated_at,
            "market_session_date": session_date,
            "horizon_days": 1,
        },
        "market": {
            "current_price": current_price,
            "predicted_price_next_session": predicted_price,
            "predicted_price_horizon": predicted_price,
        },
        "signals": {
            "trend_label": trend_label,
            "investment_signal": investment_signal,
            "check_engine_light": check_engine_light,
            "sentiment_score": 0.0,
            "iris_reasoning": {
                "pct_change": pct_change,
                "direction": direction,
                "top_factors": ["SMA(10)", "Day Trend", "SMA(20)"],
            },
            "model_confidence": 92.5,
        },
        "all_horizons": {
            "1D": {
                "predicted_price": predicted_price,
                "trend_label": trend_label,
                "investment_signal": investment_signal,
                "iris_reasoning": {
                    "pct_change": pct_change,
                    "direction": direction,
                    "top_factors": ["SMA(10)", "Day Trend", "SMA(20)"],
                },
                "model_confidence": 92.5,
            }
        },
    }


class DummyModel:
    feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])


class FakeIrisApp:
    def get_market_data(self, ticker):
        history_df = pd.DataFrame(
            {"rsi_14": [48.0, 56.0]},
            index=pd.to_datetime(["2026-04-01", "2026-04-02"]),
        )
        base_prices = {
            "SPY": 650.0,
            "^DJI": 46000.0,
            "^GSPC": 6550.0,
            "^IXIC": 21800.0,
        }
        return {
            "current_price": base_prices[ticker],
            "history_df": history_df,
        }

    def predict_trend(self, data, sentiment_score, horizon_days=1):
        current_price = float(data["current_price"])
        predicted_price = current_price * 1.01
        return (
            "STRONG UPTREND",
            predicted_price,
            [predicted_price],
            [predicted_price * 1.02],
            [predicted_price * 0.98],
            DummyModel(),
            88.4,
        )


class TestAlmanacIrisAPI(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()
        self.original_data_dir = app_module.DATA_DIR
        self.original_iris_app = app_module.iris_app
        app_module._iris_snapshot_cache = {"data": None, "ts": 0.0}

    def tearDown(self):
        app_module.DATA_DIR = self.original_data_dir
        app_module.iris_app = self.original_iris_app
        app_module._iris_snapshot_cache = {"data": None, "ts": 0.0}

    def make_root(self, name: str) -> Path:
        root = Path(__file__).resolve().parent.parent / "tmp_feedback_test_main" / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_iris_snapshot_reads_latest_valid_reports_from_data_dir(self):
        temp_root = self.make_root("almanac_iris_snapshot")
        try:
            (temp_root / "SPY_report.json").write_text(
                json.dumps(
                    [
                        build_report(
                            "SPY",
                            current_price=640.0,
                            predicted_price=646.4,
                            session_date="2026-04-01",
                            generated_at="2026-04-01T12:00:00Z",
                        )
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            # The latest entry is intentionally invalid due to low price and must be skipped.
            (temp_root / "^DJI_report.json").write_text(
                json.dumps(
                    [
                        build_report(
                            "^DJI",
                            current_price=46200.0,
                            predicted_price=46350.0,
                            session_date="2026-04-02",
                            generated_at="2026-04-02T12:00:00Z",
                            investment_signal="BUY",
                        ),
                        build_report(
                            "DJI",
                            current_price=88.0,
                            predicted_price=91.0,
                            session_date="2026-04-03",
                            generated_at="2026-04-03T12:00:00Z",
                            investment_signal="BUY",
                        ),
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            (temp_root / "^IXIC_report.json").write_text(
                json.dumps(
                    build_report(
                        "^IXIC",
                        current_price=21850.0,
                        predicted_price=21960.0,
                        session_date="2026-04-02",
                        generated_at="2026-04-02T11:00:00Z",
                        investment_signal="SELL",
                        trend_label="STRONG DOWNTREND",
                        check_engine_light=" RED (Risk Detected - Caution)",
                        pct_change=-0.5,
                        direction="downward",
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )

            with patch.object(app_module, "DATA_DIR", temp_root):
                resp = self.client.get("/api/almanac/iris-snapshot")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["indices"]["spy"]["available"])
        self.assertTrue(data["indices"]["dji"]["available"])
        self.assertEqual(data["indices"]["dji"]["session_date"], "2026-04-02")
        self.assertFalse(data["indices"]["gspc"]["available"])
        self.assertEqual(data["indices"]["ixic"]["direction"], "downward")
        self.assertEqual(data["indices"]["ixic"]["investment_signal"], "SELL")

    def test_iris_refresh_returns_lightweight_predictions_for_all_indices(self):
        app_module.iris_app = FakeIrisApp()
        app_module._iris_snapshot_cache = {"data": {"stale": True}, "ts": 123.0}

        resp = self.client.get("/api/almanac/iris-refresh")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(sorted(data["indices"].keys()), ["dji", "gspc", "ixic", "spy"])
        self.assertTrue(all(entry["available"] for entry in data["indices"].values()))
        self.assertEqual(data["indices"]["dji"]["direction"], "upward")
        self.assertEqual(data["indices"]["gspc"]["investment_signal"], "STRONG BUY")
        self.assertEqual(data["indices"]["ixic"]["source"], "live_rf_prediction")
        self.assertEqual(app_module._iris_snapshot_cache["data"], None)
        self.assertEqual(app_module._iris_snapshot_cache["ts"], 0.0)

    def test_default_tickers_include_spy_and_three_indices(self):
        for ticker in ("SPY", "^DJI", "^GSPC", "^IXIC"):
            self.assertIn(ticker, iris_mvp.DEFAULT_TICKERS)


if __name__ == "__main__":
    unittest.main()
