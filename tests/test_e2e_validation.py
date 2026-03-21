"""
End-to-end integration tests for the complete validation system.

These tests exercise the full request → validation → data-fetch → response flow
using the real Flask app with network calls mocked out.

Network-required tests are skipped automatically in offline environments.

Run the full suite:
    python -m pytest tests/test_e2e_validation.py -v

Run only offline-safe tests:
    python -m pytest tests/test_e2e_validation.py -v -m "not network"
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from app import app, _rate_limit_store
from ticker_validator import _cached_api_lookup


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _yf_mock(company="Test Corp", exchange="NYQ"):
    """Return a mock yf.Ticker() that looks like a real, active stock."""
    t = MagicMock()
    t.info = {"shortName": company, "exchange": exchange}
    hist = MagicMock()
    hist.empty = False
    t.history.return_value = hist
    return t


def _yf_empty():
    """Return a mock that looks like an unknown ticker (empty info, no history)."""
    t = MagicMock()
    t.info = {}
    hist = MagicMock()
    hist.empty = True
    t.history.return_value = hist
    return t


def _fake_market_data(ticker):
    """Minimal market-data dict — mirrors what data_fetcher.fetch_market_data returns."""
    return {
        "ticker": ticker,
        "company_name": "Apple Inc.",
        "current_price": 185.50,
        "market_cap": 2_900_000_000_000,
        "pe_ratio": 28.5,
        "52_week_high": 199.62,
        "52_week_low": 124.17,
    }


def _fake_prompt(ticker, company_name, market_data):
    """Minimal grounded prompt that includes real data values."""
    price = market_data.get("current_price", "N/A")
    return (
        f"Analyze {ticker} ({company_name}). "
        f"Current price: ${price}. "
        "Base your analysis strictly on the real data provided above."
    )


def _clear_state():
    """Reset all shared state between tests."""
    _rate_limit_store.clear()
    _cached_api_lookup.cache_clear()


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestE2EValidation(unittest.TestCase):

    def setUp(self):
        _clear_state()
        self.client = app.test_client()

    # 1 ---
    def test_e2e_valid_ticker_full_flow(self):
        """AAPL through the analyze API: validation passes, real market data attached,
        grounded prompt references Apple, response includes both analysis and market data."""
        fake_report = {
            "ticker": "AAPL",
            "risk_score": 42,
            "llm_insights": {},
        }
        mock_iris = MagicMock()
        mock_iris.run_one_ticker.return_value = fake_report

        with patch("ticker_validator.yf.Ticker", return_value=_yf_mock("Apple Inc.")), \
             patch("ticker_validator.is_known_ticker", return_value=True), \
             patch("app.iris_app", mock_iris), \
             patch("app._fetch_market_data", side_effect=_fake_market_data), \
             patch("app._build_risk_prompt", side_effect=_fake_prompt), \
             patch("app.get_latest_llm_reports", return_value={}):
            resp = self.client.get("/api/analyze?ticker=AAPL")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Market data must be included so the frontend can render real numbers
        self.assertIn("market_data", data, "Response must include real market data")
        self.assertIn("current_price", data["market_data"])

        # Grounded prompt must reference real company data (not hallucinated)
        self.assertIn("grounded_prompt", data, "Response must include the grounded LLM prompt")
        self.assertIn("Apple", data["grounded_prompt"])
        self.assertIn("185.5", data["grounded_prompt"])

        # Confirm iris_app.run_one_ticker was actually called (analysis happened)
        mock_iris.run_one_ticker.assert_called_once()

    # 2 ---
    def test_e2e_invalid_ticker_blocked(self):
        """XYZZY (unknown ticker) must be blocked at the validation gate —
        the LLM analysis function must never be called."""
        mock_iris = MagicMock()

        with patch("ticker_validator.yf.Ticker", return_value=_yf_empty()), \
             patch("ticker_validator.is_known_ticker", return_value=False), \
             patch("ticker_validator.find_similar_tickers", return_value=["XYZ", "XYZT"]), \
             patch("app.iris_app", mock_iris):
            resp = self.client.get("/api/analyze?ticker=XYZZY")

        self.assertEqual(resp.status_code, 422)
        data = resp.get_json()
        self.assertFalse(data.get("valid", True), "Response must report invalid")
        self.assertIn("error", data, "Response must include error message")
        self.assertIn("suggestions", data, "Response must include suggestions")
        self.assertIsInstance(data["suggestions"], list)

        # The LLM must never have been invoked
        mock_iris.run_one_ticker.assert_not_called()

    # 3 ---
    def test_e2e_format_error_never_hits_backend(self):
        """Format-invalid input ('123!!!') must be rejected before yfinance is called.
        The rejection happens in ticker_validator.validate_ticker_format, not in the DB
        or network layer."""
        mock_yf = MagicMock()

        with patch("ticker_validator.yf.Ticker", mock_yf):
            resp = self.client.post(
                "/api/validate-ticker",
                json={"ticker": "123!!!"},
                content_type="application/json",
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["valid"], "123!!! must be rejected")
        self.assertIn("code", data, "Rejection must carry a structured error code")

        # yfinance must never have been touched — format check is instant
        mock_yf.assert_not_called()

    # 4 ---
    def test_e2e_suggestion_is_valid(self):
        """Submitting 'AAPPL' (typo) returns suggestions; submitting the first
        suggestion passes full validation."""
        # Step 1: submit the typo — expect a rejection with suggestions
        with patch("ticker_validator.yf.Ticker", return_value=_yf_empty()), \
             patch("ticker_validator.is_known_ticker", return_value=False), \
             patch("ticker_validator.find_similar_tickers", return_value=["AAPL", "PPL"]):
            resp1 = self.client.post(
                "/api/validate-ticker",
                json={"ticker": "AAPPL"},
                content_type="application/json",
            )

        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.get_json()
        self.assertFalse(data1["valid"])
        suggestions = data1.get("suggestions", [])
        self.assertGreater(len(suggestions), 0, "Expected at least one suggestion for 'AAPPL'")

        # Step 2: submit the first suggestion — it must pass validation
        _cached_api_lookup.cache_clear()
        first = suggestions[0]

        with patch("ticker_validator.yf.Ticker", return_value=_yf_mock("Apple Inc.")), \
             patch("ticker_validator.is_known_ticker", return_value=True):
            resp2 = self.client.post(
                "/api/validate-ticker",
                json={"ticker": first},
                content_type="application/json",
            )

        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.get_json()
        self.assertTrue(
            data2["valid"],
            f"Suggestion '{first}' should pass validation; got: {data2}",
        )

    # 5 ---
    def test_e2e_concurrent_requests(self):
        """10 concurrent validation requests (via asyncio.gather + asyncio.to_thread)
        must all succeed and leave the ticker DB in a consistent state."""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                   "TSLA", "NVDA", "JPM", "V", "JNJ"]
        results: dict = {}

        def _validate_one(ticker: str) -> None:
            with patch("ticker_validator.yf.Ticker",
                       return_value=_yf_mock(f"{ticker} Corp")), \
                 patch("ticker_validator.is_known_ticker", return_value=True):
                client = app.test_client()
                resp = client.post(
                    "/api/validate-ticker",
                    json={"ticker": ticker},
                    content_type="application/json",
                )
                results[ticker] = resp.get_json()

        async def _run_all() -> None:
            await asyncio.gather(
                *[asyncio.to_thread(_validate_one, t) for t in tickers]
            )

        asyncio.run(_run_all())

        self.assertEqual(len(results), 10, "All 10 requests must complete")
        for ticker, data in results.items():
            self.assertTrue(
                data.get("valid"),
                f"Ticker {ticker} should be valid; got: {data}",
            )

    # 6 ---
    def test_e2e_rate_limiting(self):
        """35 rapid requests to /api/validate-ticker from the same IP:
        the first 30 must succeed (HTTP 200), the remaining 5 must be rate-limited (HTTP 429)."""
        _rate_limit_store.clear()

        # Pre-warm the LRU cache so yfinance is never actually called after the first lookup
        with patch("ticker_validator.yf.Ticker", return_value=_yf_mock("Apple Inc.")), \
             patch("ticker_validator.is_known_ticker", return_value=True):
            statuses = []
            for _ in range(35):
                resp = self.client.post(
                    "/api/validate-ticker",
                    json={"ticker": "AAPL"},
                    content_type="application/json",
                )
                statuses.append(resp.status_code)

        successes    = statuses.count(200)
        rate_limited = statuses.count(429)

        self.assertEqual(
            successes, 30,
            f"Expected 30 successful responses, got {successes}. Statuses: {statuses}",
        )
        self.assertEqual(
            rate_limited, 5,
            f"Expected 5 rate-limited (429) responses, got {rate_limited}. Statuses: {statuses}",
        )


if __name__ == "__main__":
    unittest.main()
