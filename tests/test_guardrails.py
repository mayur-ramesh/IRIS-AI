"""Tests for data_fetcher.py and prompt_builder.py guardrail modules."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import fetch_market_data
from prompt_builder import build_risk_analysis_prompt, validate_llm_output


def _sample_market_data() -> dict:
    return {
        "ticker": "AAPL",
        "currentPrice": 195.50,
        "regularMarketPrice": 195.50,
        "marketCap": 3_000_000_000_000,
        "trailingPE": 32.5,
        "forwardPE": 28.1,
        "beta": 1.24,
        "fiftyTwoWeekHigh": 220.0,
        "fiftyTwoWeekLow": 164.0,
        "dividendYield": 0.005,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "shortName": "Apple Inc.",
        "longName": "Apple Inc.",
        "price_history": [
            {
                "date": "2025-01-01",
                "open": 190.0,
                "high": 198.0,
                "low": 189.0,
                "close": 195.5,
                "volume": 50_000_000,
            }
        ],
        "fetched_at": "2025-01-01T00:00:00+00:00",
    }


class TestFetchMarketData(unittest.TestCase):

    def test_fetch_market_data_valid_ticker(self):
        """AAPL should return a dict with non-empty fields."""
        result = fetch_market_data("AAPL")
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result["ticker"], "AAPL")
        self.assertIn("currentPrice", result)
        self.assertIn("marketCap", result)
        self.assertIn("price_history", result)
        self.assertIn("fetched_at", result)
        # At least some fields must carry real values, not all DATA_NOT_AVAILABLE
        real_values = [
            v for v in result.values()
            if v not in ("DATA_NOT_AVAILABLE", [], None)
        ]
        self.assertGreater(len(real_values), 3, "Expected several real data fields for AAPL")

    def test_fetch_market_data_invalid_ticker(self):
        """An invalid ticker should return an error dict or all-unavailable fields."""
        result = fetch_market_data("XYZINVALIDTICKER999")
        if "error" in result:
            # Expected path: yfinance raised an exception
            self.assertIn("XYZINVALIDTICKER999", result["error"])
        else:
            # yfinance returned partial/empty info — all key fields should be DATA_NOT_AVAILABLE
            key_fields = ["currentPrice", "marketCap", "trailingPE"]
            all_na = all(result.get(f) == "DATA_NOT_AVAILABLE" for f in key_fields)
            self.assertTrue(
                all_na,
                "Invalid ticker key fields should be DATA_NOT_AVAILABLE, got: "
                + str({f: result.get(f) for f in key_fields}),
            )


class TestPromptBuilder(unittest.TestCase):

    def test_prompt_contains_real_data(self):
        """Built prompt must embed actual values from market_data."""
        market_data = _sample_market_data()
        prompt = build_risk_analysis_prompt("AAPL", "Apple Inc.", market_data)
        # Real price value appears in the prompt
        self.assertIn("195.5", prompt)
        # Ticker appears
        self.assertIn("AAPL", prompt)
        # Market cap value appears
        self.assertIn("3000000000000", prompt)
        # Company name appears in the analysis request section
        self.assertIn("Apple Inc.", prompt)

    def test_prompt_contains_guardrail_instructions(self):
        """Prompt must contain the hallucination-prevention instructions."""
        market_data = _sample_market_data()
        prompt = build_risk_analysis_prompt("AAPL", "Apple Inc.", market_data)
        self.assertIn("Do NOT invent", prompt)
        self.assertIn("DATA_NOT_AVAILABLE", prompt)
        self.assertIn("hallucinate", prompt)


class TestValidateLlmOutput(unittest.TestCase):

    def test_no_disclaimer_when_prices_match(self):
        """Output should be unchanged when LLM prices are within 10% of real."""
        market_data = _sample_market_data()  # currentPrice = 195.50
        text = "The stock is trading near $196.00, showing resilience."
        result = validate_llm_output(text, market_data)
        self.assertEqual(result, text, "No disclaimer expected for an accurate price")

    def test_disclaimer_appended_when_price_deviates(self):
        """Disclaimer must be appended when LLM mentions a price >10% off real."""
        market_data = _sample_market_data()  # currentPrice = 195.50
        text = "The stock recently traded at $300.00, well above its peers."
        result = validate_llm_output(text, market_data)
        self.assertIn("Note: Some figures", result)
        self.assertIn("verified numbers", result)


if __name__ == "__main__":
    unittest.main()
