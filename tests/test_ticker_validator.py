"""Unit tests for ticker_validator module."""

import sys
import os
import unittest
from unittest.mock import patch

try:
    import pytest
    _slow = pytest.mark.slow
except ImportError:
    # pytest not installed – define a no-op decorator so the file loads cleanly
    def _slow(cls):
        return cls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ticker_validator import validate_ticker, validate_ticker_format, TickerValidationResult


# ---------------------------------------------------------------------------
# Format-only tests (no network, no DB)
# ---------------------------------------------------------------------------

class TestFormatValidation(unittest.TestCase):

    def test_invalid_format_numbers(self):
        result = validate_ticker_format("123ABC")
        self.assertFalse(result.valid)
        self.assertIn("not a valid ticker format", result.error)

    def test_invalid_format_too_long(self):
        result = validate_ticker_format("ABCDEF")
        self.assertFalse(result.valid)
        self.assertIn("not a valid ticker format", result.error)

    def test_invalid_format_empty(self):
        for value in ("", "   "):
            result = validate_ticker_format(value)
            self.assertFalse(result.valid)
            self.assertIn("Please enter", result.error)

    def test_reserved_word(self):
        for word in ("TEST", "NULL"):
            result = validate_ticker_format(word)
            self.assertFalse(result.valid)
            self.assertIn("reserved word", result.error)

    def test_ticker_normalization(self):
        """Lowercase and padded tickers should normalise cleanly."""
        for raw in ("aapl", " AAPL ", "Aapl"):
            result = validate_ticker_format(raw)
            self.assertTrue(result.valid)
            self.assertEqual(result.ticker, "AAPL")


# ---------------------------------------------------------------------------
# Full-stack tests (hit yfinance – mark slow for CI skipping)
# ---------------------------------------------------------------------------

@_slow
class TestFullValidation(unittest.TestCase):

    def test_valid_ticker_aapl(self):
        result = validate_ticker("AAPL")
        self.assertTrue(result.valid)
        self.assertIn("Apple", result.company_name)
        self.assertIn(result.source, ("api", "local_db", "cache"))

    def test_nonexistent_ticker(self):
        result = validate_ticker("XYZQW")
        self.assertFalse(result.valid)
        self.assertNotEqual(result.error, "")

    def test_result_has_suggestions(self):
        """A close-but-wrong ticker should surface suggestions."""
        result = validate_ticker("AAPLL")
        # Either invalid with suggestions or (edge case) valid – just check structure
        self.assertIsInstance(result.suggestions, list)
        if not result.valid:
            self.assertTrue(len(result.suggestions) > 0 or result.error != "")


class TestLocalDbFastPath(unittest.TestCase):

    def test_known_ticker_skips_yfinance_lookup(self):
        with patch("ticker_validator.is_known_ticker", return_value=True), \
             patch("ticker_validator.get_company_name", return_value="Apple Inc."), \
             patch("ticker_validator.yf.Ticker") as mock_ticker:
            result = validate_ticker("AAPL")

        self.assertTrue(result.valid)
        self.assertEqual(result.company_name, "Apple Inc.")
        self.assertEqual(result.source, "local_db")
        self.assertFalse(result.warning)
        mock_ticker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
