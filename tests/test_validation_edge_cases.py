"""
10 edge-case tests for the hardened validation layer.

All network calls are mocked so the suite is fast and deterministic.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ticker_validator import (
    ErrorCode,
    _cached_api_lookup,
    sanitize_ticker_input,
    validate_ticker,
    validate_ticker_format,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yf_ticker_mock(company="Test Corp", exchange="NYQ", history_empty=False):
    """Return a mock yf.Ticker() object with controllable behaviour."""
    t = MagicMock()
    t.info = {"shortName": company, "exchange": exchange}
    hist = MagicMock()
    hist.empty = history_empty
    t.history.return_value = hist
    return t


def _yf_ticker_empty():
    """Return a mock that looks like an unknown ticker (empty info)."""
    t = MagicMock()
    t.info = {}
    hist = MagicMock()
    hist.empty = True
    t.history.return_value = hist
    return t


def setUp_cache():
    """Clear the lru_cache before each test that exercises live-path logic."""
    _cached_api_lookup.cache_clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInputSanitisation(unittest.TestCase):

    def setUp(self):
        setUp_cache()

    # 1 ---
    def test_dollar_prefix_stripped(self):
        """$AAPL should validate the same as AAPL."""
        with patch(
            "ticker_validator.yf.Ticker", return_value=_yf_ticker_mock("Apple Inc.")
        ), patch("ticker_validator.is_known_ticker", return_value=True):
            result = validate_ticker("$AAPL")
        self.assertTrue(result.valid, f"Expected valid, got: {result.error}")
        self.assertEqual(result.ticker, "AAPL")

    # 2 ---
    def test_internal_spaces_stripped(self):
        """'A A P L' should resolve to 'AAPL' and validate."""
        with patch(
            "ticker_validator.yf.Ticker", return_value=_yf_ticker_mock("Apple Inc.")
        ), patch("ticker_validator.is_known_ticker", return_value=True):
            result = validate_ticker("A A P L")
        self.assertTrue(result.valid, f"Expected valid, got: {result.error}")
        self.assertEqual(result.ticker, "AAPL")

    # 3 ---
    def test_ticker_with_dot(self):
        """BRK.B should pass format validation (dot-suffix allowed)."""
        result = validate_ticker_format("BRK.B")
        self.assertTrue(result.valid, f"Expected valid format for BRK.B, got: {result.error}")
        self.assertEqual(result.ticker, "BRK.B")

    # 4 ---
    def test_crypto_ticker_rejected(self):
        """BTC should be rejected with RESERVED_WORD code and a helpful message."""
        result = validate_ticker_format("BTC")
        self.assertFalse(result.valid)
        self.assertEqual(result.code, ErrorCode.RESERVED_WORD)
        self.assertIn("crypto", result.error.lower())

    # 5 ---
    def test_etf_ticker_valid(self):
        """SPY (ETF) should be valid — ETFs live in the SEC database."""
        with patch(
            "ticker_validator.yf.Ticker", return_value=_yf_ticker_mock("SPDR S&P 500 ETF", "PCX")
        ), patch("ticker_validator.is_known_ticker", return_value=True):
            result = validate_ticker("SPY")
        self.assertTrue(result.valid, f"Expected ETF SPY to be valid, got: {result.error}")

    # 6 ---
    def test_very_long_input_rejected(self):
        """A 50-character input string must be rejected after sanitisation."""
        long_input = "A" * 50
        # After the 20-char cap, sanitised value is "AAAAAAAAAAAAAAAAAAAA" (20 chars)
        # which fails the 1-5 letter regex → INVALID_FORMAT
        result = validate_ticker_format(long_input)
        self.assertFalse(result.valid)
        self.assertIn(result.code, (ErrorCode.INVALID_FORMAT, ErrorCode.EMPTY_INPUT))

    # 7 ---
    def test_special_characters_rejected(self):
        """'AAPL!' must be rejected as INVALID_FORMAT."""
        result = validate_ticker_format("AAPL!")
        self.assertFalse(result.valid)
        self.assertEqual(result.code, ErrorCode.INVALID_FORMAT)


class TestGracefulDegradation(unittest.TestCase):

    def setUp(self):
        setUp_cache()

    # 8 ---
    def test_graceful_degradation_api_down(self):
        """When yfinance times out but ticker IS in local DB, return valid with warning."""
        with patch(
            "ticker_validator.yf.Ticker", side_effect=TimeoutError("Connection timed out")
        ), patch("ticker_validator.is_known_ticker", return_value=True):
            result = validate_ticker("AAPL")

        self.assertTrue(result.valid, "Should degrade gracefully to local DB")
        self.assertIn("local database", result.warning.lower())
        self.assertEqual(result.source, "local_db")

    # 9 ---
    def test_both_services_down(self):
        """When both yfinance AND the local DB are unavailable, return a specific error."""
        with patch(
            "ticker_validator.yf.Ticker", side_effect=Exception("API unreachable")
        ), patch(
            "ticker_validator.is_known_ticker", side_effect=Exception("DB corrupted")
        ):
            result = validate_ticker("AAPL")

        self.assertFalse(result.valid)
        self.assertIn("temporarily unavailable", result.error.lower())
        self.assertEqual(result.code, ErrorCode.API_ERROR)


class TestErrorCodePresence(unittest.TestCase):

    def setUp(self):
        setUp_cache()

    # 10 ---
    def test_error_code_present_on_every_rejection(self):
        """Every rejection scenario must carry a non-empty 'code' field."""
        cases = [
            ("",          ErrorCode.EMPTY_INPUT),
            ("123",       ErrorCode.INVALID_FORMAT),
            ("BTC",       ErrorCode.RESERVED_WORD),
            ("NULL",      ErrorCode.RESERVED_WORD),
            ("TOOLONGTIC", ErrorCode.INVALID_FORMAT),
        ]
        for raw, expected_code in cases:
            result = validate_ticker_format(raw)
            self.assertFalse(result.valid, f"Expected '{raw}' to be invalid")
            self.assertEqual(
                result.code, expected_code,
                f"Input '{raw}': expected code {expected_code!r}, got {result.code!r}",
            )

        # API-path rejections also carry codes
        with patch(
            "ticker_validator.yf.Ticker", return_value=_yf_ticker_empty()
        ), patch(
            "ticker_validator.is_known_ticker", return_value=False
        ), patch(
            "ticker_validator.find_similar_tickers", return_value=[]
        ):
            result = validate_ticker("ZZZZ")
        self.assertFalse(result.valid)
        self.assertTrue(result.code, f"Expected a non-empty code, got: {result.code!r}")
        self.assertEqual(result.code, ErrorCode.TICKER_NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
