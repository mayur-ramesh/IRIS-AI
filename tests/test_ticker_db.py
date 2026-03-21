"""Unit tests for ticker_db module."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ticker_db


SAMPLE_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]


def _patch_db(tickers: list[str]):
    """Context manager: replace the in-memory cache with a controlled set."""
    return patch.object(ticker_db, "_ticker_cache", set(tickers))


class TestKnownTicker(unittest.TestCase):
    def test_known_ticker_found(self):
        with _patch_db(SAMPLE_TICKERS):
            self.assertTrue(ticker_db.is_known_ticker("AAPL"))
            self.assertTrue(ticker_db.is_known_ticker("MSFT"))
            self.assertTrue(ticker_db.is_known_ticker("GOOGL"))

    def test_unknown_ticker_not_found(self):
        with _patch_db(SAMPLE_TICKERS):
            self.assertFalse(ticker_db.is_known_ticker("XYZABC"))
            self.assertFalse(ticker_db.is_known_ticker("123"))
            self.assertFalse(ticker_db.is_known_ticker(""))

    def test_ticker_normalization(self):
        with _patch_db(SAMPLE_TICKERS):
            self.assertTrue(ticker_db.is_known_ticker("aapl"))
            self.assertTrue(ticker_db.is_known_ticker(" AAPL "))


class TestSimilarTickers(unittest.TestCase):
    def test_similar_tickers_returns_results(self):
        with _patch_db(SAMPLE_TICKERS):
            results = ticker_db.find_similar_tickers("AAPPL")
            self.assertIn("AAPL", results)

    def test_similar_tickers_no_match(self):
        with _patch_db(SAMPLE_TICKERS):
            results = ticker_db.find_similar_tickers("ZZZZZQQQQQ")
            self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
