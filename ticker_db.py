"""
Local ticker database for fast offline validation of stock ticker symbols.
Downloads SEC company tickers JSON and caches locally for quick lookups.
"""

import difflib
import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

_DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "valid_tickers.json")
_SEC_URL = "https://www.sec.gov/files/company_tickers.json"
_USER_AGENT = "IRIS-AI-Demo admin@iris-ai.app"

_ticker_cache: set[str] | None = None


def initialize_ticker_db() -> set[str]:
    """Download SEC tickers, parse into a set, and save to data/valid_tickers.json.

    Falls back to the local file if the download fails.
    Returns the set of uppercase ticker symbols.
    """
    tickers: set[str] | None = None

    try:
        response = requests.get(
            _SEC_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        tickers = {entry["ticker"].upper() for entry in data.values()}
        logger.info("Downloaded %d tickers from SEC.", len(tickers))

        os.makedirs(os.path.dirname(_DATA_FILE), exist_ok=True)
        with open(_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(tickers), f)
        logger.info("Saved ticker database to %s.", _DATA_FILE)

    except Exception as exc:
        logger.warning("Failed to download SEC tickers: %s", exc)
        if os.path.exists(_DATA_FILE):
            logger.info("Falling back to local file %s.", _DATA_FILE)
            with open(_DATA_FILE, encoding="utf-8") as f:
                tickers = set(json.load(f))
            logger.info("Loaded %d tickers from local fallback.", len(tickers))
        else:
            logger.error("No local ticker file found. Returning empty set.")
            tickers = set()

    return tickers


def load_ticker_db() -> set[str]:
    """Return the set of valid tickers, loading from disk (or SEC) as needed.

    The result is cached in memory for the lifetime of the process.
    """
    global _ticker_cache
    if _ticker_cache is not None:
        return _ticker_cache

    if os.path.exists(_DATA_FILE):
        with open(_DATA_FILE, encoding="utf-8") as f:
            _ticker_cache = set(json.load(f))
        logger.info("Loaded %d tickers from %s.", len(_ticker_cache), _DATA_FILE)
    else:
        _ticker_cache = initialize_ticker_db()

    return _ticker_cache


def is_known_ticker(ticker: str) -> bool:
    """Return True if *ticker* is present in the local ticker database."""
    return ticker.strip().upper() in load_ticker_db()


def find_similar_tickers(ticker: str, max_results: int = 3) -> list[str]:
    """Return up to *max_results* ticker symbols close to *ticker*.

    Uses difflib.get_close_matches with cutoff=0.6.
    Returns an empty list if no close matches are found.
    """
    normalized = ticker.strip().upper()
    db = load_ticker_db()
    return difflib.get_close_matches(normalized, db, n=max_results, cutoff=0.6)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    initialize_ticker_db()

    print(is_known_ticker("AAPL"))    # True
    print(is_known_ticker("XYZABC")) # False
    print(find_similar_tickers("AAPPL"))  # ['AAPL', ...]
