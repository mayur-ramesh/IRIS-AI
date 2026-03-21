"""
Local ticker database for fast offline validation of stock ticker symbols.
Downloads SEC company tickers JSON and caches locally for quick lookups.
Includes background refresh scheduling and startup integrity checks.
"""

import difflib
import json
import logging
import os
import tempfile
import threading
import time

import requests

try:
    from filelock import FileLock as _FileLock
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False
    _FileLock = None

logger = logging.getLogger(__name__)

_DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "valid_tickers.json")
_NAME_FILE = os.path.join(os.path.dirname(__file__), "data", "ticker_names.json")
_LOCK_FILE = os.path.join(os.path.dirname(__file__), "data", "valid_tickers.lock")
_SEC_URL = "https://www.sec.gov/files/company_tickers.json"
_USER_AGENT = "IRIS-AI-Demo admin@iris-ai.app"
_MIN_TICKER_COUNT = 5000

_ticker_cache: set[str] | None = None
_name_cache: dict[str, str] | None = None
_cache_lock = threading.RLock()   # protects in-memory cache updates


# ---------------------------------------------------------------------------
# File-lock context manager (falls back to no-op if filelock not installed)
# ---------------------------------------------------------------------------

class _NoOpLock:
    """Drop-in replacement used when filelock is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


def _file_lock():
    """Return a file lock for the ticker database, or a no-op lock."""
    if _FILELOCK_AVAILABLE:
        os.makedirs(os.path.dirname(_LOCK_FILE), exist_ok=True)
        return _FileLock(_LOCK_FILE, timeout=30)
    return _NoOpLock()


# ---------------------------------------------------------------------------
# Age / staleness helpers
# ---------------------------------------------------------------------------

def get_db_file_age_hours() -> float | None:
    """Return the age of the ticker database file in hours.

    Returns None if the file does not exist.
    """
    if not os.path.exists(_DATA_FILE):
        return None
    mtime = os.path.getmtime(_DATA_FILE)
    return (time.time() - mtime) / 3600.0


def is_db_stale(threshold_hours: float = 48.0) -> bool:
    """Return True if the database file is older than *threshold_hours*."""
    age = get_db_file_age_hours()
    return age is None or age > threshold_hours


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, data) -> None:
    """Write *data* as JSON to *path* atomically (via a temp file + rename)."""
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Core database functions
# ---------------------------------------------------------------------------

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
        names = {entry["ticker"].upper(): entry.get("title", "") for entry in data.values()}
        tickers = set(names.keys())
        logger.info("Downloaded %d tickers from SEC.", len(tickers))

        with _file_lock():
            _atomic_write_json(_DATA_FILE, sorted(tickers))
            logger.info("Saved ticker database to %s.", _DATA_FILE)
            _atomic_write_json(_NAME_FILE, names)
            logger.info("Saved ticker names to %s.", _NAME_FILE)

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
    File access is protected by a file lock to prevent reading during writes.
    """
    global _ticker_cache
    with _cache_lock:
        if _ticker_cache is not None:
            return _ticker_cache

        if os.path.exists(_DATA_FILE):
            with _file_lock():
                with open(_DATA_FILE, encoding="utf-8") as f:
                    _ticker_cache = set(json.load(f))
            logger.info("Loaded %d tickers from %s.", len(_ticker_cache), _DATA_FILE)
        else:
            _ticker_cache = initialize_ticker_db()

        return _ticker_cache


def refresh_ticker_db() -> dict:
    """Download fresh SEC data, diff against existing, save atomically.

    Updates the in-memory cache on success.
    On failure, logs the error and keeps the existing data.

    Returns a dict with keys: status, ticker_count, added, removed (or error).
    """
    global _ticker_cache, _name_cache

    # Snapshot the old set for comparison (work from the cache to avoid a disk read)
    with _cache_lock:
        old_tickers: set[str] = set(_ticker_cache) if _ticker_cache is not None else set()
        if not old_tickers and os.path.exists(_DATA_FILE):
            try:
                with open(_DATA_FILE, encoding="utf-8") as f:
                    old_tickers = set(json.load(f))
            except Exception:
                pass

    try:
        response = requests.get(
            _SEC_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        names = {entry["ticker"].upper(): entry.get("title", "") for entry in data.values()}
        new_tickers = set(names.keys())

        added = new_tickers - old_tickers
        removed = old_tickers - new_tickers
        logger.info(
            "Ticker DB refresh: %d total (+%d added, -%d removed)",
            len(new_tickers), len(added), len(removed),
        )

        with _file_lock():
            _atomic_write_json(_DATA_FILE, sorted(new_tickers))
            _atomic_write_json(_NAME_FILE, names)

        # Update in-memory caches under the cache lock
        with _cache_lock:
            _ticker_cache = new_tickers
            _name_cache = names

        logger.info("Ticker DB refresh saved to disk and cache updated.")
        return {
            "status": "ok",
            "ticker_count": len(new_tickers),
            "added": len(added),
            "removed": len(removed),
        }

    except Exception as exc:
        logger.error("Ticker DB refresh failed: %s", exc)
        with _cache_lock:
            current_count = len(_ticker_cache) if _ticker_cache is not None else 0
        return {
            "status": "error",
            "error": str(exc),
            "ticker_count": current_count,
            "added": 0,
            "removed": 0,
        }


def run_startup_checks() -> None:
    """Run integrity checks on the ticker database at app startup.

    1. If the database file is missing, download it now.
    2. If the file is older than 7 days, trigger a background refresh.
    3. If the loaded set has fewer than 5000 tickers, re-download.
    """
    # 1. File missing → download synchronously so the app starts with data
    if not os.path.exists(_DATA_FILE):
        logger.warning("Ticker database not found. Downloading from SEC…")
        initialize_ticker_db()
        return

    # 2. File older than 7 days → background refresh (non-blocking)
    age_hours = get_db_file_age_hours()
    if age_hours is not None and age_hours > 7 * 24:
        logger.warning(
            "Ticker database is %.1f hours old (>7 days). Triggering background refresh.",
            age_hours,
        )
        threading.Thread(target=refresh_ticker_db, daemon=True, name="ticker-db-refresh").start()

    # 3. Sanity check on loaded count
    db = load_ticker_db()
    if len(db) < _MIN_TICKER_COUNT:
        logger.error(
            "Ticker database appears corrupted (%d tickers < %d minimum). Re-downloading.",
            len(db), _MIN_TICKER_COUNT,
        )
        threading.Thread(target=initialize_ticker_db, daemon=True, name="ticker-db-reinit").start()


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

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


def load_ticker_names() -> dict[str, str]:
    """Return a dict mapping uppercase ticker symbol → company name.

    Loaded from data/ticker_names.json if present; returns an empty dict otherwise.
    Result is cached in memory for the lifetime of the process.
    """
    global _name_cache
    with _cache_lock:
        if _name_cache is not None:
            return _name_cache
        if os.path.exists(_NAME_FILE):
            with open(_NAME_FILE, encoding="utf-8") as f:
                _name_cache = json.load(f)
            logger.info("Loaded %d ticker names from %s.", len(_name_cache), _NAME_FILE)
        else:
            logger.warning("Ticker names file not found at %s; names unavailable.", _NAME_FILE)
            _name_cache = {}
        return _name_cache


def search_tickers(query: str, limit: int = 8) -> list[dict]:
    """Return tickers whose symbol starts with *query* (case-insensitive prefix match).

    Results are sorted alphabetically. Each entry is a dict with keys
    ``ticker`` (str) and ``name`` (str, may be empty if names not loaded).
    Returns an empty list if *query* is empty.
    """
    q = query.strip().upper()
    if not q:
        return []
    db = load_ticker_db()
    names = load_ticker_names()
    matches = sorted(t for t in db if t.startswith(q))[:limit]
    return [{"ticker": t, "name": names.get(t, "")} for t in matches]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    initialize_ticker_db()

    print(is_known_ticker("AAPL"))    # True
    print(is_known_ticker("XYZABC")) # False
    print(find_similar_tickers("AAPPL"))  # ['AAPL', ...]
