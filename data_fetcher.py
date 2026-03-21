"""
Fetches verified market data from yfinance for a validated ticker.
Provides a clean data dict that anchors LLM analysis to real numbers.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

_NA = "DATA_NOT_AVAILABLE"

_INFO_FIELDS = [
    "currentPrice",
    "regularMarketPrice",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "beta",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "dividendYield",
    "sector",
    "industry",
    "shortName",
    "longName",
]


def _clean(value: Any) -> Any:
    """Return value if meaningful, else DATA_NOT_AVAILABLE."""
    if value is None:
        return _NA
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return _NA
    return value


def fetch_market_data(ticker: str) -> dict:
    """Fetch real market data from yfinance for *ticker*.

    Returns a dict with:
      - Info fields from yfinance (None / NaN → "DATA_NOT_AVAILABLE")
      - price_history: list of {date, open, high, low, close, volume}
        for the last 30 calendar days (empty list if unavailable)
      - fetched_at: ISO-8601 UTC timestamp

    On any unrecoverable error returns {"error": "Failed to fetch data for {ticker}"}.
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return {"error": f"Failed to fetch data for {ticker}"}

    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}

        data: dict = {
            "ticker": symbol,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        for field in _INFO_FIELDS:
            data[field] = _clean(info.get(field))

        # 30-day daily price history
        try:
            hist = stock.history(period="30d", interval="1d")
            price_history: list[dict] = []
            if hist is not None and not hist.empty:
                for ts, row in hist.iterrows():
                    try:
                        date_str = str(ts.date()) if hasattr(ts, "date") else str(ts)[:10]
                    except Exception:
                        date_str = str(ts)[:10]

                    def _safe_float(series_val) -> Any:
                        try:
                            v = float(series_val)
                            return _clean(v)
                        except (TypeError, ValueError):
                            return _NA

                    def _safe_int(series_val) -> Any:
                        try:
                            v = int(series_val)
                            return v
                        except (TypeError, ValueError):
                            return _NA

                    price_history.append({
                        "date": date_str,
                        "open":   _safe_float(row.get("Open")),
                        "high":   _safe_float(row.get("High")),
                        "low":    _safe_float(row.get("Low")),
                        "close":  _safe_float(row.get("Close")),
                        "volume": _safe_int(row.get("Volume")),
                    })
            data["price_history"] = price_history
        except Exception as hist_err:
            logger.warning("Could not fetch price history for %s: %s", symbol, hist_err)
            data["price_history"] = []

        # Warn when yfinance returns suspiciously sparse info
        if len(info) < 3:
            logger.warning("yfinance returned very sparse info for %s — ticker may be invalid", symbol)

        return data

    except Exception as exc:
        logger.error("fetch_market_data failed for %s: %s", symbol, exc)
        return {"error": f"Failed to fetch data for {ticker}"}
