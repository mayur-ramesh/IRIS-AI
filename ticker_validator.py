"""
Multi-layer ticker validation: format check → local DB → live yfinance API.

Usage:
    from ticker_validator import validate_ticker
    result = validate_ticker("AAPL")
    if result.valid:
        print(result.company_name)
    else:
        print(result.error, result.suggestions)
"""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

import yfinance as yf

from ticker_db import find_similar_tickers, is_known_ticker

logger = logging.getLogger(__name__)

_RESERVED_WORDS = {"HELP", "TEST", "NULL", "NONE", "NA", "N/A"}


@dataclass
class TickerValidationResult:
    valid: bool
    ticker: str
    company_name: str = ""
    error: str = ""
    suggestions: list[str] = field(default_factory=list)
    source: str = ""  # "cache" | "local_db" | "api" | ""


# ---------------------------------------------------------------------------
# Layer 1 – format
# ---------------------------------------------------------------------------

def validate_ticker_format(ticker: str) -> TickerValidationResult:
    """Check that *ticker* has a valid format before any DB or API call."""
    normalized = ticker.strip().upper()

    if not normalized:
        return TickerValidationResult(
            valid=False,
            ticker=normalized,
            error="Please enter a stock ticker symbol.",
        )

    if not re.fullmatch(r"[A-Z]{1,5}", normalized):
        return TickerValidationResult(
            valid=False,
            ticker=normalized,
            error=(
                f'"{ticker.strip()}" is not a valid ticker format. '
                "US stock tickers are 1-5 letters (e.g., AAPL, MSFT, TSLA)."
            ),
        )

    if normalized in _RESERVED_WORDS:
        return TickerValidationResult(
            valid=False,
            ticker=normalized,
            error=f'"{normalized}" is a reserved word, not a stock ticker.',
        )

    return TickerValidationResult(valid=True, ticker=normalized)


# ---------------------------------------------------------------------------
# Layer 2 + 3 – local DB then live API (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _cached_api_lookup(ticker: str) -> TickerValidationResult:
    """
    Inner function that hits yfinance; result is cached per ticker string.
    Only successful (non-network-error) results are stored by lru_cache because
    callers re-raise network exceptions before this function returns normally.
    ticker must already be normalised uppercase.
    """
    in_local_db = is_known_ticker(ticker)

    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("shortName") or info.get("longName") or ""

        if not company_name and not in_local_db:
            return TickerValidationResult(
                valid=False,
                ticker=ticker,
                error=f'Ticker "{ticker}" was not found. Please check the symbol and try again.',
                suggestions=find_similar_tickers(ticker),
            )

        # Valid so far – check for recent trading activity
        history = yf.Ticker(ticker).history(period="5d")
        if history.empty and company_name:
            return TickerValidationResult(
                valid=False,
                ticker=ticker,
                company_name=company_name,
                error=(
                    f'"{ticker}" ({company_name}) appears to be delisted '
                    "or has no recent trading data."
                ),
            )

        source = "local_db" if in_local_db else "api"
        return TickerValidationResult(
            valid=True,
            ticker=ticker,
            company_name=company_name or "(verified offline)",
            source=source,
        )

    except Exception as exc:
        # Network / timeout – bubble up so the caller decides what to return
        # (we must NOT return from here so lru_cache doesn't store the failure)
        logger.warning("yfinance lookup failed for %s: %s", ticker, exc)
        raise


def validate_ticker_exists(ticker: str) -> TickerValidationResult:
    """
    Full existence check: format → local DB → yfinance API.
    Handles network failures gracefully.
    """
    fmt = validate_ticker_format(ticker)
    if not fmt.valid:
        return fmt

    normalized = fmt.ticker
    in_local_db = is_known_ticker(normalized)

    try:
        result = _cached_api_lookup(normalized)
        # Upgrade source to "cache" on a cache hit (lru_cache doesn't tell us,
        # but we can detect it via cache_info between calls if needed; for now
        # we keep whatever _cached_api_lookup set and let the caller know it
        # came from our process cache by checking cache_info externally).
        return result
    except Exception:
        if in_local_db:
            return TickerValidationResult(
                valid=True,
                ticker=normalized,
                company_name="(verified offline)",
                source="local_db",
            )
        return TickerValidationResult(
            valid=False,
            ticker=normalized,
            error=f'Could not verify "{normalized}". Please check your connection and try again.',
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_ticker(ticker: str) -> TickerValidationResult:
    """Orchestrate the full validation flow and return a single result."""
    return validate_ticker_exists(ticker)
