"""
Multi-layer ticker validation: sanitise → format → local DB → live yfinance API.

All error responses carry a structured error code in the ``code`` field.
Implements a graceful fallback chain when external services are degraded.
"""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

import yfinance as yf

from ticker_db import find_similar_tickers, get_company_name, is_known_ticker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured error codes
# ---------------------------------------------------------------------------

class ErrorCode:
    EMPTY_INPUT       = "EMPTY_INPUT"
    INVALID_FORMAT    = "INVALID_FORMAT"
    RESERVED_WORD     = "RESERVED_WORD"
    TICKER_NOT_FOUND  = "TICKER_NOT_FOUND"
    TICKER_DELISTED   = "TICKER_DELISTED"
    API_TIMEOUT       = "API_TIMEOUT"
    API_ERROR         = "API_ERROR"
    RATE_LIMITED      = "RATE_LIMITED"
    DATA_FETCH_FAILED = "DATA_FETCH_FAILED"
    INTERNAL_ERROR    = "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Known non-stock inputs
# ---------------------------------------------------------------------------

_RESERVED_WORDS = {"HELP", "TEST", "NULL", "NONE", "NA", "N/A"}

_CRYPTO_TICKERS = {
    "BTC", "ETH", "XRP", "LTC", "BNB", "SOL", "ADA", "DOT",
    "AVAX", "DOGE", "MATIC", "SHIB", "TRX", "LINK", "ATOM", "USDT", "USDC",
}

_CRYPTO_MESSAGE = (
    "IRIS-AI analyzes stocks and ETFs. "
    "For cryptocurrency analysis, please use a crypto-specific platform."
)

# OTC exchange identifiers returned by yfinance
_OTC_EXCHANGES = {"PNK", "OTC", "OTCQB", "OTCQX", "PINK", "GREY", "EXPERT"}

_MAX_RAW_LENGTH = 20   # chars before any processing


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TickerValidationResult:
    valid: bool
    ticker: str
    company_name: str = ""
    error: str = ""
    code: str = ""          # structured error code (empty on success)
    warning: str = ""       # non-fatal advisory (e.g. OTC market)
    suggestions: list[str] = field(default_factory=list)
    source: str = ""        # "cache" | "local_db" | "api" | "special_symbol" | ""


# ---------------------------------------------------------------------------
# Layer 0 – input sanitisation
# ---------------------------------------------------------------------------

def sanitize_ticker_input(raw: str) -> str:
    """Return a cleaned, uppercase ticker string from arbitrary user input.

    Steps applied in order:
    1. Enforce a 20-character hard cap before any further processing.
    2. Remove leading ``$`` or ``#`` characters.
    3. Remove ``ticker:`` prefix (case-insensitive).
    4. Remove common trailing words: ``stock``, ``etf``, ``shares``.
    5. Collapse all internal whitespace so "A A P L" becomes "AAPL".
    6. Uppercase.
    """
    s = str(raw or "").strip()
    if len(s) > _MAX_RAW_LENGTH:
        s = s[:_MAX_RAW_LENGTH]
    s = re.sub(r"^[\$#]+", "", s)
    s = re.sub(r"^ticker:", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(stock|etf|shares)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "", s)
    return s.upper()


# ---------------------------------------------------------------------------
# Layer 1 – format validation
# ---------------------------------------------------------------------------

# Standard US tickers: 1-5 letters, optionally ONE dot + 1-2 letters.
# Covers BRK.B, BRK.A, class shares, etc.
_STANDARD_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
# Preferred share tickers: base symbol + hyphen + series code.
# Covers T-PA, BAC-PB, WFC-PL, JPM-PC, etc.
_PREFERRED_TICKER_RE = re.compile(r"^[A-Z]{1,5}-[A-Z0-9]{1,3}$")
# Yahoo special symbols:
# - Indices: ^GSPC, ^IXIC, ^DJI
# - Futures: CL=F, GC=F, SI=F, HG=F
# - Composite symbols: DX-Y.NYB
_INDEX_TICKER_RE = re.compile(r"^\^[A-Z0-9.\-]{1,14}$")
_FUTURES_TICKER_RE = re.compile(r"^[A-Z0-9]{1,8}=F$")
_COMPOSITE_TICKER_RE = re.compile(r"^[A-Z0-9]{1,8}-[A-Z0-9]{1,8}\.[A-Z]{1,6}$")


def _is_special_market_symbol(ticker: str) -> bool:
    return bool(
        _INDEX_TICKER_RE.fullmatch(ticker)
        or _FUTURES_TICKER_RE.fullmatch(ticker)
        or _COMPOSITE_TICKER_RE.fullmatch(ticker)
    )


def validate_ticker_format(ticker: str) -> TickerValidationResult:
    """Check that *ticker* has a valid format (sanitises input first)."""
    normalized = sanitize_ticker_input(ticker)

    if not normalized:
        return TickerValidationResult(
            valid=False, ticker=normalized, code=ErrorCode.EMPTY_INPUT,
            error="Please enter a stock ticker symbol.",
        )

    if normalized in _CRYPTO_TICKERS:
        return TickerValidationResult(
            valid=False, ticker=normalized, code=ErrorCode.RESERVED_WORD,
            error=_CRYPTO_MESSAGE,
        )

    is_standard = bool(_STANDARD_TICKER_RE.fullmatch(normalized))
    is_preferred = bool(_PREFERRED_TICKER_RE.fullmatch(normalized))
    is_special = _is_special_market_symbol(normalized)
    if not (is_standard or is_preferred or is_special):
        return TickerValidationResult(
            valid=False, ticker=normalized, code=ErrorCode.INVALID_FORMAT,
            error=(
                f'"{normalized}" is not a valid ticker format. '
                "Use stock format (e.g., AAPL, BRK.B), preferred shares (e.g., T-PA, BAC-PB), "
                "or special market symbols (e.g., ^GSPC, CL=F)."
            ),
        )

    if normalized in _RESERVED_WORDS:
        return TickerValidationResult(
            valid=False, ticker=normalized, code=ErrorCode.RESERVED_WORD,
            error=f'"{normalized}" is a reserved word, not a stock ticker.',
        )

    return TickerValidationResult(valid=True, ticker=normalized)


# ---------------------------------------------------------------------------
# Layers 2 + 3 – local DB then live yfinance API (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _cached_api_lookup(ticker: str) -> TickerValidationResult:
    """Hit yfinance for *ticker* (already normalised uppercase).

    lru_cache stores only successful returns; raised exceptions are never cached,
    so transient network failures do not permanently poison the cache.
    """
    in_local_db = is_known_ticker(ticker)

    # info call — may raise on network error; let it propagate so the caller
    # can apply the graceful-degradation fallback chain.
    info = yf.Ticker(ticker).info
    company_name = info.get("shortName") or info.get("longName") or ""

    is_special_symbol = _is_special_market_symbol(ticker)
    if not company_name and not in_local_db and not is_special_symbol:
        return TickerValidationResult(
            valid=False, ticker=ticker, code=ErrorCode.TICKER_NOT_FOUND,
            error=f'Ticker "{ticker}" was not found. Please check the symbol and try again.',
            suggestions=find_similar_tickers(ticker),
        )

    # history call — a network failure here doesn't mean the ticker is bad
    history_empty = False
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        history_empty = hist is None or (hasattr(hist, "empty") and hist.empty)
    except Exception:
        history_empty = False   # service hiccup; don't penalise the ticker

    if history_empty and company_name:
        return TickerValidationResult(
            valid=False, ticker=ticker, company_name=company_name,
            code=ErrorCode.TICKER_DELISTED,
            error=(
                f'"{ticker}" ({company_name}) appears to be delisted '
                "or has no recent trading data."
            ),
        )

    # OTC / pink-sheet advisory
    exchange = (info.get("exchange") or info.get("market") or "").upper()
    warning = ""
    if exchange in _OTC_EXCHANGES:
        warning = f"Note: {ticker} trades on the OTC market. Data may be limited."

    source = "special_symbol" if is_special_symbol and not in_local_db else ("local_db" if in_local_db else "api")
    return TickerValidationResult(
        valid=True, ticker=ticker,
        company_name=company_name or ("(special market symbol)" if is_special_symbol else "(verified offline)"),
        warning=warning,
        source=source,
    )


def validate_ticker_exists(ticker: str) -> TickerValidationResult:
    """Full existence check with graceful-degradation fallback chain.

    Fallback behaviour when services are degraded:
    - API down + ticker in local DB  → valid with warning
    - API down + ticker NOT in DB    → rejection with API_TIMEOUT / API_ERROR
    - DB corrupted/missing + API up  → rely on API only
    - Both services down             → rejection explaining both are unavailable
    """
    fmt = validate_ticker_format(ticker)
    if not fmt.valid:
        return fmt

    normalized = fmt.ticker
    is_special_symbol = _is_special_market_symbol(normalized)

    # Probe local DB (may fail if DB file is corrupted or missing)
    in_local_db = False
    local_db_available = True
    try:
        in_local_db = is_known_ticker(normalized)
    except Exception:
        local_db_available = False
        logger.warning("Local ticker DB unavailable when checking %s", normalized)

    try:
        if in_local_db and not is_special_symbol:
            return TickerValidationResult(
                valid=True,
                ticker=normalized,
                company_name=get_company_name(normalized) or "(verified offline)",
                source="local_db",
            )

        return _cached_api_lookup(normalized)

    except Exception as exc:
        exc_str = str(exc).lower()
        is_timeout = (
            "timeout" in exc_str
            or "timed out" in exc_str
            or isinstance(exc, TimeoutError)
        )
        api_code = ErrorCode.API_TIMEOUT if is_timeout else ErrorCode.API_ERROR
        logger.warning("yfinance lookup failed for %s: %s", normalized, exc)

        # Both services unavailable
        if not local_db_available:
            return TickerValidationResult(
                valid=False, ticker=normalized, code=ErrorCode.API_ERROR,
                error=(
                    "Validation services are temporarily unavailable. "
                    "Please try again shortly."
                ),
            )

        # API down but ticker confirmed in local DB — degrade gracefully
        if in_local_db:
            return TickerValidationResult(
                valid=True, ticker=normalized,
                company_name="(verified offline)",
                source="local_db",
                warning=(
                    "Ticker verified from local database. "
                    "Real-time data verification is temporarily unavailable."
                ),
            )

        # API down for special symbols: allow format-validated pass-through.
        if is_special_symbol:
            return TickerValidationResult(
                valid=True,
                ticker=normalized,
                company_name="(special market symbol)",
                source="special_symbol",
                warning="Special market symbol accepted while live validation is temporarily unavailable.",
            )

        # API down, ticker not in local DB - cannot verify
        return TickerValidationResult(
            valid=False, ticker=normalized, code=api_code,
            error=(
                "Cannot verify this ticker right now. "
                "Please try again in a few minutes or use a well-known ticker."
            ),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_ticker(ticker: str) -> TickerValidationResult:
    """Orchestrate the full validation flow and return a single result."""
    return validate_ticker_exists(ticker)
