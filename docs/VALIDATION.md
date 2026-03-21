# IRIS-AI Ticker Validation System

Developer reference for the multi-layer ticker validation system introduced in the
`feat: add comprehensive error codes, edge case handling, and graceful degradation`
commit series.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Validation Flow](#validation-flow)
3. [Error Codes](#error-codes)
4. [API Reference](#api-reference)
5. [Local Ticker Database](#local-ticker-database)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Testing](#testing)

---

## Architecture Overview

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│  Layer 0 – Input Sanitisation               │
│  ticker_validator.sanitize_ticker_input()   │
│  • Strip $/#/ticker: prefixes               │
│  • Remove trailing "stock"/"etf"/"shares"   │
│  • Collapse internal whitespace             │
│  • Enforce 20-char hard cap                 │
│  • Uppercase                                │
└──────────────────────┬──────────────────────┘
                       │ cleaned string
                       ▼
┌─────────────────────────────────────────────┐
│  Layer 1 – Format Validation  (instant)     │
│  ticker_validator.validate_ticker_format()  │
│  • Regex: ^[A-Z]{1,5}(\.[A-Z]{1,2})?$      │
│  • Rejects crypto tickers (BTC, ETH, …)     │
│  • Rejects reserved words (NULL, TEST, …)   │
│  • No network I/O — always fast             │
└──────────────────────┬──────────────────────┘
                       │ valid format
                       ▼
┌─────────────────────────────────────────────┐
│  Layer 2 – Local SEC Database               │
│  ticker_db.is_known_ticker()                │
│  • In-memory set loaded from               │
│    data/valid_tickers.json                  │
│  • ~13 000 SEC-registered tickers           │
│  • Refreshed every 24 h in the background   │
│  • Thread-safe reads (threading.RLock)      │
└──────────────────────┬──────────────────────┘
                       │ lookup result
                       ▼
┌─────────────────────────────────────────────┐
│  Layer 3 – Live yfinance API  (cached)      │
│  ticker_validator._cached_api_lookup()      │
│  • lru_cache(maxsize=512)                   │
│  • Fetches info + 5-day history             │
│  • Detects OTC / pink-sheet listings        │
│  • Graceful degradation if API is down      │
└──────────────────────┬──────────────────────┘
                       │ TickerValidationResult
                       ▼
┌─────────────────────────────────────────────┐
│  Layer 4 – Data Guardrails                  │
│  data_fetcher.fetch_market_data()           │
│  prompt_builder.build_risk_analysis_prompt()│
│  • Anchors LLM to real price / market-cap   │
│  • Sanity-checks LLM output post-generation │
└─────────────────────────────────────────────┘
```

**Files:**

| File | Role |
|---|---|
| `ticker_validator.py` | Layers 0–3: sanitisation, format check, DB probe, API lookup |
| `ticker_db.py` | Local SEC ticker database: load, refresh, search, similarity |
| `ticker_scheduler.py` | Background 24 h refresh timer |
| `data_fetcher.py` | Layer 4: real market data for LLM grounding |
| `prompt_builder.py` | Layer 4: grounded prompt construction + output sanity check |
| `app.py` | Flask wiring, rate limiter, API endpoints |
| `static/tickerValidation.js` | Client-side Layers 0–1 (mirrors Python, no network) |

---

## Validation Flow

### When a user types a ticker and presses Enter

1. **Client-side (instant)**
   - `sanitizeTicker(raw)` strips prefixes/spaces, uppercases.
   - `validateTickerFormat(cleaned)` runs the regex and crypto/reserved-word checks.
   - If format fails: inline hint shown immediately — no network call.

2. **Server-side `POST /api/validate-ticker`**
   - Rate limit checked (30 req / 60 s per IP).
   - `validate_ticker()` runs all four layers.
   - Response includes `valid`, `ticker`, `company_name`, `warning`, `code`, `suggestions`.

3. **If valid, `GET /api/analyze?ticker=AAPL`**
   - Validation runs again server-side (defence-in-depth).
   - `fetch_market_data(ticker)` gets live price, market cap, P/E, 52-week range.
   - `build_risk_analysis_prompt(ticker, company_name, market_data)` produces a
     data-grounded LLM prompt.
   - `iris_app.run_one_ticker(ticker)` runs the full analysis pipeline.
   - `validate_llm_output(text, market_data)` sanity-checks any pre-built insights.
   - Response includes `market_data` and `grounded_prompt` alongside analysis.

### Graceful degradation fallback chain

```
yfinance OK?
  YES → return API result
  NO  →
        ticker in local DB?
          YES → return valid + warning("verified offline")
          NO  →
                local DB available?
                  YES → return error(API_TIMEOUT / API_ERROR)
                  NO  → return error(API_ERROR, "both services unavailable")
```

---

## Error Codes

All rejection responses carry a `code` field for structured handling.

| Code | HTTP | Meaning | Typical user-facing message |
|---|---|---|---|
| `EMPTY_INPUT` | 200 | Ticker string is empty after sanitisation | "Please enter a stock ticker symbol." |
| `INVALID_FORMAT` | 200 | Doesn't match `^[A-Z]{1,5}(\.[A-Z]{1,2})?$` | "Tickers are 1–5 letters, with an optional class suffix (e.g., BRK.B)." |
| `RESERVED_WORD` | 200 | Crypto ticker or reserved word (NULL, TEST…) | "IRIS-AI analyzes stocks and ETFs. For cryptocurrency analysis, please use a crypto-specific platform." |
| `TICKER_NOT_FOUND` | 200 | Passes format but unknown to both DB and API | "Ticker was not found. Please check the symbol and try again." |
| `TICKER_DELISTED` | 200 | Company found but no recent trading data | "Appears to be delisted or has no recent trading data." |
| `API_TIMEOUT` | 200 | yfinance timed out, ticker not in local DB | "Cannot verify this ticker right now. Please try again." |
| `API_ERROR` | 200 | Network error or both services down | "Validation services are temporarily unavailable." |
| `RATE_LIMITED` | 429 | IP exceeded 30 requests / 60 s | "Too many requests. Please wait before trying again." |
| `DATA_FETCH_FAILED` | 502 | market data fetch failed before LLM call | "Could not retrieve market data. Please try again later." |
| `INTERNAL_ERROR` | 500 | Unhandled exception in analysis pipeline | "An internal error occurred during analysis." |

**Python constant:** `ticker_validator.ErrorCode.FIELD_NAME`
**JavaScript constant:** `TickerValidation.ErrorCodes.FIELD_NAME`

---

## API Reference

### `POST /api/validate-ticker`

Real-time ticker validation. Returns HTTP 200 for both valid and invalid results
(only 429 on rate-limit).

**Request**
```json
{ "ticker": "AAPL" }
```

**Response — valid**
```json
{
  "valid": true,
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "warning": ""
}
```

**Response — invalid**
```json
{
  "valid": false,
  "error": "Ticker \"XYZZY\" was not found. Please check the symbol and try again.",
  "code": "TICKER_NOT_FOUND",
  "suggestions": ["XYZ", "XYZT"]
}
```

**Response — rate limited (HTTP 429)**
```json
{
  "error": "Too many requests. Please wait before trying again.",
  "code": "RATE_LIMITED"
}
```

---

### `GET /api/analyze?ticker=AAPL`

Full analysis endpoint. Runs all validation layers, fetches market data, calls LLM pipeline.

**Query parameters**

| Parameter | Default | Description |
|---|---|---|
| `ticker` | *(required)* | Stock ticker symbol |
| `timeframe` | — | Preset: `1D`, `5D`, `1M`, `6M`, `YTD`, `1Y`, `5Y` |
| `period` | `60d` | yfinance period string (used when `timeframe` is absent) |
| `interval` | `1d` | yfinance interval string |

**Response — success (200)**
```json
{
  "ticker": "AAPL",
  "risk_score": 42,
  "llm_insights": { ... },
  "market_data": {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "current_price": 185.50,
    "market_cap": 2900000000000,
    "pe_ratio": 28.5,
    "52_week_high": 199.62,
    "52_week_low": 124.17
  },
  "grounded_prompt": "Analyze AAPL (Apple Inc.). Current price: $185.5. ..."
}
```

**Response — validation failure (422)**
```json
{
  "valid": false,
  "error": "...",
  "code": "TICKER_NOT_FOUND",
  "suggestions": ["..."]
}
```

---

### `GET /api/tickers/search?q=APP`

Typeahead autocomplete. Returns up to 8 matching tickers from the local DB.

**Response**
```json
[
  { "ticker": "AAPL", "name": "Apple Inc.", "exchange": "Nasdaq" },
  { "ticker": "APP",  "name": "Applovin Corp", "exchange": "Nasdaq" }
]
```

---

### `GET /api/health`

Service health check. Reports ticker DB status, age, and staleness.

**Response (200)**
```json
{
  "status": "healthy",
  "ticker_db_loaded": true,
  "ticker_count": 13247,
  "ticker_db_age_hours": 3.2,
  "ticker_db_stale": false
}
```

---

### `POST /api/admin/refresh-ticker-db`

Trigger a manual ticker database refresh (downloads fresh SEC data).

**Response**
```json
{
  "status": "success",
  "added": 12,
  "removed": 3,
  "total": 13256
}
```

---

## Local Ticker Database

### How it works

The database is a flat JSON array of uppercase ticker symbols downloaded from the
[SEC EDGAR company tickers endpoint](https://www.sec.gov/files/company_tickers.json).
It covers all SEC-registered companies (~13 000 symbols).

On first startup, `run_startup_checks()` detects a missing or severely outdated file and
triggers a download. Subsequent refreshes run in a background daemon thread every 24 hours
via `ticker_scheduler.py`.

### File locations

| File | Purpose |
|---|---|
| `data/valid_tickers.json` | Canonical ticker set (sorted JSON array) |
| `data/valid_tickers.lock` | `filelock` lock file — prevents concurrent writes |

### Thread safety

- **Reads** are protected by `threading.RLock` (`_cache_lock` in `ticker_db.py`).
- **Writes** use a temp file + `os.replace()` atomic rename, so a crash mid-write never
  leaves a corrupt file.
- The `filelock.FileLock` prevents two processes from writing simultaneously (relevant when
  running multiple workers under gunicorn).

### Manually refreshing

```bash
# Via API (running server)
curl -X POST http://localhost:5000/api/admin/refresh-ticker-db

# Via Python
from ticker_db import refresh_ticker_db
result = refresh_ticker_db()
print(result)  # {'added': 5, 'removed': 2, 'total': 13250}
```

### Startup integrity checks (`run_startup_checks`)

| Condition | Action |
|---|---|
| `valid_tickers.json` missing | Synchronous download (blocks startup briefly) |
| File older than 7 days | Background refresh (non-blocking) |
| Fewer than 5 000 tickers loaded | Background re-initialisation |

---

## Configuration

All constants are defined in their respective source files. There are no environment
variables specific to the validation system.

### `ticker_validator.py`

| Constant | Value | Description |
|---|---|---|
| `_MAX_RAW_LENGTH` | `20` | Hard cap on raw input length before sanitisation |
| `lru_cache(maxsize=...)` | `512` | Maximum cached yfinance lookups |
| `_TICKER_RE` | `^[A-Z]{1,5}(\.[A-Z]{1,2})?$` | Valid ticker format regex |

To add a new crypto or reserved word, extend `_CRYPTO_TICKERS` or `_RESERVED_WORDS`
in `ticker_validator.py` and the matching sets in `static/tickerValidation.js`.

### `app.py`

| Constant | Value | Description |
|---|---|---|
| `_RATE_LIMIT_MAX` | `30` | Max requests per IP per window |
| `_RATE_LIMIT_WINDOW` | `60` | Window size in seconds |

### `ticker_scheduler.py`

| Constant | Value | Description |
|---|---|---|
| `_REFRESH_INTERVAL_SECONDS` | `86400` (24 h) | Background DB refresh interval |

### `ticker_db.py`

| Constant | Value | Description |
|---|---|---|
| `_SEC_URL` | SEC EDGAR endpoint | Source for ticker data |
| `_DATA_FILE` | `data/valid_tickers.json` | Local cache path |
| `is_db_stale(threshold_hours=48.0)` | 48 h | Age at which DB is considered stale |

---

## Troubleshooting

### "Validation services are temporarily unavailable"

Both yfinance **and** the local DB failed. This is rare. Check:
- `data/valid_tickers.json` exists and is readable.
- No other process is holding `data/valid_tickers.lock` indefinitely.
- Network connectivity to `sec.gov` and `query1.finance.yahoo.com`.

### Ticker DB not loading on startup

```
startup checks failed: [Errno 13] Permission denied: 'data/valid_tickers.json'
```

Ensure the process user has read/write access to the `data/` directory.

### "BTC is not found" instead of crypto rejection message

The `_CRYPTO_TICKERS` set in `ticker_validator.py` may be out of sync with
`static/tickerValidation.js`. Both sets must be kept identical — add/remove
symbols in both files.

### yfinance API returning empty info for real tickers

yfinance occasionally returns `{}` for valid tickers during outages or rate limiting.
When this happens and the ticker is in the local DB, the system degrades gracefully
and returns `valid: true` with a `warning` field. The frontend renders this as a
yellow advisory rather than an error.

### Rate limit hit during automated testing

The rate limiter is per-IP and in-memory. In tests, clear `app._rate_limit_store`
in `setUp`:

```python
from app import _rate_limit_store
_rate_limit_store.clear()
```

### LRU cache serving stale results in tests

Clear the yfinance lookup cache in `setUp`:

```python
from ticker_validator import _cached_api_lookup
_cached_api_lookup.cache_clear()
```

---

## Testing

### Test files

| File | What it tests | Network required? |
|---|---|---|
| `tests/test_validation_edge_cases.py` | Unit tests: sanitisation, format, error codes, graceful degradation | No — all mocked |
| `tests/test_e2e_validation.py` | End-to-end: full request→validation→response flow via Flask test client | No — all mocked |

### Running the tests

```bash
# All validation tests
python -m unittest tests/test_validation_edge_cases.py tests/test_e2e_validation.py -v

# Edge cases only
python -m unittest tests/test_validation_edge_cases.py -v

# E2E only
python -m unittest tests/test_e2e_validation.py -v
```

### What each E2E test covers

| Test | Scenario |
|---|---|
| `test_e2e_valid_ticker_full_flow` | AAPL passes all layers; market data and grounded prompt appear in response |
| `test_e2e_invalid_ticker_blocked` | XYZZY blocked at Layer 3; LLM never called |
| `test_e2e_format_error_never_hits_backend` | `123!!!` blocked at Layer 1; yfinance never called |
| `test_e2e_suggestion_is_valid` | Typo "AAPPL" returns suggestions; first suggestion itself passes validation |
| `test_e2e_concurrent_requests` | 10 simultaneous requests via `asyncio.gather`; all succeed without race conditions |
| `test_e2e_rate_limiting` | 35 rapid requests; first 30 return 200, next 5 return 429 |

### Writing new tests

Follow these conventions:

1. **Always clear shared state in `setUp`:**
   ```python
   from app import _rate_limit_store
   from ticker_validator import _cached_api_lookup

   def setUp(self):
       _rate_limit_store.clear()
       _cached_api_lookup.cache_clear()
   ```

2. **Mock at the module boundary, not inside the function:**
   ```python
   # Correct — patches what ticker_validator.py imports
   with patch("ticker_validator.yf.Ticker", return_value=mock):

   # Wrong — patches yfinance globally
   with patch("yfinance.Ticker", return_value=mock):
   ```

3. **Always mock `ticker_validator.is_known_ticker`** alongside `yf.Ticker`
   to control which layer the test exercises.

4. **For analyze-endpoint tests, mock `app.iris_app`** to avoid spinning up
   the full IRIS pipeline (slow, requires model files):
   ```python
   mock_iris = MagicMock()
   mock_iris.run_one_ticker.return_value = {"ticker": "AAPL", ...}
   with patch("app.iris_app", mock_iris):
       ...
   ```
