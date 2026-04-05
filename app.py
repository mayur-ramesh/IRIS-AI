from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import traceback
import os
import json
import logging
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from storage_paths import resolve_data_dir

# Fix for Windows: Disable symlink warnings which can cause the Hugging Face download to hang
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = resolve_data_dir(PROJECT_ROOT, DEMO_MODE)
SESSIONS_DIR = DATA_DIR / "sessions"
YF_CACHE_DIR = DATA_DIR / "yfinance_tz_cache"

# Import the IRIS_System from the existing MVP script
try:
    from iris_mvp import (
        IRIS_System,
        RISK_HORIZON_MAP,
        RISK_HORIZON_LABELS,
        derive_investment_signal,
        generate_rf_reasoning,
    )
    iris_app = IRIS_System()
except ImportError as e:
    print(f"Error importing iris_mvp: {e}")
    iris_app = None

app = Flask(__name__)
CORS(app) # Enable CORS for all routes
DATA_DIR.mkdir(parents=True, exist_ok=True)
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _feedback_log_path() -> Path:
    """Return the canonical feedback log path for the current runtime mode."""
    if DEMO_MODE:
        demo_dir = PROJECT_ROOT / "data" / "demo_guests"
        demo_dir.mkdir(parents=True, exist_ok=True)
        return demo_dir / "feedback_logs.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "feedback_logs.json"
try:
    cache_mod = getattr(yf, "cache", None)
    cache_setter = getattr(cache_mod, "set_cache_location", None)
    if callable(cache_setter):
        cache_setter(str(YF_CACHE_DIR))
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(YF_CACHE_DIR))
except Exception:
    pass

try:
    import sqlite3

    probe_path = YF_CACHE_DIR / ".cache_probe.sqlite3"
    conn = sqlite3.connect(str(probe_path))
    conn.execute("CREATE TABLE IF NOT EXISTS _probe (id INTEGER)")
    conn.close()
    try:
        probe_path.unlink()
    except OSError:
        pass
except Exception:
    # Some environments cannot write SQLite files in cache dirs.
    # Disable yfinance SQLite caches to avoid runtime OperationalError.
    try:
        cache_mod = getattr(yf, "cache", None)
        if cache_mod is not None:
            if hasattr(cache_mod, "_CookieCacheManager") and hasattr(cache_mod, "_CookieCacheDummy"):
                cache_mod._CookieCacheManager._Cookie_cache = cache_mod._CookieCacheDummy()
            if hasattr(cache_mod, "_ISINCacheManager") and hasattr(cache_mod, "_ISINCacheDummy"):
                cache_mod._ISINCacheManager._isin_cache = cache_mod._ISINCacheDummy()
            if hasattr(cache_mod, "_TzCacheManager") and hasattr(cache_mod, "_TzCacheDummy"):
                cache_mod._TzCacheManager._tz_cache = cache_mod._TzCacheDummy()
    except Exception:
        pass

TIMEFRAME_TO_YFINANCE = {
    "1D": ("1d", "2m"),
    "5D": ("5d", "15m"),
    "1M": ("1mo", "1h"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
}

SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOG", "NVDA", "META", "CRM", "ADBE", "INTC", "AMD", "AVGO", "ORCL", "CSCO", "IBM", "QCOM", "NOW"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "BKNG", "CMG"],
    "Communication Services": ["GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "SNAP", "PINS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "DVN"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "MDLZ", "GIS"],
    "Industrials": ["CAT", "BA", "HON", "UPS", "RTX", "DE", "LMT", "GE", "MMM", "UNP"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    "Basic Materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DOW", "NUE", "VMC", "MLM"],
}


_yf_info_cache = {}
_YF_INFO_TTL = 300  # seconds
_almanac_data = None
_accuracy_data = None
_accuracy_mtime = 0.0
_iris_snapshot_cache = {"data": None, "ts": 0.0}
_IRIS_SNAPSHOT_TTL = 300  # 5 minutes

_ALMANAC_INDEX_KEY_MAP = {
    "djia": "dow",
    "dow": "dow",
    "dow jones industrial average": "dow",
    "s&p 500": "sp500",
    "sp500": "sp500",
    "s&p500": "sp500",
    "nasdaq": "nasdaq",
}


def _get_cached_yf_info(ticker):
    """Cache yfinance Ticker.info payloads to reduce repeated network calls."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return {}

    now_ts = time.time()
    cached = _yf_info_cache.get(symbol)
    if cached and (now_ts - cached.get("ts", 0)) < _YF_INFO_TTL:
        return cached.get("info") or {}

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}
    _yf_info_cache[symbol] = {"info": info, "ts": now_ts}
    return info


def _almanac_iso_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _almanac_week_range(start_value: str):
    """Return Monday-Friday ISO dates for the requested week anchor."""
    parsed = datetime.strptime(start_value, "%Y-%m-%d").date()
    week_start = parsed - timedelta(days=parsed.weekday())
    week_end = week_start + timedelta(days=4)
    return week_start.isoformat(), week_end.isoformat()


def _nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> date:
    first_day = date(year, month, 1)
    offset = (weekday - first_day.weekday()) % 7
    return first_day + timedelta(days=offset + ((occurrence - 1) * 7))


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    """Return Gregorian Easter Sunday for the requested year."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _market_holiday_map(year: int) -> dict[date, str]:
    easter = _easter_sunday(year)
    return {
        _observed_fixed_holiday(year, 1, 1): "New Year's Day market holiday",
        _nth_weekday_of_month(year, 1, 0, 3): "Martin Luther King Jr. Day market holiday",
        _nth_weekday_of_month(year, 2, 0, 3): "Presidents' Day market holiday",
        easter - timedelta(days=2): "Good Friday market holiday",
        _last_weekday_of_month(year, 5, 0): "Memorial Day market holiday",
        _observed_fixed_holiday(year, 6, 19): "Juneteenth market holiday",
        _observed_fixed_holiday(year, 7, 4): "Independence Day market holiday",
        _nth_weekday_of_month(year, 9, 0, 1): "Labor Day market holiday",
        _nth_weekday_of_month(year, 11, 3, 4): "Thanksgiving Day market holiday",
        _observed_fixed_holiday(year, 12, 25): "Christmas Day market holiday",
    }


def _market_closure_reason(date_key: str) -> str | None:
    target = datetime.strptime(date_key, "%Y-%m-%d").date()
    return _market_holiday_map(target.year).get(target)


def _almanac_weekday_entry(date_key: str, daily: dict[str, dict], data_year: int | None):
    entry = daily.get(date_key)
    if entry:
        return {
            **entry,
            "date": date_key,
            "day": str(entry.get("day", "")).strip().upper()[:3],
            "market_open": True,
            "almanac_available": True,
            "status": "open",
            "status_reason": "",
        }

    closure_reason = _market_closure_reason(date_key)
    parsed = datetime.strptime(date_key, "%Y-%m-%d")
    if closure_reason:
        status = "closed"
        status_reason = closure_reason
        market_open = False
    else:
        status = "no_data"
        market_open = True
        year_note = f" outside the {data_year} dataset" if data_year else ""
        status_reason = f"Market open, but no Almanac entry is available for this date{year_note}."

    return {
        "date": date_key,
        "day": parsed.strftime("%a").upper()[:3],
        "d": None,
        "s": None,
        "n": None,
        "d_dir": "",
        "s_dir": "",
        "n_dir": "",
        "icon": None,
        "notes": "",
        "market_open": market_open,
        "almanac_available": False,
        "status": status,
        "status_reason": status_reason,
    }


def _almanac_table_rows(payload, table_name):
    table = payload.get(table_name, {})
    if isinstance(table, dict):
        rows = table.get("rows", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _almanac_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _almanac_int(value, default=0):
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _normalize_almanac_index(value):
    cleaned = str(value or "").strip().lower().replace(".", "")
    return _ALMANAC_INDEX_KEY_MAP.get(cleaned)


def _normalize_almanac_dump(payload):
    metadata_rows = _almanac_table_rows(payload, "metadata")
    metadata = {str(row.get("key", "")).strip(): row.get("value") for row in metadata_rows}

    month_rows = _almanac_table_rows(payload, "months")
    vital_rows = _almanac_table_rows(payload, "vital_statistics")
    daily_rows = _almanac_table_rows(payload, "daily_probabilities")
    signal_rows = _almanac_table_rows(payload, "seasonal_signals")
    heatmap_rows = _almanac_table_rows(payload, "seasonal_heatmap")

    if not month_rows or not daily_rows:
        return {"error": "Unsupported almanac JSON format"}

    months = {}
    for row in month_rows:
        month_key = str(row.get("month_key", "")).strip()
        if not month_key:
            continue
        months[month_key] = {
            "name": str(row.get("name", "")).strip(),
            "month_num": _almanac_int(row.get("month_num"), 0),
            "overview": str(row.get("overview", "")).strip(),
            "vital_stats": {},
        }

    for row in vital_rows:
        month_key = str(row.get("month_key", "")).strip()
        index_key = str(row.get("index_key", "")).strip() or _normalize_almanac_index(row.get("index_name"))
        if month_key not in months or index_key not in {"dow", "sp500", "nasdaq"}:
            continue
        months[month_key]["vital_stats"][index_key] = {
            "rank": _almanac_int(row.get("rank"), 0),
            "up": _almanac_int(row.get("years_up"), 0),
            "down": _almanac_int(row.get("years_down"), 0),
            "avg_change": _almanac_float(row.get("avg_pct_change"), 0.0),
            "midterm_avg": _almanac_float(row.get("midterm_yr_avg"), 0.0),
        }

    daily = {}
    for row in daily_rows:
        date_key = str(row.get("date", "")).strip()
        if not date_key:
            continue
        daily[date_key] = {
            "date": date_key,
            "source_month": str(row.get("source_month", "")).strip(),
            "day": str(row.get("day_of_week", "")).strip().upper()[:3],
            "d": _almanac_float(row.get("dow_prob"), 0.0),
            "s": _almanac_float(row.get("sp500_prob"), 0.0),
            "n": _almanac_float(row.get("nasdaq_prob"), 0.0),
            "d_dir": str(row.get("dow_dir", "")).strip().upper(),
            "s_dir": str(row.get("sp500_dir", "")).strip().upper(),
            "n_dir": str(row.get("nasdaq_dir", "")).strip().upper(),
            "icon": row.get("icon"),
            "notes": str(row.get("notes", "")).strip(),
        }

    seasonal_signals = []
    for row in signal_rows:
        seasonal_signals.append(
            {
                "id": str(row.get("id", "")).strip(),
                "label": str(row.get("label", row.get("signal", ""))).strip(),
                "type": str(row.get("type", row.get("relevance", ""))).strip(),
                "source_month": str(row.get("source_month", "")).strip(),
                "description": str(row.get("description", row.get("detail", ""))).strip(),
            }
        )

    seasonal_heatmap = {}
    for row in heatmap_rows:
        month_key = str(row.get("month_key", "")).strip()
        if not month_key:
            continue
        seasonal_heatmap[month_key] = {
            "bias": str(row.get("bias", "")).strip(),
            "sp500_rank": _almanac_int(row.get("sp500_rank"), 0),
            "sp500_avg": _almanac_float(row.get("sp500_avg"), 0.0),
            "sp500_midterm": _almanac_float(row.get("sp500_midterm"), 0.0),
            "sp500_midterm_rank": _almanac_int(row.get("sp500_midterm_rank"), 0),
        }

    return {
        "meta": {
            "source": str(
                metadata.get("source")
                or payload.get("_meta", {}).get("source")
                or "Stock Trader's Almanac 2026 (Wiley)"
            ),
            "year": _almanac_int(
                metadata.get("year", payload.get("_meta", {}).get("year")),
                2026,
            ),
            "generated_at": str(
                metadata.get("generated_at")
                or payload.get("_meta", {}).get("generated_at")
                or _almanac_iso_now()
            ),
        },
        "months": months,
        "daily": {date_key: daily[date_key] for date_key in sorted(daily.keys())},
        "seasonal_signals": seasonal_signals,
        "seasonal_heatmap": seasonal_heatmap,
    }


def _normalize_almanac_payload(payload):
    if not isinstance(payload, dict):
        return {"error": "Invalid almanac payload"}

    required_keys = {"meta", "months", "daily", "seasonal_signals", "seasonal_heatmap"}
    if required_keys.issubset(payload.keys()):
        return payload

    if payload.get("_meta") or payload.get("daily_probabilities") or payload.get("vital_statistics"):
        return _normalize_almanac_dump(payload)

    return {"error": "Unsupported almanac JSON format"}


def _load_almanac_data():
    """Load JSON-backed almanac data once for the comparison UI."""
    global _almanac_data
    if _almanac_data is not None:
        return _almanac_data

    almanac_dir = PROJECT_ROOT / "data" / "almanac_2026"
    candidates = [
        ("primary", almanac_dir / "almanac_2026.json"),
        ("structured-db", almanac_dir / "almanac_2026_db_dump.json"),
    ]

    for label, path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_payload = json.load(f)
            _almanac_data = _normalize_almanac_payload(raw_payload)
            if "error" in _almanac_data:
                print(f"[ALMANAC] ERROR: {path.name} could not be normalized ({_almanac_data['error']})")
                return _almanac_data
            print(f"[ALMANAC] Loaded {label} almanac data from {path}")
            return _almanac_data
        except Exception as exc:
            _almanac_data = {"error": f"Failed to load {path.name}: {exc}"}
            print(f"[ALMANAC] ERROR: {_almanac_data['error']}")
            return _almanac_data

    _almanac_data = {"error": "No almanac data found. Run build_almanac_json.py first."}
    print(f"[ALMANAC] ERROR: {_almanac_data['error']}")
    return _almanac_data


def _load_accuracy_data():
    """Load accuracy_results.json with file-mtime caching."""
    global _accuracy_data, _accuracy_mtime
    path = PROJECT_ROOT / "data" / "almanac_2026" / "accuracy_results.json"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _accuracy_data is not None and mtime <= _accuracy_mtime:
        return _accuracy_data
    try:
        with open(path, "r", encoding="utf-8") as f:
            _accuracy_data = json.load(f)
        _accuracy_mtime = mtime
        return _accuracy_data
    except Exception as e:
        print(f"[ACCURACY] Error loading: {e}")
        return None


def _iris_price_threshold(symbol: str) -> float:
    token = str(symbol or "").strip().upper()
    if "DJI" in token:
        return 10000.0
    if "IXIC" in token:
        return 5000.0
    return 400.0


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _iris_direction_from_pct_change(pct_change: float) -> str:
    if pct_change > 0:
        return "upward"
    if pct_change < 0:
        return "downward"
    return "flat"


def _iris_prediction_light(trend_label: str, sentiment_score=0.0) -> str:
    normalized_trend = str(trend_label or "").upper()
    sentiment = _safe_float(sentiment_score, 0.0)
    if sentiment < -0.05 or "STRONG DOWNTREND" in normalized_trend:
        return " RED (Risk Detected - Caution)"
    if abs(sentiment) < 0.05 and "WEAK" in normalized_trend:
        return " YELLOW (Neutral / Noise)"
    return " GREEN (Safe to Proceed)"


def _read_latest_iris_report(symbol: str):
    """Read the latest valid IRIS report for the requested symbol from DATA_DIR."""
    token = str(symbol or "").strip().upper()
    bare = token.lstrip("^_")
    filename_candidates = []
    for candidate in (
        f"{token}_report.json",
        f"^{bare}_report.json",
        f"_{bare}_report.json",
        f"{bare}_report.json",
    ):
        path = DATA_DIR / candidate
        if path not in filename_candidates:
            filename_candidates.append(path)

    min_price = _iris_price_threshold(token)

    for path in filename_candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                reports = json.load(f)
            if not isinstance(reports, list):
                reports = [reports]
            for report in reversed(reports):
                if not isinstance(report, dict):
                    continue
                current_price = _safe_float(report.get("market", {}).get("current_price"), 0.0)
                if current_price < min_price:
                    continue
                horizon_1d = report.get("all_horizons", {}).get("1D", {})
                meta = report.get("meta", {})
                horizon_days = meta.get("horizon_days", 1) if isinstance(meta, dict) else 1
                if not isinstance(horizon_1d, dict) and int(_safe_float(horizon_days, 1)) != 1:
                    continue
                return report
        except Exception:
            continue
    return None


def _format_iris_snapshot_entry(report: dict, label: str):
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    market = report.get("market", {}) if isinstance(report, dict) else {}
    signals = report.get("signals", {}) if isinstance(report, dict) else {}
    h1d = report.get("all_horizons", {}).get("1D", {}) if isinstance(report, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(market, dict):
        market = {}
    if not isinstance(signals, dict):
        signals = {}
    if not isinstance(h1d, dict):
        h1d = {}

    current_price = _safe_float(market.get("current_price"), 0.0)
    predicted_price = (
        market.get("predicted_price_next_session")
        or h1d.get("predicted_price")
        or market.get("predicted_price_horizon")
    )
    predicted_price = _safe_float(predicted_price, 0.0)

    reasoning = h1d.get("iris_reasoning") or signals.get("iris_reasoning") or {}
    if not isinstance(reasoning, dict):
        reasoning = {}
    pct_change = reasoning.get("pct_change")
    if pct_change in (None, "") and current_price:
        pct_change = ((predicted_price - current_price) / current_price) * 100
    pct_change = round(_safe_float(pct_change, 0.0), 2)

    direction = str(reasoning.get("direction", "")).strip().lower()
    if not direction:
        direction = _iris_direction_from_pct_change(pct_change)

    top_factors = reasoning.get("top_factors", [])
    if not isinstance(top_factors, list):
        top_factors = []

    return {
        "available": True,
        "label": label,
        "symbol": str(meta.get("symbol") or meta.get("source_symbol") or "").strip(),
        "session_date": str(meta.get("market_session_date", "")).strip(),
        "generated_at": str(meta.get("generated_at", "")).strip(),
        "current_price": current_price or None,
        "predicted_price": predicted_price or None,
        "trend_label": str(h1d.get("trend_label") or signals.get("trend_label", "")).strip(),
        "investment_signal": str(h1d.get("investment_signal") or signals.get("investment_signal", "")).strip(),
        "check_engine_light": str(signals.get("check_engine_light", "")).strip(),
        "pct_change": pct_change,
        "direction": direction,
        "top_factors": top_factors,
        "model_confidence": h1d.get("model_confidence") or signals.get("model_confidence"),
        "sentiment_score": _safe_float(signals.get("sentiment_score"), 0.0),
        "source": "report_snapshot",
    }


def _get_related_tickers(ticker, count=7):
    """Return a list of related tickers based on the sector of the given ticker."""
    fallback = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA"]
    symbol = str(ticker or "").strip().upper()
    try:
        info = _get_cached_yf_info(symbol)
        sector = info.get("sector", "")
        peers = SECTOR_PEERS.get(sector, fallback)
        related = [s for s in peers if s != symbol]
        return related[:count]
    except Exception:
        return [s for s in fallback if s != symbol][:count]

# ---------------------------------------------------------------------------
# Ticker validation setup
# ---------------------------------------------------------------------------

try:
    from ticker_validator import validate_ticker as _validate_ticker
    from ticker_db import (
        load_ticker_db as _load_ticker_db,
        search_tickers as _search_tickers,
        refresh_ticker_db as _refresh_ticker_db,
        run_startup_checks as _run_startup_checks,
        get_db_file_age_hours as _get_db_file_age_hours,
        is_db_stale as _is_db_stale,
    )
    _VALIDATOR_AVAILABLE = True
except ImportError:
    _VALIDATOR_AVAILABLE = False
    _load_ticker_db = None
    _search_tickers = None
    _refresh_ticker_db = None
    _run_startup_checks = None
    _get_db_file_age_hours = None
    _is_db_stale = None

try:
    from ticker_scheduler import start_scheduler as _start_scheduler
    _SCHEDULER_AVAILABLE = True
except ImportError:
    _SCHEDULER_AVAILABLE = False
    _start_scheduler = None

try:
    from data_fetcher import fetch_market_data as _fetch_market_data
    from prompt_builder import (
        build_risk_analysis_prompt as _build_risk_prompt,
        validate_llm_output as _validate_llm_output,
    )
    _GUARDRAILS_AVAILABLE = True
except ImportError:
    _GUARDRAILS_AVAILABLE = False
    _fetch_market_data = None
    _build_risk_prompt = None
    _validate_llm_output = None

_validation_logger = logging.getLogger("iris.ticker_validation")

# ---------------------------------------------------------------------------
# Startup integrity checks + background scheduler
# ---------------------------------------------------------------------------
if _VALIDATOR_AVAILABLE and _run_startup_checks is not None:
    try:
        _run_startup_checks()
    except Exception as _startup_exc:
        logging.getLogger(__name__).warning("Startup checks failed: %s", _startup_exc)

if _SCHEDULER_AVAILABLE and _start_scheduler is not None:
    try:
        _start_scheduler()
    except Exception as _sched_exc:
        logging.getLogger(__name__).warning("Could not start ticker scheduler: %s", _sched_exc)
# ---------------------------------------------------------------------------

# Simple in-memory rate limiter: {ip: [unix_timestamp, ...]}
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_MAX = 30
_RATE_LIMIT_WINDOW = 60  # seconds

# In-memory cache for /api/llm-predict.
_llm_predict_cache: dict[str, dict] = {}
_LLM_CACHE_TTL = 600  # 10 minutes

# Shared headline cache: {ticker: {headlines: [...], sentiment: float, ts: float}}
_headline_cache: dict[str, dict] = {}
_HEADLINE_CACHE_TTL = 600  # 10 minutes


def _check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed, False if rate limit exceeded."""
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > cutoff]
    if len(_rate_limit_store[ip]) >= _RATE_LIMIT_MAX:
        return False
    _rate_limit_store[ip].append(now)
    return True


def _log_validation(raw_input: str, result) -> None:
    _validation_logger.info(
        "TICKER_VALIDATION | input=%s | valid=%s | source=%s | error=%s",
        raw_input,
        result.valid if result else False,
        result.source if result else "",
        result.error if result else "validator_unavailable",
    )


# ---------------------------------------------------------------------------

def get_latest_llm_reports(symbol: str) -> dict:
    """Read the latest reports for the given symbol from the configured LLM models."""
    llm_dir = PROJECT_ROOT / "data" / "LLM reports"
    models = {
        "chatgpt52": "chatgpt_5.2.json",
        "deepseek_v3": "deepseek_v3.json",
        "gemini_v3_pro": "gemini_v3_pro.json"
    }
    
    insights = {}
    for model_key, filename in models.items():
        filepath = llm_dir / filename
        if not filepath.exists():
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                
                for report in reversed(data):
                    if str(report.get("meta", {}).get("symbol", "")).upper() == symbol.upper():
                        insights[model_key] = report
                        break
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return insights

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


@app.route('/almanac')
def almanac_comparison():
    """Serve the IRIS vs Almanac comparison dashboard."""
    return render_template('almanac_comparison.html')


@app.route('/api/almanac/daily')
def almanac_daily():
    """Return almanac daily scores."""
    data = _load_almanac_data()
    if "error" in data:
        return jsonify(data), 404

    daily = data.get("daily", {})
    date_param = str(request.args.get("date", "") or "").strip()
    from_param = str(request.args.get("from", "") or "").strip()
    to_param = str(request.args.get("to", "") or "").strip()

    if date_param:
        entry = daily.get(date_param)
        if entry is None:
            return jsonify({"error": f"No data for {date_param}"}), 404
        return jsonify(entry)

    if from_param and to_param:
        filtered = {k: v for k, v in daily.items() if from_param <= k <= to_param}
        return jsonify({"from": from_param, "to": to_param, "daily": filtered})

    return jsonify({"daily": daily})


@app.route('/api/almanac/month/<month_key>')
def almanac_month(month_key):
    """Return monthly overview plus daily scores for the selected month."""
    data = _load_almanac_data()
    if "error" in data:
        return jsonify(data), 404

    month = data.get("months", {}).get(month_key)
    if month is None:
        return jsonify({"error": f"No data for month {month_key}"}), 404

    daily = data.get("daily", {})
    month_daily = {k: v for k, v in daily.items() if k.startswith(f"{month_key}-")}
    return jsonify({"month": month, "daily": month_daily})


@app.route('/api/almanac/seasonal')
def almanac_seasonal():
    """Return seasonal heatmap, signals, and month summaries."""
    data = _load_almanac_data()
    if "error" in data:
        return jsonify(data), 404

    return jsonify(
        {
            "heatmap": data.get("seasonal_heatmap", {}),
            "signals": data.get("seasonal_signals", []),
            "months": data.get("months", {}),
        }
    )


@app.route('/api/almanac/week')
def almanac_week():
    """Return the Monday-to-Friday calendar slice for the requested week."""
    data = _load_almanac_data()
    if "error" in data:
        return jsonify(data), 404

    start = str(request.args.get("start", "") or "").strip()
    daily = data.get("daily", {})
    all_dates = sorted(daily.keys())
    if not all_dates:
        return jsonify({"error": "No almanac daily data available"}), 404

    if not start:
        start = all_dates[0]

    try:
        week_start, week_end = _almanac_week_range(start)
    except ValueError:
        return jsonify({"error": "Invalid start date. Expected YYYY-MM-DD"}), 400

    calendar_dates = [
        (datetime.strptime(week_start, "%Y-%m-%d") + timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(5)
    ]
    week_dates = [date_key for date_key in calendar_dates if date_key in daily]
    if not week_dates and not any(_market_closure_reason(date_key) for date_key in calendar_dates):
        return jsonify({"error": f"No weekday entries found for week starting {week_start}"}), 404

    week_data = {date_key: daily[date_key] for date_key in week_dates}
    data_year = data.get("meta", {}).get("year") if isinstance(data.get("meta"), dict) else None
    week_entries = [_almanac_weekday_entry(date_key, daily, data_year) for date_key in calendar_dates]
    first_available = next((entry for entry in week_entries if entry.get("almanac_available")), None)
    month_key = (
        str(first_available.get("source_month", "")).strip()
        if first_available
        else ""
    ) or (str(first_available.get("date", week_start))[:7] if first_available else week_start[:7])
    month_info = data.get("months", {}).get(month_key, {})

    return jsonify(
        {
            "week_start": week_start,
            "week_end": week_end,
            "weekdays": week_entries,
            "daily": week_data,
            "month_overview": month_info,
        }
    )


# --- Almanac Accuracy Tracking API ---

def _accuracy_unavailable_response():
    return jsonify(
        {
            "available": False,
            "message": "Run scripts/seed_accuracy.py to generate accuracy data.",
        }
    )


def _accuracy_pct(hits, total):
    if not total:
        return 0.0
    return round((hits / total) * 100, 1)


@app.route('/api/almanac/accuracy')
def almanac_accuracy():
    """Return almanac historic accuracy results."""
    data = _load_accuracy_data()
    if data is None:
        return _accuracy_unavailable_response()

    daily = data.get("daily", {})
    date_param = str(request.args.get("date", "") or "").strip()
    from_param = str(request.args.get("from", "") or "").strip()
    to_param = str(request.args.get("to", "") or "").strip()

    if date_param:
        entry = daily.get(date_param)
        if entry is None:
            return jsonify({"error": f"No accuracy data for {date_param}"}), 404
        return jsonify(entry)

    if from_param and to_param:
        filtered = {k: v for k, v in daily.items() if from_param <= k <= to_param}
        return jsonify({"from": from_param, "to": to_param, "daily": filtered})

    return jsonify({"daily": daily})


@app.route('/api/almanac/accuracy/week')
def almanac_accuracy_week():
    """Return weekly accuracy results for the requested week."""
    data = _load_accuracy_data()
    if data is None:
        return _accuracy_unavailable_response()

    start = str(request.args.get("start", "") or "").strip()
    if not start:
        return jsonify({"error": "start query parameter is required"}), 400

    try:
        week_start, week_end = _almanac_week_range(start)
    except ValueError:
        return jsonify({"error": "Invalid start date. Expected YYYY-MM-DD"}), 400

    daily = data.get("daily") or {}
    weekly = data.get("weekly") or {}
    week_dates = sorted(date_key for date_key in daily.keys() if week_start <= date_key <= week_end)

    weekly_entry = weekly.get(week_start)

    if weekly_entry is None:
        for date_key in week_dates:
            legacy_week_key = datetime.strptime(date_key, "%Y-%m-%d").strftime("%Y-W%W")
            weekly_entry = weekly.get(legacy_week_key)
            if weekly_entry is not None:
                break

    if weekly_entry is None:
        return jsonify({"error": f"No weekly accuracy found for week starting {week_start}"}), 404

    payload = dict(weekly_entry)
    payload["week_start"] = week_start
    payload["week_end"] = week_end
    return jsonify(payload)


@app.route('/api/almanac/accuracy/month')
def almanac_accuracy_month():
    """Return monthly accuracy results for the requested month."""
    data = _load_accuracy_data()
    if data is None:
        return _accuracy_unavailable_response()

    month_key = str(request.args.get("month", "") or "").strip()
    if not month_key:
        return jsonify({"error": "month query parameter is required"}), 400

    monthly_entry = (data.get("monthly") or {}).get(month_key)
    if monthly_entry is None:
        return jsonify({"error": f"No monthly accuracy found for {month_key}"}), 404
    return jsonify(monthly_entry)


@app.route('/api/almanac/accuracy/summary')
def almanac_accuracy_summary():
    """Return aggregate historic accuracy metrics."""
    data = _load_accuracy_data()
    if data is None:
        return _accuracy_unavailable_response()

    monthly = data.get("monthly") or {}
    daily = data.get("daily") or {}

    overall_hits = sum(int(month.get("hits", 0)) for month in monthly.values())
    overall_total = sum(int(month.get("total_calls", 0)) for month in monthly.values())

    per_index = {}
    for index_key in ("dow", "sp500", "nasdaq"):
        hits = sum(int(month.get(index_key, {}).get("hits", 0)) for month in monthly.values())
        total = sum(int(month.get(index_key, {}).get("total", 0)) for month in monthly.values())
        per_index[index_key] = {
            "hits": hits,
            "total": total,
            "pct": _accuracy_pct(hits, total),
        }

    return jsonify(
        {
            "overall": {
                "hits": overall_hits,
                "total_calls": overall_total,
                "accuracy": _accuracy_pct(overall_hits, overall_total),
            },
            "monthly": monthly,
            "per_index": per_index,
            "last_scored_date": max(daily.keys()) if daily else None,
            "total_days": len(daily),
        }
    )


# --- IRIS Snapshot for Almanac Dashboard ---

@app.route('/api/almanac/iris-snapshot')
def almanac_iris_snapshot():
    """Return the latest cached IRIS index predictions from on-disk report files."""
    now = time.time()
    if (
        _iris_snapshot_cache["data"] is not None
        and (now - _iris_snapshot_cache["ts"]) < _IRIS_SNAPSHOT_TTL
    ):
        return jsonify(_iris_snapshot_cache["data"])

    symbols = {
        "spy": {"file_symbol": "SPY", "label": "SPY (S&P 500 ETF)"},
        "dji": {"file_symbol": "^DJI", "label": "Dow Jones"},
        "gspc": {"file_symbol": "^GSPC", "label": "S&P 500 Index"},
        "ixic": {"file_symbol": "^IXIC", "label": "NASDAQ"},
    }

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "indices": {},
    }

    for key, info in symbols.items():
        report = _read_latest_iris_report(info["file_symbol"])
        if not report:
            result["indices"][key] = {
                "available": False,
                "label": info["label"],
            }
            continue
        result["indices"][key] = _format_iris_snapshot_entry(report, info["label"])

    _iris_snapshot_cache["data"] = result
    _iris_snapshot_cache["ts"] = now
    return jsonify(result)


@app.route('/api/almanac/iris-refresh')
def almanac_iris_refresh():
    """Run lightweight 1D IRIS predictions for the dashboard's major indices."""
    if not iris_app:
        return jsonify({"error": "IRIS not initialized"}), 500

    tickers = {
        "spy": {"ticker": "SPY", "label": "SPY (S&P 500 ETF)"},
        "dji": {"ticker": "^DJI", "label": "Dow Jones"},
        "gspc": {"ticker": "^GSPC", "label": "S&P 500 Index"},
        "ixic": {"ticker": "^IXIC", "label": "NASDAQ"},
    }
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = {"generated_at": generated_at, "indices": {}}

    for key, info in tickers.items():
        ticker = info["ticker"]
        try:
            data = iris_app.get_market_data(ticker)
            if not data:
                result["indices"][key] = {
                    "available": False,
                    "label": info["label"],
                }
                continue

            trend_label, predicted_price, trajectory, traj_upper, traj_lower, rf_model, model_confidence = iris_app.predict_trend(
                data,
                sentiment_score=0.0,
                horizon_days=1,
            )

            current_price = _safe_float(data.get("current_price"), 0.0)
            pct_change = ((predicted_price - current_price) / current_price * 100) if current_price else 0.0
            history_df = data.get("history_df")
            last_rsi = 50.0
            session_date = time.strftime("%Y-%m-%d")
            if history_df is not None and "rsi_14" in history_df.columns and len(history_df):
                last_rsi = float(history_df["rsi_14"].iloc[-1])
            if history_df is not None and len(history_df):
                try:
                    session_date = str(pd.Timestamp(history_df.index[-1]).date())
                except Exception:
                    session_date = time.strftime("%Y-%m-%d")

            investment_signal = derive_investment_signal(pct_change, 0.0, last_rsi, 1)
            reasoning = {}
            if rf_model is not None:
                try:
                    reasoning = generate_rf_reasoning(
                        rf_model,
                        None,
                        current_price,
                        predicted_price,
                        "1 Day",
                    )
                except Exception:
                    reasoning = {}

            result["indices"][key] = {
                "available": True,
                "label": info["label"],
                "symbol": ticker,
                "session_date": session_date,
                "generated_at": generated_at,
                "current_price": round(current_price, 6),
                "predicted_price": round(float(predicted_price), 6),
                "trend_label": trend_label,
                "investment_signal": investment_signal,
                "check_engine_light": _iris_prediction_light(trend_label, 0.0).strip(),
                "pct_change": round(float(pct_change), 2),
                "direction": _iris_direction_from_pct_change(pct_change),
                "top_factors": reasoning.get("top_factors", []) if isinstance(reasoning, dict) else [],
                "model_confidence": round(float(model_confidence), 1),
                "sentiment_score": 0.0,
                "source": "live_rf_prediction",
            }
        except Exception as e:
            result["indices"][key] = {
                "available": False,
                "label": info["label"],
                "error": str(e),
            }

    _iris_snapshot_cache["data"] = None
    _iris_snapshot_cache["ts"] = 0.0
    return jsonify(result)


@app.route('/api/history/<ticker>', methods=['GET'])
def get_history(ticker):
    """Return lightweight market history points directly from yfinance for chart rendering."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return jsonify({"error": "Ticker parameter is required"}), 400

    period = str(request.args.get('period', '1y') or '1y').strip()
    interval = str(request.args.get('interval', '1d') or '1d').strip()

    try:
        def _normalize_download_frame(frame, symbol_token: str):
            if frame is None or frame.empty:
                return frame
            if isinstance(frame.columns, pd.MultiIndex):
                # Single-ticker download typically uses MultiIndex [Price, Ticker].
                try:
                    if symbol_token in frame.columns.get_level_values(-1):
                        frame = frame.xs(symbol_token, axis=1, level=-1, drop_level=True)
                    else:
                        frame.columns = [str(col[0]) for col in frame.columns]
                except Exception:
                    frame.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in frame.columns]
            return frame

        def _fetch_history_with_fallbacks(stock, req_period: str, req_interval: str):
            # yfinance can return empty for some intraday interval/period combinations;
            # progressively widen interval while keeping the requested period.
            normalized_period = str(req_period or "").strip().lower() or "1y"
            normalized_interval = str(req_interval or "").strip().lower() or "1d"
            if normalized_interval == "1h":
                normalized_interval = "60m"

            attempts = [(normalized_period, normalized_interval)]
            if normalized_interval == "2m":
                attempts.extend([(normalized_period, "5m"), (normalized_period, "15m"), (normalized_period, "30m")])
            elif normalized_interval == "15m":
                attempts.extend([(normalized_period, "30m"), (normalized_period, "60m")])
            elif normalized_interval == "60m":
                attempts.append((normalized_period, "1d"))

            tried = set()
            for p, i in attempts:
                key = (p, i)
                if key in tried:
                    continue
                tried.add(key)
                try:
                    frame = stock.history(period=p, interval=i, auto_adjust=False, actions=False)
                    if frame is not None and not frame.empty and "Close" in frame.columns:
                        return frame, i
                except Exception:
                    pass

                # Fallback path: direct download API can succeed when Ticker.history fails.
                try:
                    frame = yf.download(
                        symbol,
                        period=p,
                        interval=i,
                        progress=False,
                        auto_adjust=False,
                        actions=False,
                        threads=False,
                    )
                    frame = _normalize_download_frame(frame, symbol)
                except Exception:
                    continue
                if frame is not None and not frame.empty and "Close" in frame.columns:
                    return frame, i
            return None, normalized_interval

        def _index_to_unix_seconds(index_values):
            if isinstance(index_values, pd.DatetimeIndex):
                dt_index = index_values
            else:
                dt_index = pd.to_datetime(index_values, utc=True, errors="coerce")

            if getattr(dt_index, "tz", None) is None:
                dt_index = dt_index.tz_localize("UTC")
            else:
                dt_index = dt_index.tz_convert("UTC")

            # `asi8` is robust for tz-aware/naive DatetimeIndex and returns ns since epoch.
            raw_ns = np.asarray(dt_index.asi8, dtype=np.int64)
            return np.asarray(raw_ns // 10**9, dtype=np.int64)

        stock = yf.Ticker(symbol)
        hist, resolved_interval = _fetch_history_with_fallbacks(stock, period, interval)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return jsonify({
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "message": "No historical data returned by market data provider for the selected range.",
                "data": [],
            })

        close_series = pd.to_numeric(hist["Close"], errors="coerce")
        open_series = pd.to_numeric(hist["Open"], errors="coerce") if "Open" in hist.columns else close_series
        high_series = pd.to_numeric(hist["High"], errors="coerce") if "High" in hist.columns else close_series
        low_series = pd.to_numeric(hist["Low"], errors="coerce") if "Low" in hist.columns else close_series
        volume_series = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0) if "Volume" in hist.columns else pd.Series(0, index=hist.index)
        unix_seconds = _index_to_unix_seconds(hist.index)

        close_values = np.asarray(close_series, dtype=np.float64)
        open_values = np.asarray(open_series, dtype=np.float64)
        high_values = np.asarray(high_series, dtype=np.float64)
        low_values = np.asarray(low_series, dtype=np.float64)
        volume_values = np.asarray(volume_series, dtype=np.float64)
        
        valid_mask = np.isfinite(close_values) & np.isfinite(open_values) & np.isfinite(high_values) & np.isfinite(low_values) & np.isfinite(unix_seconds) & (unix_seconds > 0)
        data = [
            {
                "time": int(ts), 
                "open": float(o), 
                "high": float(h), 
                "low": float(l), 
                "close": float(c), 
                "value": float(c), 
                "volume": float(vol)
            }
            for ts, o, h, l, c, vol in zip(
                unix_seconds[valid_mask], 
                open_values[valid_mask], 
                high_values[valid_mask], 
                low_values[valid_mask], 
                close_values[valid_mask], 
                volume_values[valid_mask]
            )
        ]
        return jsonify({
            "symbol": symbol,
            "period": period,
            "interval": resolved_interval,
            "data": data,
        })
    except Exception:
        print(f"Error fetching chart history for {symbol}: {traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred while fetching chart history."}), 500

@app.route('/api/related/<ticker>', methods=['GET'])
def get_related(ticker):
    """Return related stock tickers with mini price data for a Recommended for you section."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return jsonify({"error": "Ticker parameter is required"}), 400

    print(f"API Request for Related Tickers: {symbol}")

    def _normalize_related_frame(frame, sym):
        if frame is None or frame.empty:
            return frame
        if isinstance(frame.columns, pd.MultiIndex):
            try:
                if sym in frame.columns.get_level_values(-1):
                    frame = frame.xs(sym, axis=1, level=-1, drop_level=True)
                else:
                    frame.columns = [str(col[0]) for col in frame.columns]
            except Exception:
                frame.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in frame.columns]
        return frame

    try:
        related_symbols = _get_related_tickers(symbol)
        results = []
        for sym in related_symbols:
            try:
                frame = yf.download(
                    sym,
                    period="5d",
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                    actions=False,
                    threads=False,
                )
                frame = _normalize_related_frame(frame, sym)
                if frame is None or frame.empty or "Close" not in frame.columns:
                    continue
                close_series = pd.to_numeric(frame["Close"], errors="coerce")
                closes = [float(x) for x in close_series if np.isfinite(x)]
                if len(closes) < 2:
                    continue
                current_price = closes[-1]
                previous_close = closes[-2]
                price_change = current_price - previous_close
                price_change_pct = (price_change / previous_close * 100) if previous_close else 0.0
                try:
                    name = yf.Ticker(sym).info.get("shortName", sym)
                except Exception:
                    name = sym
                results.append({
                    "symbol": sym,
                    "name": name,
                    "current_price": round(current_price, 2),
                    "price_change": round(price_change, 2),
                    "price_change_pct": round(price_change_pct, 4),
                    "sparkline": closes,
                })
            except Exception:
                continue
        return jsonify({"ticker": symbol, "related": results})
    except Exception:
        print(f"Error in /api/related/{symbol}: {traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred"}), 500


@app.route('/api/analyze', methods=['GET'])
def analyze_ticker():
    """API endpoint to analyze a specific ticker."""
    if not iris_app:
        return jsonify({"error": "IRIS System failed to initialize on the server."}), 500

    raw_ticker = request.args.get('ticker')
    if not raw_ticker:
        return jsonify({"error": "Ticker parameter is required"}), 400

    # --- Validation gate (Layer 1-3) before any LLM / heavy computation -----
    if _VALIDATOR_AVAILABLE:
        val_result = _validate_ticker(str(raw_ticker))
        _log_validation(raw_ticker, val_result)
        if not val_result.valid:
            return jsonify({
                "error": val_result.error,
                "code": val_result.code,
                "suggestions": val_result.suggestions,
                "valid": False,
            }), 422
        ticker = val_result.ticker
        company_name = val_result.company_name  # confirmed context for LLM
    else:
        ticker = str(raw_ticker).strip().upper()
        company_name = ""
    # -------------------------------------------------------------------------

    # --- Data guardrail layer (enabled by default; can be disabled explicitly) ---
    market_data = None
    grounded_prompt = None
    guardrails_enabled = str(request.args.get('guardrails', '1') or '1').strip().lower() not in {"0", "false", "no", "off"}
    if _GUARDRAILS_AVAILABLE and guardrails_enabled:
        market_data = _fetch_market_data(ticker)
        if "error" in market_data:
            return jsonify({
                "error": f"Could not retrieve market data for {ticker}. Please try again later."
            }), 502
        grounded_prompt = _build_risk_prompt(ticker, company_name, market_data)
    # -------------------------------------------------------------------------

    timeframe = str(request.args.get('timeframe', '') or '').strip().upper()
    horizon = str(request.args.get('horizon', '1D') or '1D').strip()

    if timeframe:
        mapped = TIMEFRAME_TO_YFINANCE.get(timeframe)
        if not mapped:
            return jsonify({
                "error": "Invalid timeframe. Supported values: 1D, 5D, 1M, 6M, 1Y, 5Y."
            }), 400
        period, interval = mapped
    else:
        period = str(request.args.get('period', '60d') or '60d').strip()
        interval = str(request.args.get('interval', '1d') or '1d').strip()

    try:
        print(
            f"API Request for Analysis: {ticker} ({company_name or 'unknown'}) | "
            f"timeframe={timeframe or 'custom'} | period={period} interval={interval} | horizon={horizon}"
        )
        # Run the analysis for the single ticker quietly
        report = iris_app.run_one_ticker(
            ticker,
            quiet=True,
            period=period,
            interval=interval,
            include_chart_history=True,
            risk_horizon=horizon,
            fast_mode=True,
            persist_report=False,
            generate_chart_artifact=False,
        )
        
        if report:
            # Cache analyzed headlines for /api/llm-predict to avoid re-running news pipeline.
            evidence_headlines = report.get("evidence", {}).get("headlines_used", [])
            report_sentiment = report.get("signals", {}).get("sentiment_score", 0.0)
            cache_symbol = str(report.get("meta", {}).get("symbol", ticker) or ticker).strip().upper()
            _headline_cache[cache_symbol] = {
                "headlines": evidence_headlines if isinstance(evidence_headlines, list) else [],
                "sentiment": float(report_sentiment or 0.0),
                "ts": time.time(),
            }

            report["llm_insights"] = get_latest_llm_reports(ticker)

            # Attach guardrail data so the frontend renders real numbers
            if market_data is not None:
                report["market_data"] = market_data
            if grounded_prompt is not None:
                report["grounded_prompt"] = grounded_prompt

            # Post-processing sanity check on any pre-built LLM insight text
            if _GUARDRAILS_AVAILABLE and market_data is not None:
                for insight in report["llm_insights"].values():
                    if isinstance(insight, dict):
                        for text_key in ("summary", "analysis", "text", "content"):
                            if isinstance(insight.get(text_key), str):
                                insight[text_key] = _validate_llm_output(
                                    insight[text_key], market_data
                                )

            return jsonify(report)
        else:
             return jsonify({"error": f"Failed to analyze {ticker}. Stock not found or connection error."}), 404
             
    except Exception:
        print(f"Error during analysis: {traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred during analysis.", "code": "INTERNAL_ERROR"}), 500


@app.route('/api/predict', methods=['GET'])
def predict_only():
    """Lightweight prediction endpoint: RF model only, no news re-fetch."""
    if not iris_app:
        return jsonify({"error": "IRIS System not initialized"}), 500

    ticker = str(request.args.get('ticker', '') or '').strip().upper()
    horizon = str(request.args.get('horizon', '1D') or '1D').strip().upper()
    if not ticker:
        return jsonify({"error": "Ticker parameter required"}), 400

    horizon_days = RISK_HORIZON_MAP.get(horizon, 1)
    horizon_label = RISK_HORIZON_LABELS.get(horizon, '1 Day')

    try:
        data = iris_app.get_market_data(ticker)
        if not data:
            return jsonify({"error": f"No market data for {ticker}"}), 404

        trend_label, predicted_price, trajectory, traj_upper, traj_lower, _ = iris_app.predict_trend(
            data,
            sentiment_score=0.0,
            horizon_days=horizon_days,
        )

        current_price = float(data["current_price"])
        pct_change = ((predicted_price - current_price) / current_price * 100) if current_price else 0.0
        last_rsi = 50.0
        history_df = data.get("history_df")
        if history_df is not None and "rsi_14" in history_df.columns and len(history_df):
            last_rsi = float(history_df["rsi_14"].iloc[-1])
        investment_signal = derive_investment_signal(pct_change, 0.0, last_rsi, horizon_days)

        return jsonify({
            "ticker": ticker,
            "horizon": horizon,
            "horizon_days": horizon_days,
            "horizon_label": horizon_label,
            "predicted_price": float(predicted_price),
            "prediction_trajectory": [float(p) for p in trajectory],
            "prediction_trajectory_upper": [float(p) for p in traj_upper],
            "prediction_trajectory_lower": [float(p) for p in traj_lower],
            "trend_label": trend_label,
            "investment_signal": investment_signal,
        })
    except Exception:
        print(f"Error in /api/predict: {traceback.format_exc()}")
        return jsonify({"error": "Prediction failed"}), 500


@app.route('/api/llm-predict', methods=['GET'])
def llm_predict_endpoint():
    """Parallel LLM prediction using cached headlines (no full news pipeline rerun)."""
    ticker = str(request.args.get('ticker', '') or '').strip().upper()
    horizon = str(request.args.get('horizon', '1D') or '1D').strip().upper()
    if not ticker:
        return jsonify({"error": "Ticker parameter required"}), 400
    _start_ts = time.time()
    print(f"[LLM-PREDICT] START ticker={ticker} horizon={horizon}")

    cache_key = f"{ticker}:{horizon}"
    cached = _llm_predict_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < _LLM_CACHE_TTL:
        print(f"[LLM-PREDICT] CACHE HIT ticker={ticker} horizon={horizon}")
        return jsonify(cached["data"])

    try:
        from generate_llm_reports import predict_with_llms, _normalize_llm_result

        horizon_days = RISK_HORIZON_MAP.get(horizon, 1)
        horizon_label = RISK_HORIZON_LABELS.get(horizon, "1 Day")

        # Gather market context. Failures here should not block LLM calls.
        current_price = 0.0
        sma_5 = current_price
        rsi_14 = 50.0
        sentiment_score = 0.0
        headlines_summary = "No recent headlines available."

        try:
            data = iris_app.get_market_data(ticker) if iris_app else None
            if data:
                current_price = float(data.get("current_price", 0.0) or 0.0)
                sma_5 = current_price
                history_df = data.get("history_df")
                if history_df is not None and len(history_df):
                    if "sma_5" in history_df.columns:
                        sma_5 = float(history_df["sma_5"].iloc[-1])
                    if "rsi_14" in history_df.columns:
                        rsi_14 = float(history_df["rsi_14"].iloc[-1])
        except Exception as e:
            print(f"[LLM-PREDICT] Market data failed for {ticker}: {e}")
        print(f"[LLM-PREDICT] Market data: {time.time() - _start_ts:.1f}s")

        # Headlines: use cached /api/analyze headlines; never rerun full analyze_news here.
        hcache = _headline_cache.get(ticker)
        if hcache and (time.time() - float(hcache.get("ts", 0.0))) < _HEADLINE_CACHE_TTL:
            sentiment_score = float(hcache.get("sentiment", 0.0) or 0.0)
            cached_headlines = hcache.get("headlines", [])
            headlines_summary = "; ".join(
                str(h.get("title", ""))[:80]
                for h in (cached_headlines or [])[:7]
                if isinstance(h, dict)
            ) or "No recent headlines available."
        else:
            # Minimal fallback without FinBERT/LLM filtering.
            try:
                stock_news = yf.Ticker(ticker).news or []
                quick_titles = []
                for item in stock_news[:10]:
                    title = ""
                    if isinstance(item, dict):
                        title = item.get("title") or ""
                        if not title:
                            content = item.get("content")
                            if isinstance(content, dict):
                                title = content.get("title", "")
                    if title:
                        quick_titles.append(str(title)[:80])
                headlines_summary = "; ".join(quick_titles[:7]) or "No recent headlines available."
            except Exception:
                pass

        results = predict_with_llms(
            symbol=ticker,
            current_price=current_price,
            sma_5=sma_5,
            rsi_14=rsi_14,
            sentiment_score=sentiment_score,
            horizon=horizon,
            horizon_days=horizon_days,
            horizon_label=horizon_label,
            headlines_summary=headlines_summary,
        )
        print(f"[LLM-PREDICT] LLM calls done: {time.time() - _start_ts:.1f}s")

        for key in results:
            results[key] = _normalize_llm_result(results[key])

        response_data = {
            "ticker": ticker,
            "horizon": horizon,
            "horizon_label": horizon_label,
            "models": results,
        }

        _llm_predict_cache[cache_key] = {"data": response_data, "ts": time.time()}
        print(f"[LLM-PREDICT] DONE ticker={ticker} horizon={horizon} total={time.time() - _start_ts:.1f}s")
        return jsonify(response_data)
    except Exception:
        print(f"[LLM-PREDICT] Unhandled error: {traceback.format_exc()}")
        fallback = {
            "ticker": ticker,
            "horizon": horizon,
            "horizon_label": RISK_HORIZON_LABELS.get(horizon, horizon),
            "models": {
                "chatgpt52": {"error": "Service error", "status": "unavailable"},
                "deepseek_v3": {"error": "Service error", "status": "unavailable"},
                "gemini_v3_pro": {"error": "Service error", "status": "unavailable"},
            },
        }
        return jsonify(fallback)


@app.route('/api/chart')
def get_chart():
    """Serve the generated chart image."""
    path = request.args.get('path')
    if not path:
        return jsonify({"error": "No path provided"}), 400

    requested = Path(str(path))
    full_path = (PROJECT_ROOT / requested).resolve() if not requested.is_absolute() else requested.resolve()
    data_root = DATA_DIR.resolve()
    try:
        full_path.relative_to(data_root)
    except ValueError:
        return jsonify({"error": "Invalid path"}), 403

    if not full_path.exists():
        return jsonify({"error": "Chart not found"}), 404

    return send_file(str(full_path), mimetype='image/png')

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Receive dashboard feedback payloads and append them to a JSON log."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    feedback_item = dict(payload)
    feedback_item["timestamp"] = datetime.now(timezone.utc).isoformat()
    log_path = _feedback_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            logs = loaded if isinstance(loaded, list) else []
    except FileNotFoundError:
        logs = []
    except (OSError, json.JSONDecodeError):
        logs = []

    logs.append(feedback_item)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except OSError:
        return jsonify({"error": "Failed to write feedback log"}), 500

    try:
        saved_to = str(log_path.relative_to(PROJECT_ROOT))
    except ValueError:
        saved_to = str(log_path)

    return jsonify({
        "status": "success",
        "message": "Feedback logged",
        "saved_to": saved_to,
        "demo_mode": DEMO_MODE,
    })


@app.route('/api/feedback/status', methods=['GET'])
def feedback_status():
    """Expose current feedback storage wiring for runtime debugging."""
    log_path = _feedback_log_path()
    exists = log_path.exists()
    count = 0
    last_timestamp = None
    if exists:
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                count = len(loaded)
                if loaded and isinstance(loaded[-1], dict):
                    last_timestamp = loaded[-1].get("timestamp")
        except (OSError, json.JSONDecodeError):
            pass

    return jsonify({
        "demo_mode": DEMO_MODE,
        "cwd": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "iris_initialized": iris_app is not None,
        "finbert_status": getattr(iris_app, "finbert_status", None) if iris_app is not None else None,
        "feedback_log_path": str(log_path),
        "feedback_log_exists": exists,
        "feedback_log_entries": count,
        "last_timestamp": last_timestamp,
    })


@app.route('/api/admin/feedback')
def admin_feedback_logs():
    """Download the current feedback log file from the runtime container."""
    relative_log_path = (
        Path("data/demo_guests/feedback_logs.json")
        if DEMO_MODE
        else Path("data/feedback_logs.json")
    )
    log_path = PROJECT_ROOT / relative_log_path
    if log_path.exists():
        return send_file(str(log_path), mimetype='application/json')
    return jsonify({
        "status": "empty",
        "message": "No feedback logs have been generated yet.",
    })

@app.route('/api/session-summary/latest')
def latest_session_summary():
    """Return the most recent session summary with comparisons."""
    path = SESSIONS_DIR / "latest_session_summary.json"
    if not path.exists():
        return jsonify({"error": "No session summary found yet."}), 404
    return send_file(str(path), mimetype="application/json")

@app.route('/api/tickers/search', methods=['GET'])
def search_tickers_endpoint():
    """Prefix search over the local ticker database for autocomplete."""
    q = str(request.args.get('q', '') or '').strip()
    if not q:
        return jsonify({"results": []}), 200
    try:
        limit = max(1, min(int(request.args.get('limit', 8)), 50))
    except (ValueError, TypeError):
        limit = 8
    if _VALIDATOR_AVAILABLE and _search_tickers is not None:
        try:
            results = _search_tickers(q, limit)
        except Exception:
            results = []
    else:
        results = []
    return jsonify({"results": results}), 200


@app.route('/api/validate-ticker', methods=['POST'])
def validate_ticker_endpoint():
    """Real-time ticker validation for the frontend (always returns HTTP 200)."""
    ip = request.remote_addr or "unknown"
    if not _check_rate_limit(ip):
        return jsonify({"error": "Too many requests. Please wait before trying again.", "code": "RATE_LIMITED"}), 429

    body = request.get_json(silent=True) or {}
    raw = body.get("ticker", "")

    if not _VALIDATOR_AVAILABLE:
        return jsonify({"valid": True, "ticker": str(raw).strip().upper(),
                        "company_name": ""}), 200

    result = _validate_ticker(str(raw))
    _log_validation(raw, result)

    if result.valid:
        return jsonify({
            "valid": True,
            "ticker": result.ticker,
            "company_name": result.company_name,
            "warning": result.warning,
        }), 200
    return jsonify({
        "valid": False,
        "error": result.error,
        "code": result.code,
        "suggestions": result.suggestions,
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Report service health and ticker database status."""
    ticker_db_loaded = False
    ticker_count = 0
    ticker_db_age_hours = None
    ticker_db_stale = False

    if _VALIDATOR_AVAILABLE and _load_ticker_db is not None:
        try:
            db = _load_ticker_db()
            ticker_db_loaded = True
            ticker_count = len(db)
        except Exception:
            pass

    if _get_db_file_age_hours is not None:
        try:
            ticker_db_age_hours = _get_db_file_age_hours()
            ticker_db_age_hours = round(ticker_db_age_hours, 2) if ticker_db_age_hours is not None else None
        except Exception:
            pass

    if _is_db_stale is not None:
        try:
            ticker_db_stale = _is_db_stale(threshold_hours=48.0)
        except Exception:
            pass

    return jsonify({
        "status": "ok",
        "ticker_db_loaded": ticker_db_loaded,
        "ticker_count": ticker_count,
        "ticker_db_age_hours": ticker_db_age_hours,
        "ticker_db_stale": ticker_db_stale,
    }), 200


@app.route('/api/admin/refresh-ticker-db', methods=['POST'])
def refresh_ticker_db_endpoint():
    """Manually trigger a ticker database refresh from the SEC source."""
    if not _VALIDATOR_AVAILABLE or _refresh_ticker_db is None:
        return jsonify({"error": "Ticker database module not available."}), 503
    try:
        result = _refresh_ticker_db()
        status_code = 200 if result.get("status") == "ok" else 502
        return jsonify(result), status_code
    except Exception as exc:
        logging.getLogger(__name__).error("Manual ticker DB refresh failed: %s", exc)
        return jsonify({"status": "error", "error": "Refresh failed unexpectedly."}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
