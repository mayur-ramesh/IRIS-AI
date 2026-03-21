from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import traceback
import os
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
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
    from iris_mvp import IRIS_System
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
    "YTD": ("ytd", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
}

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
                "suggestions": val_result.suggestions,
                "valid": False,
            }), 422
        ticker = val_result.ticker
        company_name = val_result.company_name  # confirmed context for LLM
    else:
        ticker = str(raw_ticker).strip().upper()
        company_name = ""
    # -------------------------------------------------------------------------

    # --- Data guardrail layer: fetch real market data before any LLM call ---
    market_data = None
    grounded_prompt = None
    if _GUARDRAILS_AVAILABLE:
        market_data = _fetch_market_data(ticker)
        if "error" in market_data:
            return jsonify({
                "error": f"Could not retrieve market data for {ticker}. Please try again later."
            }), 502
        grounded_prompt = _build_risk_prompt(ticker, company_name, market_data)
    # -------------------------------------------------------------------------

    timeframe = str(request.args.get('timeframe', '') or '').strip().upper()

    if timeframe:
        mapped = TIMEFRAME_TO_YFINANCE.get(timeframe)
        if not mapped:
            return jsonify({
                "error": "Invalid timeframe. Supported values: 1D, 5D, 1M, 6M, YTD, 1Y, 5Y."
            }), 400
        period, interval = mapped
    else:
        period = str(request.args.get('period', '60d') or '60d').strip()
        interval = str(request.args.get('interval', '1d') or '1d').strip()

    try:
        print(
            f"API Request for Analysis: {ticker} ({company_name or 'unknown'}) | "
            f"timeframe={timeframe or 'custom'} | period={period} interval={interval}"
        )
        # Run the analysis for the single ticker quietly
        report = iris_app.run_one_ticker(
            ticker,
            quiet=True,
            period=period,
            interval=interval,
            include_chart_history=True,
        )
        
        if report:
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
        return jsonify({"error": "An internal error occurred during analysis."}), 500

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
        return jsonify({"error": "Too many requests. Please wait before trying again."}), 429

    body = request.get_json(silent=True) or {}
    raw = body.get("ticker", "")

    if not _VALIDATOR_AVAILABLE:
        return jsonify({"valid": True, "ticker": str(raw).strip().upper(),
                        "company_name": ""}), 200

    result = _validate_ticker(str(raw))
    _log_validation(raw, result)

    if result.valid:
        return jsonify({"valid": True, "ticker": result.ticker,
                        "company_name": result.company_name}), 200
    return jsonify({"valid": False, "error": result.error,
                    "suggestions": result.suggestions}), 200


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
