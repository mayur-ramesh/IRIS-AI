from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import traceback
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# Fix for Windows: Disable symlink warnings which can cause the Hugging Face download to hang
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_data_dir():
    preferred_rel = Path("data/demo_guests") if DEMO_MODE else Path("data")
    preferred = PROJECT_ROOT / preferred_rel
    if DEMO_MODE:
        try:
            (PROJECT_ROOT / Path("data/demo_guests")).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback_name = "demo_guests_data" if DEMO_MODE else "runtime_data"
        fallback = PROJECT_ROOT / fallback_name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DATA_DIR = _resolve_data_dir()
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
        unix_seconds = _index_to_unix_seconds(hist.index)

        close_values = np.asarray(close_series, dtype=np.float64)
        valid_mask = np.isfinite(close_values) & np.isfinite(unix_seconds) & (unix_seconds > 0)
        data = [
            {"time": int(ts), "value": float(val)}
            for ts, val in zip(unix_seconds[valid_mask], close_values[valid_mask])
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

    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker parameter is required"}), 400

    ticker = str(ticker).strip().upper()
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
            f"API Request for Analysis: {ticker} | timeframe={timeframe or 'custom'} | "
            f"period={period} interval={interval}"
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
            return jsonify(report)
        else:
             return jsonify({"error": f"Failed to analyze {ticker}. Stock not found or connection error."}), 404
             
    except Exception as e:
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

@app.route('/api/session-summary/latest')
def latest_session_summary():
    """Return the most recent session summary with comparisons."""
    path = SESSIONS_DIR / "latest_session_summary.json"
    if not path.exists():
        return jsonify({"error": "No session summary found yet."}), 404
    return send_file(str(path), mimetype="application/json")

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
