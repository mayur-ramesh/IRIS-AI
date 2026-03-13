import argparse
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
import yfinance as yf
from newsapi import NewsApiClient
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import time
import json
import re
from pathlib import Path
from storage_paths import resolve_data_dir

# Optional: charting (graceful fallback if matplotlib missing)
try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.dates as mdates  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Load .env if available (for NEWS_API_KEY, IRIS_TICKERS)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass
NEWS_API_KEY = os.environ.get("NEWS_API_KEY") or None
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = resolve_data_dir(PROJECT_ROOT, DEMO_MODE)
SESSIONS_DIR = DATA_DIR / "sessions"
CHARTS_DIR = DATA_DIR / "charts"
YF_CACHE_DIR = DATA_DIR / "yfinance_tz_cache"
TICKER_ALIASES = {
    "GOOGL": "GOOG",
}
COMPANY_NAME_TO_TICKERS = {
    "GOOGLE": ["GOOG", "GOOGL"],
    "ALPHABET": ["GOOG", "GOOGL"],
    "ALPHABETINC": ["GOOG", "GOOGL"],
    "ALPHABETCLASSA": ["GOOGL"],
    "ALPHABETCLASSC": ["GOOG"],
    "APPLE": ["AAPL"],
    "MICROSOFT": ["MSFT"],
    "AMAZON": ["AMZN"],
    "TESLA": ["TSLA"],
    "NVIDIA": ["NVDA"],
    "META": ["META"],
    "NIKE": ["NKE"],
}


def normalize_ticker_symbol(symbol: str):
    token = str(symbol or "").strip().upper()
    if not token:
        return token
    return TICKER_ALIASES.get(token, token)


def sanitize_company_token(value: str):
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def normalize_ticker_list(symbols):
    seen = set()
    normalized = []
    for symbol in symbols or []:
        token = normalize_ticker_symbol(symbol)
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


DEFAULT_TICKERS = normalize_ticker_list(
    os.environ.get("IRIS_TICKERS", "AAPL,MSFT,GOOG,AMZN,NVDA,META,TSLA").split(",")
)



class IRIS_System:
    def __init__(self):
        print("\n  Initializing IRIS Risk Engines...")
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
            # Fall back to in-memory/dummy cache objects when SQLite cache is not writable.
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
        
        # Setup Sentiment Brain (FinBERT - financial sentiment model)
        print("   -> Loading FinBERT AI Model (This may take a moment on first run)...")
        try:
            finbert_model_id = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(finbert_model_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                finbert_model_id,
                use_safetensors=False,
            )
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
            )
            print("   -> FinBERT Loaded Successfully!")
        except Exception as e:
            print(f"   -> Error loading FinBERT: {e}")
            self.sentiment_analyzer = None
        
        #  Setup News Connection
        self.news_api = None
        if NEWS_API_KEY:
            self.news_api = NewsApiClient(api_key=NEWS_API_KEY)
            print("   -> NewsAPI Connection: ESTABLISHED")
        else:
            print("   -> NewsAPI Connection: SIMULATION MODE (No Key Found)")
        self.merge_alias_reports()

    def _read_report_file(self, path: Path):
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, IOError):
            return []

    def resolve_user_ticker_input(self, raw_input: str, interactive_prompt: bool = False, quiet: bool = False):
        token = str(raw_input or "").strip().upper()
        if not token:
            return ""

        company_key = sanitize_company_token(token)
        candidates = COMPANY_NAME_TO_TICKERS.get(company_key, [])
        if not candidates:
            # Exact ticker input or direct alias.
            if token.isalpha() and 1 <= len(token) <= 6:
                return token
            return token

        if len(candidates) == 1:
            corrected = candidates[0]
            if not quiet:
                print(f"  Auto-correct: '{raw_input}' -> '{corrected}'")
            return corrected

        # Ambiguous company input (e.g., GOOGLE -> GOOG/GOOGL)
        if interactive_prompt:
            if not quiet:
                print(f"  Input '{raw_input}' is ambiguous.")
                print("  Did you refer to GOOG or GOOGL?")
            choice = input("Choose ticker [GOOG/GOOGL] (default GOOG): ").strip().upper()
            if choice in candidates:
                return choice
        if not quiet:
            print(f"  Auto-correct: '{raw_input}' -> '{candidates[0]}'")
        return candidates[0]

    def _report_generated_at(self, report: dict):
        if not isinstance(report, dict):
            return ""
        meta = report.get("meta", {})
        if not isinstance(meta, dict):
            return ""
        return str(meta.get("generated_at", "")).strip()

    def _report_sort_key(self, report: dict):
        session_date = self._extract_session_date(report) or ""
        generated_at = self._report_generated_at(report)
        return (session_date, generated_at)

    def merge_alias_reports(self):
        """Merge alias symbols into canonical report files (e.g. GOOGL -> GOOG) without dropping history."""
        data_dir = DATA_DIR
        if not data_dir.exists():
            return None

        for alias, canonical in TICKER_ALIASES.items():
            alias_path = data_dir / f"{alias}_report.json"
            canonical_path = data_dir / f"{canonical}_report.json"
            alias_reports = self._read_report_file(alias_path)
            if not alias_reports:
                continue
            canonical_reports = self._read_report_file(canonical_path)

            merged_reports = []
            for report in canonical_reports + alias_reports:
                if not isinstance(report, dict):
                    continue
                meta = report.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                    report["meta"] = meta
                meta["symbol"] = canonical
                merged_reports.append(report)

            merged_reports = sorted(merged_reports, key=self._report_sort_key)
            if merged_reports:
                with open(canonical_path, "w", encoding="utf-8") as f:
                    json.dump(merged_reports, f, indent=2)
            try:
                alias_path.unlink()
            except OSError:
                pass

    def _extract_session_date(self, report: dict):
        """Return market session date (YYYY-MM-DD) from a report, with legacy fallback."""
        if not isinstance(report, dict):
            return None
        meta = report.get("meta", {})
        if not isinstance(meta, dict):
            return None
        session_date = str(meta.get("market_session_date", "")).strip()
        if len(session_date) == 10:
            return session_date
        generated_at = str(meta.get("generated_at", "")).strip()
        if len(generated_at) >= 10:
            return generated_at[:10]
        return None

    def save_report(self, report: dict, filename: str):
        base_data_dir = DATA_DIR
        base_data_dir.mkdir(parents=True, exist_ok=True)
        canonical_filename = filename
        if filename.endswith("_report.json"):
            symbol = filename[:-12]
            canonical_filename = f"{normalize_ticker_symbol(symbol)}_report.json"
        out_path = base_data_dir / canonical_filename
        
        # Load existing reports if file exists
        reports = []
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Handle both array format and legacy single object format
                    reports = data if isinstance(data, list) else [data]
            except (json.JSONDecodeError, IOError):
                reports = []

        # Always append so every run remains accumulative.
        reports.append(report)
        
        # Save all reports
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        return str(out_path)

    def _load_symbol_reports(self, symbol: str):
        canonical_symbol = normalize_ticker_symbol(symbol)
        out_path = DATA_DIR / f"{canonical_symbol}_report.json"
        if not out_path.exists():
            return []
        return self._read_report_file(out_path)

    def _find_previous_session_report(self, symbol: str, current_session_date: str):
        reports = self._load_symbol_reports(symbol)
        if not reports:
            return None

        candidates = []
        for report in reports:
            session_date = self._extract_session_date(report)
            if session_date and session_date < current_session_date:
                candidates.append((session_date, report))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

        # Fallback for legacy/missing dates.
        for report in reversed(reports):
            session_date = self._extract_session_date(report)
            if session_date != current_session_date:
                return report
        return None

    def _build_comparison(self, current_report: dict, previous_report: dict):
        symbol = current_report.get("meta", {}).get("symbol", "")
        current_session_date = self._extract_session_date(current_report)
        if not previous_report:
            return {
                "symbol": symbol,
                "current_session_date": current_session_date,
                "previous_session_date": None,
                "has_previous_session": False,
                "changes": {},
            }

        previous_session_date = self._extract_session_date(previous_report)
        current_market = current_report.get("market", {})
        previous_market = previous_report.get("market", {})
        current_signals = current_report.get("signals", {})
        previous_signals = previous_report.get("signals", {})

        current_price = float(current_market.get("current_price", 0.0))
        previous_price = float(previous_market.get("current_price", 0.0))
        current_pred = float(current_market.get("predicted_price_next_session", 0.0))
        previous_pred = float(previous_market.get("predicted_price_next_session", 0.0))
        current_sent = float(current_signals.get("sentiment_score", 0.0))
        previous_sent = float(previous_signals.get("sentiment_score", 0.0))

        return {
            "symbol": symbol,
            "current_session_date": current_session_date,
            "previous_session_date": previous_session_date,
            "has_previous_session": True,
            "changes": {
                "current_price_delta": current_price - previous_price,
                "predicted_price_delta": current_pred - previous_pred,
                "sentiment_score_delta": current_sent - previous_sent,
                "trend_label_changed": current_signals.get("trend_label") != previous_signals.get("trend_label"),
                "check_engine_light_changed": current_signals.get("check_engine_light") != previous_signals.get("check_engine_light"),
            },
            "current_snapshot": {
                "current_price": current_price,
                "predicted_price_next_session": current_pred,
                "trend_label": current_signals.get("trend_label"),
                "sentiment_score": current_sent,
                "check_engine_light": current_signals.get("check_engine_light"),
            },
            "previous_snapshot": {
                "current_price": previous_price,
                "predicted_price_next_session": previous_pred,
                "trend_label": previous_signals.get("trend_label"),
                "sentiment_score": previous_sent,
                "check_engine_light": previous_signals.get("check_engine_light"),
            },
        }

    def save_session_summary(self, reports: list):
        """Save per-session aggregate with comparisons against previous session."""
        if not reports:
            return None

        session_dates = [self._extract_session_date(r) for r in reports if self._extract_session_date(r)]
        session_date = max(session_dates) if session_dates else time.strftime("%Y-%m-%d")
        generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Use one latest report per symbol for summary stability.
        latest_by_symbol = {}
        for report in reports:
            symbol = str(report.get("meta", {}).get("symbol", "")).upper()
            if symbol:
                latest_by_symbol[symbol] = report

        ordered_symbols = sorted(latest_by_symbol.keys())
        comparisons = []
        for symbol in ordered_symbols:
            current = latest_by_symbol[symbol]
            current_session = self._extract_session_date(current) or session_date
            previous = self._find_previous_session_report(symbol, current_session)
            comparisons.append(self._build_comparison(current, previous))

        payload = {
            "meta": {
                "generated_at": generated_at,
                "session_date": session_date,
                "symbols": ordered_symbols,
                "report_count": len(ordered_symbols),
            },
            "reports": [latest_by_symbol[s] for s in ordered_symbols],
            "comparisons": comparisons,
        }

        base_sessions_dir = SESSIONS_DIR
        base_sessions_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir = base_sessions_dir / session_date
        sessions_dir.mkdir(parents=True, exist_ok=True)
        summary_path = sessions_dir / "session_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        latest_path = base_sessions_dir / "latest_session_summary.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return str(summary_path)

    def _simulated_market_data(self, ticker: str, price_hint=None):
        """Build deterministic synthetic market data so demo flows still work offline."""
        symbol = str(ticker or "").strip().upper() or "DEMO"
        seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(symbol))
        rng = np.random.default_rng(seed)
        try:
            base_price = float(price_hint)
            if not np.isfinite(base_price) or base_price <= 0:
                raise ValueError
        except (TypeError, ValueError):
            base_price = float(50 + (seed % 350))

        points = 60
        trend = np.linspace(-0.03, 0.03, points)
        noise = rng.normal(0.0, 0.01, points)
        close_values = np.maximum(1.0, base_price * (1.0 + trend + noise))
        close_values[-1] = max(1.0, float(base_price))
        open_values = close_values * (1.0 - rng.normal(0.0, 0.005, points))
        high_values = np.maximum(close_values, open_values) * (1.0 + np.abs(rng.normal(0.0, 0.005, points)))
        low_values = np.minimum(close_values, open_values) * (1.0 - np.abs(rng.normal(0.0, 0.005, points)))

        date_index = pd.date_range(end=pd.Timestamp.utcnow().floor("D"), periods=points, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": open_values.astype(float),
                "High": high_values.astype(float),
                "Low": low_values.astype(float),
                "Close": close_values.astype(float),
                "Volume": rng.integers(700000, 3500000, size=points).astype(float),
            },
            index=date_index,
        )
        df["returns_1d"] = df["Close"].pct_change().fillna(0.0)
        df["sma_5"] = df["Close"].rolling(5, min_periods=1).mean()
        df["sma_10"] = df["Close"].rolling(10, min_periods=1).mean()
        df["sma_20"] = df["Close"].rolling(20, min_periods=1).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(avg_loss != 0.0, 100.0)
        rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
        df["rsi_14"] = rsi.fillna(50.0).clip(0.0, 100.0)

        unix_seconds = np.asarray(date_index.asi8 // 10**9, dtype=np.int64)
        history_points = [
            {"time": int(ts), "open": float(o), "high": float(h), "low": float(l), "close": float(c), "value": float(c)}
            for ts, o, h, l, c in zip(unix_seconds, open_values, high_values, low_values, close_values)
            if ts > 0 and np.isfinite(c)
        ]
        return {
            "current_price": float(close_values[-1]),
            "history": close_values.astype(float),
            "history_df": df,
            "history_points": history_points,
            "symbol": symbol,
        }

    def get_market_data(self, ticker, period="60d", interval="1d"):
        """Fetches market history/features and chart points using the requested yfinance period/interval."""
        try:
            stock = yf.Ticker(ticker)
            price = getattr(stock.fast_info, "last_price", None) or getattr(stock.fast_info, "previous_close", None)
            if price is None:
                return self._simulated_market_data(ticker)
            hist = stock.history(period=period, interval=interval)
            if hist is None or hist.empty:
                return self._simulated_market_data(ticker, price_hint=price)

            def _infer_unix_seconds_from_index(index_values):
                raw_index = pd.Index(index_values)

                def _convert_numeric_to_seconds(numeric_values):
                    numeric_values = np.asarray(numeric_values, dtype=np.float64)
                    if numeric_values.size == 0:
                        return np.array([], dtype=np.int64)
                    abs_median = float(np.nanmedian(np.abs(numeric_values)))
                    if not np.isfinite(abs_median):
                        return np.array([], dtype=np.int64)
                    if abs_median >= 1e17:
                        secs = numeric_values / 1e9      # nanoseconds
                    elif abs_median >= 1e14:
                        secs = numeric_values / 1e6      # microseconds
                    elif abs_median >= 1e11:
                        secs = numeric_values / 1e3      # milliseconds
                    else:
                        secs = numeric_values            # seconds
                    return np.asarray(np.rint(secs), dtype=np.int64)

                if isinstance(raw_index, pd.DatetimeIndex):
                    dt_index = raw_index.tz_localize("UTC") if raw_index.tz is None else raw_index.tz_convert("UTC")
                    raw_int = np.asarray(dt_index.asi8, dtype=np.int64)
                    # Handle malformed datetime indexes that were created with wrong epoch units.
                    abs_median = float(np.nanmedian(np.abs(raw_int))) if raw_int.size else 0.0
                    if 1e8 <= abs_median < 1e14:
                        unix_seconds = _convert_numeric_to_seconds(raw_int)
                    else:
                        unix_seconds = np.asarray(raw_int // 10**9, dtype=np.int64)
                else:
                    numeric_index = pd.to_numeric(raw_index, errors="coerce")
                    numeric_valid = np.isfinite(numeric_index)
                    if np.any(numeric_valid):
                        unix_seconds = np.full(len(raw_index), -1, dtype=np.int64)
                        unix_seconds[numeric_valid] = _convert_numeric_to_seconds(np.asarray(numeric_index[numeric_valid], dtype=np.float64))
                    else:
                        dt_index = pd.to_datetime(raw_index.astype(str), utc=True, errors="coerce")
                        unix_seconds = np.asarray(dt_index.asi8 // 10**9, dtype=np.int64)
                return unix_seconds

            close_series = pd.to_numeric(hist.get("Close"), errors="coerce") if "Close" in hist.columns else None
            if close_series is None:
                return self._simulated_market_data(ticker, price_hint=price)

            open_series = pd.to_numeric(hist.get("Open", close_series), errors="coerce")
            high_series = pd.to_numeric(hist.get("High", close_series), errors="coerce")
            low_series = pd.to_numeric(hist.get("Low", close_series), errors="coerce")
            
            close_values = np.asarray(close_series, dtype=np.float64)
            open_values = np.asarray(open_series, dtype=np.float64)
            high_values = np.asarray(high_series, dtype=np.float64)
            low_values = np.asarray(low_series, dtype=np.float64)
            
            unix_seconds_all = _infer_unix_seconds_from_index(hist.index)
            valid_chart_mask = np.isfinite(close_values) & np.isfinite(unix_seconds_all) & (unix_seconds_all >= 1e8)
            
            history_points = [
                {"time": int(ts), "open": float(o), "high": float(h), "low": float(l), "close": float(c), "value": float(c)}
                for ts, o, h, l, c in zip(
                    unix_seconds_all[valid_chart_mask], 
                    open_values[valid_chart_mask], 
                    high_values[valid_chart_mask], 
                    low_values[valid_chart_mask], 
                    close_values[valid_chart_mask]
                )
            ]
            history_values = close_values[valid_chart_mask]
            # Build feature-rich DataFrame for better model accuracy
            df = hist[["Close"]].copy()
            if "Volume" in hist.columns:
                df["Volume"] = hist["Volume"]
            else:
                df["Volume"] = 0
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
            df["returns_1d"] = df["Close"].pct_change()
            df["sma_5"] = df["Close"].rolling(5, min_periods=1).mean()
            df["sma_10"] = df["Close"].rolling(10, min_periods=1).mean()
            df["sma_20"] = df["Close"].rolling(20, min_periods=1).mean()

            # Standard RSI(14) using Wilder-style exponential averaging.
            delta = df["Close"].diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi = rsi.where(avg_loss != 0.0, 100.0)
            rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
            df["rsi_14"] = rsi

            # Handle NaN/inf values for model stability.
            df = df.replace([np.inf, -np.inf], np.nan)
            df["returns_1d"] = df["returns_1d"].fillna(0.0)
            df["rsi_14"] = df["rsi_14"].fillna(50.0).clip(0.0, 100.0)
            df["Volume"] = df["Volume"].fillna(0.0).clip(lower=0.0)
            df = df.dropna(subset=["Close", "sma_5", "sma_10", "sma_20", "returns_1d", "rsi_14", "Volume"])

            return {
                "current_price": float(price),
                "history": history_values if len(history_values) else None,
                "history_df": df if not df.empty else None,
                "history_points": history_points,
                "symbol": ticker.upper(),
            }
        except Exception:
            return self._simulated_market_data(ticker)

    def analyze_news(self, ticker):
        """Fetches headlines and calculates a Sentiment Score (-1.0 to +1.0)."""
        ticker_symbol = normalize_ticker_symbol(ticker).upper()
        headlines = []
        seen = set()

        # Build strict relevance terms: ticker + known company names mapped to this ticker.
        relevance_terms = {ticker_symbol}
        for company_name, tickers in COMPANY_NAME_TO_TICKERS.items():
            normalized = normalize_ticker_list(tickers)
            if ticker_symbol in normalized:
                relevance_terms.add(str(company_name or "").upper())

        def add_relevant_article(title, url="", description="", published_at=""):
            if len(headlines) >= 15:
                return
            clean_title = str(title or "").strip()
            if not clean_title:
                return
            clean_url = str(url or "").strip()
            clean_description = str(description or "").strip()

            combined_text = f"{clean_title} {clean_description}"
            is_relevant = False
            for term in relevance_terms:
                term_upper = str(term or "").upper().strip()
                if not term_upper:
                    continue
                # Use regex with word boundaries for strict matching (-P1 News Validation)
                pattern = r'\b' + re.escape(term_upper) + r'\b'
                if re.search(pattern, combined_text, re.IGNORECASE):
                    is_relevant = True
                    # Additionally require at least one financial keyword in the
                    # combined text so a ticker name embedded in a movie title
                    # (e.g. "META" in a film description) does not pass the filter.
                    _FINANCIAL_TERMS = re.compile(
                        r'\b(stock|share|price|market|earn|revenue|profit|loss|invest|'
                        r'analyst|quarter|fiscal|IPO|valuat|forecast|guidance|trade|'
                        r'fund|ETF|NYSE|NASDAQ|SEC|CEO|CFO|board|dividend|rally|'
                        r'downgrade|upgrade|outlook|chip|semiconductor|AI|cloud|'
                        r'data.?center|GPU|compute)\b',
                        re.IGNORECASE,
                    )
                    if not _FINANCIAL_TERMS.search(combined_text):
                        is_relevant = False   # ticker name present but zero financial context
                    break
            if not is_relevant:
                return

            # --- NON-FINANCIAL CONTENT FILTER ---
            # Reject headlines that match patterns typical of non-financial noise
            # (torrent filenames, video metadata, piracy site leakage, etc.)
            _NOISE_PATTERNS = [
                r'\b(1080p|720p|480p|2160p|4K|BluRay|WEB-?DL|WEBRip|HDTV|DVDRip|BRRip)\b',
                r'\b(x264|x265|H\.?264|H\.?265|HEVC|AVC|AAC|AC3|DTS|FLAC|MP4|MKV|AVI)\b',
                r'\b(S\d{2}E\d{2}|S\d{2}-S\d{2})\b',   # episode codes like S01E03
                r'\b(YIFY|RARBG|EZTV|BobDobbs|playWEB|Kitsune|TEPES|RAWR|MiXED|SPARKS)\b',
                r'\b(torrent|magnet|repack|proper|extended\.cut|theatrical)\b',
                r'(?i)\.\s*(mkv|mp4|avi|mov|wmv|flv)\b',
            ]
            for _pat in _NOISE_PATTERNS:
                if re.search(_pat, clean_title, re.IGNORECASE):
                    return   # silently drop non-financial noise

            dedupe_key = (clean_title, clean_url)
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)

            # Validate URL is reachable before accepting the headline
            if clean_url:
                try:
                    import urllib.request as _urlreq
                    import urllib.error as _urlerr
                    req = _urlreq.Request(
                        clean_url,
                        headers={'User-Agent': 'Mozilla/5.0 (IRIS-AI headline validator)'},
                    )
                    with _urlreq.urlopen(req, timeout=3) as resp:
                        if resp.status not in (200, 201, 202, 301, 302, 303):
                            return   # inaccessible — drop silently
                except Exception:
                    return   # any network/DNS/timeout error — drop silently

            headlines.append({
                "title": clean_title,
                "url": clean_url,
                "published_at": str(published_at or "").strip(),
            })

        # Preferred source: NewsAPI (if configured) for a larger headline baseline.
        if self.news_api:
            try:
                response = self.news_api.get_everything(
                    q=ticker,
                    language="en",
                    sort_by="publishedAt",
                    page_size=15,
                )
                if isinstance(response, dict):
                    for article in response.get("articles", []) or []:
                        if not isinstance(article, dict):
                            continue
                        title = str(article.get("title", "")).strip()
                        url = article.get("url", "")
                        description = article.get("description", "")
                        add_relevant_article(
                            title=title,
                            url=url,
                            description=description,
                            published_at=article.get("publishedAt", ""),
                        )
                        if len(headlines) >= 15:
                            break
            except Exception:
                headlines = []
                seen = set()

        # Fallback: existing yfinance extraction when NewsAPI is unavailable/failed/empty.
        if not headlines:
            try:
                stock = yf.Ticker(ticker)
                news_items = stock.news
                if news_items:
                    for item in news_items[:30]:
                        if not isinstance(item, dict):
                            continue
                        content = item.get("content") if isinstance(item.get("content"), dict) else {}
                        title = item.get("title") or content.get("title") or ""
                        description = item.get("description") or item.get("summary") or content.get("description") or content.get("summary") or ""
                        url = item.get("link") or item.get("url") or content.get("link") or content.get("url") or ""
                        _pub = item.get("providerPublishTime") or \
                               (item.get("content") or {}).get("pubDate", "")
                        add_relevant_article(
                            title=title,
                            url=url,
                            description=description,
                            published_at=_pub,
                        )
                        if len(headlines) >= 15:
                            break
            except Exception:
                pass
        
        #  Fallback: Simulation Mode (If internet/API failure)
        if not headlines:
            if ticker == "TSLA":
                simulation_items = [
                    {"title": "Tesla recalls 2 million vehicles due to autopilot risk", "url": ""},
                    {"title": "Analysts downgrade Tesla stock amid slowing EV demand", "url": ""},
                ]
            elif ticker == "NVDA":
                simulation_items = [
                    {"title": "Nvidia announces fantastic breakthrough AI chip", "url": ""},
                    {"title": "Nvidia quarterly revenue brilliantly beats expectations by 20%", "url": ""},
                ]
            else:
                simulation_items = [
                    {"title": f"{ticker_symbol} announces date for shareholder meeting", "url": ""},
                    {"title": f"{ticker_symbol} news flow remains active amid market volatility", "url": ""},
                ]
            for entry in simulation_items:
                add_relevant_article(
                    title=entry.get("title", ""),
                    url=entry.get("url", ""),
                    description="",
                )

        #  Analyze Sentiment using FinBERT
        total_score = 0
        valid_headlines = 0
        
        if self.sentiment_analyzer and headlines:
            for headline in headlines:
                try:
                    title_text = str(headline.get("title", "")).strip() if isinstance(headline, dict) else str(headline or "").strip()
                    if not title_text:
                        continue
                    # FinBERT returns labels like 'positive', 'negative', 'neutral'
                    result = self.sentiment_analyzer(title_text)[0]
                    label = result['label']
                    score = result['score'] # Confidence score 0 to 1
                    
                    if label == 'positive':
                        total_score += score
                        valid_headlines += 1
                    elif label == 'negative':
                        total_score -= score
                        valid_headlines += 1
                    else: # neutral
                        valid_headlines += 1
                        # neutral adds 0 to total score
                except Exception:
                    pass
        
        avg_score = total_score / valid_headlines if valid_headlines > 0 else 0
        return avg_score, headlines

    def predict_trend(self, data, sentiment_score):
        """
        The 'Crystal Ball': Uses Random Forest with technical and sentiment features.
        data: dict with 'history' (array) and optionally 'history_df' (DataFrame with features).
        """
        history_prices = data.get("history") if isinstance(data, dict) else data
        history_df = data.get("history_df") if isinstance(data, dict) else None
        try:
            sentiment_value = float(sentiment_score)
        except (TypeError, ValueError):
            sentiment_value = 0.0

        if history_prices is None or len(history_prices) < 5:
            return "INSUFFICIENT DATA", 0.0

        # Feature matrix: day index + technical features + sentiment.
        required_cols = ["sma_5", "sma_10", "sma_20", "returns_1d", "rsi_14", "Volume"]
        if history_df is not None and len(history_df) >= 5 and all(col in history_df.columns for col in required_cols):
            n = len(history_df)
            days = np.arange(n).reshape(-1, 1)
            X = np.hstack([
                days,
                history_df[required_cols].values,
                np.full((n, 1), sentiment_value),
            ])
            y = history_df["Close"].values
            last = history_df.iloc[-1]
            last_close = float(last["Close"])
            next_sma5 = (float(last["Close"]) + float(last["sma_5"]) * 4) / 5
            next_sma10 = (float(last["Close"]) + float(last["sma_10"]) * 9) / 10
            next_sma20 = (float(last["Close"]) + float(last["sma_20"]) * 19) / 20

            # Use next SMA(5) implied close drift for next-day return estimate.
            implied_next_close = float(next_sma5)
            next_ret = ((implied_next_close - last_close) / last_close) if last_close else float(last.get("returns_1d", 0) or 0.0)

            recent_vol = pd.to_numeric(history_df["Volume"].tail(5), errors="coerce").dropna()
            next_volume = float(recent_vol.mean()) if len(recent_vol) else float(last.get("Volume", 0.0) or 0.0)

            last_rsi = float(last.get("rsi_14", 50.0) or 50.0)
            next_rsi14 = float(np.clip(last_rsi + (next_ret * 100.0), 0.0, 100.0))

            next_row = np.array([[
                n,
                next_sma5,
                next_sma10,
                next_sma20,
                next_ret,
                next_rsi14,
                next_volume,
                sentiment_value,
            ]])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            predicted_price = float(model.predict(next_row)[0])
        else:
            n = len(history_prices)
            days = np.arange(n).reshape(-1, 1)
            close_series = pd.Series(np.array(history_prices, dtype=float))
            sma_5 = close_series.rolling(5, min_periods=1).mean()
            sma_10 = close_series.rolling(10, min_periods=1).mean()
            sma_20 = close_series.rolling(20, min_periods=1).mean()
            returns_1d = close_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

            delta = close_series.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            rsi_14 = 100.0 - (100.0 / (1.0 + rs))
            rsi_14 = rsi_14.where(avg_loss != 0.0, 100.0)
            rsi_14 = rsi_14.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0).fillna(50.0).clip(0.0, 100.0)

            volume = np.zeros(n, dtype=float)
            X = np.column_stack([
                days.ravel(),
                sma_5.values,
                sma_10.values,
                sma_20.values,
                returns_1d.values,
                rsi_14.values,
                volume,
                np.full(n, sentiment_value),
            ])
            y = close_series.values
            last_close = float(close_series.iloc[-1]) if n else 0.0
            next_sma5 = (last_close + float(sma_5.iloc[-1]) * 4) / 5 if n else 0.0
            next_sma10 = (last_close + float(sma_10.iloc[-1]) * 9) / 10 if n else 0.0
            next_sma20 = (last_close + float(sma_20.iloc[-1]) * 19) / 20 if n else 0.0
            next_ret = ((next_sma5 - last_close) / last_close) if last_close else 0.0
            next_rsi14 = float(np.clip(float(rsi_14.iloc[-1]) + (next_ret * 100.0), 0.0, 100.0)) if n else 50.0
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            predicted_price = float(model.predict(np.array([[
                n,
                next_sma5,
                next_sma10,
                next_sma20,
                next_ret,
                next_rsi14,
                0.0,
                sentiment_value,
            ]]))[0])

        current_price = float(history_prices[-1]) if len(history_prices) else 0.0
        pct_change = ((predicted_price - current_price) / current_price * 100) if current_price else 0.0

        if pct_change > 0.5:
            label = "STRONG UPTREND "
        elif pct_change > 0:
            label = "WEAK UPTREND "
        elif pct_change < -0.5:
            label = "STRONG DOWNTREND "
        else:
            label = "WEAK DOWNTREND "

        return label, predicted_price

    def draw_chart(self, symbol: str, history_df, current_price: float, predicted_price: float, trend_label: str, save_dir: str = ""):
        """Draw live price history and prediction trend; save to dated subfolder under data/charts (YYYY-MM-DD)."""
        if not _HAS_MATPLOTLIB or history_df is None or history_df.empty:
            return None
        base_dir = Path(save_dir) if str(save_dir or "").strip() else CHARTS_DIR
        if not base_dir.is_absolute():
            base_dir = PROJECT_ROOT / base_dir
        date_str = time.strftime("%Y-%m-%d")
        daily_dir = base_dir / date_str
        daily_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        dates = history_df.index
        ax.plot(dates, history_df["Close"], color="steelblue", linewidth=2, label="Close price")
        # Add predicted next point (one day after last date)
        if len(dates):
            last_ts = pd.Timestamp(dates[-1])
            next_ts = last_ts + pd.Timedelta(days=1)
            ax.scatter([next_ts], [predicted_price], color="darkorange", s=80, zorder=5, label="AI prediction")
            ax.axhline(y=current_price, color="gray", linestyle="--", alpha=0.7, label=f"Current ${current_price:.2f}")
        ax.set_title(f"{symbol} - Live price & prediction | {trend_label.strip()}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=15)
        plt.tight_layout()
        path = daily_dir / f"{symbol}_trend.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        try:
            return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return str(path)

    def run_one_ticker(
        self,
        ticker: str,
        quiet: bool = False,
        interactive_prompt: bool = False,
        period: str = "60d",
        interval: str = "1d",
        include_chart_history: bool = False,
    ):
        """Run analysis for a single ticker; returns report dict or None."""
        ticker = self.resolve_user_ticker_input(ticker, interactive_prompt=interactive_prompt, quiet=quiet)
        analyzed_ticker = str(ticker).strip().upper()
        canonical_ticker = normalize_ticker_symbol(analyzed_ticker)
        if not quiet:
            print(f"... Analyzing {analyzed_ticker} ...")
            if canonical_ticker != analyzed_ticker:
                print(f"  Note: {analyzed_ticker} will be merged into canonical symbol {canonical_ticker}.")
        data = self.get_market_data(analyzed_ticker, period=period, interval=interval)
        if not data:
            if not quiet:
                print(f"  {analyzed_ticker}: Stock not found or connection error.")
            return None

        sentiment_score, headlines = self.analyze_news(analyzed_ticker)
        trend_label, predicted_price = self.predict_trend(data, sentiment_score)
        light = " GREEN (Safe to Proceed)"
        if sentiment_score < -0.05 or "STRONG DOWNTREND" in trend_label:
            light = " RED (Risk Detected - Caution)"
        elif abs(sentiment_score) < 0.05 and "WEAK" in trend_label:
            light = " YELLOW (Neutral / Noise)"

        market_session_date = None
        history_df = data.get("history_df")
        if history_df is not None and len(history_df):
            try:
                market_session_date = str(pd.Timestamp(history_df.index[-1]).date())
            except Exception:
                market_session_date = None

        evidence_headlines = []
        if isinstance(headlines, list):
            for entry in headlines:
                if not isinstance(entry, dict):
                    continue
                title_text = str(entry.get("title", "")).strip()
                if not title_text:
                    continue
                evidence_headlines.append({
                    "title": title_text,
                    "url": str(entry.get("url", "")).strip(),
                    "published_at": str(entry.get("published_at", "")).strip(),
                })

        report = {
            "meta": {
                "symbol": canonical_ticker,
                "source_symbol": analyzed_ticker,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "market_session_date": market_session_date,
                "mode": "live" if NEWS_API_KEY else "simulation",
                "period": period,
                "interval": interval,
            },
            "market": {
                "current_price": float(data["current_price"]),
                "predicted_price_next_session": float(predicted_price),
            },
            "signals": {
                "trend_label": trend_label,
                "sentiment_score": float(sentiment_score),
                "check_engine_light": light,
            },
            "evidence": {"headlines_used": evidence_headlines},
        }

        chart_path = self.draw_chart(
            canonical_ticker,
            data.get("history_df"),
            data["current_price"],
            predicted_price,
            trend_label,
        )
        report["evidence"]["chart_path"] = chart_path
        saved_path = self.save_report(report, f"{canonical_ticker}_report.json")

        if chart_path and not quiet:
            print(f"  Chart saved: {chart_path}")
        if not quiet:
            print(f"  Report: {canonical_ticker} | {light} | Predicted next: ${predicted_price:.2f} | {saved_path}")

        # Optionally include chart history in API response without storing it in report logs.
        if include_chart_history:
            history_points = data.get("history_points", []) if isinstance(data.get("history_points", []), list) else []
            response = json.loads(json.dumps(report))
            response.setdefault("market", {})["history"] = history_points
            return response
        return report

    def run_auto(self, tickers: list):
        """Run analysis for a list of tickers (for automated daily runs)."""
        print("\n-- IRIS automated run --")
        results = []
        for t in normalize_ticker_list(tickers):
            if not t:
                continue
            report = self.run_one_ticker(t, quiet=False)
            if report:
                results.append(report)

        if results:
            summary_path = self.save_session_summary(results)
            if summary_path:
                print(f"Session summary: {summary_path}")
        return results

    def run(self):
        print("\n" + "="*50)
        print("--IRIS: INTELLIGENT RISK IDENTIFICATION SYSTEM--")
        print("    'The Check Engine Light for your Portfolio'")
        print("="*50)
        while True:
            ticker = input("\nEnter Stock Ticker or Company Name (e.g., AAPL, TSLA, Google) or 'q' to quit: ").strip().upper()
            if ticker == "Q":
                print("Shutting down IRIS...")
                break
            if not ticker:
                continue
            self.run_one_ticker(ticker, quiet=False, interactive_prompt=True)
            print("="*40)
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="IRIS - Intelligent Risk Identification System")
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers (e.g. AAPL,TSLA). Used with --auto or for one-shot run.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run once for given tickers then exit (no interactive prompt). Use for daily scheduler.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive mode (default if no --auto and no --tickers).",
    )
    args = parser.parse_args()
    app = IRIS_System()
    if args.auto or args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else DEFAULT_TICKERS
        app.run_auto(tickers)
        return
    app.run()


if __name__ == "__main__":
    main()

