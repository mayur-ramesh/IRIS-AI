import argparse
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import yfinance as yf
from newsapi import NewsApiClient
import nltk
from transformers import pipeline
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import time
import json
from pathlib import Path

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
        
        # Setup Sentiment Brain (FinBERT - financial sentiment model)
        print("   -> Loading FinBERT AI Model (This may take a moment on first run)...")
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
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
        """Merge alias symbols into canonical report files (e.g. GOOGL -> GOOG)."""
        data_dir = Path("data")
        if not data_dir.exists():
            return None

        for alias, canonical in TICKER_ALIASES.items():
            alias_path = data_dir / f"{alias}_report.json"
            canonical_path = data_dir / f"{canonical}_report.json"
            alias_reports = self._read_report_file(alias_path)
            if not alias_reports:
                continue
            canonical_reports = self._read_report_file(canonical_path)

            merged = {}
            for report in canonical_reports + alias_reports:
                if not isinstance(report, dict):
                    continue
                meta = report.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                    report["meta"] = meta
                meta["symbol"] = canonical

                session_key = self._extract_session_date(report) or self._report_generated_at(report)
                if not session_key:
                    session_key = f"legacy_{len(merged)}"

                existing = merged.get(session_key)
                if existing is None or self._report_generated_at(report) >= self._report_generated_at(existing):
                    merged[session_key] = report

            merged_reports = sorted(merged.values(), key=self._report_sort_key)
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
        Path("data").mkdir(exist_ok=True)
        canonical_filename = filename
        if filename.endswith("_report.json"):
            symbol = filename[:-12]
            canonical_filename = f"{normalize_ticker_symbol(symbol)}_report.json"
        out_path = Path("data") / canonical_filename
        
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
        
        # Upsert by market session date to avoid duplicate entries for the same trading session.
        target_session = self._extract_session_date(report)
        updated = False
        if target_session:
            for idx, existing in enumerate(reports):
                if self._extract_session_date(existing) == target_session:
                    reports[idx] = report
                    updated = True
                    break
        if not updated:
            reports.append(report)
        
        # Save all reports
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        return str(out_path)

    def _load_symbol_reports(self, symbol: str):
        canonical_symbol = normalize_ticker_symbol(symbol)
        out_path = Path("data") / f"{canonical_symbol}_report.json"
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

        sessions_dir = Path("data") / "sessions" / session_date
        sessions_dir.mkdir(parents=True, exist_ok=True)
        summary_path = sessions_dir / "session_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        latest_path = Path("data") / "sessions" / "latest_session_summary.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return str(summary_path)

    def get_market_data(self, ticker):
        """Fetches current price and 60-day history with features for the Prediction Engine."""
        try:
            stock = yf.Ticker(ticker)
            price = getattr(stock.fast_info, "last_price", None) or getattr(stock.fast_info, "previous_close", None)
            if price is None:
                return None
            # Use 60d for more robust feature computation (SMAs, etc.)
            hist = stock.history(period="60d")
            if hist is None or hist.empty or len(hist) < 10:
                return {
                    "current_price": float(price),
                    "history": None,
                    "history_df": None,
                    "symbol": ticker.upper(),
                }
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
                "history": df["Close"].values,
                "history_df": df,
                "symbol": ticker.upper(),
            }
        except Exception:
            return None

    def analyze_news(self, ticker):
        """Fetches headlines and calculates a Sentiment Score (-1.0 to +1.0)."""
        headlines = []
        
        # Fetch real, live news using yfinance
        try:
            stock = yf.Ticker(ticker)
            news_items = stock.news
            if news_items:
                for item in news_items[:5]: # Get up to 5 recent headlines
                    if 'title' in item:
                        headlines.append(item['title'])
                    elif 'content' in item and 'title' in item['content']:
                        headlines.append(item['content']['title'])
        except Exception:
            pass
        
        #  Fallback: Simulation Mode (If internet/API failure)
        if not headlines:
            if ticker == "TSLA":
                headlines = [
                    "Tesla recalls 2 million vehicles due to autopilot risk",
                    "Analysts downgrade stock amid slowing EV demand"
                ]
            elif ticker == "NVDA":
                headlines = [
                    "Nvidia announces fantastic breakthrough AI chip", 
                    "Excellent quarterly revenue brilliantly beats expectations by 20%"
                ]
            else:
                headlines = [
                    f"{ticker} announces date for shareholder meeting",
                    "Market volatility continues ahead of Fed decision"
                ]

        #  Analyze Sentiment using FinBERT
        total_score = 0
        valid_headlines = 0
        
        if self.sentiment_analyzer and headlines:
            for h in headlines:
                try:
                    # FinBERT returns labels like 'positive', 'negative', 'neutral'
                    result = self.sentiment_analyzer(h)[0]
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

    def draw_chart(self, symbol: str, history_df, current_price: float, predicted_price: float, trend_label: str, save_dir: str = "data/charts"):
        """Draw live price history and prediction trend; save to dated subfolder under data/charts (YYYY-MM-DD)."""
        if not _HAS_MATPLOTLIB or history_df is None or history_df.empty:
            return None
        base_dir = Path(save_dir)
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
        return str(path)

    def run_one_ticker(self, ticker: str, quiet: bool = False, interactive_prompt: bool = False):
        """Run analysis for a single ticker; returns report dict or None."""
        ticker = self.resolve_user_ticker_input(ticker, interactive_prompt=interactive_prompt, quiet=quiet)
        analyzed_ticker = str(ticker).strip().upper()
        canonical_ticker = normalize_ticker_symbol(analyzed_ticker)
        if not quiet:
            print(f"... Analyzing {analyzed_ticker} ...")
            if canonical_ticker != analyzed_ticker:
                print(f"  Note: {analyzed_ticker} will be merged into canonical symbol {canonical_ticker}.")
        data = self.get_market_data(analyzed_ticker)
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

        report = {
            "meta": {
                "symbol": canonical_ticker,
                "source_symbol": analyzed_ticker,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "market_session_date": market_session_date,
                "mode": "live" if NEWS_API_KEY else "simulation",
            },
            "market": {
                "current_price": float(data["current_price"]),
                "predicted_price_next_session": float(predicted_price),
                "history": [{"time": ts.strftime('%Y-%m-%d'), "value": float(price)} for ts, price in history_df["Close"].items()] if history_df is not None else [],
            },
            "signals": {
                "trend_label": trend_label,
                "sentiment_score": float(sentiment_score),
                "check_engine_light": light,
            },
            "evidence": {"headlines_used": headlines if isinstance(headlines, list) else []},
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

