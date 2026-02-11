import argparse
import os
import sys
import yfinance as yf
from newsapi import NewsApiClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import json
from pathlib import Path

# Optional: charting (graceful fallback if matplotlib missing)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Load .env if available (for NEWS_API_KEY, IRIS_TICKERS)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
NEWS_API_KEY = os.environ.get("NEWS_API_KEY") or None
DEFAULT_TICKERS = [t for t in os.environ.get("IRIS_TICKERS", "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA").split(",") if t.strip()]



class IRIS_System:
    def __init__(self):
        print("\n  Initializing IRIS Risk Engines...")
        
        #. Setup Sentiment Brain (VADER)
        # We download the dictionary required to understand words like "crash" or "soar"
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            print("   -> Downloading NLTK lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        #  Setup News Connection
        self.news_api = None
        if NEWS_API_KEY:
            self.news_api = NewsApiClient(api_key=NEWS_API_KEY)
            print("   -> NewsAPI Connection: ESTABLISHED")
        else:
            print("   -> NewsAPI Connection: SIMULATION MODE (No Key Found)")

    def save_report(self, report: dict, filename: str):
        Path("data").mkdir(exist_ok=True)
        out_path = Path("data") / filename
        
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
        
        # Append new report
        reports.append(report)
        
        # Save all reports
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        return str(out_path)

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
            df["returns_1d"] = df["Close"].pct_change()
            df["sma_5"] = df["Close"].rolling(5, min_periods=1).mean()
            df["sma_10"] = df["Close"].rolling(10, min_periods=1).mean()
            df["sma_20"] = df["Close"].rolling(20, min_periods=1).mean()
            df = df.dropna()
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
        
        #  Try to get real news
        if self.news_api:
            try:
                response = self.news_api.get_everything(
                    q=ticker, 
                    language='en', 
                    sort_by='publishedAt',
                    page_size=5
                )
                headlines = [art['title'] for art in response['articles']]
            except:
                pass # Fail silently and use simulation if API errors out
        
        #  Fallback: Simulation Mode (If no Key or API failure)
        if not headlines:
            if ticker == "TSLA":
                headlines = [
                    "Tesla recalls 2 million vehicles due to autopilot risk",
                    "Analysts downgrade stock amid slowing EV demand"
                ]
            elif ticker == "NVDA":
                headlines = [
                    "Nvidia announces breakthrough AI chip", 
                    "Quarterly revenue beats expectations by 20%"
                ]
            else:
                headlines = [
                    f"{ticker} announces date for shareholder meeting",
                    "Market volatility continues ahead of Fed decision"
                ]

        #  Analyze Sentiment
        total_score = 0
        for h in headlines:
            # compound score ranges from -1 (Negative) to +1 (Positive)
            score = self.sentiment_analyzer.polarity_scores(h)['compound']
            total_score += score
        
        avg_score = total_score / len(headlines) if headlines else 0
        return avg_score, headlines

    def predict_trend(self, data):
        """
        The 'Crystal Ball': Uses Ridge regression with multiple features for better accuracy.
        data: dict with 'history' (array) and optionally 'history_df' (DataFrame with features).
        """
        history_prices = data.get("history") if isinstance(data, dict) else data
        history_df = data.get("history_df") if isinstance(data, dict) else None

        if history_prices is None or len(history_prices) < 5:
            return "INSUFFICIENT DATA", 0.0

        # Feature matrix: day index + optional technical features for better accuracy
        if history_df is not None and len(history_df) >= 5 and "sma_5" in history_df.columns:
            n = len(history_df)
            days = np.arange(n).reshape(-1, 1)
            X = np.hstack([
                days,
                history_df[["sma_5", "sma_10", "returns_1d"]].values,
            ])
            y = history_df["Close"].values
            last = history_df.iloc[-1]
            next_sma5 = (float(last["Close"]) + float(last["sma_5"]) * 4) / 5
            next_sma10 = (float(last["Close"]) + float(last["sma_10"]) * 9) / 10
            next_ret = float(last.get("returns_1d", 0) or 0)
            next_row = np.array([[n, next_sma5, next_sma10, next_ret]])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_scaled, y)
            next_X = scaler.transform(next_row)
            predicted_price = float(model.predict(next_X)[0])
            slope = float(model.coef_[0]) if len(model.coef_) else 0
        else:
            n = len(history_prices)
            days = np.arange(n).reshape(-1, 1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(days)
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_scaled, history_prices)
            next_day_index = scaler.transform(np.array([[n]]))
            predicted_price = float(model.predict(next_day_index)[0])
            slope = float(model.coef_[0])

        # Normalize slope by typical price level for consistent thresholds
        avg_price = np.mean(history_prices)
        slope_norm = (slope / avg_price * 100) if avg_price else 0

        if slope_norm > 0.15:
            label = "STRONG UPTREND "
        elif slope_norm > 0:
            label = "WEAK UPTREND "
        elif slope_norm < -0.15:
            label = "STRONG DOWNTREND "
        else:
            label = "WEAK DOWNTREND "

        return label, predicted_price

    def draw_chart(self, symbol: str, history_df, current_price: float, predicted_price: float, trend_label: str, save_dir: str = "data/charts"):
        """Draw live price history and prediction trend; save to data/charts."""
        if not _HAS_MATPLOTLIB or history_df is None or history_df.empty:
            return None
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        dates = history_df.index
        ax.plot(dates, history_df["Close"], color="steelblue", linewidth=2, label="Close price")
        # Add predicted next point (one day after last date)
        if len(dates):
            last_ts = pd.Timestamp(dates[-1])
            next_ts = last_ts + pd.Timedelta(days=1)
            ax.scatter([next_ts], [predicted_price], color="darkorange", s=80, zorder=5, label="AI prediction")
            ax.axhline(y=current_price, color="gray", linestyle="--", alpha=0.7, label=f"Current ${current_price:.2f}")
        ax.set_title(f"{symbol} — Live price & prediction | {trend_label.strip()}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=15)
        plt.tight_layout()
        path = Path(save_dir) / f"{symbol}_trend.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    def run_one_ticker(self, ticker: str, quiet: bool = False):
        """Run analysis for a single ticker; returns report dict or None."""
        if not quiet:
            print(f"... Analyzing {ticker} ...")
        data = self.get_market_data(ticker)
        if not data:
            if not quiet:
                print(f"  {ticker}: Stock not found or connection error.")
            return None
        sentiment_score, headlines = self.analyze_news(ticker)
        trend_label, predicted_price = self.predict_trend(data)
        light = " GREEN (Safe to Proceed)"
        if sentiment_score < -0.05 or "STRONG DOWNTREND" in trend_label:
            light = " RED (Risk Detected - Caution)"
        elif abs(sentiment_score) < 0.05 and "WEAK" in trend_label:
            light = " YELLOW (Neutral / Noise)"
        report = {
            "meta": {
                "symbol": data["symbol"],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": "live" if NEWS_API_KEY else "simulation",
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
            "evidence": {"headlines_used": headlines if isinstance(headlines, list) else []},
        }
        saved_path = self.save_report(report, f"{data['symbol']}_report.json")
        chart_path = self.draw_chart(
            data["symbol"],
            data.get("history_df"),
            data["current_price"],
            predicted_price,
            trend_label,
        )
        if chart_path and not quiet:
            print(f"  Chart saved: {chart_path}")
        if not quiet:
            print(f"  Report: {data['symbol']} — {light} | Predicted next: ${predicted_price:.2f} | {saved_path}")
        return report

    def run_auto(self, tickers: list):
        """Run analysis for a list of tickers (for automated daily runs)."""
        print("\n-- IRIS automated run --")
        results = []
        for t in tickers:
            t = str(t).strip().upper()
            if not t:
                continue
            r = self.run_one_ticker(t, quiet=False)
            if r:
                results.append(r)
        return results

    def run(self):
        print("\n" + "="*50)
        print("--IRIS: INTELLIGENT RISK IDENTIFICATION SYSTEM--")
        print("    'The Check Engine Light for your Portfolio'")
        print("="*50)
        while True:
            ticker = input("\nEnter Stock Ticker (e.g., AAPL, TSLA) or 'q' to quit: ").strip().upper()
            if ticker == "Q":
                print("Shutting down IRIS...")
                break
            if not ticker:
                continue
            self.run_one_ticker(ticker, quiet=False)
            print("="*40)
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="IRIS — Intelligent Risk Identification System")
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