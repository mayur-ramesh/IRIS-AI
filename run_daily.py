"""
Run IRIS on a schedule (e.g. once per day) for automated live market data and reports.
Usage:
  python run_daily.py                    # run every day at 09:35 (configurable)
  python run_daily.py --once             # run once now then exit (for cron/Task Scheduler)
  python run_daily.py --once --tickers AAPL,TSLA
Set IRIS_TICKERS and NEWS_API_KEY in .env or environment.
"""
import argparse
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
# Ensure we run from project root (so data/ and data/charts/ are in the right place)
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

from iris_mvp import IRIS_System, DEFAULT_TICKERS

_FALLBACK_TICKERS = ["AAPL", "MSFT", "GOOGL"]


def run_once(tickers=None):
    tickers = tickers or DEFAULT_TICKERS or _FALLBACK_TICKERS
    if not tickers:
        print("No tickers configured. Set IRIS_TICKERS in .env or use --tickers.")
        return
    app = IRIS_System()
    app.run_auto(tickers)


def run_scheduled():
    import schedule  # type: ignore
    tickers = DEFAULT_TICKERS or _FALLBACK_TICKERS
    if not tickers:
        tickers = _FALLBACK_TICKERS
    app = IRIS_System()

    # Run every day at 9:35 AM (after market open; adjust time as needed)
    def job():
        app.run_auto(tickers)

    schedule.every().day.at("09:35").do(job)
    print("IRIS daily scheduler started. Next run: 09:35. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="IRIS daily automation")
    parser.add_argument("--once", action="store_true", help="Run once now then exit (for cron/Task Scheduler)")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers (default: IRIS_TICKERS or AAPL,MSFT,...)")
    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    if args.once:
        run_once(tickers)
    else:
        run_scheduled()


if __name__ == "__main__":
    main()
