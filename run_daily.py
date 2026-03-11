"""
Automate IRIS report generation and session-to-session comparison.

Usage examples:
  python run_daily.py --once
  python run_daily.py --once-if-market-open
  python run_daily.py --once --tickers AAPL,TSLA
  python run_daily.py                    # daemon loop, runs at 09:00 ET
  python run_daily.py --install-task     # task checks every 5 min and runs at 09:00 ET
  python run_daily.py --uninstall-task
"""
import argparse
from datetime import datetime, timedelta
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo
from storage_paths import resolve_data_dir

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_TASK_NAME = "IRIS-Daily-Reports"
_FALLBACK_TICKERS = ["AAPL", "MSFT", "GOOGL"]
_TICKER_ALIASES = {
    "GOOGL": "GOOG",
}
_MARKET_TZ = ZoneInfo("America/New_York")
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MINUTE = 0
_MARKET_OPEN_WINDOW_MINUTES = 10
_RUN_MARKER_PATH = None

# Ensure we run from project root (so data/ and data/charts/ are in the right place)
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass


def _resolve_runtime_data_dir():
    demo_mode = os.environ.get("DEMO_MODE", "false").lower() == "true"
    return resolve_data_dir(_PROJECT_ROOT, demo_mode)


_RUN_MARKER_PATH = _resolve_runtime_data_dir() / "sessions" / ".last_market_open_run_et_date"


def _parse_tickers(raw: str):
    if not raw:
        return None
    return _normalize_ticker_list(raw.split(","))


def _canonical_ticker(symbol: str):
    token = str(symbol or "").strip().upper()
    if not token:
        return token
    return _TICKER_ALIASES.get(token, token)


def _normalize_ticker_list(symbols):
    seen = set()
    normalized = []
    for symbol in symbols or []:
        token = _canonical_ticker(symbol)
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _load_watchlist_tickers():
    watchlist_path = _PROJECT_ROOT / "watchlist.txt"
    if not watchlist_path.exists():
        return []
    tickers = []
    for raw_line in watchlist_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[\s,]+", line) if p]
        tickers.extend(parts)
    return _normalize_ticker_list(tickers)


def _default_tickers():
    watchlist_tickers = _load_watchlist_tickers()
    if watchlist_tickers:
        return watchlist_tickers

    raw = os.environ.get("IRIS_TICKERS", "")
    env_tickers = _normalize_ticker_list(raw.split(","))
    if env_tickers:
        return env_tickers

    try:
        from iris_mvp import DEFAULT_TICKERS  # type: ignore

        if DEFAULT_TICKERS:
            return _normalize_ticker_list(DEFAULT_TICKERS)
    except Exception:
        pass

    return _normalize_ticker_list(_FALLBACK_TICKERS)


def _now_et():
    return datetime.now(_MARKET_TZ)


def _today_et_str():
    return _now_et().strftime("%Y-%m-%d")


def _already_ran_today_et():
    if not _RUN_MARKER_PATH.exists():
        return False
    try:
        return _RUN_MARKER_PATH.read_text(encoding="utf-8").strip() == _today_et_str()
    except OSError:
        return False


def _mark_ran_today_et():
    _RUN_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RUN_MARKER_PATH.write_text(_today_et_str(), encoding="utf-8")


def _is_market_open_window_et(now_et=None):
    now_et = now_et or _now_et()
    if now_et.weekday() >= 5:
        return False
    window_start = now_et.replace(hour=_MARKET_OPEN_HOUR, minute=_MARKET_OPEN_MINUTE, second=0, microsecond=0)
    window_end = window_start + timedelta(minutes=_MARKET_OPEN_WINDOW_MINUTES)
    return window_start <= now_et < window_end


def run_once(tickers=None):
    from iris_mvp import IRIS_System  # type: ignore

    target_tickers = _normalize_ticker_list(tickers or _default_tickers())
    if not target_tickers:
        print("No tickers configured. Set IRIS_TICKERS in .env or use --tickers.")
        return 1
    app = IRIS_System()
    app.run_auto(target_tickers)
    return 0


def run_once_if_market_open(tickers=None):
    now_et = _now_et()
    if not _is_market_open_window_et(now_et):
        print(
            f"Skip: now {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')} is outside the ET 09:00 open window."
        )
        return 0
    if _already_ran_today_et():
        print(f"Skip: market-open run already completed for {_today_et_str()} ET.")
        return 0
    code = run_once(tickers=tickers)
    if code == 0:
        _mark_ran_today_et()
    return code


def run_scheduled(tickers=None):
    from iris_mvp import IRIS_System  # type: ignore

    target_tickers = _normalize_ticker_list(tickers or _default_tickers())
    if not target_tickers:
        target_tickers = _FALLBACK_TICKERS
    app = IRIS_System()
    print("IRIS scheduler started: will run at 09:00 ET on market weekdays, regardless of local timezone.")
    while True:
        now_et = _now_et()
        if _is_market_open_window_et(now_et) and not _already_ran_today_et():
            print(f"Triggering run for ET market open ({now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}).")
            app.run_auto(target_tickers)
            _mark_ran_today_et()
            time.sleep(65)
            continue
        time.sleep(20)


def _build_task_command(python_exe: str, tickers=None):
    cmd_parts = [
        f'"{python_exe}"',
        f'"{_PROJECT_ROOT / "run_daily.py"}"',
        "--once-if-market-open",
    ]
    if tickers:
        cmd_parts.extend(["--tickers", ",".join(tickers)])
    return " ".join(cmd_parts)


def install_windows_task(task_name: str, python_exe: str, tickers=None):
    if os.name != "nt":
        raise RuntimeError("Windows Task Scheduler setup is only available on Windows.")

    task_command = _build_task_command(python_exe=python_exe, tickers=tickers)
    cmd = [
        "schtasks",
        "/Create",
        "/F",
        "/SC",
        "MINUTE",
        "/MO",
        "5",
        "/TN",
        task_name,
        "/TR",
        task_command,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Failed to create scheduled task.")
    print(f"Scheduled task installed: {task_name} (checks every 5 min; runs once at 09:00 ET)")


def uninstall_windows_task(task_name: str):
    if os.name != "nt":
        raise RuntimeError("Windows Task Scheduler setup is only available on Windows.")

    cmd = ["schtasks", "/Delete", "/TN", task_name, "/F"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Failed to delete scheduled task.")
    print(f"Scheduled task removed: {task_name}")


def main():
    parser = argparse.ArgumentParser(description="IRIS trading-session automation")
    parser.add_argument("--once", action="store_true", help="Run one session now and exit.")
    parser.add_argument(
        "--once-if-market-open",
        action="store_true",
        help="Run once only if current time is within the ET 09:00 market-open window.",
    )
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers.")
    parser.add_argument(
        "--install-task",
        action="store_true",
        help="Install Windows Task Scheduler checker (every 5 minutes, ET-open guarded).",
    )
    parser.add_argument("--uninstall-task", action="store_true", help="Remove Windows Task Scheduler job.")
    parser.add_argument("--task-name", type=str, default=_DEFAULT_TASK_NAME, help="Windows Task Scheduler task name.")
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable path for the scheduled task.",
    )
    args = parser.parse_args()

    tickers = _parse_tickers(args.tickers)

    if args.install_task and args.uninstall_task:
        print("Use either --install-task or --uninstall-task, not both.")
        return 2

    if args.install_task:
        install_windows_task(
            task_name=args.task_name,
            python_exe=args.python_exe,
            tickers=tickers,
        )
        return 0

    if args.uninstall_task:
        uninstall_windows_task(task_name=args.task_name)
        return 0

    if args.once:
        return run_once(tickers=tickers)
    if args.once_if_market_open:
        return run_once_if_market_open(tickers=tickers)

    run_scheduled(tickers=tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
