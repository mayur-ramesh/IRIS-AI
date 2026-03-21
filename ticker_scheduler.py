"""
Background scheduler that refreshes the ticker database every 24 hours.
Uses a self-rescheduling threading.Timer so no extra packages are required.
"""

import logging
import threading

from ticker_db import refresh_ticker_db

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL_SECONDS = 24 * 3600  # 24 hours

_timer: threading.Timer | None = None
_scheduler_lock = threading.Lock()


def _run_refresh() -> None:
    """Execute one refresh cycle, then schedule the next run."""
    global _timer
    try:
        result = refresh_ticker_db()
        logger.info(
            "Scheduled ticker DB refresh complete: %d tickers (+%d added, -%d removed)",
            result.get("ticker_count", 0),
            result.get("added", 0),
            result.get("removed", 0),
        )
    except Exception as exc:
        logger.error("Scheduled ticker DB refresh raised an unexpected error: %s", exc)
    finally:
        # Always reschedule the next run regardless of success or failure
        with _scheduler_lock:
            global _timer  # noqa: F821 – re-bind in enclosing scope
            _timer = threading.Timer(_REFRESH_INTERVAL_SECONDS, _run_refresh)
            _timer.daemon = True
            _timer.start()


def start_scheduler() -> None:
    """Start the 24-hour background refresh timer.

    Safe to call multiple times — subsequent calls are no-ops if already running.
    """
    global _timer
    with _scheduler_lock:
        if _timer is not None:
            logger.debug("Ticker DB scheduler already running; skipping start.")
            return
        _timer = threading.Timer(_REFRESH_INTERVAL_SECONDS, _run_refresh)
        _timer.daemon = True
        _timer.start()
    logger.info(
        "Ticker DB scheduler started (refresh interval: %d hours).",
        _REFRESH_INTERVAL_SECONDS // 3600,
    )


def stop_scheduler() -> None:
    """Cancel the pending refresh timer.

    The currently-running refresh (if any) will complete; only the *next*
    scheduled run is cancelled.
    """
    global _timer
    with _scheduler_lock:
        if _timer is not None:
            _timer.cancel()
            _timer = None
    logger.info("Ticker DB scheduler stopped.")
