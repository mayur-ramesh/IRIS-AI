"""
Draw a trend-direction accuracy comparison chart for IRIS and three LLMs.

Data sources (default dir: runtime_data/):
- <data-dir>/<TICKER>_report.json (IRIS reports)
- <data-dir>/LLM reports/chatgpt_5.2.json
- <data-dir>/LLM reports/deepseek_v3.json
- <data-dir>/LLM reports/gemini_v3_pro.json

Ticker selection order:
1) --tickers
2) watchlist.txt (project root)
3) discovered *_report.json files in --data-dir
4) fallback defaults
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from storage_paths import resolve_data_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FALLBACK_TICKERS = ["AMZN", "NVDA", "TSLA", "AAPL", "GOOG"]
TICKER_ALIASES = {"GOOGL": "GOOG"}

MODEL_CONFIG = {
    "iris": {"label": "IRIS", "relative_path": None},
    "chatgpt_5_2": {"label": "ChatGPT 5.2", "relative_path": Path("LLM reports") / "chatgpt_5.2.json"},
    "deepseek_v3": {"label": "DeepSeek V3", "relative_path": Path("LLM reports") / "deepseek_v3.json"},
    "gemini_v3_pro": {"label": "Gemini V3 Pro", "relative_path": Path("LLM reports") / "gemini_v3_pro.json"},
}

MODEL_COLORS = {
    "iris": "#1f77b4",
    "chatgpt_5_2": "#ff7f0e",
    "deepseek_v3": "#2ca02c",
    "gemini_v3_pro": "#d62728",
}
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Prediction:
    ticker: str
    session_date: str
    generated_at: str
    current_price: float
    predicted_price_next_session: float


def _normalize_date_arg(value: str, flag_name: str) -> Optional[str]:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        return datetime.strptime(token, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be in YYYY-MM-DD format, got: {token}") from exc


def _in_date_range(session_date: str, start_date: Optional[str], end_date: Optional[str]) -> bool:
    if not session_date:
        return False
    if start_date and session_date < start_date:
        return False
    if end_date and session_date > end_date:
        return False
    return True


def _filter_predictions_by_date(
    predictions: List[Prediction],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Prediction]:
    return [p for p in predictions if _in_date_range(p.session_date, start_date, end_date)]


def _filter_iris_by_date(
    iris_by_ticker: Dict[str, List[Prediction]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, List[Prediction]]:
    return {
        ticker: _filter_predictions_by_date(rows, start_date, end_date)
        for ticker, rows in iris_by_ticker.items()
    }


def _normalize_ticker_list(symbols: List[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for symbol in symbols:
        token = _canonical_ticker(symbol)
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _parse_tickers(raw: str) -> List[str]:
    tokenized = [p for p in re.split(r"[\s,]+", str(raw or "").strip()) if p]
    return _normalize_ticker_list(tokenized)


def _load_watchlist_tickers(watchlist_path: Path) -> List[str]:
    if not watchlist_path.exists():
        return []
    out: List[str] = []
    for raw_line in watchlist_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        out.extend([p for p in re.split(r"[\s,]+", line) if p])
    return _normalize_ticker_list(out)


def _discover_report_tickers(data_dir: Path) -> List[str]:
    suffix = "_report.json"
    discovered = []
    for path in sorted(data_dir.glob(f"*{suffix}")):
        name = path.name
        if not name.endswith(suffix):
            continue
        discovered.append(name[: -len(suffix)])
    return _normalize_ticker_list(discovered)


def resolve_tracked_tickers(data_dir: Path, ticker_arg: str) -> List[str]:
    explicit = _parse_tickers(ticker_arg)
    if explicit:
        return explicit

    watchlist_path = Path(__file__).resolve().parent / "watchlist.txt"
    from_watchlist = _load_watchlist_tickers(watchlist_path)
    if from_watchlist:
        return from_watchlist

    discovered = _discover_report_tickers(data_dir)
    if discovered:
        return discovered

    return DEFAULT_FALLBACK_TICKERS.copy()


def _collect_actual_dates(actual_direction_map: Dict[str, Dict[str, int]], tracked_tickers: List[str]) -> List[str]:
    dates = sorted(
        {
            session_date
            for ticker in tracked_tickers
            for session_date in actual_direction_map.get(ticker, {}).keys()
        }
    )
    return dates


def _format_date_range_label(start_date: Optional[str], end_date: Optional[str]) -> str:
    if start_date and end_date:
        return f"{start_date} to {end_date}"
    if start_date:
        return f"from {start_date}"
    if end_date:
        return f"up to {end_date}"
    return "all available dates"


def _canonical_ticker(symbol: object) -> str:
    token = str(symbol or "").strip().upper()
    return TICKER_ALIASES.get(token, token)


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _direction(delta: float, eps: float = 1e-9) -> int:
    if delta > eps:
        return 1
    if delta < -eps:
        return -1
    return 0


def _read_json_as_list(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _extract_session_date(meta: Optional[dict], fallback_generated_at: str = "") -> str:
    if isinstance(meta, dict):
        raw = str(meta.get("market_session_date", "")).strip()
        if len(raw) >= 10:
            return raw[:10]
        generated_at = str(meta.get("generated_at", "")).strip()
        if len(generated_at) >= 10:
            return generated_at[:10]
    if len(fallback_generated_at) >= 10:
        return fallback_generated_at[:10]
    return ""


def _upsert_prediction(store: Dict[Tuple[str, str], Prediction], item: Prediction) -> None:
    key = (item.ticker, item.session_date)
    existing = store.get(key)
    if existing is None or item.generated_at >= existing.generated_at:
        store[key] = item


def load_iris_predictions(data_dir: Path, tracked_tickers: List[str]) -> Dict[str, List[Prediction]]:
    by_ticker_raw: Dict[str, Dict[Tuple[str, str], Prediction]] = {
        ticker: {} for ticker in tracked_tickers
    }

    for ticker in tracked_tickers:
        path = data_dir / f"{ticker}_report.json"
        for item in _read_json_as_list(path):
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            market = item.get("market") if isinstance(item.get("market"), dict) else {}

            symbol = _canonical_ticker(meta.get("symbol"))
            if symbol != ticker:
                continue

            current_price = _safe_float(market.get("current_price"))
            predicted_price = _safe_float(market.get("predicted_price_next_session"))
            if current_price is None or predicted_price is None:
                continue

            generated_at = str(meta.get("generated_at", "")).strip()
            session_date = _extract_session_date(meta)
            if not session_date:
                continue

            pred = Prediction(
                ticker=ticker,
                session_date=session_date,
                generated_at=generated_at,
                current_price=current_price,
                predicted_price_next_session=predicted_price,
            )
            _upsert_prediction(by_ticker_raw[ticker], pred)

    return {
        ticker: sorted(rows.values(), key=lambda x: x.session_date)
        for ticker, rows in by_ticker_raw.items()
    }


def build_actual_direction_map(iris_by_ticker: Dict[str, List[Prediction]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {ticker: {} for ticker in iris_by_ticker.keys()}
    for ticker in iris_by_ticker.keys():
        rows = iris_by_ticker.get(ticker, [])
        for idx in range(len(rows) - 1):
            current_row = rows[idx]
            next_row = rows[idx + 1]
            out[ticker][current_row.session_date] = _direction(next_row.current_price - current_row.current_price)
    return out


def load_llm_predictions(path: Path, tracked_tickers: List[str]) -> List[Prediction]:
    store: Dict[Tuple[str, str], Prediction] = {}
    tracked_set = set(tracked_tickers)

    for item in _read_json_as_list(path):
        parent_meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        parent_generated_at = str(
            item.get("generated_time")
            or item.get("generated_at")
            or parent_meta.get("generated_at")
            or ""
        ).strip()
        parent_session_date = _extract_session_date(parent_meta, parent_generated_at)

        def add_record(
            symbol: object,
            session_date: str,
            generated_at: str,
            current_price: object,
            predicted_price: object,
        ) -> None:
            ticker = _canonical_ticker(symbol)
            if ticker not in tracked_set or not session_date:
                return
            current_value = _safe_float(current_price)
            predicted_value = _safe_float(predicted_price)
            if current_value is None or predicted_value is None:
                return
            pred = Prediction(
                ticker=ticker,
                session_date=session_date[:10],
                generated_at=generated_at,
                current_price=current_value,
                predicted_price_next_session=predicted_value,
            )
            _upsert_prediction(store, pred)

        next_session_predictions = item.get("next_session_predictions")
        if isinstance(next_session_predictions, list):
            for row in next_session_predictions:
                if not isinstance(row, dict):
                    continue
                row_session = str(row.get("market_session_date", "")).strip()[:10] or parent_session_date
                row_generated = str(row.get("generated_at", "")).strip() or parent_generated_at
                add_record(
                    symbol=row.get("ticker"),
                    session_date=row_session,
                    generated_at=row_generated,
                    current_price=row.get("current_price"),
                    predicted_price=row.get("predicted_price_next_session"),
                )
            continue

        stocks = item.get("stocks")
        if isinstance(stocks, list):
            for row in stocks:
                if not isinstance(row, dict):
                    continue
                row_session = str(row.get("market_session_date", "")).strip()[:10] or parent_session_date
                row_generated = str(row.get("generated_at", "")).strip() or parent_generated_at
                add_record(
                    symbol=row.get("ticker"),
                    session_date=row_session,
                    generated_at=row_generated,
                    current_price=row.get("current_price"),
                    predicted_price=row.get("predicted_price_next_session"),
                )
            continue

        market = item.get("market")
        if isinstance(parent_meta, dict) and isinstance(market, dict):
            add_record(
                symbol=parent_meta.get("symbol"),
                session_date=_extract_session_date(parent_meta, parent_generated_at),
                generated_at=str(parent_meta.get("generated_at", "")).strip() or parent_generated_at,
                current_price=market.get("current_price"),
                predicted_price=market.get("predicted_price_next_session"),
            )

    return sorted(store.values(), key=lambda x: (x.ticker, x.session_date))


def evaluate_model_accuracy(
    predictions: List[Prediction],
    actual_direction_map: Dict[str, Dict[str, int]],
    tracked_tickers: List[str],
) -> dict:
    per_ticker = {
        ticker: {"correct": 0, "total": 0, "accuracy": None}
        for ticker in tracked_tickers
    }
    overall_correct = 0
    overall_total = 0

    for row in predictions:
        actual_direction = actual_direction_map.get(row.ticker, {}).get(row.session_date)
        if actual_direction is None:
            continue

        predicted_direction = _direction(row.predicted_price_next_session - row.current_price)
        is_correct = predicted_direction == actual_direction

        overall_total += 1
        per_ticker[row.ticker]["total"] += 1
        if is_correct:
            overall_correct += 1
            per_ticker[row.ticker]["correct"] += 1

    for ticker in tracked_tickers:
        total = per_ticker[ticker]["total"]
        if total > 0:
            per_ticker[ticker]["accuracy"] = per_ticker[ticker]["correct"] / total

    overall_accuracy = (overall_correct / overall_total) if overall_total > 0 else None

    return {
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": overall_accuracy,
        },
        "per_ticker": per_ticker,
    }


def _format_ticker_set_label(tracked_tickers: List[str]) -> str:
    if len(tracked_tickers) <= 6:
        return ", ".join(tracked_tickers)
    preview = ", ".join(tracked_tickers[:6])
    return f"{len(tracked_tickers)} tickers ({preview}, ...)"


def draw_accuracy_chart(
    stats_by_model: dict,
    output_path: Path,
    date_range_label: str,
    tracked_tickers: List[str],
) -> None:
    labels = tracked_tickers + ["OVERALL"]
    model_ids = list(MODEL_CONFIG.keys())
    x = np.arange(len(labels), dtype=float)
    width = min(0.22, 0.8 / max(len(model_ids), 1))

    fig_width = max(13.5, 5.0 + 0.85 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    for idx, model_id in enumerate(model_ids):
        model_stats = stats_by_model[model_id]
        acc_values = []
        for ticker in tracked_tickers:
            ticker_acc = model_stats["per_ticker"][ticker]["accuracy"]
            acc_values.append(ticker_acc * 100 if ticker_acc is not None else 0.0)
        overall_acc = model_stats["overall"]["accuracy"]
        acc_values.append(overall_acc * 100 if overall_acc is not None else 0.0)

        offsets = x + (idx - (len(model_ids) - 1) / 2.0) * width
        bars = ax.bar(
            offsets,
            acc_values,
            width=width,
            color=MODEL_COLORS.get(model_id, "#777777"),
            label=MODEL_CONFIG[model_id]["label"],
            alpha=0.9,
        )

        raw_accs = [model_stats["per_ticker"][ticker]["accuracy"] for ticker in tracked_tickers] + [overall_acc]
        for bar, raw_acc in zip(bars, raw_accs):
            x_pos = bar.get_x() + bar.get_width() / 2.0
            if raw_acc is None:
                ax.text(x_pos, 1.2, "N/A", ha="center", va="bottom", fontsize=8, rotation=90)
            else:
                ax.text(x_pos, bar.get_height() + 1.2, f"{raw_acc * 100:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if len(labels) > 8 or any(len(label) > 5 for label in labels):
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel("Trend Direction Accuracy (%)")
    ax.set_title(
        f"Trend Prediction Accuracy: IRIS vs LLMs ({_format_ticker_set_label(tracked_tickers)})\n"
        f"Session range: {date_range_label}"
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_overall_accuracy_chart(
    stats_by_model: dict,
    output_path: Path,
    date_range_label: str,
) -> None:
    model_ids = list(MODEL_CONFIG.keys())
    labels = [MODEL_CONFIG[m]["label"] for m in model_ids]
    colors = [MODEL_COLORS.get(m, "#777777") for m in model_ids]

    acc_values = []
    raw_accs = []
    counts = []
    for model_id in model_ids:
        overall = stats_by_model[model_id]["overall"]
        acc = overall["accuracy"]
        raw_accs.append(acc)
        acc_values.append(acc * 100 if acc is not None else 0.0)
        counts.append((overall["correct"], overall["total"]))

    x = np.arange(len(model_ids), dtype=float)
    width = 0.45

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, acc_values, width=width, color=colors, alpha=0.9)

    for bar, acc, (correct, total) in zip(bars, raw_accs, counts):
        x_pos = bar.get_x() + bar.get_width() / 2.0
        if acc is None:
            ax.text(x_pos, 1.5, "N/A", ha="center", va="bottom", fontsize=10)
        else:
            ax.text(
                x_pos, bar.get_height() + 1.5,
                f"{acc * 100:.1f}%\n({correct}/{total})",
                ha="center", va="bottom", fontsize=9, linespacing=1.4,
            )

    ax.axhline(50, color="grey", linewidth=1, linestyle="--", alpha=0.6, label="50% baseline")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Trend Direction Accuracy (%)")
    ax.set_title(
        f"Overall Trend Prediction Accuracy: IRIS vs LLMs\nSession range: {date_range_label}"
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summary_json(
    stats_by_model: dict,
    output_png: Path,
    selected_start_date: Optional[str],
    selected_end_date: Optional[str],
    actual_dates: List[str],
    tracked_tickers: List[str],
) -> Path:
    actual_start = actual_dates[0] if actual_dates else None
    actual_end = actual_dates[-1] if actual_dates else None
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tickers": tracked_tickers,
            "selected_range": {
                "start_date": selected_start_date,
                "end_date": selected_end_date,
                "label": _format_date_range_label(selected_start_date, selected_end_date),
            },
            "effective_actual_direction_range": {
                "start_date": actual_start,
                "end_date": actual_end,
            },
            "definition": "Accuracy = predicted next-session direction matches actual next-session close direction.",
        },
        "models": {
            model_id: {
                "label": MODEL_CONFIG[model_id]["label"],
                "overall": stats_by_model[model_id]["overall"],
                "per_ticker": stats_by_model[model_id]["per_ticker"],
            }
            for model_id in MODEL_CONFIG.keys()
        },
    }

    output_json = output_png.with_suffix(".json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_json


def print_summary(stats_by_model: dict, tracked_tickers: List[str]) -> None:
    print("\nTrend Direction Accuracy Summary")
    print("Ticker set:", ", ".join(tracked_tickers))
    print("-" * 72)
    for model_id, cfg in MODEL_CONFIG.items():
        overall = stats_by_model[model_id]["overall"]
        accuracy = overall["accuracy"]
        accuracy_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
        print(
            f"{cfg['label']:<14}  Overall: {accuracy_text:>8}  "
            f"({overall['correct']}/{overall['total']})"
        )


def resolve_default_output(data_dir: Path) -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return data_dir / "charts" / date_str / "trend_accuracy_comparison_chart.png"


def resolve_default_data_dir() -> Path:
    return PROJECT_ROOT / "runtime_data"


def resolve_llm_base_dir(data_dir: Path, llm_data_dir_arg: str) -> Path:
    explicit = str(llm_data_dir_arg or "").strip()
    if explicit:
        base = Path(explicit).expanduser()
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base

    runtime_candidate = data_dir / "LLM reports"
    if runtime_candidate.exists():
        return data_dir

    repo_candidate = PROJECT_ROOT / "data" / "LLM reports"
    if repo_candidate.exists():
        return PROJECT_ROOT / "data"

    return data_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw IRIS vs LLM trend prediction accuracy chart from IRIS JSON reports."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Path to IRIS reports directory (default: active runtime dir from environment).",
    )
    parser.add_argument(
        "--llm-data-dir",
        type=str,
        default="",
        help=(
            "Optional base directory for LLM report files (expects 'LLM reports/*.json'). "
            "Default auto-fallback: data-dir, then project data/."
        ),
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma/space-separated ticker list. Overrides watchlist/discovery.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path. Default: data/charts/<UTC-date>/trend_accuracy_comparison_chart.png",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="",
        help="Inclusive start session date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="",
        help="Inclusive end session date (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else resolve_default_data_dir()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    llm_base_dir = resolve_llm_base_dir(data_dir, args.llm_data_dir)
    output_path = Path(args.output) if args.output else resolve_default_output(data_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    try:
        start_date = _normalize_date_arg(args.start_date, "--start-date")
        end_date = _normalize_date_arg(args.end_date, "--end-date")
    except ValueError as exc:
        parser.error(str(exc))
    if start_date and end_date and start_date > end_date:
        parser.error("--start-date must be <= --end-date")
    print(f"IRIS reports dir: {data_dir}")
    print(f"LLM reports base dir: {llm_base_dir}")

    tracked_tickers = resolve_tracked_tickers(data_dir, args.tickers)
    if not tracked_tickers:
        parser.error("No tickers resolved. Use --tickers or provide watchlist/report files.")

    selected_range_label = _format_date_range_label(start_date, end_date)

    iris_by_ticker_all = load_iris_predictions(data_dir, tracked_tickers)
    iris_by_ticker = _filter_iris_by_date(iris_by_ticker_all, start_date, end_date)
    actual_directions = build_actual_direction_map(iris_by_ticker)
    actual_dates = _collect_actual_dates(actual_directions, tracked_tickers)
    if not actual_dates:
        print(
            "Note: no completed next-session pairs found in the selected data range. "
            "Run at least two sessions per ticker or use --data-dir pointing to historical archives."
        )

    stats_by_model = {}
    for model_id, cfg in MODEL_CONFIG.items():
        if model_id == "iris":
            iris_predictions: List[Prediction] = []
            for ticker in tracked_tickers:
                iris_predictions.extend(iris_by_ticker.get(ticker, []))
            stats_by_model[model_id] = evaluate_model_accuracy(
                iris_predictions, actual_directions, tracked_tickers
            )
            continue

        relative_path = cfg["relative_path"]
        model_path = (llm_base_dir / relative_path) if relative_path is not None else None
        if model_path is None:
            stats_by_model[model_id] = evaluate_model_accuracy([], actual_directions, tracked_tickers)
            continue
        model_predictions = load_llm_predictions(model_path, tracked_tickers)
        model_predictions = _filter_predictions_by_date(model_predictions, start_date, end_date)
        stats_by_model[model_id] = evaluate_model_accuracy(
            model_predictions, actual_directions, tracked_tickers
        )

    draw_accuracy_chart(stats_by_model, output_path, selected_range_label, tracked_tickers)
    overall_output_path = output_path.with_name(
        output_path.stem.replace("trend_accuracy_comparison_chart", "overall_accuracy_chart") + output_path.suffix
    )
    draw_overall_accuracy_chart(stats_by_model, overall_output_path, selected_range_label)
    summary_path = save_summary_json(
        stats_by_model,
        output_path,
        selected_start_date=start_date,
        selected_end_date=end_date,
        actual_dates=actual_dates,
        tracked_tickers=tracked_tickers,
    )
    print_summary(stats_by_model, tracked_tickers)
    print(f"Date range: {selected_range_label}")
    print(f"\nChart saved: {output_path}")
    print(f"Overall chart saved: {overall_output_path}")
    print(f"Summary JSON saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
