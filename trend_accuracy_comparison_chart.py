"""
Draw a trend-direction accuracy comparison chart for IRIS and three LLMs.

Data sources:
- data/<TICKER>_report.json (IRIS reports)
- data/LLM reports/chatgpt_5.2.json
- data/LLM reports/deepseek_v3.json
- data/LLM reports/gemini_v3_pro.json

Default tracked tickers:
AMZN, NVDA, TSLA, AAPL, GOOG
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRACKED_TICKERS = ["AMZN", "NVDA", "TSLA", "AAPL", "GOOG"]
TICKER_ALIASES = {"GOOGL": "GOOG"}

MODEL_CONFIG = {
    "iris": {"label": "IRIS", "path": None},
    "chatgpt_5_2": {"label": "ChatGPT 5.2", "path": Path("data") / "LLM reports" / "chatgpt_5.2.json"},
    "deepseek_v3": {"label": "DeepSeek V3", "path": Path("data") / "LLM reports" / "deepseek_v3.json"},
    "gemini_v3_pro": {"label": "Gemini V3 Pro", "path": Path("data") / "LLM reports" / "gemini_v3_pro.json"},
}

MODEL_COLORS = {
    "iris": "#1f77b4",
    "chatgpt_5_2": "#ff7f0e",
    "deepseek_v3": "#2ca02c",
    "gemini_v3_pro": "#d62728",
}


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


def _collect_actual_dates(actual_direction_map: Dict[str, Dict[str, int]]) -> List[str]:
    dates = sorted(
        {
            session_date
            for ticker in TRACKED_TICKERS
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


def load_iris_predictions(data_dir: Path) -> Dict[str, List[Prediction]]:
    by_ticker_raw: Dict[str, Dict[Tuple[str, str], Prediction]] = {
        ticker: {} for ticker in TRACKED_TICKERS
    }

    for ticker in TRACKED_TICKERS:
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
    out: Dict[str, Dict[str, int]] = {ticker: {} for ticker in TRACKED_TICKERS}
    for ticker in TRACKED_TICKERS:
        rows = iris_by_ticker.get(ticker, [])
        for idx in range(len(rows) - 1):
            current_row = rows[idx]
            next_row = rows[idx + 1]
            out[ticker][current_row.session_date] = _direction(next_row.current_price - current_row.current_price)
    return out


def load_llm_predictions(path: Path) -> List[Prediction]:
    store: Dict[Tuple[str, str], Prediction] = {}

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
            if ticker not in TRACKED_TICKERS or not session_date:
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
) -> dict:
    per_ticker = {
        ticker: {"correct": 0, "total": 0, "accuracy": None}
        for ticker in TRACKED_TICKERS
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

    for ticker in TRACKED_TICKERS:
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


def draw_accuracy_chart(stats_by_model: dict, output_path: Path, date_range_label: str) -> None:
    labels = TRACKED_TICKERS + ["OVERALL"]
    model_ids = list(MODEL_CONFIG.keys())
    x = np.arange(len(labels), dtype=float)
    width = 0.18

    fig, ax = plt.subplots(figsize=(13.5, 7))

    for idx, model_id in enumerate(model_ids):
        model_stats = stats_by_model[model_id]
        acc_values = []
        for ticker in TRACKED_TICKERS:
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

        raw_accs = [model_stats["per_ticker"][ticker]["accuracy"] for ticker in TRACKED_TICKERS] + [overall_acc]
        for bar, raw_acc in zip(bars, raw_accs):
            x_pos = bar.get_x() + bar.get_width() / 2.0
            if raw_acc is None:
                ax.text(x_pos, 1.2, "N/A", ha="center", va="bottom", fontsize=8, rotation=90)
            else:
                ax.text(x_pos, bar.get_height() + 1.2, f"{raw_acc * 100:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Trend Direction Accuracy (%)")
    ax.set_title(
        "Trend Prediction Accuracy: IRIS vs LLMs (AMZN, NVDA, TSLA, AAPL, GOOG)\n"
        f"Session range: {date_range_label}"
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(loc="upper right")
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
) -> Path:
    actual_start = actual_dates[0] if actual_dates else None
    actual_end = actual_dates[-1] if actual_dates else None
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tickers": TRACKED_TICKERS,
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


def print_summary(stats_by_model: dict) -> None:
    print("\nTrend Direction Accuracy Summary")
    print("Ticker set:", ", ".join(TRACKED_TICKERS))
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw IRIS vs LLM trend prediction accuracy chart from /data JSON reports."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data).",
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

    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else resolve_default_output(data_dir)
    try:
        start_date = _normalize_date_arg(args.start_date, "--start-date")
        end_date = _normalize_date_arg(args.end_date, "--end-date")
    except ValueError as exc:
        parser.error(str(exc))
    if start_date and end_date and start_date > end_date:
        parser.error("--start-date must be <= --end-date")

    selected_range_label = _format_date_range_label(start_date, end_date)

    iris_by_ticker_all = load_iris_predictions(data_dir)
    iris_by_ticker = _filter_iris_by_date(iris_by_ticker_all, start_date, end_date)
    actual_directions = build_actual_direction_map(iris_by_ticker)
    actual_dates = _collect_actual_dates(actual_directions)

    stats_by_model = {}
    for model_id, cfg in MODEL_CONFIG.items():
        if model_id == "iris":
            iris_predictions: List[Prediction] = []
            for ticker in TRACKED_TICKERS:
                iris_predictions.extend(iris_by_ticker.get(ticker, []))
            stats_by_model[model_id] = evaluate_model_accuracy(iris_predictions, actual_directions)
            continue

        model_path = cfg["path"]
        if model_path is None:
            stats_by_model[model_id] = evaluate_model_accuracy([], actual_directions)
            continue
        model_predictions = load_llm_predictions(model_path)
        model_predictions = _filter_predictions_by_date(model_predictions, start_date, end_date)
        stats_by_model[model_id] = evaluate_model_accuracy(model_predictions, actual_directions)

    draw_accuracy_chart(stats_by_model, output_path, selected_range_label)
    summary_path = save_summary_json(
        stats_by_model,
        output_path,
        selected_start_date=start_date,
        selected_end_date=end_date,
        actual_dates=actual_dates,
    )
    print_summary(stats_by_model)
    print(f"Date range: {selected_range_label}")
    print(f"\nChart saved: {output_path}")
    print(f"Summary JSON saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
