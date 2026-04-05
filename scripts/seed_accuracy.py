#!/usr/bin/env python3
"""Seed Almanac historic accuracy data from local index CSV files.

Run this script from the project root so relative paths resolve against the repo:
    python scripts/seed_accuracy.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PRIMARY_ALMANAC_PATH = Path("data") / "almanac_2026" / "almanac_2026.json"
FALLBACK_ALMANAC_PATH = Path("data") / "almanac_2026" / "almanac_2026_db_dump.json"
OUTPUT_PATH = Path("data") / "almanac_2026" / "accuracy_results.json"

INDEX_CONFIG = {
    "d": {
        "csv_key": "dji",
        "summary_key": "dow",
        "label": "Dow",
        "arg": "dji",
        "default": Path("data") / "historical" / "DJI_daily.csv",
    },
    "s": {
        "csv_key": "sp500",
        "summary_key": "sp500",
        "label": "S&P 500",
        "arg": "sp500",
        "default": Path("data") / "historical" / "GSPC_daily.csv",
    },
    "n": {
        "csv_key": "nasdaq",
        "summary_key": "nasdaq",
        "label": "NASDAQ",
        "arg": "nasdaq",
        "default": Path("data") / "historical" / "IXIC_daily.csv",
    },
}


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return payload


def load_almanac_predictions(project_root: Path) -> dict[str, dict[str, object]]:
    primary_path = project_root / PRIMARY_ALMANAC_PATH
    fallback_path = project_root / FALLBACK_ALMANAC_PATH

    if primary_path.exists():
        payload = read_json(primary_path)
        daily = payload.get("daily", {})
        if isinstance(daily, dict):
            normalized = {}
            for date_key, day in daily.items():
                if not isinstance(day, dict):
                    continue
                normalized[str(date_key)] = {
                    "d": float(day.get("d", 0.0)),
                    "s": float(day.get("s", 0.0)),
                    "n": float(day.get("n", 0.0)),
                    "context": str(day.get("notes", "") or "").strip(),
                }
            if normalized:
                return normalized

    if fallback_path.exists():
        payload = read_json(fallback_path)
        table = payload.get("daily_probabilities", {})
        rows = table.get("rows", []) if isinstance(table, dict) else []
        if isinstance(rows, list):
            normalized = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                date_key = str(row.get("date", "")).strip()
                if not date_key:
                    continue
                normalized[date_key] = {
                    "d": float(row.get("dow_prob", 0.0)),
                    "s": float(row.get("sp500_prob", 0.0)),
                    "n": float(row.get("nasdaq_prob", 0.0)),
                    "context": str(row.get("notes", "") or "").strip(),
                }
            if normalized:
                return normalized

    raise FileNotFoundError(
        "No supported almanac source found. Expected "
        f"{primary_path} or {fallback_path}."
    )


def parse_close(value: str) -> float:
    return float(str(value or "").replace(",", "").strip())


def load_history_csv(path: Path) -> dict[str, dict[str, float | None]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing historical CSV: {path}")

    rows: list[tuple[datetime, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Date", "Close"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain Date and Close columns")
        for row in reader:
            date_text = str(row.get("Date", "")).strip()
            if not date_text:
                continue
            try:
                parsed_date = datetime.strptime(date_text, "%m/%d/%Y")
                close_value = parse_close(str(row.get("Close", "")))
            except ValueError as exc:
                raise ValueError(f"Unable to parse row in {path}: {row}") from exc
            rows.append((parsed_date, close_value))

    if not rows:
        raise ValueError(f"{path} did not contain any historical rows")

    rows.sort(key=lambda item: item[0])
    lookup: dict[str, dict[str, float | None]] = {}
    previous_close: float | None = None
    for trade_date, close_value in rows:
        iso_date = trade_date.strftime("%Y-%m-%d")
        lookup[iso_date] = {"close": close_value, "prev_close": previous_close}
        previous_close = close_value
    return lookup


def actual_direction(pct_change: float) -> str:
    if pct_change > 0:
        return "UP"
    if pct_change < 0:
        return "DOWN"
    return "FLAT"


def predicted_direction(probability: float) -> str | None:
    if probability > 50:
        return "UP"
    if probability < 50:
        return "DOWN"
    return None


def score_prediction(probability: float, pct_change: float) -> dict[str, str | None]:
    predicted = predicted_direction(probability)
    actual = actual_direction(pct_change)
    verdict = None

    if predicted == "UP":
        verdict = "HIT" if pct_change > 0 else "MISS"
    elif predicted == "DOWN":
        verdict = "HIT" if pct_change < 0 else "MISS"

    return {"verdict": verdict, "predicted": predicted, "actual": actual}


def pct(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((value / total) * 100, 1)


def build_daily_results(
    almanac_daily: dict[str, dict[str, object]],
    history_by_index: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, object]]:
    daily_results: dict[str, dict[str, object]] = {}

    for date_key in sorted(almanac_daily.keys()):
        current_records = {}
        for config in INDEX_CONFIG.values():
            history = history_by_index[config["csv_key"]]
            current_records[config["csv_key"]] = history.get(date_key)

        if any(record is None or record.get("prev_close") is None for record in current_records.values()):
            continue

        day_predictions = almanac_daily[date_key]
        actuals = {}
        prev_closes = {}
        pct_changes = {}
        results = {}
        hits = 0
        total_calls = 0

        for signal_key, config in INDEX_CONFIG.items():
            csv_key = config["csv_key"]
            record = current_records[csv_key] or {}
            close_value = float(record["close"])
            prev_close = float(record["prev_close"])
            pct_change = (close_value - prev_close) / prev_close
            probability = float(day_predictions.get(signal_key, 0.0))

            actuals[csv_key] = round(close_value, 6)
            prev_closes[csv_key] = round(prev_close, 6)
            pct_changes[csv_key] = round(pct_change, 6)
            results[signal_key] = score_prediction(probability, pct_change)

            if results[signal_key]["verdict"] is not None:
                total_calls += 1
                if results[signal_key]["verdict"] == "HIT":
                    hits += 1

        daily_results[date_key] = {
            "actual": actuals,
            "prev_close": prev_closes,
            "pct_change": pct_changes,
            "almanac_scores": {
                "d": float(day_predictions.get("d", 0.0)),
                "s": float(day_predictions.get("s", 0.0)),
                "n": float(day_predictions.get("n", 0.0)),
            },
            "results": results,
            "hits": hits,
            "total_calls": total_calls,
            "context": str(day_predictions.get("context", "") or "").strip(),
        }

    return daily_results


def aggregate_periods(
    daily_results: dict[str, dict[str, object]],
    key_builder,
    include_dates: bool = False,
    include_trading_days: bool = False,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "dates": [],
            "hits": 0,
            "total_calls": 0,
            "dow": {"hits": 0, "total": 0},
            "sp500": {"hits": 0, "total": 0},
            "nasdaq": {"hits": 0, "total": 0},
        }
    )

    for date_key, day in sorted(daily_results.items()):
        group_key = key_builder(date_key)
        bucket = grouped[group_key]
        bucket["dates"].append(date_key)
        bucket["hits"] += int(day.get("hits", 0))
        bucket["total_calls"] += int(day.get("total_calls", 0))

        for signal_key, config in INDEX_CONFIG.items():
            result = (day.get("results", {}) or {}).get(signal_key, {})
            verdict = result.get("verdict")
            if verdict is None:
                continue
            summary_bucket = bucket[config["summary_key"]]
            summary_bucket["total"] += 1
            if verdict == "HIT":
                summary_bucket["hits"] += 1

    summarized: dict[str, dict[str, object]] = {}
    for group_key, bucket in sorted(grouped.items()):
        record: dict[str, object] = {
            "hits": bucket["hits"],
            "total_calls": bucket["total_calls"],
            "accuracy": pct(bucket["hits"], bucket["total_calls"]),
        }
        if include_dates:
            record["dates"] = bucket["dates"]
        for index_key in ("dow", "sp500", "nasdaq"):
            index_bucket = bucket[index_key]
            record[index_key] = {
                "hits": index_bucket["hits"],
                "total": index_bucket["total"],
                "pct": pct(index_bucket["hits"], index_bucket["total"]),
            }
        if include_trading_days:
            record["trading_days"] = len(bucket["dates"])
        summarized[group_key] = record

    return summarized


def build_output(daily_results: dict[str, dict[str, object]]) -> dict[str, object]:
    weekly = aggregate_periods(
        daily_results,
        key_builder=lambda date_key: datetime.strptime(date_key, "%Y-%m-%d").strftime("%Y-W%W"),
        include_dates=True,
    )
    monthly = aggregate_periods(
        daily_results,
        key_builder=lambda date_key: date_key[:7],
        include_trading_days=True,
    )

    sorted_dates = sorted(daily_results.keys())
    return {
        "meta": {
            "last_updated": iso_utc_now(),
            "total_days_scored": len(sorted_dates),
            "data_range": {
                "from": sorted_dates[0] if sorted_dates else None,
                "to": sorted_dates[-1] if sorted_dates else None,
            },
            "source": "Historic CSV backtest via scripts/seed_accuracy.py",
        },
        "daily": daily_results,
        "weekly": weekly,
        "monthly": monthly,
    }


def format_score(hits: int, total: int) -> str:
    if total <= 0:
        return "0/0 (--%)"
    return f"{hits}/{total} ({round((hits / total) * 100):.0f}%)"


def print_summary(output: dict[str, object]) -> None:
    monthly = output.get("monthly", {})
    if not isinstance(monthly, dict):
        return

    print("=== 2026 Almanac Accuracy Backtest ===")
    print(f"{'Month':<10} {'Dow':<15} {'S&P 500':<15} {'NASDAQ':<15} {'All':<15}")

    total_hits = 0
    total_calls = 0
    per_index_totals = {
        "dow": {"hits": 0, "total": 0},
        "sp500": {"hits": 0, "total": 0},
        "nasdaq": {"hits": 0, "total": 0},
    }

    for month_key in sorted(monthly.keys()):
        month_data = monthly[month_key]
        month_name = datetime.strptime(month_key + "-01", "%Y-%m-%d").strftime("%B")
        total_hits += int(month_data.get("hits", 0))
        total_calls += int(month_data.get("total_calls", 0))
        for index_key in per_index_totals:
            per_index_totals[index_key]["hits"] += int(month_data.get(index_key, {}).get("hits", 0))
            per_index_totals[index_key]["total"] += int(month_data.get(index_key, {}).get("total", 0))

        print(
            f"{month_name:<10} "
            f"{format_score(month_data['dow']['hits'], month_data['dow']['total']):<15} "
            f"{format_score(month_data['sp500']['hits'], month_data['sp500']['total']):<15} "
            f"{format_score(month_data['nasdaq']['hits'], month_data['nasdaq']['total']):<15} "
            f"{format_score(month_data['hits'], month_data['total_calls']):<15}"
        )

    total_label = "Q1 Total" if set(monthly.keys()).issubset({"2026-01", "2026-02", "2026-03"}) else "YTD Total"
    print(
        f"{total_label:<10} "
        f"{format_score(per_index_totals['dow']['hits'], per_index_totals['dow']['total']):<15} "
        f"{format_score(per_index_totals['sp500']['hits'], per_index_totals['sp500']['total']):<15} "
        f"{format_score(per_index_totals['nasdaq']['hits'], per_index_totals['nasdaq']['total']):<15} "
        f"{format_score(total_hits, total_calls):<15}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Almanac historic accuracy results from local CSV data.")
    parser.add_argument("--dji", type=Path, default=INDEX_CONFIG["d"]["default"])
    parser.add_argument("--sp500", type=Path, default=INDEX_CONFIG["s"]["default"])
    parser.add_argument("--nasdaq", type=Path, default=INDEX_CONFIG["n"]["default"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()

    try:
        almanac_daily = load_almanac_predictions(project_root)
        history_by_index = {
            "dji": load_history_csv(project_root / Path(args.dji)),
            "sp500": load_history_csv(project_root / Path(args.sp500)),
            "nasdaq": load_history_csv(project_root / Path(args.nasdaq)),
        }
        daily_results = build_daily_results(almanac_daily, history_by_index)
        output = build_output(daily_results)

        output_path = project_root / OUTPUT_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
            handle.write("\n")

        print_summary(output)
        print(f"Wrote {output_path}")
        return 0
    except Exception as exc:
        print(f"[seed_accuracy] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
