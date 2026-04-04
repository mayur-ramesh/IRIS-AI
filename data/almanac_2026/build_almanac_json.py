from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "almanac_2026.json"
DB_DUMP_OUTPUT_PATH = BASE_DIR / "almanac_2026_db_dump.json"
SOURCE_DEFAULT = "Stock Trader's Almanac 2026 (Wiley)"
TARGET_YEAR = 2026

EXPECTED_MONTHS = [
    ("2026-01", 1, "January", "almanac_2026_01_january.md"),
    ("2026-02", 2, "February", "almanac_2026_02_february.md"),
    ("2026-03", 3, "March", "almanac_2026_03_march.md"),
    ("2026-04", 4, "April", "almanac_2026_04_april.md"),
    ("2026-05", 5, "May", "almanac_2026_05_may.md"),
    ("2026-06", 6, "June", "almanac_2026_06_june.md"),
    ("2026-07", 7, "July", "almanac_2026_07_july.md"),
    ("2026-08", 8, "August", "almanac_2026_08_august.md"),
    ("2026-09", 9, "September", "almanac_2026_09_september.md"),
    ("2026-10", 10, "October", "almanac_2026_10_october.md"),
    ("2026-11", 11, "November", "almanac_2026_11_november.md"),
    ("2026-12", 12, "December", "almanac_2026_12_december.md"),
]

INDEX_KEY_MAP = {
    "djia": "dow",
    "dow": "dow",
    "dow jones industrial average": "dow",
    "s&p 500": "sp500",
    "sp500": "sp500",
    "s&p500": "sp500",
    "nasdaq": "nasdaq",
}

INDEX_LABEL_MAP = {
    "dow": "DJIA",
    "sp500": "S&P 500",
    "nasdaq": "NASDAQ",
}

UNICODE_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2192": "->",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u00a0": " ",
}


def _normalize_text(text: str) -> str:
    normalized = text
    for src, dst in UNICODE_REPLACEMENTS.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _strip_markdown(text: str) -> str:
    cleaned = _normalize_text(text)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text

    end_marker = text.find("\n---\n", 4)
    if end_marker == -1:
        return {}, text

    frontmatter_text = text[4:end_marker]
    body = text[end_marker + 5 :]
    data = {}
    for line in frontmatter_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _strip_markdown(value.strip())
    return data, body


def _parse_sections(text: str) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            if current is not None:
                current["content"] = "\n".join(current["lines"]).strip()
                current.pop("lines", None)
                sections.append(current)
            current = {
                "level": len(heading_match.group(1)),
                "title": _strip_markdown(heading_match.group(2).strip()),
                "lines": [],
            }
            continue
        if current is not None:
            current["lines"].append(line)

    if current is not None:
        current["content"] = "\n".join(current["lines"]).strip()
        current.pop("lines", None)
        sections.append(current)
    return sections


def _find_section(sections: list[dict[str, object]], *needles: str) -> dict[str, object] | None:
    lowered_needles = [needle.lower() for needle in needles]
    for section in sections:
        title = str(section.get("title", "")).lower()
        if all(needle in title for needle in lowered_needles):
            return section
    return None


def _extract_table_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append(stripped)
        else:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    return blocks


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _parse_table(block: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    rows = [_split_table_row(line) for line in block]
    header = rows[0]
    data_rows: list[dict[str, str]] = []
    for row in rows[1:]:
        if _is_separator_row(row):
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        data_rows.append(dict(zip(header, row)))
    return header, data_rows


def _find_matching_table(
    content: str, required_headers: set[str], first_headers: tuple[str, ...]
) -> tuple[list[str], list[dict[str, str]]]:
    for block in _extract_table_blocks(content):
        header, rows = _parse_table(block)
        normalized_header = {_strip_markdown(cell) for cell in header}
        if not required_headers.issubset(normalized_header):
            continue
        if tuple(_strip_markdown(cell) for cell in header[: len(first_headers)]) != first_headers:
            continue
        return header, rows
    raise ValueError(f"Could not find table with headers {sorted(required_headers)}")


def _compact_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        stripped = _strip_markdown(raw_line).strip()
        if stripped:
            lines.append(stripped)
    return lines


def _first_two_sentences(text: str) -> str:
    normalized = _strip_markdown(text)
    if not normalized:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if len(sentences) >= 2:
        return " ".join(sentences[:2]).strip()
    return sentences[0] if sentences else ""


def _extract_overview(sections: list[dict[str, object]]) -> str:
    section = _find_section(sections, "overview")
    if section is None:
        return ""
    content = str(section.get("content", ""))
    bullet_lines = [
        _strip_markdown(re.sub(r"^\s*-\s*", "", line))
        for line in content.splitlines()
        if line.strip().startswith("-")
    ]
    if bullet_lines:
        pieces = bullet_lines[:2]
        return " ".join(piece if piece.endswith((".", "!", "?")) else f"{piece}." for piece in pieces)
    return _first_two_sentences(content)


def _parse_float(value: str) -> float:
    normalized = _normalize_text(value).replace("%", "").replace("+", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        raise ValueError(f"Could not parse float from {value!r}")
    return float(match.group(0))


def _parse_int(value: str) -> int:
    match = re.search(r"\d+", _normalize_text(value))
    if not match:
        raise ValueError(f"Could not parse int from {value!r}")
    return int(match.group(0))


def _derive_dir(score: float) -> str:
    if score >= 60.0:
        return "D"
    if score <= 40.0:
        return "N"
    return "S"


def _normalize_icon(value: str) -> str | None:
    icon = _strip_markdown(value).lower()
    if icon in {"bull", "bear", "bull_bear"}:
        return icon
    return None


def _clean_notes(value: str) -> str:
    return _strip_markdown(value)


def _extract_daily_entries(
    sections: list[dict[str, object]], month_key: str, source_month_num: int
) -> list[tuple[str, dict[str, object], bool]]:
    header = None
    rows = None
    for candidate in sections:
        title = str(candidate.get("title", "")).lower()
        if "daily probability scores" not in title and "daily d/s/n probability scores" not in title:
            continue
        try:
            header, rows = _find_matching_table(
                str(candidate.get("content", "")),
                required_headers={"Date", "Day", "D", "S", "N"},
                first_headers=("Date", "Day"),
            )
            break
        except ValueError:
            continue
    if header is None or rows is None:
        raise ValueError(f"Missing daily probability scores table for {month_key}")

    header_lookup = {_strip_markdown(cell): cell for cell in header}

    entries: list[tuple[str, dict[str, object], bool]] = []
    for row in rows:
        raw_date = row.get(header_lookup["Date"], "").strip()
        if not raw_date.startswith(f"{TARGET_YEAR}-"):
            continue

        try:
            d_score = _parse_float(row.get(header_lookup["D"], ""))
            s_score = _parse_float(row.get(header_lookup["S"], ""))
            n_score = _parse_float(row.get(header_lookup["N"], ""))
        except ValueError:
            continue

        date_month = int(raw_date[5:7])
        date_matches_month = date_month == source_month_num
        day_value = _strip_markdown(row.get(header_lookup["Day"], "")).upper()

        d_dir = (
            _strip_markdown(row.get(header_lookup.get("D Dir", ""), "")).upper()
            if "D Dir" in header_lookup
            else _derive_dir(d_score)
        )
        s_dir = (
            _strip_markdown(row.get(header_lookup.get("S Dir", ""), "")).upper()
            if "S Dir" in header_lookup
            else _derive_dir(s_score)
        )
        n_dir = (
            _strip_markdown(row.get(header_lookup.get("N Dir", ""), "")).upper()
            if "N Dir" in header_lookup
            else _derive_dir(n_score)
        )

        entry = {
            "date": raw_date,
            "day": day_value[:3],
            "d": d_score,
            "s": s_score,
            "n": n_score,
            "d_dir": d_dir or _derive_dir(d_score),
            "s_dir": s_dir or _derive_dir(s_score),
            "n_dir": n_dir or _derive_dir(n_score),
            "icon": _normalize_icon(row.get(header_lookup.get("Icon", ""), "")) if "Icon" in header_lookup else None,
            "notes": _clean_notes(row.get(header_lookup.get("Notes", ""), "")) if "Notes" in header_lookup else "",
        }
        entries.append((raw_date, entry, date_matches_month))
    return entries


def _normalize_index_name(value: str) -> str | None:
    cleaned = _strip_markdown(value).lower().replace(".", "")
    return INDEX_KEY_MAP.get(cleaned)


def _extract_vital_stats(sections: list[dict[str, object]], month_key: str) -> dict[str, dict[str, object]]:
    section = _find_section(sections, "vital statistics")
    if section is None:
        section = _find_section(sections, "monthly statistics summary")
    if section is None:
        raise ValueError(f"Missing vital stats section for {month_key}")

    header, rows = _find_matching_table(
        str(section.get("content", "")),
        required_headers={"Index", "Rank", "Up", "Down", "Avg % Chg"},
        first_headers=("Index", "Rank"),
    )
    header_lookup = {_strip_markdown(cell): cell for cell in header}
    midterm_column = (
        header_lookup.get("Midterm Yr Avg")
        or header_lookup.get("Midterm Avg")
        or header_lookup.get("Midterm Year Avg")
    )
    if not midterm_column:
        raise ValueError(f"Missing midterm column in vital stats for {month_key}")

    vital_stats: dict[str, dict[str, object]] = {}
    for row in rows:
        index_key = _normalize_index_name(row.get(header_lookup["Index"], ""))
        if index_key not in {"dow", "sp500", "nasdaq"}:
            continue
        vital_stats[index_key] = {
            "rank": _parse_int(row.get(header_lookup["Rank"], "")),
            "up": _parse_int(row.get(header_lookup["Up"], "")),
            "down": _parse_int(row.get(header_lookup["Down"], "")),
            "avg_change": _parse_float(row.get(header_lookup["Avg % Chg"], "")),
            "midterm_avg": _parse_float(row.get(midterm_column, "")),
        }

    missing = {"dow", "sp500", "nasdaq"} - set(vital_stats)
    if missing:
        raise ValueError(f"Missing vital stats rows for {month_key}: {sorted(missing)}")
    return vital_stats


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _strip_markdown(text).lower())
    return slug.strip("_")


def _extract_seasonal_signals(
    sections: list[dict[str, object]], month_key: str
) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    for section in sections:
        title = str(section.get("title", ""))
        match = re.match(r"^(.*?)\s+\[([a-z_]+)\]$", title.strip())
        if not match:
            continue
        label = _strip_markdown(match.group(1))
        signal_type = match.group(2).strip()
        description = _strip_markdown(str(section.get("content", "")))
        if not label or not description:
            continue
        signals.append(
            {
                "id": _slugify(label),
                "label": label,
                "type": signal_type,
                "source_month": month_key,
                "description": description,
            }
        )
    return signals


def _build_heatmap(months: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    month_keys_by_midterm = sorted(
        months.keys(),
        key=lambda key: (
            -float(months[key]["vital_stats"]["sp500"]["midterm_avg"]),
            key,
        ),
    )
    midterm_ranks = {month_key: idx + 1 for idx, month_key in enumerate(month_keys_by_midterm)}

    heatmap: dict[str, dict[str, object]] = {}
    for month_key, month_data in months.items():
        sp500 = month_data["vital_stats"]["sp500"]
        rank = int(sp500["rank"])
        avg_change = float(sp500["avg_change"])
        midterm_avg = float(sp500["midterm_avg"])
        if rank <= 6 and avg_change > 0 and midterm_avg > -0.5:
            bias = "bullish"
        elif rank >= 9 or midterm_avg < -1.0:
            bias = "bearish"
        else:
            bias = "mixed"
        heatmap[month_key] = {
            "bias": bias,
            "sp500_rank": rank,
            "sp500_avg": avg_change,
            "sp500_midterm": midterm_avg,
            "sp500_midterm_rank": midterm_ranks[month_key],
        }
    return heatmap


def _choose_daily_entry(
    existing: tuple[dict[str, object], bool] | None,
    candidate: tuple[dict[str, object], bool],
) -> tuple[dict[str, object], bool]:
    if existing is None:
        return candidate
    existing_entry, existing_matches = existing
    candidate_entry, candidate_matches = candidate
    if candidate_matches and not existing_matches:
        return candidate
    if existing_matches and not candidate_matches:
        return existing
    return existing_entry, existing_matches


def build_payload() -> dict[str, object]:
    months: dict[str, dict[str, object]] = {}
    daily_candidates: dict[str, tuple[dict[str, object], bool]] = {}
    seasonal_signals: list[dict[str, str]] = []
    source = SOURCE_DEFAULT

    for month_key, month_num, month_name, filename in EXPECTED_MONTHS:
        path = BASE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required almanac file: {path.name}")

        raw_text = path.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(_normalize_text(raw_text))
        if frontmatter.get("source"):
            source = frontmatter["source"]

        sections = _parse_sections(body)
        overview = _extract_overview(sections)
        vital_stats = _extract_vital_stats(sections, month_key)
        daily_entries = _extract_daily_entries(sections, month_key, month_num)
        seasonal_signals.extend(_extract_seasonal_signals(sections, month_key))

        months[month_key] = {
            "name": month_name,
            "month_num": month_num,
            "overview": overview,
            "vital_stats": vital_stats,
        }

        for date_key, entry, date_matches_month in daily_entries:
            daily_candidates[date_key] = _choose_daily_entry(
                daily_candidates.get(date_key),
                (entry, date_matches_month),
            )

    daily = {
        date_key: daily_candidates[date_key][0]
        for date_key in sorted(daily_candidates.keys())
    }

    payload = {
        "meta": {
            "source": source,
            "year": TARGET_YEAR,
            "generated_at": _iso_now(),
        },
        "months": months,
        "daily": daily,
        "seasonal_signals": seasonal_signals,
        "seasonal_heatmap": _build_heatmap(months),
    }
    return payload


def _build_table(columns: list[str], rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "columns": columns,
        "row_count": len(rows),
        "rows": rows,
    }


def build_structured_db_dump(payload: dict[str, object]) -> dict[str, object]:
    meta = dict(payload.get("meta", {}))
    months = dict(payload.get("months", {}))
    daily = dict(payload.get("daily", {}))
    seasonal_signals = list(payload.get("seasonal_signals", []))
    seasonal_heatmap = dict(payload.get("seasonal_heatmap", {}))

    metadata_rows = [
        {"key": "source", "value": meta.get("source", SOURCE_DEFAULT)},
        {"key": "year", "value": int(meta.get("year", TARGET_YEAR))},
        {"key": "generated_at", "value": meta.get("generated_at", _iso_now())},
        {"key": "format", "value": "almanac-json-db-v1"},
    ]

    month_rows: list[dict[str, object]] = []
    vital_rows: list[dict[str, object]] = []
    for month_key in sorted(months.keys()):
        month = dict(months[month_key])
        month_rows.append(
            {
                "month_key": month_key,
                "name": month.get("name", ""),
                "month_num": int(month.get("month_num", 0) or 0),
                "overview": month.get("overview", ""),
            }
        )
        for index_key in ("dow", "sp500", "nasdaq"):
            stats = dict(month.get("vital_stats", {}).get(index_key, {}))
            vital_rows.append(
                {
                    "month_key": month_key,
                    "index_key": index_key,
                    "index_name": INDEX_LABEL_MAP[index_key],
                    "rank": int(stats.get("rank", 0) or 0),
                    "years_up": int(stats.get("up", 0) or 0),
                    "years_down": int(stats.get("down", 0) or 0),
                    "avg_pct_change": float(stats.get("avg_change", 0.0) or 0.0),
                    "midterm_yr_avg": float(stats.get("midterm_avg", 0.0) or 0.0),
                }
            )

    daily_rows = []
    for date_key in sorted(daily.keys()):
        entry = dict(daily[date_key])
        daily_rows.append(
            {
                "date": date_key,
                "day_of_week": entry.get("day", ""),
                "dow_prob": float(entry.get("d", 0.0) or 0.0),
                "sp500_prob": float(entry.get("s", 0.0) or 0.0),
                "nasdaq_prob": float(entry.get("n", 0.0) or 0.0),
                "dow_dir": entry.get("d_dir", ""),
                "sp500_dir": entry.get("s_dir", ""),
                "nasdaq_dir": entry.get("n_dir", ""),
                "icon": entry.get("icon"),
                "notes": entry.get("notes", ""),
            }
        )

    signal_rows = []
    for signal in seasonal_signals:
        signal_rows.append(
            {
                "id": signal.get("id", ""),
                "label": signal.get("label", ""),
                "type": signal.get("type", ""),
                "source_month": signal.get("source_month", ""),
                "description": signal.get("description", ""),
            }
        )

    heatmap_rows = []
    for month_key in sorted(seasonal_heatmap.keys()):
        entry = dict(seasonal_heatmap[month_key])
        heatmap_rows.append(
            {
                "month_key": month_key,
                "bias": entry.get("bias", ""),
                "sp500_rank": int(entry.get("sp500_rank", 0) or 0),
                "sp500_avg": float(entry.get("sp500_avg", 0.0) or 0.0),
                "sp500_midterm": float(entry.get("sp500_midterm", 0.0) or 0.0),
                "sp500_midterm_rank": int(entry.get("sp500_midterm_rank", 0) or 0),
            }
        )

    tables = {
        "metadata": _build_table(["key", "value"], metadata_rows),
        "months": _build_table(["month_key", "name", "month_num", "overview"], month_rows),
        "vital_statistics": _build_table(
            [
                "month_key",
                "index_key",
                "index_name",
                "rank",
                "years_up",
                "years_down",
                "avg_pct_change",
                "midterm_yr_avg",
            ],
            vital_rows,
        ),
        "daily_probabilities": _build_table(
            [
                "date",
                "day_of_week",
                "dow_prob",
                "sp500_prob",
                "nasdaq_prob",
                "dow_dir",
                "sp500_dir",
                "nasdaq_dir",
                "icon",
                "notes",
            ],
            daily_rows,
        ),
        "seasonal_signals": _build_table(
            ["id", "label", "type", "source_month", "description"],
            signal_rows,
        ),
        "seasonal_heatmap": _build_table(
            [
                "month_key",
                "bias",
                "sp500_rank",
                "sp500_avg",
                "sp500_midterm",
                "sp500_midterm_rank",
            ],
            heatmap_rows,
        ),
    }
    return {
        "_meta": {
            "format": "almanac-json-db-v1",
            "source": meta.get("source", SOURCE_DEFAULT),
            "year": int(meta.get("year", TARGET_YEAR) or TARGET_YEAR),
            "generated_at": meta.get("generated_at", _iso_now()),
            "tables": list(tables.keys()),
        },
        **tables,
    }


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    db_dump = build_structured_db_dump(payload)
    DB_DUMP_OUTPUT_PATH.write_text(json.dumps(db_dump, indent=2), encoding="utf-8")
    print(
        "Generated almanac JSON:",
        f"months={len(payload['months'])}",
        f"daily={len(payload['daily'])}",
        f"signals={len(payload['seasonal_signals'])}",
    )
    print(
        "Generated almanac JSON DB:",
        f"tables={len(db_dump['_meta']['tables'])}",
        f"path={DB_DUMP_OUTPUT_PATH.name}",
    )


if __name__ == "__main__":
    main()
