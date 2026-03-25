"""
Generate prediction data for tickers in watchlist.txt using three LLMs:
- ChatGPT 5.2 (OpenAI-compatible)
- DeepSeek V3
- Gemini V3 Pro

Results are appended to the existing JSON report files in data/LLM reports/,
using the same schema as gemini_v3_pro.json.

To run the script, type in the terminal:
    python generate_llm_reports.py
"""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from iris_mvp import IRIS_System


PROJECT_ROOT = Path(__file__).resolve().parent
LLM_REPORTS_DIR = PROJECT_ROOT / "data" / "LLM reports"
WATCHLIST_PATH = PROJECT_ROOT / "watchlist.txt"

_TICKER_ALIASES = {
    "GOOGL": "GOOG",
}


def _load_env():
    """
    Load environment variables from .env at project root.

    Tries python-dotenv if available; otherwise falls back to a simple parser.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    # First try python-dotenv if installed.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path)
        return
    except Exception:
        # Fall back to manual parsing below.
        pass

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value
    except OSError:
        # If we can't read .env, just skip; callers will see missing env vars.
        return


_load_env()


def _canonical_ticker(symbol: str) -> str:
    token = str(symbol or "").strip().upper()
    if not token:
        return token
    return _TICKER_ALIASES.get(token, token)


def _normalize_ticker_list(symbols: Iterable[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for symbol in symbols or []:
        token = _canonical_ticker(symbol)
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def load_watchlist_tickers() -> List[str]:
    if not WATCHLIST_PATH.exists():
        return []
    tickers: List[str] = []
    raw_text = WATCHLIST_PATH.read_text(encoding="utf-8")
    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[\s,]+", line) if p]
        tickers.extend(parts)
    return _normalize_ticker_list(tickers)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse existing JSON file as array: {path}") from exc
    if isinstance(data, list):
        return data
    # Fallback: wrap single object
    if isinstance(data, dict):
        return [data]
    raise RuntimeError(f"Unexpected JSON root type in {path}: {type(data).__name__}")


def _parse_llm_json(raw_content: str) -> Dict[str, Any]:
    """
    Parse JSON returned by an LLM, being tolerant of common wrappers like
    Markdown code fences (```json ... ```).
    """
    text = (raw_content or "").strip()
    if text.startswith("```"):
        # Strip leading ``` or ```json and trailing ``` if present
        # Split once on newline to drop the first fence line.
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else ""
        if text.rstrip().endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return json.loads(text)


def _ensure_meta_fields(obj: Dict[str, Any], symbol: str, mode: str) -> Dict[str, Any]:
    meta = obj.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    meta["symbol"] = symbol
    meta["generated_at"] = _now_utc_iso()
    meta["mode"] = mode
    obj["meta"] = meta
    return obj


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(num):
        return fallback
    return num


def _build_forecast_prompt(
    symbol: str,
    mode: str,
    current_price: float,
    sma_5: float,
    sentiment_score: float,
) -> str:
    return f"""You are a financial forecasting assistant.

Given the stock ticker "{symbol}", produce a concise next-session forecast.

Current IRIS metrics (use these as factual context):
- current_price_usd: {current_price:.4f}
- sma_5_usd: {sma_5:.4f}
- sentiment_score: {sentiment_score:.4f}

Respond with a single JSON object with this exact structure and field names:
{{
  "meta": {{
    "symbol": "{symbol}",
    "generated_at": "<ISO8601-UTC timestamp>",
    "mode": "{mode}"
  }},
  "market": {{
    "current_price": <float>,
    "predicted_price_next_session": <float>
  }},
  "signals": {{
    "trend_label": "<string like 'WEAK UPTREND ' or 'WEAK DOWNTREND '>",
    "sentiment_score": <float between -1 and 1>,
    "check_engine_light": "<string description like ' RED (..)' or ' YELLOW (..)' or ' GREEN (..)'>"
  }},
  "evidence": {{
    "headlines_used": [
      {{"title": "<short headline 1>", "url": ""}},
      {{"title": "<short headline 2>", "url": ""}}
    ]
  }}
}}

Rules:
- Only output raw JSON (no markdown, no code fences, no commentary).
- Use realistic but approximate prices in USD.
- Set market.current_price to current_price_usd exactly.
- Set signals.sentiment_score to sentiment_score exactly.
- Use sma_5_usd relative to current_price_usd and sentiment_score for trend reasoning.
- headlines_used items must be JSON objects with "title" (string) and "url" (empty string ""). Never output raw strings in that array."""


def _build_horizon_forecast_prompt(
    symbol: str,
    mode: str,
    current_price: float,
    sma_5: float,
    rsi_14: float,
    sentiment_score: float,
    horizon_label: str,
    horizon_days: int,
    headlines_summary: str,
) -> str:
    return f"""You are a quantitative financial analyst.

Given the stock ticker "{symbol}", produce a forecast for the {horizon_label} horizon ({horizon_days} trading days).

Current metrics:
- current_price_usd: {current_price:.4f}
- sma_5_usd: {sma_5:.4f}
- rsi_14: {rsi_14:.2f}
- sentiment_score: {sentiment_score:.4f}
- horizon: {horizon_label} ({horizon_days} trading days)

Recent relevant headlines:
{headlines_summary}

Respond with ONLY a single JSON object (no markdown, no code fences):
{{
  "meta": {{
    "symbol": "{symbol}",
    "generated_at": "<ISO8601-UTC timestamp>",
    "mode": "{mode}",
    "horizon": "{horizon_label}",
    "horizon_days": {horizon_days}
  }},
  "market": {{
    "current_price": {current_price:.4f},
    "predicted_price_horizon": <float>,
    "predicted_price_next_session": <float>
  }},
  "signals": {{
    "trend_label": "<STRONG UPTREND|WEAK UPTREND|WEAK DOWNTREND|STRONG DOWNTREND>",
    "sentiment_score": {sentiment_score:.4f},
    "check_engine_light": "<GREEN (..)|YELLOW (..)|RED (..)>",
    "investment_signal": "<STRONG BUY|BUY|HOLD|SELL|STRONG SELL>"
  }},
  "evidence": {{
    "headlines_used": [
      {{"title": "<headline 1>", "url": ""}},
      {{"title": "<headline 2>", "url": ""}}
    ]
  }},
  "reasoning": "<2-3 sentence explanation of why you made this prediction>"
}}

Rules:
- investment_signal MUST be exactly one of: STRONG BUY, BUY, HOLD, SELL, STRONG SELL
- predicted_price_horizon is the price at END of the {horizon_label} period
- reasoning should reference the metrics and headlines provided
- Only output raw JSON"""


def get_chatgpt52_forecast(
    symbol: str,
    current_price: float,
    sma_5: float,
    sentiment_score: float,
    *,
    mode: str = "live_forecast",
) -> Dict[str, Any]:
    """
    Call ChatGPT 5.2 (or configured OpenAI model) to get a forecast JSON.

    Requires:
    - OPENAI_API_KEY in environment
    - Optional OPENAI_MODEL_CHATGPT52 for model override (default: gpt-4o)
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai package is not installed. Install with 'pip install openai'.") from exc

    client = OpenAI()
    model_name = os.environ.get("OPENAI_MODEL_CHATGPT52", "gpt-4o")

    prompt = _build_forecast_prompt(
        symbol,
        mode,
        current_price,
        sma_5,
        sentiment_score,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You produce structured JSON forecasts for US equities."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    content = response.choices[0].message.content or ""
    data = _parse_llm_json(content)
    return _ensure_meta_fields(data, symbol, mode)


def get_deepseek_v3_forecast(
    symbol: str,
    current_price: float,
    sma_5: float,
    sentiment_score: float,
    *,
    mode: str = "live_forecast",
) -> Dict[str, Any]:
    """
    Call DeepSeek V3 API (OpenAI-compatible HTTP) to get a forecast JSON.

    Requires:
    - DEEPSEEK_API_KEY in environment
    - Optional DEEPSEEK_BASE_URL (default: https://api.deepseek.com)
    - Optional DEEPSEEK_MODEL (default: deepseek-chat)
    """
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError("requests package is not installed. Install with 'pip install requests'.") from exc

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is required.")

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    prompt = _build_forecast_prompt(
        symbol,
        mode,
        current_price,
        sma_5,
        sentiment_score,
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You produce structured JSON forecasts for US equities."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    data = _parse_llm_json(content)
    return _ensure_meta_fields(data, symbol, mode)


def get_geminiv3_forecast(
    symbol: str,
    current_price: float,
    sma_5: float,
    sentiment_score: float,
    *,
    mode: str = "live_forecast",
) -> Dict[str, Any]:
    """
    Call Gemini V3 Pro via google-genai client to get a forecast JSON.

    Requires:
    - GEMINI_API_KEY in environment
    - Optional GEMINI_MODEL (default: gemini-3-flash-preview or similar)
    """
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai package is not installed. Install with 'pip install google-genai'.") from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required.")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    client = genai.Client(api_key=api_key)

    prompt = _build_forecast_prompt(
        symbol,
        mode,
        current_price,
        sma_5,
        sentiment_score,
    )
    response = client.models.generate_content(model=model_name, contents=prompt)
    content = response.text or ""
    data = _parse_llm_json(content)
    return _ensure_meta_fields(data, symbol, mode)


def predict_with_llms(
    symbol: str,
    current_price: float,
    sma_5: float,
    rsi_14: float,
    sentiment_score: float,
    horizon: str,
    horizon_days: int,
    horizon_label: str,
    headlines_summary: str,
    mode: str = "live_forecast",
) -> dict:
    """Call all three LLM providers in parallel and return a results dict."""
    del horizon  # Kept for compatibility with endpoint call signature.

    prompt = _build_horizon_forecast_prompt(
        symbol=symbol,
        mode=mode,
        current_price=current_price,
        sma_5=sma_5,
        rsi_14=rsi_14,
        sentiment_score=sentiment_score,
        horizon_label=horizon_label,
        horizon_days=horizon_days,
        headlines_summary=headlines_summary,
    )

    def _call_chatgpt():
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI()
            model_name = os.environ.get("OPENAI_MODEL_CHATGPT52", "gpt-4o")
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You produce structured JSON forecasts for US equities."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                timeout=15,
            )
            data = _parse_llm_json(resp.choices[0].message.content or "")
            return _ensure_meta_fields(data, symbol, mode)
        except Exception as e:
            return {"error": str(e), "status": "unavailable"}

    def _call_deepseek():
        try:
            import requests as req  # type: ignore

            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                return {"error": "DEEPSEEK_API_KEY not set", "status": "unavailable"}

            base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
            resp = req.post(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You produce structured JSON forecasts for US equities."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.4,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = _parse_llm_json(resp.json()["choices"][0]["message"]["content"])
            return _ensure_meta_fields(data, symbol, mode)
        except Exception as e:
            return {"error": str(e), "status": "unavailable"}

    def _call_gemini():
        try:
            from google import genai  # type: ignore

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return {"error": "GEMINI_API_KEY not set", "status": "unavailable"}
            model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(model=model_name, contents=prompt)
            data = _parse_llm_json(resp.text or "")
            return _ensure_meta_fields(data, symbol, mode)
        except Exception as e:
            return {"error": str(e), "status": "unavailable"}

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_call_chatgpt): "chatgpt52",
            executor.submit(_call_deepseek): "deepseek_v3",
            executor.submit(_call_gemini): "gemini_v3_pro",
        }
        try:
            for future in as_completed(futures, timeout=20):
                model_key = futures[future]
                try:
                    results[model_key] = future.result()
                except Exception as e:
                    results[model_key] = {"error": str(e), "status": "unavailable"}
        except Exception:
            # Preserve completed results and mark unresolved calls as unavailable.
            pass

    for model_key in ("chatgpt52", "deepseek_v3", "gemini_v3_pro"):
        if model_key not in results:
            results[model_key] = {"error": "timeout", "status": "unavailable"}

    return results


_VALID_SIGNALS = {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}


def _normalize_llm_result(result: dict) -> dict:
    """Normalize signal and ensure reasoning exists."""
    if "error" in result:
        return result

    signals = result.get("signals", {})
    raw_signal = str(signals.get("investment_signal", "")).strip().upper()
    if raw_signal not in _VALID_SIGNALS:
        signals["investment_signal"] = "HOLD"
    else:
        signals["investment_signal"] = raw_signal
    result["signals"] = signals

    reasoning = str(result.get("reasoning", "")).strip()
    if not reasoning or len(reasoning) < 10:
        trend = str(signals.get("trend_label", "neutral")).lower().strip()
        price = result.get("market", {}).get("predicted_price_horizon", "N/A")
        reasoning = f"Model predicts {trend} trend to ${price}."
    result["reasoning"] = reasoning[:300]
    return result


def _normalize_llm_signal(result: dict) -> dict:
    """Backward-compatible alias for legacy callers."""
    return _normalize_llm_result(result)


def generate_reports_for_watchlist(*, mode: str = "live_forecast") -> None:
    tickers = load_watchlist_tickers()
    if not tickers:
        print("No tickers found in watchlist.txt; nothing to do.")
        return

    LLM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "chatgpt52": LLM_REPORTS_DIR / "chatgpt_5.2.json",
        "deepseek_v3": LLM_REPORTS_DIR / "deepseek_v3.json",
        "gemini_v3_pro": LLM_REPORTS_DIR / "gemini_v3_pro.json",
    }

    existing: Dict[str, List[Dict[str, Any]]] = {
        key: _load_json_array(path) for key, path in files.items()
    }
    app = IRIS_System()

    for symbol in tickers:
        print(f"Fetching LLM forecasts for {symbol}...")
        market_data = app.get_market_data(symbol) or {}
        sentiment_raw, _headlines = app.analyze_news(symbol)

        current_price = _safe_float(market_data.get("current_price"), 0.0)
        sma_5 = current_price
        history_df = market_data.get("history_df")
        if history_df is not None:
            try:
                sma_5 = _safe_float(history_df["sma_5"].iloc[-1], current_price)
            except Exception:
                sma_5 = current_price
        sentiment_score = _safe_float(sentiment_raw, 0.0)

        # ChatGPT 5.2
        try:
            chatgpt_obj = get_chatgpt52_forecast(
                symbol,
                current_price,
                sma_5,
                sentiment_score,
                mode=mode,
            )
        except Exception as exc:
            print(f"  ChatGPT 5.2 error for {symbol}: {exc}")
        else:
            existing["chatgpt52"].append(chatgpt_obj)

        # DeepSeek V3
        try:
            deepseek_obj = get_deepseek_v3_forecast(
                symbol,
                current_price,
                sma_5,
                sentiment_score,
                mode=mode,
            )
        except Exception as exc:
            print(f"  DeepSeek V3 error for {symbol}: {exc}")
        else:
            existing["deepseek_v3"].append(deepseek_obj)

        # Gemini V3 Pro
        try:
            gemini_obj = get_geminiv3_forecast(
                symbol,
                current_price,
                sma_5,
                sentiment_score,
                mode=mode,
            )
        except Exception as exc:
            print(f"  Gemini V3 Pro error for {symbol}: {exc}")
        else:
            existing["gemini_v3_pro"].append(gemini_obj)

    for key, path in files.items():
        path.write_text(json.dumps(existing[key], indent=2), encoding="utf-8")
        print(f"Wrote {len(existing[key])} entries to {path}")


def main() -> int:
    # For now, we just run once in "live_forecast" mode for all watchlist tickers.
    generate_reports_for_watchlist(mode="live_forecast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

