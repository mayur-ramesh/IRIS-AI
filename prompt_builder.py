"""
Builds LLM prompts anchored to real market data to prevent hallucination.
Also provides a post-processing sanity check for LLM output.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = (
    "You are a financial risk analyst. "
    "Use ONLY the market data provided below. "
    "Do NOT invent, estimate, or hallucinate any financial figures. "
    "If a data point is marked DATA_NOT_AVAILABLE, explicitly state that "
    "the data is unavailable — do NOT fill it in with estimates."
)

_ANALYSIS_REQUEST = (
    "Provide a risk analysis for {company_name} ({ticker}) covering: "
    "overall risk score (1-10), key risk factors, volatility assessment, "
    "and recommendations."
)

_FORMAT_INSTRUCTION = (
    "Format your response with clear sections. "
    "Prefix each data point you reference with its source from the provided data."
)


def build_risk_analysis_prompt(
    ticker: str,
    company_name: str,
    market_data: dict,
) -> str:
    """Return a grounded LLM prompt that anchors the model to *market_data*.

    Embeds the full market data dict and includes explicit instructions
    not to invent or estimate any figures.
    """
    company_label = company_name or ticker

    # Summary data block (exclude raw history to keep the prompt concise)
    exclude = {"price_history", "fetched_at"}
    summary_data = {k: v for k, v in market_data.items() if k not in exclude}
    data_block = json.dumps(summary_data, indent=2, default=str)

    # Recent price history summary (last 5 sessions)
    history = market_data.get("price_history", [])
    if history:
        recent = history[-5:]
        history_lines = [
            f"  {row['date']}: close={row['close']}, volume={row['volume']}"
            for row in recent
        ]
        history_block = (
            "Recent price history (last 5 sessions):\n" + "\n".join(history_lines)
        )
    else:
        history_block = "Recent price history: DATA_NOT_AVAILABLE"

    analysis_request = _ANALYSIS_REQUEST.format(
        company_name=company_label,
        ticker=ticker,
    )

    prompt = (
        f"[SYSTEM INSTRUCTION]\n{_SYSTEM_INSTRUCTION}\n\n"
        f"[VERIFIED MARKET DATA FOR {ticker}]\n{data_block}\n\n"
        f"{history_block}\n\n"
        f"[ANALYSIS REQUEST]\n{analysis_request}\n\n"
        f"[OUTPUT FORMAT]\n{_FORMAT_INSTRUCTION}"
    )
    return prompt


def validate_llm_output(llm_response: str, market_data: dict) -> str:
    """Best-effort sanity check: compare prices/market-cap in *llm_response*
    against *market_data*.

    If a discrepancy is detected, appends a disclaimer.
    Returns the (possibly annotated) response string.
    This is a best-effort check, not a hard block.
    """
    if not llm_response or not isinstance(llm_response, str):
        return llm_response

    discrepancy_found = False

    # Price check: any dollar amount that deviates > 10% from the real price
    real_price = _extract_real_price(market_data)
    if real_price is not None and real_price > 0:
        for price in _extract_dollar_amounts(llm_response):
            if abs(price - real_price) / real_price > 0.10:
                logger.warning(
                    "LLM price $%.2f deviates >10%% from real price $%.2f",
                    price,
                    real_price,
                )
                discrepancy_found = True
                break

    # Market-cap magnitude check: ratio must be between 0.1× and 10×
    real_mcap = market_data.get("marketCap")
    if real_mcap and real_mcap != "DATA_NOT_AVAILABLE":
        try:
            real_mcap_f = float(real_mcap)
            if real_mcap_f > 0:
                for cap in _extract_market_caps(llm_response):
                    if cap > 0:
                        ratio = cap / real_mcap_f
                        if ratio < 0.1 or ratio > 10:
                            logger.warning(
                                "LLM market cap ~%.2e deviates by order of magnitude from real %.2e",
                                cap,
                                real_mcap_f,
                            )
                            discrepancy_found = True
                            break
        except (ValueError, TypeError):
            pass

    if discrepancy_found:
        disclaimer = (
            "\n\n---\n"
            "Note: Some figures in this analysis may not exactly match current market data. "
            "Please refer to the data summary above for verified numbers."
        )
        return llm_response + disclaimer

    return llm_response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_real_price(market_data: dict):
    """Return a float price from market_data, or None if unavailable."""
    for key in ("currentPrice", "regularMarketPrice"):
        val = market_data.get(key)
        if val and val != "DATA_NOT_AVAILABLE":
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    # Fall back to most recent close in price history
    history = market_data.get("price_history", [])
    if history:
        last_close = history[-1].get("close")
        if last_close and last_close != "DATA_NOT_AVAILABLE":
            try:
                return float(last_close)
            except (ValueError, TypeError):
                pass
    return None


def _extract_dollar_amounts(text: str) -> list:
    """Extract dollar amounts like $123.45 or $1,234.56 from *text*."""
    amounts = []
    for m in re.finditer(r"\$\s*([\d,]+(?:\.\d+)?)", text):
        try:
            amounts.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return amounts


def _extract_market_caps(text: str) -> list:
    """Extract market-cap mentions like '$2.5 trillion', '$300B', '$50M'."""
    amounts = []
    multipliers = {
        "trillion": 1e12,
        "billion":  1e9,
        "million":  1e6,
        "t": 1e12,
        "b": 1e9,
        "m": 1e6,
    }
    pattern = r"\$\s*([\d,.]+)\s*(trillion|billion|million|t|b|m)\b"
    for m in re.finditer(pattern, text, re.IGNORECASE):
        try:
            num = float(m.group(1).replace(",", ""))
            mult = multipliers.get(m.group(2).lower(), 1)
            amounts.append(num * mult)
        except (ValueError, TypeError):
            pass
    return amounts
