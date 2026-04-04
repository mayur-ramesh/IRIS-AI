# IRIS: Vision in Volatility
CP3405 Team Project

## Project Description
**IRIS** is an AI-driven decision support tool designed to filter market noise. We identify short-term divergences between market sentiment and price movement to flag risk and validate sentiment before a trade is made.

**Key Features:**
* **Sentiment Analysis:** Synthesizing news and social media sentiment.
* **Risk Detection:** Flagging when sentiment and price action diverge.
* **Focus Market:** US Tech Sector (e.g. AAPL, TSLA, and AMD).

## Our Goal

Our vision is **not** to build a "Magic Crystal Ball" that gives Buy/Sell advice.

Instead, our goal is to build a **decision support tool** that helps retail traders manage volatility logically. We aim to act as a "Check Engine Light," alerting users to irrational market behavior so they can pause and reassess their risk.

## The Scrum Team (Sprint 1&2)

| Role | Name |
| :--- | :--- | 
| **Product Owner** | [Mayur Ramesh](https://github.com/mayur-ramesh) | 
| **Scrum Master** | [Minzhi Liu](https://github.com/DZSMG) | 
| **Data Scientist** | [Erdong Xiao](https://github.com/XIAOERDONG933) | 
| **AI Engineer** | [Shen Chen](https://github.com/chenshen0623) | 
| **Engineer IC (FE/BE)** | [Jun Bin Yap](https://github.com/Junbinyap) | 
| **UI/UX Designer** | [Jinyu Xie](https://github.com/xiejinyu-jcu) | 

## Disclaimer

IRIS provides decision support flags based on historical data and sentiment analysis. It does not provide financial advice.

---

## Team Runtime Data (Conflict-Free)

To avoid report/cache merge conflicts across different developer machines:

* By default, runtime outputs now go to `runtime_data/` (or `demo_guests_data/` when `DEMO_MODE=true`).
* Tracked `data/` files are no longer the default write target.
* You can override with environment variables:
  * `IRIS_USE_REPO_DATA=true` to intentionally write into `data/`
  * `IRIS_DATA_DIR=<path>` to use a custom machine-local folder

This keeps generated reports, sessions, feedback logs, and yfinance caches machine-local by default.

---

## FinBERT Setup

FinBERT sentiment analysis is enabled by default for IRIS.

* Dependencies are already declared in `requirements.txt`:
  * `torch`
  * `transformers`
* On first startup, IRIS loads the `ProsusAI/finbert` model using the normal Hugging Face cache for the machine.
* Optional environment overrides:
  * `IRIS_ENABLE_FINBERT=false` to disable FinBERT explicitly
  * `IRIS_FINBERT_MODEL_ID=<hugging-face-model-id>` to swap the model
  * `IRIS_FINBERT_CACHE_DIR=<path>` to pin the model cache to a project-local folder such as `runtime_data/huggingface/transformers`

If FinBERT cannot load, IRIS now logs the exact reason and continues running with sentiment fallback behavior instead of failing app startup.

---

## Historical Reports and Trend Accuracy

Use these commands to keep the pipeline reproducible across machines.

* Generate fresh runtime reports/session summary:
  * `python run_daily.py --once --tickers AAPL,MSFT`
* Read latest session summary from active runtime dir:
  * `runtime_data/sessions/latest_session_summary.json` (default)
  * `demo_guests_data/sessions/latest_session_summary.json` (`DEMO_MODE=true`)
* Generate trend-accuracy chart using active runtime reports:
  * `python trend_accuracy_comparison_chart.py`
* Compare using historical archived reports in tracked `data/`:
  * `python trend_accuracy_comparison_chart.py --data-dir data --output runtime_data/charts/<YYYY-MM-DD>/trend_accuracy_from_repo_data.png`

Notes:
* Accuracy requires at least two market sessions per ticker in the selected dataset.
* The script auto-falls back to `data/LLM reports` for LLM baseline files when runtime dirs do not contain them.

---

## Current Work Progress

* **Live Demo (Hugging Face Space):**
  * https://huggingface.co/spaces/Brajmovech/IRIS-AI_DEMO
* **Latest Updates (March 26, 2026):**
  * Finalized headlines panel UX fixes: restored stable scroll behavior, tightened spacing/density, and added dynamic height sync so the right-column headlines card always aligns with left-column dashboard cards.
  * Added frontend height re-sync triggers after dashboard render, async LLM insights render, and browser resize for consistent layout across ticker/data states.
  * Synced frontend fixes to both local and server projects with cache-version updates for immediate deployment refresh.
* **Backend & REST API (`app.py`):**
  * Set up a Flask server with CORS support.
  * Implemented core API endpoints (`/api/analyze`, `/api/chart`, `/api/session-summary/latest`) to securely interface the frontend dashboard with the backend AI engines.
  * **LLM Integration**: Aggregates predictions from ChatGPT 5.2, DeepSeek V3, and Gemini V3 Pro to provide comparative "LLM Insights".
  * **Pipeline Efficiency**: Removed redundant news-processing work in `/api/llm-predict` to reduce repeated backend load.
  * **Reliability**: Hardened LLM prediction handling for unavailable models and failure edge cases.
* **AI & Risk Engine (`iris_mvp.py`):**
  * **Market Data:** Integrated `yfinance` to pull historical OHLCV data.
  * **Trend Prediction:** Built a "Crystal Ball" feature using `scikit-learn`'s Ridge regression to predict next-session prices and identify Uptrends/Downtrends.
  * **Sentiment Analysis:** Integrated the `transformers` library using the `ProsusAI/finbert` model to analyze the sentiment of recent market headlines.
  * **Strict News Filtering:** Implemented `Webz.io` and `NewsAPI` aggregation with strict Regex boundaries and noise reduction patterns to ensure high-quality financial context.
  * **Coverage Expansion:** Added Google News RSS ingestion and candidate scoring to improve headline coverage quality.
* **Automated Schedulers (`run_daily.py` & `generate_llm_reports.py`):**
  * Developed timezone-aware automation scripts to trigger bulk ticker analyses exactly at the US market open (09:00 ET).
  * Implemented automated LLM forecasting scripts to build baseline reference data.
* **Interactive Frontend Dashboard (`index.html` & `app.js`):**
  * Built a sleek, responsive UI to query tickers dynamically.
  * Integrated **TradingView Lightweight Charts** featuring real-time interactive **Candlestick** and **Volume Histogram** overlays.
  * Added dynamic UI widgets for the "Check Engine Light" (Risk Indicator), Market Prediction summaries, LLM Insights, and a Sentiment Meter with recent headline rendering.
  * Added LLM timeout/retry UX flow and smoother loading-state handling for model cards.
  * Improved ticker input validation to support special market symbols.
