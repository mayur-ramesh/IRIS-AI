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

## Current Work Progress

* **Backend & REST API (`app.py`):** * Set up a Flask server with CORS support.
  * Implemented core API endpoints (`/api/analyze`, `/api/chart`, `/api/session-summary/latest`) to securely interface the frontend dashboard with the backend AI engines.
* **AI & Risk Engine (`iris_mvp.py`):**
  * **Market Data:** Integrated `yfinance` to pull 60-day historical data and calculate Simple Moving Averages (SMA).
  * **Trend Prediction:** Built a "Crystal Ball" feature using `scikit-learn`'s Ridge regression to predict next-session prices and identify Uptrends/Downtrends.
  * **Sentiment Analysis:** Integrated the `transformers` library using the `ProsusAI/finbert` model to analyze the sentiment of recent market headlines and generate a normalized Sentiment Score.
  * **Charting:** Built a fallback charting module using `matplotlib` to render static trend visualizations.
* **Automated Schedulers (`run_daily.py`):**
  * Developed timezone-aware automation scripts to trigger bulk ticker analyses exactly at the US market open (09:00 ET).
  * Added features to install and manage background cron jobs via Windows Task Scheduler.
* **Interactive Frontend Dashboard (`index.html` & `app.js`):**
  * Built a sleek, responsive UI to query tickers dynamically.
  * Integrated **TradingView Lightweight Charts** for interactive, real-time visualization of stock history and AI price predictions.
  * Added dynamic UI widgets for the "Check Engine Light" (Risk Indicator), Market Prediction summaries, and a Sentiment Meter with recent headline rendering.
