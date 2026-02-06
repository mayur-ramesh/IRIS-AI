import yfinance as yf
from newsapi import NewsApiClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import time


NEWS_API_KEY = None



class IRIS_System:
    def __init__(self):
        print("\n  Initializing IRIS Risk Engines...")
        
        #. Setup Sentiment Brain (VADER)
        # We download the dictionary required to understand words like "crash" or "soar"
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            print("   -> Downloading NLTK lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        #  Setup News Connection
        self.news_api = None
        if NEWS_API_KEY:
            self.news_api = NewsApiClient(api_key=NEWS_API_KEY)
            print("   -> NewsAPI Connection: ESTABLISHED")
        else:
            print("   -> NewsAPI Connection: SIMULATION MODE (No Key Found)")

    def get_market_data(self, ticker):
        """Fetches current price and 30-day history for the Prediction Engine."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get Current Price
            price = stock.fast_info['last_price']
            
            # Get History (Last 30 days) for the Trend AI
            hist = stock.history(period="30d")
            
            return {
                "current_price": price,
                "history": hist['Close'].values if not hist.empty else None,
                "symbol": ticker.upper()
            }
        except Exception as e:
            # If stock doesn't exist or internet is down
            return None

    def analyze_news(self, ticker):
        """Fetches headlines and calculates a Sentiment Score (-1.0 to +1.0)."""
        headlines = []
        
        #  Try to get real news
        if self.news_api:
            try:
                response = self.news_api.get_everything(
                    q=ticker, 
                    language='en', 
                    sort_by='publishedAt',
                    page_size=5
                )
                headlines = [art['title'] for art in response['articles']]
            except:
                pass # Fail silently and use simulation if API errors out
        
        #  Fallback: Simulation Mode (If no Key or API failure)
        if not headlines:
            if ticker == "TSLA":
                headlines = [
                    "Tesla recalls 2 million vehicles due to autopilot risk",
                    "Analysts downgrade stock amid slowing EV demand"
                ]
            elif ticker == "NVDA":
                headlines = [
                    "Nvidia announces breakthrough AI chip", 
                    "Quarterly revenue beats expectations by 20%"
                ]
            else:
                headlines = [
                    f"{ticker} announces date for shareholder meeting",
                    "Market volatility continues ahead of Fed decision"
                ]

        #  Analyze Sentiment
        total_score = 0
        for h in headlines:
            # compound score ranges from -1 (Negative) to +1 (Positive)
            score = self.sentiment_analyzer.polarity_scores(h)['compound']
            total_score += score
        
        avg_score = total_score / len(headlines) if headlines else 0
        return avg_score, headlines

    def predict_trend(self, history_prices):
        """
        The 'Crystal Ball': Uses Linear Regression to predict the mathematical trend.
        """
        if history_prices is None or len(history_prices) < 5:
            return "INSUFFICIENT DATA", 0, 0

        # Prepare data for AI (X = Day Number, y = Price)
        days = np.array(range(len(history_prices))).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(days, history_prices)

        #  Predict Next Day Price
        next_day_index = np.array([[len(history_prices)]])
        predicted_price = model.predict(next_day_index)[0]
        
        #  Calculate Slope (Steepness of the trend)
        slope = model.coef_[0]

        #  Label the Trend
        if slope > 0.5: label = "STRONG UPTREND "
        elif slope > 0: label = "WEAK UPTREND "
        elif slope < -0.5: label = "STRONG DOWNTREND "
        else: label = "WEAK DOWNTREND "

        return label, predicted_price

    def run(self):
        print("\n" + "="*50)
        print("--IRIS: INTELLIGENT RISK IDENTIFICATION SYSTEM--")
        print("    'The Check Engine Light for your Portfolio'")
        print("="*50)
        
        while True:
            ticker = input("\nEnter Stock Ticker (e.g., AAPL, TSLA) or 'q' to quit: ").strip().upper()
            
            if ticker == 'Q':
                print("Shutting down IRIS...")
                break
            if not ticker: continue

            print(f"... Analyzing noise for {ticker} ...")
            
            #  Get Data
            data = self.get_market_data(ticker)
            if not data:
                print(" Stock not found or connection error.")
                continue

            #  Run AI Engines
            sentiment_score, headlines = self.analyze_news(ticker)
            trend_label, predicted_price = self.predict_trend(data['history'])

            #  The Check Engine Logic (Filter)
            #  If Sentiment is Bad OR Trend is crashing -> RED
            light = " GREEN (Safe to Proceed)"
            
            if sentiment_score < -0.05 or "STRONG DOWNTREND" in trend_label:
                light = " RED (Risk Detected - Caution)"
            elif abs(sentiment_score) < 0.05 and "WEAK" in trend_label:
                light = " YELLOW (Neutral / Noise)"

            # The Report
            print("\n" + "-"*40)
            print(f"REPORT: {data['symbol']}")
            print("-" * 40)
            print(f" Current Price:   ${data['current_price']:.2f}")
            print(f" AI Projection:   ${predicted_price:.2f} (Next Session)")
            print("-" * 40)
            print(f"  TREND AI:     {trend_label}")
            print(f"  SENTIMENT AI: {sentiment_score:.2f} (Scale: -1 to 1)")
            print("-" * 40)
            print(f" CHECK ENGINE LIGHT: {light}")
            print("="*40)
            time.sleep(0.5)

if __name__ == "__main__":
    app = IRIS_System()
    app.run()