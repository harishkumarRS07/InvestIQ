import numpy as np
import pandas as pd
import yfinance as yf
from textblob import TextBlob
from backend.core.logging import logger

class ExternalDataSimulator:
    """
    Handles external data sources (News Sentiment, Macroeconomics).
    Supports both Simulation (for training/backtesting) and Live Fetching (for inference).
    """
    
    @staticmethod
    def get_sentiment(ticker: str, date: pd.Timestamp = None) -> float:
        """
        Simulate News Sentiment Score (-1.0 to 1.0).
        Used during training when historical news is hard to get.
        """
        return np.random.uniform(-1.0, 1.0)

    @staticmethod
    def fetch_live_sentiment(ticker_symbol: str) -> float:
        """
        Fetch REAL news sentiment using yfinance and TextBlob.
        Used during inference for real-time prediction.
        """
        try:
            logger.info(f"Fetching live news for {ticker_symbol}...")
            
            # Formatting for yahoo finance (Indian stocks need .NS)
            if not ticker_symbol.endswith(".NS") and not ticker_symbol.endswith(".BO") and not ticker_symbol.startswith("^"):
                 # Heuristic: assume NSE
                 search_ticker = f"{ticker_symbol}.NS"
            else:
                 search_ticker = ticker_symbol

            ticker_obj = yf.Ticker(search_ticker)
            news = ticker_obj.news
            
            if not news:
                logger.warning(f"No news found for {search_ticker}, returning neutral.")
                return 0.0
                
            total_sentiment = 0
            count = 0
            
            for article in news:
                title = article.get('title', '')
                analysis = TextBlob(title)
                total_sentiment += analysis.sentiment.polarity
                count += 1
                
            if count == 0: return 0.0
            
            avg_sentiment = total_sentiment / count
            logger.info(f"Live Sentiment for {ticker_symbol}: {avg_sentiment:.4f}")
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"Failed to fetch live sentiment for {ticker_symbol}: {e}")
            return 0.0

    @staticmethod
    def get_macro_score(date: pd.Timestamp = None) -> float:
        """
        Simulate Macroeconomic Health Score (0 to 100).
        0 = Recession, 100 = Boom.
        """
        # Macro data changes slowly. 
        # For simulation, we'll return a relatively stable random number 
        # or just a random number for robust model training demonstration.
        return np.random.uniform(40, 80)

    @staticmethod
    def add_external_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Enrich dataframe with simulated external features.
        """
        logger.info(f"Adding simulated external data for {ticker}...")
        
        # Generate synthetic data for the entire dataframe
        # Using numpy for speed
        n_rows = len(df)
        
        # Sentiment: Random noise (News follows random arrival)
        # Seed by index to be deterministic per run if needed, but let's keep it random for robustness
        sentiments = np.random.uniform(-1.0, 1.0, n_rows)
        
        # Macro: Slow moving random walk
        # Start at 60, move by small steps
        macro_scores = []
        current_macro = 60.0
        for _ in range(n_rows):
            change = np.random.normal(0, 0.5)
            current_macro = np.clip(current_macro + change, 0, 100)
            macro_scores.append(current_macro)
            
        df = df.copy()
        df['Sentiment'] = sentiments
        df['Macro_Score'] = np.array(macro_scores)
        
        return df
