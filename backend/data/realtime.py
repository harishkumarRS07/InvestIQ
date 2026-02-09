import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from backend.core.logging import logger

class RealTimeDataFetcher:
    """
    Fetches real-time market data using yFinance.
    Includes caching mechanisms to respect API rate limits.
    """
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=1)

    def _get_ticker_symbol(self, symbol: str) -> str:
        """
        Adjusts symbol for Yahoo Finance (e.g., adding .NS for Indian stocks if needed).
        This is a heuristic and might need refinement based on specific user needs.
        """
        if symbol.endswith(".NS") or symbol.endswith(".BO") or symbol.startswith("^"):
            return symbol
        # Default assumption: NSE for Indian context if not specified, or US otherwise.
        # Given the context of "HDFCBANK", it's likely Indian.
        # We'll try adding .NS if it doesn't have a suffix.
        return f"{symbol}.NS"

    def fetch_price_history(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical price data.
        """
        ticker_symbol = self._get_ticker_symbol(symbol)
        cache_key = f"{ticker_symbol}_{period}_{interval}"
        
        if cache_key in self._cache and datetime.now() < self._cache_expiry[cache_key]:
            logger.info(f"Returning cached data for {ticker_symbol}")
            return self._cache[cache_key]

        try:
            logger.info(f"Fetching data for {ticker_symbol} (Period: {period}, Interval: {interval})")
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {ticker_symbol}")
                return pd.DataFrame()
            
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            
            # Cache result
            self._cache[cache_key] = df
            self._cache_expiry[cache_key] = datetime.now() + self.cache_duration
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker_symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest available price.
        """
        df = self.fetch_price_history(symbol, period="1d", interval="1m")
        if not df.empty:
            return df['Close'].iloc[-1]
        return None

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental company information.
        """
        ticker_symbol = self._get_ticker_symbol(symbol)
        try:
            ticker = yf.Ticker(ticker_symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching info for {ticker_symbol}: {e}")
            return {}
