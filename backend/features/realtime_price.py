
import yfinance as yf
import pandas as pd
from datetime import datetime
from backend.core.logging import logger

def fetch_latest_stock_data(ticker: str) -> pd.DataFrame:
    """
    Fetches the latest daily stock data (Open, High, Low, Close, Volume) 
    for the given ticker using yfinance.
    
    Args:
        ticker (str): Ticker symbol (e.g., "HDFCBANK" or "HDFCBANK.BO").
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                      containing the latest available trading day's data.
                      Returns empty DataFrame if fetch fails.
    """
    try:
        # Default to BSE (.BO) if no suffix and not an index
        search_ticker = ticker
        if not ticker.endswith(".NS") and not ticker.endswith(".BO") and not ticker.startswith("^"):
             search_ticker = f"{ticker}.BO"
        
        logger.info(f"Fetching live price data for {search_ticker}...")
        
        # Fetch 1 day of data
        # period="1d" gets the latest available trading session
        stock = yf.Ticker(search_ticker)
        df_live = stock.history(period="1d", interval="1d")
        
        if df_live.empty:
            logger.warning(f"No live data found for {search_ticker}")
            return pd.DataFrame()
            
        # Reset index to make Date a column
        df_live.reset_index(inplace=True)
        
        # Select and Rename columns to match our schema
        # yfinance columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Ensure columns exist
        available_cols = [col for col in required_cols if col in df_live.columns]
        df_live = df_live[available_cols]
        
        # Convert Date to datetime and normalize to midnight (remove timezone)
        if 'Date' in df_live.columns:
            df_live['Date'] = pd.to_datetime(df_live['Date']).dt.tz_localize(None).dt.normalize()
            
        logger.info(f"Successfully fetched live data for {search_ticker} date: {df_live['Date'].iloc[0]}")
        return df_live

    except Exception as e:
        logger.error(f"Failed to fetch live price data for {ticker}: {e}")
        return pd.DataFrame()
