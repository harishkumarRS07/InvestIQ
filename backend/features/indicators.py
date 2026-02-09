import pandas as pd
import numpy as np
import ta
from backend.core.logging import logger

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe:
    - MA (Moving Part): SMA_20, SMA_50
    - RSI (Relative Strength Index)
    - VWAP (Volume Weighted Average Price)
    - Bollinger Bands
    """
    try:
        df = df.copy()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = indicator_bb.bollinger_hband()
        df['BB_Low'] = indicator_bb.bollinger_lband()
        
        # VWAP
        # ta library VWAP requires High, Low, Close, Volume
        df['VWAP'] = ta.volume.volume_weighted_average_price(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14
        )

        # MACD
        macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # ATR (Volatility)
        atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ATR'] = atr.average_true_range()

        # Log Return (Target Variable)
        # We calculate it here so it's available for feature engineering too (autocorrelation)
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Drop NaNs created by indicators (e.g. first 50 rows for SMA_50)
        df = df.dropna()
        
        logger.info("Technical indicators added successfully")
        return df
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        raise e
