import pandas as pd
from typing import Dict

class TimeFrameProcessor:
    """
    Resamples daily data into larger timeframes (Weekly, Monthly)
    and aligns them for multi-timeframe model input.
    """
    
    @staticmethod
    def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample Daily OHLCV data.
        rule: 'W' (Weekly), 'M' (Monthly)
        """
        # Ensure 'Date' index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('Date') if 'Date' in df.columns else df
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Index must be DatetimeIndex")

        resampled = df.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Drop incomplete periods if needed, but often we want the running week/month
        return resampled.dropna()

    @staticmethod
    def create_multiframe_features(daily_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary with 'daily', 'weekly', 'monthly' dataframes.
        """
        weekly_df = TimeFrameProcessor.resample(daily_df, 'W')
        monthly_df = TimeFrameProcessor.resample(daily_df, 'ME')
        
        return {
            'daily': daily_df,
            'weekly': weekly_df,
            'monthly': monthly_df
        }

    @staticmethod
    def merge_timeframes(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, monthly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges weekly/monthly features back into daily dataframe using forward fill.
        Useful if we want to use higher-timeframe trend as a feature for daily prediction.
        """
        # Rename columns to avoid collision via suffix
        weekly = weekly_df.add_suffix('_W')
        monthly = monthly_df.add_suffix('_M')
        
        # Merge on index (Date)
        # We use reindex/ffill to propagate the last known weekly/monthly value to daily rows
        merged = daily_df.join(weekly, how='left').join(monthly, how='left')
        
        # Forward fill the NaN values introduced by joining sparse weekly/monthly data onto daily
        merged = merged.ffill()
        
        return merged
