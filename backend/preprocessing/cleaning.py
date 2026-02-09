import pandas as pd
import numpy as np
from backend.core.logging import logger
from backend.core.exceptions import DataNotFoundException, PreprocessingException

def load_data(file_path: str) -> pd.DataFrame:
    """Load stock data from CSV"""
    try:
        # User's data often has a 2nd row with metadata (Date is NaN or empty there)
        # We read it all as strings first to inspect
        df = pd.read_csv(file_path, dtype=str)
        
        # Check if the first row is metadata (e.g. Date is null or empty string)
        if pd.isna(df.iloc[0]['Date']) or df.iloc[0]['Date'].strip() == '':
            logger.info("Detected metadata row, dropping row 0")
            df = df.iloc[1:].reset_index(drop=True)
            
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise DataNotFoundException(f"File {file_path} not found")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise PreprocessingException(f"Error loading data: {e}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Convert Date to datetime
    - Sort by Date
    - Handle missing values (forward fill then backward fill)
    """
    try:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise PreprocessingException(f"Missing required columns: {missing}")

        # Ensure numeric columns
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values
        df[required_cols] = df[required_cols].ffill().bfill()
        
        # Verify no NaNs remain
        if df[required_cols].isnull().any().any():
             logger.warning("NaNs remaining after fill, dropping rows")
             df = df.dropna(subset=required_cols)

        logger.info("Data cleaned successfully")
        return df
    except Exception as e:
        logger.error(f"Error in clean_data: {e}")
        raise PreprocessingException(f"Error cleaning data: {e}")
