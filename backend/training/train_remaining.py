import os
from backend.core.config import settings
from backend.training.train import train_pipeline
from backend.core.logging import logger

if __name__ == "__main__":
    # Specific list of tickers to retrain (skipping HDFCBANK, ICICIBANK which finished)
    remaining_tickers = ['INFY', 'RELIANCE', 'TCS']
    
    print(f"Starting training for remaining tickers: {remaining_tickers}")
    
    for ticker in remaining_tickers:
        file_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            try:
                print(f"Processing {ticker}...")
                result = train_pipeline(file_path)
                print(result)
            except Exception as e:
                logger.error(f"Failed to train on {file_path}: {e}")
                print(f"Error on {ticker}: {e}")
        else:
            print(f"File not found for {ticker}")
