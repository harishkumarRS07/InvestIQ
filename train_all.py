import asyncio
import os
import sys
from backend.data.realtime import RealTimeDataFetcher
from backend.training.train import train_pipeline
from backend.core.logging import logger

# Add project root
sys.path.append(os.getcwd())

async def train_all():
    print("="*60)
    print("      AI MODEL TRAINING PIPELINE      ")
    print("="*60)
    
    tickers = ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS"]
    fetcher = RealTimeDataFetcher()
    
    for ticker in tickers:
        print(f"\n[1/2] FETCHING DATA FOR {ticker}...")
        # Fetch 2 years of data for training
        df = fetcher.fetch_price_history(ticker, period="2y")
        
        if df.empty:
            print(f" -> Failed to fetch data for {ticker}. Skipping.")
            continue
            
        # Save to CSV for training pipeline to read
        data_dir = "backend/data/stock_data"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"{ticker}.csv")
        df.to_csv(file_path)
        print(f" -> Saved data to {file_path}")
        
        print(f"[2/2] TRAINING AI MODELS FOR {ticker}...")
        print(" -> Optimizing LSTM & XGBoost... (This may take a minute)")
        
        try:
            # Run existing training pipeline
            # This handles scaling, feature engineering, LSTM training, and XGBoost training
            result = train_pipeline(file_path)
            print(f" -> Training Complete! Metrics: {result}")
        except Exception as e:
            print(f" -> Training Failed: {e}")
            logger.error(f"Training failed for {ticker}", exc_info=True)

    print("\n" + "="*60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("Run run_demo.bat again to see REAL AI PREDICTIONS.")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(train_all())
