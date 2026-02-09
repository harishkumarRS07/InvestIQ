import sys
import os
import pandas as pd

# Add the project root (parent of 'backend') to python path
# This allows 'from backend.training import ...' to work regardless of CWD
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../invest iq/backend
project_root = os.path.dirname(current_dir) # .../invest iq
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.training.train import train_pipeline
from backend.inference.predict import Predictor
from backend.core.config import settings

def run_verification():
    print("1. Verifying Data Loading...")
    # Use HDFCBANK as test case
    ticker = "HDFCBANK"
    file_path = os.path.join(settings.DATA_DIR, f"{ticker}.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    print(f"   Found {file_path}")

    # Start training is handled separately via 'python backend/training/train.py'
    print("\n2. Checking Model Artifacts (Wait for training to complete)...")
    files = [f"scaler_{ticker}.pkl", f"lstm_attention_{ticker}.pth", f"xgboost_fusion_{ticker}.pkl"]
    missing = []
    for f in files:
        path = os.path.join(settings.MODEL_DIR, f)
        if os.path.exists(path):
            print(f"   {f} found.")
        else:
            missing.append(f)
    
    if missing:
        print(f"   Warning: The following models are missing (Training might still be running): {missing}")
        print("   Cannot proceed with prediction verification until models are ready.")
        return

    print(f"\n3. Running Inference for {ticker}...")
    try:
        predictor = Predictor()
        result = predictor.predict(file_path, ticker=ticker)
        print("   Prediction Success!")
        print(f"   Current Price: {result['current_price']}")
        print(f"   Predicted Price: {result['predicted_price']:.2f}")
        print(f"   Confidence Interval: {result['confidence_interval']}")
        print(f"   Signal: {result['signal']} ({result['signal_confidence']:.2f})")
    except Exception as e:
        print(f"   Inference Failed: {e}")

if __name__ == "__main__":
    run_verification()
