
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import traceback

# Ensure backend can be imported
sys.path.append(os.getcwd())

from backend.core.config import settings
from backend.models.xgboost_fusion import XGBoostFusionModel
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators

def calculate_accuracy():
    csv_files = glob.glob(os.path.join(settings.DATA_DIR, "*.csv"))
    
    print(f"{'Ticker':<15} | {'Test Accuracy':<15} | {'Support':<10}")
    print("-" * 50)
    
    for file_path in csv_files:
        ticker = os.path.basename(file_path).replace(".csv", "")
        
        try:
            # Load and Prep
            df = load_data(file_path)
            df = clean_data(df)
            df = add_technical_indicators(df)
            
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'BB_High', 'BB_Low', 'VWAP', 'MACD', 'MACD_Signal', 'ATR', 'Log_Return']
            
            # Scale
            scaler = StockScaler()
            # Note: We must load the saved scaler to match training! 
            # Actually, standard practice is fitting on train and transforming test.
            # But here we used fit_transform on whole dataset in train.py (minor leakage issue in preprocessing, but let's stick to train.py methodology for consistency)
            # To be strictly correct to train.py logic:
            df_scaled = scaler.fit_transform(df, feature_cols)
            
            # Prepare Data
            xg_model = XGBoostFusionModel()
            
            # Try to load formatted model
            try:
                xg_model.load(f"xgboost_fusion_{ticker}.pkl")
            except Exception:
                print(f"{ticker:<15} | {'No Model':<15} | {'-':<10}")
                continue
                
            labels = xg_model.prepare_labels(df, horizon=5)
            
            # Align
            X_xgb = df_scaled[feature_cols].iloc[:-5].values
            y_xgb = labels
            
            if len(X_xgb) == 0:
                continue

            # Split (Replicate train.py logic EXACTLY)
            # But wait, we need to FUSE first.
            
            # --- FUSION LOGIC FOR ACCURACY TEST ---
            
            # 1. Load LSTM
            from backend.models.lstm_attention import LSTMAttentionModel
            import torch
            
            lstm_path = f"lstm_attention_{ticker}.pth"
            input_dim = len(feature_cols)
            lstm_model = LSTMAttentionModel(input_dim=input_dim)
            lstm_model.load_state_dict(torch.load(os.path.join(settings.MODEL_DIR, lstm_path)))
            lstm_model.eval()
            
            # 2. Extract LSTM Features
            # Create sequences 
            def create_sequences(data, seq_length):
                sequences = []
                for i in range(len(data) - seq_length):
                    seq = data[i:i+seq_length]
                    sequences.append(seq)
                return np.array(sequences)

            X_lstm_in = create_sequences(df_scaled[feature_cols].values, settings.SEQ_LENGTH)
            
            # Get Predictions
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_lstm_in)
                lstm_preds = lstm_model(X_tensor).numpy()
                
            # 3. Add External Data (Mock)
            # Need to match lengths. 
            # Train logic: X_xgb (technicals) was sliced [SEQ_LENGTH : -horizon]
            # LSTM preds sliced [:-horizon]
            # External data sliced [SEQ_LENGTH : -horizon]
            
            horizon = 5
            valid_start = settings.SEQ_LENGTH
            valid_end = -horizon
            
            # Slice Base Features
            X_xgb_base = df_scaled[feature_cols].iloc[valid_start : valid_end].values
            
            # Slice LSTM
            lstm_preds_trimmed = lstm_preds[:valid_end] # lstm_preds is missing first SEQ_LENGTH vs df
            # Wait, X_lstm_in len is N - SEQ_LENGTH.
            # So lstm_preds[0] corresponds to df[SEQ_LENGTH].
            # We want to drop the last `horizon` rows of predictions.
            lstm_preds_trimmed = lstm_preds[:-horizon]
            
            # Slice External
            from backend.features.external_data import ExternalDataSimulator
            df_enriched = ExternalDataSimulator.add_external_features(df, ticker)
            sentiments = df_enriched['Sentiment'].values[valid_start : valid_end].reshape(-1, 1)
            macros = df_enriched['Macro_Score'].values[valid_start : valid_end].reshape(-1, 1)
            
            # Fuse
            X_fused = np.hstack([X_xgb_base, lstm_preds_trimmed, sentiments, macros])
            
            # Labels
            labels = xg_model.prepare_labels(df, horizon=horizon)
            y_xgb = labels[valid_start:]
            
            if len(X_fused) == 0: continue

            # Split (Replicate train.py logic EXACTLY)
            # train_test_split(..., shuffle=False)
            X_train, X_test, y_train, y_test = train_test_split(X_fused, y_xgb, test_size=settings.TEST_SIZE, shuffle=False)
            
            # Predict
            y_pred = xg_model.predict(X_test)
            
            # Metric
            acc = accuracy_score(y_test, y_pred)
            
            print(f"{ticker:<15} | {acc:.2%}         | {len(y_test):<10}")
            
        except Exception as e:
            print(f"{ticker:<15} | {'Error':<15} | {str(e)}")
            # traceback.print_exc()

if __name__ == "__main__":
    calculate_accuracy()
