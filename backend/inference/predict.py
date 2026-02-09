import torch
import pandas as pd
import numpy as np
import os
from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators
from backend.models.lstm_attention import LSTMAttentionModel
from backend.models.xgboost_fusion import XGBoostFusionModel
from backend.core.exceptions import ModelNotTrainedException
from backend.features.realtime_price import fetch_latest_stock_data

class Predictor:
    def __init__(self):
        self.scaler = StockScaler()
        self.lstm_model = None
        self.xg_model = XGBoostFusionModel()
        self.feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'BB_High', 'BB_Low', 'VWAP', 'MACD', 'MACD_Signal', 'ATR', 'Log_Return']
        
        # self._load_models() # Models loaded lazily per ticker

    def _load_models(self, ticker: str):
        try:
            scaler_path = f"scaler_{ticker}.pkl"
            lstm_path = f"lstm_attention_{ticker}.pth"
            xg_path = f"xgboost_fusion_{ticker}.pkl"
            
            logger.info(f"Loading models for {ticker}...")
            self.scaler.load(scaler_path)
            
            # Init LSTM (dimensions need to match training, hardcoded for now or saved in config)
            # Assumption: feature_cols len is 11
            input_dim = len(self.feature_cols)
            self.lstm_model = LSTMAttentionModel(input_dim=input_dim)
            self.lstm_model.load_state_dict(torch.load(os.path.join(settings.MODEL_DIR, lstm_path)))
            self.lstm_model.eval()
            
            self.xg_model.load(xg_path)
            
        except FileNotFoundError as e:
            logger.warning(f"Models for {ticker} not found: {e}")
            self.lstm_model = None
            
    def predict(self, file_path: str, ticker: str = None):
        if not ticker:
             ticker = os.path.basename(file_path).replace(".csv", "")
             
        self._load_models(ticker)
        
        if not self.lstm_model:
            raise ModelNotTrainedException(f"Models for {ticker} are not trained or found.")

        # Load and prep latest data
        df = load_data(file_path)
        
        # --- NEW: Inject Real-time Data ---
        try:
            live_df = fetch_latest_stock_data(ticker)
            if not live_df.empty:
                # Align columns
                # load_data returns all strings or whatever, clean_data fixes types later
                # But here we want to concat. load_data primarily reads CSV. 
                # clean_data handles type conversion.
                # However, to deduplicate by Date, we need Dates to be comparable.
                
                # Parse Dates in historical df for merging purposes
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Live DF Date is already datetime (from our realtime_price.py)
                
                # 1. Update existing rows (if live date exists in history)
                live_date = live_df['Date'].iloc[0]
                
                # Remove any existing row with the same date
                df = df[df['Date'] != live_date]
                
                # 2. Append new live data
                df = pd.concat([df, live_df], ignore_index=True)
                
                logger.info(f"Integrated live data for {ticker} (Date: {live_date})")
                
        except Exception as e:
            logger.error(f"Error integrating live data: {e} - proceeding with historical data only.")
        # ----------------------------------

        df = clean_data(df)
        df = add_technical_indicators(df)
        
        # Get last sequence
        last_sequence_df = df.iloc[-settings.SEQ_LENGTH:]
        
        # Scale
        df_scaled = self.scaler.transform(last_sequence_df)
        seq_data = df_scaled[self.feature_cols].values
        
        # LSTM Prediction (Next Day Price)
        input_tensor = torch.FloatTensor(seq_data).unsqueeze(0) # [1, seq_len, input_dim]
        
        # Monte Carlo Dropout for Confidence Interval
        self.lstm_model.train() # Enable dropout
        mc_predictions = []
        for _ in range(20):
             with torch.no_grad():
                 pred = self.lstm_model(input_tensor)
                 mc_predictions.append(pred.item())
        
        self.lstm_model.eval() # Reset to eval
        
        mean_pred_scaled = np.mean(mc_predictions)
        std_pred_scaled = np.std(mc_predictions)
        
        # Inverse Transform for Log Return
        # Construct a dummy array to inverse transform just the target
        dummy_row = np.zeros((1, len(self.feature_cols)))
        target_idx = self.feature_cols.index('Log_Return')
        
        # Mean Prediction
        dummy_row[0, target_idx] = mean_pred_scaled
        pred_log_return = self.scaler.scaler.inverse_transform(dummy_row)[0, target_idx]
        
        # Lower Bound
        dummy_row[0, target_idx] = mean_pred_scaled - 1.96 * std_pred_scaled
        lower_log_return = self.scaler.scaler.inverse_transform(dummy_row)[0, target_idx]
        
        # Upper Bound
        dummy_row[0, target_idx] = mean_pred_scaled + 1.96 * std_pred_scaled
        upper_log_return = self.scaler.scaler.inverse_transform(dummy_row)[0, target_idx]
        
        # Reconstruct Price
        # Predicted_Price = Current_Price * exp(Predicted_Log_Return)
        current_price = df['Close'].iloc[-1]
        pred_price = current_price * np.exp(pred_log_return)
        
        # Bounds (Approximation: Apply return bounds to current price)
        # Note: Uncertainty expands with time, but for 1-step logic this is consistent
        lower_bound = current_price * np.exp(lower_log_return)
        upper_bound = current_price * np.exp(upper_log_return)

        # XGBoost Prediction (Signal)
        # Fuse Features: [Technicals, LSTM_Pred, Sentiment, Macro]
        
        # 1. Technicals (Last row of sequence)
        last_row_scaled = seq_data[-1] # Shape (n_features,)
        
        # 2. LSTM Prediction (Scaled Log Return)
        # mean_pred_scaled is a scalar numpy float
        
        # 3. External Data
        from backend.features.external_data import ExternalDataSimulator
        
        # Use REAL Live News Sentiment
        sentiment = ExternalDataSimulator.fetch_live_sentiment(ticker)
        
        # Macro is still simulated (API access expensive/complex for macro)
        macro = ExternalDataSimulator.get_macro_score()
        
        # Create Fused Vector
        # Ensure all are 1D arrays or scalars before concat
        # Shape 1, -1 for single sample prediction
        fused_input = np.hstack([
            last_row_scaled, 
            np.array([mean_pred_scaled]), 
            np.array([sentiment]), 
            np.array([macro])
        ]).reshape(1, -1)
        
        signal = self.xg_model.predict(fused_input)[0]
        signal_probs = self.xg_model.predict_proba(fused_input)[0]
        
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        
        return {
            "current_price": df['Close'].iloc[-1],
            "predicted_price": pred_price,
            "confidence_interval": (lower_bound, upper_bound),
            "signal": signal_map.get(int(signal), "UNKNOWN"),
            "signal_confidence": float(max(signal_probs)),
            "risk_level": "High" if std_pred_scaled > 0.05 else "Low" # Simple heuristic
        }
