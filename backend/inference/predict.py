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
            # Try to find model files with or without .NS suffix
            base_model_path = os.path.join(settings.MODEL_DIR, f"lstm_attention_{ticker}.pth")
            if not os.path.exists(base_model_path):
                # Try alternative (e.g. HDFCBANK.NS -> HDFCBANK or vice versa)
                alt_ticker = ticker.replace(".NS", "") if ".NS" in ticker else f"{ticker}.NS"
                alt_path = os.path.join(settings.MODEL_DIR, f"lstm_attention_{alt_ticker}.pth")
                if os.path.exists(alt_path):
                    ticker = alt_ticker
            
            scaler_path = f"scaler_{ticker}.pkl"
            lstm_path = f"lstm_attention_{ticker}.pth"
            xg_path = f"xgboost_fusion_{ticker}.pkl"
            
            logger.info(f"Loading models for {ticker}...")
            self.scaler.load(scaler_path)
            
            # Init LSTM (dimensions need to match training)
            # Training uses hidden_dim=64
            input_dim = len(self.feature_cols)
            self.lstm_model = LSTMAttentionModel(input_dim=input_dim, hidden_dim=64)
            self.lstm_model.eval()
            
            try:
                self.xg_model.load(xg_path)
            except Exception as e:
                 logger.warning(f"XGBoost model not found: {e}. signal will be heuristic.")
                 self.xg_model = None
            
        except FileNotFoundError as e:
            logger.warning(f"Models for {ticker} not found: {e}")
            self.lstm_model = None
            self.xg_model = None
            
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
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    # Ensure historical data is tz-naive
                    if df['Date'].dt.tz is not None:
                         df['Date'] = df['Date'].dt.tz_localize(None)
                
                # Ensure live data is tz-naive
                live_df['Date'] = pd.to_datetime(live_df['Date'])
                if live_df['Date'].dt.tz is not None:
                     live_df['Date'] = live_df['Date'].dt.tz_localize(None)
                
                live_date = live_df['Date'].iloc[0]
                df = df[df['Date'] != live_date]
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
        dummy_row = np.zeros((1, len(self.feature_cols)))
        target_idx = self.feature_cols.index('Log_Return')
        
        dummy_row[0, target_idx] = mean_pred_scaled
        pred_log_return = self.scaler.scaler.inverse_transform(dummy_row)[0, target_idx]
        
        dummy_row[0, target_idx] = mean_pred_scaled - 1.96 * std_pred_scaled
        lower_log_return = self.scaler.scaler.inverse_transform(dummy_row)[0, target_idx]
        
        dummy_row[0, target_idx] = mean_pred_scaled + 1.96 * std_pred_scaled
        upper_log_return = self.scaler.scaler.inverse_transform(dummy_row)[0, target_idx]
        
        # Reconstruct Price
        current_price = df['Close'].iloc[-1]
        pred_price = current_price * np.exp(pred_log_return)
        
        lower_bound = current_price * np.exp(lower_log_return)
        upper_bound = current_price * np.exp(upper_log_return)

        # XGBoost Prediction (Signal)
        signal = 1 # Default HOLD
        signal_probs = [0.0, 1.0, 0.0] # Default
        
        if self.xg_model and hasattr(self.xg_model, 'model'):
             try:
                # 1. Technicals (Last row of sequence)
                last_row_scaled = seq_data[-1] 
                
                # 2. LSTM Prediction
                # 3. External Data
                from backend.features.external_data import ExternalDataSimulator
                sentiment = ExternalDataSimulator.fetch_live_sentiment(ticker)
                macro = ExternalDataSimulator.get_macro_score()
                
                fused_input = np.hstack([
                    last_row_scaled, 
                    np.array([mean_pred_scaled]), 
                    np.array([sentiment]), 
                    np.array([macro])
                ]).reshape(1, -1)
                
                signal = self.xg_model.predict(fused_input)[0]
                signal_probs = self.xg_model.predict_proba(fused_input)[0]
             except Exception as e:
                 logger.error(f"XGBoost inference failed: {e}. Using heuristic.")
                 self.xg_model = None
        
        # Heuristic Fallback
        if not self.xg_model:
             # If predicted return > 1%, Buy. < -1%, Sell.
             if pred_log_return > 0.01: 
                 signal = 2 
                 signal_probs = [0.1, 0.1, 0.8]
             elif pred_log_return < -0.01: 
                 signal = 0
                 signal_probs = [0.8, 0.1, 0.1]
             else:
                 signal = 1
                 signal_probs = [0.1, 0.8, 0.1]
                 
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        
        return {
            "current_price": df['Close'].iloc[-1],
            "predicted_price": pred_price,
            "confidence_interval": (lower_bound, upper_bound),
            "signal": signal_map.get(int(signal), "UNKNOWN"),
            "signal_confidence": float(max(signal_probs)),
            "risk_level": "High" if std_pred_scaled > 0.05 else "Low"
        }
