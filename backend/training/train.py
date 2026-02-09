import torch
import glob
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from backend.core.config import settings
from backend.core.logging import logger
from backend.preprocessing.cleaning import load_data, clean_data
from backend.preprocessing.scaling import StockScaler
from backend.features.indicators import add_technical_indicators
from backend.models.lstm_attention import LSTMAttentionModel
from backend.models.xgboost_fusion import XGBoostFusionModel

def create_sequences(data: np.ndarray, seq_length: int, target_col_idx: int = 0):
    """Create sequences for LSTM"""
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length, target_col_idx]  # Predicting next step
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def train_pipeline(file_path: str):
    ticker = os.path.basename(file_path).replace(".csv", "")
    logger.info(f"Starting training pipeline for {ticker} from {file_path}")
    
    # 1. Load and Preprocess
    try:
        df = load_data(file_path)
    except Exception as e:
        logger.error(f"Skipping {ticker} due to load error: {e}")
        return f"Failed to load {ticker}"

    df = clean_data(df)
    df = add_technical_indicators(df)
    
    # Define features
    # Define features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'BB_High', 'BB_Low', 'VWAP', 'MACD', 'MACD_Signal', 'ATR', 'Log_Return']
    target_col = 'Log_Return'
    target_col_idx = feature_cols.index(target_col)
    
    # 2. Scale Data
    scaler = StockScaler()
    df_scaled = scaler.fit_transform(df, feature_cols)
    scaled_data = df_scaled[feature_cols].values
    
    # 3. Prepare LSTM Data
    X, y = create_sequences(scaled_data, settings.SEQ_LENGTH, target_col_idx)
    
    # Check if we have enough data
    if len(X) == 0:
        logger.warning(f"Not enough data for {ticker} to create sequences.")
        return f"Insufficient data for {ticker}"

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings.TEST_SIZE, shuffle=False)
    
    # Convert to Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    # 4. Train LSTM
    input_dim = X.shape[2]
    hidden_dim = 64
    lstm_model = LSTMAttentionModel(input_dim=input_dim, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=settings.LEARNING_RATE)
    
    logger.info(f"Training LSTM for {ticker}...")
    for epoch in range(settings.EPOCHS):
        lstm_model.train()
        optimizer.zero_grad()
        output = lstm_model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{settings.EPOCHS}, Loss: {loss.item():.6f}")
            
    # Save LSTM Model
    torch.save(lstm_model.state_dict(), os.path.join(settings.MODEL_DIR, f"lstm_attention_{ticker}.pth"))
    scaler.save(f"scaler_{ticker}.pkl")
    logger.info(f"LSTM Model and Scaler saved for {ticker}.")

    # Save LSTM Model
    torch.save(lstm_model.state_dict(), os.path.join(settings.MODEL_DIR, f"lstm_attention_{ticker}.pth"))
    scaler.save(f"scaler_{ticker}.pkl")
    logger.info(f"LSTM Model and Scaler saved for {ticker}.")

    # --- MODEL FUSION START ---
    
    # 5. Extract LSTM Features (The "Fusion" Step)
    lstm_model.eval()
    with torch.no_grad():
        # Get predictions for the entire X dataset (Train + Test)
        # Convert X to tensor
        X_tensor = torch.FloatTensor(X)
        lstm_preds = lstm_model(X_tensor).numpy() # Shape: (len(X), 1)
        
    # 6. Add External Data (News & Macro)
    from backend.features.external_data import ExternalDataSimulator
    
    # We need to align external data with the sequences
    # X contains data from index [0...seq_len] to [N...N+seq_len]
    # The target y is at [seq_length...N+seq_length]
    # We want features corresponding to the prediction time `t`?
    # Actually, XGBoost tries to predict "Signal" at time `t` based on features at `t`.
    # LSTM prediction at `t` is based on `t-seq_len` to `t`.
    # So `lstm_preds[i]` corresponds to the prediction made using sequence `i`.
    # Sequence `i` ends at index `i + seq_length - 1`. The prediction is for `i + seq_length`.
    
    # Let's enrich the original `df_scaled` (or rather a subset of it) with External Data first.
    # Actually, generating random data for the `trimmed` dataset `X` is easier.
    
    # Generate Mock Data matching the length of X
    # In reality, you'd match dates. Here we just generate N samples.
    n_samples = len(X)
    sentiments = np.random.uniform(-1.0, 1.0, size=(n_samples, 1))
    macros = np.random.uniform(40, 80, size=(n_samples, 1))
    
    # Fuse Features: [Original_Last_Step_Features, LSTM_Prediction, Sentiment, Macro]
    # X shape: (samples, seq_len, features)
    # We take the last time step of the input sequence as the "current state" for XGBoost
    # X_last_step = X[:, -1, :] # Shape: (samples, features)
    
    # Wait, in original code:
    # X_xgb = df_scaled[feature_cols].iloc[:-5].values 
    # original code aligned XGBoost features with Labels.
    # labels = prepare_labels(df) (which generates labels for t using t+5)
    
    # Let's align correctly.
    # LSTM X, y were created from `scaled_data`.
    # `y` in LSTM is `Log_Return` at `t+1`.
    
    # For XGBoost, we want to predict `Label` (Buy/Sell/Hold) at `t`, using info available at `t`.
    # Info available at `t`:
    # 1. Technical Indicators at `t` (from df)
    # 2. LSTM Prediction for `t+1` made at `t` (using history `t-seq`...`t`)
    # 3. Sentiment/Macro at `t`
    
    # `lstm_preds` are exactly #2.
    # `X` (LSTM input) corresponds to windows ending at `t`.
    # So `lstm_preds[i]` is the prediction made at `t`, using data ending at `t`.
    
    # We need to recreate `X_xgb` to match `lstm_preds` length.
    # LSTM `create_sequences` consumed `seq_length` rows.
    # So `lstm_preds` has length `len(df) - seq_length`.
    # `labels` generation usually drops the last `horizon` rows.
    
    # Let's rebuild the fusion dataset carefully.
    
    # 1. Base Features (from X last step?) OR from df directly?
    # Using df directly is safer for indices.
    # df_scaled len: N.
    # LSTM preds len: N - seq_length. Start index in df: seq_length.
    
    valid_indices_start = settings.SEQ_LENGTH
    
    # XGBoost Labels horizon
    horizon = 5
    # We lose last `horizon` rows for labels.
    valid_indices_end = len(df_scaled) - horizon
    
    # Intersection of valid ranges:
    # Start: SEQ_LENGTH
    # End: len(df) - horizon
    
    # Slice LSTM Preds
    # lstm_preds start at `seq_length`.
    # We need to drop the last `horizon` predictions to match labels.
    lstm_preds_trimmed = lstm_preds[:-horizon]
    
    # Slice External Data
    # Generate for full length then slice? Or generate for specific len?
    # Simulating for full `df` is cleaner.
    df_enriched = ExternalDataSimulator.add_external_features(df, ticker)
    # Extract columns
    sentiments_full = df_enriched['Sentiment'].values
    macros_full = df_enriched['Macro_Score'].values
    
    # Slice Features to match [SEQ_LENGTH : -horizon]
    # We need features at time `t`.
    X_xgb_base = df_scaled[feature_cols].iloc[valid_indices_start : -horizon].values
    sentiment_slice = sentiments_full[valid_indices_start : -horizon].reshape(-1, 1)
    macro_slice = macros_full[valid_indices_start : -horizon].reshape(-1, 1)
    
    # Labels
    xg_model = XGBoostFusionModel()
    all_labels = xg_model.prepare_labels(df, horizon=horizon) 
    # all_labels len = len(df) - horizon. 
    # We need to slice off the first `SEQ_LENGTH` to match our start.
    y_xgb = all_labels[valid_indices_start:]
    
    # Concatenate All Features
    # 1. Base Technicals
    # 2. LSTM Prediction (Future Insight)
    # 3. Sentiment
    # 4. Macro
    X_fused = np.hstack([X_xgb_base, lstm_preds_trimmed, sentiment_slice, macro_slice])
    
    # Scale External Features? 
    # XGBoost handles unscaled fine, but for consistency maybe? 
    # Sentiment is -1 to 1 (fine). Macro 40-80 (fine).
    # LSTM pred is scaled (fine).
    
    if len(X_fused) > 0:
        # Split - DISABLE SHUFFLE
        X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(X_fused, y_xgb, test_size=settings.TEST_SIZE, shuffle=False)
        
        # Use Test set for Early Stopping
        eval_set = [(X_xgb_train, y_xgb_train), (X_xgb_test, y_xgb_test)]
        
        xg_model.train(X_xgb_train, y_xgb_train, eval_set=eval_set)
        xg_model.save(f"xgboost_fusion_{ticker}.pkl")
    else:
        logger.warning(f"Not enough data for XGBoost Fusion for {ticker}")
    
    return f"Training Completed Successfully for {ticker}"

if __name__ == "__main__":
    # Train for all CSV files in the data directory
    csv_files = glob.glob(os.path.join(settings.DATA_DIR, "*.csv"))
    
    if not csv_files:
        logger.warning("No CSV files found in data directory.")
    
    for file_path in csv_files:
        # Skip sample.csv if we strictly want real data, but let's process it if it exists 
        # unless it is explicitly unwanted. Given user prompt, they replaced it, 
        # but if it exists we can just train on it too or skip it.
        # I'll process whatever is there.
        try:
            result = train_pipeline(file_path)
            print(result)
        except Exception as e:
            logger.error(f"Failed to train on {file_path}: {e}")
