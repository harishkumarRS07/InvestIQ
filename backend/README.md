# AI-Based Stock Predictor Backend

This project implements a modular, production-ready backend for stock prediction using Deep Learning (LSTM with Attention) and Decision Fusion (XGBoost).

## Architecture
The system is built with FastAPI and follows a clean directory structure:
- **app/**: API Routes and Schemas.
- **core/**: Configuration and Logging.
- **preprocessing/**: Data cleaning and scaling.
- **features/**: Technical indicator generation (RSI, SMA, VWAP, Bollinger Bands).
- **models/**:
  - `LSTMAttentionModel`: For Time-Series forecasting.
  - `XGBoostFusionModel`: For Buy/Hold/Sell signal classification.
- **training/**: Training pipelines.
- **inference/**: Prediction logic with Monte Carlo Dropout for confidence intervals.

## Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Data Setup**:
   Place your stock CSV files in `backend/data/stock_data/`.
   Format: `Date, Open, High, Low, Close, Volume`

3. **Run the API**:
   ```bash
   uvicorn backend.app.main:app --reload
   ```

## API Usage

### Health Check
`GET /api/v1/health`

### Train Model
`POST /api/v1/train`
```json
{
  "file_path": "backend/data/stock_data/sample.csv"
}
```

### Predict
`POST /api/v1/predict`
```json
{
  "symbol": "sample",
  "file_path": "backend/data/stock_data/sample.csv" 
}
```
*Note: `file_path` is optional if the symbol corresponds to a file in the data directory.*

## Features
- **Data Leakage Prevention**: Proper Train/Test splitting.
- **Confidence Intervals**: Uses Monte Carlo Dropout.
- **Signal Fusion**: Combines statistical indicators with ML models.
- **Scalable**: Modularity allows easy addition of new models or features.
