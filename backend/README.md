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

## Setup & Installation (Local Windows)

1. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Data Setup**:
   - Place your stock CSV files in `backend/data/stock_data/`.
   - Ensure you have `HDFCBANK.csv` or similar for testing.

3. **Run the API**:
   - Double-click `run_app.bat`
   - OR run manually:
     ```bash
     uvicorn backend.app.main:app --reload
     ```

## API Usage

### Health Check
`GET /api/v1/health`

### New Features
- **Sentiment Analysis**: `POST /api/v1/sentiment/analyze`
- **Portfolio Optimization**: `POST /api/v1/portfolio/optimize`
- **Risk Score**: `POST /api/v1/risk/score`
- **Train Model**: `POST /api/v1/train`
- **Predict**: `POST /api/v1/predict`

## Features
- **Real-Time Data**: Integrates `yfinance` for live prices.
- **FinBERT Sentiment**: Uses institutional-grade NLP.
- **Risk & Portfolio Engines**: Advanced financial metrics and optimization.
- **Bidirectional LSTM**: Improved deep learning architecture.
- **Model Fusion**: Stacking ensemble for better accuracy.

