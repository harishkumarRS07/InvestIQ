from fastapi import APIRouter, HTTPException, BackgroundTasks
import os
from backend.app.schemas import PredictionRequest, PredictionResponse, TrainRequest, TrainResponse
from backend.inference.predict import Predictor
from backend.training.train import train_pipeline
from backend.core.config import settings
from backend.core.logging import logger
from backend.core.exceptions import StockPredictorException

router = APIRouter()

# Instantiate predictor globally (lazy load models)
predictor = Predictor()

@router.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Determine file path
        # If file_path is provided, use it. Else construct from symbol.
        if request.file_path:
            path = request.file_path
        else:
            path = os.path.join(settings.DATA_DIR, f"{request.symbol}.csv")
            
        result = predictor.predict(path, ticker=request.symbol)
        
        return PredictionResponse(
            symbol=request.symbol,
            current_price=result['current_price'],
            predicted_price=result['predicted_price'],
            confidence_interval=result['confidence_interval'],
            signal=result['signal'],
            signal_confidence=result['signal_confidence'],
            risk_level=result['risk_level']
        )
    except StockPredictorException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    try:
        # Check if file exists
        if not os.path.exists(request.file_path):
             raise HTTPException(status_code=404, detail="Training file not found")
             
        # Run training in background to avoid blocking
        background_tasks.add_task(train_pipeline, request.file_path)
        
        return TrainResponse(
            status="accepted",
            message="Training started in background"
        )
    except Exception as e:
        logger.error(f"Training launch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from backend.app.schemas import (
    SentimentRequest, SentimentResponse,
    PortfolioRequest, PortfolioResponse,
    RiskRequest, RiskResponse
)
from backend.features.sentiment import sentiment_analyzer
from backend.data.realtime import RealTimeDataFetcher
from backend.features.portfolio import PortfolioOptimizer
from backend.features.risk import RiskEngine
import pandas as pd

realtime_fetcher = RealTimeDataFetcher()
portfolio_optimizer = PortfolioOptimizer()

@router.post("/sentiment/analyze", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    try:
        if request.symbol:
            # Fetch live news for symbol
            # This logic ideally belongs in data/realtime or features/sentiment
            # For now, we use the realtime fetcher if it had a news method, 
            # but we can fallback to the sentiment analyzer's logic or implement it here
            # Simulating fetched news for now as RealTimeDataFetcher doesn't explicitly return news text yet
            # We will use the live sentiment fetcher from external_data.py but upgraded
            from backend.features.external_data import ExternalDataSimulator
            score = ExternalDataSimulator.fetch_live_sentiment(request.symbol)
            # Todo: use FinBERT on fetched headlines
            
        elif request.text:
            score = sentiment_analyzer.analyze(request.text)
        else:
            raise HTTPException(status_code=400, detail="Either text or symbol must be provided.")
            
        label = "Neutral"
        if score > 0.1: label = "Positive"
        if score < -0.1: label = "Negative"
        
        return SentimentResponse(
            symbol=request.symbol,
            sentiment_score=score,
            sentiment_label=label
        )
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio/optimize", response_model=PortfolioResponse)
def optimize_portfolio(request: PortfolioRequest):
    try:
        # Fetch data for all symbols
        prices = pd.DataFrame()
        for symbol in request.symbols:
            df = realtime_fetcher.fetch_price_history(symbol, period=request.period)
            if not df.empty:
                prices[symbol] = df['Close']
        
        if prices.empty:
             raise HTTPException(status_code=404, detail="No data found for symbols")
             
        allocation = portfolio_optimizer.optimize(prices)
        
        # Calculate metrics for this allocation
        weights = [allocation.get(sym, 0) for sym in prices.columns]
        metrics = portfolio_optimizer.get_portfolio_metrics(weights, prices.pct_change().dropna())
        
        return PortfolioResponse(
            allocation=allocation,
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/score", response_model=RiskResponse)
def get_risk_score(request: RiskRequest):
    try:
        df = realtime_fetcher.fetch_price_history(request.symbol, period="1y")
        if df.empty:
             raise HTTPException(status_code=404, detail="Symbol not found")
             
        metrics = RiskEngine.get_risk_profile(df)
        
        return RiskResponse(
            symbol=request.symbol,
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Risk scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
