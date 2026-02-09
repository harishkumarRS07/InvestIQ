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
