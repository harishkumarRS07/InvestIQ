from pydantic import BaseModel
from typing import Optional, List, Tuple

class PredictionRequest(BaseModel):
    symbol: str
    file_path: Optional[str] = None # Optional override for direct file path

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    confidence_interval: Tuple[float, float]
    signal: str
    signal_confidence: float
    risk_level: str

class TrainRequest(BaseModel):
    file_path: str

class TrainResponse(BaseModel):
    status: str
    message: str
