from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict

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

class SentimentRequest(BaseModel):
    text: Optional[str] = None
    symbol: Optional[str] = None # If symbol provided, fetch live news

class SentimentResponse(BaseModel):
    symbol: Optional[str]
    sentiment_score: float
    sentiment_label: str # Positive/Negative/Neutral

class PortfolioRequest(BaseModel):
    symbols: List[str]
    period: str = "1y"

class PortfolioResponse(BaseModel):
    allocation: Dict[str, float]
    metrics: Dict[str, float]

class RiskRequest(BaseModel):
    symbol: str

class RiskResponse(BaseModel):
    symbol: str
    metrics: Dict[str, float]
