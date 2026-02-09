import numpy as np
import pandas as pd
from typing import List, Dict
from backend.core.logging import logger

class StackingEnsemble:
    """
    Combines predictions from multiple models.
    Level 0: LSTM (Regression/Price), XGBoost (Classification/Signal), Technical Indicators
    Level 1: Meta-learner (Simple Average or Weighted)
    """
    
    def __init__(self):
        # Weights can be learned or heuristic
        # Heuristic: 60% directional signal (XGBoost), 40% trend strength (LSTM)
        self.weights = {
            'xgboost': 0.6,
            'lstm': 0.4
        }
        
    def predict(self, lstm_pred_price: float, current_price: float, xgb_probs: np.ndarray) -> Dict[str, Any]:
        """
        Fusion logic to generate final signal.
        
        xgb_probs: [prob_sell, prob_hold, prob_buy]
        lstm_pred_price: Predicted price next step
        """
        
        # 1. Normalize LSTM signal to -1 (Sell), 0 (Hold), 1 (Buy)
        # Calculate implied return
        implied_return = (lstm_pred_price - current_price) / current_price
        
        # Convert implied return to a confidence score (-1 to 1)
        # Sigmoid-like mapping: tanh(100 * return)
        lstm_signal_score = np.tanh(implied_return * 100) 
        
        # 2. Convert XGBoost probs to scalar score
        # Buy prob - Sell prob (Hold is neutral)
        xgb_signal_score = xgb_probs[2] - xgb_probs[0]
        
        # 3. Weighted Fusion
        final_score = (self.weights['lstm'] * lstm_signal_score) + (self.weights['xgboost'] * xgb_signal_score)
        
        # 4. Final Decision
        # Thresholds: > 0.3 Buy, < -0.3 Sell, else Hold
        if final_score > 0.3:
            decision = "BUY"
        elif final_score < -0.3:
            decision = "SELL"
        else:
            decision = "HOLD"
            
        return {
            "final_score": float(final_score),
            "decision": decision,
            "components": {
                "lstm_score": float(lstm_signal_score),
                "xgb_score": float(xgb_signal_score)
            }
        }
