import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os
from backend.core.config import settings
from backend.core.logging import logger

class XGBoostFusionModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,  # 0: Sell, 1: Hold, 2: Buy
            n_estimators=500, # Increased from 100
            learning_rate=0.05, # Lower LR for better generalization
            max_depth=6,
            reg_alpha=0.1, # L1 Regularization
            reg_lambda=0.1, # L2 Regularization
            eval_metric='mlogloss',
            early_stopping_rounds=10, # Stop if no improvement
            random_state=42
        )

    def prepare_labels(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01):
        """
        Generate Buy/Sell/Hold labels based on future returns.
        Return = (Price[t+horizon] - Price[t]) / Price[t]
        Buy (2): Return > threshold
        Sell (0): Return < -threshold
        Hold (1): Otherwise
        """
        future_close = df['Close'].shift(-horizon)
        returns = (future_close - df['Close']) / df['Close']
        
        labels = np.zeros(len(df))
        labels[returns > threshold] = 2  # Buy
        labels[returns < -threshold] = 0 # Sell
        labels[(returns >= -threshold) & (returns <= threshold)] = 1 # Hold
        
        # Drop last 'horizon' rows as they have no labels
        return labels[:-horizon]

    def train(self, X: pd.DataFrame, y: np.ndarray, eval_set=None):
        """Train the model with optional validation set for early stopping"""
        logger.info(f"Training XGBoost model with {len(X)} samples")
        
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y, verbose=False)
            
        logger.info("XGBoost training completed")

    def predict(self, X: pd.DataFrame):
        """Return class predictions"""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """Return class probabilities"""
        return self.model.predict_proba(X)

    def save(self, name: str = "xgboost_fusion.pkl"):
        path = os.path.join(settings.MODEL_DIR, name)
        joblib.dump(self.model, path)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, name: str = "xgboost_fusion.pkl"):
        path = os.path.join(settings.MODEL_DIR, name)
        self.model = joblib.load(path)
        logger.info(f"XGBoost model loaded from {path}")
