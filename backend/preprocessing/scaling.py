import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
from backend.core.config import settings
from backend.core.logging import logger

class StockScaler:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = []

    def fit_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Fit scaler on columns and return transformed dataframe"""
        self.feature_columns = columns
        scaled_data = self.scaler.fit_transform(df[columns])
        df_scaled = df.copy()
        df_scaled[columns] = scaled_data
        logger.info(f"Scaled columns: {columns}")
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler"""
        if not self.feature_columns:
            raise ValueError("Scaler not fitted yet")
        
        scaled_data = self.scaler.transform(df[self.feature_columns])
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = scaled_data
        return df_scaled
    
    def inverse_transform(self, data, column_index: int = 0):
        """
        Inverse transform a specific column (usually Close price for prediction)
        Since the scaler is multivariate, we need to handle the dimensions.
        A dedicated scaler for target can be simpler, but here we assume
        we want to inverse transform the target which is usually part of features.
        """
        # Detailed inverse transform can be tricky if we only have the target column.
        # Often easier to keep a separate scaler for the target variable.
        pass

    def save(self, name: str = "scaler.pkl"):
        path = os.path.join(settings.MODEL_DIR, name)
        state = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(state, path)
        logger.info(f"Scaler saved to {path}")

    def load(self, name: str = "scaler.pkl"):
        path = os.path.join(settings.MODEL_DIR, name)
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.feature_columns = state['feature_columns']
        logger.info(f"Scaler loaded from {path}")
