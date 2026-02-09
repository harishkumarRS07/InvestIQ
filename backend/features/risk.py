import numpy as np
import pandas as pd
from typing import Dict

class RiskEngine:
    """
    Calculates financial risk metrics.
    """
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 252) -> float:
        """
        Annualized volatility.
        """
        return returns.std() * np.sqrt(window)

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """
        Maximum Drawdown from peak.
        """
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown.min()

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """
        Sharpe Ratio (Annualized).
        Assuming daily returns.
        """
        excess_returns = returns - (risk_free_rate / 252)
        if returns.std() == 0: return 0.0
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """
        Sortino Ratio (penalizes only downside volatility).
        """
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        
        downside_std = downside_returns.std()
        if downside_std == 0: return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_std

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Value at Risk (VaR) using historical method.
        """
        return np.percentile(returns, (1 - confidence_level) * 100)

    @staticmethod
    def get_risk_profile(df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all risk metrics for a given dataframe with 'Close'.
        """
        if 'Close' not in df.columns:
            return {}
            
        prices = df['Close']
        returns = prices.pct_change().dropna()
        
        return {
            "volatility_annual": RiskEngine.calculate_volatility(returns),
            "max_drawdown": RiskEngine.calculate_max_drawdown(prices),
            "sharpe_ratio": RiskEngine.calculate_sharpe_ratio(returns),
            "sortino_ratio": RiskEngine.calculate_sortino_ratio(returns),
            "value_at_risk_95": RiskEngine.calculate_var(returns)
        }
