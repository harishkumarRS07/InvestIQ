import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Tuple
from backend.core.logging import logger

class PortfolioOptimizer:
    """
    Portfolio Optimization Engine.
    Uses Mean-Variance Optimization (MVO) to find optimal weights.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def optimize(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize portfolio weights for Maximum Sharpe Ratio.
        price_data: DataFrame with columns as Tickers and rows as Dates (Close prices).
        """
        # Calculate returns
        returns = price_data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)
        
        if num_assets < 2:
            logger.warning("Portfolio optimization requires at least 2 assets.")
            return {col: 1.0 for col in price_data.columns}

        # Objective function (Negative Sharpe Ratio)
        def negative_sharpe(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe_ratio

        # Constraints: Sum of weights = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: Weights between 0 and 1 (No short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial Guess: Equal weights
        init_guess = num_assets * [1. / num_assets,]
        
        try:
            result = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
            optimal_weights = result.x
            allocation = {ticker: round(weight, 4) for ticker, weight in zip(price_data.columns, optimal_weights)}
            
            logger.info(f"Optimal Portfolio Allocation: {allocation}")
            return allocation
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {col: 1.0/num_assets for col in price_data.columns}

    def get_portfolio_metrics(self, weights: List[float], returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate expected return and volatility for a given set of weights.
        """
        weights = np.array(weights)
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            "expected_annual_return": portfolio_return,
            "annual_volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio
        }
