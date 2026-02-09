import unittest
import pandas as pd
import numpy as np
from backend.features.risk import RiskEngine
from backend.features.portfolio import PortfolioOptimizer
from backend.features.timeframes import TimeFrameProcessor

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Create dummy daily data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.df = pd.DataFrame({
            "Date": dates,
            "Open": np.random.rand(100) * 100,
            "High": np.random.rand(100) * 100,
            "Low": np.random.rand(100) * 100,
            "Close": np.linspace(100, 110, 100) + np.random.normal(0, 1, 100),
            "Volume": np.random.randint(1000, 5000, 100)
        }).set_index("Date")
        
    def test_risk_metrics(self):
        metrics = RiskEngine.get_risk_profile(self.df)
        self.assertIn("volatility_annual", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        # Check values are float and not infinite
        self.assertTrue(np.isfinite(metrics["volatility_annual"]))

    def test_portfolio_optimization(self):
        # Create multi-asset dataframe
        prices = pd.DataFrame({
            "A": np.linspace(100, 200, 50) + np.random.normal(0, 5, 50),
            "B": np.linspace(50, 60, 50) + np.random.normal(0, 2, 50)
        })
        optimizer = PortfolioOptimizer()
        allocation = optimizer.optimize(prices)
        
        self.assertIn("A", allocation)
        self.assertIn("B", allocation)
        self.assertAlmostEqual(sum(allocation.values()), 1.0, places=4)

    def test_timeframe_resampling(self):
        processed = TimeFrameProcessor.create_multiframe_features(self.df)
        self.assertIn("daily", processed)
        self.assertIn("weekly", processed)
        self.assertIn("monthly", processed)
        
        # Check weekly resampling reduced size
        self.assertLess(len(processed["weekly"]), len(processed["daily"]))
        
if __name__ == "__main__":
    unittest.main()
