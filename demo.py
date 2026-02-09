import sys
import os
import asyncio
import pandas as pd
from backend.data.realtime import RealTimeDataFetcher
from backend.features.risk import RiskEngine
from backend.features.portfolio import PortfolioOptimizer
from backend.features.sentiment import SentimentAnalyzer

# Add project root to path
sys.path.append(os.getcwd())

def main():
    print("="*60)
    print("      AI FINANCIAL INTELLIGENCE PLATFORM - LIVE DEMO      ")
    print("="*60)

    # Data storage for final report
    report_data = {
        "risk": {},
        "sentiment": {},
        "portfolio": {},
        "predictions": []
    }

    # 1. Real-time Data
    print("\n[1] FETCHING REAL-TIME DATA...")
    fetcher = RealTimeDataFetcher()
    symbol = "HDFCBANK.NS"
    
    price = fetcher.get_current_price(symbol)
    print(f" -> Current Price of {symbol}: {price}")
    
    df = fetcher.fetch_price_history(symbol, period="1y")
    print(f" -> Fetched {len(df)} days of historical data.")

    # 2. Risk Analysis
    print("\n[2] CALCULATING RISK METRICS...")
    if not df.empty:
        risk_metrics = RiskEngine.get_risk_profile(df)
        report_data["risk"] = risk_metrics
        print(f" -> Annual Volatility: {risk_metrics['volatility_annual']:.2%}")
        print(f" -> Sharpe Ratio:      {risk_metrics['sharpe_ratio']:.2f}")
        print(f" -> Max Drawdown:      {risk_metrics['max_drawdown']:.2%}")
        print(f" -> Value at Risk (95%): {risk_metrics['value_at_risk_95']:.2%}")
    else:
        print(" -> Error: No data for risk analysis.")

    # 3. Sentiment Analysis
    print("\n[3] ANALYZING NEWS SENTIMENT...")
    sentiment = SentimentAnalyzer()
    headline = "HDFC Bank reports highest ever quarterly profit, creating positive outlook."
    print(f" -> Analyzing Headline: \"{headline}\"")
    
    score = sentiment.analyze(headline)
    label = "POSITIVE" if score > 0.1 else "NEGATIVE" if score < -0.1 else "NEUTRAL"
    report_data["sentiment"] = {"headline": headline, "score": score, "label": label}
    print(f" -> Sentiment Score: {score:.4f} ({label})")

    # 4. Portfolio Optimization
    print("\n[4] OPTIMIZING PORTFOLIO...")
    tickers = ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS"]
    print(f" -> Optimizing allocation for: {tickers}")
    
    portfolio_engine = PortfolioOptimizer()
    
    # Fetch data for all
    prices = pd.DataFrame()
    for t in tickers:
        print(f"    Fetching {t}...", end="\r")
        hist = fetcher.fetch_price_history(t, period="1y")
        if not hist.empty:
            prices[t] = hist['Close']
    print("    Data fetch complete.        ")
            
    if not prices.empty:
        allocation = portfolio_engine.optimize(prices)
        report_data["portfolio"]["allocation"] = allocation
        print(" -> Optimal Allocation (Max Sharpe):")
        for ticker, weight in allocation.items():
            print(f"    - {ticker}: {weight:.1%}")
            
        # Metrics
        weights = [allocation.get(t, 0) for t in prices.columns]
        metrics = portfolio_engine.get_portfolio_metrics(weights, prices.pct_change().dropna())
        report_data["portfolio"]["metrics"] = metrics
        print(f" -> Expected Annual Return: {metrics['expected_annual_return']:.2%}")
        print(f" -> Portfolio Volatility:   {metrics['annual_volatility']:.2%}")


    # 5. Price Prediction & Signals
    print("\n[5] GENERATING TRADING SIGNALS (AI PREDICTION)...")
    print("-" * 60)
    print(f"{'TICKER':<15} {'CURRENT PRICE':<15} {'PRED PRICE (7D)':<20} {'SIGNAL':<10} {'CONFIDENCE':<10}")
    print("-" * 60)
    
    from backend.inference.predict import Predictor
    from backend.core.exceptions import ModelNotTrainedException
    
    predictor = Predictor()
    
    for t in tickers:
        # Get latest price
        current_price = fetcher.get_current_price(t)
        if not current_price: continue
            
        # Create a temporary CSV for the predictor to use (it expects a file)
        # We use the history we fetched earlier or fetch fresh
        hist = fetcher.fetch_price_history(t, period="2y") # Need enough data for sequences
        if hist.empty: continue
        
        temp_csv = f"temp_{t}.csv"
        hist.to_csv(temp_csv)
        
        try:
            # Real AI Prediction
            result = predictor.predict(temp_csv, ticker=t)
            
            pred_price = result['predicted_price']
            signal = result['signal']
            confidence = result['signal_confidence']
            
            # Store for final report
            report_data["predictions"].append({
                "ticker": t,
                "current": current_price,
                "predicted": pred_price,
                "signal": signal,
                "confidence": confidence,
                "risk_level": result.get("risk_level", "N/A")
            })

            print(f"{t:<15} {current_price:<15.2f} {pred_price:<20.2f} {signal:<10} {confidence:<10.1%}")
            
        except ModelNotTrainedException:
            print(f"{t:<15} {current_price:<15.2f} {'(Not Trained)':<20} {'N/A':<10} {'0.0%':<10}")
        except Exception as e:
            print(f"{t:<15} Error: {e}")
            
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

    print("-" * 60)

    # ==========================================
    # FINAL EXECUTIVE SUMMARY
    # ==========================================
    print("\n\n")
    print("="*80)
    print("                      FINAL EXECUTIVE REPORT")
    print("="*80)
    
    # 1. Prediction Table
    print("\n1. AI TRADING SIGNALS")
    print("-" * 80)
    print(f"{'TICKER':<15} {'CURRENT PRICE':<15} {'PRED PRICE':<15} {'SIGNAL':<10} {'CONFIDENCE':<12} {'RISK':<10}")
    print("-" * 80)
    for p in report_data["predictions"]:
        print(f"{p['ticker']:<15} {p['current']:<15.2f} {p['predicted']:<15.2f} {p['signal']:<10} {p['confidence']:<12.1%} {p['risk_level']:<10}")
    print("-" * 80)

    # 2. Portfolio
    print("\n2. PORTFOLIO RECOMMENDATION")
    print("-" * 80)
    if "allocation" in report_data["portfolio"]:
        allo = report_data["portfolio"]["allocation"]
        met = report_data["portfolio"]["metrics"]
        
        # Format as a row of tickers
        alloc_str = ", ".join([f"{k}: {v:.1%}" for k, v in allo.items() if v > 0.001])
        print(f"Optimal Allocation: {alloc_str}")
        print(f"Est. Annual Return: {met['expected_annual_return']:.2%}  |  Volatility: {met['annual_volatility']:.2%}")
    else:
        print("No portfolio data generated.")

    # 3. Market Insights
    print("\n3. MARKET INSIGHTS (SAMPLE: HDFCBANK.NS)")
    print("-" * 80)
    if report_data["risk"]:
        r = report_data["risk"]
        print(f"Risk Metrics: Volatility={r['volatility_annual']:.2%}, Max Drawdown={r['max_drawdown']:.2%}, Sharpe={r['sharpe_ratio']:.2f}")
    
    if report_data["sentiment"]:
        s = report_data["sentiment"]
        print(f"Sentiment Analysis: {s['label']} ({s['score']:.4f})")
        print(f"Headline: \"{s['headline']}\"")
        
    print("="*80)
    print("\n")

if __name__ == "__main__":
    main()

