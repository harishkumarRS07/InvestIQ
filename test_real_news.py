
import yfinance as yf
from textblob import TextBlob
import sys
import subprocess

# Ensure TextBlob is installed
try:
    import textblob
except ImportError:
    print("Installing TextBlob...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
    import textblob

try:
    import yfinance
except ImportError:
    print("Installing yfinance...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance

def get_real_sentiment(ticker_symbol):
    print(f"Fetching news for {ticker_symbol}...")
    
    # Add .NS for NSE stocks if not present
    if not ticker_symbol.endswith(".NS") and not ticker_symbol.endswith(".BO"):
         # Simple heuristic for Indian stocks in this project
         ticker_symbol += ".NS"
         
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        
        if not news:
            print("No news found via yfinance. Returning Neutral (0.0).")
            return 0.0
            
        print(f"Found {len(news)} recent articles.")
        
        total_sentiment = 0
        count = 0
        
        for article in news:
            title = article.get('title', '')
            # Publisher could be useful too
            print(f" - {title}")
            
            # Analyze Sentiment
            analysis = TextBlob(title)
            sentiment = analysis.sentiment.polarity
            total_sentiment += sentiment
            count += 1
            
        if count == 0:
            return 0.0
            
        avg_sentiment = total_sentiment / count
        print(f"\nAverage Sentiment Score: {avg_sentiment:.4f} (-1.0 to 1.0)")
        return avg_sentiment

    except Exception as e:
        print(f"Error fetching news: {e}")
        return 0.0

if __name__ == "__main__":
    ticker = "HDFCBANK"
    score = get_real_sentiment(ticker)
    
    print("\nInterpretation:")
    if score > 0.1: print("Market Mood: POSITIVE/BULLISH")
    elif score < -0.1: print("Market Mood: NEGATIVE/BEARISH")
    else: print("Market Mood: NEUTRAL")
