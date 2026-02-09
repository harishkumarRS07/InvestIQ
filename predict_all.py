import urllib.request
import json
import os
import glob

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "data", "stock_data")

def get_available_tickers():
    """Find all CSV files in the data directory."""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tickers = [os.path.basename(f).replace(".csv", "") for f in csv_files]
    # Filter out sample if desired, but user might want it. Left in.
    return sorted(tickers)

def get_prediction(ticker):
    url = f"{BASE_URL}/predict"
    payload = json.dumps({"symbol": ticker}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                return json.loads(response.read().decode('utf-8'))
            else:
                return None
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def main():
    print("Discovering tickers...")
    tickers = get_available_tickers()
    
    if not tickers:
        print("No tickers found in data directory.")
        return

    print(f"Found {len(tickers)} companies: {', '.join(tickers)}")
    print("\nfetching predictions...\n")

    results = []
    for ticker in tickers:
        pred = get_prediction(ticker)
        if pred:
            results.append(pred)
        else:
            results.append({"symbol": ticker, "error": "Failed"})

    # Display Results
    print(f"{'Company':<15} | {'Current Price':<15} | {'Predicted (Next Day)':<20} | {'Signal':<10} | {'Confidence':<10}")
    print("-" * 80)
    
    for res in results:
        if "error" in res:
            print(f"{res['symbol']:<15} | {'ERROR':<15} | {'-':<20} | {'-':<10} | {'-':<10}")
        else:
            current = f"{res['current_price']:.2f}"
            predicted = f"{res['predicted_price']:.2f}"
            signal = res['signal']
            conf = f"{res['signal_confidence']:.2f}"
            print(f"{res['symbol']:<15} | {current:<15} | {predicted:<20} | {signal:<10} | {conf:<10}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
