
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

# Define the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'stock_data')

# Map CSV filenames to yfinance tickers
# Assuming .BO as per the headers in the CSV files
TICKERS_MAP = {
    'HDFCBANK.csv': 'HDFCBANK.BO',
    'ICICIBANK.csv': 'ICICIBANK.BO',
    'INFY.csv': 'INFY.BO',
    'RELIANCE.csv': 'RELIANCE.BO',
    'TCS.csv': 'TCS.BO'
}

def update_stock_data():
    """
    Updates the stock data CSVs with the latest daily data from yfinance.
    Calculates and appends the Cumulative VWAP.
    """
    print(f"Starting stock data update at {datetime.now()}")
    
    for filename, ticker in TICKERS_MAP.items():
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping.")
            continue
            
        print(f"Processing {ticker} ({filename})...")
        
        try:
            # Load existing data
            df = pd.read_csv(file_path)
            
            # Ensure Date column is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            if df.empty:
                print(f"  {filename} is empty.")
                continue
                
            # Get the last date
            last_date = df['Date'].iloc[-1]
            print(f"  Last data date: {last_date.date()}")
            
            # Check if update is needed (if last date is before today)
            # Note: yfinance start date is inclusive, so we use last_date + 1 day
            start_date = last_date + timedelta(days=1)
            end_date = datetime.now()
            
            if start_date.date() >= end_date.date():
                print("  Data is already up to date.")
                continue
                
            # Fetch new data
            print(f"  Fetching data from {start_date.date()} to {end_date.date()}...")
            new_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if new_data.empty:
                print(f"  No new data found for {ticker}.")
                continue
                
            print(f"  Found {len(new_data)} new rows.")
            
            # Calculate cumulative values for VWAP from existing data
            # VWAP = Total_PV / Total_Volume
            # Total_PV = Last_VWAP * Total_Cumulative_Volume
            current_total_volume = df['Volume'].sum()
            last_vwap = df['VWAP'].iloc[-1]
            current_total_pv = last_vwap * current_total_volume
            
            new_rows = []
            
            # Process new data
            for index, row in new_data.iterrows():
                # Get basic values
                # yfinance usually returns multi-index columns if downloaded as list, 
                # but single ticker download might be different depending on version. 
                # Handling standard case: index is Date.
                
                # Check if columns are MultiIndex (common in recent yfinance)
                if isinstance(row['Open'], pd.Series):
                     # Extract scalar values
                    open_val = row['Open'].iloc[0]
                    high_val = row['High'].iloc[0]
                    low_val = row['Low'].iloc[0]
                    close_val = row['Close'].iloc[0]
                    vol_val = row['Volume'].iloc[0]
                else:
                    open_val = row['Open']
                    high_val = row['High']
                    low_val = row['Low']
                    close_val = row['Close']
                    vol_val = row['Volume']

                date_val = index
                
                # Calculate Typical Price and PV
                typical_price = (high_val + low_val + close_val) / 3.0
                pv = typical_price * vol_val
                
                # Update cumulatives
                current_total_pv += pv
                current_total_volume += vol_val
                
                # specific handling for zero volume to avoid division by zero (though unlikely in cumulative)
                if current_total_volume == 0:
                    vwap_val = 0
                else:
                    vwap_val = current_total_pv / current_total_volume
                
                new_rows.append({
                    'Date': date_val,
                    'Open': open_val,
                    'High': high_val,
                    'Low': low_val,
                    'Close': close_val,
                    'Volume': int(vol_val), # Convert to int
                    'VWAP': vwap_val
                })
            
            if new_rows:
                # Create DataFrame for new rows
                new_df = pd.DataFrame(new_rows)
                
                # Append to file
                # We assume the file is CSV and we just append to it
                # But to ensure formatting matches pandas default, we concat and save
                
                combined_df = pd.concat([df, new_df], ignore_index=True)
                
                # Formatting: preserve ticker header line if it existed?
                # The original file has a second line: ,RELIANCE.BO,RELIANCE.BO...
                # pandas read_csv with default args treats the first line as header and 2nd as data?
                # Let's re-verify line 2 of the file.
                # Line 1: Date,Open...
                # Line 2: ,RELIANCE.BO...
                # Line 3: 2000-01-04...
                
                # If I just used pd.read_csv(file_path), it likely made line 2 the first data row.
                # I should check if line 2 is actually skipped or treated as data.
                # If treated as data, df['Date'].iloc[-1] might be wrong if date parsing failed on it.
                
                # Re-reading carefully to skip the second line if it's metadata
                # Actually, standard pandas behavior on that file:
                # Line 1 -> Header
                # Line 2 -> Row 0. Date value matches ",RELIANCE.BO..." -> parsing to NaN or error?
                
                pass # Logic refined below
                
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

def refined_update():
    """
    Refined logic handling the specific CSV structure (2nd line junk).
    """
    for filename, ticker in TICKERS_MAP.items():
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path): continue
        
        print(f"Updating {filename}...")
        
        # Read with header=0 to get columns. 
        # Identify if row 0 is the metadata row.
        df = pd.read_csv(file_path)
        
        # Check if the first row is the ticker row (junk data in numeric cols)
        is_second_line_junk = False
        if df.iloc[0]['Open'] == ticker or df.iloc[0]['Open'] == str(ticker):
             is_second_line_junk = True
        
        # Reload properly handling dates
        # Use existing dataframe, just convert date, force errors="coerce"
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows with NaT in Date (this effectively removes the junk row if it failed parse)
        valid_data = df.dropna(subset=['Date']).copy()
        
        if valid_data.empty:
            print("No valid data found.")
            continue
            
        last_date = valid_data['Date'].iloc[-1]
        print(f"  Last date: {last_date.date()}")
        
        # Calculate date range
        start_date = last_date + timedelta(days=1)
        # yfinance 'end' is exclusive. To fetch 'today', end must be tomorrow.
        end_date = datetime.now() + timedelta(days=1)
        
        # Determine strict cutoff for "up to date"
        # If last_date is today, start_date is tomorrow.
        # If end_date is tomorrow. tomorrow >= tomorrow -> True (Skip).
        if start_date.date() >= end_date.date():
            print("  Up to date.")
            continue
            
        # Download data
        try:
            new_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        except Exception as e:
            print(f"Download failed: {e}")
            continue
            
        if new_data.empty:
            print("  No new data.")
            continue
            
        # Calculate VWAP
        # We need the sum of Volume and PV from valid_data
        # Ensure columns are numeric
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
        for c in cols:
            valid_data[c] = pd.to_numeric(valid_data[c], errors='coerce')
            
        current_total_volume = valid_data['Volume'].sum()
        # total_pv = sum(vwap_i * vol_i)? No.
        # The equation for cumulative VWAP at step T is: Sum(P_i * V_i for i=0..T) / Sum(V_i for i=0..T)
        # So Last_VWAP = Total_PV / Total_Volume
        # -> Total_PV = Last_VWAP * Total_Volume
        last_vwap = valid_data['VWAP'].iloc[-1]
        current_total_pv = last_vwap * current_total_volume
        
        new_rows = []
        for index, row in new_data.iterrows():
            # Handle yfinance dict/scalar return
            # ... (extraction logic)
            # To be safe against MultiIndex which yf returns now:
            try:
                # convert row to float
                op = float(row['Open'])
                hi = float(row['High'])
                lo = float(row['Low'])
                cl = float(row['Close'])
                vo = float(row['Volume'])
            except:
                # If validation implies it is a series
                op = float(row['Open'].iloc[0])
                hi = float(row['High'].iloc[0])
                lo = float(row['Low'].iloc[0])
                cl = float(row['Close'].iloc[0])
                vo = float(row['Volume'].iloc[0])
                
            date_str = index.strftime('%Y-%m-%d')
            
            typ_p = (hi + lo + cl) / 3.0
            pv = typ_p * vo
            current_total_pv += pv
            current_total_volume += vo
            
            if current_total_volume == 0: vwap = 0
            else: vwap = current_total_pv / current_total_volume
            
            new_rows.append({
                'Date': date_str,
                'Open': op,
                'High': hi,
                'Low': lo,
                'Close': cl,
                'Volume': int(vo),
                'VWAP': vwap
            })
            
        if new_rows:
            # We append directly to the file to avoid messing up the weird header
            # Or we can rewrite the file properly. 
            # Given the user might have other tools reading this, 
            # appending to the file is safest if we respect calculation.
            # But the file might have that 2nd line. 
            # If we rewrite, we might lose that 2nd line. 
            # Is that line important? It looks like column names again but with Ticker.
            # 'Date,Open,High,Low,Close,Volume,VWAP'
            # ',RELIANCE.BO,RELIANCE.BO,RELIANCE.BO,RELIANCE.BO,RELIANCE.BO,'
            # Just skipping it logic-wise is fine, but writing back...
            # I will Append in 'a' mode.
            
            with open(file_path, 'a') as f:
                # Ensure we start on a new line if not present
                # But CSV writer handles it.
                # Just formatting new_df to csv string without header
                
                new_df = pd.DataFrame(new_rows)
                # Ensure columns order
                new_df = new_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]
                
                # Write to string
                csv_data = new_df.to_csv(header=False, index=False, lineterminator='\n')
                f.write(csv_data)
                
            print(f"  Appended {len(new_rows)} rows.")

if __name__ == "__main__":
    refined_update()
