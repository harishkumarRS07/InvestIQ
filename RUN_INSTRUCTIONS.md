# How to Run the AI Financial Intelligence Platform

## Prerequisites
- Python 3.9+ installed
- Internet connection (for yFinance and FinBERT)

## Step 1: Install Dependencies
Open a terminal in the project root (`c:\Users\haris\OneDrive\Documents\invest iq`) and run:
```bash
pip install -r backend/requirements.txt
```

## Step 2: Run the Server
You have two options:
1.  **Double-click** the `run_app.bat` file in the folder.
2.  **OR** run this command in your terminal:
    ```bash
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

## Step 3: Verify & Interact
Once the server is running (you'll see `Uvicorn running on http://0.0.0.0:8000`), you can test the new features.

### Option A: Use the API Docs (Easiest)
1.  Open your browser to: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
2.  Expand **POST /api/v1/sentiment/analyze**, click **Try it out**, enter `{"text": "Market is crashing due to inflation"}` and execute.
3.  Expand **POST /api/v1/risk/score**, click **Try it out**, enter `{"symbol": "HDFCBANK"}` and execute.

### Option B: Use CURL / Terminal
**Check Sentiment:**
```bash
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" -H "Content-Type: application/json" -d "{\"text\": \"HDFC Bank reports record growth\"}"
```

**Get Risk Score:**
```bash
curl -X POST "http://localhost:8000/api/v1/risk/score" -H "Content-Type: application/json" -d "{\"symbol\": \"HDFCBANK\"}"
```

**Optimize Portfolio:**
```bash
curl -X POST "http://localhost:8000/api/v1/portfolio/optimize" -H "Content-Type: application/json" -d "{\"symbols\": [\"HDFCBANK\", \"RELIANCE.NS\", \"INFY.NS\"]}"
```
