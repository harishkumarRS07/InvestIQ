@echo off
echo Starting Stock Predictor API...
call backend\venv\Scripts\activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
pause
