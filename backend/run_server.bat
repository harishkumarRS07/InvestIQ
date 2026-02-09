@echo off
cd /d "%~dp0"
call venv\Scripts\activate
echo Starting Stock Predictor API...
uvicorn app.main:app --reload
pause
