@echo off
echo ===================================================
echo Setting up Stock Predictor Backend Environment
echo ===================================================

cd /d "%~dp0"

echo 1. Creating Python Virtual Environment...
python -m venv venv

echo 2. Activating Virtual Environment...
call venv\Scripts\activate

echo 3. Installing Dependencies...
pip install -r requirements.txt

echo 4. Running Verification Script...
python verify_setup.py

echo.
echo Setup Complete! To run the server:
echo    venv\Scripts\activate
echo    uvicorn app.main:app --reload
echo.
pause
