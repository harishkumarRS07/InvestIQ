@echo off
echo Starting AI Model Training...
echo This will fetch data and train deep learning models for HDFCBANK, RELIANCE, TCS, and INFY.
echo Please wait...
call backend\venv\Scripts\activate
python train_all.py
pause
