@echo off
REM One-click startup script for Windows

REM Ensure the working directory is always the script's directory
cd /d "%~dp0"

echo ==========================================================
echo   Starting Breast Cancer Histopathology Classifier...
echo   Working Directory: %~dp0
echo ==========================================================

REM Create virtual environment if not present
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Checking dependencies...
pip install -r requirements.txt

REM Launch Streamlit App
echo Launching Streamlit...
streamlit run app.py

pause
