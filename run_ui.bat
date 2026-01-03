@echo off
echo Starting Maintenance Text Summarizer UI...
echo.
echo This will launch a Streamlit web application in your browser.
echo.
echo To stop the application, press Ctrl+C in this window.
echo.
pause

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    
    echo Installing requirements...
    pip install -r requirements.txt
    echo.
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Starting Streamlit application...
streamlit run summarizer_ui.py --server.port 8501

pause