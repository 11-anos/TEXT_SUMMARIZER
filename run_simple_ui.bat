@echo off
echo Starting Simple Maintenance Text Summarizer...
echo.
echo This will launch a clean and simple Streamlit web application.
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
echo Starting Simple Streamlit UI...
streamlit run simple_ui.py --server.port 8501

pause