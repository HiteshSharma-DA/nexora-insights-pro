@echo off
setlocal
cd /d "%~dp0"
title NEXORA INSIGHTS - Hitesh Sharma
echo.
echo   NEXORA INSIGHTS  ^|  Hitesh Sharma
echo   -----------------------------------
echo.
if not exist ".venv\Scripts\python.exe" (
    echo Creating a private Python environment for this app...
    python -m venv .venv
    if errorlevel 1 goto :failed
)
echo Installing or checking required packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
echo Starting the application. Keep this window open while using it.
echo Open http://localhost:8501 if your browser does not open automatically.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :failed
goto :end
:failed
echo.
echo ERROR: Setup or launch failed. See messages above.
echo Check that Python is installed and your internet connection works.
pause
:end
endlocal
