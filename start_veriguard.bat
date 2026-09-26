@echo off
title VeriGuard AI Deepfake Detection Platform
cd /d "%~dp0"

echo ===================================================
echo       VERIGUARD - AI FORENSIC DETECTION SYSTEM
echo ===================================================
echo.
echo [1/3] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo [2/3] Starting VeriGuard Server on http://127.0.0.1:8000 ...
echo [3/3] Opening browser...
start "" http://127.0.0.1:8000

echo.
echo Press Ctrl+C in this terminal anytime to stop the server.
echo.
python app.py --host 127.0.0.1 --port 8000
pause
