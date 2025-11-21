@echo off
echo ========================================
echo MT5 Trading Bot Dashboard
echo ========================================
echo.
echo [1/2] Starting API server...
echo.

REM Start Flask API in background
start "Dashboard API" cmd /c "venv\Scripts\python.exe dashboard_api.py"

REM Wait for API to start
timeout /t 3 /nobreak > nul

echo [2/2] Opening dashboard in browser...
echo.
echo ========================================
echo Dashboard is running!
echo ========================================
echo.
echo API Server: http://localhost:5000
echo Dashboard: Opening in browser...
echo.
echo Press Ctrl+C to stop the API server
echo ========================================
echo.

REM Open dashboard in default browser
start "" "dashboard.html"

REM Keep this window open to show status
echo Dashboard opened! You can close this window to stop the API.
pause
