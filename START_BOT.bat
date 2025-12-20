@echo off
REM SinfoloBot Easy Launcher
REM Double-click this file to start the bot menu

echo Starting SinfoloBot Launcher...
echo.

REM Check if venv exists
if exist "venv\Scripts\python.exe" (
    echo Using virtual environment...
    venv\Scripts\python.exe start_bot.py
) else (
    echo Using system Python...
    python start_bot.py
)

pause
