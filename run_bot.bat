@echo off
echo ========================================
echo   Starting MT5 Trading Bot
echo ========================================
echo.
call venv\Scripts\activate.bat
python mt5_bot.py
echo.
echo ========================================
echo   Bot Stopped
echo ========================================
pause
