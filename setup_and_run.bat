@echo off
echo ========================================
echo MT5 Ultra Scalping Bot - Setup & Run
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup first.
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [2/3] Installing pymongo for MongoDB...
pip install pymongo
if %errorlevel% neq 0 (
    echo WARNING: Failed to install pymongo
    echo MongoDB logging will be disabled
    echo Bot will still run normally
    echo.
)

echo.
echo [3/3] Starting Ultra Scalping Bot...
echo.
echo ========================================
echo Bot Configuration:
echo ========================================
echo Strategy: Ultra Scalping (M1 EUR/USD)
echo Risk per trade: 1%% ($100 on $10k)
echo Stop Loss: 5 pips
echo Take Profit: 10 pips (1:2 RR)
echo Market Filters: ENABLED
echo MongoDB Logging: ENABLED
echo ========================================
echo.
echo Press Ctrl+C to stop the bot
echo.

python run_ultra_scalping.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Bot failed to start
    echo Check the error messages above
    pause
)
