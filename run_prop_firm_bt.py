"""
Prop Firm Strategy Backtest Runner
Backtest the Trend Pullback strategy on M5 timeframe
"""

import yaml
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from backtesting.backtest_engine import BacktestEngine
from strategies.trend_pullback import TrendPullbackStrategy
from core.mt5_data import MT5DataFetcher
from core.risk_manager import RiskManager
import logging
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_backtest(symbol="EURUSD", timeframe="M5", years=1):
    """
    Run backtest for the Trend Pullback strategy

    Args:
        symbol: Trading symbol (default: EURUSD)
        timeframe: Timeframe (default: M5)
        years: Number of years to backtest (default: 1)
    """

    # Load config
    with open('config/prop_firm_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize MT5
    if not mt5.initialize():
        print("[ERROR] MT5 initialization failed")
        print("Make sure MetaTrader 5 is running and logged in")
        return None

    # Calculate days
    days = years * 365

    print(f"\n{'='*70}")
    print(f"  BACKTESTING: Trend Pullback on {symbol} ({timeframe})")
    print(f"  Period: {years} year(s) ({days} days)")
    print(f"{'='*70}\n")

    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select {symbol}")
        mt5.shutdown()
        return None

    # Map timeframe
    mt5_timeframe = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }.get(timeframe, mt5.TIMEFRAME_M5)

    # Calculate bars needed
    bars_per_day = {'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48, 'H1': 24, 'H4': 6, 'D1': 1}
    target_bars = days * bars_per_day.get(timeframe, 288)

    # Fetch data in chunks
    max_bars_per_request = 99000
    print(f"Target: {target_bars:,} bars, fetching in chunks...")

    all_rates = []
    position = 0

    while position < target_bars:
        bars_to_fetch = min(max_bars_per_request, target_bars - position)
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, position, bars_to_fetch)

        if rates is None or len(rates) == 0:
            if position == 0:
                print(f"[ERROR] Failed to fetch data: {mt5.last_error()}")
                mt5.shutdown()
                return None
            else:
                print(f"Reached end of available data at {len(all_rates):,} bars")
                break

        all_rates.extend(rates)
        position += len(rates)
        print(f"Fetched {len(all_rates):,} bars...")

        if len(rates) < bars_to_fetch:
            break

    if len(all_rates) == 0:
        print("[ERROR] No data fetched")
        mt5.shutdown()
        return None

    # Convert to DataFrame
    rates_array = np.array(all_rates)
    data = pd.DataFrame(rates_array)
    data = data.iloc[::-1].reset_index(drop=True)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    data.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'tick_volume': 'Volume'
    }, inplace=True)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    actual_days = len(data) / bars_per_day.get(timeframe, 288)
    print(f"\nLoaded {len(data):,} bars ({actual_days:.1f} days)")
    print(f"Period: {data.index[0]} to {data.index[-1]}")

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    symbol_dict = {
        'volume_min': symbol_info.volume_min,
        'volume_step': symbol_info.volume_step,
        'point': symbol_info.point,
        'digits': symbol_info.digits,
    }

    # Create strategy
    strategy_config = config['strategies']['trend_pullback']
    strategy = TrendPullbackStrategy(config=strategy_config)

    # Create risk manager
    risk_manager = RiskManager(config['risk'])

    # Create backtest engine
    initial_balance = config['backtest']['initial_balance']
    backtest = BacktestEngine(
        strategy,
        risk_manager,
        initial_balance=initial_balance,
        commission=0.00007,
        slippage_pips=0.3,
        use_spread=True,
        avg_spread_pips=0.5
    )

    # Run backtest
    print("\nRunning backtest...\n")
    results = backtest.run(data, symbol_dict)

    if results is None or 'error' in results:
        print(f"[ERROR] Backtest failed: {results.get('error', 'Unknown error')}")
        mt5.shutdown()
        return None

    # Print results
    backtest.print_summary(results)

    # Generate and save plot
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"backtest_prop_firm_{symbol}_{timeframe}_{timestamp}.png"
        backtest.plot_results(results, save_path=plot_file)
        print(f"\n[SUCCESS] Backtest complete!")
        print(f"Plot saved to: {plot_file}")
    except Exception as e:
        print(f"[WARNING] Could not generate plot: {e}")

    mt5.shutdown()
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  PROP FIRM STRATEGY BACKTESTER")
    print("  Strategy: Trend Pullback (EMA 50 + RSI 14)")
    print("="*70 + "\n")

    print("Options:")
    print("1. Backtest 1 year (recommended)")
    print("2. Backtest 6 months")
    print("3. Backtest 3 months")
    print("4. Custom")

    try:
        choice = input("\nSelect option (1-4): ").strip()
    except:
        choice = "1"

    if choice == "1":
        run_backtest("EURUSD", "M5", 1)
    elif choice == "2":
        run_backtest("EURUSD", "M5", 0.5)
    elif choice == "3":
        run_backtest("EURUSD", "M5", 0.25)
    elif choice == "4":
        try:
            symbol = input("Symbol (default EURUSD): ").strip() or "EURUSD"
            years = float(input("Years (default 1): ").strip() or "1")
            run_backtest(symbol, "M5", years)
        except:
            print("Invalid input, running default...")
            run_backtest("EURUSD", "M5", 1)
    else:
        run_backtest("EURUSD", "M5", 1)
