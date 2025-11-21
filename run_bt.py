"""
Ultra Scalping Backtest Runner
5-year backtest for the ultra scalping strategy
"""

import yaml
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from backtesting.backtest_engine import BacktestEngine
from strategies.ultra_scalping import UltraScalpingStrategy
from core.mt5_data import MT5DataFetcher
from core.risk_manager import RiskManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_backtest(symbol="EURUSD", timeframe="M1", years=5):
    """
    Run a backtest for the ultra scalping strategy

    Args:
        symbol: Trading symbol (default: EURUSD)
        timeframe: Timeframe (default: M1)
        years: Number of years to backtest (default: 5)
    """

    # Load config
    with open('config/ultra_scalping_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize MT5
    if not mt5.initialize():
        print("[ERROR] MT5 initialization failed")
        print("Make sure MetaTrader 5 is running and logged in")
        return None

    # Calculate days
    days = years * 365

    print(f"\n{'='*70}")
    print(f"  BACKTESTING: Ultra Scalping on {symbol} ({timeframe})")
    print(f"  Period: {years} years ({days} days)")
    print(f"{'='*70}\n")

    # Fetch historical data
    print(f"Fetching {years} years of {timeframe} data for {symbol}...")
    fetcher = MT5DataFetcher()

    # Select symbol in Market Watch (required for data access)
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select {symbol}")
        mt5.shutdown()
        return None

    # Map timeframe string to MT5 constant
    mt5_timeframe = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }.get(timeframe, mt5.TIMEFRAME_M1)

    # Calculate how many bars we need
    bars_per_day = {'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48, 'H1': 24, 'H4': 6, 'D1': 1}
    target_bars = days * bars_per_day.get(timeframe, 24)

    # MT5 has a limit of ~100,000 bars per request, fetch in chunks
    max_bars_per_request = 99000

    print(f"Target: {target_bars:,} bars, fetching in chunks of {max_bars_per_request:,}...")

    all_rates = []
    position = 0

    while position < target_bars:
        bars_to_fetch = min(max_bars_per_request, target_bars - position)

        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, position, bars_to_fetch)

        if rates is None or len(rates) == 0:
            if position == 0:
                error = mt5.last_error()
                print(f"[ERROR] Failed to fetch data: {error}")
                mt5.shutdown()
                return None
            else:
                print(f"Reached end of available data at {len(all_rates):,} bars")
                break

        all_rates.extend(rates)
        position += len(rates)
        print(f"Fetched {len(all_rates):,} bars...")

        # If we got fewer bars than requested, we've reached the end
        if len(rates) < bars_to_fetch:
            print(f"Reached end of available data")
            break

    if len(all_rates) == 0:
        print("[ERROR] No data fetched")
        mt5.shutdown()
        return None

    # Convert to DataFrame
    import numpy as np
    import pandas as pd

    # all_rates is a list of numpy structured array records
    # Convert to numpy array first, then to DataFrame
    rates_array = np.array(all_rates)
    data = pd.DataFrame(rates_array)

    # Reverse to get chronological order (position 0 = newest, so we need oldest first)
    data = data.iloc[::-1].reset_index(drop=True)

    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    data.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'tick_volume': 'Volume'
    }, inplace=True)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Calculate bars per day for display
    bars_per_day = {
        'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48,
        'H1': 24, 'H4': 6, 'D1': 1
    }

    actual_days = len(data) / bars_per_day.get(timeframe, 24)
    print(f"Loaded {len(data):,} bars ({actual_days:.1f} days)")
    print(f"Period: {data.index[0]} to {data.index[-1]}")

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Failed to get symbol info for {symbol}")
        mt5.shutdown()
        return None

    symbol_dict = {
        'volume_min': symbol_info.volume_min,
        'volume_step': symbol_info.volume_step,
        'point': symbol_info.point,
        'digits': symbol_info.digits,
    }

    # Create strategy with config
    strategy_config = config['strategies']['ultra_scalping']
    strategy = UltraScalpingStrategy(config=strategy_config)

    # Create risk manager
    risk_manager = RiskManager(config['risk'])

    # Create backtest engine with realistic costs
    initial_balance = config['backtest']['initial_balance']
    backtest = BacktestEngine(
        strategy,
        risk_manager,
        initial_balance=initial_balance,
        commission=0.00007,      # 0.7 pip round-trip
        slippage_pips=0.3,       # 0.3 pip slippage
        use_spread=True,
        avg_spread_pips=0.5      # Average spread for EUR/USD
    )

    # Run backtest
    print("\nRunning backtest... (this may take a while for M1 data)\n")
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
        plot_file = f"backtest_{symbol}_{timeframe}_{timestamp}.png"
        backtest.plot_results(results, save_path=plot_file)
        print(f"\n[SUCCESS] Backtest complete!")
        print(f"Plot saved to: {plot_file}")
    except Exception as e:
        print(f"[WARNING] Could not generate plot: {e}")
        print("Results printed above.")

    mt5.shutdown()
    return results


def run_multi_pair_backtest(years=5):
    """
    Run backtest on multiple pairs and compare results

    Args:
        years: Number of years to backtest
    """
    # Load config to get symbols
    with open('config/ultra_scalping_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get top pairs for testing
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'EURJPY']

    print(f"\n{'='*70}")
    print(f"  MULTI-PAIR BACKTEST: {years} Years")
    print(f"{'='*70}\n")

    all_results = {}

    for symbol in symbols:
        print(f"\n--- Testing {symbol} ---")
        results = run_backtest(symbol, "M1", years)
        if results and 'error' not in results:
            all_results[symbol] = results

    # Print comparison
    if len(all_results) > 0:
        print(f"\n{'='*70}")
        print("  MULTI-PAIR COMPARISON")
        print(f"{'='*70}\n")
        print(f"{'Symbol':<10} {'Return %':<12} {'Win Rate':<10} {'Trades':<10} {'Max DD %':<10} {'Sharpe':<10}")
        print("-" * 70)

        for symbol, res in all_results.items():
            print(f"{symbol:<10} {res['return_pct']:<12.2f} {res['win_rate']:<10.1f} "
                  f"{res['num_trades']:<10} {res['max_drawdown']:<10.2f} {res['sharpe_ratio']:<10.2f}")

        # Find best performer
        best_symbol = max(all_results.keys(), key=lambda x: all_results[x]['return_pct'])
        print(f"\nBest performer: {best_symbol} with {all_results[best_symbol]['return_pct']:.2f}% return")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ULTRA SCALPING STRATEGY BACKTESTER")
    print("="*70 + "\n")

    print("Options:")
    print("1. Single pair backtest (EURUSD, 5 years)")
    print("2. Multi-pair backtest comparison")
    print("3. Custom backtest")

    try:
        choice = input("\nSelect option (1-3): ").strip()
    except:
        choice = "1"

    if choice == "1":
        run_backtest("EURUSD", "M1", 5)
    elif choice == "2":
        run_multi_pair_backtest(5)
    elif choice == "3":
        try:
            symbol = input("Symbol (default EURUSD): ").strip() or "EURUSD"
            years = int(input("Years (default 5): ").strip() or "5")
            run_backtest(symbol, "M1", years)
        except:
            print("Invalid input, running default...")
            run_backtest("EURUSD", "M1", 5)
    else:
        run_backtest("EURUSD", "M1", 5)
