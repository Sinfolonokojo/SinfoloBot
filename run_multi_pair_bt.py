"""
Multi-Pair Strategy Backtest Runner
Tests multiple pairs with their optimal timeframes and combines results
"""

import yaml
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from backtesting.backtest_engine import BacktestEngine
from strategies.trend_pullback import TrendPullbackStrategy
from core.risk_manager import RiskManager
import logging
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_single_pair_backtest(symbol, timeframe, config, years=1):
    """
    Run backtest for a single pair

    Returns:
        dict: Backtest results or None if failed
    """
    # Calculate days
    days = years * 365

    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select {symbol}")
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
    all_rates = []
    position = 0

    while position < target_bars:
        bars_to_fetch = min(max_bars_per_request, target_bars - position)
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, position, bars_to_fetch)

        if rates is None or len(rates) == 0:
            if position == 0:
                return None
            else:
                break

        all_rates.extend(rates)
        position += len(rates)

        if len(rates) < bars_to_fetch:
            break

    if len(all_rates) == 0:
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

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    symbol_dict = {
        'volume_min': symbol_info.volume_min,
        'volume_step': symbol_info.volume_step,
        'point': symbol_info.point,
        'digits': symbol_info.digits,
    }

    # Get pair-specific strategy config
    if symbol in config['strategies']:
        strategy_config = config['strategies'][symbol]
    else:
        strategy_config = config['strategies']['trend_pullback']

    # Create strategy
    strategy = TrendPullbackStrategy(config=strategy_config)

    # Create risk manager with reduced risk for multi-pair
    risk_config = config['risk'].copy()
    risk_manager = RiskManager(risk_config)

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
    results = backtest.run(data, symbol_dict)

    return results


def run_multi_pair_backtest(years=1):
    """
    Run backtest for all enabled pairs and combine results
    """
    # Load config
    with open('config/multi_pair_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize MT5
    if not mt5.initialize():
        print("[ERROR] MT5 initialization failed")
        print("Make sure MetaTrader 5 is running and logged in")
        return None

    print("\n" + "=" * 70)
    print("  MULTI-PAIR BACKTEST")
    print("  Strategy: Trend Pullback (Optimized per pair)")
    print("=" * 70)

    # Get enabled pairs
    enabled_pairs = []
    for symbol, settings in config['trading']['pairs'].items():
        if settings.get('enabled', False):
            enabled_pairs.append({
                'symbol': symbol,
                'timeframe': settings['timeframe']
            })

    print(f"\nTesting {len(enabled_pairs)} pairs:")
    for pair in enabled_pairs:
        print(f"  - {pair['symbol']} on {pair['timeframe']}")

    # Run backtests for each pair
    all_results = {}
    combined_trades = []
    combined_equity = []
    total_profit = 0

    for pair in enabled_pairs:
        symbol = pair['symbol']
        timeframe = pair['timeframe']

        print(f"\n{'=' * 50}")
        print(f"  Testing {symbol} on {timeframe}")
        print(f"{'=' * 50}")

        results = run_single_pair_backtest(symbol, timeframe, config, years)

        if results and 'error' not in results:
            all_results[symbol] = results
            total_profit += results['total_return']

            # Add pair info to trades
            for trade in results['trades']:
                trade['symbol'] = symbol
                trade['timeframe'] = timeframe
                combined_trades.append(trade)

            print(f"\n{symbol} Results:")
            print(f"  Return: ${results['total_return']:+,.2f} ({results['return_pct']:+.2f}%)")
            print(f"  Win Rate: {results['win_rate']:.1f}%")
            print(f"  Trades: {results['num_trades']}")
            print(f"  Profit Factor: {results['profit_factor']:.2f}")
            print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
        else:
            print(f"\n{symbol}: No trades or error")
            all_results[symbol] = None

    # Calculate combined statistics
    print("\n" + "=" * 70)
    print("  COMBINED MULTI-PAIR RESULTS")
    print("=" * 70)

    if combined_trades:
        df_trades = pd.DataFrame(combined_trades)

        # Sort by exit time
        df_trades = df_trades.sort_values('exit_time')

        # Calculate combined metrics
        initial_balance = config['backtest']['initial_balance']
        num_trades = len(df_trades)
        winning_trades = df_trades[df_trades['profit'] > 0]
        losing_trades = df_trades[df_trades['profit'] < 0]

        total_return = df_trades['profit'].sum()
        return_pct = (total_return / initial_balance) * 100
        win_rate = (len(winning_trades) / num_trades) * 100 if num_trades > 0 else 0

        avg_profit = df_trades['profit'].mean()
        avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0

        gross_profit = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate combined equity curve
        equity = initial_balance
        peak = initial_balance
        max_drawdown = 0

        for _, trade in df_trades.iterrows():
            equity += trade['profit']
            if equity > peak:
                peak = equity
            drawdown = (equity - peak) / peak * 100
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        final_balance = initial_balance + total_return

        # Print combined results
        print(f"\nStarting Capital: ${initial_balance:,.2f}")
        print(f"Final Balance:    ${final_balance:,.2f}")
        print(f"Total Return:     ${total_return:,.2f} ({return_pct:.2f}%)")

        print(f"\nTrades by Pair:")
        for symbol in all_results:
            if all_results[symbol]:
                pair_trades = df_trades[df_trades['symbol'] == symbol]
                pair_profit = pair_trades['profit'].sum()
                print(f"  {symbol}: {len(pair_trades)} trades, ${pair_profit:+,.2f}")

        print(f"\nCombined Statistics:")
        print(f"  Total Trades: {num_trades}")
        print(f"  Winning: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"  Losing: {len(losing_trades)}")

        print(f"\nProfit Metrics:")
        print(f"  Avg Profit: ${avg_profit:.2f}")
        print(f"  Avg Win: ${avg_win:.2f}")
        print(f"  Avg Loss: ${avg_loss:.2f}")
        print(f"  Best Trade: ${df_trades['profit'].max():.2f}")
        print(f"  Worst Trade: ${df_trades['profit'].min():.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Profit Factor: {profit_factor:.2f}")

        # Performance grade
        if return_pct >= 20:
            grade = 'A+'
        elif return_pct >= 10:
            grade = 'A'
        elif return_pct >= 5:
            grade = 'B'
        elif return_pct >= 0:
            grade = 'C'
        else:
            grade = 'D'

        print(f"\nPerformance Grade: {grade}")
        print("=" * 70)

        # Return combined results
        return {
            'pairs_tested': list(all_results.keys()),
            'individual_results': all_results,
            'combined': {
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'return_pct': return_pct,
                'num_trades': num_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'grade': grade
            },
            'trades': df_trades.to_dict('records')
        }

    else:
        print("\nNo trades executed across any pairs")
        return None

    mt5.shutdown()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MULTI-PAIR STRATEGY BACKTESTER")
    print("  Pairs: USDJPY (M5), EURUSD (H1), AUDUSD (M5)")
    print("=" * 70 + "\n")

    print("Options:")
    print("1. Backtest 1 year (recommended)")
    print("2. Backtest 6 months")
    print("3. Backtest 3 months")
    print("4. Custom period")

    try:
        choice = input("\nSelect option (1-4): ").strip()
    except:
        choice = "1"

    if choice == "1":
        run_multi_pair_backtest(1)
    elif choice == "2":
        run_multi_pair_backtest(0.5)
    elif choice == "3":
        run_multi_pair_backtest(0.25)
    elif choice == "4":
        try:
            years = float(input("Years (default 1): ").strip() or "1")
            run_multi_pair_backtest(years)
        except:
            print("Invalid input, running default...")
            run_multi_pair_backtest(1)
    else:
        run_multi_pair_backtest(1)
