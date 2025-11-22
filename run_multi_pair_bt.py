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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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


def plot_backtest_results(results):
    """
    Plot backtest results: Balance vs Time, Drawdown vs Time, Max Win/Loss
    """
    if not results or 'trades' not in results:
        print("No results to plot")
        return

    df_trades = pd.DataFrame(results['trades'])
    if df_trades.empty:
        print("No trades to plot")
        return

    initial_balance = results['combined']['initial_balance']

    # Calculate equity curve
    df_trades = df_trades.sort_values('exit_time')
    df_trades['cumulative_profit'] = df_trades['profit'].cumsum()
    df_trades['balance'] = initial_balance + df_trades['cumulative_profit']

    # Calculate drawdown
    df_trades['peak'] = df_trades['balance'].cummax()
    df_trades['drawdown_pct'] = (df_trades['balance'] - df_trades['peak']) / df_trades['peak'] * 100

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])

    # Plot 1: Balance vs Time (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_trades['exit_time'], df_trades['balance'], 'b-', linewidth=1.5, label='Balance')
    ax1.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.7, label=f'Initial: ${initial_balance:,.0f}')
    ax1.fill_between(df_trades['exit_time'], initial_balance, df_trades['balance'],
                     where=df_trades['balance'] >= initial_balance, alpha=0.3, color='green')
    ax1.fill_between(df_trades['exit_time'], initial_balance, df_trades['balance'],
                     where=df_trades['balance'] < initial_balance, alpha=0.3, color='red')
    ax1.set_title('Account Balance Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Balance ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Drawdown vs Time (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(df_trades['exit_time'], 0, df_trades['drawdown_pct'],
                     color='red', alpha=0.5)
    ax2.plot(df_trades['exit_time'], df_trades['drawdown_pct'], 'r-', linewidth=1)
    ax2.set_title('Drawdown Over Time', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    max_dd = df_trades['drawdown_pct'].min()
    ax2.axhline(y=max_dd, color='darkred', linestyle='--', alpha=0.7,
                label=f'Max DD: {max_dd:.2f}%')
    ax2.legend(loc='lower left')

    # Plot 3: Trade Distribution (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    wins = df_trades[df_trades['profit'] > 0]['profit']
    losses = df_trades[df_trades['profit'] < 0]['profit']
    ax3.hist(wins, bins=20, alpha=0.7, color='green', label=f'Wins ({len(wins)})')
    ax3.hist(losses, bins=20, alpha=0.7, color='red', label=f'Losses ({len(losses)})')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Trade Profit Distribution', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Profit ($)')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Max Win/Loss Stats (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    max_win = df_trades['profit'].max()
    max_loss = df_trades['profit'].min()
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    categories = ['Max Win', 'Avg Win', 'Avg Loss', 'Max Loss']
    values = [max_win, avg_win, avg_loss, max_loss]
    colors = ['darkgreen', 'green', 'red', 'darkred']

    bars = ax4.barh(categories, values, color=colors, alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Win/Loss Statistics', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Profit ($)')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        x_pos = val + (10 if val >= 0 else -10)
        ax4.text(x_pos, bar.get_y() + bar.get_height()/2, f'${val:.2f}',
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    # Plot 5: Profit by Pair (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    pair_profits = df_trades.groupby('symbol')['profit'].sum().sort_values()
    colors = ['green' if p > 0 else 'red' for p in pair_profits.values]
    bars = ax5.barh(pair_profits.index, pair_profits.values, color=colors, alpha=0.7)
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax5.set_title('Total Profit by Pair', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Total Profit ($)')

    # Add value labels
    for bar, val in zip(bars, pair_profits.values):
        x_pos = val + (5 if val >= 0 else -5)
        ax5.text(x_pos, bar.get_y() + bar.get_height()/2, f'${val:.0f}',
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to 'backtest_results.png'")


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

    results = None
    if choice == "1":
        results = run_multi_pair_backtest(1)
    elif choice == "2":
        results = run_multi_pair_backtest(0.5)
    elif choice == "3":
        results = run_multi_pair_backtest(0.25)
    elif choice == "4":
        try:
            years = float(input("Years (default 1): ").strip() or "1")
            results = run_multi_pair_backtest(years)
        except:
            print("Invalid input, running default...")
            results = run_multi_pair_backtest(1)
    else:
        results = run_multi_pair_backtest(1)

    # Plot results if available
    if results:
        plot_backtest_results(results)
