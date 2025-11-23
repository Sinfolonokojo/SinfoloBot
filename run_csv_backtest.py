"""
CSV Historical Data Backtest Runner
Tests strategy using downloaded historical data from HistData.com
Supports multi-year backtesting with M1 data resampled to any timeframe
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from backtesting.backtest_engine import BacktestEngine
from strategies.trend_pullback import TrendPullbackStrategy
from core.risk_manager import RiskManager
import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_histdata_csv(file_path):
    """
    Load HistData ASCII format CSV file
    Format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
    """
    df = pd.read_csv(
        file_path,
        sep=';',
        header=None,
        names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    )

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
    df.set_index('datetime', inplace=True)

    return df


def load_multiple_years(data_folder, symbol, years=None):
    """
    Load and combine multiple years of historical data

    Args:
        data_folder: Path to folder containing CSV files
        symbol: Symbol name (e.g., 'EURUSD')
        years: List of years to load, or None for all available

    Returns:
        DataFrame with combined data
    """
    data_path = Path(data_folder)
    all_data = []

    # Find all CSV files for the symbol
    pattern = f"DAT_ASCII_{symbol}_M1_*.csv"
    files = sorted(data_path.glob(pattern))

    if not files:
        print(f"[ERROR] No files found matching {pattern} in {data_folder}")
        return None

    print(f"\nFound {len(files)} data files:")

    for file in files:
        # Extract year from filename
        year_str = file.stem.split('_')[-1]

        # Filter by years if specified
        if years and year_str[:4] not in [str(y) for y in years]:
            continue

        print(f"  Loading {file.name}...")
        df = load_histdata_csv(file)
        all_data.append(df)
        print(f"    -> {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    if not all_data:
        print("[ERROR] No data loaded")
        return None

    # Combine all data
    combined = pd.concat(all_data)
    combined = combined.sort_index()

    # Remove duplicates
    combined = combined[~combined.index.duplicated(keep='first')]

    print(f"\nTotal: {len(combined)} M1 bars")
    print(f"Period: {combined.index[0]} to {combined.index[-1]}")

    return combined


def resample_to_timeframe(df, timeframe):
    """
    Resample M1 data to higher timeframe

    Args:
        df: DataFrame with M1 OHLCV data
        timeframe: Target timeframe ('M5', 'M15', 'M30', 'H1', 'H4', 'D1')

    Returns:
        Resampled DataFrame
    """
    # Timeframe mapping
    tf_map = {
        'M1': '1min',
        'M5': '5min',
        'M15': '15min',
        'M30': '30min',
        'H1': '1h',
        'H4': '4h',
        'D1': '1D'
    }

    if timeframe not in tf_map:
        print(f"[ERROR] Unknown timeframe: {timeframe}")
        return None

    if timeframe == 'M1':
        return df

    # Resample OHLCV
    resampled = df.resample(tf_map[timeframe]).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Drop rows with NaN (incomplete candles)
    resampled = resampled.dropna()

    return resampled


def get_symbol_info(symbol):
    """
    Get symbol info for position sizing
    Returns dict with volume_min, volume_step, point, digits
    """
    # Symbol info for forex, gold, and indices
    symbol_info = {
        # Forex pairs
        'EURUSD': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.00001, 'digits': 5},
        'GBPUSD': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.00001, 'digits': 5},
        'USDJPY': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.001, 'digits': 3},
        'AUDUSD': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.00001, 'digits': 5},
        'USDCAD': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.00001, 'digits': 5},
        'NZDUSD': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.00001, 'digits': 5},
        'EURJPY': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.001, 'digits': 3},
        'GBPJPY': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.001, 'digits': 3},
        # Gold
        'XAUUSD': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.01, 'digits': 2},
        # US Indices
        'US30': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.01, 'digits': 2},
        'US100': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.01, 'digits': 2},
        'US500': {'volume_min': 0.01, 'volume_step': 0.01, 'point': 0.01, 'digits': 2},
    }

    return symbol_info.get(symbol, symbol_info['EURUSD'])


def run_csv_backtest(symbol, timeframe, data_folder, config, years=None):
    """
    Run backtest using CSV historical data

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe to test ('M5', 'H1', etc.)
        data_folder: Path to CSV data folder
        config: Configuration dict
        years: List of years to include, or None for all

    Returns:
        dict: Backtest results
    """
    print(f"\n{'=' * 60}")
    print(f"  Loading {symbol} data from CSV")
    print(f"{'=' * 60}")

    # Load M1 data
    m1_data = load_multiple_years(data_folder, symbol, years)

    if m1_data is None or len(m1_data) == 0:
        print(f"[ERROR] Failed to load data for {symbol}")
        return None

    # Resample to target timeframe
    print(f"\nResampling to {timeframe}...")
    data = resample_to_timeframe(m1_data, timeframe)
    print(f"Resampled: {len(data)} {timeframe} bars")

    # Get symbol info
    symbol_dict = get_symbol_info(symbol)

    # Get strategy config
    if symbol in config['strategies']:
        strategy_config = config['strategies'][symbol]
    else:
        strategy_config = config['strategies']['trend_pullback']

    # Create strategy
    strategy = TrendPullbackStrategy(config=strategy_config)

    # Create risk manager
    risk_config = config['risk'].copy()
    risk_manager = RiskManager(risk_config)

    # Create backtest engine with risk limits from config
    initial_balance = config['backtest']['initial_balance']
    max_daily_loss = config['risk'].get('max_daily_loss', {}).get('percentage', 4.0)
    max_total_dd = config['risk'].get('max_total_drawdown', {}).get('percentage', 10.0)

    backtest = BacktestEngine(
        strategy,
        risk_manager,
        initial_balance=initial_balance,
        commission=0.00007,
        slippage_pips=0.3,
        use_spread=True,
        avg_spread_pips=0.5,
        max_daily_loss_pct=max_daily_loss,
        max_total_drawdown_pct=max_total_dd
    )

    # Run backtest
    print(f"\nRunning backtest...")
    results = backtest.run(data, symbol_dict)

    return results


def plot_backtest_results(results, symbol):
    """
    Plot backtest results
    """
    if not results or 'trades' not in results:
        print("No results to plot")
        return

    df_trades = pd.DataFrame(results['trades'])
    if df_trades.empty:
        print("No trades to plot")
        return

    initial_balance = results.get('initial_balance', 10000)

    # Calculate equity curve
    df_trades = df_trades.sort_values('exit_time')
    df_trades['cumulative_profit'] = df_trades['profit'].cumsum()
    df_trades['balance'] = initial_balance + df_trades['cumulative_profit']

    # Calculate drawdown
    df_trades['peak'] = df_trades['balance'].cummax()
    df_trades['drawdown_pct'] = (df_trades['balance'] - df_trades['peak']) / df_trades['peak'] * 100

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])

    # Plot 1: Balance vs Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_trades['exit_time'], df_trades['balance'], 'b-', linewidth=1.5, label='Balance')
    ax1.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.7, label=f'Initial: ${initial_balance:,.0f}')
    ax1.fill_between(df_trades['exit_time'], initial_balance, df_trades['balance'],
                     where=df_trades['balance'] >= initial_balance, alpha=0.3, color='green')
    ax1.fill_between(df_trades['exit_time'], initial_balance, df_trades['balance'],
                     where=df_trades['balance'] < initial_balance, alpha=0.3, color='red')
    ax1.set_title(f'{symbol} - Account Balance Over Time (CSV Backtest)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Balance ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(df_trades['exit_time'], 0, df_trades['drawdown_pct'], color='red', alpha=0.5)
    ax2.plot(df_trades['exit_time'], df_trades['drawdown_pct'], 'r-', linewidth=1)
    ax2.set_title('Drawdown Over Time', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    max_dd = df_trades['drawdown_pct'].min()
    ax2.axhline(y=max_dd, color='darkred', linestyle='--', alpha=0.7, label=f'Max DD: {max_dd:.2f}%')
    ax2.legend(loc='lower left')

    # Plot 3: Trade Distribution
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

    # Plot 4: Win/Loss Stats
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

    for bar, val in zip(bars, values):
        x_pos = val + (10 if val >= 0 else -10)
        ax4.text(x_pos, bar.get_y() + bar.get_height()/2, f'${val:.2f}',
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    # Plot 5: Monthly Returns
    ax5 = fig.add_subplot(gs[2, 1])
    df_trades['month'] = pd.to_datetime(df_trades['exit_time']).dt.to_period('M')
    monthly = df_trades.groupby('month')['profit'].sum()
    colors = ['green' if p > 0 else 'red' for p in monthly.values]
    ax5.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
    ax5.set_title('Monthly Returns', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Profit ($)')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Set x-ticks for monthly
    if len(monthly) > 12:
        step = len(monthly) // 12
        ax5.set_xticks(range(0, len(monthly), step))
        ax5.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), step)], rotation=45)

    plt.tight_layout()
    plt.savefig(f'backtest_results_{symbol}_csv.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to 'backtest_results_{symbol}_csv.png'")


def main():
    """
    Main function to run CSV backtest
    """
    print("\n" + "=" * 70)
    print("  CSV HISTORICAL DATA BACKTESTER")
    print("  Using downloaded HistData.com data")
    print("=" * 70)

    # Load config
    with open('config/multi_pair_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Settings - Choose symbol
    print("\nSelect symbol:")
    print("1. EURUSD")
    print("2. GBPUSD")
    print("3. USDJPY")
    print("4. AUDUSD")
    print("5. USDCAD")

    try:
        sym_choice = input("\nSelect symbol (1-5): ").strip()
    except:
        sym_choice = "1"

    symbol_map = {
        "1": ("EURUSD", "historic_data_forex_eurusd"),
        "2": ("GBPUSD", "historic_data_forex_gbpusd"),
        "3": ("USDJPY", "historic_data_forex_usdjpy"),
        "4": ("AUDUSD", "historic_data_forex_audusd"),
        "5": ("USDCAD", "historic_data_forex_usdcad"),
    }

    symbol, data_folder = symbol_map.get(sym_choice, ("EURUSD", "historic_data_forex_eurusd"))

    print(f"\nSymbol: {symbol}")
    print(f"Data folder: {data_folder}")

    # Options
    print("\nOptions:")
    print("1. Test on M5 timeframe (5 years)")
    print("2. Test on H1 timeframe (5 years)")
    print("3. Test on M15 timeframe (5 years)")
    print("4. Custom timeframe")

    try:
        choice = input("\nSelect option (1-4): ").strip()
    except:
        choice = "1"

    if choice == "1":
        timeframe = "M5"
    elif choice == "2":
        timeframe = "H1"
    elif choice == "3":
        timeframe = "M15"
    elif choice == "4":
        timeframe = input("Enter timeframe (M5/M15/M30/H1/H4/D1): ").strip().upper()
    else:
        timeframe = "M5"

    print(f"\nTesting {symbol} on {timeframe}")

    # Run backtest
    results = run_csv_backtest(
        symbol=symbol,
        timeframe=timeframe,
        data_folder=data_folder,
        config=config,
        years=None  # All available years
    )

    if results and 'error' not in results:
        # Print results
        print("\n" + "=" * 70)
        print(f"  {symbol} BACKTEST RESULTS ({timeframe})")
        print("=" * 70)

        initial_balance = config['backtest']['initial_balance']
        final_balance = initial_balance + results['total_return']

        print(f"\nStarting Capital: ${initial_balance:,.2f}")
        print(f"Final Balance:    ${final_balance:,.2f}")
        print(f"Total Return:     ${results['total_return']:,.2f} ({results['return_pct']:.2f}%)")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {results['num_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")

        # Calculate average profit
        if results['num_trades'] > 0:
            avg_profit = results['total_return'] / results['num_trades']
            print(f"  Avg Profit/Trade: ${avg_profit:.2f}")

        # Performance grade
        return_pct = results['return_pct']
        if return_pct >= 100:
            grade = 'A+'
        elif return_pct >= 50:
            grade = 'A'
        elif return_pct >= 25:
            grade = 'B+'
        elif return_pct >= 10:
            grade = 'B'
        elif return_pct >= 0:
            grade = 'C'
        else:
            grade = 'D'

        print(f"\nPerformance Grade: {grade}")

        # Show if trading was halted
        if results.get('trading_halted'):
            print(f"\n⚠️  TRADING HALTED: {results['halt_reason']}")

        print("=" * 70)

        # Plot results
        plot_backtest_results(results, symbol)

        return results
    else:
        print("\n[ERROR] Backtest failed")
        return None


if __name__ == "__main__":
    main()
