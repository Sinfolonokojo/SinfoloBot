"""
Parameter Optimization
Try different parameter combinations to find better performance
"""

import yaml
import MetaTrader5 as mt5
from backtesting.backtest_engine import BacktestEngine
from strategies.sma_crossover import SMACrossoverStrategy
from core.mt5_data import MT5DataFetcher
from core.risk_manager import RiskManager

def test_sma_params(fast, slow, symbol, timeframe, config):
    """Test SMA crossover with specific parameters"""

    # Fetch data
    fetcher = MT5DataFetcher()
    bars_per_day = {'H1': 24, 'H4': 6, 'D1': 1}
    num_bars = 365 * bars_per_day.get(timeframe, 24)

    data = fetcher.get_historical_data(symbol, timeframe, num_bars=num_bars)
    if data is None or len(data) < max(fast, slow) + 50:
        return None

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return None

    symbol_dict = {
        'volume_min': symbol_info.volume_min,
        'volume_step': symbol_info.volume_step,
        'point': symbol_info.point,
        'digits': symbol_info.digits,
    }

    # Create strategy with custom params
    strategy_config = {
        'fast_period': fast,
        'slow_period': slow,
        'timeframe': timeframe
    }
    strategy = SMACrossoverStrategy(config=strategy_config)

    # Run backtest
    risk_manager = RiskManager(config['risk'])
    backtest = BacktestEngine(strategy, risk_manager, initial_balance=10000)
    results = backtest.run(data, symbol_dict)

    return {
        'fast': fast,
        'slow': slow,
        'return': results['total_return'],
        'win_rate': results['win_rate'],
        'sharpe': results['sharpe_ratio'],
        'max_dd': results['max_drawdown'],
        'trades': results['num_trades']
    }

def optimize_sma():
    """Optimize SMA parameters"""

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if not mt5.initialize():
        print("[ERROR] MT5 initialization failed")
        return

    # Test on the "best" combination from comprehensive test
    symbol = "GBPUSD"
    timeframe = "H4"

    print("\n" + "="*80)
    print(f"  OPTIMIZING SMA CROSSOVER PARAMETERS")
    print(f"  Symbol: {symbol}, Timeframe: {timeframe}")
    print("="*80 + "\n")

    # Test different SMA period combinations
    fast_periods = [5, 10, 15, 20, 25]
    slow_periods = [20, 30, 40, 50, 60, 100, 200]

    results = []
    total_tests = len(fast_periods) * len(slow_periods)
    current = 0

    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:  # Skip invalid combinations
                continue

            current += 1
            print(f"[{current}/{total_tests}] Testing Fast={fast:3}, Slow={slow:3}...", end=" ")

            result = test_sma_params(fast, slow, symbol, timeframe, config)
            if result:
                results.append(result)
                status = "[PROFIT]" if result['return'] > 0 else "[LOSS]"
                print(f"{status} Return: {result['return']:>10.2f}%  Sharpe: {result['sharpe']:>6.2f}")
            else:
                print("FAILED")

    mt5.shutdown()

    # Sort by Sharpe ratio (risk-adjusted return)
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    print("\n" + "="*80)
    print("  TOP 10 PARAMETER COMBINATIONS (by Sharpe Ratio)")
    print("="*80 + "\n")

    print(f"{'Rank':<6} {'Fast':<6} {'Slow':<6} {'Return %':>12} {'Win %':>8} {'Sharpe':>8} {'Max DD %':>10} {'Trades':>8}")
    print("-"*80)

    for i, result in enumerate(results[:10], 1):
        print(f"{i:<6} {result['fast']:<6} {result['slow']:<6} {result['return']:>12.2f} "
              f"{result['win_rate']:>8.1f} {result['sharpe']:>8.2f} {result['max_dd']:>10.2f} {result['trades']:>8}")

    # Find profitable ones
    profitable = [r for r in results if r['return'] > 0]

    if profitable:
        print(f"\n[SUCCESS] Found {len(profitable)} profitable parameter combinations!")
        print("\nBest overall:")
        best = profitable[0]
        print(f"  Fast SMA: {best['fast']}")
        print(f"  Slow SMA: {best['slow']}")
        print(f"  Return: {best['return']:.2f}%")
        print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")

        print("\nTo use these parameters, update config/config.yaml:")
        print(f"  strategies:")
        print(f"    sma_crossover:")
        print(f"      fast_period: {best['fast']}")
        print(f"      slow_period: {best['slow']}")
        print(f"      timeframe: '{timeframe}'")
    else:
        print("\n[WARNING] No profitable parameters found even after optimization!")
        print("This strongly suggests the market doesn't suit this strategy.")
        print("\nLeast-bad combination:")
        if results:
            best = results[0]
            print(f"  Fast: {best['fast']}, Slow: {best['slow']}")
            print(f"  Loss: {best['return']:.2f}%")

    return results

if __name__ == "__main__":
    optimize_sma()
