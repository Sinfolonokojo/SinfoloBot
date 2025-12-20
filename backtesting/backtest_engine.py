"""
Backtesting Engine for MT5 Trading Strategies
Allows testing strategies on historical data before live trading.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mt5_data import MT5DataFetcher
from core.risk_manager import RiskManager


class BacktestEngine:
    """Backtesting engine for strategy evaluation"""

    def __init__(self, strategy, risk_manager, initial_balance=10000, commission=7.0,
                 slippage_pips=0.3, use_spread=True, avg_spread_pips=0.5,
                 max_daily_loss_pct=4.0, max_total_drawdown_pct=10.0):
        """
        Initialize backtest engine

        Args:
            strategy: Strategy instance (BaseStrategy subclass)
            risk_manager: RiskManager instance
            initial_balance: Starting capital
            commission: Commission per lot round-trip in USD (default: $7 per lot)
            slippage_pips: Slippage in pips per trade (default: 0.3 pips)
            use_spread: Apply bid/ask spread simulation (default: True)
            avg_spread_pips: Average spread in pips (default: 0.5 for EUR/USD)
            max_daily_loss_pct: Maximum daily loss percentage (default: 4.0%)
            max_total_drawdown_pct: Maximum total drawdown percentage (default: 10.0%)
        """
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage_pips = slippage_pips
        self.use_spread = use_spread
        self.avg_spread_pips = avg_spread_pips
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_total_drawdown_pct = max_total_drawdown_pct
        self.logger = logging.getLogger(__name__)

        # Results tracking
        self.reset()

    def reset(self):
        """Reset backtest state"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.trades = []
        self.equity_curve = []
        self.current_position = None

        # Risk limit tracking
        self.daily_start_balance = self.initial_balance
        self.current_day = None
        self.trading_halted = False
        self.halt_reason = ""
        self.peak_balance = self.initial_balance

    def _check_risk_limits(self, current_time):
        """
        Check if risk limits have been breached

        Args:
            current_time: Current bar timestamp

        Returns:
            tuple: (can_trade: bool, reason: str)
        """
        # Check if trading was permanently halted (max drawdown hit)
        if self.trading_halted:
            return False, self.halt_reason

        # Update peak balance for drawdown calculation
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Check total drawdown from peak
        total_drawdown_pct = ((self.balance - self.peak_balance) / self.peak_balance) * 100
        if total_drawdown_pct <= -self.max_total_drawdown_pct:
            self.trading_halted = True
            self.halt_reason = f"Max total drawdown hit: {total_drawdown_pct:.2f}% (limit: -{self.max_total_drawdown_pct}%)"
            self.logger.warning(self.halt_reason)
            return False, self.halt_reason

        # Check for new day and reset daily tracking
        current_date = current_time.date() if hasattr(current_time, 'date') else current_time
        if self.current_day != current_date:
            self.current_day = current_date
            self.daily_start_balance = self.balance

        # Check daily loss limit
        daily_loss_pct = ((self.balance - self.daily_start_balance) / self.daily_start_balance) * 100
        if daily_loss_pct <= -self.max_daily_loss_pct:
            return False, f"Daily loss limit hit: {daily_loss_pct:.2f}% (limit: -{self.max_daily_loss_pct}%)"

        return True, ""

    def run(self, data, symbol_info):
        """
        Run backtest on historical data

        Args:
            data: DataFrame with OHLC data and indicators
            symbol_info: Symbol information dict

        Returns:
            dict: Backtest results
        """
        self.logger.info(f"Starting backtest: {self.strategy.name}")
        self.logger.info(f"Data: {len(data)} bars, Period: {data.index[0]} to {data.index[-1]}")

        self.reset()

        # Prepare data with indicators
        data = self.strategy.prepare_data(data)
        if data is None:
            self.logger.error("Failed to prepare data")
            return None

        # Iterate through each bar
        for i in range(len(data)):
            current_data = data.iloc[:i+1]

            # Skip if not enough data
            if len(current_data) < self.strategy.config.get('min_data_points', 50):
                continue

            current_bar = current_data.iloc[-1]
            current_price = current_bar['Close']
            current_time = current_bar.name

            # Update equity
            if self.current_position:
                self._update_position_value(current_price)

            # Check for exit signals if in position
            if self.current_position:
                should_exit, exit_reason = self._check_exit_conditions(
                    current_data,
                    current_price,
                    current_bar
                )

                if should_exit:
                    self._close_position(current_price, current_time, exit_reason)

            # Check for entry signals if no position
            if not self.current_position:
                # Check risk limits before allowing new trades
                can_trade, limit_reason = self._check_risk_limits(current_time)

                if can_trade:
                    signal = self.strategy.generate_signals(current_data)

                    if signal['action'] == 'BUY' or signal['action'] == 'SELL':
                        self._open_position(
                            signal,
                            current_price,
                            current_time,
                            current_bar.get('ATR', current_price * 0.01),
                            symbol_info
                        )

            # Record equity
            self.equity_curve.append({
                'time': current_time,
                'equity': self.equity,
                'balance': self.balance
            })

        # Close any open position at end
        if self.current_position:
            final_price = data.iloc[-1]['Close']
            final_time = data.iloc[-1].name
            self._close_position(final_price, final_time, "Backtest end")

        # Calculate statistics
        results = self._calculate_statistics()

        self.logger.info("Backtest complete")
        return results

    def _open_position(self, signal, entry_price, entry_time, atr, symbol_info):
        """Open a new position"""
        order_type = signal['action']

        # Apply slippage and spread to entry price (realistic execution)
        actual_entry_price = self._apply_entry_costs(entry_price, order_type)

        # Calculate SL/TP based on actual entry
        sl_price = self.risk_manager.calculate_stop_loss(
            actual_entry_price,
            atr,
            self.risk_manager.config.get('stop_loss', {}),
            order_type
        )

        tp_price = self.risk_manager.calculate_take_profit(
            actual_entry_price,
            atr,
            self.risk_manager.config.get('take_profit', {}),
            sl_price,
            order_type
        )

        # Calculate position size
        sl_pips = self.risk_manager.calculate_stop_loss_pips(
            actual_entry_price,
            sl_price,
            symbol_info
        )

        lot_size = self.risk_manager.calculate_position_size(
            self.balance,
            symbol_info,
            sl_pips
        )

        # Store symbol info for profit calculations
        self._current_contract_size = symbol_info.get('trade_contract_size', 100000)
        self._current_symbol_point = symbol_info.get('point', 0.00001)

        self.current_position = {
            'type': order_type,
            'entry_price': actual_entry_price,
            'entry_time': entry_time,
            'lot_size': lot_size,
            'sl': sl_price,
            'tp': tp_price,
            'reason': signal['reason'],
            'confidence': signal['confidence']
        }

        self.logger.debug(
            f"OPEN {order_type}: Price={actual_entry_price:.5f} (slippage applied), "
            f"SL={sl_price:.5f}, TP={tp_price:.5f}, Size={lot_size:.2f}"
        )

    def _close_position(self, exit_price, exit_time, reason):
        """Close current position"""
        if not self.current_position:
            return

        entry_price = self.current_position['entry_price']
        lot_size = self.current_position['lot_size']
        order_type = self.current_position['type']

        # Apply slippage and spread to exit price (realistic execution)
        actual_exit_price = self._apply_exit_costs(exit_price, order_type)

        # Calculate profit with actual execution prices
        if order_type == 'BUY':
            price_change = actual_exit_price - entry_price
        else:  # SELL
            price_change = entry_price - actual_exit_price

        # Get symbol info from current position
        contract_size = getattr(self, '_current_contract_size', 100000)
        point = getattr(self, '_current_symbol_point', 0.00001)

        # Profit in pips - use actual point value (handles JPY pairs, XAUUSD, etc.)
        profit_pips = price_change / point

        # Profit in currency - standard forex formula
        # profit = (price_change / point) * (point_value) * lot_size
        # For most pairs: point_value = point * contract_size (for account currency = counter currency)
        # Simplified: profit = price_change * contract_size * lot_size
        profit = price_change * lot_size * contract_size

        # Apply commission (commission is per lot, typically $7 per round trip)
        # FIX: Don't multiply by contract_size! Commission is per lot only.
        profit -= self.commission * lot_size

        # Update balance
        self.balance += profit

        # Record trade
        trade = {
            'entry_time': self.current_position['entry_time'],
            'exit_time': exit_time,
            'type': order_type,
            'entry_price': entry_price,
            'exit_price': actual_exit_price,
            'lot_size': lot_size,
            'profit': profit,
            'profit_pips': profit_pips,
            'return_pct': (profit / self.initial_balance) * 100,
            'duration': exit_time - self.current_position['entry_time'],
            'exit_reason': reason,
            'entry_reason': self.current_position['reason']
        }

        self.trades.append(trade)

        self.logger.debug(
            f"CLOSE {order_type}: Price={actual_exit_price:.5f} (slippage applied), "
            f"Profit=${profit:.2f} ({profit_pips:.1f} pips), Reason={reason}"
        )

        self.current_position = None

    def _update_position_value(self, current_price):
        """Update unrealized P&L"""
        if not self.current_position:
            return

        entry_price = self.current_position['entry_price']
        lot_size = self.current_position['lot_size']
        order_type = self.current_position['type']
        contract_size = getattr(self, '_current_contract_size', 100000)

        if order_type == 'BUY':
            unrealized_profit = (current_price - entry_price) * lot_size * contract_size
        else:
            unrealized_profit = (entry_price - current_price) * lot_size * contract_size

        self.equity = self.balance + unrealized_profit

    def _check_exit_conditions(self, data, current_price, current_bar):
        """Check if position should be exited"""
        if not self.current_position:
            return False, ""

        order_type = self.current_position['type']
        sl = self.current_position['sl']
        tp = self.current_position['tp']

        # Check SL/TP hits
        if order_type == 'BUY':
            if current_price <= sl:
                return True, "Stop loss hit"
            if current_price >= tp:
                return True, "Take profit hit"

            # Check strategy exit signal
            should_exit, reason = self.strategy.should_exit_long(data)
            if should_exit:
                return True, reason

        else:  # SELL
            if current_price >= sl:
                return True, "Stop loss hit"
            if current_price <= tp:
                return True, "Take profit hit"

            should_exit, reason = self.strategy.should_exit_short(data)
            if should_exit:
                return True, reason

        return False, ""

    def _apply_entry_costs(self, price, order_type):
        """
        Apply slippage and spread to entry price (makes entry worse)

        For BUY: pay ASK + slippage (higher price = worse)
        For SELL: receive BID - slippage (lower price = worse)

        Args:
            price: Base price (close price from candle)
            order_type: 'BUY' or 'SELL'

        Returns:
            float: Actual entry price after costs
        """
        # Convert pips to price using actual point value (handles JPY, XAUUSD, etc.)
        point = getattr(self, '_current_symbol_point', 0.00001)
        slippage_value = self.slippage_pips * point
        spread_value = (self.avg_spread_pips / 2) * point  # Half spread on entry

        if order_type == 'BUY':
            # Buy at ASK (price + half spread) + slippage
            actual_price = price + spread_value + slippage_value
        else:  # SELL
            # Sell at BID (price - half spread) - slippage
            actual_price = price - spread_value - slippage_value

        return actual_price

    def _apply_exit_costs(self, price, order_type):
        """
        Apply slippage and spread to exit price (makes exit worse)

        For BUY: sell at BID - slippage (lower price = worse)
        For SELL: buy at ASK + slippage (higher price = worse)

        Args:
            price: Base price (close price from candle)
            order_type: 'BUY' or 'SELL' of the position being closed

        Returns:
            float: Actual exit price after costs
        """
        # Convert pips to price using actual point value (handles JPY, XAUUSD, etc.)
        point = getattr(self, '_current_symbol_point', 0.00001)
        slippage_value = self.slippage_pips * point
        spread_value = (self.avg_spread_pips / 2) * point  # Half spread on exit

        if order_type == 'BUY':
            # Close BUY at BID (price - half spread) - slippage
            actual_price = price - spread_value - slippage_value
        else:  # SELL
            # Close SELL at ASK (price + half spread) + slippage
            actual_price = price + spread_value + slippage_value

        return actual_price

    def _max_consecutive(self, df, column, condition):
        """
        Calculate maximum consecutive occurrences where condition is True

        Args:
            df: DataFrame with trades
            column: Column name to check
            condition: Lambda function for condition (e.g., lambda x: x > 0)

        Returns:
            int: Maximum consecutive count
        """
        if len(df) == 0:
            return 0

        max_streak = 0
        current_streak = 0

        for value in df[column]:
            if condition(value):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_statistics(self):
        """Calculate backtest performance statistics"""
        if not self.trades:
            return {
                'error': 'No trades executed',
                'num_trades': 0
            }

        df_trades = pd.DataFrame(self.trades)
        df_equity = pd.DataFrame(self.equity_curve)

        # Basic metrics
        num_trades = len(df_trades)
        winning_trades = df_trades[df_trades['profit'] > 0]
        losing_trades = df_trades[df_trades['profit'] < 0]

        total_return = self.balance - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100

        # Win rate
        win_rate = (len(winning_trades) / num_trades) * 100 if num_trades > 0 else 0

        # Average trade
        avg_profit = df_trades['profit'].mean()
        avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0

        # Max drawdown
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak'] * 100
        max_drawdown = df_equity['drawdown'].min()

        # Profit factor
        gross_profit = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (simplified)
        returns = df_trades['return_pct']
        sharpe_ratio = (returns.mean() / returns.std()) if returns.std() > 0 else 0

        # Sortino ratio (penalizes downside volatility only)
        downside_returns = df_trades[df_trades['profit'] < 0]['return_pct']
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0001  # Avoid div by zero
        sortino_ratio = (returns.mean() / downside_std) if downside_std > 0 else 0

        # Calmar ratio (return / max drawdown)
        calmar_ratio = abs(return_pct / max_drawdown) if max_drawdown != 0 else 0

        # Expectancy (average profit per trade considering win probability)
        win_prob = win_rate / 100
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        expectancy = (win_prob * avg_win_loss_ratio) - (1 - win_prob)

        # Consecutive wins/losses
        max_consecutive_wins = self._max_consecutive(df_trades, 'profit', lambda x: x > 0)
        max_consecutive_losses = self._max_consecutive(df_trades, 'profit', lambda x: x < 0)

        # Recovery factor (net profit / max drawdown)
        recovery_factor = abs(total_return / (max_drawdown * self.initial_balance / 100)) if max_drawdown != 0 else 0

        # Average Risk-Reward Ratio
        avg_rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Payoff ratio (average win / average loss)
        payoff_ratio = avg_rr_ratio

        # System quality number (SQN) - Van Tharp metric
        sqn = (avg_profit / df_trades['profit'].std()) * (num_trades ** 0.5) if df_trades['profit'].std() > 0 else 0

        return {
            # Basic metrics
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': total_return,
            'return_pct': return_pct,
            'num_trades': num_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': df_trades['profit'].max(),
            'worst_trade': df_trades['profit'].min(),

            # Risk metrics
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'recovery_factor': recovery_factor,

            # Performance ratios
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'expectancy': expectancy,
            'payoff_ratio': payoff_ratio,
            'avg_rr_ratio': avg_rr_ratio,
            'sqn': sqn,

            # Streak metrics
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,

            # Other
            'avg_trade_duration': df_trades['duration'].mean(),
            'trades': df_trades.to_dict('records'),
            'equity_curve': df_equity.to_dict('records'),

            # Risk limit info
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason if self.trading_halted else ""
        }

    def plot_results(self, results, save_path=None):
        """Plot backtest results with improved, user-friendly visualization"""
        if 'equity_curve' not in results:
            print("No data to plot")
            return

        df_equity = pd.DataFrame(results['equity_curve'])
        df_trades = pd.DataFrame(results['trades'])

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Color scheme
        colors = {
            'profit': '#2ECC71',      # Green
            'loss': '#E74C3C',        # Red
            'equity': '#3498DB',      # Blue
            'balance': '#9B59B6',     # Purple
            'drawdown': '#E74C3C',    # Red
            'neutral': '#95A5A6',     # Gray
            'background': '#F8F9FA',  # Light gray
        }

        fig = plt.figure(figsize=(16, 12), facecolor=colors['background'])

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title with performance grade
        return_pct = results['return_pct']
        if return_pct >= 20:
            grade, grade_color = 'A+', '#27AE60'
        elif return_pct >= 10:
            grade, grade_color = 'A', '#2ECC71'
        elif return_pct >= 5:
            grade, grade_color = 'B', '#F39C12'
        elif return_pct >= 0:
            grade, grade_color = 'C', '#E67E22'
        else:
            grade, grade_color = 'D', '#E74C3C'

        fig.suptitle(f'{self.strategy.name} Backtest Report',
                    fontsize=20, fontweight='bold', y=0.98)

        # 1. EQUITY CURVE (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(colors['background'])

        # Plot equity with gradient fill
        ax1.plot(df_equity['time'], df_equity['equity'],
                color=colors['equity'], linewidth=2.5, label='Equity')
        ax1.fill_between(df_equity['time'], self.initial_balance, df_equity['equity'],
                        where=(df_equity['equity'] >= self.initial_balance),
                        color=colors['profit'], alpha=0.3)
        ax1.fill_between(df_equity['time'], self.initial_balance, df_equity['equity'],
                        where=(df_equity['equity'] < self.initial_balance),
                        color=colors['loss'], alpha=0.3)

        # Mark trades on equity curve
        for trade in results['trades']:
            marker_color = colors['profit'] if trade['profit'] > 0 else colors['loss']
            ax1.scatter(trade['exit_time'],
                       df_equity[df_equity['time'] <= trade['exit_time']]['equity'].iloc[-1] if len(df_equity[df_equity['time'] <= trade['exit_time']]) > 0 else self.initial_balance,
                       color=marker_color, s=50, zorder=5, alpha=0.7)

        ax1.axhline(y=self.initial_balance, color=colors['neutral'],
                   linestyle='--', linewidth=1, alpha=0.7, label='Initial Capital')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Account Value ($)', fontsize=10)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Add return annotation
        ax1.annotate(f'{return_pct:+.2f}%',
                    xy=(df_equity['time'].iloc[-1], df_equity['equity'].iloc[-1]),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    color=colors['profit'] if return_pct >= 0 else colors['loss'])

        # 2. WIN/LOSS PIE CHART
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(colors['background'])

        wins = results['winning_trades']
        losses = results['losing_trades']
        sizes = [wins, losses]
        pie_colors = [colors['profit'], colors['loss']]
        explode = (0.05, 0)

        wedges, texts, autotexts = ax2.pie(sizes, explode=explode, colors=pie_colors,
                                          autopct='%1.1f%%', startangle=90,
                                          shadow=True, textprops={'fontsize': 11})
        autotexts[0].set_color('white')
        autotexts[1].set_color('white')
        ax2.set_title(f'Win Rate: {results["win_rate"]:.1f}%',
                     fontsize=12, fontweight='bold', pad=10)
        ax2.legend([f'Wins ({wins})', f'Losses ({losses})'],
                  loc='lower center', framealpha=0.9)

        # 3. TRADE PROFIT/LOSS BARS
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(colors['background'])

        profits = df_trades['profit'].values
        bar_colors = [colors['profit'] if p > 0 else colors['loss'] for p in profits]
        bars = ax3.bar(range(len(profits)), profits, color=bar_colors, alpha=0.8, edgecolor='white')

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.axhline(y=results['avg_profit'], color=colors['equity'],
                   linestyle='--', linewidth=1.5, label=f'Avg: ${results["avg_profit"]:.2f}')
        ax3.set_title('Trade Results', fontsize=12, fontweight='bold', pad=10)
        ax3.set_xlabel('Trade #', fontsize=10)
        ax3.set_ylabel('Profit/Loss ($)', fontsize=10)
        ax3.legend(loc='upper right', framealpha=0.9)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. DRAWDOWN CHART
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor(colors['background'])

        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak'] * 100

        ax4.fill_between(df_equity['time'], df_equity['drawdown'], 0,
                        color=colors['drawdown'], alpha=0.4)
        ax4.plot(df_equity['time'], df_equity['drawdown'],
                color=colors['drawdown'], linewidth=1.5)

        # Mark max drawdown
        max_dd_idx = df_equity['drawdown'].idxmin()
        ax4.scatter(df_equity.loc[max_dd_idx, 'time'],
                   df_equity.loc[max_dd_idx, 'drawdown'],
                   color=colors['loss'], s=100, zorder=5, marker='v')
        ax4.annotate(f'{results["max_drawdown"]:.1f}%',
                    xy=(df_equity.loc[max_dd_idx, 'time'], df_equity.loc[max_dd_idx, 'drawdown']),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=colors['loss'])

        ax4.set_title('Drawdown', fontsize=12, fontweight='bold', pad=10)
        ax4.set_xlabel('Date', fontsize=10)
        ax4.set_ylabel('Drawdown (%)', fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. KEY METRICS PANEL
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.set_facecolor(colors['background'])
        ax5.axis('off')

        # Create metrics table
        metrics_data = [
            ['Final Balance', f'${results["final_balance"]:,.2f}',
             'Profit Factor', f'{results["profit_factor"]:.2f}'],
            ['Total Return', f'${results["total_return"]:+,.2f} ({return_pct:+.1f}%)',
             'Sharpe Ratio', f'{results["sharpe_ratio"]:.2f}'],
            ['Total Trades', f'{results["num_trades"]}',
             'Sortino Ratio', f'{results["sortino_ratio"]:.2f}'],
            ['Avg Win', f'${results["avg_win"]:.2f}',
             'Max Drawdown', f'{results["max_drawdown"]:.1f}%'],
            ['Avg Loss', f'${results["avg_loss"]:.2f}',
             'Recovery Factor', f'{results["recovery_factor"]:.2f}'],
            ['Best Trade', f'${results["best_trade"]:.2f}',
             'Expectancy', f'{results["expectancy"]:.2f}'],
            ['Worst Trade', f'${results["worst_trade"]:.2f}',
             'Win Streak', f'{results["max_consecutive_wins"]}'],
        ]

        table = ax5.table(cellText=metrics_data, loc='center', cellLoc='left',
                         colWidths=[0.22, 0.28, 0.22, 0.28])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Style table
        for i in range(len(metrics_data)):
            for j in range(4):
                cell = table[(i, j)]
                cell.set_facecolor(colors['background'])
                if j in [0, 2]:  # Label columns
                    cell.set_text_props(fontweight='bold')

        ax5.set_title('Performance Metrics', fontsize=14, fontweight='bold',
                     pad=20, y=1.02)

        # 6. PERFORMANCE GRADE
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.set_facecolor(colors['background'])
        ax6.axis('off')

        # Draw grade circle
        circle = plt.Circle((0.5, 0.55), 0.35, color=grade_color, alpha=0.2)
        ax6.add_patch(circle)
        ax6.text(0.5, 0.55, grade, fontsize=60, fontweight='bold',
                ha='center', va='center', color=grade_color)
        ax6.text(0.5, 0.15, 'Performance\nGrade', fontsize=12,
                ha='center', va='center', color=colors['neutral'])
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)

        plt.tight_layout()

        # Save to file if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=colors['background'])
            print(f"\n📊 Plot saved to: {save_path}")

        plt.show()

    def print_summary(self, results):
        """Print backtest summary"""
        print("\n" + "=" * 70)
        print(f"  BACKTEST RESULTS: {self.strategy.name}")
        print("=" * 70)
        print(f"\nStarting Capital: ${results['initial_balance']:,.2f}")
        print(f"Final Balance:    ${results['final_balance']:,.2f}")
        print(f"Total Return:     ${results['total_return']:,.2f} ({results['return_pct']:.2f}%)")
        print(f"\nTrades:")
        print(f"  Total: {results['num_trades']}")
        print(f"  Winning: {results['winning_trades']} ({results['win_rate']:.1f}%)")
        print(f"  Losing: {results['losing_trades']}")
        print(f"\nProfit Metrics:")
        print(f"  Avg Profit: ${results['avg_profit']:.2f}")
        print(f"  Avg Win: ${results['avg_win']:.2f}")
        print(f"  Avg Loss: ${results['avg_loss']:.2f}")
        print(f"  Best Trade: ${results['best_trade']:.2f}")
        print(f"  Worst Trade: ${results['worst_trade']:.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Recovery Factor: {results['recovery_factor']:.2f}")
        print(f"\nPerformance Ratios:")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio: {results['calmar_ratio']:.2f}")
        print(f"  Payoff Ratio: {results['payoff_ratio']:.2f}")
        print(f"  System Quality (SQN): {results['sqn']:.2f}")
        print(f"\nExpectancy & Streaks:")
        print(f"  Expectancy: {results['expectancy']:.2f}")
        print(f"  Max Consecutive Wins: {results['max_consecutive_wins']}")
        print(f"  Max Consecutive Losses: {results['max_consecutive_losses']}")

        # Show risk limit status
        if results.get('trading_halted'):
            print(f"\n⚠️  TRADING HALTED: {results['halt_reason']}")
        print("=" * 70 + "\n")
