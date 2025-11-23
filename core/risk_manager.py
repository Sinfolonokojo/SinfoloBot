"""
Risk Management Module
Handles position sizing, stop loss, take profit, and risk controls.
"""

import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta


class RiskManager:
    """Manages trading risk and position sizing"""

    def __init__(self, config):
        """
        Initialize Risk Manager

        Args:
            config: Risk configuration dict from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Risk parameters
        self.risk_per_trade = config.get('risk_per_trade', 1.0)
        self.max_daily_loss_pct = config.get('max_daily_loss', {}).get('percentage', 5.0)
        self.max_daily_loss_enabled = config.get('max_daily_loss', {}).get('enabled', True)
        self.max_exposure = config.get('max_total_exposure', 10.0)

        # Tracking
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.starting_balance = None
        self.last_reset_date = datetime.now().date()

    def calculate_position_size(self, account_balance, symbol_info, stop_loss_pips):
        """
        Calculate lot size based on risk percentage

        Args:
            account_balance: Current account balance
            symbol_info: Symbol information dict from MT5
            stop_loss_pips: Stop loss distance in pips

        Returns:
            float: Lot size (rounded to min lot step)
        """
        try:
            # Calculate risk amount in account currency
            risk_amount = account_balance * (self.risk_per_trade / 100)

            # Get pip value
            pip_value = self.calculate_pip_value(symbol_info)

            # Calculate lot size
            # Risk Amount = Stop Loss (pips) × Pip Value × Lot Size
            # Therefore: Lot Size = Risk Amount / (Stop Loss × Pip Value)
            lot_size = risk_amount / (stop_loss_pips * pip_value)

            # Round to symbol's volume step
            volume_step = symbol_info.get('volume_step', 0.01)
            lot_size = round(lot_size / volume_step) * volume_step

            # Apply min/max constraints
            min_lot = symbol_info.get('volume_min', 0.01)
            max_lot = symbol_info.get('volume_max', 100.0)

            lot_size = max(min_lot, min(lot_size, max_lot))

            return lot_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return symbol_info.get('volume_min', 0.01)  # Return minimum lot

    def calculate_pip_value(self, symbol_info):
        """
        Calculate the value of one pip for the symbol

        Args:
            symbol_info: Symbol information dict

        Returns:
            float: Pip value in account currency per standard lot
        """
        point = symbol_info.get('point', 0.00001)
        digits = symbol_info.get('digits', 5)
        contract_size = symbol_info.get('trade_contract_size', 100000)

        # Calculate pip size based on broker digits
        # 5-digit broker (EUR/USD: 1.12345): 1 pip = 10 points = 0.0001
        # 4-digit broker (EUR/USD: 1.1234): 1 pip = 1 point = 0.0001
        # 3-digit broker (JPY pairs: 123.45): 1 pip = 10 points = 0.01
        # 2-digit broker (JPY pairs: 123.4): 1 pip = 1 point = 0.01
        if digits == 5 or digits == 3:
            pip_size = 10 * point
        else:
            pip_size = point

        # For XXX/USD pairs, pip value per lot = pip_size * contract_size
        # Example: EUR/USD with pip_size=0.0001, contract_size=100,000
        # pip_value = 0.0001 * 100,000 = $10 per lot
        pip_value = pip_size * contract_size

        return pip_value

    def calculate_stop_loss(self, entry_price, atr, sl_config, order_type='BUY'):
        """
        Calculate stop loss price

        Args:
            entry_price: Entry price
            atr: Average True Range value
            sl_config: Stop loss configuration dict
            order_type: 'BUY' or 'SELL'

        Returns:
            float: Stop loss price
        """
        sl_type = sl_config.get('type', 'atr')

        if sl_type == 'atr':
            multiplier = sl_config.get('atr_multiplier', 2.0)
            sl_distance = atr * multiplier

        elif sl_type == 'fixed_pips':
            pips = sl_config.get('fixed_pips', 50)
            sl_distance = pips * 0.0001  # Assuming 4-digit broker

        elif sl_type == 'percentage':
            pct = sl_config.get('percentage', 2.0)
            sl_distance = entry_price * (pct / 100)

        else:
            # Default to ATR
            sl_distance = atr * 2.0

        # Calculate SL price
        if order_type == 'BUY':
            sl_price = entry_price - sl_distance
        else:  # SELL
            sl_price = entry_price + sl_distance

        return sl_price

    def calculate_take_profit(self, entry_price, atr, tp_config, sl_price=None, order_type='BUY'):
        """
        Calculate take profit price

        Args:
            entry_price: Entry price
            atr: Average True Range value
            tp_config: Take profit configuration dict
            sl_price: Optional stop loss price (for risk-reward calculation)
            order_type: 'BUY' or 'SELL'

        Returns:
            float: Take profit price
        """
        tp_type = tp_config.get('type', 'atr')

        if tp_type == 'atr':
            multiplier = tp_config.get('atr_multiplier', 4.0)
            tp_distance = atr * multiplier

        elif tp_type == 'fixed_pips':
            pips = tp_config.get('fixed_pips', 100)
            tp_distance = pips * 0.0001

        elif tp_type == 'percentage':
            pct = tp_config.get('percentage', 4.0)
            tp_distance = entry_price * (pct / 100)

        elif tp_type == 'risk_reward' and sl_price:
            # Calculate based on risk-reward ratio
            rr_ratio = tp_config.get('risk_reward_ratio', 2.0)
            sl_distance = abs(entry_price - sl_price)
            tp_distance = sl_distance * rr_ratio

        else:
            # Default to ATR
            tp_distance = atr * 4.0

        # Calculate TP price
        if order_type == 'BUY':
            tp_price = entry_price + tp_distance
        else:  # SELL
            tp_price = entry_price - tp_distance

        return tp_price

    def calculate_stop_loss_pips(self, entry_price, sl_price, symbol_info):
        """
        Calculate stop loss distance in pips

        Args:
            entry_price: Entry price
            sl_price: Stop loss price
            symbol_info: Symbol information

        Returns:
            float: Stop loss distance in pips
        """
        point = symbol_info.get('point', 0.00001)
        digits = symbol_info.get('digits', 5)

        # Most pairs: 1 pip = 10 points (5 digits)
        # JPY pairs: 1 pip = 1 point (3 digits)
        pip_size = 10 * point if digits == 5 else point

        pips = abs(entry_price - sl_price) / pip_size
        return round(pips, 1)

    def check_daily_loss_limit(self, account_balance):
        """
        Check if daily loss limit has been reached

        Args:
            account_balance: Current account balance

        Returns:
            tuple: (is_limit_reached: bool, reason: str)
        """
        if not self.max_daily_loss_enabled:
            return False, ""

        # Reset daily tracking if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.reset_daily_tracking(account_balance)

        # Calculate daily loss
        if self.starting_balance is None:
            self.starting_balance = account_balance
            return False, ""

        daily_loss_pct = ((account_balance - self.starting_balance) / self.starting_balance) * 100

        if daily_loss_pct <= -self.max_daily_loss_pct:
            reason = (
                f"Daily loss limit reached: {daily_loss_pct:.2f}% "
                f"(limit: {self.max_daily_loss_pct}%)"
            )
            self.logger.warning(reason)
            return True, reason

        return False, ""

    def check_exposure_limit(self, current_exposure, new_position_size):
        """
        Check if adding new position would exceed exposure limit

        Args:
            current_exposure: Current exposure as % of account
            new_position_size: Size of new position as %

        Returns:
            tuple: (is_limit_exceeded: bool, reason: str)
        """
        total_exposure = current_exposure + new_position_size

        if total_exposure > self.max_exposure:
            reason = (
                f"Exposure limit exceeded: {total_exposure:.2f}% "
                f"(limit: {self.max_exposure}%)"
            )
            self.logger.warning(reason)
            return True, reason

        return False, ""

    def reset_daily_tracking(self, current_balance):
        """Reset daily tracking at start of new day"""
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.starting_balance = current_balance
        self.last_reset_date = datetime.now().date()
        self.logger.info(f"Daily tracking reset. Starting balance: ${current_balance:.2f}")

    def record_trade(self, trade_info):
        """
        Record a completed trade

        Args:
            trade_info: Dict with trade details (profit, volume, symbol, etc.)
        """
        trade_info['timestamp'] = datetime.now()
        self.daily_trades.append(trade_info)

        profit = trade_info.get('profit', 0.0)
        self.daily_pnl += profit

        self.logger.info(
            f"Trade recorded: {trade_info.get('symbol')} | "
            f"Profit: ${profit:.2f} | Daily P&L: ${self.daily_pnl:.2f}"
        )

    def get_daily_stats(self):
        """
        Get daily trading statistics

        Returns:
            dict: Daily stats
        """
        if not self.daily_trades:
            return {
                'num_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0
            }

        winning_trades = [t for t in self.daily_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.daily_trades if t.get('profit', 0) < 0]

        return {
            'num_trades': len(self.daily_trades),
            'total_pnl': self.daily_pnl,
            'win_rate': (len(winning_trades) / len(self.daily_trades)) * 100,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_profit': self.daily_pnl / len(self.daily_trades),
            'largest_win': max([t.get('profit', 0) for t in self.daily_trades]),
            'largest_loss': min([t.get('profit', 0) for t in self.daily_trades])
        }

    def should_apply_trailing_stop(self, position, current_price, trailing_config):
        """
        Check if trailing stop should be applied

        Args:
            position: Position dict with entry_price, sl, tp
            current_price: Current market price
            trailing_config: Trailing stop configuration

        Returns:
            tuple: (should_trail: bool, new_sl: float)
        """
        if not trailing_config.get('enabled', False):
            return False, None

        entry_price = position.get('price_open')
        current_sl = position.get('sl')
        position_type = position.get('type')  # BUY or SELL

        trigger_pips = trailing_config.get('trigger_pips', 20)
        distance_pips = trailing_config.get('distance_pips', 15)

        # Calculate profit in pips
        if position_type == mt5.ORDER_TYPE_BUY:
            profit_pips = (current_price - entry_price) / 0.0001

            # Check if profit exceeds trigger
            if profit_pips >= trigger_pips:
                new_sl = current_price - (distance_pips * 0.0001)

                # Only move SL up, never down
                if new_sl > current_sl:
                    return True, new_sl

        else:  # SELL
            profit_pips = (entry_price - current_price) / 0.0001

            if profit_pips >= trigger_pips:
                new_sl = current_price + (distance_pips * 0.0001)

                # Only move SL down, never up
                if new_sl < current_sl:
                    return True, new_sl

        return False, None

    def should_apply_breakeven_stop(self, position, current_price, breakeven_config, symbol_info):
        """
        Check if break-even stop loss should be applied

        Args:
            position: Position dict with entry_price, sl, tp
            current_price: Current market price
            breakeven_config: Break-even configuration dict
            symbol_info: Symbol information for spread calculation

        Returns:
            tuple: (should_apply: bool, new_sl: float)
        """
        if not breakeven_config.get('enabled', False):
            return False, None

        entry_price = position.get('price_open')
        current_sl = position.get('sl')
        position_type = position.get('type')  # BUY or SELL

        # Get trigger threshold (default: 3 pips for ultra scalping)
        trigger_pips = breakeven_config.get('trigger_pips', 3)

        # Get buffer to add beyond break-even (default: spread + 0.2 pips)
        spread = symbol_info.get('spread', 0)
        point = symbol_info.get('point', 0.00001)
        buffer_pips = breakeven_config.get('buffer_pips', 0.2)
        buffer = (spread * point) + (buffer_pips * 0.0001)

        # Calculate profit in pips
        if position_type == mt5.ORDER_TYPE_BUY:
            profit_pips = (current_price - entry_price) / 0.0001

            # Check if profit exceeds trigger threshold
            if profit_pips >= trigger_pips:
                # Move SL to break-even + buffer (protects against spread on close)
                new_sl = entry_price + buffer

                # Only move SL up to break-even, never down
                # And only if current SL is below break-even
                if current_sl < entry_price and new_sl > current_sl:
                    self.logger.info(
                        f"Break-even trigger: Profit={profit_pips:.1f} pips, "
                        f"Moving SL from {current_sl:.5f} to {new_sl:.5f} (entry + buffer)"
                    )
                    return True, new_sl

        else:  # SELL
            profit_pips = (entry_price - current_price) / 0.0001

            if profit_pips >= trigger_pips:
                # Move SL to break-even - buffer
                new_sl = entry_price - buffer

                # Only move SL down to break-even, never up
                # And only if current SL is above break-even
                if current_sl > entry_price and new_sl < current_sl:
                    self.logger.info(
                        f"Break-even trigger: Profit={profit_pips:.1f} pips, "
                        f"Moving SL from {current_sl:.5f} to {new_sl:.5f} (entry - buffer)"
                    )
                    return True, new_sl

        return False, None

    def validate_order(self, order_params, account_balance, symbol_info):
        """
        Validate order before submission

        Args:
            order_params: Order parameters dict
            account_balance: Current account balance
            symbol_info: Symbol information

        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        # Check lot size
        lot_size = order_params.get('volume', 0)
        min_lot = symbol_info.get('volume_min', 0.01)
        max_lot = symbol_info.get('volume_max', 100.0)

        if lot_size < min_lot:
            return False, f"Lot size {lot_size} below minimum {min_lot}"

        if lot_size > max_lot:
            return False, f"Lot size {lot_size} above maximum {max_lot}"

        # Check if enough margin
        # Required margin = Lot Size × Contract Size / Leverage
        # This is simplified; actual margin calculation varies by broker

        # Check SL/TP validity
        entry_price = order_params.get('price', 0)
        sl = order_params.get('sl', 0)
        tp = order_params.get('tp', 0)

        if sl and tp:
            if order_params.get('type') == mt5.ORDER_TYPE_BUY:
                if sl >= entry_price:
                    return False, "Stop loss must be below entry price for BUY orders"
                if tp <= entry_price:
                    return False, "Take profit must be above entry price for BUY orders"
            else:  # SELL
                if sl <= entry_price:
                    return False, "Stop loss must be above entry price for SELL orders"
                if tp >= entry_price:
                    return False, "Take profit must be below entry price for SELL orders"

        return True, ""

    def get_risk_summary(self):
        """Get summary of risk parameters"""
        return {
            'risk_per_trade': f"{self.risk_per_trade}%",
            'max_daily_loss': f"{self.max_daily_loss_pct}%" if self.max_daily_loss_enabled else "Disabled",
            'max_exposure': f"{self.max_exposure}%",
            'daily_pnl': f"${self.daily_pnl:.2f}",
            'trades_today': len(self.daily_trades)
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    risk_config = {
        'risk_per_trade': 1.0,
        'max_daily_loss': {'enabled': True, 'percentage': 5.0},
        'max_total_exposure': 10.0,
        'stop_loss': {'type': 'atr', 'atr_multiplier': 2.0},
        'take_profit': {'type': 'atr', 'atr_multiplier': 4.0},
        'trailing_stop': {'enabled': True, 'trigger_pips': 20, 'distance_pips': 15}
    }

    rm = RiskManager(risk_config)
    print("Risk Manager initialized")
    print(rm.get_risk_summary())
