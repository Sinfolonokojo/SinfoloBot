"""
Base Strategy Class
Abstract base class that all trading strategies must inherit from.
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""

    def __init__(self, name, config=None):
        """
        Initialize base strategy

        Args:
            name: Strategy name
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.data = None
        self.current_position = None  # 'long', 'short', or None

    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals based on market data

        Args:
            data: DataFrame with OHLC data and indicators

        Returns:
            dict: {
                'action': 'BUY', 'SELL', or 'HOLD',
                'confidence': float (0-1),
                'reason': str (explanation of signal),
                'stop_loss': float (optional),
                'take_profit': float (optional)
            }
        """
        pass

    @abstractmethod
    def calculate_indicators(self, data):
        """
        Calculate technical indicators for the strategy

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame: Data with indicators added
        """
        pass

    def validate_data(self, data):
        """
        Validate that data has required columns and sufficient rows

        Args:
            data: DataFrame to validate

        Returns:
            bool: True if valid
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        if data is None or data.empty:
            self.logger.error("Data is empty")
            return False

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False

        min_rows = self.config.get('min_data_points', 50)
        if len(data) < min_rows:
            self.logger.warning(f"Insufficient data: {len(data)} rows (need {min_rows})")
            return False

        return True

    def prepare_data(self, data):
        """
        Prepare data by calculating indicators

        Args:
            data: Raw OHLC data

        Returns:
            DataFrame: Data with indicators
        """
        if not self.validate_data(data):
            return None

        self.data = self.calculate_indicators(data.copy())
        return self.data

    def should_enter_long(self, data):
        """
        Check if should enter long position

        Args:
            data: DataFrame with indicators

        Returns:
            tuple: (should_enter: bool, confidence: float, reason: str)
        """
        signal = self.generate_signals(data)

        if signal['action'] == 'BUY':
            return True, signal['confidence'], signal['reason']

        return False, 0.0, ""

    def should_enter_short(self, data):
        """
        Check if should enter short position

        Args:
            data: DataFrame with indicators

        Returns:
            tuple: (should_enter: bool, confidence: float, reason: str)
        """
        signal = self.generate_signals(data)

        if signal['action'] == 'SELL':
            return True, signal['confidence'], signal['reason']

        return False, 0.0, ""

    def should_exit_position(self, data, position_type):
        """
        Check if should exit current position

        Args:
            data: DataFrame with indicators
            position_type: 'long' or 'short'

        Returns:
            tuple: (should_exit: bool, reason: str)
        """
        signal = self.generate_signals(data)

        # Exit long if SELL signal
        if position_type == 'long' and signal['action'] == 'SELL':
            return True, signal['reason']

        # Exit short if BUY signal
        if position_type == 'short' and signal['action'] == 'BUY':
            return True, signal['reason']

        return False, ""

    def should_exit_long(self, data):
        """
        Check if should exit long position (for backtest compatibility)

        Args:
            data: DataFrame with indicators

        Returns:
            tuple: (should_exit: bool, reason: str)
        """
        return self.should_exit_position(data, 'long')

    def should_exit_short(self, data):
        """
        Check if should exit short position (for backtest compatibility)

        Args:
            data: DataFrame with indicators

        Returns:
            tuple: (should_exit: bool, reason: str)
        """
        return self.should_exit_position(data, 'short')

    def get_entry_price(self, data, action):
        """
        Get entry price for a trade

        Args:
            data: DataFrame with OHLC data
            action: 'BUY' or 'SELL'

        Returns:
            float: Entry price (usually current close or ask/bid)
        """
        return data['Close'].iloc[-1]

    def get_position_size(self, account_balance, risk_percent, stop_loss_pips, pip_value):
        """
        Calculate position size based on risk parameters

        Args:
            account_balance: Current account balance
            risk_percent: Risk per trade as percentage (e.g., 1.0 for 1%)
            stop_loss_pips: Stop loss distance in pips
            pip_value: Value of one pip for the symbol

        Returns:
            float: Lot size
        """
        risk_amount = account_balance * (risk_percent / 100)
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        return round(lot_size, 2)

    def log_signal(self, signal):
        """Log trading signal"""
        if signal['action'] != 'HOLD':
            self.logger.info(
                f"Signal: {signal['action']} | "
                f"Confidence: {signal['confidence']:.2%} | "
                f"Reason: {signal['reason']}"
            )

    def get_strategy_state(self):
        """
        Get current state of the strategy

        Returns:
            dict: Strategy state information
        """
        return {
            'name': self.name,
            'current_position': self.current_position,
            'config': self.config,
            'data_points': len(self.data) if self.data is not None else 0
        }

    def reset(self):
        """Reset strategy state"""
        self.data = None
        self.current_position = None
        self.logger.info(f"Strategy {self.name} reset")

    def __str__(self):
        """String representation"""
        return f"{self.name} Strategy"

    def __repr__(self):
        """Detailed representation"""
        return f"<{self.__class__.__name__}(name='{self.name}', config={self.config})>"


class SignalStrength:
    """Helper class for signal strength constants"""
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0


class TradingAction:
    """Helper class for trading action constants"""
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'
    CLOSE = 'CLOSE'
