"""
Trend Pullback Strategy for Prop Firm Compliance
Designed for The5ers and similar prop firm challenges.

Strategy Logic:
- Timeframe: M5
- Trend: EMA(50)
- Entry: RSI(14) pullback in trend direction
- Exit: 1:2 R:R or RSI reversal

Buy Signal: Price > EMA(50) AND RSI < 40 (oversold in uptrend)
Sell Signal: Price < EMA(50) AND RSI > 60 (overbought in downtrend)
"""

from strategies.base_strategy import BaseStrategy, TradingAction, SignalStrength
from strategies.indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_adx
import pandas as pd
import numpy as np
import logging


class TrendPullbackStrategy(BaseStrategy):
    """
    Trend Pullback Strategy for Prop Firm Trading

    Simple, high-probability setup:
    - Trade with the trend (EMA 50)
    - Enter on pullbacks (RSI extremes)
    - Fixed 1:2 Risk/Reward
    """

    def __init__(self, config=None):
        """Initialize Trend Pullback Strategy"""
        default_config = {
            # Indicator periods
            'ema_period': 50,
            'rsi_period': 14,
            'atr_period': 14,
            'adx_period': 14,

            # Entry thresholds - STRICTER for better quality
            'rsi_buy_threshold': 32,      # RSI below this for buy (deeper oversold)
            'rsi_sell_threshold': 68,     # RSI above this for sell (deeper overbought)

            # Exit thresholds (RSI reversal) - DISABLED to let trades run to TP/SL
            'use_rsi_exit': False,        # Disable RSI exit - let trades hit TP/SL
            'rsi_buy_exit': 70,           # Exit long if RSI > 70 (only if enabled)
            'rsi_sell_exit': 30,          # Exit short if RSI < 30 (only if enabled)

            # Risk management
            'risk_reward': 2.0,           # 1:2 R:R ratio
            'sl_atr_multiplier': 1.5,     # SL = 1.5 x ATR

            # Filters
            'min_atr': 0.0003,            # Minimum ATR to trade (avoid low volatility)
            'min_ema_distance': 0.0002,   # Minimum distance from EMA
            'require_ema_slope': True,    # Require EMA to be sloping in trade direction
            'min_adx': 20,                # Minimum ADX for trend strength
            'require_rsi_momentum': True, # Require RSI to be turning in trade direction

            # General
            'min_confidence': 0.70,       # Balanced confidence for quality
            'min_data_points': 60,        # Need enough data for EMA 50
        }

        if config:
            default_config.update(config)

        super().__init__('TrendPullback', default_config)
        self.logger = logging.getLogger(__name__)

    def calculate_indicators(self, data):
        """Calculate indicators for the strategy"""
        df = data.copy()

        # EMA for trend direction
        df['EMA'] = calculate_ema(df, self.config['ema_period'])

        # RSI for entry timing
        df['RSI'] = calculate_rsi(df, self.config['rsi_period'])

        # ATR for stop loss calculation
        df['ATR'] = calculate_atr(df, self.config['atr_period'])

        # ADX for trend strength
        df['ADX'] = calculate_adx(df, self.config.get('adx_period', 14))

        # Price distance from EMA (normalized)
        df['EMA_Distance'] = (df['Close'] - df['EMA']) / df['ATR']

        # Trend direction
        df['Trend'] = np.where(df['Close'] > df['EMA'], 'UP', 'DOWN')

        # EMA slope (for trend strength confirmation)
        df['EMA_Slope'] = df['EMA'].diff(5) / df['ATR']  # 5-bar slope normalized by ATR

        # RSI momentum (for confirmation)
        df['RSI_Change'] = df['RSI'].diff(3)  # 3-bar RSI change

        return df

    def generate_signals(self, data):
        """
        Generate trading signals based on trend pullback logic

        Returns:
            dict: Signal with action, confidence, and reason
        """
        if not self.validate_data(data):
            return self._no_signal("Invalid data")

        # Ensure indicators are calculated
        if 'EMA' not in data.columns:
            data = self.calculate_indicators(data)

        # Get latest values
        current = data.iloc[-1]

        # Check for sufficient data
        if pd.isna(current['EMA']) or pd.isna(current['RSI']):
            return self._no_signal("Insufficient indicator data")

        # Get values
        price = current['Close']
        ema = current['EMA']
        rsi = current['RSI']
        atr = current['ATR']
        adx = current.get('ADX', 0)
        rsi_change = current.get('RSI_Change', 0)

        # Check minimum ATR (avoid low volatility)
        if atr < self.config['min_atr']:
            return self._no_signal(f"ATR too low ({atr:.5f})")

        # Check ADX for trend strength (avoid ranging markets)
        min_adx = self.config.get('min_adx', 25)
        if pd.notna(adx) and adx < min_adx:
            return self._no_signal(f"ADX too low ({adx:.1f} < {min_adx})")

        # Determine trend
        is_uptrend = price > ema
        is_downtrend = price < ema

        # Calculate distance from EMA
        ema_distance = abs(price - ema)
        min_distance = self.config['min_ema_distance']

        # Get EMA slope for trend strength
        ema_slope = current.get('EMA_Slope', 0)

        # Check for BUY signal (uptrend + oversold RSI + positive EMA slope + RSI turning up)
        if is_uptrend and rsi < self.config['rsi_buy_threshold']:
            # Check EMA slope if required
            if self.config.get('require_ema_slope', True) and ema_slope <= 0:
                pass  # Skip - EMA not sloping up
            # Check RSI momentum if required (RSI should be turning up)
            elif self.config.get('require_rsi_momentum', True) and rsi_change <= 0:
                pass  # Skip - RSI not turning up
            elif ema_distance >= min_distance:
                confidence = self._calculate_buy_confidence(current, data)
                if confidence >= self.config['min_confidence']:
                    return {
                        'action': TradingAction.BUY,
                        'confidence': confidence,
                        'reason': f"Uptrend pullback | RSI={rsi:.1f} | ADX={adx:.1f} | EMA slope={ema_slope:.2f}",
                        'stop_loss': None,
                        'take_profit': None,
                        'atr': atr,
                        'rsi': rsi,
                        'ema_distance': ema_distance
                    }

        # Check for SELL signal (downtrend + overbought RSI + negative EMA slope + RSI turning down)
        if is_downtrend and rsi > self.config['rsi_sell_threshold']:
            # Check EMA slope if required
            if self.config.get('require_ema_slope', True) and ema_slope >= 0:
                pass  # Skip - EMA not sloping down
            # Check RSI momentum if required (RSI should be turning down)
            elif self.config.get('require_rsi_momentum', True) and rsi_change >= 0:
                pass  # Skip - RSI not turning down
            elif ema_distance >= min_distance:
                confidence = self._calculate_sell_confidence(current, data)
                if confidence >= self.config['min_confidence']:
                    return {
                        'action': TradingAction.SELL,
                        'confidence': confidence,
                        'reason': f"Downtrend pullback | RSI={rsi:.1f} | ADX={adx:.1f} | EMA slope={ema_slope:.2f}",
                        'stop_loss': None,
                        'take_profit': None,
                        'atr': atr,
                        'rsi': rsi,
                        'ema_distance': ema_distance
                    }

        return self._no_signal("No setup detected")

    def _calculate_buy_confidence(self, current, data):
        """Calculate confidence score for BUY signal"""
        confidence = 0.0

        rsi = current['RSI']
        ema_distance = abs(current['Close'] - current['EMA'])
        atr = current['ATR']
        adx = current.get('ADX', 25)
        rsi_change = current.get('RSI_Change', 0)

        # RSI depth (lower = more oversold = better)
        # RSI 30 = 0.5, RSI 20 = 0.75, RSI 10 = 1.0
        rsi_score = (self.config['rsi_buy_threshold'] - rsi) / 20
        rsi_score = max(0, min(1, rsi_score + 0.5))
        confidence += rsi_score * 0.30  # 30% weight

        # EMA distance (further from EMA = stronger trend)
        distance_score = min(1.0, ema_distance / (atr * 2))
        confidence += distance_score * 0.20  # 20% weight

        # RSI momentum (RSI should be rising from low)
        if rsi_change > 0:  # RSI turning up
            momentum_score = min(1.0, rsi_change / 5)  # Normalize
            confidence += momentum_score * 0.20  # 20% weight
        else:
            confidence += 0.05

        # ADX trend strength (higher = stronger trend)
        if pd.notna(adx):
            adx_score = min(1.0, (adx - 20) / 30)  # ADX 20=0, ADX 50=1
            confidence += max(0, adx_score) * 0.20  # 20% weight

        # ATR health (good volatility)
        avg_atr = data['ATR'].tail(20).mean()
        if atr > avg_atr * 0.8:
            confidence += 0.10  # 10% weight

        return min(1.0, confidence)

    def _calculate_sell_confidence(self, current, data):
        """Calculate confidence score for SELL signal"""
        confidence = 0.0

        rsi = current['RSI']
        ema_distance = abs(current['Close'] - current['EMA'])
        atr = current['ATR']
        adx = current.get('ADX', 25)
        rsi_change = current.get('RSI_Change', 0)

        # RSI height (higher = more overbought = better)
        # RSI 70 = 0.5, RSI 80 = 0.75, RSI 90 = 1.0
        rsi_score = (rsi - self.config['rsi_sell_threshold']) / 20
        rsi_score = max(0, min(1, rsi_score + 0.5))
        confidence += rsi_score * 0.30  # 30% weight

        # EMA distance
        distance_score = min(1.0, ema_distance / (atr * 2))
        confidence += distance_score * 0.20  # 20% weight

        # RSI momentum (RSI should be falling from high)
        if rsi_change < 0:  # RSI turning down
            momentum_score = min(1.0, abs(rsi_change) / 5)  # Normalize
            confidence += momentum_score * 0.20  # 20% weight
        else:
            confidence += 0.05

        # ADX trend strength (higher = stronger trend)
        if pd.notna(adx):
            adx_score = min(1.0, (adx - 20) / 30)  # ADX 20=0, ADX 50=1
            confidence += max(0, adx_score) * 0.20  # 20% weight

        # ATR health
        avg_atr = data['ATR'].tail(20).mean()
        if atr > avg_atr * 0.8:
            confidence += 0.10  # 10% weight

        return min(1.0, confidence)

    def _no_signal(self, reason=""):
        """Return a HOLD signal"""
        return {
            'action': TradingAction.HOLD,
            'confidence': 0.0,
            'reason': reason,
            'stop_loss': None,
            'take_profit': None
        }

    def should_exit_position(self, data, position_type):
        """
        Check if position should be exited based on RSI reversal

        NOTE: RSI exit is DISABLED by default to let trades run to TP/SL
        This improves the risk/reward ratio significantly.

        Args:
            data: DataFrame with indicators
            position_type: 'long' or 'short'

        Returns:
            tuple: (should_exit: bool, reason: str)
        """
        # Check if RSI exit is enabled
        if not self.config.get('use_rsi_exit', False):
            return False, ""  # Let trade run to TP/SL

        if 'RSI' not in data.columns:
            data = self.calculate_indicators(data)

        current = data.iloc[-1]
        rsi = current['RSI']

        # Exit long on RSI overbought
        if position_type == 'long':
            if rsi > self.config['rsi_buy_exit']:
                return True, f"RSI reversal exit ({rsi:.1f} > {self.config['rsi_buy_exit']})"

        # Exit short on RSI oversold
        if position_type == 'short':
            if rsi < self.config['rsi_sell_exit']:
                return True, f"RSI reversal exit ({rsi:.1f} < {self.config['rsi_sell_exit']})"

        return False, ""

    def get_sl_tp_prices(self, entry_price, atr, direction):
        """
        Calculate Stop Loss and Take Profit prices

        CRITICAL: SL must always be set for prop firm compliance

        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: 'BUY' or 'SELL'

        Returns:
            tuple: (sl_price, tp_price)
        """
        sl_distance = atr * self.config['sl_atr_multiplier']
        tp_distance = sl_distance * self.config['risk_reward']

        if direction == 'BUY':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # SELL
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        return sl_price, tp_price


if __name__ == "__main__":
    # Test the strategy
    import logging
    logging.basicConfig(level=logging.INFO)

    strategy = TrendPullbackStrategy()
    print(f"Strategy initialized: {strategy}")
    print(f"Config: {strategy.config}")
