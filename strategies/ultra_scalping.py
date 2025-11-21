"""
Ultra Scalping Strategy for M1 Timeframe
Optimized for EUR/USD with conservative risk management.
Focuses on quick entries/exits with high probability setups.
"""

from strategies.base_strategy import BaseStrategy, TradingAction, SignalStrength
from strategies.indicators import (
    calculate_ema, calculate_rsi, calculate_atr,
    calculate_bollinger_bands, calculate_macd,
    calculate_adx, detect_crossover, detect_crossunder
)
from strategies.market_filters import MarketFilters
import pandas as pd
import numpy as np


class UltraScalpingStrategy(BaseStrategy):
    """
    Ultra-fast scalping strategy for M1 timeframe.

    Features:
    - Fast EMA (5, 13, 21) for trend direction
    - RSI for momentum confirmation
    - Bollinger Bands for volatility squeeze detection
    - MACD for additional momentum confirmation
    - Volume analysis for entry strength
    - Conservative exit on opposite signals

    Entry Conditions (BUY):
    1. Price above EMA 21 (trend filter)
    2. EMA 5 crosses above EMA 13
    3. RSI between 40-70 (avoiding extremes)
    4. MACD histogram positive and increasing
    5. Price near or bouncing off lower Bollinger Band
    6. Recent volume spike (optional confirmation)

    Entry Conditions (SELL):
    1. Price below EMA 21 (trend filter)
    2. EMA 5 crosses below EMA 13
    3. RSI between 30-60
    4. MACD histogram negative and decreasing
    5. Price near or bouncing off upper Bollinger Band

    Exit Strategy:
    - Quick profit targets (5-10 pips)
    - Tight stop losses (3-5 pips)
    - Exit on opposite EMA cross
    - Exit on RSI extremes (>80 or <20)
    """

    def __init__(self, config=None):
        """Initialize Ultra Scalping Strategy"""
        default_config = {
            'ema_fast': 5,
            'ema_medium': 13,
            'ema_slow': 21,
            'rsi_period': 9,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'rsi_extreme_low': 20,
            'rsi_extreme_high': 80,
            'bb_period': 20,
            'bb_std': 2.0,
            'macd_fast': 5,
            'macd_slow': 13,
            'macd_signal': 5,
            'atr_period': 14,
            'adx_period': 14,
            'volume_ma_period': 20,
            'min_confidence': 0.55,  # Lowered from 0.65 for more trades
            'min_data_points': 50,
            # Market filters configuration
            'market_filters': {
                'volatility_filter': {'enabled': True, 'atr_min_ratio': 0.7, 'atr_max_ratio': 1.5},
                'spread_filter': {'enabled': True, 'max_spread_pips': 1.2, 'max_spread_atr_pct': 0.3},
                'session_filter': {'enabled': True, 'allowed_sessions': ['london', 'ny', 'overlap']},
                'trend_filter': {'enabled': True, 'min_strength': 0.2},  # Lowered from 0.3
                'rollover_filter': {'enabled': True, 'start_hour': 21, 'end_hour': 22}
            }
        }

        if config:
            default_config.update(config)

        super().__init__('UltraScalping', default_config)

        # Initialize market filters
        self.market_filters = MarketFilters(self.config.get('market_filters', {}))

    def calculate_indicators(self, data):
        """Calculate all indicators needed for the strategy"""
        df = data.copy()

        # EMAs for trend and crossover signals
        df['EMA_Fast'] = calculate_ema(df, self.config['ema_fast'])
        df['EMA_Medium'] = calculate_ema(df, self.config['ema_medium'])
        df['EMA_Slow'] = calculate_ema(df, self.config['ema_slow'])

        # RSI for momentum
        df['RSI'] = calculate_rsi(df, self.config['rsi_period'])

        # Bollinger Bands for volatility
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            df, self.config['bb_period'], self.config['bb_std']
        )
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower

        # MACD for trend confirmation
        macd, signal, hist = calculate_macd(
            df,
            self.config['macd_fast'],
            self.config['macd_slow'],
            self.config['macd_signal']
        )
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist

        # ATR for volatility measurement
        df['ATR'] = calculate_atr(df, self.config['atr_period'])

        # ADX for trend strength confirmation
        df['ADX'] = calculate_adx(df, self.config.get('adx_period', 14))

        # Volume analysis
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(
                window=self.config['volume_ma_period']
            ).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        else:
            df['Volume_MA'] = 0
            df['Volume_Ratio'] = 1

        # Price position relative to BBands
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Trend strength
        df['Trend_Strength'] = (df['EMA_Fast'] - df['EMA_Slow']) / df['ATR']

        return df

    def generate_signals(self, data):
        """
        Generate trading signals based on ultra scalping logic

        Returns:
            dict: Signal with action, confidence, and reason
        """
        if not self.validate_data(data):
            return self._no_signal("Invalid data")

        # Ensure indicators are calculated
        if 'EMA_Fast' not in data.columns:
            data = self.calculate_indicators(data)

        # Get latest values
        current = data.iloc[-1]
        previous = data.iloc[-2]

        # Check for sufficient data
        if pd.isna(current['EMA_Fast']) or pd.isna(current['RSI']):
            return self._no_signal("Insufficient indicator data")

        # Apply market condition filters
        # NOTE: symbol_info needs to be passed in for full filtering
        # For now, we'll do the filters that only need data
        if self.config.get('market_filters'):
            # Check volatility filter
            volatility_config = self.config['market_filters'].get('volatility_filter', {})
            if volatility_config.get('enabled', True):
                is_valid, reason = self.market_filters.check_volatility(
                    data,
                    atr_min_ratio=volatility_config.get('atr_min_ratio', 0.7),
                    atr_max_ratio=volatility_config.get('atr_max_ratio', 1.5)
                )
                if not is_valid:
                    return self._no_signal(f"Market filter: {reason}")

            # Check trend strength filter
            trend_config = self.config['market_filters'].get('trend_filter', {})
            if trend_config.get('enabled', True):
                is_valid, reason = self.market_filters.check_trend_strength(
                    data,
                    min_strength=trend_config.get('min_strength', 0.3)
                )
                if not is_valid:
                    return self._no_signal(f"Market filter: {reason}")

        # Detect EMA crossovers
        bullish_cross = detect_crossover(data['EMA_Fast'], data['EMA_Medium'])
        bearish_cross = detect_crossunder(data['EMA_Fast'], data['EMA_Medium'])

        # Check for BUY signal
        if bullish_cross:
            buy_signal = self._check_buy_conditions(current, previous, data)
            if buy_signal['action'] == TradingAction.BUY:
                return buy_signal

        # Check for SELL signal
        if bearish_cross:
            sell_signal = self._check_sell_conditions(current, previous, data)
            if sell_signal['action'] == TradingAction.SELL:
                return sell_signal

        # No clear signal
        return self._no_signal("No setup detected")

    def _check_buy_conditions(self, current, previous, data):
        """Check all conditions for a BUY signal"""
        confidence_score = 0
        max_score = 11  # Increased from 8 to accommodate new boosters
        reasons = []

        # 1. Trend filter: Price above EMA 21
        if current['Close'] > current['EMA_Slow']:
            confidence_score += 2
            reasons.append("Price above trend EMA")
        else:
            # Weak trend, lower confidence
            confidence_score += 0.5
            reasons.append("Price below trend EMA (weak)")

        # 2. EMA crossover already confirmed by caller
        confidence_score += 1.5
        reasons.append("EMA 5/13 bullish cross")

        # 3. RSI in favorable range (not overbought, has room to move up)
        rsi = current['RSI']
        if self.config['rsi_oversold'] < rsi < self.config['rsi_overbought']:
            confidence_score += 1.5
            reasons.append(f"RSI optimal ({rsi:.1f})")
        elif rsi < self.config['rsi_oversold']:
            confidence_score += 1
            reasons.append(f"RSI oversold ({rsi:.1f})")

        # 4. MACD confirmation
        if current['MACD_Hist'] > 0 and current['MACD_Hist'] > previous['MACD_Hist']:
            confidence_score += 1.5
            reasons.append("MACD histogram increasing")
        elif current['MACD_Hist'] > 0:
            confidence_score += 0.5
            reasons.append("MACD histogram positive")

        # 5. Bollinger Band position (buying near lower band is ideal)
        bb_pos = current['BB_Position']
        if bb_pos < 0.3:
            confidence_score += 1
            reasons.append("Near lower BB (oversold)")
        elif bb_pos < 0.5:
            confidence_score += 0.5
            reasons.append("Below BB middle")

        # 6. Volume confirmation (if available)
        if current['Volume_Ratio'] > 1.2:
            confidence_score += 0.5
            reasons.append("High volume")

        # 7. ADX trend strength confirmation (soft booster)
        adx_score, adx_value = self.market_filters.get_adx_score(data)
        if adx_value >= 25:
            confidence_score += 1.5
            reasons.append(f"Strong trend ADX ({adx_value:.1f})")
        elif adx_value >= 20:
            confidence_score += 1.0
            reasons.append(f"Trending ADX ({adx_value:.1f})")
        elif adx_value >= 15:
            confidence_score += 0.5
            reasons.append(f"Weak ADX ({adx_value:.1f})")

        # 8. EMA slope confirmation (soft booster)
        slope_score, slope_direction = self.market_filters.get_ema_slope_score(data)
        if slope_direction == "bullish":
            confidence_score += 1.5
            reasons.append("EMA slope bullish")
        elif slope_direction == "flat":
            confidence_score += 0.3
            reasons.append("EMA slope flat")
        # No points for bearish slope on BUY

        # Calculate final confidence
        confidence = confidence_score / max_score

        # Only signal if confidence meets minimum
        if confidence >= self.config['min_confidence']:
            return {
                'action': TradingAction.BUY,
                'confidence': confidence,
                'reason': ' | '.join(reasons),
                'stop_loss': None,  # Will be calculated by risk manager
                'take_profit': None,
                # Additional metrics for multi-pair ranking
                'trend_strength': abs(current.get('Trend_Strength', 0)),
                'rsi': current.get('RSI', 50),
                'bb_position': current.get('BB_Position', 0.5),
                'volume_ratio': current.get('Volume_Ratio', 1.0),
                'adx': adx_value,
                'slope_direction': slope_direction
            }

        return self._no_signal(f"Low confidence ({confidence:.1%})")

    def _check_sell_conditions(self, current, previous, data):
        """Check all conditions for a SELL signal"""
        confidence_score = 0
        max_score = 11  # Increased from 8 to accommodate new boosters
        reasons = []

        # 1. Trend filter: Price below EMA 21
        if current['Close'] < current['EMA_Slow']:
            confidence_score += 2
            reasons.append("Price below trend EMA")
        else:
            confidence_score += 0.5
            reasons.append("Price above trend EMA (weak)")

        # 2. EMA crossunder already confirmed
        confidence_score += 1.5
        reasons.append("EMA 5/13 bearish cross")

        # 3. RSI in favorable range
        rsi = current['RSI']
        if self.config['rsi_oversold'] < rsi < self.config['rsi_overbought']:
            confidence_score += 1.5
            reasons.append(f"RSI optimal ({rsi:.1f})")
        elif rsi > self.config['rsi_overbought']:
            confidence_score += 1
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # 4. MACD confirmation
        if current['MACD_Hist'] < 0 and current['MACD_Hist'] < previous['MACD_Hist']:
            confidence_score += 1.5
            reasons.append("MACD histogram decreasing")
        elif current['MACD_Hist'] < 0:
            confidence_score += 0.5
            reasons.append("MACD histogram negative")

        # 5. Bollinger Band position (selling near upper band is ideal)
        bb_pos = current['BB_Position']
        if bb_pos > 0.7:
            confidence_score += 1
            reasons.append("Near upper BB (overbought)")
        elif bb_pos > 0.5:
            confidence_score += 0.5
            reasons.append("Above BB middle")

        # 6. Volume confirmation
        if current['Volume_Ratio'] > 1.2:
            confidence_score += 0.5
            reasons.append("High volume")

        # 7. ADX trend strength confirmation (soft booster)
        adx_score, adx_value = self.market_filters.get_adx_score(data)
        if adx_value >= 25:
            confidence_score += 1.5
            reasons.append(f"Strong trend ADX ({adx_value:.1f})")
        elif adx_value >= 20:
            confidence_score += 1.0
            reasons.append(f"Trending ADX ({adx_value:.1f})")
        elif adx_value >= 15:
            confidence_score += 0.5
            reasons.append(f"Weak ADX ({adx_value:.1f})")

        # 8. EMA slope confirmation (soft booster)
        slope_score, slope_direction = self.market_filters.get_ema_slope_score(data)
        if slope_direction == "bearish":
            confidence_score += 1.5
            reasons.append("EMA slope bearish")
        elif slope_direction == "flat":
            confidence_score += 0.3
            reasons.append("EMA slope flat")
        # No points for bullish slope on SELL

        # Calculate final confidence
        confidence = confidence_score / max_score

        # Only signal if confidence meets minimum
        if confidence >= self.config['min_confidence']:
            return {
                'action': TradingAction.SELL,
                'confidence': confidence,
                'reason': ' | '.join(reasons),
                'stop_loss': None,
                'take_profit': None,
                # Additional metrics for multi-pair ranking
                'trend_strength': abs(current.get('Trend_Strength', 0)),
                'rsi': current.get('RSI', 50),
                'bb_position': current.get('BB_Position', 0.5),
                'volume_ratio': current.get('Volume_Ratio', 1.0),
                'adx': adx_value,
                'slope_direction': slope_direction
            }

        return self._no_signal(f"Low confidence ({confidence:.1%})")

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
        Determine if current position should be exited
        Uses stricter exit criteria for scalping
        """
        if 'EMA_Fast' not in data.columns:
            data = self.calculate_indicators(data)

        current = data.iloc[-1]

        # Exit on RSI extremes (quick profit taking)
        if position_type == 'long':
            # Exit long if RSI too high or bearish cross
            if current['RSI'] > self.config['rsi_extreme_high']:
                return True, "RSI extreme high - take profit"

            if detect_crossunder(data['EMA_Fast'], data['EMA_Medium']):
                return True, "EMA bearish crossunder"

        elif position_type == 'short':
            # Exit short if RSI too low or bullish cross
            if current['RSI'] < self.config['rsi_extreme_low']:
                return True, "RSI extreme low - take profit"

            if detect_crossover(data['EMA_Fast'], data['EMA_Medium']):
                return True, "EMA bullish crossover"

        return False, ""


if __name__ == "__main__":
    # Test the strategy
    import logging
    logging.basicConfig(level=logging.INFO)

    strategy = UltraScalpingStrategy()
    print(f"Strategy initialized: {strategy}")
    print(f"Config: {strategy.config}")
