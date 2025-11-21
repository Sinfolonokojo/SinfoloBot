"""
Market Condition Filters
Filters to prevent trading during unfavorable market conditions.
Critical for ultra scalping strategies with tight stops.
"""

import logging
from datetime import datetime, time
import pytz
import pandas as pd


class MarketFilters:
    """Market condition filters to improve trade quality and reduce false signals"""

    def __init__(self, config=None):
        """
        Initialize market filters

        Args:
            config: Optional filter configuration dict
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

    def check_volatility(self, data, symbol_info=None, atr_min_ratio=0.7, atr_max_ratio=1.5):
        """
        Filter based on ATR (Average True Range) to avoid ranging or overly volatile markets

        Args:
            data: DataFrame with ATR indicator
            symbol_info: Optional symbol info dict
            atr_min_ratio: Minimum ATR as ratio of average (default 0.7 = 70% of avg)
            atr_max_ratio: Maximum ATR as ratio of average (default 1.5 = 150% of avg)

        Returns:
            tuple: (is_valid: bool, reason: str)
        """
        try:
            if 'ATR' not in data.columns:
                return True, ""  # Skip if ATR not available

            current_atr = data['ATR'].iloc[-1]
            avg_atr = data['ATR'].tail(20).mean()  # 20-period average

            if avg_atr == 0:
                return True, ""  # Avoid division by zero

            atr_ratio = current_atr / avg_atr

            # Too low = ranging market, whipsaw risk
            if atr_ratio < atr_min_ratio:
                self.logger.debug(
                    f"Volatility filter: ATR too low ({atr_ratio:.2f}x avg) - "
                    f"ranging market, high whipsaw risk"
                )
                return False, f"ATR too low ({atr_ratio:.2f}x avg) - ranging market"

            # Too high = news event or unusual volatility, unpredictable
            if atr_ratio > atr_max_ratio:
                self.logger.debug(
                    f"Volatility filter: ATR too high ({atr_ratio:.2f}x avg) - "
                    f"possible news event or unusual volatility"
                )
                return False, f"ATR too high ({atr_ratio:.2f}x avg) - volatile market"

            return True, ""

        except Exception as e:
            self.logger.error(f"Error in volatility filter: {e}")
            return True, ""  # Don't block on errors

    def check_spread(self, symbol_info, atr=None, max_spread_pips=2.0, max_spread_atr_pct=0.3):
        """
        Filter based on current spread - critical for scalping

        Args:
            symbol_info: Symbol information dict from MT5
            atr: Current ATR value (optional, for ratio check)
            max_spread_pips: Maximum allowed spread in pips (default 2.0)
            max_spread_atr_pct: Maximum spread as % of ATR (default 0.3 = 30%)

        Returns:
            tuple: (is_valid: bool, reason: str)
        """
        try:
            spread = symbol_info.get('spread', 0)
            point = symbol_info.get('point', 0.00001)

            # Convert spread to pips (for 5-digit broker, divide by 10)
            spread_pips = spread * 0.1

            # Check absolute spread limit
            if spread_pips > max_spread_pips:
                self.logger.debug(
                    f"Spread filter: Spread too wide ({spread_pips:.1f} pips) - "
                    f"exceeds maximum {max_spread_pips} pips"
                )
                return False, f"Spread too wide ({spread_pips:.1f} pips)"

            # Check spread relative to ATR (if provided)
            if atr and atr > 0:
                spread_value = spread * point
                spread_atr_ratio = spread_value / atr

                if spread_atr_ratio > max_spread_atr_pct:
                    self.logger.debug(
                        f"Spread filter: Spread is {spread_atr_ratio:.1%} of ATR - "
                        f"exceeds maximum {max_spread_atr_pct:.0%}"
                    )
                    return False, f"Spread too high ({spread_atr_ratio:.1%} of ATR)"

            return True, ""

        except Exception as e:
            self.logger.error(f"Error in spread filter: {e}")
            return True, ""  # Don't block on errors

    def check_trading_session(self, symbol, current_time=None, allowed_sessions=None):
        """
        Filter based on trading session - avoid low liquidity periods

        For EUR/USD:
        - Best: London (08:00-16:00 UTC) and NY (13:00-21:00 UTC)
        - Overlap: 13:00-16:00 UTC (highest liquidity)
        - Avoid: Asian session (22:00-08:00 UTC) - low volume, wider spreads

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            current_time: Current time (datetime object, UTC)
            allowed_sessions: List of allowed session names ['london', 'ny', 'overlap']

        Returns:
            tuple: (is_valid: bool, reason: str)
        """
        try:
            if current_time is None:
                current_time = datetime.now(pytz.UTC)

            # Ensure time is in UTC
            if current_time.tzinfo is None:
                current_time = pytz.UTC.localize(current_time)
            else:
                current_time = current_time.astimezone(pytz.UTC)

            current_hour = current_time.hour
            current_minute = current_time.minute
            current_decimal_hour = current_hour + (current_minute / 60.0)

            # Define sessions (UTC time)
            sessions = {
                'asian': (22, 8),      # 22:00-08:00 UTC (Tokyo)
                'london': (8, 16),     # 08:00-16:00 UTC
                'ny': (13, 21),        # 13:00-21:00 UTC
                'overlap': (13, 16),   # 13:00-16:00 UTC (London + NY)
            }

            # Default allowed sessions for EUR/USD
            if allowed_sessions is None:
                if 'EUR' in symbol or 'GBP' in symbol:
                    allowed_sessions = ['london', 'ny', 'overlap']
                elif 'JPY' in symbol:
                    allowed_sessions = ['asian', 'london', 'ny']
                else:
                    allowed_sessions = ['london', 'ny']

            # Check if current time is in any allowed session
            in_session = False
            active_session = None

            for session_name in allowed_sessions:
                if session_name not in sessions:
                    continue

                start, end = sessions[session_name]

                # Handle sessions that cross midnight
                if start > end:  # e.g., Asian session 22:00-08:00
                    if current_hour >= start or current_hour < end:
                        in_session = True
                        active_session = session_name
                        break
                else:
                    if start <= current_decimal_hour < end:
                        in_session = True
                        active_session = session_name
                        break

            if not in_session:
                self.logger.debug(
                    f"Session filter: Current time {current_time.strftime('%H:%M')} UTC "
                    f"not in allowed sessions {allowed_sessions}"
                )
                return False, f"Outside trading hours (current: {current_time.strftime('%H:%M')} UTC)"

            self.logger.debug(f"Session filter: Trading allowed - {active_session} session")
            return True, ""

        except Exception as e:
            self.logger.error(f"Error in session filter: {e}")
            return True, ""  # Don't block on errors

    def check_trend_strength(self, data, min_strength=0.3):
        """
        Filter based on trend strength - avoid choppy/sideways markets

        Uses existing Trend_Strength indicator from ultra_scalping strategy

        Args:
            data: DataFrame with Trend_Strength indicator
            min_strength: Minimum absolute trend strength (default 0.3)

        Returns:
            tuple: (is_valid: bool, reason: str)
        """
        try:
            if 'Trend_Strength' not in data.columns:
                return True, ""  # Skip if not available

            trend_strength = data['Trend_Strength'].iloc[-1]

            # Check absolute trend strength
            if abs(trend_strength) < min_strength:
                self.logger.debug(
                    f"Trend filter: Trend strength too weak ({trend_strength:.2f}) - "
                    f"choppy/sideways market"
                )
                return False, f"Weak trend ({abs(trend_strength):.2f}) - choppy market"

            return True, ""

        except Exception as e:
            self.logger.error(f"Error in trend strength filter: {e}")
            return True, ""  # Don't block on errors

    def check_rollover_time(self, current_time=None, rollover_start=21, rollover_end=22):
        """
        Filter to avoid trading during daily rollover when spreads widen

        Args:
            current_time: Current time (datetime object, UTC)
            rollover_start: Start hour of rollover period (default 21 UTC)
            rollover_end: End hour of rollover period (default 22 UTC)

        Returns:
            tuple: (is_valid: bool, reason: str)
        """
        try:
            if current_time is None:
                current_time = datetime.now(pytz.UTC)

            # Ensure time is in UTC
            if current_time.tzinfo is None:
                current_time = pytz.UTC.localize(current_time)
            else:
                current_time = current_time.astimezone(pytz.UTC)

            current_hour = current_time.hour

            # Check if in rollover window
            if rollover_start <= current_hour < rollover_end:
                self.logger.debug(
                    f"Rollover filter: Current time {current_time.strftime('%H:%M')} UTC "
                    f"is in rollover window ({rollover_start}:00-{rollover_end}:00 UTC)"
                )
                return False, f"Rollover period ({current_time.strftime('%H:%M')} UTC) - spreads widening"

            return True, ""

        except Exception as e:
            self.logger.error(f"Error in rollover filter: {e}")
            return True, ""  # Don't block on errors

    def get_adx_score(self, data, min_adx=20, strong_adx=25):
        """
        Get ADX-based trend confirmation score (not a hard filter)

        Args:
            data: DataFrame with ADX indicator
            min_adx: Minimum ADX for trending market (default 20)
            strong_adx: ADX level for strong trend (default 25)

        Returns:
            tuple: (score: float, adx_value: float)
                   score is 0-1 where higher = stronger trend
        """
        try:
            if 'ADX' not in data.columns:
                return 0.5, 0  # Neutral if not available

            adx = data['ADX'].iloc[-1]

            if pd.isna(adx):
                return 0.5, 0

            # Calculate score based on ADX level
            if adx < min_adx:
                # Weak trend - low score but don't block
                score = adx / min_adx * 0.5  # 0 to 0.5
            elif adx < strong_adx:
                # Moderate trend
                score = 0.5 + (adx - min_adx) / (strong_adx - min_adx) * 0.3  # 0.5 to 0.8
            else:
                # Strong trend
                score = min(1.0, 0.8 + (adx - strong_adx) / 25 * 0.2)  # 0.8 to 1.0

            return score, adx

        except Exception as e:
            self.logger.error(f"Error calculating ADX score: {e}")
            return 0.5, 0

    def get_ema_slope_score(self, data, lookback=5):
        """
        Get EMA slope confirmation score

        Args:
            data: DataFrame with EMA_Slow indicator
            lookback: Number of bars to calculate slope (default 5)

        Returns:
            tuple: (score: float, slope_direction: str)
                   score is 0-1 where higher = steeper slope in trade direction
        """
        try:
            if 'EMA_Slow' not in data.columns:
                return 0.5, "neutral"

            if len(data) < lookback:
                return 0.5, "neutral"

            # Calculate slope over lookback period
            ema_now = data['EMA_Slow'].iloc[-1]
            ema_prev = data['EMA_Slow'].iloc[-lookback]

            if pd.isna(ema_now) or pd.isna(ema_prev):
                return 0.5, "neutral"

            # Normalize by ATR to get relative slope
            atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else 0.0001
            if atr == 0:
                atr = 0.0001

            slope = (ema_now - ema_prev) / atr

            # Determine direction and score
            if abs(slope) < 0.1:
                return 0.3, "flat"
            elif slope > 0:
                score = min(1.0, 0.5 + abs(slope) * 0.5)
                return score, "bullish"
            else:
                score = min(1.0, 0.5 + abs(slope) * 0.5)
                return score, "bearish"

        except Exception as e:
            self.logger.error(f"Error calculating EMA slope score: {e}")
            return 0.5, "neutral"

    def check_all_filters(self, data, symbol_info, symbol, current_time=None, config=None):
        """
        Run all market condition filters

        Args:
            data: DataFrame with indicators
            symbol_info: Symbol information dict from MT5
            symbol: Trading symbol
            current_time: Current datetime (UTC)
            config: Filter configuration dict

        Returns:
            tuple: (all_passed: bool, failed_reasons: list)
        """
        if config is None:
            config = self.config

        failed_reasons = []

        # 1. Volatility filter (ATR)
        if config.get('volatility_filter', {}).get('enabled', True):
            atr_min = config.get('volatility_filter', {}).get('atr_min_ratio', 0.7)
            atr_max = config.get('volatility_filter', {}).get('atr_max_ratio', 1.5)

            is_valid, reason = self.check_volatility(data, symbol_info, atr_min, atr_max)
            if not is_valid:
                failed_reasons.append(reason)

        # 2. Spread filter
        if config.get('spread_filter', {}).get('enabled', True):
            max_spread_pips = config.get('spread_filter', {}).get('max_spread_pips', 2.0)
            max_spread_atr_pct = config.get('spread_filter', {}).get('max_spread_atr_pct', 0.3)

            current_atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else None
            is_valid, reason = self.check_spread(
                symbol_info, current_atr, max_spread_pips, max_spread_atr_pct
            )
            if not is_valid:
                failed_reasons.append(reason)

        # 3. Trading session filter
        if config.get('session_filter', {}).get('enabled', True):
            allowed_sessions = config.get('session_filter', {}).get('allowed_sessions')
            is_valid, reason = self.check_trading_session(symbol, current_time, allowed_sessions)
            if not is_valid:
                failed_reasons.append(reason)

        # 4. Trend strength filter
        if config.get('trend_filter', {}).get('enabled', True):
            min_strength = config.get('trend_filter', {}).get('min_strength', 0.3)
            is_valid, reason = self.check_trend_strength(data, min_strength)
            if not is_valid:
                failed_reasons.append(reason)

        # 5. Rollover time filter
        if config.get('rollover_filter', {}).get('enabled', True):
            rollover_start = config.get('rollover_filter', {}).get('start_hour', 21)
            rollover_end = config.get('rollover_filter', {}).get('end_hour', 22)
            is_valid, reason = self.check_rollover_time(current_time, rollover_start, rollover_end)
            if not is_valid:
                failed_reasons.append(reason)

        all_passed = len(failed_reasons) == 0

        if not all_passed:
            self.logger.info(f"Market filters blocked trade: {', '.join(failed_reasons)}")

        return all_passed, failed_reasons

    def get_filter_scores(self, data, symbol_info, config=None):
        """
        Get quality scores for each filter condition.
        Used for ranking pairs when comparing opportunities.

        Args:
            data: DataFrame with indicators
            symbol_info: Symbol information dict from MT5
            config: Filter configuration dict

        Returns:
            dict: Quality scores for each filter (0-1 scale, higher is better)
        """
        if config is None:
            config = self.config

        scores = {
            'volatility_score': 0.5,
            'spread_score': 0.5,
            'trend_score': 0.5
        }

        try:
            # Volatility score (closer to 1.0 ratio = better)
            if 'ATR' in data.columns:
                current_atr = data['ATR'].iloc[-1]
                avg_atr = data['ATR'].tail(20).mean()
                if avg_atr > 0:
                    atr_ratio = current_atr / avg_atr
                    # Optimal is around 1.0, score decreases as it moves away
                    deviation = abs(atr_ratio - 1.0)
                    scores['volatility_score'] = max(0, 1 - deviation)

            # Spread score (lower spread = higher score)
            if symbol_info:
                spread_pips = symbol_info.get('spread', 0) * 0.1
                # 0 pips = 1.0 score, 3+ pips = 0 score
                scores['spread_score'] = max(0, 1 - (spread_pips / 3.0))

            # Trend score (stronger trend = higher score)
            if 'Trend_Strength' in data.columns:
                trend_strength = abs(data['Trend_Strength'].iloc[-1])
                # Normalize: 0.5+ trend strength = 1.0 score
                scores['trend_score'] = min(1.0, trend_strength / 0.5)

        except Exception as e:
            self.logger.error(f"Error calculating filter scores: {e}")

        return scores

    def get_composite_score(self, data, symbol_info, config=None):
        """
        Get a single composite quality score combining all filters.

        Args:
            data: DataFrame with indicators
            symbol_info: Symbol information dict from MT5
            config: Filter configuration dict

        Returns:
            float: Composite score (0-1 scale)
        """
        scores = self.get_filter_scores(data, symbol_info, config)

        # Weighted average: volatility 30%, spread 40%, trend 30%
        composite = (
            scores['volatility_score'] * 0.3 +
            scores['spread_score'] * 0.4 +
            scores['trend_score'] * 0.3
        )

        return composite


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    sample_data = pd.DataFrame({
        'ATR': [0.0008, 0.0009, 0.0010, 0.0009, 0.0012],  # Last is high
        'Trend_Strength': [0.5, 0.6, 0.7, 0.4, 0.2],  # Last is weak
    })

    sample_symbol_info = {
        'spread': 5,  # 0.5 pips for 5-digit broker
        'point': 0.00001
    }

    filters = MarketFilters()

    # Test individual filters
    print("\n=== Testing Volatility Filter ===")
    is_valid, reason = filters.check_volatility(sample_data)
    print(f"Valid: {is_valid}, Reason: {reason}")

    print("\n=== Testing Spread Filter ===")
    is_valid, reason = filters.check_spread(sample_symbol_info, atr=0.001)
    print(f"Valid: {is_valid}, Reason: {reason}")

    print("\n=== Testing Session Filter ===")
    test_time = datetime(2025, 1, 15, 14, 30, tzinfo=pytz.UTC)  # 14:30 UTC (London/NY overlap)
    is_valid, reason = filters.check_trading_session('EURUSD', test_time)
    print(f"Valid: {is_valid}, Reason: {reason}")

    print("\n=== Testing Trend Strength Filter ===")
    is_valid, reason = filters.check_trend_strength(sample_data)
    print(f"Valid: {is_valid}, Reason: {reason}")

    # Test all filters
    print("\n=== Testing All Filters Together ===")
    config = {
        'volatility_filter': {'enabled': True, 'atr_min_ratio': 0.7, 'atr_max_ratio': 1.5},
        'spread_filter': {'enabled': True, 'max_spread_pips': 1.0, 'max_spread_atr_pct': 0.3},
        'session_filter': {'enabled': True, 'allowed_sessions': ['london', 'ny']},
        'trend_filter': {'enabled': True, 'min_strength': 0.3}
    }

    all_passed, failed_reasons = filters.check_all_filters(
        sample_data, sample_symbol_info, 'EURUSD', test_time, config
    )
    print(f"All passed: {all_passed}")
    if not all_passed:
        print(f"Failed reasons: {', '.join(failed_reasons)}")
