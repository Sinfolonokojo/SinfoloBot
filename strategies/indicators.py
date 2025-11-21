"""
Technical Indicators Module
Contains all indicator calculation functions for trading strategies.
Ported from enhanced_backtest.py and optimized for MT5 integration.
"""

import pandas as pd
import numpy as np


def calculate_sma(data, period):
    """
    Calculate Simple Moving Average

    Args:
        data: Pandas Series or DataFrame with 'Close' column
        period: Period for SMA calculation

    Returns:
        Pandas Series with SMA values
    """
    if isinstance(data, pd.DataFrame):
        return data['Close'].rolling(window=period).mean()
    return pd.Series(data).rolling(window=period).mean()


def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average

    Args:
        data: Pandas Series or DataFrame with 'Close' column
        period: Period for EMA calculation

    Returns:
        Pandas Series with EMA values
    """
    if isinstance(data, pd.DataFrame):
        return data['Close'].ewm(span=period, adjust=False).mean()
    return pd.Series(data).ewm(span=period, adjust=False).mean()


def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI)

    Args:
        data: Pandas Series or DataFrame with 'Close' column
        period: Period for RSI calculation (default: 14)

    Returns:
        Pandas Series with RSI values (0-100)
    """
    if isinstance(data, pd.DataFrame):
        prices = data['Close']
    else:
        prices = pd.Series(data)

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Calculate Bollinger Bands

    Args:
        data: Pandas DataFrame with 'Close' column
        period: Period for moving average (default: 20)
        std_dev: Standard deviations for bands (default: 2)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if isinstance(data, pd.DataFrame):
        close = data['Close']
    else:
        close = pd.Series(data)

    middle_band = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return upper_band, middle_band, lower_band


def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Args:
        data: Pandas Series or DataFrame with 'Close' column
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if isinstance(data, pd.DataFrame):
        close = data['Close']
    else:
        close = pd.Series(data)

    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR) - volatility indicator

    Args:
        data: Pandas DataFrame with 'High', 'Low', 'Close' columns
        period: Period for ATR calculation (default: 14)

    Returns:
        Pandas Series with ATR values
    """
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()

    return atr


def calculate_stochastic(data, period=14, smooth_k=3, smooth_d=3):
    """
    Calculate Stochastic Oscillator

    Args:
        data: Pandas DataFrame with 'High', 'Low', 'Close' columns
        period: Look-back period (default: 14)
        smooth_k: %K smoothing (default: 3)
        smooth_d: %D smoothing (default: 3)

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = data['Low'].rolling(window=period).min()
    highest_high = data['High'].rolling(window=period).max()

    k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    k_smooth = k_percent.rolling(window=smooth_k).mean()
    d_smooth = k_smooth.rolling(window=smooth_d).mean()

    return k_smooth, d_smooth


def calculate_adx(data, period=14):
    """
    Calculate Average Directional Index (ADX) - trend strength

    Args:
        data: Pandas DataFrame with 'High', 'Low', 'Close' columns
        period: Period for ADX calculation (default: 14)

    Returns:
        Pandas Series with ADX values
    """
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    # Calculate True Range
    tr = calculate_atr(data, period=1) * period

    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx


def apply_all_indicators(data, config=None):
    """
    Apply all technical indicators to a DataFrame.
    This is the main function to be called by trading strategies.

    Args:
        data: Pandas DataFrame with OHLC columns
        config: Optional dict with indicator parameters

    Returns:
        DataFrame with all indicators added as new columns
    """
    df = data.copy()

    # Default configuration
    if config is None:
        config = {
            'sma_fast': 10,
            'sma_slow': 20,
            'sma_trend': 50,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14
        }

    # Simple Moving Averages
    df['SMA_Fast'] = calculate_sma(df, config.get('sma_fast', 10))
    df['SMA_Slow'] = calculate_sma(df, config.get('sma_slow', 20))
    df['SMA_Trend'] = calculate_sma(df, config.get('sma_trend', 50))

    # Exponential Moving Average
    df['EMA_Fast'] = calculate_ema(df, config.get('sma_fast', 10))

    # RSI
    df['RSI'] = calculate_rsi(df, config.get('rsi_period', 14))

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        df,
        config.get('bb_period', 20),
        config.get('bb_std', 2.0)
    )
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower

    # MACD
    macd, signal, hist = calculate_macd(
        df,
        config.get('macd_fast', 12),
        config.get('macd_slow', 26),
        config.get('macd_signal', 9)
    )
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist

    # ATR
    df['ATR'] = calculate_atr(df, config.get('atr_period', 14))

    # Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic(df)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d

    # ADX (Trend Strength)
    df['ADX'] = calculate_adx(df, config.get('adx_period', 14))

    return df


def detect_crossover(fast_series, slow_series):
    """
    Detect bullish crossover (fast crosses above slow)

    Args:
        fast_series: Fast moving average series
        slow_series: Slow moving average series

    Returns:
        Boolean: True if crossover detected on last bar
    """
    if len(fast_series) < 2 or len(slow_series) < 2:
        return False

    # Previous bar: fast was below slow
    # Current bar: fast is above slow
    prev_below = fast_series.iloc[-2] < slow_series.iloc[-2]
    curr_above = fast_series.iloc[-1] > slow_series.iloc[-1]

    return prev_below and curr_above


def detect_crossunder(fast_series, slow_series):
    """
    Detect bearish crossunder (fast crosses below slow)

    Args:
        fast_series: Fast moving average series
        slow_series: Slow moving average series

    Returns:
        Boolean: True if crossunder detected on last bar
    """
    if len(fast_series) < 2 or len(slow_series) < 2:
        return False

    # Previous bar: fast was above slow
    # Current bar: fast is below slow
    prev_above = fast_series.iloc[-2] > slow_series.iloc[-2]
    curr_below = fast_series.iloc[-1] < slow_series.iloc[-1]

    return prev_above and curr_below
