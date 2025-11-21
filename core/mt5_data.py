"""
MT5 Data Fetching Module
Handles historical and real-time data retrieval from MetaTrader 5.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging


class MT5DataFetcher:
    """Fetches market data from MetaTrader 5"""

    # Timeframe mapping
    TIMEFRAMES = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }

    def __init__(self):
        """Initialize data fetcher"""
        self.logger = logging.getLogger(__name__)

    def get_historical_data(self, symbol, timeframe, num_bars=500, start_date=None, end_date=None):
        """
        Fetch historical OHLCV data

        Args:
            symbol: Symbol name (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "H1", "M15")
            num_bars: Number of bars to fetch (default: 500)
            start_date: Optional start date (datetime object)
            end_date: Optional end date (datetime object)

        Returns:
            pandas.DataFrame: OHLCV data with datetime index
        """
        if timeframe not in self.TIMEFRAMES:
            self.logger.error(f"Invalid timeframe: {timeframe}")
            self.logger.info(f"Valid timeframes: {list(self.TIMEFRAMES.keys())}")
            return None

        mt5_timeframe = self.TIMEFRAMES[timeframe]

        try:
            # Fetch data using different methods based on parameters
            if start_date and end_date:
                rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            elif start_date:
                rates = mt5.copy_rates_from(symbol, mt5_timeframe, start_date, num_bars)
            else:
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_bars)

            if rates is None or len(rates) == 0:
                error_code, error_msg = mt5.last_error()
                self.logger.error(f"Failed to fetch data for {symbol}: {error_msg} (Code: {error_code})")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)

            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Rename columns to standard OHLC format
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume',
                'real_volume': 'Real_Volume'
            }, inplace=True)

            # Select only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            self.logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return None

    def get_current_price(self, symbol):
        """
        Get current bid/ask price for a symbol

        Args:
            symbol: Symbol name (e.g., "EURUSD")

        Returns:
            dict: {'bid': float, 'ask': float, 'spread': float, 'time': datetime}
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error(f"Failed to get tick for {symbol}: {mt5.last_error()}")
            return None

        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time)
        }

    def get_latest_bar(self, symbol, timeframe):
        """
        Get the most recent completed bar

        Args:
            symbol: Symbol name
            timeframe: Timeframe string (e.g., "H1")

        Returns:
            dict: OHLCV data for latest bar
        """
        df = self.get_historical_data(symbol, timeframe, num_bars=1)
        if df is None or df.empty:
            return None

        latest = df.iloc[-1]
        return {
            'time': latest.name,
            'open': latest['Open'],
            'high': latest['High'],
            'low': latest['Low'],
            'close': latest['Close'],
            'volume': latest['Volume']
        }

    def get_multiple_symbols_data(self, symbols, timeframe, num_bars=500):
        """
        Fetch data for multiple symbols

        Args:
            symbols: List of symbol names
            timeframe: Timeframe string
            num_bars: Number of bars to fetch

        Returns:
            dict: {symbol: DataFrame}
        """
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, timeframe, num_bars)
            if df is not None:
                data[symbol] = df
            else:
                self.logger.warning(f"Skipping {symbol} - failed to fetch data")

        return data

    def resample_data(self, df, target_timeframe):
        """
        Resample data to a different timeframe

        Args:
            df: DataFrame with OHLCV data
            target_timeframe: Target timeframe (e.g., '4H', '1D')

        Returns:
            DataFrame: Resampled data
        """
        resampled = df.resample(target_timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

        # Remove rows with NaN (incomplete periods)
        resampled.dropna(inplace=True)

        return resampled

    def get_tick_data(self, symbol, num_ticks=1000, start_date=None, end_date=None):
        """
        Get tick data (all price changes)

        Args:
            symbol: Symbol name
            num_ticks: Number of ticks to fetch
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame: Tick data with bid/ask prices
        """
        try:
            if start_date and end_date:
                ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
            elif start_date:
                ticks = mt5.copy_ticks_from(symbol, start_date, num_ticks, mt5.COPY_TICKS_ALL)
            else:
                # Get ticks from current position
                now = datetime.now()
                ticks = mt5.copy_ticks_from(symbol, now, num_ticks, mt5.COPY_TICKS_ALL)

            if ticks is None or len(ticks) == 0:
                self.logger.error(f"Failed to fetch ticks for {symbol}")
                return None

            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching tick data: {e}")
            return None

    def calculate_spread_stats(self, symbol, hours=24):
        """
        Calculate spread statistics over a time period

        Args:
            symbol: Symbol name
            hours: Number of hours to analyze

        Returns:
            dict: Spread statistics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)

        ticks = self.get_tick_data(symbol, start_date=start_date, end_date=end_date)
        if ticks is None:
            return None

        ticks['spread'] = ticks['ask'] - ticks['bid']

        return {
            'avg_spread': ticks['spread'].mean(),
            'min_spread': ticks['spread'].min(),
            'max_spread': ticks['spread'].max(),
            'current_spread': ticks['spread'].iloc[-1],
            'period_hours': hours
        }

    def is_market_open(self, symbol):
        """
        Check if market is currently open for trading

        Args:
            symbol: Symbol name

        Returns:
            bool: True if market is open
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False

        # Check if trading is allowed
        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False

        # Check current time against trading sessions
        current_time = datetime.now()

        # For forex, check if it's weekend
        if current_time.weekday() >= 5:  # Saturday or Sunday
            return False

        return True

    def wait_for_new_bar(self, symbol, timeframe, timeout=300):
        """
        Wait for a new bar to form (useful for bar-based strategies)

        Args:
            symbol: Symbol name
            timeframe: Timeframe to monitor
            timeout: Maximum seconds to wait

        Returns:
            dict: New bar data or None if timeout
        """
        import time

        initial_bar = self.get_latest_bar(symbol, timeframe)
        if initial_bar is None:
            return None

        initial_time = initial_bar['time']
        start_wait = datetime.now()

        while (datetime.now() - start_wait).seconds < timeout:
            time.sleep(1)
            current_bar = self.get_latest_bar(symbol, timeframe)

            if current_bar and current_bar['time'] > initial_time:
                self.logger.info(f"New {timeframe} bar formed for {symbol}")
                return current_bar

        self.logger.warning(f"Timeout waiting for new bar on {symbol}")
        return None


# Helper function for quick data access
def get_data(symbol, timeframe='H1', num_bars=500):
    """
    Quick helper to fetch data

    Args:
        symbol: Symbol name
        timeframe: Timeframe string
        num_bars: Number of bars

    Returns:
        DataFrame: OHLCV data
    """
    fetcher = MT5DataFetcher()
    return fetcher.get_historical_data(symbol, timeframe, num_bars)


if __name__ == "__main__":
    # Example usage
    import MetaTrader5 as mt5

    logging.basicConfig(level=logging.INFO)

    # Initialize MT5 (make sure it's running)
    if not mt5.initialize():
        print("Failed to initialize MT5")
        quit()

    # Create data fetcher
    fetcher = MT5DataFetcher()

    # Fetch EURUSD hourly data
    print("\nFetching EURUSD H1 data...")
    df = fetcher.get_historical_data("EURUSD", "H1", num_bars=100)
    if df is not None:
        print(df.tail())
        print(f"\nData shape: {df.shape}")

    # Get current price
    print("\nCurrent EURUSD price:")
    price = fetcher.get_current_price("EURUSD")
    print(price)

    # Check if market is open
    print(f"\nMarket open: {fetcher.is_market_open('EURUSD')}")

    mt5.shutdown()
