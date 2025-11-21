"""
Prop Firm Trading Bot
Designed for The5ers and similar prop firm challenges.

COMPLIANCE RULES:
1. EVERY order has Stop Loss attached
2. NO trading 2 min before/after high-impact news
3. Minimum 2 min hold time (no tick scalping)
4. Max daily loss protection
5. Simple technical strategy (no arbitrage)
"""

import MetaTrader5 as mt5
import yaml
import time
import logging
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.trend_pullback import TrendPullbackStrategy
from core.mt5_data import MT5DataFetcher
from core.risk_manager import RiskManager
from core.news_manager import NewsManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/prop_firm_bot.log')
    ]
)
logger = logging.getLogger(__name__)


class PropFirmBot:
    """
    Prop Firm Compliant Trading Bot

    Enforces all prop firm rules while trading the Trend Pullback strategy.
    """

    def __init__(self, config_path='config/prop_firm_config.yaml'):
        """Initialize the prop firm bot"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.data_fetcher = MT5DataFetcher()
        self.strategy = TrendPullbackStrategy(self.config['strategies']['trend_pullback'])
        self.risk_manager = RiskManager(self.config['risk'])
        self.news_manager = NewsManager(
            'inputs/blocked_times.csv',
            self.config['trading']['compliance']['news_buffer_minutes']
        )

        # State tracking
        self.is_running = False
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.open_positions = {}
        self.trade_open_times = {}  # Track when trades were opened

        # Compliance settings
        self.min_hold_minutes = self.config['trading']['compliance']['min_hold_minutes']
        self.max_trades_per_day = self.config['trading']['compliance']['max_trades_per_day']

        logger.info("Prop Firm Bot initialized")
        logger.info(f"Strategy: {self.strategy.name}")
        logger.info(f"News buffer: {self.news_manager.buffer_minutes} minutes")
        logger.info(f"Min hold time: {self.min_hold_minutes} minutes")

    def connect(self):
        """Connect to MetaTrader 5"""
        mt5_config = self.config['mt5']

        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False

        # Login
        authorized = mt5.login(
            mt5_config['login'],
            password=mt5_config['password'],
            server=mt5_config['server']
        )

        if not authorized:
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            return False

        account_info = mt5.account_info()
        logger.info(f"Connected to MT5: {account_info.name}")
        logger.info(f"Account balance: ${account_info.balance:.2f}")
        logger.info(f"Account equity: ${account_info.equity:.2f}")

        return True

    def check_compliance(self):
        """
        Check all compliance rules before trading

        Returns:
            tuple: (can_trade: bool, reason: str)
        """
        # 1. Check news blackout
        if self.news_manager.is_news_event():
            return False, "NEWS BLACKOUT - High impact news event"

        # 2. Check daily trade limit
        if self.daily_trades >= self.max_trades_per_day:
            return False, f"MAX TRADES REACHED ({self.daily_trades}/{self.max_trades_per_day})"

        # 3. Check daily loss limit
        daily_loss_limit = self.config['risk']['max_daily_loss']['percentage']
        account = mt5.account_info()
        if account:
            daily_loss_pct = (self.daily_pnl / account.balance) * 100
            if daily_loss_pct <= -daily_loss_limit:
                return False, f"DAILY LOSS LIMIT REACHED ({daily_loss_pct:.2f}%)"

        # 4. Check max open positions
        max_positions = self.config['trading']['max_open_positions']
        if len(self.open_positions) >= max_positions:
            return False, f"MAX POSITIONS OPEN ({len(self.open_positions)}/{max_positions})"

        # 5. Check trading hours
        if self.config['trading']['trading_hours']['enabled']:
            now = datetime.now()
            start = datetime.strptime(self.config['trading']['trading_hours']['start'], '%H:%M')
            end = datetime.strptime(self.config['trading']['trading_hours']['end'], '%H:%M')
            start = start.replace(year=now.year, month=now.month, day=now.day)
            end = end.replace(year=now.year, month=now.month, day=now.day)

            if not (start <= now <= end):
                return False, f"OUTSIDE TRADING HOURS ({now.strftime('%H:%M')})"

        return True, "OK"

    def check_min_hold_time(self, position_ticket):
        """
        Check if position has been held for minimum time

        Args:
            position_ticket: MT5 position ticket

        Returns:
            bool: True if can close (held long enough)
        """
        if position_ticket not in self.trade_open_times:
            return True  # Unknown position, allow close

        open_time = self.trade_open_times[position_ticket]
        hold_duration = (datetime.now() - open_time).total_seconds() / 60

        if hold_duration < self.min_hold_minutes:
            logger.debug(
                f"Position {position_ticket} held for {hold_duration:.1f} min "
                f"(min: {self.min_hold_minutes} min)"
            )
            return False

        return True

    def open_trade(self, signal, symbol):
        """
        Open a trade with mandatory Stop Loss

        CRITICAL: Never send order without SL attached

        Args:
            signal: Trade signal from strategy
            symbol: Trading symbol

        Returns:
            bool: True if trade opened successfully
        """
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return False

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return False

        # Determine order type and price
        if signal['action'] == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # Calculate SL and TP - MANDATORY
        atr = signal.get('atr', 0.001)
        sl_price, tp_price = self.strategy.get_sl_tp_prices(price, atr, signal['action'])

        # Calculate position size
        sl_pips = abs(price - sl_price) / symbol_info.point / 10
        account = mt5.account_info()
        lot_size = self.risk_manager.calculate_position_size(
            account.balance,
            {'volume_min': symbol_info.volume_min, 'volume_step': symbol_info.volume_step},
            sl_pips
        )

        # Prepare order request - SL is MANDATORY
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': lot_size,
            'type': order_type,
            'price': price,
            'sl': sl_price,  # MANDATORY - Never remove this
            'tp': tp_price,
            'deviation': 20,
            'magic': 123456,
            'comment': f'PropFirm_{signal["action"]}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        # Verify SL is set (double-check)
        if request['sl'] is None or request['sl'] == 0:
            logger.error("CRITICAL: Attempted to send order without Stop Loss!")
            return False

        # Send order
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return False

        # Track the trade
        self.daily_trades += 1
        self.open_positions[result.order] = {
            'symbol': symbol,
            'type': signal['action'],
            'entry_price': price,
            'sl': sl_price,
            'tp': tp_price,
            'lot_size': lot_size
        }
        self.trade_open_times[result.order] = datetime.now()

        logger.info(
            f"TRADE OPENED: {signal['action']} {symbol} @ {price:.5f} | "
            f"SL={sl_price:.5f} | TP={tp_price:.5f} | Lots={lot_size:.2f}"
        )
        logger.info(f"Reason: {signal['reason']}")

        return True

    def manage_positions(self, symbol):
        """
        Manage open positions - check for RSI reversal exits

        Args:
            symbol: Trading symbol
        """
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return

        # Get current data for exit signals
        timeframe = self.config['strategies']['trend_pullback']['timeframe']
        data = self.data_fetcher.get_historical_data(symbol, timeframe, num_bars=100)
        if data is None:
            return

        data = self.strategy.calculate_indicators(data)

        for position in positions:
            # Check minimum hold time
            if not self.check_min_hold_time(position.ticket):
                continue

            # Check for RSI reversal exit
            position_type = 'long' if position.type == mt5.POSITION_TYPE_BUY else 'short'
            should_exit, reason = self.strategy.should_exit_position(data, position_type)

            if should_exit:
                self.close_position(position, reason)

    def close_position(self, position, reason):
        """
        Close a position

        Args:
            position: MT5 position object
            reason: Reason for closing
        """
        # Determine close order type
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask

        # Close request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': position.symbol,
            'volume': position.volume,
            'type': order_type,
            'position': position.ticket,
            'price': price,
            'deviation': 20,
            'magic': 123456,
            'comment': f'Close_{reason[:20]}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Calculate P&L
            if position.type == mt5.POSITION_TYPE_BUY:
                pnl = (price - position.price_open) * position.volume * 100000
            else:
                pnl = (position.price_open - price) * position.volume * 100000

            self.daily_pnl += pnl

            # Clean up tracking
            if position.ticket in self.open_positions:
                del self.open_positions[position.ticket]
            if position.ticket in self.trade_open_times:
                del self.trade_open_times[position.ticket]

            logger.info(f"POSITION CLOSED: {position.symbol} | P&L=${pnl:.2f} | Reason: {reason}")
        else:
            logger.error(f"Failed to close position: {result.retcode}")

    def run(self):
        """Main trading loop"""
        logger.info("="*60)
        logger.info("  PROP FIRM BOT STARTING")
        logger.info("="*60)

        if not self.connect():
            logger.error("Failed to connect to MT5")
            return

        self.is_running = True
        scan_interval = self.config['bot']['scan_interval']

        # Reset daily counters at start
        self.daily_trades = 0
        self.daily_pnl = 0.0

        logger.info(f"Scan interval: {scan_interval} seconds")
        logger.info("Bot is now running. Press Ctrl+C to stop.")

        try:
            while self.is_running:
                # Check compliance before any trading
                can_trade, reason = self.check_compliance()

                if not can_trade:
                    logger.info(f"Trading blocked: {reason}")
                    time.sleep(scan_interval)
                    continue

                # Process each symbol
                for symbol in self.config['trading']['symbols']:
                    # Select symbol
                    if not mt5.symbol_select(symbol, True):
                        continue

                    # Get market data
                    timeframe = self.config['strategies']['trend_pullback']['timeframe']
                    data = self.data_fetcher.get_historical_data(symbol, timeframe, num_bars=100)

                    if data is None or len(data) < 60:
                        continue

                    # Manage existing positions
                    self.manage_positions(symbol)

                    # Check for new signals (only if no open position for this symbol)
                    positions = mt5.positions_get(symbol=symbol)
                    if positions and len(positions) > 0:
                        continue  # Already have position

                    # Generate signal
                    signal = self.strategy.generate_signals(data)

                    if signal['action'] in ['BUY', 'SELL']:
                        logger.info(f"Signal: {signal['action']} {symbol} | Confidence: {signal['confidence']:.2%}")

                        # Open trade
                        if self.open_trade(signal, symbol):
                            # Check news status
                            next_event = self.news_manager.get_next_event()
                            if next_event:
                                logger.info(
                                    f"Next news event: {next_event['event']} "
                                    f"in {next_event['minutes_until']:.0f} min"
                                )

                time.sleep(scan_interval)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.is_running = False
            mt5.shutdown()
            logger.info("Bot shutdown complete")


def main():
    """Entry point"""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    bot = PropFirmBot()
    bot.run()


if __name__ == "__main__":
    main()
