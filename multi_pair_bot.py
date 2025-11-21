"""
Multi-Pair Live Trading Bot
Trades USDJPY (M5), EURUSD (H1), AUDUSD (M5) simultaneously
Optimized settings per pair based on backtest results
"""

import MetaTrader5 as mt5
import yaml
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from strategies.trend_pullback import TrendPullbackStrategy
from core.risk_manager import RiskManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multi_pair_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiPairBot:
    """
    Multi-Pair Trading Bot
    Trades multiple currency pairs with optimized settings per pair
    """

    def __init__(self, config_path='config/multi_pair_config.yaml'):
        """Initialize the multi-pair bot"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.running = False
        self.positions = {}  # Track positions per pair
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None

        # Initialize strategies and risk managers per pair
        self.strategies = {}
        self.risk_managers = {}
        self.timeframes = {}

        # Setup pairs
        self._setup_pairs()

        logger.info("Multi-Pair Bot initialized")
        logger.info(f"Trading pairs: {list(self.strategies.keys())}")

    def _setup_pairs(self):
        """Setup strategies and risk managers for each enabled pair"""
        for symbol, settings in self.config['trading']['pairs'].items():
            if settings.get('enabled', False):
                # Get pair-specific config or default
                if symbol in self.config['strategies']:
                    strategy_config = self.config['strategies'][symbol]
                else:
                    strategy_config = self.config['strategies']['trend_pullback']

                # Create strategy for this pair
                self.strategies[symbol] = TrendPullbackStrategy(config=strategy_config)
                self.risk_managers[symbol] = RiskManager(self.config['risk'])
                self.timeframes[symbol] = settings['timeframe']
                self.positions[symbol] = None

                logger.info(f"Setup {symbol} on {settings['timeframe']}")

    def connect(self):
        """Connect to MT5"""
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
            mt5.shutdown()
            return False

        account_info = mt5.account_info()
        logger.info(f"Connected to MT5: {account_info.login}")
        logger.info(f"Balance: ${account_info.balance:,.2f}")
        logger.info(f"Equity: ${account_info.equity:,.2f}")

        # Select all trading symbols
        for symbol in self.strategies.keys():
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Failed to select {symbol}")

        return True

    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        logger.info("Disconnected from MT5")

    def get_market_data(self, symbol, timeframe, bars=100):
        """Get market data for a symbol"""
        mt5_timeframe = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }.get(timeframe, mt5.TIMEFRAME_M5)

        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get data for {symbol}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'tick_volume': 'Volume'
        }, inplace=True)

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def check_trading_hours(self):
        """Check if within trading hours"""
        if not self.config['trading']['trading_hours']['enabled']:
            return True

        now = datetime.utcnow().time()
        start = datetime.strptime(self.config['trading']['trading_hours']['start'], '%H:%M').time()
        end = datetime.strptime(self.config['trading']['trading_hours']['end'], '%H:%M').time()

        return start <= now <= end

    def check_daily_limits(self):
        """Check daily trade and loss limits"""
        # Reset daily counters
        today = datetime.utcnow().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today

        # Check trade limit
        max_trades = self.config['trading']['compliance']['max_trades_per_day']
        if self.daily_trades >= max_trades:
            return False, f"Daily trade limit reached ({max_trades})"

        # Check loss limit
        if self.config['risk']['max_daily_loss']['enabled']:
            account = mt5.account_info()
            max_loss_pct = self.config['risk']['max_daily_loss']['percentage']
            max_loss = account.balance * (max_loss_pct / 100)

            if self.daily_pnl <= -max_loss:
                return False, f"Daily loss limit reached (${-self.daily_pnl:.2f})"

        return True, ""

    def get_open_positions(self):
        """Get all open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return {}

        pos_dict = {}
        for pos in positions:
            if pos.symbol in self.strategies:
                pos_dict[pos.symbol] = {
                    'ticket': pos.ticket,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'open_time': datetime.fromtimestamp(pos.time)
                }

        return pos_dict

    def open_position(self, symbol, signal, data):
        """Open a new position"""
        order_type = signal['action']
        atr = signal.get('atr', data['Close'].iloc[-1] * 0.01)

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return False

        price = tick.ask if order_type == 'BUY' else tick.bid

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return False

        symbol_dict = {
            'volume_min': symbol_info.volume_min,
            'volume_step': symbol_info.volume_step,
            'point': symbol_info.point,
            'digits': symbol_info.digits,
        }

        # Calculate SL/TP
        risk_manager = self.risk_managers[symbol]

        sl_price = risk_manager.calculate_stop_loss(
            price, atr,
            self.config['risk']['stop_loss'],
            order_type
        )

        tp_price = risk_manager.calculate_take_profit(
            price, atr,
            self.config['risk']['take_profit'],
            sl_price, order_type
        )

        # Calculate position size
        account = mt5.account_info()
        sl_pips = risk_manager.calculate_stop_loss_pips(price, sl_price, symbol_dict)
        lot_size = risk_manager.calculate_position_size(account.balance, symbol_dict, sl_pips)

        # Prepare order request
        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': lot_size,
            'type': mt5_order_type,
            'price': price,
            'sl': sl_price,
            'tp': tp_price,
            'deviation': 10,
            'magic': 123456,
            'comment': f'MultiPair_{symbol}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Order send failed for {symbol}: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed for {symbol}: {result.comment}")
            return False

        logger.info(f"OPENED {order_type} {symbol}: Price={price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}, Lot={lot_size}")
        self.daily_trades += 1

        return True

    def close_position(self, symbol, reason="Signal"):
        """Close an open position"""
        positions = mt5.positions_get(symbol=symbol)

        if positions is None or len(positions) == 0:
            return False

        for pos in positions:
            # Determine close order type
            if pos.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask

            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': pos.volume,
                'type': close_type,
                'position': pos.ticket,
                'price': price,
                'deviation': 10,
                'magic': 123456,
                'comment': f'Close_{reason}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close {symbol}: {result.comment if result else mt5.last_error()}")
                return False

            self.daily_pnl += pos.profit
            logger.info(f"CLOSED {symbol}: Profit=${pos.profit:.2f}, Reason={reason}")

        return True

    def process_pair(self, symbol):
        """Process trading logic for a single pair"""
        timeframe = self.timeframes[symbol]
        strategy = self.strategies[symbol]

        # Get market data
        data = self.get_market_data(symbol, timeframe, bars=100)
        if data is None:
            return

        # Calculate indicators
        data = strategy.calculate_indicators(data)

        # Get current positions
        open_positions = self.get_open_positions()
        has_position = symbol in open_positions

        # Check for exit signals if in position
        if has_position:
            pos = open_positions[symbol]
            pos_type = 'long' if pos['type'] == 'BUY' else 'short'

            should_exit, exit_reason = strategy.should_exit_position(data, pos_type)
            if should_exit:
                self.close_position(symbol, exit_reason)
                return

        # Check for entry signals if no position
        if not has_position:
            signal = strategy.generate_signals(data)

            if signal['action'] in ['BUY', 'SELL']:
                # Check if we can open more positions
                total_positions = len(open_positions)
                max_positions = self.config['trading']['max_open_positions']

                if total_positions >= max_positions:
                    logger.debug(f"Max positions reached ({max_positions}), skipping {symbol}")
                    return

                logger.info(f"Signal for {symbol}: {signal['action']} | Confidence: {signal['confidence']:.2f} | Reason: {signal['reason']}")
                self.open_position(symbol, signal, data)

    def run(self):
        """Main bot loop"""
        if not self.connect():
            return

        self.running = True
        scan_interval = self.config['bot']['scan_interval']

        logger.info("=" * 50)
        logger.info("  MULTI-PAIR BOT STARTED")
        logger.info(f"  Pairs: {', '.join(self.strategies.keys())}")
        logger.info(f"  Scan interval: {scan_interval}s")
        logger.info("=" * 50)

        try:
            while self.running:
                # Check trading hours
                if not self.check_trading_hours():
                    logger.debug("Outside trading hours, waiting...")
                    time.sleep(60)
                    continue

                # Check daily limits
                can_trade, reason = self.check_daily_limits()
                if not can_trade:
                    logger.warning(f"Trading paused: {reason}")
                    time.sleep(300)  # Wait 5 minutes
                    continue

                # Process each pair
                for symbol in self.strategies.keys():
                    try:
                        self.process_pair(symbol)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")

                # Log status
                open_positions = self.get_open_positions()
                account = mt5.account_info()

                if open_positions:
                    total_profit = sum(p['profit'] for p in open_positions.values())
                    logger.info(f"Open positions: {len(open_positions)} | Unrealized P/L: ${total_profit:.2f}")

                logger.debug(f"Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f} | Daily trades: {self.daily_trades}")

                # Wait for next scan
                time.sleep(scan_interval)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("Stopping Multi-Pair Bot...")

        # Option to close all positions
        open_positions = self.get_open_positions()
        if open_positions:
            logger.warning(f"Leaving {len(open_positions)} positions open")
            for symbol, pos in open_positions.items():
                logger.info(f"  {symbol}: {pos['type']} | P/L: ${pos['profit']:.2f}")

        self.disconnect()
        logger.info("Multi-Pair Bot stopped")


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("  MULTI-PAIR TRADING BOT")
    print("  Pairs: USDJPY (M5), EURUSD (H1), AUDUSD (M5)")
    print("  Strategy: Trend Pullback (Optimized per pair)")
    print("=" * 60 + "\n")

    print("Options:")
    print("1. Start live trading")
    print("2. Paper trading (simulation)")
    print("3. Exit")

    try:
        choice = input("\nSelect option (1-3): ").strip()
    except:
        choice = "3"

    if choice == "1":
        print("\n[WARNING] Starting LIVE trading!")
        confirm = input("Type 'YES' to confirm: ").strip()
        if confirm == "YES":
            bot = MultiPairBot()
            bot.run()
        else:
            print("Cancelled")
    elif choice == "2":
        print("\nPaper trading mode not yet implemented")
        print("Use the backtest runner: python run_multi_pair_bt.py")
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
