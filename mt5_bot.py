"""
MT5 Forex Trading Bot - Main Script
24/7 automated trading bot for MetaTrader 5
"""

import MetaTrader5 as mt5
import yaml
import logging
import time
from datetime import datetime
import sys
import os
from colorlog import ColoredFormatter

# Import our modules
from core.mt5_connector import MT5Connector
from core.mt5_data import MT5DataFetcher
from core.risk_manager import RiskManager
from core.database import TradeDatabase
from core.pair_scanner import PairScanner
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.enhanced_multi import EnhancedMultiStrategy
from strategies.ultra_scalping import UltraScalpingStrategy


class MT5TradingBot:
    """Main trading bot class"""

    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize trading bot

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.connector = None
        self.data_fetcher = MT5DataFetcher()
        self.risk_manager = RiskManager(self.config['risk'])
        self.strategy = self.load_strategy()
        self.pair_scanner = None  # Initialized after connector in initialize()

        # Initialize database (optional - hybrid approach)
        self.db = None
        if self.config.get('database', {}).get('enabled', False):
            self.db = TradeDatabase(self.config['database'])
            if self.db.enabled:
                self.logger.info("Database logging enabled")
            else:
                self.logger.warning("Database initialization failed - continuing without database")
        else:
            self.logger.info("Database logging disabled")

        # Bot state
        self.running = False
        self.open_positions = {}
        self.last_heartbeat = datetime.now()
        self.session_id = None
        self.last_equity_log = datetime.now()

    def setup_logging(self):
        """Setup colored logging"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))

        # Create formatter
        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # File handler
        if log_config.get('log_to_file', False):
            file_handler = logging.FileHandler(log_config.get('log_file', 'logs/trading_bot.log'))
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))

        # Setup root logger
        logging.basicConfig(
            level=log_level,
            handlers=[console_handler] + ([file_handler] if log_config.get('log_to_file') else [])
        )

    def load_strategy(self):
        """Load trading strategy from config"""
        strategy_name = self.config['trading'].get('active_strategy', 'sma_crossover')
        strategy_config = self.config['strategies'].get(strategy_name, {})

        if strategy_name == 'sma_crossover':
            return SMACrossoverStrategy(strategy_config)
        elif strategy_name == 'enhanced_multi':
            return EnhancedMultiStrategy(strategy_config)
        elif strategy_name == 'ultra_scalping':
            return UltraScalpingStrategy(strategy_config)
        else:
            self.logger.error(f"Unknown strategy: {strategy_name}")
            sys.exit(1)

    def initialize(self):
        """Initialize MT5 connection"""
        self.logger.info("=" * 70)
        self.logger.info("  MT5 FOREX TRADING BOT")
        self.logger.info("=" * 70)

        # Connect to MT5
        mt5_config = self.config['mt5']
        self.connector = MT5Connector(
            mt5_config['login'],
            mt5_config['password'],
            mt5_config['server'],
            mt5_config.get('timeout', 60000)
        )

        if not self.connector.initialize():
            self.logger.error("Failed to initialize MT5 connection")
            return False

        # Verify symbols
        symbols = self.config['trading']['symbols']
        self.logger.info(f"Verifying {len(symbols)} symbols...")

        for symbol in symbols:
            if not self.connector.check_symbol(symbol):
                self.logger.warning(f"Symbol {symbol} not available")

        self.logger.info(f"Strategy: {self.strategy.name}")
        self.logger.info(f"Risk per trade: {self.risk_manager.risk_per_trade}%")

        # Initialize pair scanner for multi-pair trading
        self.pair_scanner = PairScanner(
            self.config,
            self.data_fetcher,
            self.strategy,
            self.connector
        )

        # Log multi-pair scan settings
        multi_pair_config = self.config.get('trading', {}).get('multi_pair_scan', {})
        if multi_pair_config.get('enabled', False):
            self.logger.info(f"Multi-pair scanning: ENABLED")
            self.logger.info(f"  Max trades per scan: {multi_pair_config.get('max_trades_per_scan', 3)}")
            self.logger.info(f"  Pairs to scan: {len(symbols)}")
        else:
            self.logger.info("Multi-pair scanning: DISABLED (single pair mode)")

        self.logger.info("Bot initialization complete")

        return True

    def scan_markets(self):
        """Scan all configured symbols for trading opportunities"""
        symbols = self.config['trading']['symbols']
        timeframe = self.config['strategies'][self.config['trading']['active_strategy']]['timeframe']

        # Check if multi-pair scanning is enabled
        multi_pair_config = self.config.get('trading', {}).get('multi_pair_scan', {})
        if multi_pair_config.get('enabled', False):
            self._scan_markets_multi_pair(symbols, timeframe)
        else:
            self._scan_markets_single_pair(symbols, timeframe)

    def _scan_markets_multi_pair(self, symbols, timeframe):
        """
        Multi-pair scanning mode: Scan all pairs, rank them, trade top N.

        Args:
            symbols: List of currency pairs to scan
            timeframe: Timeframe for analysis
        """
        # Get top opportunities from all pairs
        top_opportunities = self.pair_scanner.get_top_opportunities(symbols, timeframe)

        if not top_opportunities:
            self.logger.debug("No qualifying opportunities found in multi-pair scan")
            return

        # Execute trades on top opportunities
        trades_executed = 0
        max_trades = self.config.get('trading', {}).get('multi_pair_scan', {}).get('max_trades_per_scan', 3)

        for result in top_opportunities:
            if trades_executed >= max_trades:
                break

            symbol = result['symbol']
            signal = result['signal']
            data = result['data']

            # Check position limits
            if not self._can_open_new_position(symbol):
                self.logger.info(f"Skipping {symbol}: Position limit reached")
                continue

            # Log the trade attempt
            self.logger.info(f"Trading #{trades_executed + 1}: {symbol}")
            self.logger.info(f"  Action: {signal['action']}")
            self.logger.info(f"  Confidence: {signal['confidence']:.1%}")
            self.logger.info(f"  Score: {result['ranking_score']:.3f}")
            self.logger.info(f"  Reason: {signal.get('reason', 'N/A')[:60]}")

            # Execute trade
            self.execute_trade(symbol, signal, data)
            trades_executed += 1

        if trades_executed > 0:
            self.logger.info(f"Multi-pair scan complete: {trades_executed} trade(s) executed")

    def _scan_markets_single_pair(self, symbols, timeframe):
        """
        Single-pair scanning mode: Original behavior, trade first valid signal.

        Args:
            symbols: List of currency pairs to scan
            timeframe: Timeframe for analysis
        """
        for symbol in symbols:
            try:
                # Check position limits
                if not self._can_open_new_position(symbol):
                    continue

                # Fetch market data
                data = self.data_fetcher.get_historical_data(symbol, timeframe, num_bars=500)
                if data is None or len(data) < 100:
                    self.logger.warning(f"Insufficient data for {symbol}")
                    continue

                # Prepare data with indicators
                data = self.strategy.prepare_data(data)
                if data is None:
                    continue

                # Generate signals
                signal = self.strategy.generate_signals(data)

                if signal['action'] in ['BUY', 'SELL']:
                    self.logger.info(f"Signal detected for {symbol}: {signal['action']}")
                    self.logger.info(f"  Reason: {signal['reason']}")
                    self.logger.info(f"  Confidence: {signal['confidence']:.1%}")

                    # Execute trade
                    self.execute_trade(symbol, signal, data)

            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")

    def execute_trade(self, symbol, signal, data):
        """
        Execute trade based on signal

        Args:
            symbol: Trading symbol
            signal: Trading signal dict
            data: Market data with indicators
        """
        # Get account info
        account_info = self.connector.get_account_info()
        if not account_info:
            self.logger.error("Cannot get account info")
            return

        # Check daily loss limit
        loss_limit_reached, reason = self.risk_manager.check_daily_loss_limit(account_info['balance'])
        if loss_limit_reached:
            self.logger.warning(f"Trade blocked: {reason}")
            return

        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)
        if not symbol_info:
            return

        # Get current price
        current_price = self.data_fetcher.get_current_price(symbol)
        if not current_price:
            return

        # Calculate SL/TP
        atr = data['ATR'].iloc[-1]

        sl_price = self.risk_manager.calculate_stop_loss(
            current_price['ask'] if signal['action'] == 'BUY' else current_price['bid'],
            atr,
            self.config['risk']['stop_loss'],
            signal['action']
        )

        tp_price = self.risk_manager.calculate_take_profit(
            current_price['ask'] if signal['action'] == 'BUY' else current_price['bid'],
            atr,
            self.config['risk']['take_profit'],
            sl_price,
            signal['action']
        )

        # Calculate position size
        sl_pips = self.risk_manager.calculate_stop_loss_pips(
            current_price['ask'] if signal['action'] == 'BUY' else current_price['bid'],
            sl_price,
            symbol_info
        )

        lot_size = self.risk_manager.calculate_position_size(
            account_info['balance'],
            symbol_info,
            sl_pips
        )

        # Prepare order
        order_type = mt5.ORDER_TYPE_BUY if signal['action'] == 'BUY' else mt5.ORDER_TYPE_SELL
        price = current_price['ask'] if signal['action'] == 'BUY' else current_price['bid']

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"{self.strategy.name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Validate order
        is_valid, error_msg = self.risk_manager.validate_order(request, account_info['balance'], symbol_info)
        if not is_valid:
            self.logger.error(f"Order validation failed: {error_msg}")
            return

        # Send order
        self.logger.info(f"Sending {signal['action']} order for {symbol}")
        self.logger.info(f"  Price: {price:.5f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}, Size: {lot_size:.2f}")

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed: {result.comment}")
        else:
            self.logger.info(f"Order executed successfully! Deal: {result.deal}")
            self.logger.info(f"  Fill price: {result.price:.5f}")

            # Track position
            entry_time = datetime.now()
            self.open_positions[result.deal] = {
                'symbol': symbol,
                'type': signal['action'],
                'entry_price': result.price,
                'sl': sl_price,
                'tp': tp_price,
                'lot_size': lot_size,
                'time': entry_time,
                'signal_reason': signal.get('reason', ''),
                'signal_confidence': signal.get('confidence', 0)
            }

            # Log to database (optional)
            if self.db and self.db.enabled:
                try:
                    # Note: This logs entry, exit will be logged when position closes
                    self.db.log_signal({
                        'timestamp': entry_time,
                        'symbol': symbol,
                        'strategy': self.strategy.name,
                        'signal': signal,
                        'market_state': {
                            'price': result.price,
                            'atr': atr,
                            'spread': symbol_info.get('spread', 0)
                        },
                        'executed': True
                    })
                except Exception as e:
                    self.logger.error(f"Failed to log signal to database: {e}")

    def manage_open_positions(self):
        """Monitor and manage open positions (trailing stops, etc.)"""
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return

        for position in positions:
            symbol = position.symbol

            # Check if this is our bot's position
            if position.magic != 234000:
                continue

            # Get current price
            current_price = self.data_fetcher.get_current_price(symbol)
            if not current_price:
                continue

            # Check trailing stop
            price = current_price['bid'] if position.type == mt5.ORDER_TYPE_BUY else current_price['ask']

            should_trail, new_sl = self.risk_manager.should_apply_trailing_stop(
                {
                    'price_open': position.price_open,
                    'sl': position.sl,
                    'tp': position.tp,
                    'type': mt5.ORDER_TYPE_BUY if position.type == 0 else mt5.ORDER_TYPE_SELL
                },
                price,
                self.config['risk'].get('trailing_stop', {})
            )

            if should_trail:
                self.modify_position_sl(position, new_sl)

    def modify_position_sl(self, position, new_sl):
        """Modify position stop loss"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl,
            "tp": position.tp,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"Trailing stop updated for {position.symbol}: SL={new_sl:.5f}")
        else:
            self.logger.warning(f"Failed to update trailing stop: {result.comment}")

    def _can_open_new_position(self, symbol):
        """Check if can open new position based on limits"""
        # Check max positions
        max_positions = self.config['trading'].get('max_open_positions', 3)
        current_positions = len(mt5.positions_get() or [])

        if current_positions >= max_positions:
            return False

        # Check max per symbol
        max_per_symbol = self.config['trading'].get('max_positions_per_pair', 1)
        symbol_positions = [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == 234000]

        if len(symbol_positions) >= max_per_symbol:
            return False

        return True

    def heartbeat(self):
        """Log heartbeat message and equity snapshots"""
        heartbeat_interval = self.config['bot'].get('heartbeat_interval', 300)

        if (datetime.now() - self.last_heartbeat).seconds >= heartbeat_interval:
            account_info = self.connector.get_account_info()
            if account_info:
                self.logger.info(
                    f"Heartbeat | Balance: ${account_info['balance']:,.2f} | "
                    f"Equity: ${account_info['equity']:,.2f} | "
                    f"Positions: {len(mt5.positions_get() or [])}"
                )

                # Log equity snapshot to database
                equity_interval = self.config.get('database', {}).get('log_equity_interval', 300)
                if self.db and self.db.enabled:
                    if (datetime.now() - self.last_equity_log).seconds >= equity_interval:
                        try:
                            self.db.log_equity_snapshot({
                                'timestamp': datetime.now(),
                                'balance': account_info['balance'],
                                'equity': account_info['equity'],
                                'open_positions': len(mt5.positions_get() or []),
                                'daily_pnl': self.risk_manager.daily_pnl
                            })
                            self.last_equity_log = datetime.now()
                        except Exception as e:
                            self.logger.error(f"Failed to log equity snapshot: {e}")

            self.last_heartbeat = datetime.now()

    def run(self):
        """Main bot loop"""
        if not self.initialize():
            self.logger.error("Initialization failed. Exiting.")
            return

        self.running = True
        scan_interval = self.config['bot'].get('scan_interval', 60)

        # Start trading session (database logging)
        if self.db and self.db.enabled:
            account_info = self.connector.get_account_info()
            self.session_id = self.db.start_session({
                'strategy': self.strategy.name,
                'config_snapshot': self.config,
                'initial_balance': account_info['balance'] if account_info else 0
            })

        self.logger.info(f"Bot is running. Scan interval: {scan_interval} seconds")
        self.logger.info("Press Ctrl+C to stop")

        try:
            while self.running:
                # Check connection
                if not self.connector.is_connected():
                    self.logger.warning("Connection lost. Attempting to reconnect...")
                    if not self.connector.reconnect():
                        self.logger.error("Reconnection failed. Stopping bot.")
                        break

                # Scan markets for opportunities
                self.scan_markets()

                # Manage existing positions
                self.manage_open_positions()

                # Heartbeat
                self.heartbeat()

                # Sleep
                time.sleep(scan_interval)

        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)

        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down bot...")
        self.running = False

        # End trading session (database logging)
        if self.db and self.db.enabled and self.session_id:
            try:
                account_info = self.connector.get_account_info()
                self.db.end_session(self.session_id, {
                    'final_balance': account_info['balance'] if account_info else 0,
                    'total_trades': len(self.risk_manager.daily_trades),
                    'total_pnl': self.risk_manager.daily_pnl
                })
            except Exception as e:
                self.logger.error(f"Failed to end session: {e}")

        # Close positions (optional - ask user)
        # positions = mt5.positions_get()
        # if positions:
        #     self.logger.info(f"Warning: {len(positions)} positions still open")

        # Disconnect
        if self.connector:
            self.connector.shutdown()

        # Close database connection
        if self.db:
            self.db.close()

        self.logger.info("Bot shutdown complete")


if __name__ == "__main__":
    bot = MT5TradingBot()
    bot.run()
