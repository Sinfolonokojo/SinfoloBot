"""
MongoDB Database Integration for Trading Bot
Stores trades, equity snapshots, and signals for analysis and dashboard.
Optional module - bot works without it if MongoDB is not available.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import pytz


class TradeDatabase:
    """
    MongoDB database handler for trading bot data.
    Stores trades, equity curves, and signals for analysis.
    """

    def __init__(self, config: Dict):
        """
        Initialize database connection

        Args:
            config: Database configuration dict with:
                - connection_string: MongoDB connection string
                - database_name: Database name
                - enabled: Whether to use database
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = config.get('enabled', False)
        self.client = None
        self.db = None

        # Collection references
        self.trades_collection = None
        self.equity_collection = None
        self.signals_collection = None
        self.sessions_collection = None

        if self.enabled:
            self._connect()

    def _connect(self):
        """Connect to MongoDB database"""
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure

            connection_string = self.config.get('connection_string', 'mongodb://localhost:27017')
            database_name = self.config.get('database_name', 'mt5_trading_bot')

            self.logger.info(f"Connecting to MongoDB: {database_name}")

            # Create client with timeout
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000
            )

            # Test connection
            self.client.admin.command('ping')

            # Get database
            self.db = self.client[database_name]

            # Get collections
            self.trades_collection = self.db.trades
            self.equity_collection = self.db.equity_snapshots
            self.signals_collection = self.db.signals
            self.sessions_collection = self.db.trading_sessions

            # Create indexes for performance
            self._create_indexes()

            self.logger.info("MongoDB connected successfully")

        except ImportError:
            self.logger.warning(
                "pymongo not installed. Install with: pip install pymongo\n"
                "Database logging disabled."
            )
            self.enabled = False
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            self.logger.warning("Database logging disabled. Bot will continue without database.")
            self.enabled = False

    def _create_indexes(self):
        """Create indexes for better query performance"""
        if not self.enabled:
            return

        try:
            # Trades indexes
            self.trades_collection.create_index([("entry_time", -1)])  # Most recent first
            self.trades_collection.create_index([("symbol", 1), ("entry_time", -1)])
            self.trades_collection.create_index([("strategy", 1), ("entry_time", -1)])
            self.trades_collection.create_index([("exit_time", -1)])

            # Equity snapshots indexes
            self.equity_collection.create_index([("timestamp", -1)])

            # Signals indexes
            self.signals_collection.create_index([("timestamp", -1)])
            self.signals_collection.create_index([("symbol", 1), ("timestamp", -1)])

            # Sessions indexes
            self.sessions_collection.create_index([("start_time", -1)])

            self.logger.debug("Database indexes created")

        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")

    def log_trade(self, trade_data: Dict):
        """
        Log a completed trade to database

        Args:
            trade_data: Dict with trade information:
                - symbol: Trading pair
                - strategy: Strategy name
                - signal: Signal details (action, confidence, reason)
                - entry: Entry details (time, price, sl, tp)
                - exit: Exit details (time, price, reason)
                - performance: Profit/loss details
                - config_snapshot: Strategy config used
        """
        if not self.enabled:
            return

        try:
            # Ensure timestamp is timezone-aware
            if 'entry' in trade_data and 'time' in trade_data['entry']:
                entry_time = trade_data['entry']['time']
                if isinstance(entry_time, datetime) and entry_time.tzinfo is None:
                    trade_data['entry']['time'] = pytz.UTC.localize(entry_time)

            if 'exit' in trade_data and 'time' in trade_data['exit']:
                exit_time = trade_data['exit']['time']
                if isinstance(exit_time, datetime) and exit_time.tzinfo is None:
                    trade_data['exit']['time'] = pytz.UTC.localize(exit_time)

            # Add timestamp for database record
            trade_data['logged_at'] = datetime.now(pytz.UTC)

            # Insert into database
            result = self.trades_collection.insert_one(trade_data)

            self.logger.debug(f"Trade logged to database: {result.inserted_id}")

        except Exception as e:
            self.logger.error(f"Failed to log trade: {e}")
            # Don't raise - database logging is optional

    def log_equity_snapshot(self, equity_data: Dict):
        """
        Log equity snapshot for equity curve tracking

        Args:
            equity_data: Dict with:
                - timestamp: Current time
                - balance: Account balance
                - equity: Account equity
                - open_positions: Number of open positions
                - daily_pnl: P&L for the day
                - strategy_breakdown: P&L per strategy (optional)
        """
        if not self.enabled:
            return

        try:
            # Ensure timestamp is timezone-aware
            if 'timestamp' in equity_data:
                timestamp = equity_data['timestamp']
                if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
                    equity_data['timestamp'] = pytz.UTC.localize(timestamp)
            else:
                equity_data['timestamp'] = datetime.now(pytz.UTC)

            # Insert into database
            self.equity_collection.insert_one(equity_data)

            self.logger.debug("Equity snapshot logged")

        except Exception as e:
            self.logger.error(f"Failed to log equity snapshot: {e}")

    def log_signal(self, signal_data: Dict):
        """
        Log trading signal (both executed and rejected)

        Args:
            signal_data: Dict with:
                - timestamp: Signal time
                - symbol: Trading pair
                - strategy: Strategy name
                - signal: Signal details (action, confidence, reason)
                - market_state: Current market conditions
                - executed: Whether signal was executed
        """
        if not self.enabled or not self.config.get('log_all_signals', False):
            return  # Only log if explicitly enabled

        try:
            # Ensure timestamp is timezone-aware
            if 'timestamp' in signal_data:
                timestamp = signal_data['timestamp']
                if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
                    signal_data['timestamp'] = pytz.UTC.localize(timestamp)
            else:
                signal_data['timestamp'] = datetime.now(pytz.UTC)

            # Insert into database
            self.signals_collection.insert_one(signal_data)

            self.logger.debug(f"Signal logged: {signal_data['signal']['action']}")

        except Exception as e:
            self.logger.error(f"Failed to log signal: {e}")

    def start_session(self, session_data: Dict) -> Optional[str]:
        """
        Start a new trading session

        Args:
            session_data: Dict with:
                - start_time: Session start time
                - strategy: Strategy name
                - config_snapshot: Configuration used
                - initial_balance: Starting balance

        Returns:
            str: Session ID if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            session_data['start_time'] = datetime.now(pytz.UTC)
            session_data['status'] = 'active'

            result = self.sessions_collection.insert_one(session_data)

            self.logger.info(f"Trading session started: {result.inserted_id}")
            return str(result.inserted_id)

        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            return None

    def end_session(self, session_id: str, final_data: Dict):
        """
        End a trading session

        Args:
            session_id: Session ID from start_session
            final_data: Dict with:
                - end_time: Session end time
                - final_balance: Ending balance
                - total_trades: Number of trades
                - total_pnl: Total profit/loss
        """
        if not self.enabled or not session_id:
            return

        try:
            from bson import ObjectId

            final_data['end_time'] = datetime.now(pytz.UTC)
            final_data['status'] = 'completed'

            self.sessions_collection.update_one(
                {'_id': ObjectId(session_id)},
                {'$set': final_data}
            )

            self.logger.info(f"Trading session ended: {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to end session: {e}")

    def get_trades(self, filters: Optional[Dict] = None, limit: int = 100) -> List[Dict]:
        """
        Retrieve trades from database

        Args:
            filters: Optional MongoDB query filters
            limit: Maximum number of trades to return

        Returns:
            List of trade dicts
        """
        if not self.enabled:
            return []

        try:
            filters = filters or {}
            trades = list(
                self.trades_collection
                .find(filters)
                .sort('entry.time', -1)
                .limit(limit)
            )

            # Convert ObjectId to string for JSON serialization
            for trade in trades:
                trade['_id'] = str(trade['_id'])

            return trades

        except Exception as e:
            self.logger.error(f"Failed to retrieve trades: {e}")
            return []

    def get_equity_curve(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Retrieve equity curve data

        Args:
            start_time: Start time (optional)
            end_time: End time (optional)

        Returns:
            List of equity snapshots
        """
        if not self.enabled:
            return []

        try:
            query = {}
            if start_time or end_time:
                query['timestamp'] = {}
                if start_time:
                    query['timestamp']['$gte'] = start_time
                if end_time:
                    query['timestamp']['$lte'] = end_time

            snapshots = list(
                self.equity_collection
                .find(query)
                .sort('timestamp', 1)
            )

            # Convert ObjectId to string
            for snapshot in snapshots:
                snapshot['_id'] = str(snapshot['_id'])

            return snapshots

        except Exception as e:
            self.logger.error(f"Failed to retrieve equity curve: {e}")
            return []

    def get_performance_stats(self, strategy: Optional[str] = None,
                             start_time: Optional[datetime] = None) -> Dict:
        """
        Calculate performance statistics from database

        Args:
            strategy: Filter by strategy name (optional)
            start_time: Start time for calculation (optional)

        Returns:
            Dict with performance metrics
        """
        if not self.enabled:
            return {}

        try:
            # Build query
            query = {}
            if strategy:
                query['strategy'] = strategy
            if start_time:
                query['entry.time'] = {'$gte': start_time}

            # Get trades
            trades = list(self.trades_collection.find(query))

            if not trades:
                return {'error': 'No trades found'}

            # Calculate stats
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['performance']['profit_usd'] > 0]
            losing_trades = [t for t in trades if t['performance']['profit_usd'] < 0]

            total_profit = sum(t['performance']['profit_usd'] for t in trades)
            avg_profit = total_profit / total_trades

            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'avg_win': sum(t['performance']['profit_usd'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                'avg_loss': sum(t['performance']['profit_usd'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate performance stats: {e}")
            return {'error': str(e)}

    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test configuration
    config = {
        'enabled': True,
        'connection_string': 'mongodb://localhost:27017',
        'database_name': 'mt5_trading_bot_test',
        'log_all_signals': False
    }

    # Initialize database
    db = TradeDatabase(config)

    if db.enabled:
        print("Database connected successfully!")

        # Example: Log a trade
        sample_trade = {
            'symbol': 'EURUSD',
            'strategy': 'ultra_scalping',
            'signal': {
                'action': 'BUY',
                'confidence': 0.85,
                'reason': 'EMA bullish cross | RSI optimal'
            },
            'entry': {
                'time': datetime.now(pytz.UTC),
                'price': 1.08234,
                'sl': 1.08184,
                'tp': 1.08334
            },
            'exit': {
                'time': datetime.now(pytz.UTC),
                'price': 1.08289,
                'reason': 'Take profit hit'
            },
            'performance': {
                'profit_pips': 5.5,
                'profit_usd': 55.00,
                'duration_minutes': 12
            }
        }

        db.log_trade(sample_trade)
        print("Sample trade logged")

        # Example: Log equity snapshot
        equity_snapshot = {
            'timestamp': datetime.now(pytz.UTC),
            'balance': 10250.50,
            'equity': 10289.30,
            'open_positions': 2,
            'daily_pnl': 45.20
        }

        db.log_equity_snapshot(equity_snapshot)
        print("Equity snapshot logged")

        # Retrieve trades
        trades = db.get_trades(limit=10)
        print(f"\nRetrieved {len(trades)} trades from database")

        # Get performance stats
        stats = db.get_performance_stats()
        print(f"\nPerformance Stats: {stats}")

        db.close()
    else:
        print("Database connection failed - running in fallback mode")
