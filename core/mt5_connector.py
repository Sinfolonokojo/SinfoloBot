"""
MetaTrader 5 Connection Module
Handles initialization, authentication, and connection management for MT5.
"""

import MetaTrader5 as mt5
import logging
from datetime import datetime
import time


class MT5Connector:
    """Manages connection to MetaTrader 5 terminal"""

    def __init__(self, login, password, server, timeout=60000):
        """
        Initialize MT5 Connector

        Args:
            login: MT5 account number
            password: MT5 account password
            server: Broker server name
            timeout: Connection timeout in milliseconds
        """
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self.connected = False
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        """
        Initialize connection to MT5 terminal

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False

            self.logger.info("MT5 terminal initialized successfully")

            # Attempt to login
            authorized = mt5.login(
                login=self.login,
                password=self.password,
                server=self.server,
                timeout=self.timeout
            )

            if not authorized:
                error_code, error_msg = mt5.last_error()
                self.logger.error(f"Login failed: {error_msg} (Code: {error_code})")
                mt5.shutdown()
                return False

            self.connected = True
            self.logger.info(f"Logged in to account {self.login} on server {self.server}")

            # Display account info
            self._display_account_info()

            return True

        except Exception as e:
            self.logger.error(f"Error during MT5 initialization: {e}")
            return False

    def _display_account_info(self):
        """Display account information"""
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.warning("Failed to retrieve account info")
            return

        self.logger.info("=" * 60)
        self.logger.info("ACCOUNT INFORMATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Account ID: {account_info.login}")
        self.logger.info(f"Server: {account_info.server}")
        self.logger.info(f"Balance: ${account_info.balance:,.2f}")
        self.logger.info(f"Equity: ${account_info.equity:,.2f}")
        self.logger.info(f"Margin: ${account_info.margin:,.2f}")
        self.logger.info(f"Free Margin: ${account_info.margin_free:,.2f}")
        self.logger.info(f"Leverage: 1:{account_info.leverage}")
        self.logger.info(f"Currency: {account_info.currency}")
        self.logger.info("=" * 60)

    def get_account_info(self):
        """
        Get account information

        Returns:
            dict: Account information or None if not connected
        """
        if not self.is_connected():
            self.logger.error("Not connected to MT5")
            return None

        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None

        return {
            'login': account_info.login,
            'server': account_info.server,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'margin_free': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'profit': account_info.profit,
            'leverage': account_info.leverage,
            'currency': account_info.currency,
            'trade_allowed': account_info.trade_allowed,
            'trade_expert': account_info.trade_expert
        }

    def is_connected(self):
        """
        Check if connected to MT5

        Returns:
            bool: Connection status
        """
        if not self.connected:
            return False

        # Verify connection is still active
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            self.connected = False
            return False

        return True

    def reconnect(self, max_attempts=5, delay=5):
        """
        Attempt to reconnect to MT5

        Args:
            max_attempts: Maximum number of reconnection attempts
            delay: Delay between attempts in seconds

        Returns:
            bool: True if reconnected successfully
        """
        self.logger.info("Attempting to reconnect to MT5...")

        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"Reconnection attempt {attempt}/{max_attempts}")

            # Shutdown existing connection
            if self.connected:
                self.shutdown()

            # Wait before retry
            if attempt > 1:
                time.sleep(delay)

            # Try to initialize
            if self.initialize():
                self.logger.info("Reconnection successful!")
                return True

        self.logger.error(f"Failed to reconnect after {max_attempts} attempts")
        return False

    def check_symbol(self, symbol):
        """
        Check if symbol is available and enabled

        Args:
            symbol: Symbol name (e.g., "EURUSD")

        Returns:
            bool: True if symbol is available
        """
        if not self.is_connected():
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {symbol} not found")
            return False

        if not symbol_info.visible:
            self.logger.info(f"Symbol {symbol} is not visible, attempting to enable...")
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to enable symbol {symbol}")
                return False

        self.logger.info(f"Symbol {symbol} is available")
        return True

    def get_symbol_info(self, symbol):
        """
        Get detailed information about a symbol

        Args:
            symbol: Symbol name (e.g., "EURUSD")

        Returns:
            dict: Symbol information or None
        """
        if not self.is_connected():
            return None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Failed to get info for symbol {symbol}")
            return None

        return {
            'name': symbol_info.name,
            'description': symbol_info.description,
            'point': symbol_info.point,
            'digits': symbol_info.digits,
            'spread': symbol_info.spread,
            'trade_contract_size': symbol_info.trade_contract_size,
            'trade_tick_value': symbol_info.trade_tick_value,
            'trade_tick_size': symbol_info.trade_tick_size,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step,
            'currency_base': symbol_info.currency_base,
            'currency_profit': symbol_info.currency_profit,
            'bid': symbol_info.bid,
            'ask': symbol_info.ask,
            'trade_allowed': symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
        }

    def get_terminal_info(self):
        """
        Get MT5 terminal information

        Returns:
            dict: Terminal information
        """
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            return None

        return {
            'connected': terminal_info.connected,
            'trade_allowed': terminal_info.trade_allowed,
            'tradeapi_disabled': terminal_info.tradeapi_disabled,
            'mqid': terminal_info.mqid,
            'build': terminal_info.build,
            'company': terminal_info.company,
            'name': terminal_info.name,
            'path': terminal_info.path,
            'ping_last': terminal_info.ping_last
        }

    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 connection closed")

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()

    def __del__(self):
        """Destructor - ensure connection is closed"""
        self.shutdown()


def test_connection(login, password, server):
    """
    Test MT5 connection with provided credentials

    Args:
        login: MT5 account number
        password: MT5 password
        server: Broker server name

    Returns:
        bool: True if connection successful
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing MT5 connection...")

    connector = MT5Connector(login, password, server)

    if connector.initialize():
        logger.info("Connection test PASSED")
        account_info = connector.get_account_info()
        terminal_info = connector.get_terminal_info()

        logger.info(f"Terminal Build: {terminal_info.get('build')}")
        logger.info(f"Trade Allowed: {terminal_info.get('trade_allowed')}")

        connector.shutdown()
        return True
    else:
        logger.error("Connection test FAILED")
        return False


if __name__ == "__main__":
    # Example usage
    print("MT5 Connector Test")
    print("=" * 60)

    # Replace these with your actual credentials
    LOGIN = 12345678
    PASSWORD = "your_password"
    SERVER = "Broker-Demo"

    test_connection(LOGIN, PASSWORD, SERVER)
