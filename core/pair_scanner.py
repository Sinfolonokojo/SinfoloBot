"""
Multi-Pair Scanner Module
Scans multiple currency pairs and ranks them by signal quality.
Returns the best trading opportunities based on confidence and filter conditions.
"""

import logging
from datetime import datetime
import pytz
from typing import List, Dict, Any, Optional, Tuple


class PairScanner:
    """
    Scans multiple currency pairs and ranks them by trading opportunity quality.

    Workflow:
    1. Fetch data for all configured pairs
    2. Generate signals with confidence scores
    3. Apply market condition filters
    4. Rank pairs by confidence score
    5. Return top N tradeable opportunities
    """

    def __init__(self, config: Dict[str, Any], data_fetcher, strategy, connector):
        """
        Initialize the pair scanner.

        Args:
            config: Bot configuration dict
            data_fetcher: MT5DataFetcher instance
            strategy: Trading strategy instance
            connector: MT5Connector instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.data_fetcher = data_fetcher
        self.strategy = strategy
        self.connector = connector

        # Multi-pair scan settings
        self.scan_config = config.get('trading', {}).get('multi_pair_scan', {})
        self.enabled = self.scan_config.get('enabled', False)
        self.max_trades = self.scan_config.get('max_trades_per_scan', 3)
        self.min_confidence = self.scan_config.get('min_confidence_threshold', 0.65)
        self.require_all_filters = self.scan_config.get('require_all_filters', True)
        self.log_results = self.scan_config.get('log_scan_results', True)

    def scan_all_pairs(self, symbols: List[str], timeframe: str) -> List[Dict[str, Any]]:
        """
        Scan all configured currency pairs for trading opportunities.

        Args:
            symbols: List of currency pair symbols to scan
            timeframe: Timeframe to analyze (e.g., 'M1', 'H1')

        Returns:
            List of scan results for each pair, including:
            - symbol: Currency pair symbol
            - signal: Signal dict with action, confidence, reason
            - data: Market data with indicators
            - filters_passed: Whether all filters passed
            - filter_reasons: List of failed filter reasons
            - symbol_info: MT5 symbol information
        """
        scan_results = []

        for symbol in symbols:
            try:
                result = self._scan_single_pair(symbol, timeframe)
                if result:
                    scan_results.append(result)
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")

        return scan_results

    def _scan_single_pair(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Scan a single currency pair for trading opportunity.

        Args:
            symbol: Currency pair symbol
            timeframe: Timeframe to analyze

        Returns:
            Scan result dict or None if failed
        """
        # Fetch market data
        data = self.data_fetcher.get_historical_data(symbol, timeframe, num_bars=500)
        if data is None or len(data) < 100:
            self.logger.debug(f"Insufficient data for {symbol}")
            return None

        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)
        if not symbol_info:
            self.logger.debug(f"Cannot get symbol info for {symbol}")
            return None

        # Prepare data with indicators
        data = self.strategy.prepare_data(data)
        if data is None:
            return None

        # Generate signal
        signal = self.strategy.generate_signals(data)

        # Check market filters
        filters_passed, filter_reasons = self._check_all_filters(data, symbol_info, symbol)

        # Extract additional metrics for ranking
        current = data.iloc[-1]
        metrics = {
            'trend_strength': abs(current.get('Trend_Strength', 0)),
            'atr': current.get('ATR', 0),
            'rsi': current.get('RSI', 50),
            'spread_pips': symbol_info.get('spread', 0) * 0.1,
        }

        return {
            'symbol': symbol,
            'signal': signal,
            'data': data,
            'filters_passed': filters_passed,
            'filter_reasons': filter_reasons,
            'symbol_info': symbol_info,
            'metrics': metrics
        }

    def _check_all_filters(self, data, symbol_info: Dict, symbol: str) -> Tuple[bool, List[str]]:
        """
        Check all market condition filters for a pair.

        Args:
            data: Market data with indicators
            symbol_info: MT5 symbol information
            symbol: Currency pair symbol

        Returns:
            Tuple of (all_passed, failed_reasons)
        """
        if not hasattr(self.strategy, 'market_filters'):
            return True, []

        # Get current time in UTC
        current_time = datetime.now(pytz.UTC)

        # Get filter config from strategy
        filter_config = self.config.get('strategies', {}).get(
            self.config['trading']['active_strategy'], {}
        ).get('market_filters', {})

        return self.strategy.market_filters.check_all_filters(
            data, symbol_info, symbol, current_time, filter_config
        )

    def rank_pairs(self, scan_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank scanned pairs by signal quality.

        Ranking criteria:
        1. Must have BUY or SELL signal (not HOLD)
        2. Must pass all market filters (if require_all_filters is True)
        3. Must meet minimum confidence threshold
        4. Ranked by confidence score (higher is better)
        5. Tie-breaker: trend strength

        Args:
            scan_results: List of scan results from scan_all_pairs()

        Returns:
            Sorted list of qualifying pairs, best opportunity first
        """
        qualifying_pairs = []

        for result in scan_results:
            signal = result['signal']

            # Skip if no actionable signal
            if signal['action'] in ['HOLD', None]:
                continue

            # Skip if filters didn't pass (when required)
            if self.require_all_filters and not result['filters_passed']:
                continue

            # Skip if below confidence threshold
            confidence = signal.get('confidence', 0)
            if confidence < self.min_confidence:
                continue

            # Add to qualifying pairs with ranking score
            result['ranking_score'] = self._calculate_ranking_score(result)
            qualifying_pairs.append(result)

        # Sort by ranking score (descending)
        qualifying_pairs.sort(key=lambda x: x['ranking_score'], reverse=True)

        return qualifying_pairs

    def _calculate_ranking_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate composite ranking score for a pair.

        Score components:
        - Confidence: 70% weight (primary factor)
        - Trend strength: 20% weight
        - Spread quality: 10% weight (lower spread = higher score)

        Args:
            result: Scan result dict

        Returns:
            Ranking score (0-1 scale)
        """
        confidence = result['signal'].get('confidence', 0)
        metrics = result.get('metrics', {})

        # Confidence (70% weight)
        confidence_score = confidence * 0.7

        # Trend strength (20% weight) - normalize to 0-1
        trend_strength = min(metrics.get('trend_strength', 0) / 1.0, 1.0)
        trend_score = trend_strength * 0.2

        # Spread quality (10% weight) - lower is better
        spread_pips = metrics.get('spread_pips', 2.0)
        # Normalize: 0 pips = 1.0, 3+ pips = 0
        spread_quality = max(0, 1 - (spread_pips / 3.0))
        spread_score = spread_quality * 0.1

        return confidence_score + trend_score + spread_score

    def get_top_opportunities(self, symbols: List[str], timeframe: str,
                              max_opportunities: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the top trading opportunities from all configured pairs.

        This is the main method to call for multi-pair scanning.

        Args:
            symbols: List of currency pair symbols to scan
            timeframe: Timeframe to analyze
            max_opportunities: Maximum number of opportunities to return
                             (defaults to config value)

        Returns:
            List of top opportunities, ranked by quality
        """
        if max_opportunities is None:
            max_opportunities = self.max_trades

        # Scan all pairs
        scan_results = self.scan_all_pairs(symbols, timeframe)

        # Rank pairs
        ranked_pairs = self.rank_pairs(scan_results)

        # Log scan results if enabled
        if self.log_results:
            self._log_scan_results(scan_results, ranked_pairs)

        # Return top N
        return ranked_pairs[:max_opportunities]

    def _log_scan_results(self, scan_results: List[Dict], ranked_pairs: List[Dict]):
        """
        Log detailed scan results for monitoring.

        Args:
            scan_results: All scan results
            ranked_pairs: Ranked qualifying pairs
        """
        self.logger.info("=" * 60)
        self.logger.info("MULTI-PAIR SCAN RESULTS")
        self.logger.info("=" * 60)

        # Summary
        total_scanned = len(scan_results)
        total_signals = sum(1 for r in scan_results
                          if r['signal']['action'] not in ['HOLD', None])
        total_qualifying = len(ranked_pairs)

        self.logger.info(f"Pairs scanned: {total_scanned}")
        self.logger.info(f"Signals generated: {total_signals}")
        self.logger.info(f"Qualifying (filters + confidence): {total_qualifying}")
        self.logger.info("-" * 60)

        # Show all pairs with signals
        for result in scan_results:
            symbol = result['symbol']
            signal = result['signal']
            action = signal['action']
            confidence = signal.get('confidence', 0)

            if action in ['HOLD', None]:
                status = "NO SIGNAL"
            elif not result['filters_passed']:
                status = f"FILTERED ({', '.join(result['filter_reasons'][:2])})"
            elif confidence < self.min_confidence:
                status = f"LOW CONF ({confidence:.1%})"
            else:
                status = f"{action} - {confidence:.1%}"

            self.logger.info(f"  {symbol}: {status}")

        # Show top opportunities
        if ranked_pairs:
            self.logger.info("-" * 60)
            self.logger.info("TOP OPPORTUNITIES:")
            for i, result in enumerate(ranked_pairs[:self.max_trades], 1):
                symbol = result['symbol']
                signal = result['signal']
                score = result['ranking_score']
                self.logger.info(
                    f"  #{i} {symbol}: {signal['action']} | "
                    f"Conf: {signal['confidence']:.1%} | Score: {score:.3f}"
                )
                self.logger.info(f"      Reason: {signal.get('reason', 'N/A')[:60]}")
        else:
            self.logger.info("No qualifying opportunities found")

        self.logger.info("=" * 60)

    def get_scan_summary(self, scan_results: List[Dict]) -> Dict[str, Any]:
        """
        Get a summary of scan results for database logging.

        Args:
            scan_results: List of scan results

        Returns:
            Summary dict for logging
        """
        return {
            'timestamp': datetime.now(),
            'total_pairs': len(scan_results),
            'pairs_with_signals': sum(1 for r in scan_results
                                     if r['signal']['action'] not in ['HOLD', None]),
            'pairs_passing_filters': sum(1 for r in scan_results
                                        if r['filters_passed']),
            'pairs_list': [
                {
                    'symbol': r['symbol'],
                    'action': r['signal']['action'],
                    'confidence': r['signal'].get('confidence', 0),
                    'filters_passed': r['filters_passed']
                }
                for r in scan_results
            ]
        }
