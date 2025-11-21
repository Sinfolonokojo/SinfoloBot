"""
Trading Bot Dashboard API
Flask backend to serve MongoDB data to React frontend
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from bson import json_util
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'mt5_trading_bot')

client = MongoClient(MONGO_URI)
db = client[DB_NAME]


def parse_json(data):
    """Parse MongoDB documents to JSON"""
    return json.loads(json_util.dumps(data))


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        client.admin.command('ping')
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall trading statistics"""
    try:
        # Get all trades
        trades = list(db.trades.find())

        if not trades:
            return jsonify({
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'best_trade': 0,
                'worst_trade': 0
            })

        # Calculate statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('performance', {}).get('profit_usd', 0) > 0]
        losing_trades = [t for t in trades if t.get('performance', {}).get('profit_usd', 0) < 0]

        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        profits = [t.get('performance', {}).get('profit_usd', 0) for t in trades]
        total_profit = sum(profits)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        best_trade = max(profits) if profits else 0
        worst_trade = min(profits) if profits else 0

        # Get latest session
        latest_session = db.trading_sessions.find_one(sort=[('start_time', -1)])

        return jsonify({
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_profit': round(total_profit, 2),
            'avg_profit': round(avg_profit, 2),
            'best_trade': round(best_trade, 2),
            'worst_trade': round(worst_trade, 2),
            'current_session': parse_json(latest_session) if latest_session else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trade history with pagination"""
    try:
        # Pagination parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        skip = (page - 1) * limit

        # Filter parameters
        symbol = request.args.get('symbol')
        strategy = request.args.get('strategy')

        # Build query
        query = {}
        if symbol:
            query['symbol'] = symbol
        if strategy:
            query['strategy'] = strategy

        # Get trades
        total = db.trades.count_documents(query)
        trades = list(db.trades.find(query).sort('entry.time', -1).skip(skip).limit(limit))

        return jsonify({
            'trades': parse_json(trades),
            'total': total,
            'page': page,
            'limit': limit,
            'pages': (total + limit - 1) // limit
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades/<trade_id>', methods=['GET'])
def get_trade(trade_id):
    """Get single trade details"""
    try:
        from bson.objectid import ObjectId
        trade = db.trades.find_one({'_id': ObjectId(trade_id)})

        if not trade:
            return jsonify({'error': 'Trade not found'}), 404

        return jsonify(parse_json(trade))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/equity', methods=['GET'])
def get_equity_curve():
    """Get equity curve data"""
    try:
        # Time range parameters
        days = int(request.args.get('days', 7))
        start_date = datetime.utcnow() - timedelta(days=days)

        # Get equity snapshots
        snapshots = list(db.equity_snapshots.find({
            'timestamp': {'$gte': start_date}
        }).sort('timestamp', 1))

        return jsonify({
            'snapshots': parse_json(snapshots),
            'count': len(snapshots)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get recent signals"""
    try:
        limit = int(request.args.get('limit', 20))

        signals = list(db.signals.find().sort('timestamp', -1).limit(limit))

        return jsonify({
            'signals': parse_json(signals),
            'count': len(signals)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get trading sessions"""
    try:
        limit = int(request.args.get('limit', 10))

        sessions = list(db.trading_sessions.find().sort('start_time', -1).limit(limit))

        return jsonify({
            'sessions': parse_json(sessions),
            'count': len(sessions)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get detailed performance metrics"""
    try:
        # Get all trades
        trades = list(db.trades.find())

        if not trades:
            return jsonify({
                'metrics': {},
                'message': 'No trades available'
            })

        # Calculate metrics
        profits = [t.get('performance', {}).get('profit_usd', 0) for t in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        total_profit = sum(profits)
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0

        # Win rate
        win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0

        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average RR ratio
        avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for profit in profits:
            if profit > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        return jsonify({
            'metrics': {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': round(win_rate, 2),
                'total_profit': round(total_profit, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'avg_rr_ratio': round(avg_rr, 2),
                'profit_factor': round(profit_factor, 2),
                'best_trade': round(max(profits), 2) if profits else 0,
                'worst_trade': round(min(profits), 2) if profits else 0,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("  TRADING BOT DASHBOARD API")
    print("=" * 70)
    print(f"MongoDB: {MONGO_URI}")
    print(f"Database: {DB_NAME}")
    print("API running on http://localhost:5000")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
