# SinfoloBot - Quick Start Guide

## 🚀 Easy Launch

### Option 1: Windows Batch File (Easiest!)
Simply **double-click** `START_BOT.bat` and choose from the menu.

### Option 2: Python Script
```bash
python start_bot.py
```

### Option 3: Direct Commands
```bash
# Run backtest
python run_multi_pair_bt.py

# Start dashboard
streamlit run dashboard.py

# Start live trading (careful!)
python multi_pair_bot.py
```

---

## 📋 Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Or if using virtual environment:
```bash
venv\Scripts\pip install -r requirements.txt
```

### 2. Install Streamlit (for Dashboard)
```bash
pip install streamlit
```

---

## 🎯 Menu Options

### [1] Run Backtest
- Tests strategy on historical data
- Shows performance metrics
- Saves results to `backtest_results.png`
- **Safe** - No real trading

### [2] Start Live Trading Bot
- Trades multiple pairs automatically
- **WARNING:** Uses real money!
- Requires MT5 running and logged in
- Test on demo first!

### [3] Start Prop Firm Bot
- Prop firm compliant bot
- Enforces strict risk rules
- **WARNING:** Uses real money!
- Only use with prop firm accounts

### [4] Open Dashboard
- Web-based UI (opens in browser)
- View backtest results
- Check strategy configuration
- Read strategy guide
- **Safe** - Read-only

### [5] View Strategy Summary
- Terminal-based summary
- Quick overview of strategy
- Performance metrics

---

## 📊 Dashboard Features

The dashboard (`streamlit run dashboard.py`) provides:

- **📈 Overview:** Key metrics and performance summary
- **📊 Backtest Results:** Visual charts and detailed statistics
- **⚙️ Configuration:** Current risk settings and parameters
- **📖 Strategy Guide:** Full strategy documentation

---

## ⚠️ Important Notes

### Before Live Trading:
1. ✅ Test on demo account for 2-4 weeks
2. ✅ Verify MT5 is running and logged in
3. ✅ Check config files (config/multi_pair_config.yaml)
4. ✅ Understand the risks
5. ✅ Start with minimum position sizes

### Current Strategy:
- **Pair:** XAUUSD H1 only
- **Annual Return:** 8.22%
- **Max Drawdown:** -1.73%
- **Risk per Trade:** 0.5%
- **Win Rate:** 53.8%

---

## 🔧 Troubleshooting

### "MetaTrader5 not found"
Install MT5 Python package:
```bash
pip install MetaTrader5
```

### "Streamlit not found"
Install Streamlit:
```bash
pip install streamlit
```

### "Config file not found"
Make sure you're running from the project root directory.

### Dashboard not opening
Try manually:
```bash
python -m streamlit run dashboard.py
```

---

## 📁 Important Files

- `START_BOT.bat` - Windows launcher (double-click)
- `start_bot.py` - Python launcher script
- `dashboard.py` - Web dashboard UI
- `run_multi_pair_bt.py` - Backtest script
- `multi_pair_bot.py` - Live trading bot
- `config/multi_pair_config.yaml` - Strategy configuration
- `STRATEGY_SUMMARY.md` - Full strategy documentation

---

## 🆘 Need Help?

- Read `STRATEGY_SUMMARY.md` for strategy details
- Check `config/multi_pair_config.yaml` for settings
- Run backtest first to understand performance
- Test on demo account before live trading

---

**Good luck with your trading!** 📈
