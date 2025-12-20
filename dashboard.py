"""
SinfoloBot Dashboard
Web-based UI to view backtest results, analysis, and bot status
"""

import streamlit as st
import pandas as pd
import yaml
from PIL import Image
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="SinfoloBot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 24px;
        font-weight: bold;
    }
    .green {
        color: #2ECC71;
    }
    .red {
        color: #E74C3C;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def load_config():
    """Load strategy configuration"""
    try:
        with open('config/multi_pair_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return None

def load_strategy_summary():
    """Load strategy summary"""
    try:
        with open('STRATEGY_SUMMARY.md', 'r') as f:
            return f.read()
    except:
        return None

def main():
    # Sidebar
    st.sidebar.title("📈 SinfoloBot")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "📊 Backtest Results", "⚙️ Configuration", "📖 Strategy Guide"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")

    # Load config
    config = load_config()

    if config:
        st.sidebar.metric("Risk per Trade", f"{config['risk']['risk_per_trade']}%")
        st.sidebar.metric("Max Drawdown Limit", f"{config['risk']['max_total_drawdown']['percentage']}%")

        # Count enabled pairs
        enabled_pairs = [symbol for symbol, settings in config['trading']['pairs'].items() if settings.get('enabled', False)]
        st.sidebar.metric("Active Pairs", len(enabled_pairs))

    # Main content
    if page == "🏠 Overview":
        show_overview()
    elif page == "📊 Backtest Results":
        show_backtest_results()
    elif page == "⚙️ Configuration":
        show_configuration()
    elif page == "📖 Strategy Guide":
        show_strategy_guide()

def show_overview():
    """Show overview page"""
    st.title("📈 SinfoloBot Dashboard")
    st.markdown("### XAUUSD Conservative Strategy")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Annual Return",
            value="8.22%",
            delta="Positive",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Max Drawdown",
            value="-1.73%",
            delta="-2.27% vs target",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="Win Rate",
            value="53.8%",
            delta="+3.8% vs 50%",
            delta_color="normal"
        )

    with col4:
        st.metric(
            label="Profit Factor",
            value="3.27",
            delta="Excellent",
            delta_color="normal"
        )

    st.markdown("---")

    # Performance summary
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Performance Summary")
        st.markdown("""
        **Backtest Period:** 1 Year (Jun 2024 - Jun 2025)

        - **Starting Balance:** $10,000
        - **Final Balance:** $10,822
        - **Total Profit:** $822
        - **Total Trades:** 13
        - **Winning Trades:** 7 (53.8%)
        - **Losing Trades:** 6 (46.2%)
        - **Grade:** B
        """)

    with col2:
        st.subheader("🎯 Strategy Details")
        st.markdown("""
        **Pair:** XAUUSD (Gold) H1

        **Risk Management:**
        - Risk per Trade: 0.5%
        - Stop Loss: 1.2x ATR
        - Take Profit: 2.5:1 R:R
        - Daily DD Limit: 2.0%
        - Total DD Limit: 4.0%

        **Trade Quality:**
        - Avg Win: $169.08
        - Avg Loss: -$60.27
        - Best Trade: $218.79
        - Worst Trade: -$73.41
        """)

    st.markdown("---")

    # Status indicators
    st.subheader("🔧 System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Check if MT5 connection is possible
        try:
            import MetaTrader5 as mt5
            mt5_status = "✅ Installed"
        except:
            mt5_status = "❌ Not Installed"
        st.info(f"**MT5:** {mt5_status}")

    with col2:
        # Check if config exists
        config_status = "✅ Loaded" if load_config() else "❌ Missing"
        st.info(f"**Config:** {config_status}")

    with col3:
        # Check if backtest results exist
        results_status = "✅ Available" if os.path.exists("backtest_results.png") else "❌ Missing"
        st.info(f"**Results:** {results_status}")

def show_backtest_results():
    """Show backtest results page"""
    st.title("📊 Backtest Results")

    # Display backtest image
    if os.path.exists("backtest_results.png"):
        st.image("backtest_results.png", caption="XAUUSD H1 - 1 Year Backtest Results", use_container_width=True)
    else:
        st.warning("Backtest results image not found. Run a backtest first using `python run_multi_pair_bt.py`")

    st.markdown("---")

    # Detailed metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("💰 Profitability")
        st.markdown("""
        - **Total Return:** $821.90
        - **Return %:** 8.22%
        - **Avg Profit/Trade:** $63.22
        - **Profit Factor:** 3.27
        """)

    with col2:
        st.subheader("📉 Risk Metrics")
        st.markdown("""
        - **Max Drawdown:** -1.73%
        - **Avg Win:** $169.08
        - **Avg Loss:** -$60.27
        - **Win/Loss Ratio:** 2.8:1
        """)

    with col3:
        st.subheader("📈 Trade Statistics")
        st.markdown("""
        - **Total Trades:** 13
        - **Win Rate:** 53.8%
        - **Winners:** 7
        - **Losers:** 6
        """)

    st.markdown("---")

    # Monthly projections
    st.subheader("📅 Monthly Projections")

    months = [1, 3, 6, 12]
    balances = [10068, 10206, 10411, 10822]
    returns = [0.68, 2.06, 4.11, 8.22]

    df = pd.DataFrame({
        'Month': months,
        'Balance': [f'${b:,}' for b in balances],
        'Return %': [f'{r:.2f}%' for r in returns]
    })

    st.dataframe(df, use_container_width=True, hide_index=True)

def show_configuration():
    """Show configuration page"""
    st.title("⚙️ Current Configuration")

    config = load_config()

    if not config:
        st.error("Configuration file not found!")
        return

    # Risk Management
    st.subheader("🛡️ Risk Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk per Trade", f"{config['risk']['risk_per_trade']}%")
        st.metric("Stop Loss", f"{config['risk']['stop_loss']['atr_multiplier']}x ATR")

    with col2:
        st.metric("Take Profit R:R", f"{config['risk']['take_profit']['risk_reward_ratio']}:1")
        st.metric("Daily DD Limit", f"{config['risk']['max_daily_loss']['percentage']}%")

    with col3:
        st.metric("Total DD Limit", f"{config['risk']['max_total_drawdown']['percentage']}%")
        st.metric("Max Positions", config['trading']['max_open_positions'])

    st.markdown("---")

    # Trading Pairs
    st.subheader("💱 Trading Pairs")

    pairs_data = []
    for symbol, settings in config['trading']['pairs'].items():
        if settings.get('enabled', False):
            pairs_data.append({
                'Symbol': symbol,
                'Timeframe': settings['timeframe'],
                'Status': '✅ Enabled'
            })

    if pairs_data:
        df = pd.DataFrame(pairs_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No pairs are currently enabled!")

    st.markdown("---")

    # Strategy Parameters (XAUUSD)
    if 'XAUUSD' in config['strategies']:
        st.subheader("📊 XAUUSD Strategy Parameters")

        xau_config = config['strategies']['XAUUSD']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            **Indicators:**
            - EMA Period: {xau_config['ema_period']}
            - RSI Period: {xau_config['rsi_period']}
            - ATR Period: {xau_config['atr_period']}
            - ADX Period: {xau_config['adx_period']}
            """)

        with col2:
            st.markdown(f"""
            **Entry Filters:**
            - RSI Buy: {xau_config['rsi_buy_threshold']}
            - RSI Sell: {xau_config['rsi_sell_threshold']}
            - Min ADX: {xau_config['min_adx']}
            - Min Confidence: {xau_config['min_confidence']}
            """)

        with col3:
            st.markdown(f"""
            **Position Sizing:**
            - Min ATR: {xau_config['min_atr']}
            - Min EMA Distance: {xau_config['min_ema_distance']}
            - SL ATR Multiplier: {xau_config['sl_atr_multiplier']}
            - Risk/Reward: {xau_config['risk_reward']}
            """)

def show_strategy_guide():
    """Show strategy guide page"""
    st.title("📖 Strategy Guide")

    summary = load_strategy_summary()

    if summary:
        st.markdown(summary)
    else:
        st.error("Strategy summary not found!")

    st.markdown("---")
    st.info("""
    **💡 Tip:** For detailed backtest results and analysis,
    navigate to the 'Backtest Results' page.
    """)

if __name__ == "__main__":
    main()
