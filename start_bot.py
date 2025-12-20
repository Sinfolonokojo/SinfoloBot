"""
SinfoloBot - Easy Launcher Script
Simple menu to run backtests, live trading, or dashboard
"""

import os
import sys
import subprocess

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print bot banner"""
    print("=" * 60)
    print("  SINFOLOBOT - Trading Bot Launcher")
    print("  XAUUSD Conservative Strategy (8.22% annual, -1.73% DD)")
    print("=" * 60)
    print()

def print_menu():
    """Print main menu"""
    print("Please select an option:")
    print()
    print("  [1] Run Backtest (Test strategy on historical data)")
    print("  [2] Start Live Trading Bot (Multi-Pair)")
    print("  [3] Start Prop Firm Bot (Compliant)")
    print("  [4] Open Dashboard (View analysis & results)")
    print("  [5] View Strategy Summary")
    print("  [0] Exit")
    print()

def run_backtest():
    """Run backtest"""
    clear_screen()
    print_banner()
    print("Running backtest on XAUUSD H1...")
    print()

    try:
        # Use venv python if available, otherwise system python
        python_cmd = "venv\\Scripts\\python.exe" if os.path.exists("venv\\Scripts\\python.exe") else "python"
        subprocess.run([python_cmd, "run_multi_pair_bt.py"])
    except Exception as e:
        print(f"Error running backtest: {e}")

    input("\nPress Enter to continue...")

def start_multi_pair_bot():
    """Start multi-pair live trading bot"""
    clear_screen()
    print_banner()
    print("⚠️  WARNING: This will start LIVE TRADING!")
    print()
    print("Make sure:")
    print("  - MetaTrader 5 is running and logged in")
    print("  - You have tested on demo account first")
    print("  - Your account settings are correct in config/multi_pair_config.yaml")
    print()

    confirm = input("Type 'START' to begin live trading: ")

    if confirm.upper() == 'START':
        print("\nStarting Multi-Pair Bot...")
        try:
            python_cmd = "venv\\Scripts\\python.exe" if os.path.exists("venv\\Scripts\\python.exe") else "python"
            subprocess.run([python_cmd, "multi_pair_bot.py"])
        except Exception as e:
            print(f"Error starting bot: {e}")
    else:
        print("Cancelled.")

    input("\nPress Enter to continue...")

def start_prop_firm_bot():
    """Start prop firm compliant bot"""
    clear_screen()
    print_banner()
    print("⚠️  WARNING: This will start LIVE TRADING (Prop Firm Mode)!")
    print()
    print("Make sure:")
    print("  - MetaTrader 5 is running and logged in to prop firm account")
    print("  - You have tested on demo account first")
    print("  - Your account settings are correct in config/prop_firm_config.yaml")
    print()

    confirm = input("Type 'START' to begin live trading: ")

    if confirm.upper() == 'START':
        print("\nStarting Prop Firm Bot...")
        try:
            python_cmd = "venv\\Scripts\\python.exe" if os.path.exists("venv\\Scripts\\python.exe") else "python"
            subprocess.run([python_cmd, "prop_firm_bot.py"])
        except Exception as e:
            print(f"Error starting bot: {e}")
    else:
        print("Cancelled.")

    input("\nPress Enter to continue...")

def open_dashboard():
    """Open the analysis dashboard"""
    clear_screen()
    print_banner()
    print("Opening Dashboard...")
    print()
    print("The dashboard will open in your web browser.")
    print("Press Ctrl+C in this window to stop the dashboard.")
    print()

    try:
        python_cmd = "venv\\Scripts\\python.exe" if os.path.exists("venv\\Scripts\\python.exe") else "python"
        subprocess.run([python_cmd, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error opening dashboard: {e}")
        print("\nMake sure streamlit is installed:")
        print("  pip install streamlit")

    input("\nPress Enter to continue...")

def view_summary():
    """View strategy summary"""
    clear_screen()
    print_banner()

    try:
        with open("STRATEGY_SUMMARY.md", "r") as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"Error reading summary: {e}")

    input("\nPress Enter to continue...")

def main():
    """Main menu loop"""
    while True:
        clear_screen()
        print_banner()
        print_menu()

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            run_backtest()
        elif choice == '2':
            start_multi_pair_bot()
        elif choice == '3':
            start_prop_firm_bot()
        elif choice == '4':
            open_dashboard()
        elif choice == '5':
            view_summary()
        elif choice == '0':
            clear_screen()
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\nBot launcher stopped. Goodbye!")
        sys.exit(0)
