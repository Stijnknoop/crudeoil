import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests

# Functie om de ruwe koersdata op te halen (nodig voor dag-plots)
def fetch_raw_data():
    user, repo = "Stijnknoop", "crudeoil"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref=master"
    res = requests.get(api_url, headers=headers)
    if res.status_code == 200:
        csv_file = next((f for f in res.json() if f['name'].endswith('.csv')), None)
        if csv_file: return pd.read_csv(csv_file['download_url'])
    return None

def generate_performance_plot():
    log_dir = "Trading_details"
    plot_dir = os.path.join(log_dir, "plots")
    log_path = os.path.join(log_dir, "trading_logs.csv")
    
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(log_path): return

    df_logs = pd.read_csv(log_path)
    df_trades = df_logs[df_logs['exit_reason'] != 'No Trade'].copy()
    if df_trades.empty: return

    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades = df_trades.sort_values('entry_time')

    # --- 1. OVERALL PERFORMANCE PLOT (Equity Curve) ---
    leverage = 5
    df_trades['leverage_return'] = df_trades['return'] * leverage
    df_trades['equity_compound'] = (1 + df_trades['leverage_return']).cumprod()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_ylabel('Portfolio Waarde', color='darkgreen', fontweight='bold')
    ax1.plot(df_trades['entry_time'], df_trades['equity_compound'], color='darkgreen', linewidth=2.5, label='Compound Equity (5x)')
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    ax2 = ax1.twinx()
    colors = ['#a1d99b' if r > 0 else '#fb9a99' for r in df_trades['leverage_return']]
    ax2.bar(df_trades['entry_time'], df_trades['leverage_return'] * 100, color=colors, alpha=0.6, width=0.6)
    ax2.axhline(0, color='black', linewidth=1, alpha=0.5)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.title("Portfolio Performance & Daily Returns")
    plt.savefig(os.path.join(log_dir, "equity_curve.png"))
    plt.close()

    # --- 2. INDIVIDUELE DAG-PLOTS ---
    raw_data = fetch_raw_data()
    if raw_data is not None:
        raw_data['time'] = pd.to_datetime(raw_data['time'])
        
        for _, trade in df_trades.iterrows():
            trade_date = trade['entry_time'].date()
            # Filter koersdata voor die specifieke dag
            day_data = raw_data[raw_data['time'].dt.date == trade_date].copy()
            
            if not day_data.empty:
                plt.figure(figsize=(10, 5))
                plt.plot(day_data['time'], day_data['close_bid'], color='blue', alpha=0.5, label='Price')
                
                # Markeer Entry en Exit
                plt.scatter(trade['entry_time'], trade['entry_p'], color='green', marker='^', s=100, label='Entry', zorder=5)
                if not pd.isna(trade['exit_time']):
                    exit_t = pd.to_datetime(trade['exit_time'])
                    plt.scatter(exit_t, trade['exit_p'], color='red', marker='v', s=100, label='Exit', zorder=5)
                
                plt.title(f"Trade Detail: {trade_date} ({trade['side']}) | Return: {trade['return']:.2%}")
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Opslaan als Trading_details/plots/2025-12-24.png
                file_name = f"{trade_date}.png"
                plt.savefig(os.path.join(plot_dir, file_name))
                plt.close()
                print(f"Gegenereerd: {file_name}")

if __name__ == "__main__":
    generate_performance_plot()
