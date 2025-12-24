import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')

def download_latest_csv():
    user = "Stijnknoop"
    repo = "crudeoil"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref=master"
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200: raise Exception(f"API Error: {response.status_code}")
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    return pd.read_csv(csv_file['download_url'])

def generate_visuals():
    output_dir = "Trading_details/plots"
    log_path = "Trading_details/trading_logs.csv"
    equity_path = "Trading_details/equity_history.png"
    
    if not os.path.exists(log_path): return

    df_raw = download_latest_csv()
    df_raw['time'] = pd.to_datetime(df_raw['time'])
    logs = pd.read_csv(log_path)
    os.makedirs(output_dir, exist_ok=True)
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # --- 1. SCHOONMAAK EN PLOTS ---
    for date_str in [today_str, yesterday_str]:
        target = os.path.join(output_dir, f"plot_{date_str}.png")
        if os.path.exists(target): os.remove(target)

    for _, trade in logs.iterrows():
        if pd.isna(trade.get('entry_time')) or trade.get('exit_reason') == "No Trade": continue
        
        entry_dt = pd.to_datetime(trade['entry_time'])
        file_date = entry_dt.strftime('%Y-%m-%d')
        plot_filename = os.path.join(output_dir, f"plot_{file_date}.png")
        is_pending = trade.get('exit_reason') == "Data End (Pending)"

        if not os.path.exists(plot_filename):
            day_data = df_raw[df_raw['time'].dt.date == entry_dt.date()].sort_values('time')
            if not day_data.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(day_data['time'], day_data['close_bid'], color='black', alpha=0.3, label='Koers')
                plt.scatter(pd.to_datetime(trade['entry_time']), trade['entry_p'], marker='^', color='blue', s=100, label='Entry')
                
                if is_pending:
                    plt.scatter(day_data['time'].iloc[-1], day_data['close_bid'].iloc[-1], color='dodgerblue', s=200, edgecolors='white', label=f'LIVE ({trade["return"]:.2%})')
                else:
                    color = 'green' if trade['return'] > 0 else 'red'
                    plt.scatter(pd.to_datetime(trade['exit_time']), trade['exit_p'], marker='x', color=color, s=120, label=f'Exit ({trade["return"]:.2%})')
                
                plt.legend()
                plt.grid(True, alpha=0.15)
                plt.savefig(plot_filename)
                plt.close()

    # --- 2. SYNC EQUITY CURVE (1:1 GEEN HEFBOOM) ---
    equity = [1.0]
    valid_trades = logs[logs['exit_reason'] != "No Trade"].copy()
    valid_trades['entry_time'] = pd.to_datetime(valid_trades['entry_time'])
    valid_trades = valid_trades.sort_values('entry_time')

    for r in valid_trades['return'].values:
        if not pd.isna(r):
            equity.append(equity[-1] * (1 + r))

    plt.figure(figsize=(10, 5))
    plt.plot(equity, color='darkgreen', lw=2.5)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    plt.scatter(len(equity)-1, equity[-1], color='darkgreen', s=50)
    plt.annotate(f'Nu: {equity[-1]:.4f}', xy=(len(equity)-1, equity[-1]), xytext=(10, 0), textcoords='offset points', weight='bold')
    plt.title(f"Portfolio Performance (1:1 Sync)\nUpdate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    plt.savefig(equity_path, dpi=150)
    plt.close()

if __name__ == "__main__": generate_visuals()
