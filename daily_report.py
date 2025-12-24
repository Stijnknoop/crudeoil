import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

def download_latest_csv():
    user = "Stijnknoop"
    repo = "crudeoil"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref=master"
    response = requests.get(api_url, headers=headers)
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

    # 1. Dagplots (Forceer update voor Pending)
    for _, trade in logs.iterrows():
        if pd.isna(trade.get('entry_time')) or trade.get('exit_reason') == "No Trade":
            continue

        entry_dt = pd.to_datetime(trade['entry_time'])
        file_date = entry_dt.strftime('%Y-%m-%d')
        plot_filename = os.path.join(output_dir, f"plot_{file_date}.png")
        is_pending = trade['exit_reason'] == "Data End (Pending)"

        if not os.path.exists(plot_filename) or is_pending:
            day_data = df_raw[df_raw['time'].dt.date == entry_dt.date()].sort_values('time')
            if not day_data.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(day_data['time'], day_data['close_bid'], color='black', alpha=0.3)
                plt.scatter(pd.to_datetime(trade['entry_time']), trade['entry_p'], marker='^', color='blue', s=100)
                
                # Exit punt (Gebruik laatste koers als het pending is)
                ex_p = trade['exit_p'] if not is_pending else day_data['close_bid'].iloc[-1]
                ex_t = trade['exit_time'] if not is_pending else day_data['time'].iloc[-1]
                
                plt.scatter(pd.to_datetime(ex_t), ex_p, marker='x', color='green' if trade['return'] > 0 else 'red', s=120)
                plt.title(f"Day: {file_date} | Return: {trade['return']:.4%} | {'LIVE' if is_pending else 'CLOSED'}")
                plt.savefig(plot_filename); plt.close()

    # 2. Equity Curve (Inclusief de laatste pending trade)
    equity = [1.0]
    valid_trades = logs[logs['exit_reason'] != "No Trade"]
    for r in valid_trades['return'].values:
        equity.append(equity[-1] * (1 + (r / 0.004 * 0.02)))

    plt.figure(figsize=(10, 5))
    plt.plot(equity, color='darkgreen', lw=2.5)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    plt.title(f"Institutional Equity Curve (Live) | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    plt.ylabel("Value"); plt.xlabel("Trades")
    plt.savefig(equity_path, dpi=150); plt.close()

if __name__ == "__main__":
    generate_visuals()
