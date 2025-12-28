import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests
import datetime
import numpy as np

# 1. DATA OPHALEN
def fetch_raw_data():
    user, repo = "Stijnknoop", "crudeoil"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref=master"
    try:
        res = requests.get(api_url, headers=headers)
        if res.status_code == 200:
            csv_file = next((f for f in res.json() if f['name'].endswith('.csv')), None)
            if csv_file:
                return pd.read_csv(csv_file['download_url'])
    except Exception as e:
        print(f"Fout bij ophalen koersdata: {e}")
    return None

def generate_performance_plots():
    log_dir = "Trading_details"
    plot_dir = os.path.join(log_dir, "plots")
    log_path = os.path.join(log_dir, "trading_logs.csv")
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    if not os.path.exists(log_path):
        return

    df_logs = pd.read_csv(log_path)
    df_trades = df_logs[~df_logs['exit_reason'].isin(['No Trade', 'Data End (Pending)'])].copy()
    
    if df_trades.empty:
        return

    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades = df_trades.sort_values('entry_time')

    raw_data = fetch_raw_data()
    if raw_data is not None:
        raw_data['time'] = pd.to_datetime(raw_data['time'])
        raw_data = raw_data.sort_values('time')
        # Groepeer data per dag voor de historie-plots
        raw_data['date'] = raw_data['time'].dt.date
        all_dates = sorted(raw_data['date'].unique())

    # --- 1. EQUITY CURVE (Blijft ongewijzigd) ---
    leverage = 5
    returns = df_trades['return'].values * leverage
    equity_curve = [1.0]
    for r in returns: equity_curve.append(equity_curve[-1] * (1 + r))
    
    start_time = df_trades['entry_time'].min() - pd.Timedelta(hours=1)
    plot_times = [start_time] + df_trades['entry_time'].tolist()

    fig_eq, ax_eq = plt.subplots(figsize=(12, 6))
    ax_eq.plot(plot_times, equity_curve, color='darkgreen', linewidth=2, label=f'Bot ({leverage}x)')
    ax_eq.set_title("Cumulatief Rendement")
    ax_eq.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, "equity_curve.png"), bbox_inches='tight')
    plt.close()

    # --- 2. INDIVIDUELE DAG-PLOTS (3 SUBPLOTS) ---
    if raw_data is not None:
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        yesterday_str = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        refresh_dates = [today_str, yesterday_str]

        for _, trade in df_trades.iterrows():
            trade_date_dt = trade['entry_time'].date()
            trade_date_str = trade_date_dt.strftime('%Y-%m-%d')
            file_path = os.path.join(plot_dir, f"{trade_date_str}.png")

            if os.path.exists(file_path) and (trade_date_str not in refresh_dates):
                continue
            
            # Reconstructie van Training/Validation sets (identiek aan analyse script)
            try:
                current_idx = all_dates.index(trade_date_dt)
                history_dates = all_dates[max(0, current_idx-40):current_idx]
                split = int(len(history_dates) * 0.75)
                train_dates = history_dates[:split]
                val_dates = history_dates[split:]
            except ValueError:
                train_dates, val_dates = [], []

            # Maak de 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

            # Subplot 1: Training Overlay (Groen)
            for d in train_dates:
                d_pts = raw_data[raw_data['date'] == d]['close_bid'].values
                ax1.plot(d_pts, color='green', alpha=0.1)
            ax1.set_title(f"1. Training ({len(train_dates)}d)")
            ax1.axis('off')

            # Subplot 2: Validatie Overlay (Oranje)
            for d in val_dates:
                d_pts = raw_data[raw_data['date'] == d]['close_bid'].values
                ax2.plot(d_pts, color='orange', alpha=0.2)
            ax2.set_title(f"2. Validatie ({len(val_dates)}d)")
            ax2.axis('off')

            # Subplot 3: Live Dag + Trade (Blauw)
            day_data = raw_data[raw_data['date'] == trade_date_dt].copy()
            if not day_data.empty:
                ax3.plot(day_data['time'], day_data['close_bid'], color='blue', alpha=0.5)
                ax3.scatter(trade['entry_time'], trade['entry_p'], color='green', marker='^', s=100, zorder=5)
                if not pd.isna(trade['exit_time']):
                    ax3.scatter(pd.to_datetime(trade['exit_time']), trade['exit_p'], color='red', marker='v', s=100, zorder=5)
                
                ax3.set_title(f"3. Trade: {trade['return']:.2%}")
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax3.get_xticklabels(), rotation=30)
            
            plt.suptitle(f"Analyse: {trade_date_str}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            print(f"3-fase plot gegenereerd: {trade_date_str}.png")

if __name__ == "__main__":
    generate_performance_plots()
