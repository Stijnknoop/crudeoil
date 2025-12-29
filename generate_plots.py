import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests
import datetime

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
    
    # Zorg dat de tijdstippen echte datetime objecten zijn
    df_logs['entry_time'] = pd.to_datetime(df_logs['entry_time'])
    
    # Filter trades voor de equity curve (alleen gesloten trades)
    df_trades_finished = df_logs[~df_logs['exit_reason'].isin(['No Trade', 'Data End (Pending)', 'No Trade (Init)'])].copy()
    df_trades_finished = df_trades_finished.sort_values('entry_time')

    raw_data = fetch_raw_data()
    if raw_data is not None:
        raw_data['time'] = pd.to_datetime(raw_data['time'])
        raw_data = raw_data.sort_values('time')

    # --- 1. EQUITY CURVE ---
    if not df_trades_finished.empty:
        leverage = 5
        returns = df_trades_finished['return'].values * leverage
        equity_curve = [1.0]
        for r in returns:
            equity_curve.append(equity_curve[-1] * (1 + r))
        
        start_time = df_trades_finished['entry_time'].min() - pd.Timedelta(hours=1)
        plot_times = [start_time] + df_trades_finished['entry_time'].tolist()

        fig, ax1 = plt.subplots(figsize=(12, 7))
        if raw_data is not None:
            mask = raw_data['time'] >= start_time
            df_bh = raw_data.loc[mask].copy()
            if not df_bh.empty:
                first_price = df_bh['close_bid'].iloc[0]
                df_bh['buy_hold_factor'] = df_bh['close_bid'] / first_price
                ax1.plot(df_bh['time'], df_bh['buy_hold_factor'], color='royalblue', linestyle='--', alpha=0.7, label='Buy & Hold (1x)')

        ax1.plot(plot_times, equity_curve, color='darkgreen', linewidth=2.5, label=f'Bot Strategy ({leverage}x)')
        ax1.fill_between(plot_times, 1, equity_curve, color='green', alpha=0.1)
        ax1.set_title(f"Performance Overzicht\nUpdate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        ax1.set_ylabel("Factor (Start = 1.0)")
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        ax1.legend(loc='upper left')
        fig.autofmt_xdate()
        plt.savefig(os.path.join(log_dir, "equity_curve.png"), bbox_inches='tight', dpi=150)
        plt.close()

    # --- 2. INDIVIDUELE DAG-PLOTS ---
    if raw_data is not None:
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        yesterday_str = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        refresh_dates = [today_str, yesterday_str]

        all_trading_days = raw_data['time'].dt.date.unique()

        for trade_date_dt in all_trading_days:
            trade_date_str = trade_date_dt.strftime('%Y-%m-%d')
            file_name = f"{trade_date_str}.png"
            file_path = os.path.join(plot_dir, file_name)
            
            if os.path.exists(file_path) and (trade_date_str not in refresh_dates):
                continue
            
            day_data = raw_data[raw_data['time'].dt.date == trade_date_dt].copy()
            if day_data.empty: continue

            plt.figure(figsize=(10, 5))
            plt.plot(day_data['time'], day_data['close_bid'], color='blue', alpha=0.4, label='Prijs')
            
            # Zoek de log entry voor deze dag
            day_log = df_logs[df_logs['entry_time'].dt.date == trade_date_dt]
            
            title_suffix = "No Trade"
            if not day_log.empty:
                row = day_log.iloc[0]
                
                # Check of er een Entry is (ongeacht of hij pending is)
                # We kijken of entry_p een geldig getal is
                if not pd.isna(row.get('entry_p')):
                    plt.scatter(row['entry_time'], row['entry_p'], color='green', marker='^', s=150, label="Entry", zorder=5)
                    
                    if row['exit_reason'] == "Data End (Pending)":
                        title_suffix = "LIVE / PENDING"
                    else:
                        title_suffix = f"Return: {row['return']:.2%}"
                        # Teken Exit alleen als die er is
                        if not pd.isna(row.get('exit_time')):
                            exit_t = pd.to_datetime(row['exit_time'])
                            plt.scatter(exit_t, row['exit_p'], color='red', marker='v', s=150, label="Exit", zorder=5)

            plt.title(f"Handelsdag: {trade_date_str} | {title_suffix}")
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.grid(True, alpha=0.2)
            plt.legend()
            
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            print(f"Plot aangemaakt/ververst: {file_name}")

if __name__ == "__main__":
    generate_performance_plots()
