import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests

# Functie om de ruwe koersdata op te halen
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
        print("Geen trading_logs.csv gevonden.")
        return

    # Logs inladen
    df_logs = pd.read_csv(log_path)
    df_trades = df_logs[~df_logs['exit_reason'].isin(['No Trade', 'Data End (Pending)'])].copy()
    
    if df_trades.empty:
        print("Nog geen trades om te plotten.")
        return

    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades = df_trades.sort_values('entry_time')

    # Haal koersdata één keer op voor alle plots
    raw_data = fetch_raw_data()
    if raw_data is not None:
        raw_data['time'] = pd.to_datetime(raw_data['time'])

    # --- 1. EQUITY CURVE PLOT (Portfolio vs Olieprijs) ---
    leverage = 5
    returns = df_trades['return'].values * leverage
    equity_curve = [1.0]
    for r in returns:
        equity_curve.append(equity_curve[-1] * (1 + r))
    
    start_time = df_trades['entry_time'].min() - pd.Timedelta(hours=1)
    plot_times = [start_time] + df_trades['entry_time'].tolist()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if raw_data is not None:
        mask = raw_data['time'] >= start_time
        ax2 = ax1.twinx()
        ax2.plot(raw_data.loc[mask, 'time'], raw_data.loc[mask, 'close_bid'], 
                 color='gray', alpha=0.2, label='Olieprijs (Achtergrond)')
        ax2.set_ylabel('Olieprijs ($)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    ax1.plot(plot_times, equity_curve, color='darkgreen', linewidth=2, label=f'Portfolio Groei ({leverage}x)')
    ax1.fill_between(plot_times, 1, equity_curve, color='green', alpha=0.1)
    
    ax1.set_title(f"Oil Trading Bot: Performance Overzicht\nUpdate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    ax1.set_ylabel("Vermogensfactor (Start = 1.0)")
    ax1.set_xlim(left=start_time)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate()

    lines, labels = ax1.get_legend_handles_labels()
    if raw_data is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    plt.savefig(os.path.join(log_dir, "equity_curve.png"), bbox_inches='tight')
    plt.close()
    print("Equity curve gegenereerd.")

    # --- 2. INDIVIDUELE DAG-PLOTS ---
    if raw_data is not None:
        for _, trade in df_trades.iterrows():
            trade_date = trade['entry_time'].date()
            file_name = f"{trade_date}.png"
            file_path = os.path.join(plot_dir, file_name)
            
            # Sla over als de plot al bestaat
            if os.path.exists(file_path):
                continue
                
            day_data = raw_data[raw_data['time'].dt.date == trade_date].copy()
            
            if not day_data.empty:
                plt.figure(figsize=(10, 5))
                plt.plot(day_data['time'], day_data['close_bid'], color='blue', alpha=0.5, label='Olieprijs')
                
                # Markeer Entry
                plt.scatter(trade['entry_time'], trade['entry_p'], color='green', marker='^', s=100, label=f"Entry ({trade['side']})", zorder=5)
                
                # Markeer Exit
                if not pd.isna(trade['exit_time']):
                    exit_t = pd.to_datetime(trade['exit_time'])
                    plt.scatter(exit_t, trade['exit_p'], color='red', marker='v', s=100, label=f"Exit ({trade['exit_reason']})", zorder=5)
                
                plt.title(f"Trade Analyse: {trade_date} | Return: {trade['return']:.2%}")
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.grid(True, alpha=0.2)
                plt.legend()
                
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                print(f"Dagplot opgeslagen: {file_name}")

if __name__ == "__main__":
    generate_performance_plots()
