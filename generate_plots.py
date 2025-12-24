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
    log_path = os.path.join(log_dir, "trading_logs.csv")
    
    if not os.path.exists(log_path):
        print("Geen trading_logs.csv gevonden.")
        return

    df_logs = pd.read_csv(log_path)
    df_trades = df_logs[~df_logs['exit_reason'].isin(['No Trade', 'Data End (Pending)'])].copy()
    
    if df_trades.empty:
        print("Nog geen trades om te plotten.")
        return

    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades = df_trades.sort_values('entry_time')

    # --- BEREKENING EQUITY MET STARTPUNT 1.0 ---
    leverage = 5
    returns = df_trades['return'].values * leverage
    equity_curve = [1.0]  # Startwaarde
    for r in returns:
        equity_curve.append(equity_curve[-1] * (1 + r))
    
    # Maak een tijdlijn die begint bij de eerste entry, maar een fractie eerder
    start_time = df_trades['entry_time'].min() - pd.Timedelta(hours=1)
    plot_times = [start_time] + df_trades['entry_time'].tolist()

    # --- PLOTTEN ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 1. Koers weergeven op de achtergrond (rechter y-as)
    raw_data = fetch_raw_data()
    if raw_data is not None:
        raw_data['time'] = pd.to_datetime(raw_data['time'])
        # Filter koersdata vanaf het begin van de trades
        mask = raw_data['time'] >= start_time
        ax2 = ax1.twinx()
        ax2.plot(raw_data.loc[mask, 'time'], raw_data.loc[mask, 'close_bid'], 
                 color='gray', alpha=0.3, label='Olieprijs (Bid)')
        ax2.set_ylabel('Olieprijs ($)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    # 2. Equity Curve (linker y-as)
    ax1.plot(plot_times, equity_curve, color='darkgreen', linewidth=2, 
             label=f'Portfolio Groei ({leverage}x leverage)')
    ax1.fill_between(plot_times, 1, equity_curve, color='green', alpha=0.1)

    # Opmaak
    ax1.set_title(f"Oil Trading Bot: Portfolio vs Olieprijs\nUpdate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    ax1.set_ylabel("Vermogensfactor (Start = 1.0)")
    ax1.set_ylim(min(equity_curve) - 0.02, max(equity_curve) + 0.02)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Zorg dat de x-as netjes start aan het begin
    ax1.set_xlim(left=start_time)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig.autofmt_xdate()
    
    # Legenda's combineren
    lines, labels = ax1.get_legend_handles_labels()
    if raw_data is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    plt.savefig(os.path.join(log_dir, "equity_curve.png"), bbox_inches='tight')
    plt.close()
    print("Equity curve met koersachtergrond gegenereerd.")

if __name__ == "__main__":
    generate_performance_plots()
