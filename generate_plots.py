import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests

# Functie om de ruwe koersdata op te halen voor de dagelijkse plots
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
    
    # Maak mappen aan als ze niet bestaan
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    if not os.path.exists(log_path):
        print("Geen trading_logs.csv gevonden.")
        return

    # Inladen van de logs
    df_logs = pd.read_csv(log_path)
    
    # Filter 'No Trade' en 'Pending' trades uit voor de equity curve
    df_trades = df_logs[~df_logs['exit_reason'].isin(['No Trade', 'Data End (Pending)'])].copy()
    
    if df_trades.empty:
        print("Nog geen voltooide trades om te plotten.")
        return

    # Datums converteren
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades = df_trades.sort_values('entry_time')

    # --- 1. OVERALL PERFORMANCE PLOT (Equity Curve) ---
    leverage = 5
    df_trades['leverage_return'] = df_trades['return'] * leverage
    df_trades['equity_compound'] = (1 + df_trades['leverage_return']).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_trades['entry_time'], df_trades['equity_compound'], color='darkgreen', linewidth=2, label=f'Equity Curve ({leverage}x leverage)')
    plt.fill_between(df_trades['entry_time'], 1, df_trades['equity_compound'], color='green', alpha=0.1)
    
    # Opmaak
    plt.title(f"Oil Trading Bot: Portfolio Growth\nLaatste update: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    plt.ylabel("Vermogensfactor (1.0 = Start)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.legend()
    
    plt.savefig(os.path.join(log_dir, "equity_curve.png"))
    plt.close()
    print("Equity curve gegenereerd.")

    # --- 2. INDIVIDUELE DAG-PLOTS ---
    raw_data = fetch_raw_data()
    if raw_data is not None:
        raw_data['time'] = pd.to_datetime(raw_data['time'])
        
        for _, trade in df_trades.iterrows():
            trade_date = trade['entry_time'].date()
            file_name = f"{trade_date}.png"
            file_path = os.path.join(plot_dir, file_name)
            
            # Sla over als de plot al bestaat (bespaart tijd)
            if os.path.exists(file_path):
                continue
                
            # Filter koersdata voor die dag
            day_data = raw_data[raw_data['time'].dt.date == trade_date].copy()
            
            if not day_data.empty:
                plt.figure(figsize=(10, 5))
                plt.plot(day_data['time'], day_data['close_bid'], color='blue', alpha=0.4, label='Prijs (Bid)')
                
                # Markeer Entry
                plt.scatter(trade['entry_time'], trade['entry_p'], color='green', marker='^', s=100, label=f"Entry ({trade['side']})", zorder=5)
                
                # Markeer Exit
                if not pd.isna(trade['exit_time']):
                    exit_t = pd.to_datetime(trade['exit_time'])
                    plt.scatter(exit_t, trade['exit_p'], color='red', marker='v', s=100, label=f"Exit ({trade['exit_reason']})", zorder=5)
                
                plt.title(f"Trade Detail: {trade_date} | Resultaat: {trade['return']:.2%}")
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.grid(True, alpha=0.2)
                plt.legend()
                
                plt.savefig(file_path)
                plt.close()
                print(f"Nieuwe dagplot gemaakt: {file_name}")

if __name__ == "__main__":
    generate_performance_plots()
