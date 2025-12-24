import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from datetime import datetime

# Zorg voor de juiste backend voor GitHub Actions
import matplotlib
matplotlib.use('Agg')

def download_latest_csv():
    user = "Stijnknoop"
    repo = "crudeoil"
    branch = "master"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref={branch}"
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")
        
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    
    if not csv_file:
        raise Exception("Geen CSV-bestand gevonden in de repository.")
        
    print(f"Koers data ophalen voor visualisatie: {csv_file['name']}...")
    return pd.read_csv(csv_file['download_url'])

def generate_visuals():
    output_dir = "Trading_details/plots"
    log_path = "Trading_details/trading_logs.csv"
    equity_path = "Trading_details/equity_history.png"
    
    if not os.path.exists(log_path):
        print(f"FOUT: {log_path} niet gevonden.")
        return

    try:
        df_raw = download_latest_csv()
        df_raw['time'] = pd.to_datetime(df_raw['time'])
        logs = pd.read_csv(log_path)
    except Exception as e:
        print(f"Fout bij inladen data: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. DAGELIJKS PLOTS (INCLUSIEF PENDING) ---
    for _, trade in logs.iterrows():
        # Sla alleen over als er echt geen trade was
        if pd.isna(trade.get('entry_time')) or trade.get('exit_reason') == "No Trade":
            continue

        entry_dt = pd.to_datetime(trade['entry_time'])
        file_date = entry_dt.strftime('%Y-%m-%d')
        plot_filename = os.path.join(output_dir, f"plot_{file_date}.png")

        # We overschrijven de plot ALTIJD als hij "Pending" is, zodat hij ververst
        is_pending = trade.get('exit_reason') == "Data End (Pending)"
        
        if not os.path.exists(plot_filename) or is_pending:
            day_data = df_raw[df_raw['time'].dt.date == entry_dt.date()].sort_values('time')
            
            if not day_data.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(day_data['time'], day_data['close_bid'], color='black', alpha=0.3, label='Koers')
                
                # Entry
                plt.scatter(pd.to_datetime(trade['entry_time']), trade['entry_p'], 
                            marker='^', color='blue', s=100, label='Entry', zorder=5)
                
                # Exit (gebruik laatste data punt als hij pending is)
                exit_time = trade.get('exit_time')
                exit_p = trade.get('exit_p')
                
                # Als er geen exit_p is (omdat het script nog loopt), pakken we de laatste prijs van de dag
                if pd.isna(exit_p) or is_pending:
                    exit_time = day_data['time'].iloc[-1]
                    exit_p = day_data['close_bid'].iloc[-1]
                    status_label = f"LOPENDE TRADE (Nu: {trade['return']:.2%})"
                else:
                    status_label = f"EXIT ({trade['return']:.2%})"

                exit_color = 'green' if trade['return'] > 0 else 'red'
                plt.scatter(pd.to_datetime(exit_time), exit_p, 
                            marker='x', color=exit_color, s=120, label=status_label, zorder=5)
                
                plt.title(f"Trade Detail: {file_date} | Status: {trade['exit_reason']}")
                plt.legend()
                plt.grid(True, alpha=0.15)
                plt.savefig(plot_filename)
                plt.close()

    # --- 2. EQUITY CURVE (INCLUSIEF PENDING) ---
    print("Equity Curve aan het bijwerken inclusief lopende trades...")
    RISK_PER_TRADE = 0.02
    FIXED_SL = 0.004
    equity = [1.0]
    
    # We nemen hier ALLE regels behalve "No Trade"
    trades_for_equity = logs[logs['exit_reason'] != "No Trade"]
    
    for r in trades_for_equity['return'].values:
        if not pd.isna(r):
            actual_gain = (r / FIXED_SL) * RISK_PER_TRADE
            equity.append(equity[-1] * (1 + actual_gain))

    plt.figure(figsize=(10, 5))
    plt.plot(equity, color='darkgreen', lw=2.5)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    
    # Laatste waarde markeren als tekst
    current_val = equity[-1]
    plt.annotate(f'{current_val:.2f}', xy=(len(equity)-1, current_val), xytext=(5, 5),
                 textcoords='offset points', color='darkgreen', weight='bold')
    
    plt.title(f"Live Equity Curve (Incl. Pending)\nLaatste update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    plt.grid(True, which='both', linestyle='-', alpha=0.1)
    plt.ylabel("Relatieve Waarde")
    plt.xlabel("Aantal Trades")
    
    plt.savefig(equity_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    generate_visuals()
