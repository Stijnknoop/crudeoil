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
        print(f"FOUT: {log_path} niet gevonden. Run eerst daily_report.py.")
        return

    # 1. Bestanden inladen
    try:
        df_raw = download_latest_csv()
        df_raw['time'] = pd.to_datetime(df_raw['time'])
        logs = pd.read_csv(log_path)
    except Exception as e:
        print(f"Fout bij inladen data: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Logbestand geladen: {len(logs)} regels gevonden.")

    # 2. Individuele Dag-plots genereren
    # We lopen door de hele log heen om te kijken of er plots ontbreken
    for _, trade in logs.iterrows():
        # Sla over als er geen trade was (No Trade) of als de data nog Pending is
        if pd.isna(trade.get('entry_time')) or trade.get('exit_reason') in ["No Trade", "Data End (Pending)"]:
            continue

        entry_dt = pd.to_datetime(trade['entry_time'])
        file_date = entry_dt.strftime('%Y-%m-%d')
        plot_filename = os.path.join(output_dir, f"plot_{file_date}.png")

        # Forceer creatie als het bestand nog niet bestaat
        if not os.path.exists(plot_filename):
            print(f"Bezig met genereren van plot voor: {file_date}")
            day_data = df_raw[df_raw['time'].dt.date == entry_dt.date()].sort_values('time')
            
            if not day_data.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(day_data['time'], day_data['close_bid'], color='black', alpha=0.3, label='Koers (Bid)')
                
                # Teken Entry
                plt.scatter(pd.to_datetime(trade['entry_time']), trade['entry_p'], 
                            marker='^', color='blue', s=100, label='Entry', zorder=5)
                
                # Teken Exit (indien aanwezig)
                if not pd.isna(trade.get('exit_time')):
                    exit_color = 'green' if trade['return'] > 0 else 'red'
                    plt.scatter(pd.to_datetime(trade['exit_time']), trade['exit_p'], 
                                marker='x', color=exit_color, s=120, label=f'Exit ({trade["return"]:.2%})', zorder=5)
                
                plt.title(f"Trade Detail: {file_date} | Resultaat: {trade['return']:.4%} | Reden: {trade['exit_reason']}")
                plt.legend()
                plt.grid(True, alpha=0.15)
                plt.savefig(plot_filename)
                plt.close()

    # 3. Equity Curve genereren (Overzicht van alle trades)
    print("Equity Curve aan het bijwerken...")
    RISK_PER_TRADE = 0.02
    FIXED_SL = 0.004
    equity = [1.0]
    
    # Gebruik alleen afgeronde trades voor de curve
    finished_trades = logs[~logs['exit_reason'].isin(["No Trade", "Data End (Pending)"])]
    
    for r in finished_trades['return'].values:
        if not pd.isna(r):
            actual_gain = (r / FIXED_SL) * RISK_PER_TRADE
            equity.append(equity[-1] * (1 + actual_gain))

    plt.figure(figsize=(10, 5))
    plt.plot(equity, color='darkgreen', lw=2.5)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    
    plt.title(f"Institutional Grade Equity Curve\nLaatste update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    plt.grid(True, which='both', linestyle='-', alpha=0.1)
    plt.ylabel("Relatieve Waarde")
    plt.xlabel("Aantal Trades")
    
    plt.savefig(equity_path, dpi=150)
    plt.close()
    print(f"Equity curve succesvol opgeslagen: {equity_path}")

if __name__ == "__main__":
    generate_visuals()
