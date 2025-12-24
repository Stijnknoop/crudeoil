import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from datetime import datetime, timedelta

# Zorg voor de juiste backend voor GitHub Actions
import matplotlib
matplotlib.use('Agg')

def download_latest_csv():
    user = "Stijnknoop"
    repo = "crudeoil"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref=master"
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")
        
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    
    if not csv_file:
        raise Exception("Geen CSV-bestand gevonden.")
        
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
    
    # --- STAP 1: DEFINIEER EN VERWIJDER OUDE PLOTS ---
    today_str = datetime.now().strftime('%Y-%m-%d')
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    force_update_dates = [today_str, yesterday_str]

    print(f"Schoonmaak: Verwijderen van plots voor {force_update_dates} indien aanwezig...")
    for date_str in force_update_dates:
        target_file = os.path.join(output_dir, f"plot_{date_str}.png")
        if os.path.exists(target_file):
            os.remove(target_file)
            print(f"Verwijderd: {target_file}")

    # --- STAP 2: DAGELIJKSE PLOTS GENEREREN ---
    for _, trade in logs.iterrows():
        if pd.isna(trade.get('entry_time')) or trade.get('exit_reason') == "No Trade":
            continue

        entry_dt = pd.to_datetime(trade['entry_time'])
        file_date = entry_dt.strftime('%Y-%m-%d')
        plot_filename = os.path.join(output_dir, f"plot_{file_date}.png")
        
        is_pending = trade.get('exit_reason') == "Data End (Pending)"
        
        # Alleen tekenen als het bestand niet bestaat (wat voor vandaag/gisteren nu zo is)
        if not os.path.exists(plot_filename):
            print(f"Nieuwe plot maken voor {file_date}...")
            day_data = df_raw[df_raw['time'].dt.date == entry_dt.date()].sort_values('time')
            
            if not day_data.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(day_data['time'], day_data['close_bid'], color='black', alpha=0.3, label='Koers')
                
                # Entry punt
                plt.scatter(pd.to_datetime(trade['entry_time']), trade['entry_p'], 
                            marker='^', color='blue', s=100, label='Entry', zorder=5)
                
                if is_pending:
                    # LIVE INDICATOR (Blauwe stip op laatste bekende prijs)
                    current_p = day_data['close_bid'].iloc[-1]
                    current_t = day_data['time'].iloc[-1]
                    plt.scatter(current_t, current_p, color='dodgerblue', s=200, 
                                edgecolors='white', linewidths=2, label=f'LIVE ({trade["return"]:.2%})', zorder=6)
                    plt.title(f"Trade Detail: {file_date} | STATUS: LIVE")
                else:
                    # GESLOTEN TRADE (Kruisje op exit prijs)
                    exit_color = 'green' if trade['return'] > 0 else 'red'
                    plt.scatter(pd.to_datetime(trade['exit_time']), trade['exit_p'], 
                                marker='x', color=exit_color, s=120, label=f'Exit ({trade["return"]:.2%})', zorder=5)
                    plt.title(f"Trade Detail: {file_date} | STATUS: GESLOTEN ({trade['exit_reason']})")
                
                plt.legend(loc='upper left')
                plt.grid(True, alpha=0.15)
                plt.savefig(plot_filename)
                plt.close()

    # --- STAP 3: EQUITY CURVE ---
    equity = [1.0]
    valid_trades = logs[logs['exit_reason'] != "No Trade"]
    for r in valid_trades['return'].values:
        if not pd.isna(r):
            # Formule: (Return / 0.4% SL) * 2% Risk
            equity.append(equity[-1] * (1 + (r / 0.004) * 0.02))

    plt.figure(figsize=(10, 5))
    plt.plot(equity, color='darkgreen', lw=2.5)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    plt.scatter(len(equity)-1, equity[-1], color='darkgreen', s=50)
    plt.annotate(f'Nu: {equity[-1]:.3f}', xy=(len(equity)-1, equity[-1]), 
                 xytext=(10, 0), textcoords='offset points', weight='bold')
    plt.title(f"Live Portfolio Performance\nUpdate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    plt.savefig(equity_path, dpi=150)
    plt.close()
    print("Visualisaties voltooid en opgeschoond.")

if __name__ == "__main__":
    generate_visuals()
