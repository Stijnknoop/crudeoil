import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
import re

# Zorg voor de juiste backend voor GitHub Actions
import matplotlib
matplotlib.use('Agg')

def download_latest_csv():
    # Gebruik dezelfde logica als daily_report om de juiste file te vinden
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
    # Zoek naar het eerste CSV bestand
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    
    if not csv_file:
        raise Exception("Geen CSV-bestand gevonden in de repository.")
        
    print(f"Data downloaden van: {csv_file['name']}...")
    return pd.read_csv(csv_file['download_url'])

def generate_visuals():
    output_dir = "Trading_details/plots"
    log_path = "Trading_details/trading_logs.csv"
    
    if not os.path.exists(log_path):
        print("Geen trading_logs.csv gevonden. Er valt niets te plotten.")
        return

    # Data en logs laden
    try:
        df_raw = download_latest_csv()
    except Exception as e:
        print(f"Fout bij downloaden data: {e}")
        return

    df_raw['time'] = pd.to_datetime(df_raw['time'])
    logs = pd.read_csv(log_path)

    os.makedirs(output_dir, exist_ok=True)

    for _, trade in logs.iterrows():
        # Sla regels zonder trade over
        if pd.isna(trade['entry_time']) or trade['exit_reason'] == "No Trade":
            continue

        # Datum bepalen voor de bestandsnaam
        entry_dt = pd.to_datetime(trade['entry_time'])
        file_date = entry_dt.strftime('%Y-%m-%d')
        plot_filename = f"{output_dir}/plot_{file_date}.png"

        # Check of plot al bestaat
        if os.path.exists(plot_filename):
            continue

        print(f"Grafiek maken voor datum: {file_date}")

        # Filter data voor die dag (marge van 1 uur voor/na de trade)
        day_data = df_raw[df_raw['time'].dt.date == entry_dt.date()].sort_values('time')

        if day_data.empty:
            continue

        plt.figure(figsize=(12, 6))
        plt.plot(day_data['time'], day_data['close_bid'], color='black', alpha=0.3, label='Price')
        
        # Entry
        plt.scatter(pd.to_datetime(trade['entry_time']), trade['entry_p'], 
                    marker='^', color='blue', s=100, label='Entry', zorder=5)
        
        # Exit
        if not pd.isna(trade['exit_time']):
            exit_color = 'green' if trade['return'] > 0 else 'red'
            plt.scatter(pd.to_datetime(trade['exit_time']), trade['exit_p'], 
                        marker='x', color=exit_color, s=120, label=f'Exit ({trade["exit_reason"]})', zorder=5)

        plt.title(f"Trade Report: {file_date} | Return: {trade['return']:.4%}")
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        plt.savefig(plot_filename)
        plt.close()

if __name__ == "__main__":
    generate_visuals()
