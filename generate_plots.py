import pandas as pd
import matplotlib.pyplot as plt
import os
import requests

# Zorg voor de juiste backend
import matplotlib
matplotlib.use('Agg')

def download_data():
    # We halen de ruwe data opnieuw op om de koersen te plotten
    url = "https://raw.githubusercontent.com/Stijnknoop/crudeoil/master/crude_oil_data.csv" # Pas aan naar jouw juiste CSV url
    return pd.read_csv(url)

def generate_visuals():
    output_dir = "Trading_details/plots"
    log_path = "Trading_details/trading_logs.csv"
    
    if not os.path.exists(log_path):
        print("Geen logs gevonden om te plotten.")
        return

    # Data en logs laden
    df_raw = download_data()
    df_raw['time'] = pd.to_datetime(df_raw['time'])
    logs = pd.read_csv(log_path)

    os.makedirs(output_dir, exist_ok=True)

    for _, trade in logs.iterrows():
        # Alleen plotten als er een entry was
        if pd.isna(trade['entry_time']):
            continue

        # Datum bepalen voor de bestandsnaam
        entry_dt = pd.to_datetime(trade['entry_time'])
        file_date = entry_dt.strftime('%Y-%m-%d')
        plot_filename = f"{output_dir}/plot_{file_date}.png"

        # Check of plot al bestaat (voorkom dubbel werk)
        if os.path.exists(plot_filename):
            continue

        print(f"Grafiek genereren voor {file_date}...")

        # Selecteer data van die specifieke dag (24u venster rond entry)
        mask = (df_raw['time'].dt.date == entry_dt.date())
        df_day = df_raw[mask].sort_values('time')

        if df_day.empty:
            continue

        plt.figure(figsize=(12, 6))
        plt.plot(df_day['time'], df_day['close_bid'], color='gray', alpha=0.4, label='Price (Bid)')
        
        # Entry markeren
        plt.scatter(pd.to_datetime(trade['entry_time']), trade['entry_p'], 
                    marker='^', color='blue', s=100, label='ENTRY', zorder=5)
        
        # Exit markeren
        if not pd.isna(trade['exit_time']):
            c = 'green' if trade['return'] > 0 else 'red'
            plt.scatter(pd.to_datetime(trade['exit_time']), trade['exit_p'], 
                        marker='x', color=c, s=120, label=f'EXIT ({trade["exit_reason"]})', zorder=5)

        plt.title(f"Crude Oil Trade: {file_date} | Return: {trade['return']:.4%}")
        plt.xlabel("Tijd")
        plt.ylabel("Prijs")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(plot_filename)
        plt.close()

if __name__ == "__main__":
    generate_visuals()
