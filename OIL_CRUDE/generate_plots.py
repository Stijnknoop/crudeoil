import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests
import numpy as np

# ==============================================================================
# CONFIGURATIE
# ==============================================================================
START_CAPITAL = 10000.0  # Beginbedrag (bijv. 100 of 10000)
MAX_SLOTS = 10           # Aantal batches waarin je het geld opdeelt
LEVERAGE = 1             # Hefboom (zet op 1 voor spot, hoger voor futures)

# Paden
LOG_DIR = "OIL_CRUDE/Trading_details"
PLOT_DIR = os.path.join(LOG_DIR, "plots")
LOG_FILE = "trading_logs.csv"

# ==============================================================================
# 1. DATA OPHALEN
# ==============================================================================
def fetch_raw_data():
    user = "Stijnknoop"
    repo = "crudeoil"
    folder_path = "OIL_CRUDE"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder_path}?ref=master"
    
    try:
        res = requests.get(api_url, headers=headers)
        if res.status_code == 200:
            files = res.json()
            # Zoek CSV bestand
            csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
            if csv_file:
                df = pd.read_csv(csv_file['download_url'])
                df['time'] = pd.to_datetime(df['time'], format='ISO8601', errors='coerce')
                return df.sort_values('time')
    except Exception as e:
        print(f"Fout bij ophalen ruwe data: {e}")
    return None

# ==============================================================================
# 2. EQUITY CURVE BEREKENEN (EVENT DRIVEN)
# ==============================================================================
def calculate_compounding_equity(df_logs):
    """
    Simuleert een portfolio met 10 slots.
    Winst wordt direct herinvesteerd in de volgende trade (Compounding).
    """
    # Stap 1: Maak een tijdlijn van Events (Entries en Exits)
    events = []
    
    for idx, row in df_logs.iterrows():
        # ENTRY Event
        events.append({
            'time': row['entry_time'],
            'type': 'ENTRY',
            'trade_id': idx,
            'return': 0.0 # Nog niet bekend bij entry
        })
        
        # EXIT Event
        # Als er nog geen exit time is (open trade), gebruiken we 'nu' of slaan we over voor equity berekening
        if pd.notnull(row['exit_time']):
            events.append({
                'time': row['exit_time'],
                'type': 'EXIT',
                'trade_id': idx,
                'return': row['return']
            })
            
    # Sorteren op tijd. Belangrijk: Bij gelijke tijd eerst EXIT verwerken (cash vrijmaken), dan ENTRY.
    # We voegen een sort_key toe: ENTRY=1, EXIT=0
    df_events = pd.DataFrame(events)
    df_events['sort_order'] = df_events['type'].apply(lambda x: 1 if x == 'ENTRY' else 0)
    df_events = df_events.sort_values(by=['time', 'sort_order'])
    
    # Stap 2: De Simulatie Loop
    cash = START_CAPITAL
    open_positions = {} # Key: trade_id, Value: invested_amount
    equity_curve = []   # Lijst met (tijd, totale_waarde)
    
    # Startpunt
    equity_curve.append({'time': df_events['time'].min() - pd.Timedelta(minutes=1), 'equity': START_CAPITAL})

    for _, event in df_events.iterrows():
        current_time = event['time']
        
        # Huidige waarde van portfolio = Cash + Som van inleg open posities
        # (We negeren ongerealiseerde winst tussendoor voor de eenvoud, we updaten bij events)
        current_equity = cash + sum(open_positions.values())
        
        if event['type'] == 'ENTRY':
            # Hoeveel slots zijn er in totaal? 10.
            # Regel: We investeren 1/10e van de HUIDIGE equity per trade.
            # Dit zorgt voor het compounding effect.
            
            # Check: hebben we ruimte? (De generator script bewaakt dit al, maar voor de zekerheid)
            if len(open_positions) < MAX_SLOTS:
                position_size = current_equity / MAX_SLOTS
                
                # Als cash minder is dan position_size (door drawdown), gebruiken we wat er is
                if cash < position_size:
                    position_size = cash
                
                cash -= position_size
                open_positions[event['trade_id']] = position_size
                
        elif event['type'] == 'EXIT':
            if event['trade_id'] in open_positions:
                invested_amount = open_positions.pop(event['trade_id'])
                
                # Bereken resultaat: Inleg * (1 + (Return * Leverage))
                trade_return = event['return'] * LEVERAGE
                returned_amount = invested_amount * (1 + trade_return)
                
                cash += returned_amount

        # Update Equity Curve
        total_equity = cash + sum(open_positions.values())
        equity_curve.append({'time': current_time, 'equity': total_equity})

    return pd.DataFrame(equity_curve)

# ==============================================================================
# 3. GENERATE PLOTS
# ==============================================================================
def generate_performance_plots():
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    
    log_path = os.path.join(LOG_DIR, "trading_logs.csv")
    if not os.path.exists(log_path):
        print("Geen logbestand gevonden.")
        return

    # Data laden
    df_logs = pd.read_csv(log_path)
    df_logs['entry_time'] = pd.to_datetime(df_logs['entry_time'], format='ISO8601', errors='coerce')
    df_logs['exit_time'] = pd.to_datetime(df_logs['exit_time'], format='ISO8601', errors='coerce')

    # Filter valide trades
    df_trades = df_logs[
        (~df_logs['exit_reason'].isin(['No Trade', 'No Trade (Init)'])) & 
        (df_logs['entry_p'].notna())
    ].copy()

    if df_trades.empty:
        print("Geen trades om te plotten.")
        return

    # --- BEREKEN EQUITY CURVE (Compounding) ---
    df_equity = calculate_compounding_equity(df_trades)
    
    # --- BEREKEN BUY & HOLD (Benchmark) ---
    raw_data = fetch_raw_data()
    
    # Plotten
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 1. Buy & Hold
    if raw_data is not None and not df_equity.empty:
        start_date = df_equity['time'].min()
        mask = raw_data['time'] >= start_date
        df_bh = raw_data.loc[mask].copy()
        
        if not df_bh.empty:
            # Normaliseer Buy & Hold naar Start Capital
            first_price = df_bh['close_bid'].iloc[0]
            df_bh['normalized'] = (df_bh['close_bid'] / first_price) * START_CAPITAL
            ax1.plot(df_bh['time'], df_bh['normalized'], color='gray', linestyle='--', alpha=0.5, label='Buy & Hold')

    # 2. Strategy Equity
    ax1.plot(df_equity['time'], df_equity['equity'], color='#00ff00', linewidth=2, label=f'Strategy (Compounding {MAX_SLOTS} slots)')
    ax1.fill_between(df_equity['time'], START_CAPITAL, df_equity['equity'], color='#00ff00', alpha=0.1)
    
    # Opmaak
    final_equity = df_equity.iloc[-1]['equity']
    total_return = ((final_equity - START_CAPITAL) / START_CAPITAL) * 100
    
    ax1.set_title(f"Portfolio Performance (Compounding)\nStart: €{START_CAPITAL:.0f} | Eind: €{final_equity:.2f} | Return: {total_return:.2f}%")
    ax1.set_ylabel(f"Portfolio Waarde (€)")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
    ax1.legend(loc='upper left')
    
    # Nul-lijn (Break even)
    ax1.axhline(START_CAPITAL, color='black', linewidth=1)
    
    fig.autofmt_xdate()
    save_path = os.path.join(LOG_DIR, "equity_curve_compounding.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Grafiek opgeslagen: {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_performance_plots()
