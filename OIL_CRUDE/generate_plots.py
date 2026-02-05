import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import os
import requests
import numpy as np

# ==============================================================================
# 1. CONFIGURATIE
# ==============================================================================
GITHUB_USER = "Stijnknoop"
GITHUB_REPO = "crudeoil"
FOLDER_PATH = "OIL_CRUDE"
OUTPUT_DIR = "OIL_CRUDE/Trading_details"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_FILE = "trading_logs.csv"

# Account Settings
START_CAPITAL = 10000.0
MAX_SLOTS = 10
LEVERAGE = 1  # Hefboom voor jouw strategie (niet voor de olieprijs!)

# ==============================================================================
# 2. DATA FUNCTIES
# ==============================================================================
def fetch_raw_data():
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{FOLDER_PATH}?ref=master"
    
    try:
        res = requests.get(api_url, headers=headers)
        if res.status_code == 200:
            files = res.json()
            csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
            if csv_file:
                # We gebruiken parse_dates=['time'] voor betere datumherkenning
                df = pd.read_csv(csv_file['download_url'], parse_dates=['time'])
                df = df.sort_values('time')
                return df
    except Exception as e:
        print(f"Fout bij ophalen ruwe data: {e}")
    return None

def calculate_compounding_equity(df_logs):
    """
    Berekent de portfolio waarde met compounding (winst herinvesteren).
    """
    events = []
    
    # 1. Zet trades om in events
    for idx, row in df_logs.iterrows():
        # Entry Event
        events.append({
            'time': row['entry_time'], 
            'type': 'ENTRY', 
            'trade_id': idx, 
            'return': 0.0
        })
        # Exit Event (alleen als exit tijd bestaat)
        if pd.notnull(row['exit_time']):
            events.append({
                'time': row['exit_time'], 
                'type': 'EXIT', 
                'trade_id': idx, 
                'return': row['return']
            })
            
    df_events = pd.DataFrame(events)
    # Sorteer: bij gelijke tijd eerst EXIT (geld vrijmaken), dan ENTRY
    df_events['sort_order'] = df_events['type'].apply(lambda x: 1 if x == 'ENTRY' else 0)
    df_events = df_events.sort_values(by=['time', 'sort_order'])
    
    # 2. Loop door de tijd
    cash = START_CAPITAL
    open_positions = {} # {trade_id: invested_amount}
    equity_curve = []
    
    # Startpunt (iets voor de eerste trade)
    if not df_events.empty:
        start_time = df_events['time'].min() - pd.Timedelta(minutes=1)
        equity_curve.append({'time': start_time, 'equity': START_CAPITAL})

    for _, event in df_events.iterrows():
        current_time = event['time']
        
        # Huidige waarde berekenen (Cash + Inleg van open posities)
        # Note: We negeren hier even de floating P&L van open posities voor de snelheid
        current_equity = cash + sum(open_positions.values())
        
        if event['type'] == 'ENTRY':
            # Is er plek? (Max 10 slots)
            if len(open_positions) < MAX_SLOTS:
                # Bereken positie grootte: 1/10e van huidige totale waarde
                position_size = current_equity / MAX_SLOTS
                
                # Veiligheid: niet meer inleggen dan cash
                if cash < position_size: 
                    position_size = cash
                
                cash -= position_size
                open_positions[event['trade_id']] = position_size
                
        elif event['type'] == 'EXIT':
            if event['trade_id'] in open_positions:
                invested_amount = open_positions.pop(event['trade_id'])
                
                # Bereken resultaat MET LEVERAGE
                # Als return 0.001 is (0.1%) en leverage is 10, dan is trade_return 0.01 (1%)
                trade_return = event['return'] * LEVERAGE
                
                returned_amount = invested_amount * (1 + trade_return)
                cash += returned_amount

        # Update curve punt
        total_equity = cash + sum(open_positions.values())
        equity_curve.append({'time': current_time, 'equity': total_equity})

    return pd.DataFrame(equity_curve)

# ==============================================================================
# 3. PLOTTEN (DE EERLIJKE VERGELIJKING)
# ==============================================================================
def generate_performance_plots():
    log_path = os.path.join(OUTPUT_DIR, LOG_FILE)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    
    if not os.path.exists(log_path):
        print("Geen logbestand gevonden.")
        return

    # Data laden (Logs)
    # We laten pandas de datum format raden, dat is vaak robuuster
    df_logs = pd.read_csv(log_path)
    df_logs['entry_time'] = pd.to_datetime(df_logs['entry_time'], errors='coerce')
    df_logs['exit_time'] = pd.to_datetime(df_logs['exit_time'], errors='coerce')

    # Filter trades
    df_trades = df_logs[
        (~df_logs['exit_reason'].isin(['No Trade', 'No Trade (Init)', 'No Trade (Data Error)'])) & 
        (df_logs['entry_p'].notna())
    ].copy()

    if df_trades.empty:
        print("Geen trades om te plotten.")
        return

    # --- 1. BEREKEN JOUW STRATEGIE ---
    df_equity = calculate_compounding_equity(df_trades)
    
    # --- 2. BEREKEN DE OLIEPRIJS (BENCHMARK) ---
    raw_data = fetch_raw_data()
    
    # Plot Setup
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    if raw_data is not None and not df_equity.empty:
        start_time = df_equity['time'].min()
        end_time = df_equity['time'].max()
        
        # We pakken EXACT dezelfde periode uit de ruwe data
        mask = (raw_data['time'] >= start_time) & (raw_data['time'] <= end_time)
        df_bh = raw_data.loc[mask].copy()
        
        if not df_bh.empty:
            # --- DIAGNOSE START ---
            price_start = df_bh['close_bid'].iloc[0]
            price_end = df_bh['close_bid'].iloc[-1]
            price_change_pct = ((price_end - price_start) / price_start) * 100
            print(f"\n--- DATA CHECK ---")
            print(f"Grafiek start datum: {start_time}")
            print(f"Grafiek eind datum:  {end_time}")
            print(f"Olieprijs Start:     ${price_start:.2f}")
            print(f"Olieprijs Eind:      ${price_end:.2f}")
            print(f"Olieprijs Groei:     {price_change_pct:.2f}% (Dit is waarom de grijze lijn beweegt!)")
            print(f"------------------\n")
            # --- DIAGNOSE EIND ---

            # Normaliseer: We doen alsof je €10.000 olie kocht op t=0
            # Formule: (Huidige Prijs / Start Prijs) * Start Kapitaal
            df_bh['portfolio_value'] = (df_bh['close_bid'] / price_start) * START_CAPITAL
            
            bh_ret = ((df_bh['portfolio_value'].iloc[-1] - START_CAPITAL) / START_CAPITAL) * 100
            
            # Plot Olie (Grijs)
            ax1.plot(df_bh['time'], df_bh['portfolio_value'], 
                    color='gray', linestyle='-', alpha=0.6, label=f'Olieprijs (Buy & Hold 1x): {bh_ret:+.1f}%')

    # Plot Jouw Strategie (Groen)
    if not df_equity.empty:
        strat_final = df_equity.iloc[-1]['equity']
        strat_ret = ((strat_final - START_CAPITAL) / START_CAPITAL) * 100
        
        ax1.plot(df_equity['time'], df_equity['equity'], 
                 color='#00CC00', linewidth=2.5, label=f'Jouw Strategie ({LEVERAGE}x): {strat_ret:+.1f}%')
        
        # Groene gloed onder de lijn
        ax1.fill_between(df_equity['time'], START_CAPITAL, df_equity['equity'], color='#00CC00', alpha=0.1)

    # --- OPMAAK (1 AS, LINEAIR) ---
    ax1.set_title(f"Eerlijke Vergelijking: Wat is je €{START_CAPITAL:.0f} nu waard?", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Portfolio Waarde (€)", fontsize=12)
    
    # Zwarte lijn op startbedrag (Break-even)
    ax1.axhline(START_CAPITAL, color='black', linewidth=1.5, linestyle='-')

    # Valuta op de Y-as (Euro teken)
    fmt = '€{x:,.0f}'
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter(fmt))
    
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=11, frameon=True, facecolor='white', framealpha=1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    
    fig.autofmt_xdate()
    
    save_path = os.path.join(PLOT_DIR, "equity_linear_comparison.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Grafiek opgeslagen: {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_performance_plots()
