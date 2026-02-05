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
# LEVERAGE = 10  <-- DEZE IS WEG! We lezen het nu uit de data.

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
                df = pd.read_csv(csv_file['download_url'], parse_dates=['time'])
                df = df.sort_values('time')
                return df
    except Exception as e:
        print(f"Fout bij ophalen ruwe data: {e}")
    return None

def calculate_compounding_equity(df_logs):
    """
    Berekent de portfolio waarde.
    Leest de leverage PER TRADE uit de log file.
    """
    events = []
    
    for idx, row in df_logs.iterrows():
        # We halen de leverage op uit de CSV. Als die kolom mist, standaard 1.
        trade_leverage = row.get('leverage', 1.0)
        
        events.append({
            'time': row['entry_time'], 
            'type': 'ENTRY', 
            'trade_id': idx, 
            'return': 0.0,
            'leverage': trade_leverage # Opslaan voor later
        })
        
        if pd.notnull(row['exit_time']):
            events.append({
                'time': row['exit_time'], 
                'type': 'EXIT', 
                'trade_id': idx, 
                'return': row['return'],
                'leverage': trade_leverage # Opslaan voor later
            })
            
    df_events = pd.DataFrame(events)
    df_events['sort_order'] = df_events['type'].apply(lambda x: 1 if x == 'ENTRY' else 0)
    df_events = df_events.sort_values(by=['time', 'sort_order'])
    
    cash = START_CAPITAL
    open_positions = {} 
    equity_curve = []
    
    # Startpunt
    if not df_events.empty:
        start_time = df_events['time'].min() - pd.Timedelta(minutes=1)
        equity_curve.append({'time': start_time, 'equity': START_CAPITAL})

    for _, event in df_events.iterrows():
        current_time = event['time']
        current_equity = cash + sum(open_positions.values())
        
        if event['type'] == 'ENTRY':
            if len(open_positions) < MAX_SLOTS:
                position_size = current_equity / MAX_SLOTS
                if cash < position_size: position_size = cash
                cash -= position_size
                open_positions[event['trade_id']] = position_size
                
        elif event['type'] == 'EXIT':
            if event['trade_id'] in open_positions:
                invested_amount = open_positions.pop(event['trade_id'])
                
                # HIER GEBRUIKEN WE DE LEVERAGE UIT DE CSV
                actual_leverage = event['leverage']
                trade_return = event['return'] * actual_leverage
                
                returned_amount = invested_amount * (1 + trade_return)
                cash += returned_amount

        total_equity = cash + sum(open_positions.values())
        equity_curve.append({'time': current_time, 'equity': total_equity})

    return pd.DataFrame(equity_curve)

# ==============================================================================
# 3. PLOTTEN
# ==============================================================================
def generate_performance_plots():
    log_path = os.path.join(OUTPUT_DIR, LOG_FILE)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    
    if not os.path.exists(log_path):
        print("Geen logbestand gevonden.")
        return

    df_logs = pd.read_csv(log_path)
    df_logs['entry_time'] = pd.to_datetime(df_logs['entry_time'], errors='coerce')
    df_logs['exit_time'] = pd.to_datetime(df_logs['exit_time'], errors='coerce')

    df_trades = df_logs[
        (~df_logs['exit_reason'].isin(['No Trade', 'No Trade (Init)', 'No Trade (Data Error)'])) & 
        (df_logs['entry_p'].notna())
    ].copy()

    if df_trades.empty:
        print("Geen trades om te plotten.")
        return

    # 1. Bereken Strategy Equity (Gebruikt leverage uit CSV)
    df_equity = calculate_compounding_equity(df_trades)
    
    # 2. Bereken Benchmark
    raw_data = fetch_raw_data()
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    if raw_data is not None and not df_equity.empty:
        start_time = df_equity['time'].min()
        end_time = df_equity['time'].max()
        mask = (raw_data['time'] >= start_time) & (raw_data['time'] <= end_time)
        df_bh = raw_data.loc[mask].copy()
        
        if not df_bh.empty:
            first_price = df_bh['close_bid'].iloc[0]
            # Normaliseer naar Start Capital
            df_bh['portfolio_value'] = (df_bh['close_bid'] / first_price) * START_CAPITAL
            
            bh_ret = ((df_bh['portfolio_value'].iloc[-1] - START_CAPITAL) / START_CAPITAL) * 100
            
            ax1.plot(df_bh['time'], df_bh['portfolio_value'], 
                    color='gray', linestyle='-', alpha=0.6, label=f'Olieprijs (Buy & Hold 1x): {bh_ret:+.1f}%')

    if not df_equity.empty:
        strat_final = df_equity.iloc[-1]['equity']
        strat_ret = ((strat_final - START_CAPITAL) / START_CAPITAL) * 100
        
        # We pakken de leverage van de laatste trade puur voor de label
        last_leverage = df_trades.iloc[-1].get('leverage', 'N/A')
        
        ax1.plot(df_equity['time'], df_equity['equity'], 
                 color='#00CC00', linewidth=2.5, label=f'Jouw Strategie (Dyn. Lev): {strat_ret:+.1f}%')
        
        ax1.fill_between(df_equity['time'], START_CAPITAL, df_equity['equity'], color='#00CC00', alpha=0.1)

    # Opmaak
    ax1.set_title(f"Eerlijke Vergelijking: Wat is je €{START_CAPITAL:.0f} nu waard?", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Portfolio Waarde (€)", fontsize=12)
    ax1.axhline(START_CAPITAL, color='black', linewidth=1.5, linestyle='-')
    
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
