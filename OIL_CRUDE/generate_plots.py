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
START_CAPITAL = 65.0
MAX_SLOTS = 10

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
    Berekent de portfolio waarde over de hele periode.
    """
    events = []
    
    for idx, row in df_logs.iterrows():
        trade_leverage = row.get('leverage', 1.0)
        
        events.append({
            'time': row['entry_time'], 
            'type': 'ENTRY', 
            'trade_id': idx, 
            'return': 0.0,
            'leverage': trade_leverage
        })
        
        if pd.notnull(row['exit_time']):
            events.append({
                'time': row['exit_time'], 
                'type': 'EXIT', 
                'trade_id': idx, 
                'return': row['return'],
                'leverage': trade_leverage
            })
            
    df_events = pd.DataFrame(events)
    if df_events.empty:
        return pd.DataFrame()

    df_events['sort_order'] = df_events['type'].apply(lambda x: 1 if x == 'ENTRY' else 0)
    df_events = df_events.sort_values(by=['time', 'sort_order'])
    
    cash = START_CAPITAL
    open_positions = {} 
    equity_curve = []
    
    # Startpunt
    start_time = df_events['time'].min() - pd.Timedelta(minutes=1)
    equity_curve.append({'time': start_time, 'equity': START_CAPITAL})

    for _, event in df_events.iterrows():
        current_time = event['time']
        
        # Huidige equity berekenen (Cash + Open Posities)
        # Note: Voor een perfecte intraday curve zou je hier eigenlijk live prijzen moeten gebruiken,
        # maar dit berekent de equity op de 'event' momenten (entry/exit), wat meestal voldoende is.
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
                actual_leverage = event['leverage']
                trade_return = event['return'] * actual_leverage
                returned_amount = invested_amount * (1 + trade_return)
                cash += returned_amount

        total_equity = cash + sum(open_positions.values())
        equity_curve.append({'time': current_time, 'equity': total_equity})

    return pd.DataFrame(equity_curve)

# ==============================================================================
# 3. GENERATE DAILY PLOTS (NIEUWE FUNCTIE)
# ==============================================================================
def generate_daily_plots(df_trades, df_equity, df_raw):
    """
    Maakt per dag een plot met:
    1. Prijsverloop + Entry/Exit markers
    2. Equity verloop van die dag
    """
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        
    # We groeperen de trades per datum
    unique_dates = df_trades['entry_time'].dt.date.unique()
    
    print(f"Start genereren van {len(unique_dates)} dagelijkse plots...")

    for trade_date in unique_dates:
        # Filter data voor deze specifieke dag
        day_start = pd.Timestamp(trade_date)
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # 1. Trades op deze dag
        daily_trades = df_trades[df_trades['entry_time'].dt.date == trade_date]
        
        # 2. Ruwe data (Prijs) voor deze dag
        if df_raw is not None:
            mask_raw = (df_raw['time'] >= day_start) & (df_raw['time'] <= day_end)
            daily_raw = df_raw.loc[mask_raw]
        else:
            daily_raw = pd.DataFrame()

        # 3. Equity data voor deze dag
        # We pakken ook het laatste punt van de vorige dag mee (indien beschikbaar) voor een mooie lijn vanaf start
        mask_eq = (df_equity['time'] >= day_start) & (df_equity['time'] <= day_end)
        daily_equity = df_equity.loc[mask_eq]
        
        # --- PLOTTEN ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # SUBPLOT 1: PRIJS & TRADES
        if not daily_raw.empty:
            ax1.plot(daily_raw['time'], daily_raw['close_bid'], color='gray', alpha=0.5, label='Price (Bid)')
        
        # Entries (Groen Pijltje Omhoog)
        ax1.scatter(daily_trades['entry_time'], daily_trades['entry_p'], 
                    color='green', marker='^', s=100, label='Buy', zorder=5)
        
        # Exits (Rood Pijltje Omlaag) - alleen als ze op dezelfde dag zijn
        # (Als een trade overnacht, wordt de exit hier niet getoond op de entry dag, wat logisch is)
        daily_exits = daily_trades[daily_trades['exit_time'].dt.date == trade_date]
        ax1.scatter(daily_exits['exit_time'], daily_exits['exit_p'], 
                    color='red', marker='v', s=100, label='Sell', zorder=5)

        ax1.set_title(f"Trading Recap: {trade_date}", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Prijs ($)")
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # SUBPLOT 2: EQUITY CURVE VANDAAG
        if not daily_equity.empty:
            ax2.plot(daily_equity['time'], daily_equity['equity'], color='#00CC00', linewidth=2)
            ax2.fill_between(daily_equity['time'], daily_equity['equity'].min(), daily_equity['equity'], color='#00CC00', alpha=0.1)
            
            # Start en Eind saldo van de dag in titel
            start_bal = daily_equity.iloc[0]['equity']
            end_bal = daily_equity.iloc[-1]['equity']
            pnl = end_bal - start_bal
            ax2.set_title(f"Equity: €{start_bal:,.0f} -> €{end_bal:,.0f} (PnL: €{pnl:+,.2f})", fontsize=10)
        
        ax2.set_ylabel("Equity (€)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # X-as formatting (Tijd)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Bestand opslaan
        filename = f"{trade_date}_daily_recap.png"
        save_path = os.path.join(PLOT_DIR, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() # Belangrijk om geheugen vrij te maken
        
    print(f"Klaar! {len(unique_dates)} afbeeldingen opgeslagen in {PLOT_DIR}")

# ==============================================================================
# 4. HOOFDFUNCTIE
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

    # 1. Data voorbereiden
    raw_data = fetch_raw_data()
    df_equity = calculate_compounding_equity(df_trades)
    
    # 2. GLOBAL PLOT (Totaal overzicht) -> In OUTPUT_DIR
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    if raw_data is not None and not df_equity.empty:
        start_time = df_equity['time'].min()
        end_time = df_equity['time'].max()
        mask = (raw_data['time'] >= start_time) & (raw_data['time'] <= end_time)
        df_bh = raw_data.loc[mask].copy()
        
        if not df_bh.empty:
            first_price = df_bh['close_bid'].iloc[0]
            df_bh['portfolio_value'] = (df_bh['close_bid'] / first_price) * START_CAPITAL
            bh_ret = ((df_bh['portfolio_value'].iloc[-1] - START_CAPITAL) / START_CAPITAL) * 100
            ax1.plot(df_bh['time'], df_bh['portfolio_value'], 
                     color='gray', linestyle='-', alpha=0.6, label=f'Olieprijs (Buy & Hold): {bh_ret:+.1f}%')

    if not df_equity.empty:
        strat_final = df_equity.iloc[-1]['equity']
        strat_ret = ((strat_final - START_CAPITAL) / START_CAPITAL) * 100
        ax1.plot(df_equity['time'], df_equity['equity'], 
                 color='#00CC00', linewidth=2.5, label=f'Strategie: {strat_ret:+.1f}%')
        ax1.fill_between(df_equity['time'], START_CAPITAL, df_equity['equity'], color='#00CC00', alpha=0.1)

    ax1.set_title("Totale Equity Curve", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Portfolio Waarde (€)", fontsize=12)
    ax1.axhline(START_CAPITAL, color='black', linewidth=1.5, linestyle='-')
    
    fmt = '€{x:,.0f}'
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter(fmt))
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    
    fig.autofmt_xdate()
    save_path_global = os.path.join(OUTPUT_DIR, "equity_linear_comparison.png")
    plt.savefig(save_path_global, bbox_inches='tight', dpi=150)
    print(f"Totaal grafiek opgeslagen: {save_path_global}")
    plt.close()

    # 3. DAILY PLOTS (Per dag) -> In PLOT_DIR
    generate_daily_plots(df_trades, df_equity, raw_data)

if __name__ == "__main__":
    generate_performance_plots()
