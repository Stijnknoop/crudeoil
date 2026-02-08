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

# Account Settings (Moet matchen met new_strategy.py)
START_CAPITAL = 65.0
MAX_SLOTS = 10
LEVERAGE = 10

# ==============================================================================
# 2. DATA FUNCTIES
# ==============================================================================
def fetch_raw_data():
    """Haalt de prijsdata op van GitHub om de grafiek te tekenen."""
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

def standardize_columns(df):
    """Zorgt dat kolomnamen matchen met de nieuwe strategie output."""
    rename_map = {
        'entry_time': 'Entry_Time',
        'exit_time': 'Exit_Time',
        'entry_p': 'Entry_Price',
        'exit_p': 'Exit_Price',
        'profit_abs': 'Profit_Euro',
        'side': 'Side'
    }
    # Hernoem alleen als de oude naam bestaat
    df.rename(columns=rename_map, inplace=True)
    return df

def calculate_equity_curve(df_trades):
    """
    Berekent de equity curve simpelweg door de gerealiseerde winsten (Profit_Euro)
    op te tellen bij het startkapitaal.
    """
    # We maken een tijdlijn op basis van EXIT tijden (want dan is de winst gerealiseerd)
    df_equity = df_trades[['Exit_Time', 'Profit_Euro']].copy()
    df_equity = df_equity.sort_values('Exit_Time')
    
    # Cumulatieve winst berekenen
    df_equity['Cumulative_Profit'] = df_equity['Profit_Euro'].cumsum()
    df_equity['Equity'] = START_CAPITAL + df_equity['Cumulative_Profit']
    
    # Voeg een startpunt toe (Tijdstip van eerste entry - 1 minuut, Equity = Start Capital)
    start_time = df_trades['Entry_Time'].min() - pd.Timedelta(minutes=10)
    start_row = pd.DataFrame([{'Exit_Time': start_time, 'Profit_Euro': 0, 'Cumulative_Profit': 0, 'Equity': START_CAPITAL}])
    
    df_final = pd.concat([start_row, df_equity], ignore_index=True)
    return df_final

# ==============================================================================
# 3. GENERATE DAILY PLOTS
# ==============================================================================
def generate_daily_plots(df_trades, df_equity, df_raw):
    """
    Maakt per dag een plot met:
    1. Prijsverloop + Entry/Exit markers
    2. Equity verloop van die dag
    """
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        
    unique_dates = df_trades['Entry_Time'].dt.date.unique()
    
    print(f"Start genereren van {len(unique_dates)} dagelijkse plots...")

    for trade_date in unique_dates:
        # Filter data voor deze specifieke dag
        day_start = pd.Timestamp(trade_date)
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # 1. Trades op deze dag
        daily_trades = df_trades[df_trades['Entry_Time'].dt.date == trade_date]
        
        # 2. Ruwe data (Prijs) voor deze dag
        if df_raw is not None:
            mask_raw = (df_raw['time'] >= day_start) & (df_raw['time'] <= day_end)
            daily_raw = df_raw.loc[mask_raw]
        else:
            daily_raw = pd.DataFrame()

        # 3. Equity data voor deze dag
        # Pak de equity stand aan het begin van de dag (laatste waarde van gisteren of startkapitaal)
        prev_equity_data = df_equity[df_equity['Exit_Time'] < day_start]
        if not prev_equity_data.empty:
            start_of_day_equity = prev_equity_data.iloc[-1]['Equity']
        else:
            start_of_day_equity = START_CAPITAL

        # Filter equity events van vandaag
        mask_eq = (df_equity['Exit_Time'] >= day_start) & (df_equity['Exit_Time'] <= day_end)
        daily_equity_events = df_equity.loc[mask_eq].copy()
        
        # Maak een nette tijdlijn voor de equity plot (stapgrafiek)
        # We voegen het startpunt van de dag toe
        start_point = pd.DataFrame([{'Exit_Time': day_start, 'Equity': start_of_day_equity}])
        daily_equity_plot = pd.concat([start_point, daily_equity_events], ignore_index=True)

        # --- PLOTTEN ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # SUBPLOT 1: PRIJS & TRADES
        if not daily_raw.empty:
            ax1.plot(daily_raw['time'], daily_raw['close_bid'], color='gray', alpha=0.5, label='Price (Bid)')
        
        # Entries
        ax1.scatter(daily_trades['Entry_Time'], daily_trades['Entry_Price'], 
                    color='green', marker='^', s=100, label='Buy', zorder=5)
        
        # Exits (alleen als ze op dezelfde dag vallen)
        daily_exits = daily_trades[daily_trades['Exit_Time'].dt.date == trade_date]
        ax1.scatter(daily_exits['Exit_Time'], daily_exits['Exit_Price'], 
                    color='red', marker='v', s=100, label='Sell', zorder=5)

        ax1.set_title(f"Trading Recap: {trade_date}", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Prijs ($)")
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # SUBPLOT 2: EQUITY CURVE VANDAAG (Step plot is logischer voor trades)
        if not daily_equity_plot.empty:
            # Step-post zorgt ervoor dat de lijn pas omhoog gaat op het moment van exit
            ax2.step(daily_equity_plot['Exit_Time'], daily_equity_plot['Equity'], where='post', color='#00CC00', linewidth=2)
            
            # Start en Eind saldo van de dag in titel
            end_bal = daily_equity_plot.iloc[-1]['Equity']
            pnl = end_bal - start_of_day_equity
            ax2.set_title(f"Equity: €{start_of_day_equity:,.2f} -> €{end_bal:,.2f} (PnL: €{pnl:+,.2f})", fontsize=10)
        
        ax2.set_ylabel("Equity (€)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # X-as formatting (Tijd)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Bestand opslaan
        filename = f"{trade_date}_daily_recap.png"
        save_path = os.path.join(PLOT_DIR, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
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
    
    # 1. Standardiseer kolomnamen (New Strategy Compatibiliteit)
    df_logs = standardize_columns(df_logs)
    
    # Datum conversie
    df_logs['Entry_Time'] = pd.to_datetime(df_logs['Entry_Time'], errors='coerce')
    df_logs['Exit_Time'] = pd.to_datetime(df_logs['Exit_Time'], errors='coerce')

    # Filter valide trades
    if 'Exit_Reason' in df_logs.columns:
        df_trades = df_logs[
            (~df_logs['Exit_Reason'].astype(str).str.contains('No Trade', case=False, na=False)) & 
            (df_logs['Entry_Price'].notna())
        ].copy()
    else:
        df_trades = df_logs.copy()

    if df_trades.empty:
        print("Geen trades om te plotten.")
        return

    # 2. Data voorbereiden
    raw_data = fetch_raw_data()
    df_equity = calculate_equity_curve(df_trades)
    
    # 3. GLOBAL PLOT (Totaal overzicht)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Buy & Hold Curve
    if raw_data is not None and not df_equity.empty:
        start_time = df_equity['Exit_Time'].min()
        end_time = df_equity['Exit_Time'].max()
        mask = (raw_data['time'] >= start_time) & (raw_data['time'] <= end_time)
        df_bh = raw_data.loc[mask].copy()
        
        if not df_bh.empty:
            first_price = df_bh['close_bid'].iloc[0]
            # Normaliseer Buy & Hold naar Startkapitaal
            df_bh['portfolio_value'] = (df_bh['close_bid'] / first_price) * START_CAPITAL
            bh_ret = ((df_bh['portfolio_value'].iloc[-1] - START_CAPITAL) / START_CAPITAL) * 100
            ax1.plot(df_bh['time'], df_bh['portfolio_value'], 
                     color='gray', linestyle='-', alpha=0.6, label=f'Olieprijs (Buy & Hold): {bh_ret:+.1f}%')

    # Strategy Curve
    if not df_equity.empty:
        strat_final = df_equity.iloc[-1]['Equity']
        strat_ret = ((strat_final - START_CAPITAL) / START_CAPITAL) * 100
        
        # Step plot is realistischer omdat equity pas update na een exit
        ax1.step(df_equity['Exit_Time'], df_equity['Equity'], where='post',
                 color='#00CC00', linewidth=2.5, label=f'Strategie: {strat_ret:+.1f}%')
        
        # Fill area
        ax1.fill_between(df_equity['Exit_Time'], START_CAPITAL, df_equity['Equity'], step='post', color='#00CC00', alpha=0.1)

    ax1.set_title(f"Equity Curve (Start: €{START_CAPITAL})", fontsize=14, fontweight='bold')
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

    # 4. DAILY PLOTS
    generate_daily_plots(df_trades, df_equity, raw_data)

if __name__ == "__main__":
    generate_performance_plots()
