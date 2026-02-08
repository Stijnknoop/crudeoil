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
LEVERAGE = 10

# ==============================================================================
# 2. DATA FUNCTIES
# ==============================================================================
def fetch_raw_data():
    """Haalt de prijsdata op van GitHub."""
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
    """Zorgt dat kolomnamen consistent zijn."""
    rename_map = {
        'entry_time': 'Entry_Time',
        'exit_time': 'Exit_Time',
        'entry_p': 'Entry_Price',
        'exit_p': 'Exit_Price',
        'profit_abs': 'Profit_Euro',
        'side': 'Side'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def calculate_equity_curve(df_trades):
    """Berekent de equity curve (Startkapitaal + Winsten)."""
    df_equity = df_trades[['Exit_Time', 'Profit_Euro']].copy()
    df_equity = df_equity.sort_values('Exit_Time')
    
    df_equity['Cumulative_Profit'] = df_equity['Profit_Euro'].cumsum()
    df_equity['Equity'] = START_CAPITAL + df_equity['Cumulative_Profit']
    
    start_time = df_trades['Entry_Time'].min() - pd.Timedelta(minutes=10)
    start_row = pd.DataFrame([{'Exit_Time': start_time, 'Profit_Euro': 0, 'Cumulative_Profit': 0, 'Equity': START_CAPITAL}])
    
    df_final = pd.concat([start_row, df_equity], ignore_index=True)
    return df_final

# ==============================================================================
# 3. GENERATE DAILY PLOTS (MET LIJNEN)
# ==============================================================================
def generate_daily_plots(df_trades, df_equity, df_raw):
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        
    unique_dates = df_trades['Entry_Time'].dt.date.unique()
    print(f"Start genereren van {len(unique_dates)} dagelijkse plots...")

    for trade_date in unique_dates:
        # Filter data voor deze dag
        day_start = pd.Timestamp(trade_date)
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        daily_trades = df_trades[df_trades['Entry_Time'].dt.date == trade_date]
        
        if df_raw is not None:
            mask_raw = (df_raw['time'] >= day_start) & (df_raw['time'] <= day_end)
            daily_raw = df_raw.loc[mask_raw]
        else:
            daily_raw = pd.DataFrame()

        # Equity data van deze dag
        prev_equity_data = df_equity[df_equity['Exit_Time'] < day_start]
        start_of_day_equity = prev_equity_data.iloc[-1]['Equity'] if not prev_equity_data.empty else START_CAPITAL

        mask_eq = (df_equity['Exit_Time'] >= day_start) & (df_equity['Exit_Time'] <= day_end)
        daily_equity_events = df_equity.loc[mask_eq].copy()
        
        start_point = pd.DataFrame([{'Exit_Time': day_start, 'Equity': start_of_day_equity}])
        daily_equity_plot = pd.concat([start_point, daily_equity_events], ignore_index=True)

        # --- PLOTTEN ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # --- SUBPLOT 1: PRIJS & TRADE LIJNEN ---
        if not daily_raw.empty:
            # Prijs op de achtergrond (lichtgrijs)
            ax1.plot(daily_raw['time'], daily_raw['close_bid'], color='#e0e0e0', alpha=1.0, label='Price', linewidth=1.5, zorder=1)
        
        # Loop over elke trade om lijnen te tekenen
        win_count = 0
        loss_count = 0
        
        for _, trade in daily_trades.iterrows():
            entry_t = trade['Entry_Time']
            exit_t = trade['Exit_Time']
            entry_p = trade['Entry_Price']
            exit_p = trade['Exit_Price']
            profit = trade['Profit_Euro']
            
            # Kleur bepalen
            if profit > 0:
                color = '#00aa00' # Groen
                win_count += 1
            else:
                color = '#ff0000' # Rood
                loss_count += 1
            
            # 1. Teken de lijn van Entry naar Exit
            ax1.plot([entry_t, exit_t], [entry_p, exit_p], color=color, linewidth=2, alpha=0.9, zorder=3)
            
            # 2. Teken Entry marker (Blauw driehoekje, zoals in je voorbeeld)
            ax1.scatter(entry_t, entry_p, color='blue', marker='^', s=80, zorder=4)
            
            # 3. Optioneel: Exit punt (klein bolletje in kleur van resultaat)
            ax1.scatter(exit_t, exit_p, color=color, marker='o', s=20, zorder=4)

        ax1.set_title(f"Trading Recap: {trade_date} | Wins: {win_count} | Losses: {loss_count}", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Prijs ($)")
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- SUBPLOT 2: EQUITY ---
        if not daily_equity_plot.empty:
            ax2.step(daily_equity_plot['Exit_Time'], daily_equity_plot['Equity'], where='post', color='blue', linewidth=2)
            
            end_bal = daily_equity_plot.iloc[-1]['Equity']
            pnl = end_bal - start_of_day_equity
            pnl_color = "green" if pnl >= 0 else "red"
            
            ax2.set_title(f"Dagwinst: €{pnl:+,.2f}", fontsize=12, color=pnl_color, fontweight='bold')
            ax2.fill_between(daily_equity_plot['Exit_Time'], start_of_day_equity, daily_equity_plot['Equity'], step='post', color='blue', alpha=0.1)
        
        ax2.set_ylabel("Equity (€)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # Opmaak X-as
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=0)
        
        filename = f"{trade_date}_daily_recap.png"
        save_path = os.path.join(PLOT_DIR, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    print(f"Klaar! {len(unique_dates)} plots opgeslagen in {PLOT_DIR}")

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
    df_logs = standardize_columns(df_logs)
    
    df_logs['Entry_Time'] = pd.to_datetime(df_logs['Entry_Time'], errors='coerce')
    df_logs['Exit_Time'] = pd.to_datetime(df_logs['Exit_Time'], errors='coerce')

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

    raw_data = fetch_raw_data()
    df_equity = calculate_equity_curve(df_trades)
    
    # Global Plot (Blijft hetzelfde)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    if not df_equity.empty:
        strat_final = df_equity.iloc[-1]['Equity']
        strat_ret = ((strat_final - START_CAPITAL) / START_CAPITAL) * 100
        ax1.step(df_equity['Exit_Time'], df_equity['Equity'], where='post', color='blue', linewidth=2, label=f'Strategie: {strat_ret:+.1f}%')
        ax1.fill_between(df_equity['Exit_Time'], START_CAPITAL, df_equity['Equity'], step='post', color='blue', alpha=0.1)

    ax1.set_title(f"Totale Equity Curve (Start: €{START_CAPITAL})", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Portfolio (€)")
    ax1.axhline(START_CAPITAL, color='red', linestyle='--')
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('€{x:,.0f}'))
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    fig.autofmt_xdate()
    
    save_path_global = os.path.join(OUTPUT_DIR, "equity_linear_comparison.png")
    plt.savefig(save_path_global, bbox_inches='tight', dpi=150)
    plt.close()

    # Daily Plots (Nieuwe stijl)
    generate_daily_plots(df_trades, df_equity, raw_data)

if __name__ == "__main__":
    generate_performance_plots()
