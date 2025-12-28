import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests
import datetime
import numpy as np

def generate_performance_plots():
    log_dir = "Trading_details"
    plot_dir = os.path.join(log_dir, "plots")
    log_path = os.path.join(log_dir, "trading_logs.csv")
    
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(log_path): return

    df_logs = pd.read_csv(log_path)
    df_trades = df_logs[~df_logs['exit_reason'].isin(['No Trade', 'Data End (Pending)'])].copy()
    if df_trades.empty: return

    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    raw_data = fetch_raw_data() # Gebruik je bestaande fetch functie
    
    if raw_data is not None:
        raw_data['time'] = pd.to_datetime(raw_data['time'])
        raw_data['date'] = raw_data['time'].dt.date
        all_dates = sorted(raw_data['date'].unique())

        refresh_dates = [datetime.date.today().strftime('%Y-%m-%d'), 
                         (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')]

        for _, trade in df_trades.iterrows():
            trade_date_dt = trade['entry_time'].date()
            trade_date_str = trade_date_dt.strftime('%Y-%m-%d')
            file_path = os.path.join(plot_dir, f"{trade_date_str}.png")

            # Alleen verversen als het nieuw is of van vandaag/gisteren
            if os.path.exists(file_path) and (trade_date_str not in refresh_dates):
                continue
            
            # --- DATA VOORBEREIDING ---
            day_data = raw_data[raw_data['date'] == trade_date_dt].copy()
            if day_data.empty: continue
            
            try:
                curr_idx = all_dates.index(trade_date_dt)
                hist_dates = all_dates[max(0, curr_idx-40):curr_idx]
                split = int(len(hist_dates) * 0.75)
                train_dates, val_dates = hist_dates[:split], hist_dates[split:]
            except: train_dates, val_dates = [], []

            # --- PLOTTING ---
            # We maken 3 subplots naast elkaar (1 rij, 3 kolommen)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            
            # 1. Training (Groen)
            for d in train_dates:
                d_pts = raw_data[raw_data['date'] == d]['close_bid'].values
                ax1.plot(d_pts, color='green', alpha=0.1, linewidth=0.5)
            ax1.set_title(f"Fase 1: Training ({len(train_dates)} dagen)")
            ax1.set_xticks([]); ax1.set_ylabel("Prijs")

            # 2. Validatie (Oranje)
            for d in val_dates:
                d_pts = raw_data[raw_data['date'] == d]['close_bid'].values
                ax2.plot(d_pts, color='orange', alpha=0.2, linewidth=0.8)
            ax2.set_title(f"Fase 2: Validatie ({len(val_dates)} dagen)")
            ax2.set_xticks([])

            # 3. Live Test Dag (Blauw + Trades)
            ax3.plot(day_data['time'], day_data['close_bid'], color='blue', alpha=0.4, label='Koers')
            ax3.scatter(trade['entry_time'], trade['entry_p'], color='green', marker='^', s=150, label='ENTRY', zorder=5)
            if not pd.isna(trade['exit_time']):
                ax3.scatter(pd.to_datetime(trade['exit_time']), trade['exit_p'], color='red', marker='v', s=150, label='EXIT', zorder=5)
            
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax3.set_title(f"Fase 3: Live Resultaat ({trade['return']:.2%})")
            ax3.legend(loc='upper left')
            plt.setp(ax3.get_xticklabels(), rotation=30)

            # Algemene layout
            plt.suptitle(f"Trading Analyse Rapport: {trade_date_str}", fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            plt.savefig(file_path, bbox_inches='tight', dpi=120)
            plt.close(fig)
            print(f"✅ Succesvol hersteld: {trade_date_str}.png")
