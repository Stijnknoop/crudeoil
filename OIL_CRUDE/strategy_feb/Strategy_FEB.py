import os
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# =============================================================
# 0. CONFIGURATIE VOOR GITHUB
# =============================================================
OUTPUT_DIR = "OIL_CRUDE/strategy_feb/TradingDetails"
DAILY_PLOTS_DIR = os.path.join(OUTPUT_DIR, "DailyPlots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DAILY_PLOTS_DIR, exist_ok=True)

# =============================================================
# 1. DATA OPHALEN & SCHOONMAKEN
# =============================================================
def fetch_crude_data(user="Stijnknoop", repo="crudeoil", folder="OIL_CRUDE"):
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder}?ref=master"

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        csv_file = next((f for f in response.json() if f['name'].endswith('.csv')), None)
        
        if not csv_file:
            print("❌ Geen CSV-bestanden gevonden.")
            return None
        
        df = pd.read_csv(csv_file['download_url'])
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        time_ny = df['time'].dt.tz_localize('Europe/Amsterdam', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('America/New_York')
        df['time_us'] = time_ny.dt.tz_localize(None)
        
        return df.sort_values('time').reset_index(drop=True)
    except Exception as e:
        print(f"❌ Fout bij ophalen data: {e}")
        return None

def prep_and_engineer_features(df):
    is_new_session = df['time'].diff() > pd.Timedelta(minutes=45)
    is_new_session.iloc[0] = True
    df['session_id'] = is_new_session.cumsum()

    df = df.set_index('time').groupby('session_id').resample('1min').ffill()
    df = df.drop(columns=['session_id'], errors='ignore').reset_index()

    df['time_bucket'] = df['time_us'].dt.floor('30min').dt.time
    df['session_open'] = df.groupby('session_id')['close_bid'].transform('first')
    df['sessie_trend'] = np.where(df['close_bid'] >= df['session_open'], 'Groen (Up)', 'Rood (Down)')

    df['cum_high'] = df.groupby('session_id')['high_bid'].cummax()
    df['cum_low'] = df.groupby('session_id')['low_bid'].cummin()
    df['current_range'] = df['cum_high'] - df['cum_low']
    
    df['price_pos_pct'] = np.where(df['current_range'] == 0, 0.5, (df['close_bid'] - df['cum_low']) / df['current_range'])
    df['range_bucket'] = pd.cut(df['price_pos_pct'], 
                                bins=[-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, np.inf], 
                                labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'])

    df['MA_50'] = df.groupby('session_id')['close_bid'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
    df['dist_ma_pct'] = ((df['close_bid'] - df['MA_50']) / df['MA_50']) * 100
    df['ma_bucket'] = pd.cut(df['dist_ma_pct'], 
                             bins=[-np.inf, -0.15, -0.03, 0.03, 0.15, np.inf], 
                             labels=['Sterk Eronder', 'Eronder', 'Neutraal', 'Erboven', 'Sterk Erboven'])

    df['future_high_bid'] = df.iloc[::-1].groupby('session_id')['high_bid'].cummax()[::-1]
    df['future_low_bid']  = df.iloc[::-1].groupby('session_id')['low_bid'].cummin()[::-1]

    return df

# =============================================================
# 2. STRATEGIE & SIMULATIE 
# =============================================================
def genereer_signalen_dynamisch(df, min_kans, min_samples, max_kans_daling, scale_out_multiplier):
    unique_sessions = sorted(df['session_id'].unique())
    alle_trades = []
    ALT_XS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    for sessie_idx in range(40, len(unique_sessions)):
        huidige_sessie = unique_sessions[sessie_idx]
        sessie_data = df[df['session_id'] == huidige_sessie].sort_values('time')
        historie = df[df['session_id'].isin(unique_sessions[sessie_idx - 40 : sessie_idx])]

        scan_momenten = pd.date_range(start=sessie_data['time'].iloc[0], end=sessie_data['time'].iloc[-1], freq='15min')

        for moment in scan_momenten:
            data_tot_moment = sessie_data[sessie_data['time'] <= moment]
            if data_tot_moment.empty: continue
            
            actuele_rij = data_tot_moment.iloc[-1]
            hist_specifiek = historie[
                (historie['time_bucket'] == actuele_rij['time_bucket']) &
                (historie['range_bucket'] == actuele_rij['range_bucket']) &
                (historie['ma_bucket'] == actuele_rij['ma_bucket']) &
                (historie['sessie_trend'] == actuele_rij['sessie_trend'])
            ]

            if len(hist_specifiek) < min_samples: continue

            beste_multiplier = None
            for x_val in ALT_XS:
                kans_up = ((hist_specifiek['future_high_bid'] - hist_specifiek['close_ask']) >= (x_val * actuele_rij['current_range'])).mean()
                kans_down = ((hist_specifiek['close_ask'] - hist_specifiek['future_low_bid']) >= (x_val * actuele_rij['current_range'])).mean()

                if kans_up >= min_kans and kans_down <= max_kans_daling:
                    beste_multiplier = x_val

            if beste_multiplier is not None:
                c_prijs_ask = actuele_rij['close_ask']
                signaal_tijd = actuele_rij['time']
                
                target_full = c_prijs_ask + (beste_multiplier * actuele_rij['current_range'])
                target_1 = c_prijs_ask + (scale_out_multiplier * beste_multiplier * actuele_rij['current_range'])
                
                eval_start_tijd = signaal_tijd + pd.Timedelta(minutes=2)
                data_evaluatie = sessie_data[sessie_data['time'] >= eval_start_tijd]

                if data_evaluatie.empty: continue

                hit_fill = data_evaluatie[data_evaluatie['low_bid'] <= c_prijs_ask]
                hit_cancel = data_evaluatie[data_evaluatie['high_bid'] >= target_1]

                idx_fill = hit_fill.index[0] if not hit_fill.empty else float('inf')
                idx_cancel = hit_cancel.index[0] if not hit_cancel.empty else float('inf')
                
                if idx_fill == float('inf') or idx_cancel <= idx_fill:
                    alle_trades.append({
                        'Sessie': huidige_sessie, 'Signaal_Tijd': signaal_tijd,
                        'Entry_Tijd': pd.NaT, 'Entry_Prijs': c_prijs_ask,
                        'Target_Prijs_1': target_1, 'Target_Prijs_2': target_full, 
                        'Exit_Tijd_1': pd.NaT, 'Exit_Prijs_1': target_1,
                        'Exit_Tijd_2': pd.NaT, 'Exit_Prijs_2': target_full,
                        'Multiplier': beste_multiplier, 'Resultaat': 'GEANNULEERD 🚫',
                        'Status': 'Pending/Canceled'
                    })
                    continue

                echte_entry_tijd = hit_fill['time'].iloc[0]
                data_na = sessie_data[sessie_data['time'] > echte_entry_tijd]

                if not data_na.empty:
                    winst_momenten_1 = data_na[data_na['high_bid'] >= target_1]
                    
                    if winst_momenten_1.empty:
                        exit_t_1 = exit_t_2 = data_na['time'].iloc[-1]
                        exit_p_1 = exit_p_2 = data_na['close_bid'].iloc[-1]
                        res = 'VERLIES ❌'
                    else:
                        exit_t_1 = winst_momenten_1['time'].iloc[0]
                        exit_p_1 = target_1
                        
                        data_na_half = data_na[data_na['time'] > exit_t_1]
                        
                        if data_na_half.empty:
                            exit_t_2, exit_p_2, res = exit_t_1, exit_p_1, 'HALVE WINST / EINDE ✅⏳'
                        else:
                            hit_full = data_na_half[data_na_half['high_bid'] >= target_full]
                            hit_be = data_na_half[data_na_half['low_bid'] <= c_prijs_ask]
                            
                            idx_full = hit_full.index[0] if not hit_full.empty else float('inf')
                            idx_be = hit_be.index[0] if not hit_be.empty else float('inf')
                            
                            if idx_full < idx_be and idx_full != float('inf'):
                                exit_t_2, exit_p_2, res = hit_full['time'].iloc[0], target_full, 'VOLLEDIGE WINST ✅✅'
                            elif idx_be <= idx_full and idx_be != float('inf'):
                                exit_t_2, exit_p_2, res = hit_be['time'].iloc[0], c_prijs_ask, 'HALVE WINST / BE ✅➖'
                            else:
                                exit_t_2, exit_p_2, res = data_na_half['time'].iloc[-1], data_na_half['close_bid'].iloc[-1], 'HALVE WINST / EINDE ✅⏳'

                    alle_trades.append({
                        'Sessie': huidige_sessie, 'Signaal_Tijd': signaal_tijd,
                        'Entry_Tijd': echte_entry_tijd, 'Entry_Prijs': c_prijs_ask,
                        'Target_Prijs_1': target_1, 'Target_Prijs_2': target_full, 
                        'Exit_Tijd_1': exit_t_1, 'Exit_Prijs_1': exit_p_1,
                        'Exit_Tijd_2': exit_t_2, 'Exit_Prijs_2': exit_p_2,
                        'Multiplier': beste_multiplier, 'Resultaat': res,
                        'Status': 'Executed'
                    })
    
    return pd.DataFrame(alle_trades)

def run_portfolio_sim(df_trades, max_slots, start_kapitaal=100000, contract_multiplier=10):
    if df_trades.empty: return pd.DataFrame()

    huidige_balans = start_kapitaal
    actieve_trades = []
    geaccepteerde_trades = []
    
    df_executed = df_trades[df_trades['Status'] == 'Executed'].sort_values('Entry_Tijd').reset_index(drop=True)

    for _, trade in df_executed.iterrows():
        nog_actief = []
        for ot in actieve_trades:
            if trade['Entry_Tijd'] >= ot['exit_time']: 
                huidige_balans += ot['pnl']
            else: 
                nog_actief.append(ot)
        actieve_trades = nog_actief

        if len(actieve_trades) < max_slots:
            inleg = huidige_balans / max_slots
            eenheden = int(inleg / trade['Entry_Prijs'])
            
            if eenheden == 0: continue
            
            eenheden_exit_1 = int(eenheden / 2)
            eenheden_exit_2 = eenheden - eenheden_exit_1
            
            pnl_deel_1 = (trade['Exit_Prijs_1'] - trade['Entry_Prijs']) * eenheden_exit_1 * contract_multiplier
            pnl_deel_2 = (trade['Exit_Prijs_2'] - trade['Entry_Prijs']) * eenheden_exit_2 * contract_multiplier
            totale_pnl = pnl_deel_1 + pnl_deel_2
            
            actieve_trades.append({'exit_time': trade['Exit_Tijd_2'], 'pnl': totale_pnl})
            
            trade_data = trade.copy()
            trade_data['Gekochte_Eenheden'] = eenheden
            trade_data['PnL_Deel_1'] = pnl_deel_1  
            trade_data['PnL_Deel_2'] = pnl_deel_2  
            trade_data['PnL_Euro'] = totale_pnl
            geaccepteerde_trades.append(trade_data)

    for ot in actieve_trades: huidige_balans += ot['pnl']
    return pd.DataFrame(geaccepteerde_trades)

def bereken_zuivere_score(df_p):
    if df_p.empty or len(df_p) < 15: return 0, 0
    dag_pnl = df_p.groupby(df_p['Entry_Tijd'].dt.date)['PnL_Euro'].sum()
    std_pnl = dag_pnl.std()
    
    if std_pnl == 0: return 0, 0
    annualized_sharpe = (dag_pnl.mean() / std_pnl) * np.sqrt(252)
    return annualized_sharpe * df_p['PnL_Euro'].sum(), annualized_sharpe

# =============================================================
# 3. VISUALISATIE
# =============================================================
def visualiseer_alle_dagen(df_data, df_signals, df_portfolio, max_slots=3, output_dir=DAILY_PLOTS_DIR):
    if df_signals.empty: 
        print("ℹ️ Geen signalen om te visualiseren.")
        return
    
    unieke_datums = df_signals['Signaal_Tijd'].dt.strftime('%Y-%m-%d').unique()
    print(f"📈 Genereren van {len(unieke_datums)} dag-grafieken...")

    for datum_str in unieke_datums:
        dag_data = df_data[df_data['time'].dt.strftime('%Y-%m-%d') == datum_str].copy()
        dag_signals = df_signals[df_signals['Signaal_Tijd'].dt.strftime('%Y-%m-%d') == datum_str].copy()

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.plot(dag_data['time'], dag_data['close_bid'], label='Close Prijs', color='black', linewidth=1.5)
        ax.fill_between(dag_data['time'], dag_data['low_bid'], dag_data['high_bid'], color='gray', alpha=0.2)
        
        tekst_lijnen = [f"TRADING OVERZICHT: {datum_str}", "="*45]
        totale_winst_euro = 0
        start_balans_dag = None

        for i, row in dag_signals.sort_values('Signaal_Tijd').iterrows():
            tijd_str = row['Signaal_Tijd'].strftime('%H:%M')
            
            if row['Status'] == 'Pending/Canceled':
                ax.hlines(y=row['Entry_Prijs'], xmin=row['Signaal_Tijd'], xmax=row['Signaal_Tijd'] + pd.Timedelta(minutes=30), color='gray', linestyle=':', linewidth=2)
                ax.scatter(row['Signaal_Tijd'], row['Entry_Prijs'], color='gray', marker='x', s=100, zorder=5)
                tekst_lijnen.append(f"Tijd: {tijd_str} | [x] CANCELLED")
                tekst_lijnen.append("-" * 45)
            else:
                port_row = df_portfolio[df_portfolio['Entry_Tijd'] == row['Entry_Tijd']]
                
                if not port_row.empty:
                    port_data = port_row.iloc[0]
                    eenheden = port_data.get('Gekochte_Eenheden', 0)
                    pnl_euro = port_data.get('PnL_Euro', 0)
                    pnl_deel_1 = port_data.get('PnL_Deel_1', 0)
                    pnl_deel_2 = port_data.get('PnL_Deel_2', 0)
                    huidige_balans = port_data.get('Account_Balance', 100000)
                    
                    if start_balans_dag is None:
                        start_balans_dag = huidige_balans - pnl_euro
                    totale_winst_euro += pnl_euro

                    target_1 = row.get('Target_Prijs_1', row['Exit_Prijs_1'])
                    target_2 = row.get('Target_Prijs_2', row['Exit_Prijs_2'])

                    ax.hlines(y=target_1, xmin=row['Entry_Tijd'], xmax=row['Exit_Tijd_2'], colors='orange', linestyles=':', alpha=0.8, label='TP1 Doel' if i==0 else "")
                    ax.hlines(y=target_2, xmin=row['Entry_Tijd'], xmax=row['Exit_Tijd_2'], colors='blue', linestyles=':', alpha=0.8, label='TP2 Doel' if i==0 else "")
                    ax.hlines(y=row['Entry_Prijs'], xmin=row['Entry_Tijd'], xmax=row['Exit_Tijd_2'], colors='green', linestyles='-.', alpha=0.5, label='Entry' if i==0 else "")

                    ax.scatter(row['Entry_Tijd'], row['Entry_Prijs'], color='green', marker='^', s=150, zorder=6, edgecolors='black')
                    ax.scatter(row['Exit_Tijd_1'], row['Exit_Prijs_1'], color='orange', marker='o', s=100, zorder=6, edgecolors='black')
                    ax.scatter(row['Exit_Tijd_2'], row['Exit_Prijs_2'], color='blue', marker='v', s=150, zorder=6, edgecolors='black')
                    
                    ax.plot([row['Entry_Tijd'], row['Exit_Tijd_1']], [row['Entry_Prijs'], row['Exit_Prijs_1']], color='orange', linestyle='--', alpha=0.6)
                    ax.plot([row['Exit_Tijd_1'], row['Exit_Tijd_2']], [row['Exit_Prijs_1'], row['Exit_Prijs_2']], color='blue', linestyle='--', alpha=0.6)

                    inleg = eenheden * row['Entry_Prijs']
                    
                    tp1_pure_rit = ((row['Exit_Prijs_1'] - row['Entry_Prijs']) / row['Entry_Prijs']) * 100
                    
                    # Berekenen van de afzonderlijke procenten voor TP1 en TP2 op de inleg
                    pct_tp1 = (pnl_deel_1 / inleg) * 100 if inleg > 0 else 0
                    pct_tp2 = (pnl_deel_2 / inleg) * 100 if inleg > 0 else 0
                    
                    definitief_pct_inleg = (pnl_euro / inleg) * 100 if inleg > 0 else 0
                    definitief_pct_portfolio = definitief_pct_inleg / max_slots
                    
                    clean_res = row['Resultaat'].replace('✅', '').replace('❌', '').replace('⏳', '').replace('➖', '').strip()

                    tekst_lijnen.append(f"Tijd: {tijd_str} | {clean_res}")
                    tekst_lijnen.append(f"   -> Werkelijke Rit tp1: {tp1_pure_rit:+.3f}%")
                    tekst_lijnen.append(f"   -> Winst op tp1/tp2  : {pct_tp1:+.2f}% en {pct_tp2:+.2f}%")
                    tekst_lijnen.append(f"   -> Winst einde trade : {definitief_pct_inleg:+.2f}%")
                    tekst_lijnen.append(f"   -> Winst Portfolio   : {definitief_pct_portfolio:+.2f}% (1/{max_slots} size)")
                    tekst_lijnen.append(f"   -> Units: {eenheden}")
                    tekst_lijnen.append("-" * 45)

        # Dag samenvatting
        dag_pct_totaal = 0
        if start_balans_dag is not None and start_balans_dag > 0:
            dag_pct_totaal = (totale_winst_euro / start_balans_dag) * 100

        hypo_start = 100000.00
        hypo_eind = hypo_start * (1 + (dag_pct_totaal / 100))
        
        tekst_lijnen.append(f"Startbalans : Euro {hypo_start:,.0f}")
        tekst_lijnen.append(f"Eindbalans  : Euro {hypo_eind:,.0f}")
        tekst_lijnen.append(f"Dag Winst   : {dag_pct_totaal:+.2f}%")

        volledige_tekst = "\n".join(tekst_lijnen)
        plt.subplots_adjust(right=0.70) 
        fig.text(0.72, 0.5, volledige_tekst, fontsize=10, family='monospace', 
                 verticalalignment='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=1'))

        ax.set_title(f'Trade Verloop incl. Pending Orders & Limits - {datum_str}', fontsize=16)
        ax.set_xlabel('Tijd', fontsize=12)
        ax.set_ylabel('Prijs', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='x', rotation=45)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        file_path = os.path.join(output_dir, f"dag_{datum_str}_verloop.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig) 

# =============================================================
# 4. UITVOEREN & OPSLAAN
# =============================================================
if __name__ == "__main__":
    BESTE_KANS = 0.7  
    BESTE_DALING = 0.4    
    SCALE_OUT_MULTIPLIER = 0.5 
    BESTE_SAMPLES = 20      
    BESTE_SLOTS = 3         
    
    print("📥 Data ophalen en voorbereiden...")
    df_raw = fetch_crude_data()
    
    if df_raw is not None:
        print("⚙️ Features engineeren...")
        df_ready = prep_and_engineer_features(df_raw)
        
        print(f"🚀 Signalen genereren met Limit Order Logica (Kans {BESTE_KANS}, Dal {BESTE_DALING}, Scale-Out {SCALE_OUT_MULTIPLIER})...")
        df_signals = genereer_signalen_dynamisch(
            df_ready, 
            min_kans=BESTE_KANS, 
            min_samples=BESTE_SAMPLES, 
            max_kans_daling=BESTE_DALING,
            scale_out_multiplier=SCALE_OUT_MULTIPLIER
        )
        
        print("💼 Portfolio simuleren...")
        df_portfolio = run_portfolio_sim(df_signals, max_slots=BESTE_SLOTS)

        if not df_portfolio.empty:
            log_path = os.path.join(OUTPUT_DIR, "trading_log.csv")
            df_portfolio.to_csv(log_path, index=False)
            print(f"✅ Trading log opgeslagen in: {log_path}")

            final_score, sharpe_ann = bereken_zuivere_score(df_portfolio)
            df_portfolio['Account_Balance'] = 100000 + df_portfolio['PnL_Euro'].cumsum()

            plt.figure(figsize=(15, 8))
            plt.plot(df_portfolio['Entry_Tijd'], df_portfolio['Account_Balance'], color='#D4AF37', linewidth=2.5,
                     label=f'Pending Order Strategie (Kans: {BESTE_KANS} | Target 1: {SCALE_OUT_MULTIPLIER}x | Slots: {BESTE_SLOTS})')
            
            plt.axhline(100000, color='black', alpha=0.5, linestyle='--')
            plt.title('Equity Curve: Limit Order met Auto-Cancel', fontsize=16)
            plt.xlabel('Tijdlijn', fontsize=12)
            plt.ylabel('Kapitaal (€)', fontsize=12)
            
            ax = plt.gca()
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('€{x:,.0f}'))
            plt.grid(True, which='both', linestyle=':', alpha=0.6)
            plt.legend(loc='upper left')
            plt.xticks(rotation=30)

            eindbalans = df_portfolio['Account_Balance'].iloc[-1]
            winst = eindbalans - 100000
            stats_text = (f"Eindbalans: €{eindbalans:,.0f}\n"
                          f"Netto Winst: €{winst:,.0f}\n"
                          f"Gevulde Trades: {len(df_portfolio)}\n"
                          f"Ann. Sharpe: {sharpe_ann:.2f}")

            plt.text(0.02, 0.75, stats_text, transform=ax.transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='gold'))

            plt.tight_layout()
            
            equity_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
            plt.savefig(equity_path)
            print(f"✅ Equity curve opgeslagen als: {equity_path}")
            plt.close()
            
            print(f"📈 Visualiseren van alle trading dagen...")
            visualiseer_alle_dagen(df_ready, df_signals, df_portfolio, max_slots=BESTE_SLOTS)

            print("\n📊 Samenvatting Resultaten (alleen gevulde orders):")
            print(df_portfolio['Resultaat'].value_counts())
            
            gem_units = int(df_portfolio['Gekochte_Eenheden'].mean())
            print(f"📦 Gemiddeld aantal eenheden per trade: {gem_units}")
            print("--- KLAAR ---")
        else:
            print("❌ Geen trades gevuld met deze instellingen.")
