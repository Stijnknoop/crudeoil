import os
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Cruciaal voor GitHub Actions
import matplotlib.pyplot as plt

# =============================================================
# 0. CONFIGURATIE VOOR GITHUB (De nieuwe strakke paden!)
# =============================================================
OUTPUT_DIR = "OIL_CRUDE/strategy_feb/TradingDetails"
DAILY_PLOTS_DIR = os.path.join(OUTPUT_DIR, "DailyPlots")

# Zorg dat beide mappen bestaan als het script draait
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
                
                target_full = c_prijs_ask + (beste_multiplier * actuele_rij['current_range'])
                target_1 = c_prijs_ask + (scale_out_multiplier * beste_multiplier * actuele_rij['current_range'])
                
                data_na = sessie_data[sessie_data['time'] > actuele_rij['time']]

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
                        'Sessie': huidige_sessie, 'Entry_Tijd': actuele_rij['time'], 
                        'Entry_Prijs': c_prijs_ask, 
                        'Exit_Tijd_1': exit_t_1, 'Exit_Prijs_1': exit_p_1,
                        'Exit_Tijd_2': exit_t_2, 'Exit_Prijs_2': exit_p_2,
                        'Multiplier': beste_multiplier, 'Resultaat': res
                    })
    
    return pd.DataFrame(alle_trades)

def run_portfolio_sim(df_trades, max_slots, start_kapitaal=100000, contract_multiplier=1):
    if df_trades.empty: return pd.DataFrame()

    huidige_balans = start_kapitaal
    actieve_trades = []
    geaccepteerde_trades = []
    df_exact = df_trades.sort_values('Entry_Tijd').reset_index(drop=True)

    for _, trade in df_exact.iterrows():
        nog_actief = []
        for ot in actieve_trades:
            if trade['Entry_Tijd'] >= ot['exit_time']: 
                huidige_balans += ot['pnl']
            else: 
                nog_actief.append(ot)
        actieve_trades = nog_actief

        if len(actieve_trades) < max_slots:
            inleg = huidige_balans / max_slots
            eenheden = inleg / trade['Entry_Prijs']
            
            pnl_deel_1 = (trade['Exit_Prijs_1'] - trade['Entry_Prijs']) * (eenheden * 0.5) * contract_multiplier
            pnl_deel_2 = (trade['Exit_Prijs_2'] - trade['Entry_Prijs']) * (eenheden * 0.5) * contract_multiplier
            totale_pnl = pnl_deel_1 + pnl_deel_2
            
            actieve_trades.append({'exit_time': trade['Exit_Tijd_2'], 'pnl': totale_pnl})
            
            trade_data = trade.copy()
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
def visualiseer_sessie(session_id, df_data, df_trades, output_dir=DAILY_PLOTS_DIR):
    sessie_data = df_data[df_data['session_id'] == session_id]
    
    if sessie_data.empty:
        print(f"❌ Sessie {session_id} niet gevonden in de dataset.")
        return
        
    sessie_trades = df_trades[df_trades['Sessie'] == session_id].copy()
    
    plt.figure(figsize=(14, 7))
    plt.plot(sessie_data['time'], sessie_data['close_bid'], label='Close Prijs', color='black', linewidth=1.5)
    plt.fill_between(sessie_data['time'], sessie_data['low_bid'], sessie_data['high_bid'], color='gray', alpha=0.2, label='High/Low Range')
    
    if not sessie_trades.empty:
        sessie_trades['Gemiddelde_Exit'] = (sessie_trades['Exit_Prijs_1'] + sessie_trades['Exit_Prijs_2']) / 2
        sessie_trades['Winst_pct_num'] = ((sessie_trades['Gemiddelde_Exit'] - sessie_trades['Entry_Prijs']) / sessie_trades['Entry_Prijs']) * 100
        totale_sessie_winst = sessie_trades['Winst_pct_num'].sum()
        sessie_trades['Winst_%'] = sessie_trades['Winst_pct_num'].round(3).astype(str) + '%'
        
        for i, trade in sessie_trades.iterrows():
            is_first = (i == sessie_trades.index[0])
            plt.scatter(trade['Entry_Tijd'], trade['Entry_Prijs'], color='green', marker='^', s=150, zorder=5, edgecolors='black', label='Entry' if is_first else "")
            plt.scatter(trade['Exit_Tijd_1'], trade['Exit_Prijs_1'], color='orange', marker='o', s=100, zorder=5, edgecolors='black', label='Exit 1 (50%)' if is_first else "")
            
            kleur_exit2 = 'green' if '✅✅' in trade['Resultaat'] else ('blue' if '➖' in trade['Resultaat'] else 'red')
            plt.scatter(trade['Exit_Tijd_2'], trade['Exit_Prijs_2'], color=kleur_exit2, marker='v', s=150, zorder=5, edgecolors='black', label='Exit 2 (Restant)' if is_first else "")
            
            plt.plot([trade['Entry_Tijd'], trade['Exit_Tijd_1']], [trade['Entry_Prijs'], trade['Exit_Prijs_1']], color='orange', linestyle='--', alpha=0.6)
            plt.plot([trade['Exit_Tijd_1'], trade['Exit_Tijd_2']], [trade['Exit_Prijs_1'], trade['Exit_Prijs_2']], color=kleur_exit2, linestyle='--', alpha=0.6)
            
        print(f"📊 {len(sessie_trades)} trade(s) gevonden in sessie {session_id}.")
        print(f"📈 Totale winst/verlies van de dag: {totale_sessie_winst:.3f}%\n")
        
    else:
        print(f"ℹ️ Geen trades uitgevoerd in sessie {session_id}.")
        
    plt.title(f'Trade Verloop - Sessie {session_id}', fontsize=16)
    plt.xlabel('Tijd', fontsize=12)
    plt.ylabel('Prijs', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, f"sessie_{session_id}_verloop.png")
    plt.savefig(file_path)
    print(f"✅ Sessie-grafiek (Daily Plot) opgeslagen als: {file_path}")
    plt.close()

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
        
        print(f"🚀 Signalen genereren met 2-Staps Exit (Kans {BESTE_KANS}, Dal {BESTE_DALING}, Scale-Out {SCALE_OUT_MULTIPLIER})...")
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
            # 1. Trading Log opslaan in TradingDetails map
            log_path = os.path.join(OUTPUT_DIR, "trading_log.csv")
            df_portfolio.to_csv(log_path, index=False)
            print(f"✅ Trading log opgeslagen in: {log_path}")

            # 2. Score berekenen
            final_score, sharpe_ann = bereken_zuivere_score(df_portfolio)
            df_portfolio['Account_Balance'] = 100000 + df_portfolio['PnL_Euro'].cumsum()

            # 3. Equity Curve Plotten & Opslaan in TradingDetails map
            plt.figure(figsize=(15, 8))
            plt.plot(df_portfolio['Entry_Tijd'], df_portfolio['Account_Balance'], color='#D4AF37', linewidth=2.5,
                     label=f'Scale-Out Strategie (Kans: {BESTE_KANS} | Target 1: {SCALE_OUT_MULTIPLIER}x | Slots: {BESTE_SLOTS})')
            
            plt.axhline(100000, color='black', alpha=0.5, linestyle='--')
            plt.title('Equity Curve: Scale-Out & Breakeven Stop', fontsize=16)
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
                          f"Aantal Trades: {len(df_portfolio)}\n"
                          f"Ann. Sharpe: {sharpe_ann:.2f}")

            plt.text(0.02, 0.75, stats_text, transform=ax.transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='gold'))

            plt.tight_layout()
            
            equity_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
            plt.savefig(equity_path)
            print(f"✅ Equity curve opgeslagen als: {equity_path}")
            plt.close()
            
            # 4. De LAATSTE sessie plotten en opslaan (in DailyPlots submap!)
            laatste_sessie_id = df_ready['session_id'].max()
            print(f"📈 Visualiseren van laatste sessie ({laatste_sessie_id})...")
            visualiseer_sessie(laatste_sessie_id, df_ready, df_portfolio)

            print("\n📊 Samenvatting Resultaten:")
            print(df_portfolio['Resultaat'].value_counts())
            print("--- KLAAR ---")
        else:
            print("❌ Geen trades gevonden met deze instellingen. (Geen bestanden opgeslagen)")
