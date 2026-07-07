import os
import glob
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================================
# 🎛️ CENTRAL CONFIGURATION PANEL (AUTONOMOUS LAYERED Z-SCORE STRATEGY)
# =========================================================================
DATA_LIMIT = 5000           # Totaal aantal synchrone minuten om in te laden
RATIO_LOOKBACK = 240         # 4 uur rolling window voor basis-statistiek
MIN_EXPECTED_WIN_PCT = 0.10  # Minimale verwachte winst per instap-slot

# De 4 onafhankelijke instap-slots met bijbehorende Z-score drempels
SLOT_THRESHOLDS = {
    1: 1.5,
    2: 2.0,
    3: 2.5,
    4: 3
}

# REGIME FILTERS & RISICOMANAGEMENT
MAX_MEAN_SLOPE_LIMIT = 0.05  # Maximaal toegestane helling (%) van het 240m gemiddelde (30m delta)
MAX_DWELL_ENTRY_LIMIT = 15   # Maximaal toegestane plaktijd (minuten) buiten |Z|>=2.0 voor NIEUWE entries
CRITICAL_DWELL_EXIT = 30     # Harde exit als een ACTIEVE TRADE langer dan 60 min vastzit zonder convergentie

# Target mappenstructuur voor de schone start
BASE_RESULTS_DIR = os.path.join("Strategies", "results", "daily_analysis_z_score_strategy")

def load_and_prepare_raw_asset(folder_name):
    """Laadt de meest recente rauwe database rechtstreeks uit de asset-map."""
    search_pattern = os.path.join(folder_name, "outputs_merged_*.csv")
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise FileNotFoundError(f"❌ Geen rauwe CSV-data gevonden in map: {folder_name}")
    
    df = pd.read_csv(files[-1])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df['mid'] = (df['close_bid'] + df['close_ask']) / 2
    
    return df[['time', 'mid', 'close_bid', 'close_ask']].rename(
        columns={
            'mid': f'{folder_name}_price',
            'close_bid': f'{folder_name}_close_bid',
            'close_ask': f'{folder_name}_close_ask'
        }
    )

def run_layered_backtest():
    print("🚀 MANTRA Autonomous Layered Z-Score Engine Gestart...")
    
    # 1. Directe data alignment vanuit de rauwe bronbestanden
    try:
        us500 = load_and_prepare_raw_asset("US500")
        gold = load_and_prepare_raw_asset("GOLD")
    except Exception as e:
        print(f"❌ Data Ingestie Fout: {str(e)}")
        return

    print("🔗 Synchroniseren van de tijdsassen...")
    df = pd.merge(us500, gold, on='time', how='inner').sort_values('time').reset_index(drop=True)
    
    if len(df) > DATA_LIMIT:
        df = df.tail(DATA_LIMIT).reset_index(drop=True)

    # 2. Wiskundige berekeningen (Global rolling window voor continuïteit over de dagen)
    df['ratio'] = df['US500_price'] / df['GOLD_price']
    df['ratio_mean'] = df['ratio'].rolling(window=RATIO_LOOKBACK).mean()
    df['ratio_std'] = df['ratio'].rolling(window=RATIO_LOOKBACK).std()
    df['z_score'] = (df['ratio'] - df['ratio_mean']) / df['ratio_std']
    
    # DIAGNOSTISCHE INDICATOR 1: Helling (Slope) van de Ratio Mean (30m % verandering van de 240m basislijn)
    df['mean_slope_30m'] = (df['ratio_mean'] - df['ratio_mean'].shift(30)) / df['ratio_mean'].shift(30) * 100
    
    df = df.dropna(subset=['z_score', 'mean_slope_30m']).reset_index(drop=True)

    # DIAGNOSTISCHE INDICATOR 2: Z-Score Dwell Time (Opeenvolgende minuten buiten |Z| >= 2.0 voor instap-filtering)
    z_abs = df['z_score'].abs().values
    dwell = np.zeros(len(df))
    for k in range(1, len(df)):
        if z_abs[k] >= 2.0:
            dwell[k] = dwell[k-1] + 1
        else:
            dwell[k] = 0
    df['z_dwell_time'] = dwell

    # Zorg voor een gegarandeerd chronologische sortering van unieke datums
    df['date_str'] = df['time'].dt.strftime('%Y-%m-%d')
    unique_dates = sorted(df['date_str'].unique())
    
    # Bepaal dynamisch de twee meest recente datums in de dataset
    latest_date = unique_dates[-1] if len(unique_dates) > 0 else None
    previous_date = unique_dates[-2] if len(unique_dates) > 1 else None
    
    print(f"📅 Totaal aantal unieke handelsdagen in dataset: {len(unique_dates)}")
    print(f"🔄 Volatiele overschrijf-zones gedetecteerd: Laatste ({latest_date}) en Vorige ({previous_date})")

    # 3. Dag-voor-Dag Slimme Overschrijf Loop
    for target_date in unique_dates:
        day_output_dir = os.path.join(BASE_RESULTS_DIR, target_date)
        
        is_volatile = (target_date == latest_date or target_date == previous_date)
        
        if os.path.exists(day_output_dir):
            if is_volatile:
                print(f"♻️ [Hercalculatie] {target_date} gedetecteerd. Oude map wordt verwijderd en opnieuw berekend...")
                shutil.rmtree(day_output_dir, ignore_errors=True)
            else:
                print(f"⏩ [Overslaan] {target_date} is historische data (map blijft ongewijzigd staan).")
                continue
            
        day_df = df[df['date_str'] == target_date].copy().reset_index(drop=True)
        if len(day_df) < 10:  
            continue
            
        print(f"🔥 [Data Run] Analyse uitvoeren voor handelssessie: {target_date}...")
        
        max_slope = day_df['mean_slope_30m'].abs().max()
        max_dwell = day_df['z_dwell_time'].max()
        print(f"   📈 [Regime Diagnose] Max Base Mean Slope (30m): {max_slope:.4f}% | Max Z-Score Dwell Time: {max_dwell:.0f} min")

        os.makedirs(day_output_dir, exist_ok=True)

        active_slots = {}   
        current_regime = None  
        trades_log = []
        
        equity_curve_base = [0.0]
        equity_curve_10x = [0.0]
        
        # Minute-by-Minute Session Simulation
        for i in range(len(day_df)):
            row = day_df.iloc[i]
            curr_time = row['time'].time()
            z_curr = row['z_score']
            
            # REGIME CHECKS FASE
            is_slope_healthy = abs(row['mean_slope_30m']) <= MAX_MEAN_SLOPE_LIMIT
            is_dwell_healthy = row['z_dwell_time'] < MAX_DWELL_ENTRY_LIMIT
            
            is_inside_hours = time(4, 0) <= curr_time <= time(22, 0)
            # Nieuwe instappen mogen alleen plaatsvinden als het regime 'gezond' is
            can_open_new = (time(4, 0) <= curr_time <= time(20, 0)) and is_slope_healthy and is_dwell_healthy
            is_forced_close_time = curr_time >= time(22, 0) or i == (len(day_df) - 1)

            if current_regime is not None:
                hit_reversion = False
                hit_slope_stop = False
                hit_dwell_stop = False
                exit_reason = ""
                
                # A. Mean Reversion Convergence Check
                if current_regime == 'SHORT_PAIR' and z_curr <= 0:
                    hit_reversion = True
                    exit_reason = "MEAN_REVERSION_CONVERGENCE"
                elif current_regime == 'LONG_PAIR' and z_curr >= 0:
                    hit_reversion = True
                    exit_reason = "MEAN_REVERSION_CONVERGENCE"
                
                # B. 🔥 NIEUW: Regime Shift Slope Exit (Achterdeur Beveiliging tijdens lopende trade)
                if not hit_reversion and not is_slope_healthy:
                    hit_slope_stop = True
                    exit_reason = "REGIME_SHIFT_SLOPE_EXIT"
                
                # C. 🔥 NIEUW: Ononderbroken Trade Holding Exit (Voorkomt de Dwell Reset Trap)
                if not hit_reversion and not hit_slope_stop and len(active_slots) > 0:
                    oldest_entry_idx = min([s['entry_idx'] for s in active_slots.values()])
                    trade_duration_mins = i - oldest_entry_idx
                    if trade_duration_mins >= CRITICAL_DWELL_EXIT:
                        hit_dwell_stop = True
                        exit_reason = "CRITICAL_DWELL_TIME_EXCEEDED"
                
                # Afhandeling van alle Exits
                if hit_reversion or hit_slope_stop or hit_dwell_stop or is_forced_close_time:
                    reason = exit_reason if (hit_reversion or hit_slope_stop or hit_dwell_stop) else "FORCED_EOD_CLOSE"
                    
                    for slot_id, slot_data in active_slots.items():
                        if current_regime == 'SHORT_PAIR':
                            pct_us500 = ((slot_data['entry_us500'] - row['US500_close_ask']) / slot_data['entry_us500']) * 100
                            pct_gold = ((row['GOLD_close_bid'] - slot_data['entry_gold']) / slot_data['entry_gold']) * 100
                            exit_us500 = row['US500_close_ask']
                            exit_gold = row['GOLD_close_bid']
                        else:
                            pct_us500 = ((row['US500_close_bid'] - slot_data['entry_us500']) / slot_data['entry_us500']) * 100
                            pct_gold = ((slot_data['entry_gold'] - row['GOLD_close_ask']) / slot_data['entry_gold']) * 100
                            exit_us500 = row['US500_close_bid']
                            exit_gold = row['GOLD_close_ask']
                            
                        pnl_comb = (pct_us500 + pct_gold) / 2
                        
                        trades_log.append({
                            'slot': slot_id, 'type': current_regime,
                            'entry_time': slot_data['entry_time'], 'exit_time': row['time'],
                            'entry_idx': slot_data['entry_idx'], 'exit_idx': i,
                            'entry_us500': slot_data['entry_us500'], 'exit_us500': exit_us500,
                            'entry_gold': slot_data['entry_gold'], 'exit_gold': exit_gold,
                            'pct_us500': pct_us500, 'pct_gold': pct_gold,
                            'pnl_pct': pnl_comb, 'reason': reason
                        })
                        
                        cash_pnl_1x = pnl_comb / 4
                        equity_curve_base.append(equity_curve_base[-1] + cash_pnl_1x)
                        equity_curve_10x.append(equity_curve_10x[-1] + (cash_pnl_1x * 10))
                    
                    active_slots = {}
                    current_regime = None
                    if is_forced_close_time:
                        continue

                elif can_open_new:
                    expected_win = (abs(row['ratio'] - row['ratio_mean']) / row['ratio']) * 100 / 2
                    for slot_id, threshold in SLOT_THRESHOLDS.items():
                        if slot_id not in active_slots:
                            if current_regime == 'SHORT_PAIR' and z_curr >= threshold:
                                if expected_win >= MIN_EXPECTED_WIN_PCT:
                                    active_slots[slot_id] = {
                                        'entry_time': row['time'], 'entry_idx': i,
                                        'entry_us500': row['US500_close_bid'], 'entry_gold': row['GOLD_close_ask']
                                    }
                            elif current_regime == 'LONG_PAIR' and z_curr <= -threshold:
                                if expected_win >= MIN_EXPECTED_WIN_PCT:
                                    active_slots[slot_id] = {
                                        'entry_time': row['time'], 'entry_idx': i,
                                        'entry_us500': row['US500_close_ask'], 'entry_gold': row['GOLD_close_bid']
                                    }
            else:
                if can_open_new:
                    expected_win = (abs(row['ratio'] - row['ratio_mean']) / row['ratio']) * 100 / 2
                    if expected_win >= MIN_EXPECTED_WIN_PCT:
                        for slot_id, threshold in SLOT_THRESHOLDS.items():
                            if z_curr >= threshold:
                                current_regime = 'SHORT_PAIR'
                                active_slots[slot_id] = {
                                    'entry_time': row['time'], 'entry_idx': i,
                                    'entry_us500': row['US500_close_bid'], 'entry_gold': row['GOLD_close_ask']
                                }
                                break
                            elif z_curr <= -threshold:
                                current_regime = 'LONG_PAIR'
                                active_slots[slot_id] = {
                                    'entry_time': row['time'], 'entry_idx': i,
                                    'entry_us500': row['US500_close_ask'], 'entry_gold': row['GOLD_close_bid']
                                }
                                break

        # Rapportering wegschrijven
        trades_df = pd.DataFrame(trades_log)
        report_path = os.path.join(day_output_dir, "multi_backtest_report.md")
        with open(report_path, 'w') as f:
            f.write(f"# 📊 MANTRA: Layered Z-Score Session Report ({target_date})\n\n")
            f.write(f"* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID WITH ACTIVE REGIME EXITS`\n")
            f.write(f"* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)\n")
            f.write(f"* **Regime Control Thresholds:** Max Slope (`±{MAX_MEAN_SLOPE_LIMIT}%`) | Max Entry Dwell (`{MAX_DWELL_ENTRY_LIMIT}m`) | Max Trade Holding (`{CRITICAL_DWELL_EXIT}m`)\n")
            f.write(f"* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`\n\n")
            
            f.write(f"### 🔍 Trend vs. Mean-Reversion Regime Indicators\n")
            f.write(f"* **Max Ratio 240m Mean Slope (30m Delta):** `{max_slope:.4f}%`\n")
            f.write(f"* **Max Z-Score Dwell Time (|Z| >= 2.0):** `{max_dwell:.0f} minutes`\n\n")
            
            if len(trades_df) > 0:
                winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
                total_comb_pnl = trades_df['pnl_pct'].sum()
                avg_comb_pnl = trades_df['pnl_pct'].mean()
                net_portfolio_1x = total_comb_pnl / 4
                net_portfolio_10x = net_portfolio_1x * 10
                avg_slot_1x = avg_comb_pnl / 4
                avg_slot_10x = avg_slot_1x * 10
                
                f.write(f"### 📈 Session Key Performance Metrics\n")
                f.write(f"* **Total Scaled Batches Executed:** {len(trades_df)}\n")
                f.write(f"* **Batch Win Rate:** {(winning_trades / len(trades_df)) * 100:.2f}%\n")
                f.write(f"* **Pure Combination Trade Yield (Rauw Totaal):** {total_comb_pnl:.4f}%\n")
                f.write(f"* **Net Portfolio Session Yield (1x Base Portfolio):** {net_portfolio_1x:.4f}%\n")
                f.write(f"* **Net Portfolio Session Yield (10x Leveraged Portfolio):** **{net_portfolio_10x:.4f}%**\n")
                f.write(f"* **Average Yield per Executed Slot (1x Base Portfolio):** {avg_slot_1x:.4f}%\n")
                f.write(f"* **Average Yield per Executed Slot (10x Leveraged Portfolio):** {avg_slot_10x:.4f}%\n\n")
                
                f.write("### 📜 Session Transaction Ledger (Slot Decomposition)\n")
                f.write("| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for idx, r in trades_df.iterrows():
                    us500_pos = "SHORT" if "SHORT" in r['type'] else "LONG"
                    gold_pos = "LONG" if "SHORT" in r['type'] else "SHORT"
                    cash_pnl_1x = r['pnl_pct'] / 4
                    cash_pnl_10x = cash_pnl_1x * 10
                    f.write(f"| **Slot {r['slot']}** | {r['entry_time'].strftime('%H:%M')} | {r['exit_time'].strftime('%H:%M')} | "
                            f"`{us500_pos}` | {r['entry_us500']:.2f} | {r['exit_us500']:.2f} | {r['pct_us500']:.4f}% | "
                            f"`{gold_pos}` | {r['entry_gold']:.2f} | {r['exit_gold']:.2f} | {r['pct_gold']:.4f}% | "
                            f"**{r['pnl_pct']:.4f}%** | {cash_pnl_1x:.4f}% | **{cash_pnl_10x:.4f}%** | `{r['reason']}` |\n")
            else:
                f.write("### 📭 Session Report\nNo layered arbitrage boundaries were hit within active market hours today.")

        if len(trades_df) > 0:
            plt.figure(figsize=(11, 5.5))
            plt.plot(range(len(equity_curve_base)), equity_curve_base, color='purple', linewidth=2, marker='o', label='Cash PnL (1x Base Portfolio) (%)')
            plt.plot(range(len(equity_curve_10x)), equity_curve_10x, color='#e65c00', linewidth=2, marker='s', linestyle='--', label='Cash PnL (10x Leveraged Portfolio) (%)')
            plt.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.title(f"MANTRA Session Capital Growth Curve: {target_date} (True Portfolio Return)", fontsize=11, fontweight='bold', loc='left')
            plt.xlabel("Sequence of Closed Batches")
            plt.ylabel("Portfolio Cash Return (%)")
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(loc="upper left", frameon=True, shadow=True)
            plt.tight_layout()
            plt.savefig(os.path.join(day_output_dir, "multi_backtest_chart.png"), dpi=300)
            plt.close()

            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
            
            # 1. US500
            ax1.plot(day_df.index, day_df['US500_price'], color='#1f78b4', alpha=0.5, label='US500 Mid')
            ax1.set_ylabel("US500 ($)")
            ax1.grid(True, linestyle=':', alpha=0.3)
            ax1.set_title(f"MANTRA Real-time Session Dashboard: {target_date}", fontsize=12, fontweight='bold', loc='left')
            
            # 2. GOLD
            ax2.plot(day_df.index, day_df['GOLD_price'], color='#ffd700', alpha=0.6, label='GOLD Mid')
            ax2.set_ylabel("GOLD ($)")
            ax2.grid(True, linestyle=':', alpha=0.3)
            
            # 3. Z-SCORE
            ax3.plot(day_df.index, day_df['z_score'], color='#6a3d9a', alpha=0.8, label='Z-Score')
            ax3.axhline(0, color='black', linestyle='-', alpha=0.4)
            for s_id, thresh in SLOT_THRESHOLDS.items():
                ax3.axhline(thresh, color='red', linestyle='--', alpha=0.3)
                ax3.axhline(-thresh, color='red', linestyle='--', alpha=0.3)
            ax3.set_ylabel("Z-Score")
            ax3.grid(True, linestyle=':', alpha=0.3)

            # 4. HELLING VAN DE RATIO MEAN MET DREMPELLIJNEN
            ax4.plot(day_df.index, day_df['mean_slope_30m'], color='#e31a1c', linewidth=1.5, label='240m Mean Slope (30m % Change)')
            ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax4.axhline(MAX_MEAN_SLOPE_LIMIT, color='red', linestyle='--', alpha=0.7, label=f'Healthy Bound (+{MAX_MEAN_SLOPE_LIMIT}%)')
            ax4.axhline(-MAX_MEAN_SLOPE_LIMIT, color='red', linestyle='--', alpha=0.7, label=f'Healthy Bound (-{MAX_MEAN_SLOPE_LIMIT}%)')
            ax4.set_ylabel("Slope (%)")
            ax4.grid(True, linestyle=':', alpha=0.3)
            ax4.legend(loc="upper left")

            # 5. Z-SCORE DWELL TIME MET DREMPELLIJNEN
            ax5.plot(day_df.index, day_df['z_dwell_time'], color='#33a02c', linewidth=1.5, label='Z-Score Dwell Time (|Z| >= 2.0)')
            ax5.axhline(MAX_DWELL_ENTRY_LIMIT, color='orange', linestyle='--', alpha=0.8, label=f'Entry Block ({MAX_DWELL_ENTRY_LIMIT}m)')
            ax5.axhline(CRITICAL_DWELL_EXIT, color='red', linestyle='--', alpha=0.8, label=f'Trade Max Hold ({CRITICAL_DWELL_EXIT}m)')
            ax5.set_ylabel("Minutes")
            ax5.grid(True, linestyle=':', alpha=0.3)
            ax5.legend(loc="upper left")

            for t in trades_log:
                e_idx = t['entry_idx']
                x_idx = t['exit_idx']
                for ax in (ax1, ax2, ax3, ax4, ax5):
                    ax.axvspan(e_idx, x_idx, color='purple', alpha=0.05)
                
                if t['type'] == 'LONG_PAIR':
                    ax1.scatter(e_idx, t['entry_us500'], color='green', marker='^', s=80, zorder=5)
                    ax2.scatter(e_idx, t['entry_gold'], color='red', marker='v', s=80, zorder=5)
                else:
                    ax1.scatter(e_idx, t['entry_us500'], color='red', marker='v', s=80, zorder=5)
                    ax2.scatter(e_idx, t['entry_gold'], color='green', marker='^', s=80, zorder=5)
            
            num_ticks = 6
            tick_indices = np.linspace(0, len(day_df) - 1, num_ticks, dtype=int)
            plt.xticks(tick_indices, day_df['time'].dt.strftime('%H:%M').iloc[tick_indices].values, rotation=0)
            plt.xlabel("Timeline (Session Trading Minutes)")
            plt.tight_layout()
            plt.savefig(os.path.join(day_output_dir, "multi_execution_chart.png"), dpi=300)
            plt.close()
            
        print(f"✅ Handelsdag {target_date} succesvol berekend met Actieve Regime Exits.")

if __name__ == '__main__':
    run_layered_backtest()
