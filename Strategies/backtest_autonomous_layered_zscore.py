import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================================
# 🎛️ CENTRAL CONFIGURATION PANEL (AUTONOMOUS LAYERED Z-SCORE STRATEGY)
# =========================================================================
DATA_LIMIT = 10000           # Totaal aantal synchrone minuten om in te laden
RATIO_LOOKBACK = 240         # 4 uur rolling window voor basis-statistiek
MIN_EXPECTED_WIN_PCT = 0.10  # Minimale verwachte winst per instap-slot

# De 4 onafhankelijke instap-slots met bijbehorende Z-score drempels
SLOT_THRESHOLDS = {
    1: 1.5,
    2: 2.0,
    3: 2.5,
    4: 3.0
}

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
    df = df.dropna(subset=['z_score']).reset_index(drop=True)
    
    df['date_str'] = df['time'].dt.strftime('%Y-%m-%d')
    unique_dates = df['date_str'].unique()
    
    print(f"📅 Totaal aantal unieke handelsdagen gedetecteerd: {len(unique_dates)}")

    # 3. Dag-voor-Dag Simulatie Loop
    for target_date in unique_dates:
        day_df = df[df['date_str'] == target_date].copy().reset_index(drop=True)
        if len(day_df) < 10:  
            continue
            
        print(f"📊 Analyseren van handelssessie: {target_date}...")
        
        # Maak de specifieke dagmap aan binnen de resultatenstructuur
        day_output_dir = os.path.join(BASE_RESULTS_DIR, target_date)
        os.makedirs(day_output_dir, exist_ok=True)

        active_slots = {}   # Volgt welke van de 4 slots open staan
        current_regime = None  # 'SHORT_PAIR' of 'LONG_PAIR'
        trades_log = []
        equity_curve = [0.0]
        
        # Minute-by-Minute Session Simulation
        for i in range(len(day_df)):
            row = day_df.iloc[i]
            curr_time = row['time'].time()
            z_curr = row['z_score']
            
            # Tijdswindows afdwingen volgens specificaties
            is_inside_hours = time(4, 0) <= curr_time <= time(22, 0)
            can_open_new = time(4, 0) <= curr_time <= time(20, 0)
            is_forced_close_time = curr_time >= time(22, 0) or i == (len(day_df) - 1)

            # -----------------------------------------------------------------
            # EVALUATIE ACTIEVE REGIME (Check Exits, Sluitingen & Opschaling)
            # -----------------------------------------------------------------
            if current_regime is not None:
                # Conditie 1: Volledig statistisch herstel bereikt (Z-score tikt de 0 aan/kruist de 0)
                hit_reversion = False
                if current_regime == 'SHORT_PAIR' and z_curr <= 0:
                    hit_reversion = True
                    exit_reason = "MEAN_REVERSION_CONVERGENCE"
                elif current_regime == 'LONG_PAIR' and z_curr >= 0:
                    hit_reversion = True
                    exit_reason = "MEAN_REVERSION_CONVERGENCE"
                
                # Conditie 2: Harde parachute sluiting aan het einde van de handelsdag (22:00)
                if hit_reversion or is_forced_close_time:
                    reason = exit_reason if hit_reversion else "FORCED_EOD_CLOSE"
                    
                    # Sluit alle momenteel actieve slots tegelijkertijd af
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
                        equity_curve.append(equity_curve[-1] + pnl_comb)
                    
                    # Reset sessie-parameters na volledige exit
                    active_slots = {}
                    current_regime = None
                    if is_forced_close_time:
                        continue

                # Conditie 3: Indien nog binnen de openingsuren (< 20:00), check voor verdere opschaling/overlap
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

            # -----------------------------------------------------------------
            # GEEN ACTIEVE POSITIES (Wacht op initiële trigger van Slot 1 t/m 4)
            # -----------------------------------------------------------------
            else:
                if can_open_new:
                    expected_win = (abs(row['ratio'] - row['ratio_mean']) / row['ratio']) * 100 / 2
                    
                    if expected_win >= MIN_EXPECTED_WIN_PCT:
                        # Check welk instap-niveau als eerste geraakt wordt
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

        # ---------------------------------------------------------------------
        # 📝 RAPPORTERING PER DAG (.MD LEDGER MET CASH EN MULTIPLIER KOLOMMEN)
        # ---------------------------------------------------------------------
        trades_df = pd.DataFrame(trades_log)
        report_path = os.path.join(day_output_dir, "multi_backtest_report.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# 📊 MANTRA: Layered Z-Score Session Report ({target_date})\n\n")
            f.write(f"* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID`\n")
            f.write(f"* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)\n")
            f.write(f"* **Operational Windows:** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`\n\n")
            
            if len(trades_df) > 0:
                winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
                f.write(f"### 📈 Session Key Performance Metrics\n")
                f.write(f"* **Total Scaled Batches Executed:** {len(trades_df)}\n")
                f.write(f"* **Batch Win Rate:** {(winning_trades / len(trades_df)) * 100:.2f}%\n")
                f.write(f"* **Net Session Yield (Total Capital ROI):** {trades_df['pnl_pct'].sum():.4f}%\n")
                f.write(f"* **Average Yield per Executed Slot:** {trades_df['pnl_pct'].mean():.4f}%\n\n")
                
                f.write("### 📜 Session Transaction Ledger (Slot Decomposition)\n")
                # 🔥 NIEUW: Toegevoegde kolommen voor Cash PnL (1x) en Cash PnL (10x Leverage)
                f.write("| Slot | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Cash PnL (1x) | Cash PnL (10x Leverage) | Reason |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                
                for idx, r in trades_df.iterrows():
                    us500_pos = "SHORT" if "SHORT" in r['type'] else "LONG"
                    gold_pos = "LONG" if "SHORT" in r['type'] else "SHORT"
                    
                    # Bereken de impact op de cashflow op basis van je 4 kapitaal-slots
                    cash_pnl_1x = r['pnl_pct'] / 4
                    cash_pnl_10x = cash_pnl_1x * 10
                    
                    f.write(f"| **Slot {r['slot']}** | {r['entry_time'].strftime('%H:%M')} | {r['exit_time'].strftime('%H:%M')} | "
                            f"`{us500_pos}` | {r['entry_us500']:.2f} | {r['exit_us500']:.2f} | {r['pct_us500']:.4f}% | "
                            f"`{gold_pos}` | {r['entry_gold']:.2f} | {r['exit_gold']:.2f} | {r['pct_gold']:.4f}% | "
                            f"**{r['pnl_pct']:.4f}%** | {cash_pnl_1x:.4f}% | **{cash_pnl_10x:.4f}%** | `{r['reason']}` |\n")
            else:
                f.write("### 📭 Session Report\nNo layered arbitrage boundaries were hit within active market hours today.")

        if len(trades_df) > 0:
            # Grafiek 1: Cumulative Session growth Curve
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(equity_curve)), equity_curve, color='purple', linewidth=2, marker='o', label='Session Layered ROI (%)')
            plt.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.title(f"MANTRA Session Curve: {target_date} (Net Return in %)", fontsize=11, fontweight='bold', loc='left')
            plt.xlabel("Sequence of Closed Batches")
            plt.ylabel("Combined Return (%)")
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(day_output_dir, "multi_backtest_chart.png"), dpi=300)
            plt.close()

            # Grafiek 2: 3-Panel Session Execution Dashboard
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
            ax1.plot(day_df.index, day_df['US500_price'], color='#1f78b4', alpha=0.5, label='US500 Mid')
            ax2.plot(day_df.index, day_df['GOLD_price'], color='#ffd700', alpha=0.6, label='GOLD Mid')
            ax3.plot(day_df.index, day_df['z_score'], color='#6a3d9a', alpha=0.8, label='Z-Score')
            
            ax3.axhline(0, color='black', linestyle='-', alpha=0.4)
            for s_id, thresh in SLOT_THRESHOLDS.items():
                ax3.axhline(thresh, color='red', linestyle='--', alpha=0.3)
                ax3.axhline(-thresh, color='red', linestyle='--', alpha=0.3)
            
            for t in trades_log:
                e_idx = t['entry_idx']
                x_idx = t['exit_idx']
                ax1.axvspan(e_idx, x_idx, color='purple', alpha=0.05)
                ax2.axvspan(e_idx, x_idx, color='purple', alpha=0.05)
                ax3.axvspan(e_idx, x_idx, color='purple', alpha=0.05)
                
                # Markeer instappen op de prijslijnen
                if t['type'] == 'LONG_PAIR':
                    ax1.scatter(e_idx, t['entry_us500'], color='green', marker='^', s=80, zorder=5)
                    ax2.scatter(e_idx, t['entry_gold'], color='red', marker='v', s=80, zorder=5)
                else:
                    ax1.scatter(e_idx, t['entry_us500'], color='red', marker='v', s=80, zorder=5)
                    ax2.scatter(e_idx, t['entry_gold'], color='green', marker='^', s=80, zorder=5)

            ax1.set_ylabel("US500 ($)")
            ax1.grid(True, linestyle=':', alpha=0.3)
            ax1.set_title(f"MANTRA Real-time Session Dashboard: {target_date}", fontsize=12, fontweight='bold', loc='left')
            
            ax2.set_ylabel("GOLD ($)")
            ax2.grid(True, linestyle=':', alpha=0.3)
            
            ax3.set_ylabel("Z-Score")
            ax3.grid(True, linestyle=':', alpha=0.3)
            
            num_ticks = 6
            tick_indices = np.linspace(0, len(day_df) - 1, num_ticks, dtype=int)
            plt.xticks(tick_indices, day_df['time'].dt.strftime('%H:%M').iloc[tick_indices].values, rotation=0)
            plt.xlabel("Timeline (Session Trading Minutes)")
            
            plt.tight_layout()
            plt.savefig(os.path.join(day_output_dir, "multi_execution_chart.png"), dpi=300)
            plt.close()
            
        print(f"✅ Sessie {target_date} succesvol verwerkt en opgeslagen.")

if __name__ == '__main__':
    run_layered_backtest()
