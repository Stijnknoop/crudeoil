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
SLOT_THRESHOLDS = {1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0}

# 🛡️ INSTITUTIONEEL PORTFOLIO RISICOMANAGEMENT (Berekend op 1x Base Portfolio Cash)
PORTFOLIO_STOP_LOSS_PCT = -0.25    # Harde stop: grijpt in bij -2.5% op je 10x leveraged account
TRAIL_ACTIVATION_PCT = 0.15       # Activatie: Schaduw-stop start bij +1.5% op je 10x leveraged account
TRAIL_DROP_PCT = 0.04             # Winstbeveiliging: Sluit als winst 0.4% daalt vanaf de piek (10x scale)

BASE_RESULTS_DIR = os.path.join("Strategies", "results", "daily_analysis_z_score_strategy")

def load_and_prepare_raw_asset(folder_name):
    search_pattern = os.path.join(folder_name, "outputs_merged_*.csv")
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise FileNotFoundError(f"❌ Geen rauwe CSV-data gevonden in map: {folder_name}")
    df = pd.read_csv(files[-1])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df['mid'] = (df['close_bid'] + df['close_ask']) / 2
    return df[['time', 'mid', 'close_bid', 'close_ask']].rename(
        columns={'mid': f'{folder_name}_price', 'close_bid': f'{folder_name}_close_bid', 'close_ask': f'{folder_name}_close_ask'}
    )

def run_layered_backtest():
    print("🚀 MANTRA Autonomous Layered Z-Score Engine Gestart...")
    try:
        us500 = load_and_prepare_raw_asset("US500")
        gold = load_and_prepare_raw_asset("GOLD")
    except Exception as e:
        print(f"❌ Data Ingestie Fout: {str(e)}")
        return

    df = pd.merge(us500, gold, on='time', how='inner').sort_values('time').reset_index(drop=True)
    if len(df) > DATA_LIMIT:
        df = df.tail(DATA_LIMIT).reset_index(drop=True)

    try:
        df['time_ny'] = df['time'].dt.tz_localize('Europe/Amsterdam', ambiguous='NaT').dt.tz_convert('America/New_York')
    except:
        df['time_ny'] = df['time'].dt.tz_localize(None).dt.tz_localize('Europe/Amsterdam').dt.tz_convert('America/New_York')

    df['ratio'] = df['US500_price'] / df['GOLD_price']
    df['ratio_mean'] = df['ratio'].rolling(window=RATIO_LOOKBACK).mean()
    df['ratio_std'] = df['ratio'].rolling(window=RATIO_LOOKBACK).std()
    df['z_score'] = (df['ratio'] - df['ratio_mean']) / df['ratio_std']
    df = df.dropna(subset=['z_score']).reset_index(drop=True)
    
    df['date_str'] = df['time'].dt.strftime('%Y-%m-%d')
    unique_dates = sorted(df['date_str'].unique())
    latest_date = unique_dates[-1] if len(unique_dates) > 0 else None
    previous_date = unique_dates[-2] if len(unique_dates) > 1 else None

    for target_date in unique_dates:
        day_output_dir = os.path.join(BASE_RESULTS_DIR, target_date)
        is_volatile = (target_date == latest_date or target_date == previous_date)
        
        if os.path.exists(day_output_dir):
            if is_volatile:
                shutil.rmtree(day_output_dir, ignore_errors=True)
            else:
                continue
            
        day_df = df[df['date_str'] == target_date].copy().reset_index(drop=True)
        if len(day_df) < 10:  
            continue
            
        os.makedirs(day_output_dir, exist_ok=True)
        active_slots = {}   
        current_regime = None  
        trades_log = []
        
        equity_curve_base = [0.0]
        equity_curve_10x = [0.0]
        
        # Volgparameters voor de winstbeveiliging (trailing stop)
        peak_unrealized_pnl = -999.0
        
        for i in range(len(day_df)):
            row = day_df.iloc[i]
            curr_time_nl = row['time'].time()
            curr_time_ny = row['time_ny'].time()
            z_curr = row['z_score']
            
            is_inside_hours = time(4, 0) <= curr_time_nl <= time(22, 0)
            can_open_new = time(4, 0) <= curr_time_nl <= time(20, 0)
            is_forced_close_time = curr_time_nl >= time(22, 0) or i == (len(day_df) - 1)
            
            is_us_open_danger_zone = time(8, 30) <= curr_time_ny <= time(10, 30)
            if is_us_open_danger_zone:
                can_open_new = False

            # 🛑 REALTIME PORTFOLIO PNLEVALUATIE (Risk Management Monitor)
            total_unrealized_pnl = 0.0
            if current_regime is not None and len(active_slots) > 0:
                for s_id, s_data in active_slots.items():
                    if current_regime == 'SHORT_PAIR':
                        p_us = ((s_data['entry_us500'] - row['US500_close_ask']) / s_data['entry_us500']) * 100
                        p_au = ((row['GOLD_close_bid'] - s_data['entry_gold']) / s_data['entry_gold']) * 100
                    else:
                        p_us = ((row['US500_close_bid'] - s_data['entry_us500']) / s_data['entry_us500']) * 100
                        p_au = ((s_data['entry_gold'] - row['GOLD_close_ask']) / s_data['entry_gold']) * 100
                    total_unrealized_pnl += ((p_us + p_au) / 2) / 4 # Gewogen naar totale portfolio cash (1/4)
                
                # Update de hoogst gemeten winstpiek van deze lopende sessie
                if total_unrealized_pnl > peak_unrealized_pnl:
                    peak_unrealized_pnl = total_unrealized_pnl

            # -----------------------------------------------------------------
            # EVALUATIE EN EXIT TRIGGERS
            # -----------------------------------------------------------------
            if current_regime is not None:
                hit_reversion = False
                exit_reason = ""
                
                # Trigger 1: Regulier statistisch herstel
                if current_regime == 'SHORT_PAIR' and z_curr <= 0:
                    hit_reversion = True
                    exit_reason = "MEAN_REVERSION_CONVERGENCE"
                elif current_regime == 'LONG_PAIR' and z_curr >= 0:
                    hit_reversion = True
                    exit_reason = "MEAN_REVERSION_CONVERGENCE"
                
                # Trigger 2: 🔥 NIEUW - Harde Portefeuille Noodrem (Stop-Loss)
                if total_unrealized_pnl <= PORTFOLIO_STOP_LOSS_PCT:
                    hit_reversion = True
                    exit_reason = "PORTFOLIO_HARD_STOP_LOSS"
                    
                # Trigger 3: 🔥 NIEUW - Trailing Profit Snatcher (Winsten Beveiligen)
                elif peak_unrealized_pnl >= TRAIL_ACTIVATION_PCT and total_unrealized_pnl <= (peak_unrealized_pnl - TRAIL_DROP_PCT):
                    hit_reversion = True
                    exit_reason = "PORTFOLIO_TRAILING_TAKE_PROFIT"
                
                if hit_reversion or is_forced_close_time:
                    reason = exit_reason if hit_reversion else "FORCED_EOD_CLOSE"
                    
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
                    peak_unrealized_pnl = -999.0 # Reset piek voor de volgende ronde
                    if is_forced_close_time:
                        continue

                elif can_open_new: # Grid mag nu weer vrijuit ademen en slots laden!
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

        # Ledger genereren
        trades_df = pd.DataFrame(trades_log)
        report_path = os.path.join(day_output_dir, "multi_backtest_report.md")
        with open(report_path, 'w') as f:
            f.write(f"# 📊 MANTRA: Layered Z-Score Session Report ({target_date})\n\n")
            f.write(f"* **Strategy Architecture:** `PURE MATHEMATICAL MULTI-SLOT GRID`\n")
            f.write(f"* **Configured Slot Thresholds:** Slot 1 (`1.5`), Slot 2 (`2.0`), Slot 3 (`2.5`), Slot 4 (`3.0`)\n")
            f.write(f"* **Risk Regulators:** Trailing Take Profit active ({TRAIL_ACTIVATION_PCT}%) | Emergency Portfolio Stop active ({PORTFOLIO_STOP_LOSS_PCT}%)\n")
            f.write(f"* **Operational Windows (NL):** Entries `04:00 - 20:00` | Forced Hard EOD Close `22:00`\n\n")
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
            
        print(f"✅ Handelsdag {target_date} berekend met herstelde grid-capaciteit en actieve trailing-uitstappen.")

if __name__ == '__main__':
    run_layered_backtest()
