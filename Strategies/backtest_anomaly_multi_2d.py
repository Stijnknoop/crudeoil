import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================================
# 🎛️ CENTRAL CONFIGURATION PANEL (AI-FILTERED STATISTICAL ARBITRAGE)
# =========================================================================
DATA_LIMIT = 5000         
RATIO_LOOKBACK = 240       
Z_THRESHOLD = 1          # Met de AI-veiligheidsgordel om kunnen we eerder instappen (1.5)

# Minimale wiskundig verwachte winst in % om de trade te accepteren
MIN_EXPECTED_WIN_PCT = 0.20  

MAX_DURATION = 60         # Parachute: Harde maximale duration timeout in minuten

# Mappenstructuur
RESULT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_multi_2d")
INPUT_CSV = os.path.join(RESULT_DIR, "multi_asset_2d_analyzed_data.csv")
OUTPUT_REPORT = os.path.join(RESULT_DIR, "multi_backtest_report.md")

# Output grafieken
OUTPUT_CHART_ROI = os.path.join(RESULT_DIR, "multi_backtest_chart.png")       
OUTPUT_CHART_EXEC = os.path.join(RESULT_DIR, "multi_execution_chart.png")    

def run_ai_backtest():
    print(f"🚀 MANTRA AI-Filtered Arbitrage Engine Gestart... [AI-Filter: AAN]")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Fout: {INPUT_CSV} ontbreekt. Run eerst de multi ML engine!")
        return

    df = pd.read_csv(INPUT_CSV)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    if len(df) > DATA_LIMIT:
        df = df.tail(DATA_LIMIT).reset_index(drop=True)

    # Wiskundige Spread & Ratio Berekening
    df['ratio'] = df['US500_price'] / df['GOLD_price']
    df['ratio_mean'] = df['ratio'].rolling(window=RATIO_LOOKBACK).mean()
    df['ratio_std'] = df['ratio'].rolling(window=RATIO_LOOKBACK).std()
    df['z_score'] = (df['ratio'] - df['ratio_mean']) / df['ratio_std']
    
    # Zorg dat we alleen rijen overhouden waar de AI-score ook echt bekend is
    if 'is_system_anomaly' not in df.columns:
        print("❌ Fout: Kolom 'is_system_anomaly' ontbreekt in de CSV. Heb je de juiste data engine gedraaid?")
        return
        
    df = df.dropna(subset=['z_score', 'is_system_anomaly']).reset_index(drop=True)

    position = None  
    entry_idx = 0
    entry_time = None
    entry_us500 = 0.0
    entry_gold = 0.0

    trades_log = []
    equity_curve = [0.0]
    skipped_trades_count = 0  

    # Marktsimulatie Loop
    for i in range(len(df)):
        row = df.iloc[i]
        curr_time = row['time'].time()
        is_inside_hours = time(0, 30) <= curr_time <= time(22, 0)
        z_curr = row['z_score']

        # ---------------------------------------------------------------------
        # CASE A: ER IS EEN ACTIEVE PAIRS TRADE (Wacht op Convergence, Timeout of EOD)
        # ---------------------------------------------------------------------
        if position is not None:
            if position == 'US500_SHORT_GOLD_LONG':
                pct_us500 = ((entry_us500 - row['US500_close_ask']) / entry_us500) * 100
                pct_gold = ((row['GOLD_close_bid'] - entry_gold) / entry_gold) * 100
            elif position == 'US500_LONG_GOLD_SHORT':
                pct_us500 = ((row['US500_close_bid'] - entry_us500) / entry_us500) * 100
                pct_gold = ((entry_gold - row['GOLD_close_ask']) / entry_gold) * 100
            
            float_pnl_combination = (pct_us500 + pct_gold) / 2
            
            converged = False
            if position == 'US500_SHORT_GOLD_LONG' and z_curr <= 0:
                converged = True
                reason = "MEAN_REVERSION_CONVERGENCE"
            elif position == 'US500_LONG_GOLD_SHORT' and z_curr >= 0:
                converged = True
                reason = "MEAN_REVERSION_CONVERGENCE"
            
            timeout = (i - entry_idx) >= MAX_DURATION
            forced_eod = curr_time > time(22, 0)

            if converged or timeout or forced_eod:
                if not converged:
                    reason = "FORCED_EOD_CLOSE" if forced_eod else "MAX_DURATION_TIMEOUT"
                
                exit_us500 = row['US500_close_ask'] if position == 'US500_SHORT_GOLD_LONG' else row['US500_close_bid']
                exit_gold = row['GOLD_close_bid'] if position == 'US500_SHORT_GOLD_LONG' else row['GOLD_close_ask']
                
                trades_log.append({
                    'type': position, 'entry_time': entry_time, 'exit_time': row['time'],
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'entry_us500': entry_us500, 'exit_us500': exit_us500,
                    'entry_gold': entry_gold, 'exit_gold': exit_gold,
                    'pct_us500': pct_us500, 'pct_gold': pct_gold,
                    'pnl_pct': float_pnl_combination, 'reason': reason
                })
                equity_curve.append(equity_curve[-1] + float_pnl_combination)
                position = None
                continue

        # ---------------------------------------------------------------------
        # CASE B: GEEN OPENDE POSITIE (🔥 AI MODE: Kijkt naar Z-score + Anomalie!)
        # ---------------------------------------------------------------------
        else:
            if is_inside_hours and row['is_system_anomaly'] == 1 and abs(z_curr) >= Z_THRESHOLD:
                
                expected_win_pct = (abs(row['ratio'] - row['ratio_mean']) / row['ratio']) * 100 / 2
                
                if expected_win_pct < MIN_EXPECTED_WIN_PCT:
                    skipped_trades_count += 1
                    continue
                
                if z_curr >= Z_THRESHOLD:
                    position = 'US500_SHORT_GOLD_LONG'
                    entry_us500 = row['US500_close_bid']  
                    entry_gold = row['GOLD_close_ask']    
                    entry_time = row['time']
                    entry_idx = i
                    
                elif z_curr <= -Z_THRESHOLD:
                    position = 'US500_LONG_GOLD_SHORT'
                    entry_us500 = row['US500_close_ask']   
                    entry_gold = row['GOLD_close_bid']     
                    entry_time = row['time']
                    entry_idx = i

    # ---------------------------------------------------------------------
    # 📝 LEDGER RAPPORTAGE GENEREREN (.MD)
    # ---------------------------------------------------------------------
    trades_df = pd.DataFrame(trades_log)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 📊 MANTRA: AI-Filtered Statistical Arbitrage Ledger\n\n")
        f.write("* **Strategy Mode:** `HIGH-PRECISION ISOLATION FOREST + Z-SCORE`\n")
        if len(trades_df) > 0:
            winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
            f.write(f"* **Total Systemic Trades Executed:** {len(trades_df)}\n")
            f.write(f"* **Trades Skipped by Expected-Win Filter:** {skipped_trades_count}\n")
            f.write(f"* **Arbitrage Win Rate:** {(winning_trades / len(trades_df)) * 100:.2f}%\n")
            f.write(f"* **Net Combined Strategy Yield (Total Capital ROI):** {trades_df['pnl_pct'].sum():.4f}%\n")
            f.write(f"* **Average Return per Trade Combination:** {trades_df['pnl_pct'].mean():.4f}%\n\n")
            
            f.write("### 📜 Geavanceerd Transactie Ledger (Leg Decomposition)\n")
            f.write("| # | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | PnL Trade Combination | Reason |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            for idx, r in trades_df.iterrows():
                us500_pos = "SHORT" if "US500_SHORT" in r['type'] else "LONG"
                gold_pos = "LONG" if "GOLD_LONG" in r['type'] else "SHORT"
                
                f.write(f"| {idx+1} | {r['entry_time'].strftime('%m-%d %H:%M')} | {r['exit_time'].strftime('%m-%d %H:%M')} | "
                        f"`{us500_pos}` | {r['entry_us500']:.2f} | {r['exit_us500']:.2f} | {r['pct_us500']:.4f}% | "
                        f"`{gold_pos}` | {r['entry_gold']:.2f} | {r['exit_gold']:.2f} | {r['pct_gold']:.4f}% | "
                        f"**{r['pnl_pct']:.4f}%** | `{r['reason']}` |\n")
        else:
            f.write("Geen cross-asset arbitrages geactiveerd binnen de huidige parameters.\n\n")
            f.write(f"* **Trades Skipped by Expected-Win Filter:** {skipped_trades_count}\n")

    if len(trades_df) > 0:
        # GRAFIEK 1: ROI
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(equity_curve)), equity_curve, color='purple', linewidth=2, marker='o', label='Combined Pairs ROI (%)')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.title("MANTRA Arbitrage Node: Cumulative Growth Curve (Return in %)", fontsize=11, fontweight='bold', loc='left')
        plt.xlabel("Sequence of Closed Pairs Trades")
        plt.ylabel("Net Combined Return (%)")
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUTPUT_CHART_ROI, dpi=300)
        plt.close()

        # GRAFIEK 2: 3-LAGIG DASHBOARD
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        ax1.plot(df.index, df['US500_price'], color='#1f78b4', alpha=0.4, label='US500 Mid Price')
        ax2.plot(df.index, df['GOLD_price'], color='#ffd700', alpha=0.5, label='GOLD Mid Price')
        ax3.plot(df.index, df['z_score'], color='#6a3d9a', alpha=0.7, linewidth=1.5, label='Real-time Z-Score')
        
        ax3.axhline(0, color='black', linestyle='-', alpha=0.4)
        ax3.axhline(Z_THRESHOLD, color='red', linestyle='--', alpha=0.6, label=f'Trigger Bound (+/-{Z_THRESHOLD})')
        ax3.axhline(-Z_THRESHOLD, color='red', linestyle='--', alpha=0.6)
        
        legend_added = {"US_LONG": False, "US_SHORT": False, "AU_LONG": False, "AU_SHORT": False}

        for t in trades_log:
            e_idx = t['entry_idx']
            x_idx = t['exit_idx']
            
            ax1.axvspan(e_idx, x_idx, color='purple', alpha=0.08)
            ax2.axvspan(e_idx, x_idx, color='purple', alpha=0.08)
            ax3.axvspan(e_idx, x_idx, color='purple', alpha=0.08)
            
            if t['type'] == 'US500_LONG_GOLD_SHORT':
                lbl = 'Buy Order (LONG)' if not legend_added["US_LONG"] else ""
                ax1.scatter(e_idx, t['entry_us500'], color='green', marker='^', s=120, zorder=5, label=lbl)
                legend_added["US_LONG"] = True
                
                lbl = 'Sell Order (SHORT)' if not legend_added["AU_SHORT"] else ""
                ax2.scatter(e_idx, t['entry_gold'], color='red', marker='v', s=120, zorder=5, label=lbl)
                legend_added["AU_SHORT"] = True
                
            elif t['type'] == 'US500_SHORT_GOLD_LONG':
                lbl = 'Sell Order (SHORT)' if not legend_added["US_SHORT"] else ""
                ax1.scatter(e_idx, t['entry_us500'], color='red', marker='v', s=120, zorder=5, label=lbl)
                legend_added["US_SHORT"] = True
                
                lbl = 'Buy Order (LONG)' if not legend_added["AU_LONG"] else ""
                ax2.scatter(e_idx, t['entry_gold'], color='green', marker='^', s=120, zorder=5, label=lbl)
                legend_added["AU_LONG"] = True

        ax1.set_ylabel("US500 Index Price ($)", fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.4)
        ax1.legend(loc="upper left", frameon=True, shadow=True)
        ax1.set_title("MANTRA Arbitrage Node: Real-time AI-Filtered Execution Dashboard", fontsize=12, fontweight='bold', loc='left')

        ax2.set_ylabel("Gold Price ($)", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.legend(loc="upper left", frameon=True, shadow=True)

        ax3.set_ylabel("Statistical Z-Score", fontsize=10)
        ax3.grid(True, linestyle=':', alpha=0.4)
        ax3.legend(loc="upper left", frameon=True, shadow=True)
        
        num_ticks = 8
        tick_indices = np.linspace(0, len(df) - 1, num_ticks, dtype=int)
        plt.xticks(tick_indices, df['time'].dt.strftime('%m-%d %H:%M').iloc[tick_indices].values, rotation=20)
        plt.xlabel("Timeline (Synchronized Market Open Minutes)", fontsize=10)

        plt.tight_layout()
        plt.savefig(OUTPUT_CHART_EXEC, dpi=300)
        plt.close()
        print(f"✅ AI-Filtered Dashboard succesvol bijgewerkt op: {OUTPUT_CHART_EXEC}\n")

if __name__ == "__main__":
    run_ai_backtest()
