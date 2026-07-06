import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================================
# 🎛️ CENTRAL CONFIGURATION PANEL (STATISTICAL ARBITRAGE)
# =========================================================================
DATA_LIMIT = 5000         # Match met je ML engine
RATIO_LOOKBACK = 240       # 4 uur rolling window om de 'normale' verhouding te bepalen
Z_THRESHOLD = 1.5          # Vanaf welke Z-score we de elastiek-trade triggeren
MAX_DURATION = 30         # Strikte 30-minuten time exit voor scalping MR trades

# Mappenstructuur
RESULT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_multi_2d")
INPUT_CSV = os.path.join(RESULT_DIR, "multi_asset_2d_analyzed_data.csv")
OUTPUT_REPORT = os.path.join(RESULT_DIR, "multi_backtest_report.md")

# 📊 GEUPDATE: Twee aparte PNG outputs gedefinieerd om beide grafieken te bewaren
OUTPUT_CHART_ROI = os.path.join(RESULT_DIR, "multi_backtest_chart.png")       # De paarse ROI vermogensgrafiek
OUTPUT_CHART_EXEC = os.path.join(RESULT_DIR, "multi_execution_chart.png")    # Het gelaagde buy/sell dashboard

def run_multi_backtest():
    print(f"🚀 MANTRA Multi-Asset Dual-Plot Backtest Engine Gestart...")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Fout: {INPUT_CSV} ontbreekt. Run eerst de multi ML engine!")
        return

    # Inladen gesynchroniseerde data
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
    
    df = df.dropna(subset=['z_score']).reset_index(drop=True)

    position = None  
    entry_idx = 0
    entry_time = None
    entry_us500 = 0.0
    entry_gold = 0.0

    trades_log = []
    equity_curve = [0.0]

    # Marktsimulatie Loop
    for i in range(len(df)):
        row = df.iloc[i]
        curr_time = row['time'].time()
        is_inside_hours = time(0, 30) <= curr_time <= time(22, 0)

        # ---------------------------------------------------------------------
        # CASE A: ER IS EEN ACTIEVE PAIRS TRADE (Check Timeout of EOD)
        # ---------------------------------------------------------------------
        if position is not None:
            if curr_time > time(22, 0) or (i - entry_idx) >= MAX_DURATION:
                reason = "FORCED_EOD_CLOSE" if curr_time > time(22, 0) else "MAX_DURATION_TIMEOUT"
                
                if position == 'US500_SHORT_GOLD_LONG':
                    pct_us500 = ((entry_us500 - row['US500_close_ask']) / entry_us500) * 100
                    pct_gold = ((row['GOLD_close_bid'] - entry_gold) / entry_gold) * 100
                    exit_us500 = row['US500_close_ask']
                    exit_gold = row['GOLD_close_bid']
                
                elif position == 'US500_LONG_GOLD_SHORT':
                    pct_us500 = ((row['US500_close_bid'] - entry_us500) / entry_us500) * 100
                    pct_gold = ((entry_gold - row['GOLD_close_ask']) / entry_gold) * 100
                    exit_us500 = row['US500_close_bid']
                    exit_gold = row['GOLD_close_ask']
                
                total_pnl_pct = pct_us500 + pct_gold
                
                trades_log.append({
                    'type': position, 'entry_time': entry_time, 'exit_time': row['time'],
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'entry_us500': entry_us500, 'exit_us500': exit_us500,
                    'entry_gold': entry_gold, 'exit_gold': exit_gold,
                    'pnl_pct': total_pnl_pct, 'reason': reason
                })
                equity_curve.append(equity_curve[-1] + total_pnl_pct)
                position = None
                continue

        # ---------------------------------------------------------------------
        # CASE B: GEEN OPENDE POSITIE (Wacht op ML Anomaly + Z-Score Extreem)
        # ---------------------------------------------------------------------
        else:
            if is_inside_hours and row['is_system_anomaly'] == 1:
                z = row['z_score']
                
                if z >= Z_THRESHOLD:
                    position = 'US500_SHORT_GOLD_LONG'
                    entry_us500 = row['US500_close_bid']  
                    entry_gold = row['GOLD_close_ask']    
                    entry_time = row['time']
                    entry_idx = i
                    
                elif z <= -Z_THRESHOLD:
                    position = 'US500_LONG_GOLD_SHORT'
                    entry_us500 = row['US500_close_ask']   
                    entry_gold = row['GOLD_close_bid']     
                    entry_time = row['time']
                    entry_idx = i

    # ---------------------------------------------------------------------
    # 📝 PERFORMANCE LEDGER RAPPORTAGE (.MD)
    # ---------------------------------------------------------------------
    trades_df = pd.DataFrame(trades_log)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 📊 MANTRA: Cross-Asset Statistical Arbitrage Ledger\n\n")
        if len(trades_df) > 0:
            winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
            f.write(f"* **Total Systemic Trades Executed:** {len(trades_df)}\n")
            f.write(f"* **Arbitrage Win Rate:** {(winning_trades / len(trades_df)) * 100:.2f}%\n")
            f.write(f"* **Net Combined Strategy Yield:** {trades_df['pnl_pct'].sum():.4f}%\n")
            f.write(f"* **Average Return per Pair:** {trades_df['pnl_pct'].mean():.4f}%\n\n")
            
            f.write("### 📜 Transactie Ledger\n")
            f.write("| # | Arbitrage Type | Entry Time | Exit Time | Net Return (%) | Close Reason |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for idx, r in trades_df.iterrows():
                f.write(f"| {idx+1} | `{r['type']}` | {r['entry_time'].strftime('%m-%d %H:%M')} | "
                        f"{r['exit_time'].strftime('%m-%d %H:%M')} | {r['pnl_pct']:.4f}% | `{r['reason']}` |\n")
        else:
            f.write("Geen cross-asset arbitrages geactiveerd binnen de huidige parameters.")

    if len(trades_df) > 0:
        # ---------------------------------------------------------------------
        # 📊 GRAFIEK 1: CUMULATIEVE VERMOGENSKROMME (ROI IN %)
        # ---------------------------------------------------------------------
        print("📊 Genereren van de Cumulative ROI Curve...")
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(equity_curve)), equity_curve, color='purple', linewidth=2, marker='o', label='Combined Pairs ROI (%)')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.title("MANTRA Arbitrage Node: Cumulative Growth Curve (Return in %)", fontsize=11, fontweight='bold', loc='left')
        plt.xlabel("Sequence of Closed Pairs Trades")
        plt.ylabel("Net Combined Return (%)")
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(OUTPUT_CHART_ROI, dpi=300)
        plt.close()
        print(f"✅ Gecumuleerde ROI-grafiek succesvol opgeslagen op: {OUTPUT_CHART_ROI}")

        # ---------------------------------------------------------------------
        # 📊 GRAFIEK 2: DUAL-ASSET EXECUTION DASHBOARD (MET ARROWS & SHADING)
        # ---------------------------------------------------------------------
        print("📊 Genereren van het gelaagde Execution Dashboard...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Plot de baselines voor index en safe-haven
        ax1.plot(df.index, df['US500_price'], color='#1f78b4', alpha=0.4, label='US500 Mid Price')
        ax2.plot(df.index, df['GOLD_price'], color='#ffd700', alpha=0.5, label='GOLD Mid Price')
        
        legend_added = {"US_LONG": False, "US_SHORT": False, "AU_LONG": False, "AU_SHORT": False}

        # Loop door de logs om orders visueel in te tekenen
        for t in trades_log:
            e_idx = t['entry_idx']
            x_idx = t['exit_idx']
            
            # Schaduwvlak over de exacte duur van de actieve arbitrage-leg
            ax1.axvspan(e_idx, x_idx, color='purple', alpha=0.08)
            ax2.axvspan(e_idx, x_idx, color='purple', alpha=0.08)
            
            if t['type'] == 'US500_LONG_GOLD_SHORT':
                # US500 kreeg een LONG (Groene driehoek omhoog)
                lbl = 'Buy Order (LONG)' if not legend_added["US_LONG"] else ""
                ax1.scatter(e_idx, t['entry_us500'], color='green', marker='^', s=120, zorder=5, label=lbl)
                legend_added["US_LONG"] = True
                
                # Goud kreeg een SHORT (Rode driehoek omlaag)
                lbl = 'Sell Order (SHORT)' if not legend_added["AU_SHORT"] else ""
                ax2.scatter(e_idx, t['entry_gold'], color='red', marker='v', s=120, zorder=5, label=lbl)
                legend_added["AU_SHORT"] = True
                
            elif t['type'] == 'US500_SHORT_GOLD_LONG':
                # US500 kreeg een SHORT (Rode driehoek omlaag)
                lbl = 'Sell Order (SHORT)' if not legend_added["US_SHORT"] else ""
                ax1.scatter(e_idx, t['entry_us500'], color='red', marker='v', s=120, zorder=5, label=lbl)
                legend_added["US_SHORT"] = True
                
                # Goud kreeg een LONG (Groene driehoek omhoog)
                lbl = 'Buy Order (LONG)' if not legend_added["AU_LONG"] else ""
                ax2.scatter(e_idx, t['entry_gold'], color='green', marker='^', s=120, zorder=5, label=lbl)
                legend_added["AU_LONG"] = True

        # Opmaak voor de S&P 500 grafiek
        ax1.set_ylabel("US500 Index Price ($)", fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.4)
        ax1.legend(loc="upper left", frameon=True, shadow=True)
        ax1.set_title("MANTRA Arbitrage Node: Real-time Pairs Trading Execution Dashboard", fontsize=12, fontweight='bold', loc='left')

        # Opmaak voor de Goud grafiek
        ax2.set_ylabel("Gold Price ($)", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.legend(loc="upper left", frameon=True, shadow=True)
        
        # Synchronisatie van de X-as tijdsnotatie
        num_ticks = 8
        tick_indices = np.linspace(0, len(df) - 1, num_ticks, dtype=int)
        plt.xticks(tick_indices, df['time'].dt.strftime('%m-%d %H:%M').iloc[tick_indices].values, rotation=20)
        plt.xlabel("Timeline (Synchronized Market Open Minutes)", fontsize=10)

        plt.tight_layout()
        plt.savefig(OUTPUT_CHART_EXEC, dpi=300)
        plt.close()
        print(f"✅ Gelaagd execution dashboard succesvol opgeslagen op: {OUTPUT_CHART_EXEC}\n")

if __name__ == "__main__":
    run_multi_backtest()
