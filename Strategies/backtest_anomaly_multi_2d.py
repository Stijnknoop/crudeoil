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
OUTPUT_CHART = os.path.join(RESULT_DIR, "multi_backtest_chart.png")

def run_multi_backtest():
    print(f"🚀 MANTRA Cross-Asset Percentage Pairs Trading Engine Gestart...")
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
                    # Procentuele PnL berekening (Inclusief Spreads!)
                    pct_us500 = ((entry_us500 - row['US500_close_ask']) / entry_us500) * 100
                    pct_gold = ((row['GOLD_close_bid'] - entry_gold) / entry_gold) * 100
                    exit_us500 = row['US500_close_ask']
                    exit_gold = row['GOLD_close_bid']
                
                elif position == 'US500_LONG_GOLD_SHORT':
                    # Procentuele PnL berekening (Inclusief Spreads!)
                    pct_us500 = ((row['US500_close_bid'] - entry_us500) / entry_us500) * 100
                    pct_gold = ((entry_gold - row['GOLD_close_ask']) / entry_gold) * 100
                    exit_us500 = row['US500_close_bid']
                    exit_gold = row['GOLD_close_ask']
                
                # Het gecombineerde procentuele rendement van de Pairs Trade
                total_pnl_pct = pct_us500 + pct_gold
                
                trades_log.append({
                    'type': position, 'entry_time': entry_time, 'exit_time': row['time'],
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
    # 📝 PERFORMANCE LEDGER RAPPORTAGE IN PERCENTAGES (.MD)
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
            
            f.write("### 📜 Transactie Ledger (Pairs Percentage Execution)\n")
            f.write("| # | Arbitrage Type | Entry Time | Exit Time | Net Return (%) | Close Reason |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for idx, r in trades_df.iterrows():
                f.write(f"| {idx+1} | `{r['type']}` | {r['entry_time'].strftime('%m-%d %H:%M')} | "
                        f"{r['exit_time'].strftime('%m-%d %H:%M')} | {r['pnl_pct']:.4f}% | `{r['reason']}` |\n")
        else:
            f.write("Geen cross-asset arbitrages geactiveerd binnen de huidige parameters.")

    # EQUITY CURVE GENEREREN IN %
    if len(trades_df) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(equity_curve)), equity_curve, color='purple', linewidth=2, marker='o', label='Combined Pairs ROI (%)')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.title("MANTRA Arbitrage Node: Cumulative Growth Curve (Return in %)", fontsize=11, fontweight='bold', loc='left')
        plt.xlabel("Sequence of Closed Pairs Trades")
        plt.ylabel("Net Combined Return (%)")
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(OUTPUT_CHART, dpi=300)
        plt.close()
        print(f"✅ Gecumuleerde arbitragegrafiek (%) opgeslagen op: {OUTPUT_CHART}\n")

if __name__ == "__main__":
    run_multi_backtest()
