import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================================
# 🎛️ CENTRAL CONFIGURATION PANEL (ADVANCED STATISTICAL ARBITRAGE)
# =========================================================================
DATA_LIMIT = 5000         # Match met je ML engine
RATIO_LOOKBACK = 240       # 4 uur rolling window om de 'normale' verhouding te bepalen
Z_THRESHOLD = 2.2          # 🔥 GEOPTIMALISEERD: Alleen instappen bij extreme elastiek-spanning
TP_VOL_MULTIPLIER = 0.5    # Multiplier op de rolling ratio volatiliteit voor het winstdoel
MAX_DURATION = 30         # Parachute: Harde maximale duration timeout in minuten

# Mappenstructuur
RESULT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_multi_2d")
INPUT_CSV = os.path.join(RESULT_DIR, "multi_asset_2d_analyzed_data.csv")
OUTPUT_REPORT = os.path.join(RESULT_DIR, "multi_backtest_report.md")

# Twee aparte PNG-bestanden om beide grafieken te behouden
OUTPUT_CHART_ROI = os.path.join(RESULT_DIR, "multi_backtest_chart.png")       # De paarse ROI vermogensgrafiek
OUTPUT_CHART_EXEC = os.path.join(RESULT_DIR, "multi_execution_chart.png")    # Het gelaagde buy/sell dashboard

def run_multi_backtest():
    print(f"🚀 MANTRA Arbitrage Dynamic-TP Engine Gestart...")
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
    active_tp = 0.0  # Onthoudt het dynamische winstdoel van de lopende trade

    trades_log = []
    equity_curve = [0.0]

    # Marktsimulatie Loop
    for i in range(len(df)):
        row = df.iloc[i]
        curr_time = row['time'].time()
        is_inside_hours = time(0, 30) <= curr_time <= time(22, 0)
        z_curr = row['z_score']

        # ---------------------------------------------------------------------
        # CASE A: ER IS EEN ACTIEVE PAIRS TRADE (Check Dynamic TP, Convergence, Timeout)
        # ---------------------------------------------------------------------
        if position is not None:
            # Bereken de realtime zwevende procentuele PnL op de sluiting van deze minuut
            if position == 'US500_SHORT_GOLD_LONG':
                pct_us500 = ((entry_us500 - row['US500_close_ask']) / entry_us500) * 100
                pct_gold = ((row['GOLD_close_bid'] - entry_gold) / entry_gold) * 100
            elif position == 'US500_LONG_GOLD_SHORT':
                pct_us500 = ((row['US500_close_bid'] - entry_us500) / entry_us500) * 100
                pct_gold = ((entry_gold - row['GOLD_close_ask']) / entry_gold) * 100
            
            # Reëel rendement over de totale portfolio-inleg (gemiddelde van de legs)
            float_pnl_combination = (pct_us500 + pct_gold) / 2
            
            trigger_exit = False
            reason = ""

            # 1. Check of de Vaste/Dynamische Volatility Take Profit (%) is bereikt
            if float_pnl_combination >= active_tp:
                trigger_exit = True
                reason = "DYNAMIC_TAKE_PROFIT"
            
            # 2. Check op Statistisch Herstel naar het gemiddelde (Z-score Convergence)
            elif position == 'US500_SHORT_GOLD_LONG' and z_curr <= 0:
                trigger_exit = True
                reason = "MEAN_REVERSION_CONVERGENCE"
            elif position == 'US500_LONG_GOLD_SHORT' and z_curr >= 0:
                trigger_exit = True
                reason = "MEAN_REVERSION_CONVERGENCE"
            
            # 3. Check op Timeouts & Marktsluiting
            elif (i - entry_idx) >= MAX_DURATION:
                trigger_exit = True
                reason = "MAX_DURATION_TIMEOUT"
            elif curr_time > time(22, 0):
                trigger_exit = True
                reason = "FORCED_EOD_CLOSE"

            if trigger_exit:
                exit_us500 = row['US500_close_ask'] if position == 'US500_SHORT_GOLD_LONG' else row['US500_close_bid']
                exit_gold = row['GOLD_close_bid'] if position == 'US500_SHORT_GOLD_LONG' else row['GOLD_close_ask']
                
                trades_log.append({
                    'type': position, 'entry_time': entry_time, 'exit_time': row['time'],
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'entry_us500': entry_us500, 'exit_us500': exit_us500,
                    'entry_gold': entry_gold, 'exit_gold': exit_gold,
                    'pct_us500': pct_us500, 'pct_gold': pct_gold,
                    'pnl_combination': float_pnl_combination, 'target_tp': active_tp, 'reason': reason
                })
                equity_curve.append(equity_curve[-1] + float_pnl_combination)
                position = None
                continue

        # ---------------------------------------------------------------------
        # CASE B: GEEN OPENDE POSITIE (Wacht op ML Anomaly + Extreme Z-Score)
        # ---------------------------------------------------------------------
        else:
            if is_inside_hours and row['is_system_anomaly'] == 1:
                if abs(z_curr) >= Z_THRESHOLD:
                    
                    # 🔍 BEREKENING VAN DE GEKOPPELDE TRANSACTIE-VLOER (SPREAD FILTER)
                    spread_us500_pct = ((row['US500_close_ask'] - row['US500_close_bid']) / row['US500_price']) * 100
                    spread_gold_pct = ((row['GOLD_close_ask'] - row['GOLD_close_bid']) / row['GOLD_price']) * 100
                    total_spread_friction = (spread_us500_pct + spread_gold_pct) / 2
                    
                    # Dynamisch winstdoel op basis van de huidige marktbeweeglijkheid
                    raw_dynamic_tp = (row['ratio_std'] / row['ratio_mean']) * 100 * TP_VOL_MULTIPLIER
                    
                    # 🔥 DE HARD FLOOR: Moet altijd groter zijn dan 1.5x de broker frictiekosten
                    active_tp = max(raw_dynamic_tp, total_spread_friction * 1.5)
                    
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
    # 📝 LEDGER RAPPORTAGE MET PORTFOLIO METRICS (.MD)
    # ---------------------------------------------------------------------
    trades_df = pd.DataFrame(trades_log)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 📊 MANTRA: Cross-Asset Statistical Arbitrage Ledger\n\n")
        if len(trades_df) > 0:
            winning_trades = len(trades_df[trades_df['pnl_combination'] > 0])
            f.write(f"* **Total Systemic Trades Executed:** {len(trades_df)}\n")
            f.write(f"* **Arbitrage Win Rate:** {(winning_trades / len(trades_df)) * 100:.2f}%\n")
            f.write(f"* **Net Combined Strategy Yield (Total Capital ROI):** {trades_df['pnl_combination'].sum():.4f}%\n")
            f.write(f"* **Average Return per Trade Combination:** {trades_df['pnl_combination'].mean():.4f}%\n\n")
            
            f.write("### 📜 Geavanceerd Transactie Ledger (Leg Decomposition)\n")
            f.write("| # | Entry Time | Exit Time | US500 Pos | Entry US500 | Exit US500 | PnL US500 | Gold Pos | Entry GOLD | Exit GOLD | PnL GOLD | Assigned Target TP | PnL Trade Combination | Reason |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            for idx, r in trades_df.iterrows():
                us500_pos = "SHORT" if "US500_SHORT" in r['type'] else "LONG"
                gold_pos = "LONG" if "GOLD_LONG" in r['type'] else "SHORT"
                
                f.write(f"| {idx+1} | {r['entry_time'].strftime('%m-%d %H:%M')} | {r['exit_time'].strftime('%m-%d %H:%M')} | "
                        f"`{us500_pos}` | {r['entry_us500']:.2f} | {r['exit_us500']:.2f} | {r['pct_us500']:.4f}% | "
                        f"`{gold_pos}` | {r['entry_gold']:.2f} | {r['exit_gold']:.2f} | {r['pct_gold']:.4f}% | "
                        f"{r['target_tp']:.4f}% | **{r['pnl_combination'].strftime if isinstance(r['pnl_combination'], str) else f'{r[...]:.4f}' if False else f'{r[\'pnl_combination\']:.4f}%'}** | `{r['reason']}` |\n")
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

        # ---------------------------------------------------------------------
        # 📊 GRAFIEK 2: DUAL-ASSET EXECUTION DASHBOARD
        # ---------------------------------------------------------------------
        print("📊 Genereren van het gelaagde Execution Dashboard...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        ax1.plot(df.index, df['US500_price'], color='#1f78b4', alpha=0.4, label='US500 Mid Price')
        ax2.plot(df.index, df['GOLD_price'], color='#ffd700', alpha=0.5, label='GOLD Mid Price')
        
        legend_added = {"US_LONG": False, "US_SHORT": False, "AU_LONG": False, "AU_SHORT": False}

        for t in trades_log:
            e_idx = t['entry_idx']
            x_idx = t['exit_idx']
            
            ax1.axvspan(e_idx, x_idx, color='purple', alpha=0.08)
            ax2.axvspan(e_idx, x_idx, color='purple', alpha=0.08)
            
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
        ax1.set_title("MANTRA Arbitrage Node: Real-time Pairs Trading Execution Dashboard", fontsize=12, fontweight='bold', loc='left')

        ax2.set_ylabel("Gold Price ($)", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.legend(loc="upper left", frameon=True, shadow=True)
        
        num_ticks = 8
        tick_indices = np.linspace(0, len(df) - 1, num_ticks, dtype=int)
        plt.xticks(tick_indices, df['time'].dt.strftime('%m-%d %H:%M').iloc[tick_indices].values, rotation=20)
        plt.xlabel("Timeline (Synchronized Market Open Minutes)", fontsize=10)

        plt.tight_layout()
        plt.savefig(OUTPUT_CHART_EXEC, dpi=300)
        plt.close()
        print(f"✅ Dashboard succesvol bijgewerkt op: {OUTPUT_CHART_EXEC}\n")

if __name__ == "__main__":
    run_multi_backtest()
