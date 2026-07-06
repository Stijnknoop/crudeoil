import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================================
# 🎛️ CENTRAL CONFIGURATION PANEL (AANPASBARE PARAMETERS)
# =========================================================================
DATA_LIMIT = 3000         # Moet matchen met de ML engine voor gelijke data-lengte
VOLATILITY_WINDOW = 15    # Venster voor risicoberekening (standaarddeviatie)
SL_MULTIPLIER = 2.5       # Ruime Stop Loss ademruimte
TP_MULTIPLIER = 0.5       # Krappe Take Profit winstneming
MAX_DURATION = 30         # Maximum trade timeout in minuten

# Mappenstructuur
RESULT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_oil")
INPUT_CSV = os.path.join(RESULT_DIR, "oil_analyzed_data.csv")
OUTPUT_REPORT = os.path.join(RESULT_DIR, "backtest_report.md")
OUTPUT_CHART = os.path.join(RESULT_DIR, "backtest_chart.png")

def run_oil_backtest():
    print(f"🚀 MANTRA Olie Backtest Engine Gestart (Window: {DATA_LIMIT})...")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Fout: {INPUT_CSV} ontbreekt. Run eerst de ML engine!")
        return

    # Inladen data
    df = pd.read_csv(INPUT_CSV)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Slicing controle
    if len(df) > DATA_LIMIT:
        df = df.tail(DATA_LIMIT).reset_index(drop=True)

    # Dynamische volatiliteit berekenen
    df['rolling_vol'] = df['close_mid'].rolling(window=VOLATILITY_WINDOW).std().bfill()

    position, entry_price, entry_time, entry_idx = None, 0.0, None, 0
    sl_price, tp_price = 0.0, 0.0
    trades_log, equity_curve = [], [0.0]
    skipped_trades_count = 0

    # Markt simulatie loop
    for i in range(len(df)):
        row = df.iloc[i]
        curr_time = row['time'].time()
        is_inside_hours = time(0, 30) <= curr_time <= time(22, 0)

        # ---------------------------------------------------------------------
        # CASE A: ER IS EEN ACTIEVE POSITIE (Check SL, TP, Timeout of EOD)
        # ---------------------------------------------------------------------
        if position is not None:
            # 1. EOD Geforceerd Sluiter
            if curr_time > time(22, 0):
                pnl = (row['close_bid'] - entry_price) if position == 'LONG' else (entry_price - row['close_ask'])
                trades_log.append({'type': position, 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': row['close_bid'] if position == 'LONG' else row['close_ask'], 'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': "FORCED_EOD_CLOSE", 'entry_time': entry_time, 'exit_time': row['time']})
                equity_curve.append(equity_curve[-1] + pnl)
                position = None
                continue

            # 2. Time-Stop na 30 minuten
            if (i - entry_idx) >= MAX_DURATION:
                pnl = (row['close_bid'] - entry_price) if position == 'LONG' else (entry_price - row['close_ask'])
                trades_log.append({'type': position, 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': row['close_bid'] if position == 'LONG' else row['close_ask'], 'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': "MAX_DURATION_TIMEOUT", 'entry_time': entry_time, 'exit_time': row['time']})
                equity_curve.append(equity_curve[-1] + pnl)
                position = None
                continue

            # 3. High-Fidelity SL/TP Order Matching
            if position == 'LONG':
                if row['low_bid'] <= sl_price:
                    pnl = sl_price - entry_price
                    trades_log.append({'type': 'LONG', 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': sl_price, 'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': 'STOP_LOSS', 'entry_time': entry_time, 'exit_time': row['time']})
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None
                elif row['high_bid'] >= tp_price:
                    pnl = tp_price - entry_price
                    trades_log.append({'type': 'LONG', 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': tp_price, 'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': 'TAKE_PROFIT', 'entry_time': entry_time, 'exit_time': row['time']})
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None
            elif position == 'SHORT':
                if row['high_ask'] >= sl_price:
                    pnl = entry_price - sl_price
                    trades_log.append({'type': 'SHORT', 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': sl_price, 'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': 'STOP_LOSS', 'entry_time': entry_time, 'exit_time': row['time']})
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None
                elif row['low_ask'] <= tp_price:
                    pnl = entry_price - tp_price
                    trades_log.append({'type': 'SHORT', 'entry_idx': entry_idx, 'exit_idx': i, 'entry_price': entry_price, 'exit_price': tp_price, 'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': 'TAKE_PROFIT', 'entry_time': entry_time, 'exit_time': row['time']})
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None

        # ---------------------------------------------------------------------
        # CASE B: GEEN GEOPENDE POSITIE (Check Triggers met Spread-Filter)
        # ---------------------------------------------------------------------
        else:
            if is_inside_hours and row['is_anomaly'] == 1:
                vol = row['rolling_vol'] if row['rolling_vol'] > 0 else 1.0
                current_spread = row['close_ask'] - row['close_bid']
                intended_tp_distance = TP_MULTIPLIER * vol

                # Spread Filter: de verwachte TP moet de transactiekosten dekken
                if intended_tp_distance <= current_spread:
                    skipped_trades_count += 1
                    continue

                if row['anomaly_type'] == 'DOWN_SHOCK':
                    position, entry_price, entry_time, entry_idx = 'LONG', row['close_ask'], row['time'], i
                    sl_price = entry_price - (SL_MULTIPLIER * vol)
                    tp_price = entry_price + intended_tp_distance
                elif row['anomaly_type'] == 'UP_SHOCK':
                    position, entry_price, entry_time, entry_idx = 'SHORT', row['close_bid'], row['time'], i
                    sl_price = entry_price + (SL_MULTIPLIER * vol)
                    tp_price = entry_price - intended_tp_distance

    # ---------------------------------------------------------------------
    # 📝 RAPPORTAGE EN GRAFIEKEN GENEREREN
    # ---------------------------------------------------------------------
    trades_df = pd.DataFrame(trades_log)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 📊 MANTRA: OIL_CRUDE Strategy Backtest Performance\n\n")
        if len(trades_df) > 0:
            f.write(f"* **Total Executed Trades:** {len(trades_df)}\n")
            f.write(f"* **Trades Skipped by Spread-Filter:** {skipped_trades_count}\n")
            f.write(f"* **Strategy Win Rate:** {(len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)) * 100:.2f}%\n")
            f.write(f"* **Net Strategy Yield:** {trades_df['pnl'].sum():.2f} Points\n\n")
            
            f.write("### 📜 Transactie Ledger\n")
            f.write("| # | Type | Entry | Exit | Entry ($) | Exit ($) | PnL | Reason |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for idx, r in trades_df.iterrows():
                f.write(f"| {idx+1} | {r['type']} | {r['entry_time'].strftime('%m-%d %H:%M')} | {r['exit_time'].strftime('%m-%d %H:%M')} | {r['entry_price']:.2f} | {r['exit_price']:.2f} | {r['pnl']:.2f} | `{r['reason']}` |\n")
        else:
            f.write("No trades executed within the historical sample parameters.")

    # Renderen van de strakke Bracket Grafiek
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['close_mid'], color='#ff7f0e', alpha=0.4, label='OIL_CRUDE Baseline', linewidth=1.2)
    
    added = {"L": False, "S": False, "TP": False, "SL": False}
    for t in trades_log:
        r = np.arange(t['entry_idx'], t['exit_idx'] + 1)
        plt.plot(r, np.full(len(r), t['tp_price']), color='#ffd700', linestyle='-.', linewidth=1.8, label='Take Profit Target' if not added["TP"] else "")
        plt.plot(r, np.full(len(r), t['sl_price']), color='#ff4500', linestyle='--', linewidth=1.5, label='Stop Loss Target' if not added["SL"] else "")
        added["TP"], added["SL"] = True, True
        
        if t['type'] == 'LONG':
            plt.scatter(t['entry_idx'], t['entry_price'], color='green', marker='^', s=100, label='Buy Order (LONG)' if not added["L"] else "", zorder=5)
            added["L"] = True
        else:
            plt.scatter(t['entry_idx'], t['entry_price'], color='red', marker='v', s=100, label='Sell Order (SHORT)' if not added["S"] else "", zorder=5)
            added["S"] = True

    plt.xticks(np.linspace(0, len(df)-1, 8, dtype=int), df['time'].dt.strftime('%m-%d %H:%M').iloc[np.linspace(0, len(df)-1, 8, dtype=int)].values, rotation=25)
    plt.title(f"MANTRA Oil Execution Node: Asymmetric Bracket Orders (Window: {DATA_LIMIT}m)", fontsize=12, fontweight='bold', loc='left')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=300)
    plt.close()
    print("✅ Backtest Engine succesvol afgerond.")

if __name__ == "__main__":
    run_oil_backtest()
