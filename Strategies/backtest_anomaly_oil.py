import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

RESULT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_oil")
INPUT_CSV = os.path.join(RESULT_DIR, "oil_analyzed_data.csv")
OUTPUT_REPORT = os.path.join(RESULT_DIR, "backtest_report.md")
OUTPUT_CHART = os.path.join(RESULT_DIR, "backtest_chart.png")

VOLATILITY_WINDOW = 15
SL_MULTIPLIER = 2.5    # Ruime Stop Loss
TP_MULTIPLIER = 0.5    # Krappe Take Profit
MAX_DURATION = 30      # 30 minuten Time-Stop

def run_oil_backtest():
    print("🚀 MANTRA Quantitative Oil Backtest Engine Opstarten...")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Fout: {INPUT_CSV} ontbreekt.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    df['rolling_vol'] = df['close_mid'].rolling(window=VOLATILITY_WINDOW).std().bfill()

    position, entry_price, entry_time, entry_idx = None, 0.0, None, 0
    sl_price, tp_price = 0.0, 0.0
    trades_log, equity_curve = [], [0.0]
    skipped_trades_count = 0

    for i in range(len(df)):
        row = df.iloc[i]
        curr_time = row['time'].time()
        is_inside_hours = time(0, 30) <= curr_time <= time(22, 0)

        if position is not None:
            # 1. EOD close
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

            # 3. Check SL/TP
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
        else:
            if is_inside_hours and row['is_anomaly'] == 1:
                vol = row['rolling_vol'] if row['rolling_vol'] > 0 else 1.0
                current_spread = row['close_ask'] - row['close_bid']
                intended_tp_distance = TP_MULTIPLIER * vol

                # Spread-filter check
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

    # Rapport schrijven
    trades_df = pd.DataFrame(trades_log)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 📊 MANTRA: OIL_CRUDE Strategy Backtest Performance\n\n")
        if len(trades_df) > 0:
            f.write(f"* **Total Trades:** {len(trades_df)}\n* **Skipped by Spread-Filter:** {skipped_trades_count}\n")
            f.write(f"* **Win Rate:** {(len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)) * 100:.2f}%\n")
            f.write(f"* **Net PnL:** {trades_df['pnl'].sum():.2f} Points\n")
        else:
            f.write("No trades executed.")

    # Execution chart genereren
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['close_mid'], color='#ff7f0e', alpha=0.4, label='OIL_CRUDE Baseline', linewidth=1.2)
    
    added = {"L": False, "S": False, "TP": False, "SL": False}
    for t in trades_log:
        r = np.arange(t['entry_idx'], t['exit_idx'] + 1)
        plt.plot(r, np.full(len(r), t['tp_price']), color='#ffd700', linestyle='-.', linewidth=1.8, label='Take Profit' if not added["TP"] else "")
        plt.plot(r, np.full(len(r), t['sl_price']), color='#ff4500', linestyle='--', linewidth=1.5, label='Stop Loss' if not added["SL"] else "")
        added["TP"], added["SL"] = True, True
        
        if t['type'] == 'LONG':
            plt.scatter(t['entry_idx'], t['entry_price'], color='green', marker='^', s=100, label='LONG Entry' if not added["L"] else "", zorder=5)
            added["L"] = True
        else:
            plt.scatter(t['entry_idx'], t['entry_price'], color='red', marker='v', s=100, label='SHORT Entry' if not added["S"] else "", zorder=5)
            added["S"] = True

    plt.xticks(np.linspace(0, len(df)-1, 8, dtype=int), df['time'].dt.strftime('%m-%d %H:%M').iloc[np.linspace(0, len(df)-1, 8, dtype=int)].values, rotation=25)
    plt.title("MANTRA Strategy Execution Node: OIL_CRUDE Bracket Orders", fontsize=12, fontweight='bold', loc='left')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=300)
    plt.close()
    print("✅ Backtest Engine succesvol afgerond.")

if __name__ == "__main__":
    run_oil_backtest()
