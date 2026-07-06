import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# Instellingen voor mappen en bestanden
RESULT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_single")
INPUT_CSV = os.path.join(RESULT_DIR, "us500_analyzed_data.csv")
OUTPUT_REPORT = os.path.join(RESULT_DIR, "backtest_report.md")
OUTPUT_CHART = os.path.join(RESULT_DIR, "backtest_chart.png")

VOLATILITY_WINDOW = 15  # Volatilitetsmeting van de afgelopen 15 minuten

# Asymmetrische risico-multipliers
SL_MULTIPLIER = 1.5    # Ruime Stop Loss
TP_MULTIPLIER = 0.5    # Krappe Take Profit

# NIEUW: Maximale doorlooptijd van een trade in minuten (bars)
MAX_DURATION = 30      

def run_backtest():
    print(f"🚀 MANTRA Backtest Engine: Risico-afstelling activeert Time-Stop na {MAX_DURATION} minuten...")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Fout: Analysebestand {INPUT_CSV} niet gevonden.")
        return

    # Load data
    df = pd.read_csv(INPUT_CSV)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # 1️⃣ Bereken de dynamische volatiliteit
    df['rolling_vol'] = df['close_mid'].rolling(window=VOLATILITY_WINDOW).std()
    df['rolling_vol'] = df['rolling_vol'].bfill()

    # Backtest variabelen
    position = None      
    entry_price = 0.0
    entry_time = None
    entry_idx = 0
    sl_price = 0.0
    tp_price = 0.0
    
    trades_log = []
    equity_curve = [0.0]

    for i in range(len(df)):
        row = df.iloc[i]
        curr_datetime = row['time']
        curr_time = curr_datetime.time()
        
        start_trade_zone = time(0, 30)
        end_trade_zone = time(22, 0)
        is_inside_trading_hours = start_trade_zone <= curr_time <= end_trade_zone

        # ---------------------------------------------------------------------
        # CASE A: ER IS EEN ACTIEVE POSITIE (Check SL, TP, Timeout of EOD)
        # ---------------------------------------------------------------------
        if position is not None:
            # 1. Tijdrestrictie: Handelsdag voorbij (> 22:00) -> Geforceerd uitstappen
            if curr_time > end_trade_zone:
                if position == 'LONG':
                    pnl = row['close_bid'] - entry_price
                    exit_price = row['close_bid']
                elif position == 'SHORT':
                    pnl = entry_price - row['close_ask']
                    exit_price = row['close_ask']
                
                trades_log.append({
                    'type': position, 'entry_time': entry_time, 'exit_time': curr_datetime,
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'entry_price': entry_price, 'exit_price': exit_price, 
                    'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': "FORCED_EOD_CLOSE"
                })
                equity_curve.append(equity_curve[-1] + pnl)
                position = None
                continue

            # 2. NIEUW: Max Duration Timeout Check (Trade duurt te lang)
            if (i - entry_idx) >= MAX_DURATION:
                if position == 'LONG':
                    pnl = row['close_bid'] - entry_price
                    exit_price = row['close_bid']
                elif position == 'SHORT':
                    pnl = entry_price - row['close_ask']
                    exit_price = row['close_ask']
                
                trades_log.append({
                    'type': position, 'entry_time': entry_time, 'exit_time': curr_datetime,
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'entry_price': entry_price, 'exit_price': exit_price, 
                    'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': "MAX_DURATION_TIMEOUT"
                })
                equity_curve.append(equity_curve[-1] + pnl)
                position = None
                continue

            # 3. Check SL/TP voor LONG positie
            if position == 'LONG':
                if row['low_bid'] <= sl_price:
                    pnl = sl_price - entry_price
                    trades_log.append({
                        'type': 'LONG', 'entry_time': entry_time, 'exit_time': curr_datetime,
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'entry_price': entry_price, 'exit_price': sl_price, 
                        'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': 'STOP_LOSS'
                    })
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None
                elif row['high_bid'] >= tp_price:
                    pnl = tp_price - entry_price
                    trades_log.append({
                        'type': 'LONG', 'entry_time': entry_time, 'exit_time': curr_datetime,
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'entry_price': entry_price, 'exit_price': tp_price, 
                        'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': 'TAKE_PROFIT'
                    })
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None

            # 4. Check SL/TP voor SHORT positie
            elif position == 'SHORT':
                if row['high_ask'] >= sl_price:
                    pnl = entry_price - sl_price
                    trades_log.append({
                        'type': 'SHORT', 'entry_time': entry_time, 'exit_time': curr_datetime,
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'entry_price': entry_price, 'exit_price': sl_price, 
                        'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': 'STOP_LOSS'
                    })
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None
                elif row['low_ask'] <= tp_price:
                    pnl = entry_price - tp_price
                    trades_log.append({
                        'type': 'SHORT', 'entry_time': entry_time, 'exit_time': curr_datetime,
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'entry_price': entry_price, 'exit_price': tp_price, 
                        'sl_price': sl_price, 'tp_price': tp_price, 'pnl': pnl, 'reason': 'TAKE_PROFIT'
                    })
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None

        # ---------------------------------------------------------------------
        # CASE B: GEEN GEOPENDE POSITIE (Check Triggers)
        # ---------------------------------------------------------------------
        else:
            if is_inside_trading_hours and row['is_anomaly'] == 1:
                vol = row['rolling_vol']
                if pd.isna(vol) or vol <= 0:
                    vol = 1.0

                if row['anomaly_type'] == 'DOWN_SHOCK':
                    position = 'LONG'
                    entry_price = row['close_ask']
                    entry_time = curr_datetime
                    entry_idx = i
                    sl_price = entry_price - (SL_MULTIPLIER * vol)
                    tp_price = entry_price + (TP_MULTIPLIER * vol)
                    
                elif row['anomaly_type'] == 'UP_SHOCK':
                    position = 'SHORT'
                    entry_price = row['close_bid']
                    entry_time = curr_datetime
                    entry_idx = i
                    sl_price = entry_price + (SL_MULTIPLIER * vol)
                    tp_price = entry_price - (TP_MULTIPLIER * vol)

    # 2️⃣ GENEREREN VAN PERFORMANCE RAPPORT (.MD)
    trades_df = pd.DataFrame(trades_log)
    total_trades = len(trades_df)
    if total_trades > 0:
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
    else:
        win_rate, total_pnl, avg_pnl = 0.0, 0.0, 0.0

    print("📝 Schrijven van backtest statistieken naar Markdown...")
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 📊 MANTRA: Mean-Reversion Strategy Performance Ledger\n\n")
        f.write("### 📈 Core Performance Metrics\n")
        f.write(f"* **Total Executed Trades:** {total_trades}\n")
        f.write(f"* **Strategy Win Rate:** {win_rate:.2f}%\n")
        f.write(f"* **Cumulative Strategy Yield (Gross PnL):** {total_pnl:.2f} Index Points\n")
        f.write(f"* **Average Return per Trade:** {avg_pnl:.2f} Points\n\n")
        
        f.write("### 📜 Volledig Transactie Logboek\n")
        f.write("| # | Type | Entry Time | Exit Time | Entry ($) | Exit ($) | Return (Pts) | Close Reason |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for idx, r in trades_df.iterrows():
            f.write(f"| {idx+1} | {r['type']} | {r['entry_time'].strftime('%m-%d %H:%M')} | "
                    f"{r['exit_time'].strftime('%m-%d %H:%M')} | {r['entry_price']:.2f} | "
                    f"{r['exit_price']:.2f} | {r['pnl']:.2f} | `{r['reason']}` |\n")

    # ---------------------------------------------------------------------
    # 📊 3️⃣ HIGH-FIDELITY EXECUTION CHART GENEREREN
    # ---------------------------------------------------------------------
    print("📊 Genereren van Asymmetric Execution Chart met Time-Stops...")
    plt.figure(figsize=(15, 8))
    
    plt.plot(df.index, df['close_mid'], color='#1f78b4', alpha=0.5, label='US500 Mid Price Baseline', linewidth=1.2)
    
    legend_labels_added = {"LONG_ENTRY": False, "SHORT_ENTRY": False, "TP_LINE": False, "SL_LINE": False}
    
    for t in trades_log:
        e_idx = t['entry_idx']
        x_idx = t['exit_idx']
        
        trade_x_range = np.arange(e_idx, x_idx + 1)
        tp_series = np.full(len(trade_x_range), t['tp_price'])
        sl_series = np.full(len(trade_x_range), t['sl_price'])
        
        # Take Profit Lijn
        lbl_tp = 'Take Profit Target (Tight)' if not legend_labels_added["TP_LINE"] else ""
        plt.plot(trade_x_range, tp_series, color='#ffd700', linestyle='-.', linewidth=1.8, label=lbl_tp)
        legend_labels_added["TP_LINE"] = True
        
        # Stop Loss Lijn
        lbl_sl = 'Stop Loss Target (Wide)' if not legend_labels_added["SL_LINE"] else ""
        plt.plot(trade_x_range, sl_series, color='#ff4500', linestyle='--', linewidth=1.5, label=lbl_sl)
        legend_labels_added["SL_LINE"] = True
        
        # Entry Markers
        if t['type'] == 'LONG':
            lbl_ent = 'Buy Order (LONG Entry)' if not legend_labels_added["LONG_ENTRY"] else ""
            plt.scatter(e_idx, t['entry_price'], color='green', marker='^', s=100, label=lbl_ent, zorder=5)
            legend_labels_added["LONG_ENTRY"] = True
        elif t['type'] == 'SHORT':
            lbl_ent = 'Sell Order (SHORT Entry)' if not legend_labels_added["SHORT_ENTRY"] else ""
            plt.scatter(e_idx, t['entry_price'], color='red', marker='v', s=100, label=lbl_ent, zorder=5)
            legend_labels_added["SHORT_ENTRY"] = True

    # As-opmaak
    num_ticks = 8
    tick_indices = np.linspace(0, len(df) - 1, num_ticks, dtype=int)
    tick_labels = df['time'].dt.strftime('%m-%d %H:%M').iloc[tick_indices].values
    plt.xticks(tick_indices, tick_labels, rotation=25)
    
    plt.title("MANTRA Strategy Execution Node: US500 Asymmetric Bracket Orders (With 30m Time-Stop)", fontsize=12, fontweight='bold', loc='left')
    plt.xlabel("Timeline (Market Open Minutes)", fontsize=10)
    plt.ylabel("Index Price ($)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc="upper left", frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=300)
    plt.close()
    print(f"✅ Execution-grafiek met Time-Stops succesvol opgeslagen op: {OUTPUT_CHART}\n")

if __name__ == "__main__":
    run_backtest()
