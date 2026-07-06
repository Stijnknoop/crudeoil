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
VOL_MULTIPLIER = 2.0    # SL/TP op 2x de standaarddeviatie (aanpasbaar)

def run_backtest():
    print("🚀 MANTRA Quantitative Backtest Engine Opstarten...")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Fout: Analysebestand {INPUT_CSV} niet gevonden. Run eerst de anomaly engine!")
        return

    # Load data
    df = pd.read_csv(INPUT_CSV)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # 1️⃣ Bereken de dynamische volatiliteit (Standaarddeviatie van de mid-prijs over 15m)
    df['rolling_vol'] = df['close_mid'].rolling(window=VOLATILITY_WINDOW).std()
    
    # Vul eventuele lege waarden aan het begin op met de eerste geldige waarde
    df['rolling_vol'] = df['rolling_vol'].bfill()

    # Backtest variabelen
    position = None      # Kan 'LONG', 'SHORT' of None zijn
    entry_price = 0.0
    entry_time = None
    sl_price = 0.0
    tp_price = 0.0
    
    trades_log = []
    equity_curve = [0.0] # Volgt gecumuleerde winst/verlies in punten/dollars

    print("📊 Simulatie van marktorders start met bid/ask spread en tijdrestricties...")

    for i in range(len(df)):
        row = df.iloc[i]
        curr_datetime = row['time']
        curr_time = curr_datetime.time()
        
        # Definieer handelsuren (00:30 tot 22:00)
        start_trade_zone = time(0, 30)
        end_trade_zone = time(22, 0)
        is_inside_trading_hours = start_trade_zone <= curr_time <= end_trade_zone

        # ---------------------------------------------------------------------
        # CASE A: ER IS EEN ACTIEVE POSITIE (Check SL, TP of Einde Handelsdag)
        # ---------------------------------------------------------------------
        if position is not None:
            # Tijd is verstreken (> 22:00) -> Geforceerd uitstappen
            if curr_time > end_trade_zone:
                if position == 'LONG':
                    pnl = row['close_bid'] - entry_price
                    exit_reason = "FORCED_EOD_CLOSE"
                    exit_price = row['close_bid']
                elif position == 'SHORT':
                    pnl = entry_price - row['close_ask']
                    exit_reason = "FORCED_EOD_CLOSE"
                    exit_price = row['close_ask']
                
                trades_log.append({
                    'type': position, 'entry_time': entry_time, 'exit_time': curr_datetime,
                    'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl, 'reason': exit_reason
                })
                equity_curve.append(equity_curve[-1] + pnl)
                position = None
                continue

            # Check SL/TP voor LONG positie
            if position == 'LONG':
                # Gestopt op SL (gebruik low_bid voor realisme)
                if row['low_bid'] <= sl_price:
                    pnl = sl_price - entry_price
                    trades_log.append({
                        'type': 'LONG', 'entry_time': entry_time, 'exit_time': curr_datetime,
                        'entry_price': entry_price, 'exit_price': sl_price, 'pnl': pnl, 'reason': 'STOP_LOSS'
                    })
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None
                # Winst gepakt op TP (gebruik high_bid voor realisme)
                elif row['high_bid'] >= tp_price:
                    pnl = tp_price - entry_price
                    trades_log.append({
                        'type': 'LONG', 'entry_time': entry_time, 'exit_time': curr_datetime,
                        'entry_price': entry_price, 'exit_price': tp_price, 'pnl': pnl, 'reason': 'TAKE_PROFIT'
                    })
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None

            # Check SL/TP voor SHORT positie
            elif position == 'SHORT':
                # Gestopt op SL (gebruik high_ask voor realisme)
                if row['high_ask'] >= sl_price:
                    pnl = entry_price - sl_price
                    trades_log.append({
                        'type': 'SHORT', 'entry_time': entry_time, 'exit_time': curr_datetime,
                        'entry_price': entry_price, 'exit_price': sl_price, 'pnl': pnl, 'reason': 'STOP_LOSS'
                    })
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None
                # Winst gepakt op TP (gebruik low_ask voor realisme)
                elif row['low_ask'] <= tp_price:
                    pnl = entry_price - tp_price
                    trades_log.append({
                        'type': 'SHORT', 'entry_time': entry_time, 'exit_time': curr_datetime,
                        'entry_price': entry_price, 'exit_price': tp_price, 'pnl': pnl, 'reason': 'TAKE_PROFIT'
                    })
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None

        # ---------------------------------------------------------------------
        # CASE B: GEEN GEOPENDE POSITIE (Check Triggers binnen handelsuren)
        # ---------------------------------------------------------------------
        else:
            if is_inside_trading_hours and row['is_anomaly'] == 1:
                vol = row['rolling_vol']
                # Beveiliging tegen een volatiliteit van 0 om brekende SL/TP te voorkomen
                if pd.isna(vol) or vol <= 0:
                    vol = 1.0

                # Trigger 1: DOWN_SHOCK gedetecteerd -> Koop (LONG)
                if row['anomaly_type'] == 'DOWN_SHOCK':
                    position = 'LONG'
                    entry_price = row['close_ask'] # Kopen op de ask
                    entry_time = curr_datetime
                    sl_price = entry_price - (VOL_MULTIPLIER * vol)
                    tp_price = entry_price + (VOL_MULTIPLIER * vol)
                    
                # Trigger 2: UP_SHOCK gedetecteerd -> Verkoop (SHORT)
                elif row['anomaly_type'] == 'UP_SHOCK':
                    position = 'SHORT'
                    entry_price = row['close_bid'] # Shorten op de bid
                    entry_time = curr_datetime
                    sl_price = entry_price + (VOL_MULTIPLIER * vol)
                    tp_price = entry_price - (VOL_MULTIPLIER * vol)

    # ---------------------------------------------------------------------
    # 📝 4️⃣ GENEREREN VAN PERFORMANCE RAPPORT (.MD)
    # ---------------------------------------------------------------------
    trades_df = pd.DataFrame(trades_log)
    
    total_trades = len(trades_df)
    if total_trades > 0:
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
    else:
        win_rate = 0.0
        total_pnl = 0.0
        avg_pnl = 0.0

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
        
        for idx, row in trades_df.iterrows():
            f.write(f"| {idx+1} | {row['type']} | {row['entry_time'].strftime('%m-%d %H:%M')} | "
                    f"{row['exit_time'].strftime('%m-%d %H:%M')} | {row['entry_price']:.2f} | "
                    f"{row['exit_price']:.2f} | {row['pnl']:.2f} | `{row['reason']}` |\n")

    print(f"✅ Strategie rapport opgeslagen: {OUTPUT_REPORT}")

    # ---------------------------------------------------------------------
    # 📊 5️⃣ VISUALISATIE: CUMULATIEVE EQUITY CURVE GENEREREN
    # ---------------------------------------------------------------------
    if total_trades > 0:
        print("📊 Tekenen van Equity Curve...")
        plt.figure(figsize=(12, 6))
        
        # Plot de vermogensontwikkeling stap voor stap
        plt.plot(range(len(equity_curve)), equity_curve, color='#2ca02c', linewidth=2.0, marker='o', label='Strategy Capital Growth')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        plt.title("MANTRA Backtest Performance: Cumulative Equity Curve (US500)", fontsize=12, fontweight='bold', loc='left')
        plt.xlabel("Sequence of Closed Positions (Trade Number)", fontsize=10)
        plt.ylabel("Gains / Losses in Index Points ($)", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc="upper left")
        
        plt.tight_layout()
        plt.savefig(OUTPUT_CHART, dpi=300)
        plt.close()
        print(f"✅ Gecumuleerde vermogensgrafiek opgeslagen: {OUTPUT_CHART}\n")
    else:
        print("⚠️ Geen trades gegenereerd. Grafiek overgeslagen.")

if __name__ == "__main__":
    run_backtest()
