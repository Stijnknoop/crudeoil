import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Instellingen voor mappen
INPUT_FOLDER = "OIL_CRUDE"
OUTPUT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_oil")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "oil_analyzed_data.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "oil_anomalies_chart.png")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "anomaly_report.md")

AGGREGATION_MINUTES = 15  # Macro-regime van 15 minuten
WINDOW_SIZE = 240         # Rolling training window van 4 uur

def detect_oil_anomalies():
    print(f"🧠 ML Engine gestart voor data uit map: {INPUT_FOLDER}...")

    search_pattern = os.path.join(INPUT_FOLDER, "outputs_merged_*.csv")
    csv_files = sorted(glob.glob(search_pattern))

    if not csv_files:
        print(f"❌ Fout: Geen samengevoegd bestand gevonden in map '{INPUT_FOLDER}'")
        return

    latest_merged_file = csv_files[-1]
    df = pd.read_csv(latest_merged_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Feature Engineering
    df['close_mid'] = (df['close_bid'] + df['close_ask']) / 2
    df['OIL_return'] = df['close_mid'].pct_change(periods=AGGREGATION_MINUTES)

    df = df.dropna(subset=['OIL_return', 'volume']).reset_index(drop=True)

    if len(df) < (WINDOW_SIZE + 10):
        print("⚠️ Te weinig datahistorie.")
        return

    market_features = df[['OIL_return', 'volume']].values

    print(f"📈 Berekenen van rolling out-of-sample anomalieën ({WINDOW_SIZE}m)...")
    rolling_anomalies = np.zeros(len(df))
    rolling_scores = np.zeros(len(df))

    for i in range(WINDOW_SIZE, len(df)):
        train_slice = market_features[i - WINDOW_SIZE : i]
        current_sample = market_features[i].reshape(1, -1)
        
        rolling_model = IsolationForest(contamination=0.01, random_state=42, n_estimators=50, n_jobs=-1)
        rolling_model.fit(train_slice)
        
        rolling_scores[i] = rolling_model.decision_function(current_sample)[0]
        rolling_anomalies[i] = 1 if rolling_model.predict(current_sample)[0] == -1 else 0

    df['anomaly_score'] = rolling_scores
    df['is_anomaly'] = rolling_anomalies

    # Strategische Classificatie
    df['anomaly_type'] = 'NORMAL'
    df['rolling_vol_threshold'] = df['volume'].rolling(window=WINDOW_SIZE).quantile(0.95)

    for idx in df[df['is_anomaly'] == 1].index:
        ret_val = df.loc[idx, 'OIL_return']
        vol_val = df.loc[idx, 'volume']
        vol_thresh = df.loc[idx, 'rolling_vol_threshold']

        if ret_val > 0.0005:
            df.loc[idx, 'anomaly_type'] = 'UP_SHOCK'
        elif ret_val < -0.0005:
            df.loc[idx, 'anomaly_type'] = 'DOWN_SHOCK'
        elif vol_val > vol_thresh:
            df.loc[idx, 'anomaly_type'] = 'VOLUME_SURGE'
        else:
            df.loc[idx, 'anomaly_type'] = 'VOLATILITY_SPKE'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df.to_csv(OUTPUT_CSV, index=False)

    # Rapport genereren
    anomalies_only = df[df['is_anomaly'] == 1].copy()
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 🚨 MANTRA: OIL_CRUDE Signal Audit Ledger\n\n")
        f.write(f"Totaal aantal gedetecteerde anomalieën: **{len(anomalies_only)}**\n\n")
        f.write("| Timestamp (UTC) | Price ($) | 15m Return % | Volume | Classification |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for _, row in anomalies_only.iterrows():
            time_str = row['time'].strftime('%Y-%m-%d %H:%M')
            f.write(f"| {time_str} | ${row['close_mid']:.2f} | {row['OIL_return'] * 100:.3f}% | {int(row['volume'])} | **{row['anomaly_type']}** |\n")

    # Grafiek genereren zonder weekend gaten
    active_df = df.iloc[WINDOW_SIZE:].reset_index(drop=True)
    plt.figure(figsize=(14, 7))
    plt.plot(active_df.index, active_df['close_mid'], color='#ff7f0e', alpha=0.6, label='OIL_CRUDE Mid Price Baseline', linewidth=1.5)
    
    colors = {'UP_SHOCK': 'green', 'DOWN_SHOCK': 'red', 'VOLUME_SURGE': 'blue', 'VOLATILITY_SPKE': 'purple'}
    markers = {'UP_SHOCK': '^', 'DOWN_SHOCK': 'v', 'VOLUME_SURGE': 'o', 'VOLATILITY_SPKE': 's'}
    
    for a_type in ['UP_SHOCK', 'DOWN_SHOCK', 'VOLUME_SURGE', 'VOLATILITY_SPKE']:
        type_df = active_df[active_df['anomaly_type'] == a_type]
        if not type_df.empty:
            plt.scatter(type_df.index, type_df['close_mid'], color=colors[a_type], marker=markers[a_type], s=65, label=f'ML {a_type}', zorder=5)
            
    num_ticks = 8
    tick_indices = np.linspace(0, len(active_df) - 1, num_ticks, dtype=int)
    plt.xticks(tick_indices, active_df['time'].dt.strftime('%m-%d %H:%M').iloc[tick_indices].values, rotation=25)
    plt.title("MANTRA Production Engine: Directional OIL_CRUDE Anomaly Matrix", fontsize=12, fontweight='bold', loc='left')
    plt.xlabel("Timeline (Market Open Minutes)", fontsize=10)
    plt.ylabel("Crude Oil Price ($)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    print("✅ ML Engine succesvol afgerond.")

if __name__ == "__main__":
    detect_oil_anomalies()
