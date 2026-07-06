import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Instellingen voor mappen (Input blijft uit US500 komen, outputs gaan naar de resultatenmap)
INPUT_FOLDER = "US500"
OUTPUT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_single")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "us500_analyzed_data.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "us500_anomalies_chart.png")

AGGREGATION_MINUTES = 15  # Het macro-regime van 15 minuten
WINDOW_SIZE = 240         # Rolling training window van 4 uur (240 minuten)

def detect_us500_anomalies():
    print(f"🧠 ML Engine gestart voor data uit map: {INPUT_FOLDER}...")

    # 1️⃣ Automatisch het merged CSV-bestand opsporen in de map US500
    search_pattern = os.path.join(INPUT_FOLDER, "outputs_merged_*.csv")
    csv_files = sorted(glob.glob(search_pattern))

    if not csv_files:
        print(f"❌ Fout: Geen samengevoegd bestand gevonden (outputs_merged_*.csv) in map '{INPUT_FOLDER}'")
        return

    latest_merged_file = csv_files[-1]
    print(f"📂 Input bestand ingeladen: {latest_merged_file}")

    df = pd.read_csv(latest_merged_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # 2️⃣ Feature Engineering (Mid-prijs & Overlapping Returns berekenen)
    df['close_mid'] = (df['close_bid'] + df['close_ask']) / 2
    df['US500_return'] = df['close_mid'].pct_change(periods=AGGREGATION_MINUTES)

    df = df.dropna(subset=['US500_return', 'volume']).reset_index(drop=True)

    if len(df) < (WINDOW_SIZE + 10):
        print(f"⚠️ Te weinig datahistorie om het rolling window van {WINDOW_SIZE} minuten te vullen.")
        return

    market_features = df[['US500_return', 'volume']].values

    # 3️⃣ Rolling Window Isolation Forest Matrix
    print(f"📈 Berekenen van out-of-sample anomalieën via rolling window ({WINDOW_SIZE}m)...")
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

    # Maak de geneste resultatenmap automatisch aan als deze nog niet bestaat
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 Nieuwe resultatenmap aangemaakt: {OUTPUT_DIR}")

    # Sla de resultaten op in de specifieke resultatenmap
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Geanalyseerde data succesvol opgeslagen: {OUTPUT_CSV}")

    # =========================================================================
    # 📊 VISUALISATIE GENEREREN
    # =========================================================================
    print("📊 Grafiek genereren met triggers...")
    active_df = df.iloc[WINDOW_SIZE:].reset_index(drop=True)
    anomalies_df = active_df[active_df['is_anomaly'] == 1]

    plt.figure(figsize=(14, 7))
    plt.plot(active_df['time'], active_df['close_mid'], color='#1f78b4', alpha=0.8, label='US500 Mid Price Baseline', linewidth=1.5)
    plt.scatter(anomalies_df['time'], anomalies_df['close_mid'], color='purple', marker='^', s=60, label='ML Anomaly Trigger', zorder=5)
    
    plt.title("MANTRA Single-Asset Production Node: US500 Anomaly Detection", fontsize=12, fontweight='bold', loc='left')
    plt.xlabel("Timeline (UTC)", fontsize=10)
    plt.ylabel("Index Mid Price ($)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc="upper left")
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    print(f"✅ Anomaly dashboard succesvol opgeslagen op: {OUTPUT_PLOT}\n")

if __name__ == "__main__":
    detect_us500_anomalies()
