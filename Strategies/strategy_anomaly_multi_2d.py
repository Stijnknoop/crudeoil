import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

# =========================================================================
# 🎛️ CENTRAL CONFIGURATION PANEL (ML ANOMALY DETECTION)
# =========================================================================
DATA_LIMIT = 5000         # Aantal synchrone minuten om te analyseren
AGGREGATION_MINUTES = 15  # Return window voor de onderlinge relatie
WINDOW_SIZE = 240         # 4 uur rolling lookback voor het trainen van de AI

# Output mappen
OUTPUT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_multi_2d")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multi_asset_2d_analyzed_data.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "multi_asset_2d_dashboard.png")

def load_and_prepare_asset(folder_name):
    search_pattern = os.path.join(folder_name, "outputs_merged_*.csv")
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise FileNotFoundError(f"❌ Geen data gevonden in map: {folder_name}")
    
    df = pd.read_csv(files[-1])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    df['mid'] = (df['close_bid'] + df['close_ask']) / 2
    df[f'{folder_name}_return'] = df['mid'].pct_change(periods=AGGREGATION_MINUTES)
    
    return df[['time', 'mid', 'close_bid', 'close_ask', f'{folder_name}_return']].rename(
        columns={
            'mid': f'{folder_name}_price',
            'close_bid': f'{folder_name}_close_bid',
            'close_ask': f'{folder_name}_close_ask'
        }
    )

def run_multi_anomaly_engine():
    print("⚡ MANTRA AI Engine Opstarten (US500 vs GOLD) ⚡")
    print("🔄 Simulatie: Volledige rolling her-training van de Isolation Forest...\n")
    
    us500 = load_and_prepare_asset("US500")
    gold = load_and_prepare_asset("GOLD")
    
    merged_df = pd.merge(us500, gold, on='time', how='inner').sort_values('time').reset_index(drop=True)
    merged_df = merged_df.dropna().reset_index(drop=True)
    
    if DATA_LIMIT is not None and len(merged_df) > DATA_LIMIT:
        merged_df = merged_df.tail(DATA_LIMIT).reset_index(drop=True)
        
    print(f"📊 Matrix gebouwd: {len(merged_df)} synchrone minuten gevonden.")

    feature_cols = ['US500_return', 'GOLD_return']
    matrix_features = merged_df[feature_cols].values
    
    system_anomalies = np.zeros(len(merged_df))
    rolling_scores = np.zeros(len(merged_df))
    
    print("🧠 Starten van de intensieve minuteloop voor de AI...")
    for i in tqdm(range(WINDOW_SIZE, len(merged_df))):
        train_slice = matrix_features[i - WINDOW_SIZE : i]
        
        active_model = IsolationForest(contamination=0.01, random_state=42, n_estimators=50, n_jobs=-1)
        active_model.fit(train_slice)
        
        current_sample = matrix_features[i].reshape(1, -1)
        rolling_scores[i] = active_model.decision_function(current_sample)[0]
        system_anomalies[i] = 1 if active_model.predict(current_sample)[0] == -1 else 0

    merged_df['anomaly_score'] = rolling_scores
    merged_df['is_system_anomaly'] = system_anomalies

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Multi-Asset data INCLUSIEF AI-kolom opgeslagen in: {OUTPUT_CSV}\n")

if __name__ == "__main__":
    run_multi_anomaly_engine()
