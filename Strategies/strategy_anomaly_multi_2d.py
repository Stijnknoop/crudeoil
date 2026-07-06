import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

# =========================================================================
# 🎛️ CENTRAL CONFIGURATION PANEL (PURE LIVE MINUTE-BY-MINUTE SIMULATION)
# =========================================================================
DATA_LIMIT = 5000         # Aantal synchrone minuten om te analyseren
AGGREGATION_MINUTES = 15  # Return window voor de onderlinge relatie
WINDOW_SIZE = 240         # 4 uur rolling lookback voor de markt-relatie

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
    
    # Bereken mid-prijs en procentueel rendement
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
    print("⚡ MANTRA Safe-Haven 2D Engine Opstarten (US500 vs GOLD) ⚡")
    print("🔄 Simulatie: Volledige her-training van het ML-model bij ÉLKE minuut...\n")
    
    print("📂 Laden van asset-databases...")
    us500 = load_and_prepare_asset("US500")
    gold = load_and_prepare_asset("GOLD")
    
    print("🔗 Synchroniseren van de tijdsassen (Cross-Asset Alignment)...")
    merged_df = pd.merge(us500, gold, on='time', how='inner')
    merged_df = merged_df.sort_values('time').reset_index(drop=True)
    merged_df = merged_df.dropna().reset_index(drop=True)
    
    if DATA_LIMIT is not None and len(merged_df) > DATA_LIMIT:
        merged_df = merged_df.tail(DATA_LIMIT).reset_index(drop=True)
        
    print(f"📊 Matrix succesvol gebouwd: {len(merged_df)} synchrone handelsminuten gevonden.")

    feature_cols = ['US500_return', 'GOLD_return']
    matrix_features = merged_df[feature_cols].values
    
    system_anomalies = np.zeros(len(merged_df))
    rolling_scores = np.zeros(len(merged_df))
    
    print(f"🧠 Starten van de intensieve minuteloop...")
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
        os.makedirs(OUTPUT_DIR)
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Multi-Asset 2D data succesvol opgeslagen in: {OUTPUT_CSV}")

    print("📊 Genereren van clean 2D Dashboard...")
    active_df = merged_df.iloc[WINDOW_SIZE:].reset_index(drop=True)
    anomalies_only = active_df[active_df['is_system_anomaly'] == 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    ax1.plot(active_df.index, active_df['US500_price'], color='#1f78b4', alpha=0.6, label='US500 Baseline')
    ax1.scatter(anomalies_only.index, anomalies_only['US500_price'], color='purple', marker='o', s=50, zorder=5, label='Systemic Trigger')
    ax1.set_ylabel("Index Price ($)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left')
    ax1.set_title("MANTRA Macro Node: Pure Live Risk-On / Risk-Off Relationship Matrix", fontsize=12, fontweight='bold', loc='left')

    ax2.plot(active_df.index, active_df['GOLD_price'], color='#ffd700', alpha=0.6, label='GOLD Baseline')
    ax2.scatter(anomalies_only.index, anomalies_only['GOLD_price'], color='purple', marker='o', s=50, zorder=5)
    ax2.set_ylabel("Gold Price ($)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper left')
    
    num_ticks = 8
    tick_indices = np.linspace(0, len(active_df) - 1, num_ticks, dtype=int)
    plt.xticks(tick_indices, active_df['time'].dt.strftime('%m-%d %H:%M').iloc[tick_indices].values, rotation=20)
    plt.xlabel("Timeline (Synchronized Market Open Minutes)", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    print(f"✅ Dashboard succesvol opgeslagen op: {OUTPUT_PLOT}\n")

if __name__ == "__main__":
    run_multi_anomaly_engine()
