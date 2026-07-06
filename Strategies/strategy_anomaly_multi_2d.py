import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 🎛️ CENTRAL CONFIGURATION PANEL (PURE DUAL-ASSET DATA ALIGNMENT)
# =========================================================================
DATA_LIMIT = 5000         # Aantal synchrone minuten om te synchroniseren
AGGREGATION_MINUTES = 15  # Return window voor de historische database

# Output mappenstructuur conform het MANTRA-framework
OUTPUT_DIR = os.path.join("Strategies", "results", "strategy_anomaly_multi_2d")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multi_asset_2d_analyzed_data.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "multi_asset_2d_dashboard.png")

def load_and_prepare_asset(folder_name):
    """Laadt de meest recente database van een asset en berekent de mid-prijzen."""
    search_pattern = os.path.join(folder_name, "outputs_merged_*.csv")
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise FileNotFoundError(f"❌ Geen CSV-data gevonden in map: {folder_name}")
    
    # Pak het allernieuwste bestand uit de data-ingestie
    df = pd.read_csv(files[-1])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Bereken de zuivere mid-prijs en procentuele rendementen op de close
    df['mid'] = (df['close_bid'] + df['close_ask']) / 2
    df[f'{folder_name}_return'] = df['mid'].pct_change(periods=AGGREGATION_MINUTES)
    
    return df[['time', 'mid', 'close_bid', 'close_ask', f'{folder_name}_return']].rename(
        columns={
            'mid': f'{folder_name}_price',
            'close_bid': f'{folder_name}_close_bid',
            'close_ask': f'{folder_name}_close_ask'
        }
    )

def run_pure_data_alignment():
    print("⚡ MANTRA Safe-Haven 2D Data Alignment Engine Gestart (Pure Mode) ⚡")
    print("🔄 Synchroniseren van live marktdata voor pure statistische arbitrage...\n")
    
    print("📂 Databases inlezen van beide legs...")
    try:
        us500 = load_and_prepare_asset("US500")
        gold = load_and_prepare_asset("GOLD")
    except Exception as e:
        print(f"❌ Fout tijdens laden: {str(e)}")
        return
    
    print("🔗 Tijdsassen matchen (Cross-Asset Minute-Alignment)...")
    # Zorgt ervoor dat we alleen minuten vergelijken waarop beide markten live stonden te handelen
    merged_df = pd.merge(us500, gold, on='time', how='inner')
    merged_df = merged_df.sort_values('time').reset_index(drop=True)
    merged_df = merged_df.dropna().reset_index(drop=True)
    
    if DATA_LIMIT is not None and len(merged_df) > DATA_LIMIT:
        merged_df = merged_df.tail(DATA_LIMIT).reset_index(drop=True)
        
    print(f"📊 Matrix succesvol opgebouwd: {len(merged_df)} synchrone handelsminuten veiliggesteld.")

    # Sla de opgeschoonde database op
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Matrix succesvol weggeschreven naar: {OUTPUT_CSV}")

    # ---------------------------------------------------------------------
    # 📊 SCHOON VISUEEL BASELINE DASHBOARD GENEREREN
    # ---------------------------------------------------------------------
    print("📊 Genereren van clean 2D Price Baseline Dashboard...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # S&P 500 Prijslijn
    ax1.plot(merged_df.index, merged_df['US500_price'], color='#1f78b4', alpha=0.8, label='US500 Mid Price')
    ax1.set_ylabel("Index Price ($)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left')
    ax1.set_title("MANTRA Macro Node: Synchronized Price Baselines (Pure Z-Score Mode)", fontsize=12, fontweight='bold', loc='left')

    # Goud Prijslijn
    ax2.plot(merged_df.index, merged_df['GOLD_price'], color='#ffd700', alpha=0.8, label='GOLD Mid Price')
    ax2.set_ylabel("Gold Price ($)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper left')
    
    # Tijdsnotatie op de X-as strakzetten
    num_ticks = 8
    tick_indices = np.linspace(0, len(merged_df) - 1, num_ticks, dtype=int)
    plt.xticks(tick_indices, merged_df['time'].dt.strftime('%m-%d %H:%M').iloc[tick_indices].values, rotation=20)
    plt.xlabel("Timeline (Synchronized Market Open Minutes)", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    print(f"✅ Dashboard succesvol opgeslagen op: {OUTPUT_PLOT}\n")

if __name__ == "__main__":
    run_pure_data_alignment()
