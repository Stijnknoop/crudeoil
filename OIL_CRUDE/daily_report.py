import requests
import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ==============================================================================
# 1. DATA & FEATURE ENGINEERING
# ==============================================================================
def read_latest_csv_from_crudeoil():
    user, repo, folder_path = "Stijnknoop", "crudeoil", "OIL_CRUDE"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder_path}?ref=master"
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200: raise Exception(f"GitHub error: {response.status_code}")
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    return pd.read_csv(csv_file['download_url'])

def add_features(df_in):
    df = df_in.copy().sort_values('time')
    df['hour'] = df['time'].dt.hour
    
    # --- MTF FEATURE: 4-Uurs EMA ---
    df['4h_ema'] = df['close_bid'].ewm(span=240).mean()
    df['mtf_trend'] = np.where(df['close_bid'] > df['4h_ema'], 1, -1)
    
    # Basis Features
    close = df['close_bid']
    df['day_progression'] = np.clip((df['hour'] * 60 + df['time'].dt.minute) / 1380.0, 0, 1)
    df['volatility_proxy'] = (df['high_bid'] - df['low_bid']).rolling(15).mean() / (close + 1e-9)
    df['z_score_30m'] = (close - close.rolling(30).mean()) / (close.rolling(30).std() + 1e-9)
    df['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    
    # Prijs voor entry berekening (vorig tijdstip)
    df['prev_close_bid'] = df['close_bid'].shift(1)
    df['prev_close_ask'] = df['close_ask'].shift(1)
    return df.dropna()

def get_tbm_labels(df, horizon=30, pt=0.002, sl=0.001):
    """
    Triple Barrier Labeling: 1 = Profit Target gehaald binnen tijd, 0 = anders.
    """
    labels = []
    prices = df['close_bid'].values
    for i in range(len(prices) - horizon):
        window = prices[i+1 : i+1+horizon]
        entry = prices[i]
        ret_max = (np.max(window) - entry) / entry
        ret_min = (entry - np.min(window)) / entry
        # Label 1 als PT geraakt wordt zonder dat SL geraakt wordt
        labels.append(1 if (ret_max >= pt and ret_min < sl) else 0)
    return np.array(labels)

# ==============================================================================
# 2. DATA PREPARATION (Dagen & Gaps)
# ==============================================================================
print("--- START DATA PREP ---")
df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'], format='ISO8601')
df_raw = df_raw.sort_values('time')

full_range = pd.date_range(df_raw['time'].min(), df_raw['time'].max(), freq='min')
df = pd.DataFrame({'time': full_range}).merge(df_raw, on='time', how='left')
df['has_data'] = ~df['open_bid'].isna()
df[df.columns.difference(['has_data', 'time'])] = df[df.columns.difference(['has_data', 'time'])].ffill(limit=5)

# Bepaal handelsdagen op basis van grote gaps (EOD)
df['gap_flag'] = (~df['has_data']) & (df['time'].dt.hour >= 20)
df['gap_group'] = (df['gap_flag'] != df['gap_flag'].shift()).cumsum()
gap_groups = df[df['gap_flag']].groupby('gap_group').agg(start_time=('time', 'first'), length=('time', 'count'))
long_gaps = gap_groups[gap_groups['length'] >= 10]

df['trading_day'] = 1
for _, row in long_gaps.iterrows():
    next_idx = df.index[(df['time'] > row['start_time']) & (df['has_data'])]
    if len(next_idx) > 0: df.loc[next_idx[0]:, 'trading_day'] += 1

dag_dict = {f'dag_{i}': d[d['has_data']].reset_index(drop=True) for i, (day, d) in enumerate(df.groupby('trading_day'), start=1)}
sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

# ==============================================================================
# 3. ANALYSIS LOOP & LOGGING
# ==============================================================================
output_dir = "OIL_CRUDE/Trading_details"
log_path = os.path.join(output_dir, "trading_logs.csv")
os.makedirs(output_dir, exist_ok=True)

f_selected = ['z_score_30m', 'macd', 'day_progression', 'volatility_proxy', 'mtf_trend']
new_records = []

print(f"--- ANALYSEREN: {len(sorted_keys)-40} dagen ---")

for i in range(40, len(sorted_keys)):
    current_key = sorted_keys[i]
    history_keys = sorted_keys[i-40:i]
    
    # A. REGIME CLUSTERING (K-Means met 3 clusters)
    regime_feat = []
    for k in history_keys:
        d = dag_dict[k]
        vol = (d['high_bid'].max() - d['low_bid'].min()) / d['close_bid'].mean()
        std = d['close_bid'].pct_change().std()
        regime_feat.append([vol, std])
    
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.fit_transform(regime_feat))
    
    df_day = add_features(dag_dict[current_key])
    if df_day.empty: continue
    
    curr_v = [[(df_day['high_bid'].max() - df_day['low_bid'].min()) / df_day['close_bid'].mean(), df_day['close_bid'].pct_change().std()]]
    regime_id = kmeans.predict(scaler.transform(curr_v))[0]

    # B. TRAINING (Classifier i.p.v. Regressor)
    X_train, y_train = [], []
    for k in history_keys:
        h_df = add_features(dag_dict[k])
        lbls = get_tbm_labels(h_df)
        if len(lbls) > 0:
            X_train.append(h_df[f_selected].values[:len(lbls)])
            y_train.append(lbls)
    
    if not X_train: continue
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42).fit(np.vstack(X_train), np.concatenate(y_train))
    
    # C. SNIPER EXECUTION
    probs = model.predict_proba(df_day[f_selected].values)[:, 1]
    day_res = {
        "day": current_key, 
        "regime": regime_id, 
        "return": 0, 
        "exit_reason": "No Trade", 
        "entry_time": str(df_day['time'].iloc[0]),
        "side": "None"
    }
    
    # Loop door de dag voor een sniper shot
    for j in range(len(df_day) - 31):
        # Condities: Kans > 75% EN 4H Trend is positief
        if probs[j] > 0.75 and df_day['mtf_trend'].iloc[j] == 1:
            ent_p = df_day['prev_close_ask'].iloc[j]
            exit_p = df_day['close_bid'].iloc[j+30]
            ret = (exit_p - ent_p) / ent_p
            
            day_res.update({
                "entry_time": str(df_day['time'].iloc[j]),
                "entry_p": round(ent_p, 4),
                "exit_p": round(exit_p, 4),
                "return": ret,
                "exit_reason": "Time-out (30m)",
                "side": "Long"
            })
            break # Stop na de eerste sniper trade van de dag
            
    new_records.append(day_res)

# OPSLAAN
final_df = pd.DataFrame(new_records)
final_df.to_csv(log_path, index=False)

# SNELLE ANALYSE PRINTEN
print("\n--- PERFORMANCE PER REGIME ---")
if not final_df.empty:
    summary = final_df.groupby('regime')['return'].agg(['count', 'sum', 'mean'])
    print(summary)

print(f"\n--- VOLTOOID --- Logboek opgeslagen in: {log_path}")
