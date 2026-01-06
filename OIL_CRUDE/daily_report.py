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
# 1. DATA & GEAVANCEERDE FEATURES
# ==============================================================================
def read_latest_csv_from_crudeoil():
    user, repo, folder_path = "Stijnknoop", "crudeoil", "OIL_CRUDE"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder_path}?ref=master"
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200: raise Exception(f"GitHub error: {response.status_code}")
    csv_file = next((f for f in response.json() if f['name'].endswith('.csv')), None)
    return pd.read_csv(csv_file['download_url'])

def add_features(df_in):
    df = df_in.copy().sort_values('time')
    df['hour'] = df['time'].dt.hour
    
    # --- MTF FEATURE: 4-Uurs Trend (EMA) ---
    df['4h_ema'] = df['close_bid'].ewm(span=240).mean()
    df['mtf_trend'] = np.where(df['close_bid'] > df['4h_ema'], 1, -1)
    
    # Basis Features
    df['day_progression'] = np.clip((df['hour'] * 60 + df['time'].dt.minute) / 1380.0, 0, 1)
    close = df['close_bid']
    df['volatility_proxy'] = (df['high_bid'] - df['low_bid']).rolling(15).mean() / (close + 1e-9)
    ma30, std30 = close.rolling(30).mean(), close.rolling(30).std()
    df['z_score_30m'] = (close - ma30) / (std30 + 1e-9)
    df['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    
    df['prev_close_bid'] = df['close_bid'].shift(1)
    df['prev_close_ask'] = df['close_ask'].shift(1)
    return df.dropna()

def get_tbm_labels(df, horizon=30, pt=0.002, sl=0.001):
    labels = []
    prices = df['close_bid'].values
    for i in range(len(prices) - horizon):
        window = prices[i+1 : i+1+horizon]
        entry = prices[i]
        ret_max = (np.max(window) - entry) / entry
        ret_min = (entry - np.min(window)) / entry
        
        # Label 1 als Profit Target geraakt wordt voor Stop Loss
        labels.append(1 if (ret_max >= pt and ret_min < sl) else 0)
    return np.array(labels)

# ==============================================================================
# 2. DATA PREP (Gap Handling & Day Splitting)
# ==============================================================================
df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'], format='ISO8601')
df_raw = df_raw.sort_values('time')

full_range = pd.date_range(df_raw['time'].min(), df_raw['time'].max(), freq='min')
df = pd.DataFrame({'time': full_range}).merge(df_raw, on='time', how='left')
df['has_data'] = ~df['open_bid'].isna()
df[df.columns.difference(['has_data', 'time'])] = df[df.columns.difference(['has_data', 'time'])].ffill(limit=5)

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
# 3. ANALYSIS LOOP
# ==============================================================================
f_selected = ['z_score_30m', 'macd', 'day_progression', 'volatility_proxy', 'mtf_trend']
output_dir = "OIL_CRUDE/Trading_details"; os.makedirs(output_dir, exist_ok=True)
results = []

for i in range(40, len(sorted_keys)):
    current_key = sorted_keys[i]
    history_keys = sorted_keys[i-40:i]
    
    # 1. Regime Clustering (K-Means)
    regime_features = []
    for k in history_keys:
        d = dag_dict[k]
        vol = (d['high_bid'].max() - d['low_bid'].min()) / d['close_bid'].mean()
        regime_features.append([vol, d['close_bid'].pct_change().std()])
    
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaler.fit_transform(regime_features))
    
    curr_df = add_features(dag_dict[current_key])
    if curr_df.empty: continue
    
    curr_vol = [[(curr_df['high_bid'].max() - curr_df['low_bid'].min()) / curr_df['close_bid'].mean(), curr_df['close_bid'].pct_change().std()]]
    current_regime = kmeans.predict(scaler.transform(curr_vol))[0]

    # 2. Training op historie
    X_train, y_train = [], []
    for k in history_keys:
        h_df = add_features(dag_dict[k])
        labels = get_tbm_labels(h_df)
        if len(labels) > 0:
            X_train.append(h_df[f_selected].values[:len(labels)])
            y_train.append(labels)
    
    if not X_train: continue
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(np.vstack(X_train), np.concatenate(y_train))

    # 3. Trading met Multi-Timeframe Filter
    preds_proba = model.predict_proba(curr_df[f_selected].values)[:, 1]
    
    daily_ret = 0
    trade_executed = False
    for j in range(len(curr_df)-31):
        # Sniper condities: Hoge probablity + MTF Trend bevestiging
        if preds_proba[j] > 0.75 and curr_df['mtf_trend'].iloc[j] == 1:
            entry = curr_df['prev_close_ask'].iloc[j]
            exit_p = curr_df['close_bid'].iloc[j+30] # Time-based exit
            daily_ret = (exit_p - entry) / entry
            trade_executed = True
            break 

    results.append({
        "day": current_key, 
        "regime": current_regime, 
        "return": daily_ret, 
        "trade": 1 if trade_executed else 0
    })

pd.DataFrame(results).to_csv(os.path.join(output_dir, "advanced_results.csv"), index=False)
print("--- ANALYSE VOLTOOID ---")
