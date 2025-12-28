import requests
import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

# 1. DATA OPHALEN
def read_latest_csv_from_crudeoil():
    user = "Stijnknoop"
    repo = "crudeoil"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref=master"
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    return pd.read_csv(csv_file['download_url'])

print("--- START ANALYSE ---")
df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'])
df_raw = df_raw.sort_values('time')

# Data groeperen in dagen
full_range = pd.date_range(df_raw['time'].min(), df_raw['time'].max(), freq='1T')
df = pd.DataFrame({'time': full_range}).merge(df_raw, on='time', how='left')
df['has_data'] = ~df['open_bid'].isna()
df = df.set_index('time')
df[df.columns.difference(['has_data'])] = df[df.columns.difference(['has_data'])].ffill(limit=5)
df = df.reset_index()

df['date'] = df['time'].dt.date
valid_dates = df.groupby('date')['has_data'].any()
valid_dates = valid_dates[valid_dates].index
df = df[df['date'].isin(valid_dates)].copy()

df['gap_flag'] = (~df['has_data']) & (df['time'].dt.hour >= 20)
df['gap_group'] = (df['gap_flag'] != df['gap_flag'].shift()).cumsum()
gap_groups = df[df['gap_flag']].groupby('gap_group').agg(start_time=('time', 'first'), length=('time', 'count'))
long_gaps = gap_groups[gap_groups['length'] >= 10]

df['trading_day'] = 1
for _, row in long_gaps.iterrows():
    next_idx = df.index[(df['time'] > row['start_time']) & (df['has_data'])]
    if len(next_idx) > 0: df.loc[next_idx[0]:, 'trading_day'] += 1

dag_dict = {f'dag_{i}': d[d['has_data']].reset_index(drop=True) 
            for i, (day, d) in enumerate(df.groupby('trading_day'), start=1)}

# 2. FEATURE ENGINEERING
def add_features(df_in):
    df = df_in.copy().sort_values('time')
    df['hour'] = df['time'].dt.hour
    df['day_progression'] = np.clip((df['hour'] * 60 + df['time'].dt.minute) / 1380.0, 0, 1)
    close = df['close_bid']
    df['volatility_proxy'] = (df['high_bid'] - df['low_bid']).rolling(15).mean() / (close + 1e-9)
    ma30, std30 = close.rolling(30).mean(), close.rolling(30).std()
    df['z_score_30m'] = (close - ma30) / (std30 + 1e-9)
    df['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    delta = close.diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    for tf, freq in [('15min', '15min'), ('1h', '60min')]:
        df_tf = df.set_index('time')['close_bid'].resample(freq).last().shift(1).rename(f'prev_{tf}_close')
        df['tf_key'] = df['time'].dt.floor(freq)
        df = df.merge(df_tf, left_on='tf_key', right_index=True, how='left')
        df[f'{tf}_trend'] = (close - df[f'prev_{tf}_close']) / (df[f'prev_{tf}_close'] + 1e-9)
        df.drop(columns=['tf_key'], inplace=True)
    f_cols = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
    df[f_cols] = df[f_cols].shift(1)
    return df.dropna()

f_selected = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
HORIZON = 30

def get_xy(keys, d_dict):
    X, yl, ys = [], [], []
    for k in keys:
        df_f = add_features(d_dict[k])
        if len(df_f) > HORIZON + 10:
            p = df_f['close_bid'].values
            X.append(df_f[f_selected].values[:-HORIZON])
            yl.append([(np.max(p[i+1:i+1+HORIZON]) - p[i])/p[i] for i in range(len(df_f)-HORIZON)])
            ys.append([(p[i] - np.min(p[i+1:i+1+HORIZON]))/p[i] for i in range(len(df_f)-HORIZON)])
    return (np.vstack(X), np.concatenate(yl), np.concatenate(ys)) if X else (None, None, None)

# 3. ANALYSE EN VOLLEDIGE VISUALISATIE
output_dir = "Trading_details"
log_path = os.path.join(output_dir, "trading_logs.csv")
if not os.path.exists(output_dir): os.makedirs(output_dir)

sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

for current_key in sorted_keys:
    idx = sorted_keys.index(current_key)
    history_keys = sorted_keys[max(0, idx-40):idx]
    if len(history_keys) < 20: continue
    
    split = int(len(history_keys) * 0.75)
    train_keys, val_keys = history_keys[:split], history_keys[split:]
    
    X_tr, yl_tr, ys_tr = get_xy(train_keys, dag_dict)
    m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, yl_tr)
    m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, ys_tr)
    
    X_val, _, _ = get_xy(val_keys, dag_dict)
    t_l, t_s = np.percentile(m_l.predict(X_val), 95), np.percentile(m_s.predict(X_val), 95)
    
    df_day = add_features(dag_dict[current_key]).reset_index(drop=True)
    p_l, p_s = m_l.predict(df_day[f_selected].values), m_s.predict(df_day[f_selected].values)
    bids, asks, times = df_day['close_bid'].values, df_day['close_ask'].values, df_day['time'].values
    
    # Simpele simulatie voor de plot
    res = {"day": current_key, "return": 0, "side": None}
    for j in range(len(bids)-1):
        if p_l[j] > t_l:
            res.update({"entry": times[j], "exit": times[min(j+30, len(bids)-1)], "side": "Long", "p1": asks[j], "p2": bids[min(j+30, len(bids)-1)]})
            res["return"] = (res["p2"] - res["p1"]) / res["p1"]
            break
        elif p_s[j] > t_s:
            res.update({"entry": times[j], "exit": times[min(j+30, len(bids)-1)], "side": "Short", "p1": bids[j], "p2": asks[min(j+30, len(bids)-1)]})
            res["return"] = (res["p1"] - res["p2"]) / res["p1"]
            break

    # --- DE FIX: FORCEER 3 APARTE ASSEN ---
    plt.clf()
    fig = plt.figure(figsize=(12, 16))
    
    # Subplot 1: Training
    ax1 = fig.add_subplot(3, 1, 1)
    for k in train_keys:
        ax1.plot(dag_dict[k]['close_bid'].values, color='blue', alpha=0.1)
    ax1.set_title(f"1. Training Fase ({len(train_keys)} dagen)")
    ax1.grid(True, alpha=0.2)

    # Subplot 2: Validatie
    ax2 = fig.add_subplot(3, 1, 2)
    for k in val_keys:
        ax2.plot(dag_dict[k]['close_bid'].values, color='orange', alpha=0.3)
    ax2.set_title(f"2. Validatie Fase ({len(val_keys)} dagen)")
    ax2.grid(True, alpha=0.2)

    # Subplot 3: Echte Wereld
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(df_day['time'], df_day['close_bid'], color='black', label='Prijs', linewidth=1)
    if res["side"]:
        ax3.scatter(res["entry"], res["p1"], color='green', marker='^', s=100)
        ax3.scatter(res["exit"], res["p2"], color='red', marker='v', s=100)
    ax3.set_title(f"3. Echte Wereld: {current_key} | Return: {res['return']:.2%}")
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"full_analysis_{current_key}.png"))
    plt.close(fig)

print("Klaar. Check de 'Trading_details' map voor de nieuwe afbeeldingen.")
