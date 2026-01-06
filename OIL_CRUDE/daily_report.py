import requests
import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from datetime import datetime

# ==============================================================================
# 1. DATA OPHALEN
# ==============================================================================
def read_latest_csv_from_crudeoil():
    user = "Stijnknoop"
    repo = "crudeoil"
    folder_path = "OIL_CRUDE"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder_path}?ref=master"
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200: raise Exception(f"GitHub error: {response.status_code}")
    csv_file = next((f for f in response.json() if f['name'].endswith('.csv')), None)
    return pd.read_csv(csv_file['download_url'])

# ==============================================================================
# 2. FEATURE ENGINEERING (INCLUSIEF ATR)
# ==============================================================================
def add_features(df_in):
    df = df_in.copy().sort_values('time')
    close = df['close_bid']
    
    # ATR (Average True Range) - Meet de "ruis" in de markt
    high_low = df['high_bid'] - df['low_bid']
    high_cp = np.abs(df['high_bid'] - close.shift())
    low_cp = np.abs(df['low_bid'] - close.shift())
    df['atr'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()
    
    df['hour'] = df['time'].dt.hour
    df['day_progression'] = np.clip((df['hour'] * 60 + df['time'].dt.minute) / 1380.0, 0, 1)
    df['volatility_proxy'] = (df['high_bid'] - df['low_bid']).rolling(15).mean() / (close + 1e-9)
    ma30, std30 = close.rolling(30).mean(), close.rolling(30).std()
    df['z_score_30m'] = (close - ma30) / (std30 + 1e-9)
    df['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    for tf, freq in [('15min', '15min'), ('1h', '60min')]:
        df_tf = df.set_index('time')['close_bid'].resample(freq).last().shift(1).rename(f'prev_{tf}_close')
        df['tf_key'] = df['time'].dt.floor(freq)
        df = df.merge(df_tf, left_on='tf_key', right_index=True, how='left')
        df[f'{tf}_trend'] = (close - df[f'prev_{tf}_close']) / (df[f'prev_{tf}_close'] + 1e-9)
        df.drop(columns=['tf_key'], inplace=True)
    
    df['prev_close_bid'] = df['close_bid'].shift(1)
    df['prev_close_ask'] = df['close_ask'].shift(1)
    return df.dropna()

f_selected = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'atr']
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

# ==============================================================================
# 3. TRADING LOOP (SPEARMAN 0.25 + SNIPER V2)
# ==============================================================================
print("--- START SNIPER V2 (0.25 SPEARMAN + 45M TIME EXIT) ---")
df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'], format='ISO8601')
df_raw = df_raw.sort_values('time')

# Dag-verwerking (versimpeld)
df_raw['trading_day'] = (df_raw['time'].diff() > pd.Timedelta(hours=4)).cumsum()
dag_dict = {f'dag_{i}': d.reset_index(drop=True) for i, (day, d) in enumerate(df_raw.groupby('trading_day'), start=1)}
sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

output_dir = "OIL_CRUDE/Trading_details"
os.makedirs(output_dir, exist_ok=True)
log_path = os.path.join(output_dir, "trading_logs.csv")

TARGET_TP = 0.0030  # 0.3% Winstdoel
MAX_TIME = 45       # 45 Minuten harde exit
MIN_SPEARMAN = 0.25 # Jouw gekozen grens

new_records = []
for current_key in sorted_keys:
    idx = sorted_keys.index(current_key)
    history_keys = sorted_keys[max(0, idx - 40):idx]
    if len(history_keys) < 20: continue
    
    X_tr, yl_tr, ys_tr = get_xy(history_keys[:int(len(history_keys)*0.75)], dag_dict)
    X_val, yl_val, ys_val = get_xy(history_keys[int(len(history_keys)*0.75):], dag_dict)
    if X_tr is None or X_val is None: continue

    m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, yl_tr)
    m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, ys_tr)
    corr_l, _ = spearmanr(m_l.predict(X_val), yl_val)
    corr_s, _ = spearmanr(m_s.predict(X_val), ys_val)

    df_day = add_features(dag_dict[current_key]).reset_index(drop=True)
    if df_day.empty: continue
    
    p_l, p_s = m_l.predict(df_day[f_selected].values), m_s.predict(df_day[f_selected].values)
    bids, asks, times, hours = df_day['close_bid'].values, df_day['close_ask'].values, df_day['time'].values, df_day['hour'].values
    
    active = False
    for j in range(len(bids) - 1):
        if not active:
            if (corr_l >= MIN_SPEARMAN and p_l[j] > np.percentile(m_l.predict(X_tr), 96)):
                ent_p, side, active, ent_t = df_day['prev_close_ask'].values[j], 1, True, times[j]
                curr_rec = {"day": current_key, "entry_time": str(ent_t), "side": "Long", "entry_p": ent_p}
                curr_sl = -0.004
            elif (corr_s >= MIN_SPEARMAN and p_s[j] > np.percentile(m_s.predict(X_tr), 96)):
                ent_p, side, active, ent_t = df_day['prev_close_bid'].values[j], -1, True, times[j]
                curr_rec = {"day": current_key, "entry_time": str(ent_t), "side": "Short", "entry_p": ent_p}
                curr_sl = -0.004
        else:
            r = (bids[j] - ent_p) / ent_p if side == 1 else (ent_p - asks[j]) / ent_p
            dur = (times[j] - ent_t).seconds // 60
            if r >= 0.0025: curr_sl = max(curr_sl, r - 0.002) # Trailing SL
            
            # EXIT CONDITIES
            reason = None
            if r >= TARGET_TP: reason = "TP 0.3%"
            elif r <= curr_sl: reason = "Stop Loss"
            elif dur >= MAX_TIME: reason = "Time Exit (45m)"
            elif hours[j] >= 23: reason = "EOD"
            
            if reason:
                curr_rec.update({"exit_time": str(times[j]), "return": r, "exit_reason": reason})
                new_records.append(curr_rec); active = False; break

pd.DataFrame(new_records).to_csv(log_path, index=False)
print("--- ANALYSE VOLTOOID ---")
