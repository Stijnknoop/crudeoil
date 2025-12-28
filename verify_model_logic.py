import requests
import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')

# ==============================================================================
# 1. FUNCTIES (KOPIE VAN JE ORIGINELE LOGICA)
# ==============================================================================
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

def get_xy(keys, d_dict):
    f_selected = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
    HORIZON = 30
    X, yl, ys = [], [], []
    for k in keys:
        df_f = add_features(d_dict[k])
        if len(df_f) > HORIZON + 10:
            p = df_f['close_bid'].values
            X.append(df_f[f_selected].values[:-HORIZON])
            yl.append([(np.max(p[i+1:i+1+HORIZON]) - p[i])/p[i] for i in range(len(df_f)-HORIZON)])
            ys.append([(p[i] - np.min(p[i+1:i+1+HORIZON]))/p[i] for i in range(len(df_f)-HORIZON)])
    return (np.vstack(X), np.concatenate(yl), np.concatenate(ys)) if X else (None, None, None)

def calculate_dynamic_threshold(correlation_score):
    if np.isnan(correlation_score) or correlation_score < 0.01:
        return 99.9
    elif correlation_score < 0.05:
        return 98.0
    elif correlation_score < 0.10:
        return 96.0
    else:
        return 94.0

# ==============================================================================
# 2. DATA VOORBEREIDEN
# ==============================================================================
print("--- START MODEL MAKER & INTEGRITEITS-CHECK ---")
df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'])
df_raw = df_raw.sort_values('time')

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

sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

# ==============================================================================
# 3. TRAINING OP [-41:-1] EN OPSLAAN PKL
# ==============================================================================
history_keys = sorted_keys[-41:-1]
test_day_key = sorted_keys[-1]

print(f"Training op: {history_keys[0]} t/m {history_keys[-1]}")

split_point = int(len(history_keys) * 0.75)
train_keys = history_keys[:split_point]
val_keys = history_keys[split_point:]

X_tr, yl_tr, ys_tr = get_xy(train_keys, dag_dict)
X_val, yl_val, ys_val = get_xy(val_keys, dag_dict)

m_l_temp = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, yl_tr)
m_s_temp = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, ys_tr)

pred_val_l = m_l_temp.predict(X_val); pred_val_s = m_s_temp.predict(X_val)
corr_l, _ = spearmanr(pred_val_l, yl_val); corr_s, _ = spearmanr(pred_val_s, ys_val)
pct_l = calculate_dynamic_threshold(corr_l); pct_s = calculate_dynamic_threshold(corr_s)

t_l_temp = np.percentile(m_l_temp.predict(X_tr), pct_l)
t_s_temp = np.percentile(m_s_temp.predict(X_tr), pct_s)

# WEGSCHRIJVEN NAAR DISK
os.makedirs("Model", exist_ok=True)
model_path = "Model/live_trading_model.pkl"
joblib.dump({
    "model_l": m_l_temp, 
    "model_s": m_s_temp, 
    "t_l": t_l_temp, 
    "t_s": t_s_temp,
    "f_selected": ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
}, model_path)

print(f"Model succesvol weggeschreven naar {model_path}")

# ==============================================================================
# 4. HERLADEN UIT PKL EN SIMULATIE LAATSTE DAG [-1]
# ==============================================================================
print(f"Model herladen van disk voor verificatie op: {test_day_key}...")
loaded_data = joblib.load(model_path)

# Gebruik de herladen objecten
m_l = loaded_data["model_l"]
m_s = loaded_data["model_s"]
t_l = loaded_data["t_l"]
t_s = loaded_data["t_s"]
f_selected = loaded_data["f_selected"]

df_day = add_features(dag_dict[test_day_key]).reset_index(drop=True)
p_l, p_s = m_l.predict(df_day[f_selected].values), m_s.predict(df_day[f_selected].values)
bids, asks, times, hours = df_day['close_bid'].values, df_day['close_ask'].values, df_day['time'].values, df_day['hour'].values

active, res = False, {"day": test_day_key, "return": 0, "exit_reason": "No Trade"}

for j in range(len(bids) - 1):
    if not active:
        if hours[j] < 23:
            if p_l[j] > t_l:
                ent_p, side, active = asks[j], 1, True
                res.update({"entry_time": str(times[j]), "side": "Long", "entry_p": ent_p, "pred": p_l[j], "thresh": t_l})
                curr_sl = -0.004
            elif p_s[j] > t_s:
                ent_p, side, active = bids[j], -1, True
                res.update({"entry_time": str(times[j]), "side": "Short", "entry_p": ent_p, "pred": p_s[j], "thresh": t_s})
                curr_sl = -0.004
    else:
        r = (bids[j] - ent_p) / ent_p if side == 1 else (ent_p - asks[j]) / ent_p
        if r >= 0.0025: curr_sl = max(curr_sl, r - 0.002)
        
        if r >= 0.005 or r <= curr_sl or hours[j] >= 23 or j == len(bids)-2:
            res.update({"exit_time": str(times[j]), "return": r, "exit_reason": "Closed"})
            active = False; break

print("\n--- RESULTAAT NA HERLADEN UIT BESTAND ---")
print(res)
print(f"\nVerificatie voltooid. Als de entry_time ({res.get('entry_time')}) en side ({res.get('side')}) "
      "overeenkomen met je daily_report.py, dan werkt de opslag en herlading perfect.")
