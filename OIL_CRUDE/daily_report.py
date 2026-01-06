import requests
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from datetime import datetime

# ==============================================================================
# 🎛️ STRATEGIE CONFIGURATIE (HIER AANPASSEN)
# ==============================================================================
# Filter drempels
MIN_SPEARMAN = 0.22         # Hoe betrouwbaar moet het model zijn? (0.20 - 0.30)
SIGNAL_PERCENTILE = 96      # Hoe uniek moet het signaal zijn? (90-99. Hoger = minder trades)

# Exit drempels
TARGET_TP = 0.0030          # Winstdoel (0.0030 = 0.3%)
INITIAL_STOP_LOSS = -0.0040 # Start stop-loss (-0.0040 = -0.4%)
MAX_TRADE_MINUTES = 45      # Maximale tijd in een trade (Time Exit)

# Training & Data
HISTORY_DAYS = 40           # Hoeveel dagen terugkijken voor training?
MIN_HISTORY_NEEDED = 20     # Minimaal aantal dagen data nodig om te starten
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

def add_features(df_in):
    df = df_in.copy().sort_values('time')
    close = df['close_bid']
    # ATR Berekening
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
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / (-delta.where(delta < 0, 0).rolling(14).mean() + 1e-9))))
    
    for tf, freq in [('15min', '15min'), ('1h', '60min')]:
        df_tf = df.set_index('time')['close_bid'].resample(freq).last().shift(1).rename(f'prev_{tf}_close')
        df['tf_key'] = df['time'].dt.floor(freq)
        df = df.merge(df_tf, left_on='tf_key', right_index=True, how='left')
        df[f'{tf}_trend'] = (close - df[f'prev_{tf}_close']) / (df[f'prev_{tf}_close'] + 1e-9)
        df.drop(columns=['tf_key'], inplace=True)
    
    df['prev_close_bid'] = close.shift(1)
    df['prev_close_ask'] = df['close_ask'].shift(1)
    return df.dropna()

def get_xy(keys, d_dict, features, horizon):
    X, yl, ys = [], [], []
    for k in keys:
        df_f = add_features(d_dict[k])
        if len(df_f) > horizon + 10:
            p = df_f['close_bid'].values
            X.append(df_f[features].values[:-horizon])
            yl.append([(np.max(p[i+1:i+1+horizon]) - p[i])/p[i] for i in range(len(df_f)-horizon)])
            ys.append([(p[i] - np.min(p[i+1:i+1+horizon]))/p[i] for i in range(len(df_f)-horizon)])
    return (np.vstack(X), np.concatenate(yl), np.concatenate(ys)) if X else (None, None, None)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
print(f"--- START SNIPER V2 (Spearman > {MIN_SPEARMAN}) ---")

f_cols = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'atr']
HORIZON = 30
output_dir = "OIL_CRUDE/Trading_details"
os.makedirs(output_dir, exist_ok=True)
log_path = os.path.join(output_dir, "trading_logs.csv")

df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'], format='ISO8601')
df_raw = df_raw.sort_values('time')
df_raw['trading_day'] = (df_raw['time'].diff() > pd.Timedelta(hours=4)).cumsum()
dag_dict = {f'dag_{i}': d.reset_index(drop=True) for i, (day, d) in enumerate(df_raw.groupby('trading_day'), start=1)}
sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

new_records = []
for current_key in sorted_keys:
    idx = sorted_keys.index(current_key)
    history_keys = sorted_keys[max(0, idx - HISTORY_DAYS):idx]
    if len(history_keys) < MIN_HISTORY_NEEDED: continue
    
    X_tr, yl_tr, ys_tr = get_xy(history_keys[:int(len(history_keys)*0.75)], dag_dict, f_cols, HORIZON)
    X_val, yl_val, ys_val = get_xy(history_keys[int(len(history_keys)*0.75):], dag_dict, f_cols, HORIZON)
    if X_tr is None or X_val is None: continue

    m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, yl_tr)
    m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, ys_tr)
    
    # Validatie voor Spearman drempel
    corr_l, _ = spearmanr(m_l.predict(X_val), yl_val)
    corr_s, _ = spearmanr(m_s.predict(X_val), ys_val)

    df_day = add_features(dag_dict[current_key]).reset_index(drop=True)
    if df_day.empty: continue
    
    p_l, p_s = m_l.predict(df_day[f_cols].values), m_s.predict(df_day[f_cols].values)
    bids, asks, times, hours = df_day['close_bid'].values, df_day['close_ask'].values, df_day['time'].values, df_day['hour'].values
    
    # Bereken drempelwaarde voor signalen (96e percentiel)
    thresh_l = np.percentile(m_l.predict(X_tr), SIGNAL_PERCENTILE)
    thresh_s = np.percentile(m_s.predict(X_tr), SIGNAL_PERCENTILE)

    active = False
    for j in range(len(bids) - 1):
        if not active:
            # Check Long
            if (corr_l >= MIN_SPEARMAN and p_l[j] > thresh_l):
                ent_p, side, active, ent_t = df_day['prev_close_ask'].values[j], 1, True, times[j]
                curr_rec = {"day": current_key, "entry_time": str(ent_t), "side": "Long", "entry_p": ent_p}
                curr_sl = INITIAL_STOP_LOSS
            # Check Short
            elif (corr_s >= MIN_SPEARMAN and p_s[j] > thresh_s):
                ent_p, side, active, ent_t = df_day['prev_close_bid'].values[j], -1, True, times[j]
                curr_rec = {"day": current_key, "entry_time": str(ent_t), "side": "Short", "entry_p": ent_p}
                curr_sl = INITIAL_STOP_LOSS
        else:
            r = (bids[j] - ent_p) / ent_p if side == 1 else (ent_p - asks[j]) / ent_p
            dur = (times[j] - ent_t).seconds // 60
            if r >= 0.0025: curr_sl = max(curr_sl, r - 0.002) # Trailing SL activering
            
            # EXIT CHECK
            reason = None
            if r >= TARGET_TP: reason = "TP"
            elif r <= curr_sl: reason = "SL"
            elif dur >= MAX_TRADE_MINUTES: reason = "TIME"
            elif hours[j] >= 23: reason = "EOD"
            
            if reason:
                curr_rec.update({"exit_time": str(times[j]), "return": r, "exit_reason": reason})
                new_records.append(curr_rec); active = False; break

# Opslaan van resultaten (met header-garantie tegen lege CSV errors)
res_df = pd.DataFrame(new_records)
if res_df.empty:
    res_df = pd.DataFrame(columns=["day", "entry_time", "side", "entry_p", "exit_time", "return", "exit_reason"])
res_df.to_csv(log_path, index=False)

# Eenvoudige plot voor visuele controle
if not res_df.empty and "return" in res_df.columns:
    plt.figure(figsize=(10,5))
    plt.plot(res_df['return'].cumsum())
    plt.title(f"Cumulative Return (Spearman > {MIN_SPEARMAN})")
    plt.savefig(os.path.join(output_dir, "performance_plot.png"))

print("--- ANALYSE VOLTOOID ---")
