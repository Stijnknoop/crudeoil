import requests
import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

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
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    return pd.read_csv(csv_file['download_url'])

print("--- START ANALYSE (SPEARMAN > 0.30 SNIPER MODE) ---")
df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'], format='ISO8601')
df_raw = df_raw.sort_values('time')

# Data voorbereiding
full_range = pd.date_range(df_raw['time'].min(), df_raw['time'].max(), freq='min')
df = pd.DataFrame({'time': full_range}).merge(df_raw, on='time', how='left')
df['has_data'] = ~df['open_bid'].isna()
df = df.set_index('time')
df[df.columns.difference(['has_data'])] = df[df.columns.difference(['has_data'])].ffill(limit=5)
df = df.reset_index()

df['date'] = df['time'].dt.date
valid_dates = df.groupby('date')['has_data'].any()
valid_dates = valid_dates[valid_dates].index
df = df[df['date'].isin(valid_dates)].copy()

# Gaps detecteren voor handelsdagen
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

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
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
    
    df['prev_close_bid'] = df['close_bid'].shift(1)
    df['prev_close_ask'] = df['close_ask'].shift(1)

    f_selected = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
    df[f_selected] = df[f_selected].shift(1)
    return df.dropna()

f_cols = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
HORIZON = 30

def get_xy(keys, d_dict):
    X, yl, ys = [], [], []
    for k in keys:
        df_f = add_features(d_dict[k])
        if len(df_f) > HORIZON + 10:
            p = df_f['close_bid'].values
            X.append(df_f[f_cols].values[:-HORIZON])
            yl.append([(np.max(p[i+1:i+1+HORIZON]) - p[i])/p[i] for i in range(len(df_f)-HORIZON)])
            ys.append([(p[i] - np.min(p[i+1:i+1+HORIZON]))/p[i] for i in range(len(df_f)-HORIZON)])
    return (np.vstack(X), np.concatenate(yl), np.concatenate(ys)) if X else (None, None, None)

# ==============================================================================
# 3. TRADING LOOP MET HARDE SPEARMAN EIS
# ==============================================================================
output_dir = "OIL_CRUDE/Trading_details"
log_path = os.path.join(output_dir, "trading_logs.csv")
insight_path = os.path.join(output_dir, "model_insights.csv")
os.makedirs(output_dir, exist_ok=True)

today_str = datetime.now().strftime('%Y-%m-%d')
sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

if os.path.exists(log_path):
    existing_logs = pd.read_csv(log_path)
    existing_logs['entry_time'] = pd.to_datetime(existing_logs['entry_time'], format='ISO8601', errors='coerce')
    mask = (existing_logs['exit_reason'] != "Data End (Pending)") & \
           (~existing_logs['entry_time'].dt.strftime('%Y-%m-%d').fillna('').str.contains(today_str))
    existing_logs = existing_logs[mask].copy()
    processed_days = set(existing_logs['day'].astype(str).tolist())
else:
    existing_logs, processed_days = pd.DataFrame(), set()

new_days = [k for k in sorted_keys if k not in processed_days]

if not new_days:
    print("Geen nieuwe dagen om te verwerken.")
else:
    new_records, insight_records = [], []
    
    for current_key in new_days:
        idx = sorted_keys.index(current_key)
        history_keys = sorted_keys[max(0, idx - 40):idx]
        
        if len(history_keys) < 20:
            continue
            
        X_tr, yl_tr, ys_tr = get_xy(history_keys[:int(len(history_keys)*0.75)], dag_dict)
        X_val, yl_val, ys_val = get_xy(history_keys[int(len(history_keys)*0.75):], dag_dict)
        
        if X_tr is None or X_val is None: continue

        m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, yl_tr)
        m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, ys_tr)
        
        corr_l, _ = spearmanr(m_l.predict(X_val), yl_val)
        corr_s, _ = spearmanr(m_s.predict(X_val), ys_val)

        # HARDE EIS: Spearman > 0.30
        MIN_SPEARMAN = 0.30
        can_trade_long = corr_l >= MIN_SPEARMAN
        can_trade_short = corr_s >= MIN_SPEARMAN
        
        # We gebruiken een streng percentiel (96e) om alleen top-signalen te pakken
        t_l = np.percentile(m_l.predict(X_tr), 96.0)
        t_s = np.percentile(m_s.predict(X_tr), 96.0)

        df_day = add_features(dag_dict[current_key]).reset_index(drop=True)
        active, day_res = False, {"day": current_key, "return": 0, "exit_reason": "No Trade", "entry_time": str(df_day['time'].iloc[0])}
        
        conflicts = 0
        if not can_trade_long and not can_trade_short:
            day_res["exit_reason"] = f"Blocked (Spearman L:{corr_l:.2f} S:{corr_s:.2f})"
        else:
            p_l, p_s = m_l.predict(df_day[f_cols].values), m_s.predict(df_day[f_cols].values)
            bids, asks = df_day['close_bid'].values, df_day['close_ask'].values
            prev_bids, prev_asks = df_day['prev_close_bid'].values, df_day['prev_close_ask'].values
            times, hours = df_day['time'].values, df_day['hour'].values

            for j in range(len(bids) - 1):
                if not active:
                    if hours[j] < 23:
                        trig_l = can_trade_long and p_l[j] > t_l
                        trig_s = can_trade_short and p_s[j] > t_s
                        
                        if trig_l and trig_s:
                            conflicts += 1
                            continue
                        
                        if trig_l:
                            ent_p, side, active = prev_asks[j], 1, True 
                            day_res.update({"entry_time": str(times[j]), "side": "Long", "entry_p": ent_p})
                            curr_sl = -0.004
                        elif trig_s:
                            ent_p, side, active = prev_bids[j], -1, True 
                            day_res.update({"entry_time": str(times[j]), "side": "Short", "entry_p": ent_p})
                            curr_sl = -0.004
                else:
                    r = (bids[j] - ent_p) / ent_p if side == 1 else (ent_p - asks[j]) / ent_p
                    if r >= 0.0025: curr_sl = max(curr_sl, r - 0.002)
                    
                    if r >= 0.005 or r <= curr_sl or hours[j] >= 23 or j == len(bids)-2:
                        day_res.update({"exit_time": str(times[j]), "exit_p": bids[j] if side == 1 else asks[j], "return": r, "exit_reason": "TP/SL/EOD"})
                        active = False; break
                    
        new_records.append(day_res)
        insight_records.append({
            "target_day": current_key, "corr_l": round(corr_l, 4), "corr_s": round(corr_s, 4),
            "min_p_l": round(t_l, 6), "min_p_s": round(t_s, 6), "conflicts": conflicts,
            "status": "Eligible" if (can_trade_long or can_trade_short) else "Blocked"
        })

    # Opslaan
    final_df = pd.concat([existing_logs, pd.DataFrame(new_records)], ignore_index=True)
    final_df.to_csv(log_path, index=False)
    
    new_ins_df = pd.DataFrame(insight_records)
    if os.path.exists(insight_path):
        new_ins_df = pd.concat([pd.read_csv(insight_path), new_ins_df], ignore_index=True).drop_duplicates(subset=['target_day'])
    new_ins_df.to_csv(insight_path, index=False)
    print("--- VOLTOOID ---")
