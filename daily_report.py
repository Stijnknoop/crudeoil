import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from sklearn.ensemble import RandomForestRegressor

# Voorkom GUI-fouten op GitHub Actions
import matplotlib
matplotlib.use('Agg')

# ==================================================
# 1. DATA INLEZEN
# ==================================================
def read_latest_csv_from_crudeoil():
    user = "Stijnknoop"
    repo = "crudeoil"
    branch = "master"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref={branch}"
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    return pd.read_csv(csv_file['download_url'])

print("Data ophalen...")
df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'])
df_raw = df_raw.sort_values('time')

# Gaps & Dagen splitsen
full_range = pd.date_range(df_raw['time'].min(), df_raw['time'].max(), freq='1T')
df = pd.DataFrame({'time': full_range}).merge(df_raw, on='time', how='left')
df['has_data'] = ~df['open_bid'].isna()
df['date'] = df['time'].dt.date
valid_dates = df.groupby('date')['has_data'].any()
valid_dates = valid_dates[valid_dates].index
df = df[df['date'].isin(valid_dates)].copy()
cols_to_ffill = df.columns.difference(['time', 'has_data', 'date'])
df[cols_to_ffill] = df.groupby(df['has_data'].cumsum())[cols_to_ffill].ffill()

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

# ==================================================
# 2. FEATURE ENGINEERING (ANTI-LEAKAGE)
# ==================================================
def add_features(df_in):
    df = df_in.copy().sort_values('time')
    df['hour'] = df['time'].dt.hour
    df['day_progression'] = np.clip((df['hour'] * 60 + df['time'].dt.minute) / 1380.0, 0, 1)
    
    close = df['close_bid']
    df['volatility_proxy'] = (df['high_bid'] - df['low_bid']).rolling(15, min_periods=15).mean() / (close + 1e-9)
    ma30, std30 = close.rolling(30).mean(), close.rolling(30).std()
    df['z_score_30m'] = (close - ma30) / (std30 + 1e-9)
    df['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    for tf, freq in [('15min', '15min'), ('1h', '60min')]:
        # Gebruik shift(1) op de resampled data om future leak te voorkomen
        df_tf = df.set_index('time')['close_bid'].resample(freq).last().shift(1).rename(f'prev_{tf}_close')
        df['tf_key'] = df['time'].dt.floor(freq)
        df = df.merge(df_tf, left_on='tf_key', right_index=True, how='left')
        df[f'{tf}_trend'] = (close - df[f'prev_{tf}_close']) / (df[f'prev_{tf}_close'] + 1e-9)
        df.drop(columns=['tf_key'], inplace=True)

    # CRUCIALE STAP: Shift alle features met 1 minuut. 
    # Voorspelling voor minuut T mag alleen data van T-1 gebruiken.
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

# ==================================================
# 3. INCREMENTELE LOGICA & PLOTS
# ==================================================
output_dir = "Trading_details"
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
log_path = os.path.join(output_dir, "trading_logs.csv")

if os.path.exists(log_path):
    existing_logs = pd.read_csv(log_path)
    processed_days = set(existing_logs['day'].astype(str).tolist())
else:
    existing_logs, processed_days = pd.DataFrame(), set()

sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
new_days = [k for k in sorted_keys if k not in processed_days]

if not new_days:
    print("Geen nieuwe dagen.")
else:
    print(f"Nieuwe dagen: {new_days}")
    new_trade_records = []
    BEST_TP, BEST_SL = 0.005, -0.004
    
    for current_key in new_days:
        idx = sorted_keys.index(current_key)
        if idx < 40: continue
        
        # Training & Threshold bepaling
        train_keys, val_keys = sorted_keys[max(0, idx-40):idx-5], sorted_keys[idx-5:idx]
        X_tr, yl_tr, ys_tr = get_xy(train_keys, dag_dict)
        X_vl, _, _ = get_xy(val_keys, dag_dict)
        
        m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_tr, yl_tr)
        m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_tr, ys_tr)
        t_l, t_s = np.percentile(m_l.predict(X_vl), 97), np.percentile(m_s.predict(X_vl), 97)
        
        df_day = add_features(dag_dict[current_key]).reset_index(drop=True)
        p_l, p_s = m_l.predict(df_day[f_selected].values), m_s.predict(df_day[f_selected].values)
        bids, asks, times, hours = df_day['close_bid'].values, df_day['close_ask'].values, df_day['time'].values, df_day['hour'].values
        
        active, day_res = False, {"day": current_key, "return": 0, "exit_reason": "No Trade"}
        for j in range(len(bids) - 1):
            if not active:
                if hours[j] < 23:
                    if p_l[j] > t_l:
                        ent_p, side, active = asks[j], 1, True
                        day_res.update({"entry_time": str(times[j]), "side": "Long", "entry_p": ent_p})
                        curr_sl = BEST_SL
                    elif p_s[j] > t_s:
                        ent_p, side, active = bids[j], -1, True
                        day_res.update({"entry_time": str(times[j]), "side": "Short", "entry_p": ent_p})
                        curr_sl = BEST_SL
            else:
                r = (bids[j] - ent_p) / ent_p if side == 1 else (ent_p - asks[j]) / ent_p
                if r >= 0.0025: curr_sl = max(curr_sl, r - 0.002)
                is_data_end, is_time_end = (j == len(bids)-2), (hours[j] >= 23)
                if r >= BEST_TP or r <= curr_sl or is_time_end or is_data_end:
                    reason = "TP/SL"
                    if is_time_end: reason = "EOD (23h)"
                    if is_data_end and not (r >= BEST_TP or r <= curr_sl): reason = "Data End (Pending)"
                    day_res.update({"exit_time": str(times[j]), "exit_p": bids[j] if side == 1 else asks[j], "return": r, "exit_reason": reason})
                    active = False; break
        
        new_trade_records.append(day_res)

        # Plot nieuwe dag
        plt.figure(figsize=(10, 4))
        plt.plot(df_day['time'], bids, color='black', alpha=0.3)
        if "entry_time" in day_res:
            c = 'green' if day_res["return"] > 0 else 'red'
            plt.scatter(pd.to_datetime(day_res["entry_time"]), day_res["entry_p"], marker='^', color='blue', s=100)
            plt.scatter(pd.to_datetime(day_res["exit_time"]), day_res["exit_p"], marker='x', color=c, s=100)
        plt.title(f"Day: {current_key} | Result: {day_res['return']:.4%}")
        plt.savefig(os.path.join(plots_dir, f"plot_{current_key}.png"))
        plt.close()

    updated_logs = pd.concat([existing_logs, pd.DataFrame(new_trade_records)], ignore_index=True)
    updated_logs.to_csv(log_path, index=False)
    existing_logs = updated_logs

# ==================================================
# 4. EQUITY OVERVIEW
# ==================================================
if not existing_logs.empty:
    equity = [1.0]
    for r in existing_logs['return'].values:
        # Risk model: risk 2% per trade op basis van de SL (0.4%)
        equity.append(equity[-1] * (1 + (r * (0.02 / 0.004))))
    plt.figure(figsize=(10, 5))
    plt.plot(equity, color='navy', lw=2)
    plt.title("Frozen Equity Curve (No Leakage)")
    plt.savefig(os.path.join(output_dir, "equity_overview.png"))
    plt.close()

print("Klaar! De backtest is nu 100% leak-free.")
