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
# 1. DATA INLEZEN MET AUTHENTICATIE
# ==================================================
def read_latest_csv_from_crudeoil():
    user = "Stijnknoop"
    repo = "crudeoil"
    branch = "master"
    
    # Gebruik een token om 403 Rate Limit errors te voorkomen in GitHub Actions
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref={branch}"
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        # Geef meer details bij fouten (zoals de 403)
        raise Exception(f"GitHub API error: {response.status_code} - {response.text}")
        
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    
    if not csv_file:
        raise Exception("Geen CSV-bestand gevonden in de repo.")
        
    return pd.read_csv(csv_file['download_url'])

print("Data ophalen...")
df_raw = read_latest_csv_from_crudeoil()

# --- Rest van de data verwerking (ongewijzigd t.o.v. vorige versie) ---
df_raw['time'] = pd.to_datetime(df_raw['time'])
df_raw = df_raw.sort_values('time')

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
    if len(next_idx) > 0:
        df.loc[next_idx[0]:, 'trading_day'] += 1

dag_dict = {f'dag_{i}': d[d['has_data']].reset_index(drop=True) 
            for i, (day, d) in enumerate(df.groupby('trading_day'), start=1)}

# ==================================================
# 2. FEATURE ENGINEERING (LEAK-FREE)
# ==================================================
def add_features(df_in):
    df = df_in.copy().sort_values('time')
    df['hour'] = df['time'].dt.hour
    df['day_progression'] = np.clip((df['hour'] * 60 + df['time'].dt.minute) / 1380.0, 0, 1)
    
    close = df['close_bid']
    df['volatility_proxy'] = (df['high_bid'] - df['low_bid']).rolling(15, min_periods=15).mean() / (close + 1e-9)
    ma30 = close.rolling(30, min_periods=30).mean()
    std30 = close.rolling(30, min_periods=30).std()
    df['z_score_30m'] = (close - ma30) / (std30 + 1e-9)
    df['macd'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    for tf, freq in [('15min', '15min'), ('1h', '60min')]:
        df_tf = df.set_index('time')['close_bid'].resample(freq).last().shift(1).rename(f'prev_{tf}_close')
        df['temp_key'] = df['time'].dt.floor(freq)
        df = df.merge(df_tf, left_on='temp_key', right_index=True, how='left').drop(columns=['temp_key'])
        df[f'{tf}_trend'] = (close - df[f'prev_{tf}_close']) / (df[f'prev_{tf}_close'] + 1e-9)
    
    return df.dropna()

f_selected = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
HORIZON = 30

def get_xy(keys, d_dict):
    X, yl, ys = [], [], []
    for k in keys:
        df_f = add_features(d_dict[k])
        if len(df_f) > HORIZON + 10:
            p = df_f['close_bid'].values
            t_l = [(np.max(p[i+1:i+1+HORIZON]) - p[i])/p[i] for i in range(len(df_f)-HORIZON)]
            t_s = [(p[i] - np.min(p[i+1:i+1+HORIZON]))/p[i] for i in range(len(df_f)-HORIZON)]
            X.append(df_f[f_selected].values[:-HORIZON])
            yl.append(t_l); ys.append(t_s)
    return (np.vstack(X), np.concatenate(yl), np.concatenate(ys)) if X else (None, None, None)

# ==================================================
# 3. DIRECTORIES VOOR OUTPUT
# ==================================================
output_dir = "Trading_details"
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ==================================================
# 4. WALK-FORWARD MET BID-ASK LOGICA
# ==================================================
sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
if len(sorted_keys) > 60: sorted_keys = sorted_keys[-60:]

WINDOW_TRAIN = int(len(sorted_keys) * 0.6)
WINDOW_VAL   = int(len(sorted_keys) * 0.2)
test_keys    = sorted_keys[WINDOW_TRAIN + WINDOW_VAL:]

trade_logs, equity = [], [1.0]
BEST_TP, BEST_SL = 0.005, -0.004
RISK, TRAILING_ACT = 0.02, 0.0025

for i, current_key in enumerate(test_keys):
    train_end = i + WINDOW_TRAIN
    val_end = train_end + WINDOW_VAL
    X_tr, yl_tr, ys_tr = get_xy(sorted_keys[i:train_end], dag_dict)
    X_vl, _, _ = get_xy(sorted_keys[train_end:val_end], dag_dict)
    
    m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_tr, yl_tr)
    m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_tr, ys_tr)
    t_l, t_s = np.percentile(m_l.predict(X_vl), 97), np.percentile(m_s.predict(X_vl), 97)
    
    df_day = add_features(dag_dict[current_key]).reset_index(drop=True)
    p_l, p_s = m_l.predict(df_day[f_selected].values), m_s.predict(df_day[f_selected].values)
    
    bids, asks = df_day['close_bid'].values, df_day['close_ask'].values
    times, hours = df_day['time'].values, df_day['hour'].values
    
    day_ret, active = 0, False
    trade_info = {"day": current_key, "entry_time": None, "exit_time": None, "side": None, "entry_p": None, "exit_p": None, "return": 0}

    for j in range(len(bids) - 1):
        if not active:
            if hours[j] < 23:
                if p_l[j] > t_l:
                    ent_p, side, active = asks[j], 1, True
                    trade_info.update({"entry_time": str(times[j]), "side": "Long", "entry_p": ent_p})
                    curr_sl = BEST_SL
                elif p_s[j] > t_s:
                    ent_p, side, active = bids[j], -1, True
                    trade_info.update({"entry_time": str(times[j]), "side": "Short", "entry_p": ent_p})
                    curr_sl = BEST_SL
        else:
            r = ((bids[j] - ent_p) / ent_p) if side == 1 else ((ent_p - asks[j]) / ent_p)
            if r >= TRAILING_ACT: curr_sl = max(curr_sl, r - 0.002)
            if r >= BEST_TP or r <= curr_sl or hours[j] >= 23 or j == len(bids)-2:
                day_ret = r
                exit_p = bids[j] if side == 1 else asks[j]
                trade_info.update({"exit_time": str(times[j]), "exit_p": exit_p, "return": r})
                trade_logs.append(trade_info)
                active = False
                break

    # DAGELIJKSE PLOT OPSLAAN
    plt.figure(figsize=(10, 4))
    plt.plot(df_day['time'], bids, color='black', alpha=0.3, label='Bid Price')
    if trade_info["entry_time"]:
        c = 'green' if trade_info["return"] > 0 else 'red'
        plt.scatter(df_day.loc[df_day['time'] == pd.to_datetime(trade_info["entry_time"]), 'time'], trade_info["entry_p"], marker='^', color='blue', label='Entry', s=100)
        plt.scatter(df_day.loc[df_day['time'] == pd.to_datetime(trade_info["exit_time"]), 'time'], trade_info["exit_p"], marker='x', color=c, label='Exit', s=100)
    plt.title(f"{current_key} | Ret: {day_ret:.4%}")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"plot_{current_key}.png"))
    plt.close()

    equity.append(equity[-1] * (1 + (day_ret * (RISK/abs(BEST_SL)))))

# ==================================================
# 5. RESULTATEN OPSLAAN
# ==================================================
df_logs = pd.DataFrame(trade_logs)
df_logs.to_csv(os.path.join(output_dir, "trading_logs.csv"), index=False)

plt.figure(figsize=(12, 6))
plt.plot(equity, label='AI Strategy (Bid-Ask)', lw=2, color='navy')
plt.title("Totaal Rendement Walk-Forward Overview")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "latest_equity_overview.png"))
plt.close()

print(f"Klaar! Resultaten opgeslagen in {output_dir}")
