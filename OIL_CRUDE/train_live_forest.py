import requests
import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from datetime import datetime

# ==============================================================================
# 1. DATA & FEATURES (Identiek aan Backtest)
# ==============================================================================
def read_latest_csv_from_crudeoil():
    user = "Stijnknoop"
    repo = "crudeoil"
    folder_path = "OIL_CRUDE"  # ✅ Geef hier de map aan
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    
    # ✅ De API URL bevat nu de folder_path
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder_path}?ref=master"
    
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}. Bestaat de map '{folder_path}' al in de repo?")
    
    files = response.json()
    
    # ✅ Filter op CSV files en sorteer op naam (zodat de nieuwste timestamp bovenaan komt)
    csv_files = [f for f in files if f['name'].endswith('.csv')]
    
    if not csv_files:
        raise Exception(f"Geen CSV bestanden gevonden in de map {folder_path}")
    
    # Sorteren op naam (omgekeerd), nieuwste bestand staat nu op index 0
    csv_files.sort(key=lambda x: x['name'], reverse=True)
    csv_file = csv_files[0]
    
    print(f"Inladen van: {csv_file['name']} uit de map {folder_path}")
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
    if np.isnan(correlation_score) or correlation_score < 0.01: return 99.9
    elif correlation_score < 0.05: return 98.0
    elif correlation_score < 0.10: return 96.0
    else: return 94.0

# ==============================================================================
# 2. TRAINING VOOR PRODUCTIE (Morgen)
# ==============================================================================
print("--- START PRODUCTIE TRAINING ---")
df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'], format='ISO8601')
df_raw = df_raw.sort_values('time')

# Data groeperen (Identiek aan backtest)
full_range = pd.date_range(df_raw['time'].min(), df_raw['time'].max(), freq='min')
df = pd.DataFrame({'time': full_range}).merge(df_raw, on='time', how='left')
df['has_data'] = ~df['open_bid'].isna()
df = df.set_index('time')
df[df.columns.difference(['has_data'])] = df[df.columns.difference(['has_data'])].ffill(limit=5)
df = df.reset_index()

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

# Neem de LAATSTE 40 dagen om het model voor MORGEN te trainen
history_keys = sorted_keys[-40:]
print(f"Training op history: {history_keys[0]} t/m {history_keys[-1]}")

X_tr, yl_tr, ys_tr = get_xy(history_keys[:int(len(history_keys)*0.75)], dag_dict)
X_val, yl_val, ys_val = get_xy(history_keys[int(len(history_keys)*0.75):], dag_dict)

m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, yl_tr)
m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, ys_tr)

# Thresholds bepalen
corr_l, _ = spearmanr(m_l.predict(X_val), yl_val)
corr_s, _ = spearmanr(m_s.predict(X_val), ys_val)
pct_l, pct_s = calculate_dynamic_threshold(corr_l), calculate_dynamic_threshold(corr_s)

t_l = np.percentile(m_l.predict(X_tr), pct_l)
t_s = np.percentile(m_s.predict(X_tr), pct_s)

# Opslaan voor Live Trader

# --- HIER KOMT DE AANPASSING ---
# 1. Definieer de map
model_map = "OIL_CRUDE/Model"

# 2. Maak de map aan (nu gebruik je de variabele)
os.makedirs(model_map, exist_ok=True)

# 3. Definieer het volledige pad naar het bestand
model_file_path = os.path.join(model_map, "live_trading_model.pkl")

# 4. Opslaan voor Live Trader (gebruik het nieuwe pad)
joblib.dump({
    "model_l": m_l, 
    "model_s": m_s, 
    "t_l": t_l, 
    "t_s": t_s,
    "f_selected": ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour'],
    "trained_on_until": history_keys[-1]
}, model_file_path)

print(f"Model succesvol opgeslagen in {model_file_path}. T_L: {t_l:.6f}, T_S: {t_s:.6f}")
