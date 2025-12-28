import requests
import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

# Zorg dat matplotlib de backend niet nodig heeft (geen GUI)
import matplotlib
matplotlib.use('Agg')

# ==============================================================================
# 1. FUNCTIES (KOPIE VAN ORIGINELE LOGICA)
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
    f_cols = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
    df[f_cols] = df[f_cols].shift(1)
    return df.dropna()

def get_xy(keys, d_dict, f_selected, HORIZON=30):
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
# 2. DATA PREPARATIE & VERIFICATIE LOGICA
# ==============================================================================

print("--- START VERIFICATIE ANALYSE ---")
df_raw = read_latest_csv_from_crudeoil()
df_raw['time'] = pd.to_datetime(df_raw['time'])
df_raw = df_raw.sort_values('time')

# Data groeperen in dagen (ffill logica)
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

# Gap-logica voor trading_day definitie
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

# VERIFICATIE SETTINGS:
test_day_key = sorted_keys[-1]      # De allerlaatste dag (die we willen testen)
history_keys = sorted_keys[-41:-1] # De 40 dagen DAARVOOR (voor training)

print(f"Test Dag: {test_day_key}")
print(f"Trainings-horizon: {history_keys[0]} t/m {history_keys[-1]} ({len(history_keys)} dagen)")

# ==============================================================================
# 3. TRAINING & VALIDATIE (MODEL BOUWEN)
# ==============================================================================

# Split 75% train / 25% val binnen de historie
split_point = int(len(history_keys) * 0.75)
train_keys = history_keys[:split_point]
val_keys = history_keys[split_point:]

f_selected = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']

# Haal XY data op
X_tr, yl_tr, ys_tr = get_xy(train_keys, dag_dict, f_selected)
X_val, yl_val, ys_val = get_xy(val_keys, dag_dict, f_selected)

if X_tr is not None:
    # 1. Train Modellen
    m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, yl_tr)
    m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, ys_tr)

    # 2. Bepaal Dynamische Thresholds via Validatie set
    if X_val is not None:
        pred_val_l = m_l.predict(X_val)
        pred_val_s = m_s.predict(X_val)
        corr_l, _ = spearmanr(pred_val_l, yl_val)
        corr_s, _ = spearmanr(pred_val_s, ys_val)
        pct_l = calculate_dynamic_threshold(corr_l)
        pct_s = calculate_dynamic_threshold(corr_s)
    else:
        pct_l, pct_s = 96.0, 96.0

    # 3. Bereken de harde threshold waarden op basis van training-distributie
    pred_tr_l = m_l.predict(X_tr)
    pred_tr_s = m_s.predict(X_tr)
    t_l = np.percentile(pred_tr_l, pct_l)
    t_s = np.percentile(pred_tr_s, pct_s)

    print(f"Gevalideerde Thresholds -> Long: {t_l:.6f} (pct {pct_l}), Short: {t_s:.6f} (pct {pct_s})")

    # ==============================================================================
    # 4. SIMULATIE TEST DAG (KOMT DIT OVEREEN MET LOGS?)
    # ==============================================================================
    
    df_test_day = add_features(dag_dict[test_day_key]).reset_index(drop=True)
    p_l = m_l.predict(df_test_day[f_selected].values)
    p_s = m_s.predict(df_test_day[f_selected].values)
    
    times = df_test_day['time'].values
    prices = df_test_day['close_bid'].values
    hours = df_test_day['hour'].values

    trade_found = False
    for j in range(len(p_l)):
        # Alleen trades voor 23:00 (EOD logica)
        if hours[j] < 23:
            if p_l[j] > t_l:
                print(f"\n[SIGNALEER] LONG gedetecteerd op {times[j]}")
                print(f"Voorspelling: {p_l[j]:.6f} > Threshold: {t_l:.6f}")
                trade_found = True
                break
            elif p_s[j] > t_s:
                print(f"\n[SIGNALEER] SHORT gedetecteerd op {times[j]}")
                print(f"Voorspelling: {p_s[j]:.6f} > Threshold: {t_s:.6f}")
                trade_found = True
                break
    
    if not trade_found:
        print("\n[RESULTAAT] Geen trade gevonden voor deze dag.")
    
    # 5. OPSLAAN ALS TEST (OPTIONEEL)
    os.makedirs("Model", exist_ok=True)
    save_data = {"model_l": m_l, "model_s": m_s, "t_l": t_l, "t_s": t_s}
    joblib.dump(save_data, "Model/verify_test_model.pkl")
    print("\n--- VERIFICATIE VOLTOOID ---")
    print("Check nu je 'trading_logs.csv' en kijk of de laatste dag hetzelfde resultaat gaf.")
else:
    print("Fout: Kon geen trainingsdata genereren.")
