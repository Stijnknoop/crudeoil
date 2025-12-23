import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import re
import os

# ==================================================
# 1. DATA OPHALEN & DATUM-GEBASEERDE VERWERKING
# ==================================================
def read_latest_csv_from_crudeoil():
    user, repo, branch = "Stijnknoop", "crudeoil", "master"
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref={branch}"
    response = requests.get(api_url)
    if response.status_code != 200: 
        raise Exception(f"GitHub API error: {response.status_code}")
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    return pd.read_csv(csv_file['download_url'])

def prepare_trading_days():
    df = read_latest_csv_from_crudeoil()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # Gap-vulling en datum-groepering
    full_range = pd.date_range(df['time'].min(), df['time'].max(), freq='min')
    df = pd.DataFrame({'time': full_range}).merge(df, on='time', how='left')
    df['has_data'] = ~df['open_bid'].isna()
    df['date'] = df['time'].dt.date
    
    valid_dates = df.groupby('date')['has_data'].any()
    df = df[df['date'].isin(valid_dates[valid_dates].index)].copy()
    
    cols_to_ffill = df.columns.difference(['time', 'has_data', 'date'])
    df[cols_to_ffill] = df.groupby(df['has_data'].cumsum())[cols_to_ffill].ffill()
    
    dag_dict = {}
    for date, group in df.groupby('date'):
        if len(group[group['has_data']]) > 200:
            dag_dict[str(date)] = group.reset_index(drop=True)
    return dag_dict

# ==================================================
# 2. FEATURE ENGINEERING
# ==================================================
def add_features(df):
    df = df.copy()
    df['hour'], df['minute'] = df['time'].dt.hour, df['time'].dt.minute
    df['day_progression'] = np.clip((df['hour'] * 60 + df['minute']) / 1380.0, 0, 1)
    df['volatility_proxy'] = (df['high_bid'] - df['low_bid']).rolling(15).mean() / (df['close_bid'] + 1e-9)
    df['z_score_30m'] = (df['close_bid'] - df['close_bid'].rolling(30).mean()) / (df['close_bid'].rolling(30).std() + 1e-9)
    df['macd'] = df['close_bid'].ewm(span=12).mean() - df['close_bid'].ewm(span=26).mean()
    
    delta = df['close_bid'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    for tf in ['15min', '1h']:
        freq = '15min' if tf == '15min' else '60min'
        df_tf = df.resample(freq, on='time').agg({'close_bid': 'ohlc'}).shift(1)
        df_tf.columns = [f'{tf}_{c}' for c in ['open', 'high', 'low', 'close']]
        df['temp_key'] = df['time'].dt.floor(freq)
        df = df.merge(df_tf, left_on='temp_key', right_index=True, how='left').drop(columns=['temp_key'])
        df[f'{tf}_trend'] = (df['close_bid'] - df[f'{tf}_close']) / (df[f'{tf}_close'] + 1e-9)
    return df.ffill().bfill()

def get_xy(keys, d_dict, f_selected):
    X, yl, ys = [], [], []
    for k in keys:
        df_f = add_features(d_dict[k]).dropna()
        p = df_f['close_bid'].values
        # Horizon van 30 minuten voor targets
        t_l = [(np.max(p[i+1:i+31]) - p[i])/p[i] for i in range(len(df_f)-30)]
        t_s = [(p[i] - np.min(p[i+1:i+31]))/p[i] for i in range(len(df_f)-30)]
        X.append(df_f[f_selected].values[:-30]); yl.append(t_l); ys.append(t_s)
    return np.vstack(X), np.concatenate(yl), np.concatenate(ys)

# ==================================================
# 3. SLIDING WINDOW LOGICA & RAPPORTAGE
# ==================================================


if __name__ == "__main__":
    dag_dict = prepare_trading_days()
    sorted_keys = sorted(dag_dict.keys())
    
    # Vaste venstergroottes
    TRAIN_SIZE = 30 
    VAL_SIZE   = 10 
    total_needed = TRAIN_SIZE + VAL_SIZE
    
    # We kunnen pas testen vanaf de dag dat we genoeg historie hebben
    test_keys = sorted_keys[total_needed:]
    
    if not test_keys:
        print(f"Nog niet genoeg data ({len(sorted_keys)} dgn). Minimaal {total_needed + 1} nodig.")
        exit(0)

    f_selected = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
    equity_val = 10000.0
    daily_logs = []

    for current_date in test_keys:
        curr_idx = sorted_keys.index(current_date)
        
        # SLIDING WINDOW: 
        # Train op [nu-40 tot nu-10]
        # Valideer op [nu-10 tot nu]
        train_keys = sorted_keys[curr_idx - total_needed : curr_idx - VAL_SIZE]
        val_keys   = sorted_keys[curr_idx - VAL_SIZE : curr_idx]
        
        # Model training
        X_t, yl_t, ys_t = get_xy(train_keys, dag_dict, f_selected)
        m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_t, yl_t)
        m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_t, ys_t)
        
        # Threshold validatie op de 10 meest recente dagen
        X_v, _, _ = get_xy(val_keys, dag_dict, f_selected)
        t_l, t_s = np.percentile(m_l.predict(X_v), 97), np.percentile(m_s.predict(X_v), 97)
        
        # Handelen op de huidige dag
        df_day = add_features(dag_dict[current_date]).dropna()
        pl, ps = m_l.predict(df_day[f_selected].values[:-30]), m_s.predict(df_day[f_selected].values[:-30])
        prices, times, hours = df_day['close_bid'].values, df_day['time'].values, df_day['hour'].values
        
        day_ret, active, current_sl = 0, False, -0.004
        trade_info = {"date": current_date, "type": "None", "entry_p": None, "exit_p": None, "pct": 0, "entry_t": None, "exit_t": None}

        for j in range(len(pl)):
            if not active:
                if hours[j] < 23:
                    if pl[j] > t_l: 
                        ent_p, side, active = prices[j], 1, True
                        trade_info.update({"type": "LONG", "entry_p": prices[j], "entry_t": times[j]})
                    elif ps[j] > t_s: 
                        ent_p, side, active = prices[j], -1, True
                        trade_info.update({"type": "SHORT", "entry_p": prices[j], "entry_t": times[j]})
            else:
                r = ((prices[j] - ent_p) / ent_p) * side
                if r >= 0.0025: current_sl = max(current_sl, r - 0.002)
                if r >= 0.005 or r <= current_sl or j == len(pl)-1 or hours[j] >= 23:
                    day_ret = r
                    trade_info.update({"exit_p": prices[j], "exit_t": times[j], "pct": r})
                    active = False; break
        
        # Bereken balans (multiplier 5)
        equity_val *= (1 + (day_ret * 5))
        trade_info.update({"dollar_profit": equity_val - (equity_val / (1 + (day_ret * 5))), "balance": equity_val})
        daily_logs.append(trade_info)

    # --- EXPORT & VISUALISATIE ---
    df_res = pd.DataFrame(daily_logs)
    df_res.to_csv('trading_log.csv', index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    last_date = test_keys[-1]
    df_plot = add_features(dag_dict[last_date]).dropna()
    last_trade = daily_logs[-1]
    
    # Plot 1: Laatste dag trades
    ax1.plot(df_plot['time'], df_plot['close_bid'], color='silver', label='Olie Prijs')
    if last_trade["type"] != "None":
        c = 'limegreen' if last_trade["type"] == "LONG" else 'crimson'
        ax1.scatter(last_trade["entry_t"], last_trade["entry_p"], color=c, marker='^', s=150, label='Entry')
        ax1.scatter(last_trade["exit_t"], last_trade["exit_p"], color='black', marker='x', s=150, label='Exit')
    ax1.set_title(f"Laatste Dag: {last_date} | Profit: ${last_trade['dollar_profit']:.2f}")
    ax1.legend(); ax1.grid(True, alpha=0.2)

    # Plot 2: Equity Curve
    
    ax2.plot(pd.to_datetime(df_res['date']), df_res['balance'], color='dodgerblue', lw=3)
    ax2.fill_between(pd.to_datetime(df_res['date']), 10000, df_res['balance'], color='dodgerblue', alpha=0.1)
    ax2.set_title(f"Trendverloop Balance (Eindstand: ${equity_val:.2f})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('last_day_plot.png')
    plt.close()
    print(f"Succes! Rapport voor {last_date} opgeslagen.")
