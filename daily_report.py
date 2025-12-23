import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import re
import os

# --------------------------------------------------
# 1. DATA OPHALEN & VOORBEREIDEN
# --------------------------------------------------
def read_latest_csv_from_crudeoil():
    user, repo, branch = "Stijnknoop", "crudeoil", "master"
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref={branch}"
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")
    
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    if not csv_file:
        raise Exception("Geen CSV-bestand gevonden")
    return pd.read_csv(csv_file['download_url'])

def prepare_trading_days():
    df = read_latest_csv_from_crudeoil()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    # Volledige reeks maken voor gap-detectie
    full_range = pd.date_range(df['time'].min(), df['time'].max(), freq='1T')
    df = pd.DataFrame({'time': full_range}).merge(df, on='time', how='left')
    df['has_data'] = ~df['open_bid'].isna()
    df['date'] = df['time'].dt.date

    # Filter dagen zonder data
    valid_dates = df.groupby('date')['has_data'].any()
    valid_dates = valid_dates[valid_dates].index
    df = df[df['date'].isin(valid_dates)].copy()

    # Forward fill binnen data-blokken
    cols_to_ffill = df.columns.difference(['time', 'has_data', 'date'])
    df[cols_to_ffill] = df.groupby(df['has_data'].cumsum())[cols_to_ffill].ffill()

    # Gap detectie (23u markt cyclus)
    df['gap_flag'] = (~df['has_data']) & (df['time'].dt.hour >= 20)
    df['gap_group'] = (df['gap_flag'] != df['gap_flag'].shift()).cumsum()
    gap_groups = df[df['gap_flag']].groupby('gap_group').agg(
        start_time=('time', 'first'), end_time=('time', 'last'), length=('time', 'count')
    )
    long_gaps = gap_groups[gap_groups['length'] >= 10]

    # Trading days markeren
    df['trading_day'] = 1
    for _, row in long_gaps.iterrows():
        next_idx = df.index[(df['time'] > row['end_time']) & (df['has_data'])]
        if len(next_idx) > 0: df.loc[next_idx[0]:, 'trading_day'] += 1

    df['market_open'] = True
    for _, row in long_gaps.iterrows():
        df.loc[(df['time'] >= row['start_time']) & (df['time'] <= row['end_time']), 'market_open'] = False

    # Naar dict en market_open filteren
    dag_dict = {f'dag_{i}': d[d['market_open']].reset_index(drop=True) 
                for i, (day, d) in enumerate(df.groupby('trading_day'), start=1)}
    return dag_dict

# --------------------------------------------------
# 2. MODEL FUNCTIES
# --------------------------------------------------
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

def get_xy(keys, d_dict, f_selected, horizon=30):
    X, yl, ys = [], [], []
    for k in keys:
        if len(d_dict[k]) > 200:
            df_f = add_features(d_dict[k]).dropna()
            p = df_f['close_bid'].values
            t_l = [(np.max(p[i+1:i+1+horizon]) - p[i])/p[i] for i in range(len(df_f)-horizon)]
            t_s = [(p[i] - np.min(p[i+1:i+1+horizon]))/p[i] for i in range(len(df_f)-horizon)]
            X.append(df_f[f_selected].values[:-horizon]); yl.append(t_l); ys.append(t_s)
    return np.vstack(X), np.concatenate(yl), np.concatenate(ys)

# --------------------------------------------------
# 3. MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    dag_dict = prepare_trading_days()
    sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    
    # Windows bepalen
    n_total = len(sorted_keys)
    WINDOW_TRAIN, WINDOW_VAL = int(n_total * 0.6), int(n_total * 0.2)
    test_keys = sorted_keys[WINDOW_TRAIN + WINDOW_VAL:]
    
    f_selected = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
    equity_val, START_CAPITAL = 10000.0, 10000.0
    BEST_TP, BEST_SL, RISK_PER_TRADE, TRAILING_ACT = 0.005, -0.004, 0.02, 0.0025
    
    daily_logs = []

    # Loop door alle testdagen voor de volledige log
    for i, current_test_key in enumerate(test_keys):
        train_idx_end = i + WINDOW_TRAIN
        val_idx_end = train_idx_end + WINDOW_VAL
        
        X_train_r, yl_train_r, ys_train_r = get_xy(sorted_keys[i:train_idx_end], dag_dict, f_selected)
        m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_train_r, yl_train_r)
        m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_train_r, ys_train_r)
        
        X_val_r, _, _ = get_xy(sorted_keys[train_idx_end:val_idx_end], dag_dict, f_selected)
        t_l, t_s = np.percentile(m_l.predict(X_val_r), 97), np.percentile(m_s.predict(X_val_r), 97)
        
        df_day = add_features(dag_dict[current_test_key]).dropna()
        pl, ps = m_l.predict(df_day[f_selected].values[:-30]), m_s.predict(df_day[f_selected].values[:-30])
        prices, times, hours = df_day['close_bid'].values, df_day['time'].values, df_day['hour'].values
        
        day_ret, active, current_sl = 0, False, BEST_SL
        trade_info = {"date": current_test_key, "type": "None", "entry_p": None, "exit_p": None, "pct": 0, "entry_t": None, "exit_t": None}

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
                if r >= TRAILING_ACT: current_sl = max(current_sl, r - 0.002)
                if r >= BEST_TP or r <= current_sl or j == len(pl)-1 or hours[j] >= 23:
                    day_ret = r
                    trade_info.update({"exit_p": prices[j], "exit_t": times[j], "pct": r})
                    active = False; break
        
        actual_gain_pct = day_ret * (RISK_PER_TRADE / abs(BEST_SL))
        dollar_profit = equity_val * actual_gain_pct
        equity_val += dollar_profit
        trade_info.update({"dollar_profit": dollar_profit, "balance": equity_val})
        daily_logs.append(trade_info)

    # --- EXPORT & PLOT LAATSTE DAG ---
    df_res = pd.DataFrame(daily_logs)
    df_res.to_csv('trading_log.csv', index=False)

    # Plot alleen de allerlaatste dag voor het rapport
    last_day_key = test_keys[-1]
    df_plot = add_features(dag_dict[last_day_key]).dropna()
    last_trade = daily_logs[-1]

    plt.figure(figsize=(14, 6))
    plt.plot(df_plot['time'], df_plot['close_bid'], color='silver', label='Olie Prijs')
    if last_trade["type"] != "None":
        c = 'limegreen' if last_trade["type"] == "LONG" else 'crimson'
        plt.scatter(last_trade["entry_t"], last_trade["entry_p"], color=c, marker='^', s=150, label='Entry')
        plt.scatter(last_trade["exit_t"], last_trade["exit_p"], color='black', marker='x', s=150, label='Exit')
    plt.title(f"Daily Report: {last_day_key} | Winst: ${last_trade['dollar_profit']:.2f}")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig('last_day_plot.png')
    plt.close()
    
    print(f"Rapport gegenereerd voor {last_day_key}. Balans: ${equity_val:.2f}")
