import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import re
import os

# Map voor output
output_dir = "Trading_details"
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# 1. DATA FUNCTIES
# --------------------------------------------------
def read_latest_csv_from_crudeoil():
    user, repo, branch = "Stijnknoop", "crudeoil", "master"
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref={branch}"
    response = requests.get(api_url)
    if response.status_code != 200: raise Exception("GitHub API Error")
    files = response.json()
    csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
    return pd.read_csv(csv_file['download_url'])

def prepare_trading_days():
    df = read_latest_csv_from_crudeoil()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    full_range = pd.date_range(df['time'].min(), df['time'].max(), freq='min')
    df = pd.DataFrame({'time': full_range}).merge(df, on='time', how='left')
    df['has_data'] = ~df['open_bid'].isna()
    df['date'] = df['time'].dt.date
    valid_dates = df.groupby('date')['has_data'].any()
    df = df[df['date'].isin(valid_dates[valid_dates].index)].copy()
    cols_to_ffill = df.columns.difference(['time', 'has_data', 'date'])
    df[cols_to_ffill] = df.groupby(df['has_data'].cumsum())[cols_to_ffill].ffill()

    df['gap_flag'] = (~df['has_data']) & (df['time'].dt.hour >= 20)
    df['gap_group'] = (df['gap_flag'] != df['gap_flag'].shift()).cumsum()
    gap_groups = df[df['gap_flag']].groupby('gap_group').agg(length=('time', 'count'), end_time=('time', 'last'))
    long_gaps = gap_groups[gap_groups['length'] >= 10]
    df['trading_day'] = 1
    for _, row in long_gaps.iterrows():
        df.loc[df['time'] > row['end_time'], 'trading_day'] += 1

    return {f'dag_{i}': d.reset_index(drop=True) 
            for i, (day, d) in enumerate(df.groupby('trading_day'), start=1)
            if len(d[d['has_data']]) > 200}

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
        t_l = [(np.max(p[i+1:i+31]) - p[i])/p[i] for i in range(len(df_f)-30)]
        t_s = [(p[i] - np.min(p[i+1:i+31]))/p[i] for i in range(len(df_f)-30)]
        X.append(df_f[f_selected].values[:-30]); yl.append(t_l); ys.append(t_s)
    return np.vstack(X), np.concatenate(yl), np.concatenate(ys)

# --------------------------------------------------
# 2. MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    dag_dict = prepare_trading_days()
    sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    
    TARGET_TRAIN, TARGET_VAL, MIN_START = 60, 20, 32
    test_keys = sorted_keys[MIN_START:]
    f_selected = ['z_score_30m', 'rsi', '1h_trend', 'macd', 'day_progression', 'volatility_proxy', 'hour']
    
    equity_val = 10000.0
    BEST_TP, BEST_SL, RISK_PER_TRADE, TRAILING_ACT = 0.005, -0.004, 0.02, 0.0025
    daily_logs = []

    for i, current_key in enumerate(test_keys):
        curr_idx = sorted_keys.index(current_key)
        current_val_size = max(8, min(TARGET_VAL, int(curr_idx * 0.25)))
        train_start = max(0, curr_idx - current_val_size - TARGET_TRAIN)
        
        X_t, yl_t, ys_t = get_xy(sorted_keys[train_start : curr_idx - current_val_size], dag_dict, f_selected)
        m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_t, yl_t)
        m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_t, ys_t)
        
        X_v, _, _ = get_xy(sorted_keys[curr_idx - current_val_size : curr_idx], dag_dict, f_selected)
        t_l, t_s = np.percentile(m_l.predict(X_v), 97), np.percentile(m_s.predict(X_v), 97)
        
        df_day = add_features(dag_dict[current_key]).dropna()
        pl, ps = m_l.predict(df_day[f_selected].values[:-30]), m_s.predict(df_day[f_selected].values[:-30])
        prices, times, hours = df_day['close_bid'].values, df_day['time'].values, df_day['hour'].values
        
        day_ret, active, current_sl = 0, False, BEST_SL
        real_date = str(dag_dict[current_key]['date'].iloc[0])
        trade_info = {"date": real_date, "type": "None", "pct": 0, "balance": equity_val}
        trade_points = []

        for j in range(len(pl)):
            if not active:
                if hours[j] < 23:
                    if pl[j] > t_l: 
                        ent_p, side, active = prices[j], 1, True
                        trade_info.update({"type": "LONG", "entry_p": prices[j], "entry_t": times[j]})
                        trade_points.append({'t': times[j], 'p': prices[j], 'm': '^', 'c': 'green'})
                    elif ps[j] > t_s: 
                        ent_p, side, active = prices[j], -1, True
                        trade_info.update({"type": "SHORT", "entry_p": prices[j], "entry_t": times[j]})
                        trade_points.append({'t': times[j], 'p': prices[j], 'm': 'v', 'c': 'red'})
            else:
                r = ((prices[j] - ent_p) / ent_p) * side
                if r >= TRAILING_ACT: current_sl = max(current_sl, r - 0.002)
                
                # Exit condities (TP, SL of tijd)
                if r >= BEST_TP or r <= current_sl or j == len(pl)-1 or hours[j] >= 23:
                    day_ret = r
                    trade_info.update({"exit_p": prices[j], "exit_t": times[j], "pct": r})
                    trade_points.append({'t': times[j], 'p': prices[j], 'm': 'x', 'c': 'black'})
                    active = False
                    break # STOP NA 1 TRADE

        gain_pct = day_ret * (RISK_PER_TRADE / abs(BEST_SL))
        equity_val *= (1 + gain_pct)
        trade_info.update({"dollar_profit": equity_val * (gain_pct/(1+gain_pct)), "balance": equity_val})
        daily_logs.append(trade_info)

        # --- PNG RAPPORTAGE PER DAG ---
        plt.figure(figsize=(12, 5))
        plt.plot(df_day['time'], df_day['close_bid'], color='silver', alpha=0.5, label='Koers')
        for tp in trade_points:
            plt.scatter(tp['t'], tp['p'], color=tp['c'], marker=tp['m'], s=150, zorder=5)
        plt.title(f"Dag Rapport: {real_date} | Win/Verlies: {day_ret:.2%}")
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(output_dir, f"report_{real_date}.png"))
        plt.close()

    # --- FINALE EXPORT ---
    pd.DataFrame(daily_logs).to_csv(os.path.join(output_dir, 'trading_log.csv'), index=False)
    plt.figure(figsize=(12, 6))
    plt.plot([d['balance'] for d in daily_logs], label='Account Balans')
    plt.title("Equity Curve (Hybride Window)")
    plt.savefig(os.path.join(output_dir, "latest_overview.png"))
