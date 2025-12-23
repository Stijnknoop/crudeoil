import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

# ==================================================
# 1. DATA OPHALEN
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
    full_range = pd.date_range(df['time'].min(), df['time'].max(), freq='min')
    df = pd.DataFrame({'time': full_range}).merge(df, on='time', how='left')
    df['has_data'] = ~df['open_bid'].isna()
    df['date'] = df['time'].dt.date
    valid_dates = df.groupby('date')['has_data'].any()
    df = df[df['date'].isin(valid_dates[valid_dates].index)].copy()
    cols_to_ffill = df.columns.difference(['time', 'has_data', 'date'])
    df[cols_to_ffill] = df.groupby(df['has_data'].cumsum())[cols_to_ffill].ffill()
    dag_dict = {str(date): group.reset_index(drop=True) 
                for date, group in df.groupby('date') if len(group[group['has_data']]) > 200}
    return dag_dict

def add_features(df):
    df = df.copy()
    df['hour'] = df['time'].dt.hour
    df['day_progression'] = (df['hour'] * 60 + df['time'].dt.minute) / 1440.0
    df['volatility_proxy'] = (df['high_bid'] - df['low_bid']).rolling(15).mean() / (df['close_bid'] + 1e-9)
    df['z_score_30m'] = (df['close_bid'] - df['close_bid'].rolling(30).mean()) / (df['close_bid'].rolling(30).std() + 1e-9)
    df['macd'] = df['close_bid'].ewm(span=12).mean() - df['close_bid'].ewm(span=26).mean()
    delta = df['close_bid'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
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

# ==================================================
# 3. EXECUTION
# ==================================================
if __name__ == "__main__":
    dag_dict = prepare_trading_days()
    sorted_keys = sorted(dag_dict.keys())
    
    output_dir = "Trading_details"
    os.makedirs(output_dir, exist_ok=True)

    # JOUW NIEUWE INSTELLINGEN:
    TRAIN_SIZE, VAL_SIZE = 24, 8
    total_needed = TRAIN_SIZE + VAL_SIZE
    test_keys = sorted_keys[total_needed:]
    
    if not test_keys:
        print(f"DEBUG: Totaal dagen in CSV: {len(sorted_keys)}. Nodig: {total_needed + 1}")
        exit(0)

    f_selected = ['z_score_30m', 'rsi', 'macd', 'day_progression', 'volatility_proxy', 'hour']
    equity_val = 10000.0
    daily_logs = []

    for current_date in test_keys:
        curr_idx = sorted_keys.index(current_date)
        train_keys = sorted_keys[curr_idx - total_needed : curr_idx - VAL_SIZE]
        val_keys = sorted_keys[curr_idx - VAL_SIZE : curr_idx]
        
        X_t, yl_t, ys_t = get_xy(train_keys, dag_dict, f_selected)
        m_l = RandomForestRegressor(n_estimators=100, max_depth=6).fit(X_t, yl_t)
        m_s = RandomForestRegressor(n_estimators=100, max_depth=6).fit(X_t, ys_t)
        
        X_v, _, _ = get_xy(val_keys, dag_dict, f_selected)
        t_l, t_s = np.percentile(m_l.predict(X_v), 97), np.percentile(m_s.predict(X_v), 97)
        
        df_day = add_features(dag_dict[current_date]).dropna()
        pl, ps = m_l.predict(df_day[f_selected].values[:-30]), m_s.predict(df_day[f_selected].values[:-30])
        prices, times, hours = df_day['close_bid'].values, df_day['time'].values, df_day['hour'].values
        
        day_ret, active, ent_p, side, current_sl = 0, False, 0, 0, -0.004
        trade_info = {"date": current_date, "type": "None", "pct": 0}

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
        
        equity_val *= (1 + (day_ret * 5))
        trade_info.update({"dollar_profit": day_ret * 5 * (equity_val/(1+day_ret*5)), "balance": equity_val})
        daily_logs.append(trade_info)

    df_res = pd.DataFrame(daily_logs)
    df_res.to_csv(os.path.join(output_dir, 'trading_log.csv'), index=False)

    # Plotten
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    last_date = test_keys[-1]
    df_plot = add_features(dag_dict[last_date]).dropna()
    last_trade = daily_logs[-1]
    
    ax1.plot(df_plot['time'], df_plot['close_bid'], color='silver')
    if last_trade["type"] != "None":
        ax1.scatter(last_trade["entry_t"], last_trade["entry_p"], color='green', marker='^')
        ax1.scatter(last_trade["exit_t"], last_trade["exit_p"], color='red', marker='x')
    ax1.set_title(f"Trades op {last_date}")

    ax2.plot(pd.to_datetime(df_res['date']), df_res['balance'], color='dodgerblue')
    ax2.set_title(f"Balance: ${equity_val:.2f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latest_overview.png"))
    plt.savefig(os.path.join(output_dir, f"report_{last_date}.png"))
    print(f"Klaar! Map {output_dir} is bijgewerkt.")
