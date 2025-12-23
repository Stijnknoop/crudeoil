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

# 1. DATA OPHALEN
def read_latest_csv_from_crudeoil():
    user, repo, branch = "Stijnknoop", "crudeoil", "master"
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents?ref={branch}"
    response = requests.get(api_url)
    if response.status_code != 200: raise Exception("API Error")
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
    return {f'dag_{i}': d.reset_index(drop=True) for i, (day, d) in enumerate(df.groupby('trading_day'), start=1) if len(d[d['has_data']]) > 200}

# 2. FEATURE ENGINEERING (LEAK-FREE)
def add_features(df):
    df = df.copy().sort_values('time')
    df['hour'] = df['time'].dt.hour
    df['volatility'] = (df['high_bid'] - df['low_bid']).rolling(15).mean()
    df['macd'] = df['close_bid'].ewm(span=12).mean() - df['close_bid'].ewm(span=26).mean()
    delta = df['close_bid'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    df_1h = df.resample('1h', on='time').agg({'close_bid': 'last'}).shift(1)
    df_1h.columns = ['last_hour_close']
    df['floor_hour'] = df['time'].dt.floor('1h')
    df = df.merge(df_1h, left_on='floor_hour', right_index=True, how='left').drop(columns=['floor_hour'])

    f_cols = ['hour', 'volatility', 'macd', 'rsi', 'last_hour_close']
    df[f_cols] = df[f_cols].shift(1)
    return df.dropna()

def get_xy(keys, d_dict, f_selected):
    X, yl, ys = [], [], []
    for k in keys:
        df_f = add_features(d_dict[k])
        if len(df_f) < 60: continue
        p = df_f['close_bid'].values
        t_l = [(np.max(p[i+1:i+31]) - p[i])/p[i] for i in range(len(p)-30)]
        t_s = [(p[i] - np.min(p[i+1:i+31]))/p[i] for i in range(len(p)-30)]
        X.append(df_f[f_selected].values[:-30]); yl.append(t_l); ys.append(t_s)
    return np.vstack(X), np.concatenate(yl), np.concatenate(ys)

# 3. BACKTEST LOOP
if __name__ == "__main__":
    dag_dict = prepare_trading_days()
    sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    
    f_selected = ['hour', 'volatility', 'macd', 'rsi', 'last_hour_close']
    initial_balance = 10000.0
    equity_val = initial_balance
    daily_logs = []
    
    for i in range(60, len(sorted_keys)):
        current_key = sorted_keys[i]
        X_t, yl_t, ys_t = get_xy(sorted_keys[i-40:i], dag_dict, f_selected)
        m_l = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1).fit(X_t, yl_t)
        m_s = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1).fit(X_t, ys_t)
        
        df_today = add_features(dag_dict[current_key])
        if df_today.empty: continue
        
        pl, ps = m_l.predict(df_today[f_selected]), m_s.predict(df_today[f_selected])
        prices, times = df_today['close_bid'].values, df_today['time'].values
        
        t_l, t_s = np.percentile(pl, 90), np.percentile(ps, 90)
        active, day_ret = False, 0
        SL, TP = -0.004, 0.005
        current_inleg = equity_val * (0.02 / abs(SL))
        
        real_date = str(times[0])[:10]
        # Altijd alle kolommen vullen om KeyErrors te voorkomen
        trade_info = {
            "date": real_date, "type": "None", "inleg_dollar": 0, 
            "pct": 0.0, "balance": equity_val, "entry_p": 0, "exit_p": 0, "dollar_profit": 0.0
        }
        pts = []

        for j in range(len(pl)-30):
            if not active:
                if pl[j] > t_l:
                    ent_p, side, active = prices[j], 1, True
                    trade_info.update({"type": "LONG", "inleg_dollar": current_inleg, "entry_p": prices[j]})
                    pts.append({'t': times[j], 'p': prices[j], 'm': '↑', 'c': 'green'})
                elif ps[j] > t_s:
                    ent_p, side, active = prices[j], -1, True
                    trade_info.update({"type": "SHORT", "inleg_dollar": current_inleg, "entry_p": prices[j]})
                    pts.append({'t': times[j], 'p': prices[j], 'm': '↓', 'c': 'red'})
            else:
                r = ((prices[j] - ent_p) / ent_p) * side
                if r <= SL or r >= TP or j == len(pl)-31:
                    day_ret = r
                    trade_info.update({"exit_p": prices[j], "pct": float(day_ret)})
                    pts.append({'t': times[j], 'p': prices[j], 'm': 'x', 'c': 'black'})
                    active = False
                    break

        old_bal = equity_val
        equity_val *= (1 + (day_ret * (0.02 / abs(SL))))
        trade_info.update({"balance": equity_val, "dollar_profit": equity_val - old_bal})
        daily_logs.append(trade_info)
        
        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(times, prices, color='gray', alpha=0.3)
        for p in pts:
            if p['m'] in ['↑', '↓']:
                plt.annotate(p['m'], xy=(p['t'], p['p']), color=p['c'], fontsize=20, fontweight='bold')
            else:
                plt.scatter(p['t'], p['p'], color='black', marker='x', s=100)
        plt.title(f"{real_date} | Inleg: ${current_inleg:.0f} | Return: {day_ret:.2%}")
        plt.savefig(os.path.join(output_dir, f"report_{real_date}.png"))
        plt.close()

    # CSV Summary met veiligheidscheck op kolommen
    if daily_logs:
        df_res = pd.DataFrame(daily_logs)
        wr = len(df_res[df_res['pct'] > 0]) / max(1, len(df_res[df_res['type'] != "None"]))
        summary = {
            "date": "SAMENVATTING", "type": f"WR: {wr:.1%}", 
            "balance": equity_val, "pct": f"Tot: {((equity_val-10000)/10000):.1%}",
            "inleg_dollar": "", "entry_p": "", "exit_p": "", "dollar_profit": ""
        }
        df_res = pd.concat([df_res, pd.DataFrame([summary])], ignore_index=True)
        df_res.to_csv(os.path.join(output_dir, "trading_log.csv"), index=False)
        print(f"Klaar! Eindbalans: ${equity_val:.2f}")
    else:
        print("Geen data gevonden om te verwerken.")
