import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import re
import os

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

# 2. FEATURE ENGINEERING (LEAK-FREE)
def add_features(df):
    df = df.copy().sort_values('time')
    
    # Basis indicatoren
    df['hour'] = df['time'].dt.hour
    df['volatility'] = (df['high_bid'] - df['low_bid']).rolling(15).mean()
    df['macd'] = df['close_bid'].ewm(span=12).mean() - df['close_bid'].ewm(span=26).mean()
    
    delta = df['close_bid'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # 1H Trend (Shifted om leakage te voorkomen)
    df_1h = df.resample('1h', on='time').agg({'close_bid': 'last'}).shift(1)
    df_1h.columns = ['last_hour_close']
    df = df.merge(df_1h, left_on=df['time'].dt.floor('1h'), right_index=True, how='left')

    # CRUCIAAL: Shift alle features met 1 minuut. 
    # Je gebruikt de data van de VOLTOOIDE vorige minuut om NU te beslissen.
    feature_cols = ['hour', 'volatility', 'macd', 'rsi', 'last_hour_close']
    df[feature_cols] = df[feature_cols].shift(1)
    
    return df.dropna() # Verwijdert de rijen waar indicatoren nog niet klaar zijn

def get_xy(keys, d_dict, f_selected):
    X, yl, ys = [], [], []
    for k in keys:
        df_f = add_features(d_dict[k])
        if len(df_f) < 60: continue
        
        p = df_f['close_bid'].values
        # Target: wat gebeurt er in de KOMENDE 30 minuten
        t_l = [(np.max(p[i+1:i+31]) - p[i])/p[i] for i in range(len(p)-30)]
        t_s = [(p[i] - np.min(p[i+1:i+31]))/p[i] for i in range(len(p)-30)]
        
        X.append(df_f[f_selected].values[:-30])
        yl.append(t_l)
        ys.append(t_s)
    return np.vstack(X), np.concatenate(yl), np.concatenate(ys)

# 3. BACKTEST LOOP
if __name__ == "__main__":
    dag_dict = prepare_trading_days() # Gebruik je bestaande functie
    sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    
    f_selected = ['hour', 'volatility', 'macd', 'rsi', 'last_hour_close']
    equity_val = 10000.0
    daily_logs = []
    
    # Start na de training buffer
    for i in range(60, len(sorted_keys)):
        current_key = sorted_keys[i]
        # Train op verleden
        X_train, yl_train, ys_train = get_xy(sorted_keys[i-40:i], dag_dict, f_selected)
        model_l = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1).fit(X_train, yl_train)
        model_s = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1).fit(X_train, ys_train)
        
        # Test op vandaag
        df_today = add_features(dag_dict[current_key])
        preds_l = model_l.predict(df_today[f_selected])
        preds_s = model_s.predict(df_today[f_selected])
        
        prices = df_today['close_bid'].values
        times = df_today['time'].values
        
        # Simpele thresholds op basis van percentile van de voorspellingen
        t_l, t_s = np.percentile(preds_l, 95), np.percentile(preds_s, 95)
        
        active = False
        day_ret = 0
        trade_info = {"date": str(times[0])[:10], "type": "None", "inleg_dollar": 0, "pct": 0, "balance": equity_val}
        pts = []

        # Risico instellingen
        SL, TP = -0.004, 0.005
        current_inleg = equity_val * (0.02 / abs(SL))

        for j in range(len(preds_l) - 30):
            if not active:
                if preds_l[j] > t_l:
                    side, ent_p, active = 1, prices[j], True
                    trade_info.update({"type": "LONG", "inleg_dollar": current_inleg})
                    pts.append({'t': times[j], 'p': prices[j], 'm': 'UP'})
                elif preds_s[j] > t_s:
                    side, ent_p, active = -1, prices[j], True
                    trade_info.update({"type": "SHORT", "inleg_dollar": current_inleg})
                    pts.append({'t': times[j], 'p': prices[j], 'm': 'DOWN'})
            else:
                rel_ret = ((prices[j] - ent_p) / ent_p) * side
                if rel_ret <= SL or rel_ret >= TP or j == len(preds_l) - 31:
                    day_ret = rel_ret
                    trade_info.update({"pct": day_ret})
                    pts.append({'t': times[j], 'p': prices[j], 'm': 'EXIT'})
                    active = False
                    break

        # Balans update
        equity_val *= (1 + (day_ret * (0.02 / abs(SL))))
        trade_info["balance"] = equity_val
        daily_logs.append(trade_info)
        
        # Plotting (pijltjes)
        plt.figure(figsize=(10,4))
        plt.plot(times, prices, color='gray', alpha=0.5)
        for p in pts:
            c = 'green' if p['m'] == 'UP' else 'red' if p['m'] == 'DOWN' else 'black'
            marker = '^' if p['m'] == 'UP' else 'v' if p['m'] == 'DOWN' else 'x'
            plt.scatter(p['t'], p['p'], color=c, marker=marker, s=100)
        plt.title(f"{trade_info['date']} | Inleg: ${current_inleg:.0f} | Profit: {day_ret:.2%}")
        plt.savefig(os.path.join(output_dir, f"report_{trade_info['date']}.png"))
        plt.close()

    pd.DataFrame(daily_logs).to_csv(os.path.join(output_dir, "trading_log.csv"), index=False)
