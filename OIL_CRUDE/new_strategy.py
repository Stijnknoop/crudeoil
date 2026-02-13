import requests
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import http.client
import json
from datetime import datetime, timedelta
import pytz

# ==============================================================================
# 1. CONFIGURATIE & CREDENTIALS
# ==============================================================================
GITHUB_USER = "Stijnknoop"
GITHUB_REPO = "crudeoil"
FOLDER_PATH = "OIL_CRUDE"
OUTPUT_DIR = "OIL_CRUDE/Trading_details"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IDENTIFIER = os.environ.get("IDENTIFIER", "stijn-knoop@live.nl")
PASSWORD = os.environ.get("PASSWORD", "Hallohallo123!")
API_KEY = os.environ.get("X_CAP_API_KEY", "FuHgMrwvmJPAYYMp")

# DE "GOEDE" PARAMETERS
BEST_PARAMS = {
    'RSI_PERIOD': 14,          
    'MA_PERIOD': 50,
    
    'LEVERAGE': 10,              
    'MAX_CONCURRENT_TRADES': 5, 
    'BATCH_COOLDOWN': 10,         
    
    'WINDOW_SIZE': 40,         
    'ENTRY_THRESHOLD': 0.7,    
    'TP_RANGE': 0.6,           
    'MAX_DROP': 0.6,           
    'MIN_OBS': 40              
}

START_CAPITAL = 10000.0 # Zelfde startbedrag als je test

# ==============================================================================
# 2. DATA FUNCTIES
# ==============================================================================

def get_session_tokens():
    conn = http.client.HTTPSConnection("api-capital.backend-capital.com")
    payload = json.dumps({"identifier": IDENTIFIER, "password": PASSWORD})
    headers = {"Content-Type": "application/json", "X-CAP-API-KEY": API_KEY}
    try:
        conn.request("POST", "/api/v1/session", payload, headers)
        res = conn.getresponse()
        if res.status == 200:
            return res.getheader("X-SECURITY-TOKEN"), res.getheader("CST")
        return None, None
    except:
        return None, None

def fetch_live_data_capital():
    sec_token, cst = get_session_tokens()
    if not sec_token: return None

    conn = http.client.HTTPSConnection("api-capital.backend-capital.com")
    headers = {"X-SECURITY-TOKEN": sec_token, "CST": cst, "X-CAP-API-KEY": API_KEY}
    
    try:
        conn.request("GET", "/api/v1/prices/OIL_CRUDE?resolution=MINUTE&max=600", "", headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode())
        prices = data.get("prices", [])
        
        if not prices: return None
        
        df = pd.json_normalize(prices)
        df = df[["snapshotTime", "openPrice.bid", "highPrice.bid", "lowPrice.bid", "closePrice.bid",
                 "openPrice.ask", "highPrice.ask", "lowPrice.ask", "closePrice.ask", "lastTradedVolume"]]
        df.columns = ["time", "open_bid", "high_bid", "low_bid", "close_bid",
                      "open_ask", "high_ask", "low_ask", "close_ask", "volume"]
        
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None) 
        return df
    except: return None

def get_data_github():
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{FOLDER_PATH}?ref=master"
    try:
        r = requests.get(api_url, headers=headers).json()
        download_url = r[0]['download_url'] if isinstance(r, list) else r['download_url']
        df = pd.read_csv(download_url)
        df['time'] = pd.to_datetime(df['time'])
        if df['time'].dt.tz is not None: df['time'] = df['time'].dt.tz_localize(None)
        return df
    except: return None

def merge_and_process(df_old, df_new):
    if df_old is None: return df_new
    if df_new is None: return df_old
    
    df = pd.concat([df_old, df_new]).drop_duplicates(subset="time", keep="last").sort_values("time").reset_index(drop=True)
    df = df.set_index('time').resample('1min').ffill().dropna().reset_index()
    
    df['price_diff'] = df['close_bid'].diff()
    df['is_flat'] = df['price_diff'] == 0
    df['block_id'] = (df['is_flat'] != df['is_flat'].shift()).cumsum()

    break_blocks = []
    stats = df[df['is_flat']].groupby('block_id').agg(start=('time', 'first'), count=('time', 'count'))
    for bid, row in stats.iterrows():
        if row['count'] > 45 and (row['start'].hour >= 21 or row['start'].hour <= 2):
            break_blocks.append(bid)

    df['is_trading_active'] = ~df['block_id'].isin(break_blocks)
    df['new_sess'] = df['is_trading_active'] & ((df['is_trading_active'].shift(1) == False) | (df.index == 0))
    df['session_id'] = df['new_sess'].cumsum()
    df.loc[~df['is_trading_active'], 'session_id'] = -1
    
    df['hour'] = df['time'].dt.hour
    df['mid_price'] = (df['close_ask'] + df['close_bid']) / 2
    
    # FEATURES (Exact zoals in jouw snippet)
    p_rsi = BEST_PARAMS['RSI_PERIOD']
    delta = df['mid_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=p_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=p_rsi).mean()
    rs_val = gain / loss
    df[f'rsi_{p_rsi}'] = 100 - (100 / (1 + rs_val))
    
    p_ma = BEST_PARAMS['MA_PERIOD']
    ma = df['mid_price'].rolling(window=p_ma).mean()
    df[f'trend_{p_ma}'] = (df['mid_price'] / ma) - 1
    
    df['vol_ratio'] = df['mid_price'].rolling(20).std() / df['mid_price'].rolling(100).std()
    
    return df.dropna().reset_index(drop=True)

# ==============================================================================
# 3. BACKTEST LOGICA (EXACTE KOPIE VAN JOUW WERKENDE FUNCTIE)
# ==============================================================================

def run_strategy(params, data):
    W_SIZE, E_THRESH = params['WINDOW_SIZE'], params['ENTRY_THRESHOLD']
    TP_R, MAX_DROP   = params['TP_RANGE'], params['MAX_DROP']
    MIN_OBS          = params['MIN_OBS']
    MAX_TRADES = params['MAX_CONCURRENT_TRADES']
    COOLDOWN   = params['BATCH_COOLDOWN']
    LEVERAGE   = params['LEVERAGE']
    
    RSI_COL = f"rsi_{params['RSI_PERIOD']}"
    TREND_COL = f"trend_{params['MA_PERIOD']}"
    
    equity = START_CAPITAL
    equity_curve = [START_CAPITAL]
    dates_curve = [data['time'].iloc[0]]
    action_log = []
    
    range_bins = np.linspace(0, 1.0, 6)
    rsi_bins = [0, 30, 70, 100]
    trend_bins = [-np.inf, -0.0005, 0.0005, np.inf]
    vol_bins = [-np.inf, 0.9, 1.2, np.inf]
    
    unique_sessions = sorted(data[data['session_id'] != -1]['session_id'].unique())
    print(f"Start simulatie over {len(unique_sessions) - W_SIZE} sessies...")
    
    for i in range(W_SIZE, len(unique_sessions)):
        test_sess_id = unique_sessions[i]
        start_train = unique_sessions[i-W_SIZE]
        end_train = unique_sessions[i-1]
        
        # Training
        mask = (data['session_id'] >= start_train) & (data['session_id'] <= end_train)
        df_h = data[mask].copy()
        
        sess_grp = df_h.groupby('session_id')
        df_h['day_high'] = sess_grp['mid_price'].cummax()
        df_h['day_low'] = sess_grp['mid_price'].cummin()
        df_h['day_rng'] = df_h['day_high'] - df_h['day_low']
        
        df_h = df_h[df_h['day_rng'] > 0].copy()
        
        target = df_h['mid_price'] + TP_R * df_h['day_rng']
        fut_max = df_h.groupby('session_id')['mid_price'].transform(lambda x: x[::-1].cummax()[::-1])
        df_h['hit'] = (fut_max >= target).astype(int)
        
        drop_tgt = df_h['mid_price'] - TP_R * df_h['day_rng']
        fut_min = df_h.groupby('session_id')['mid_price'].transform(lambda x: x[::-1].cummin()[::-1])
        df_h['loss'] = (fut_min <= drop_tgt).astype(int)
        
        df_h['b_rng'] = pd.cut((df_h['mid_price'] - df_h['day_low']) / df_h['day_rng'], bins=range_bins, labels=False)
        df_h['b_rsi'] = pd.cut(df_h[RSI_COL], bins=rsi_bins, labels=False)     
        df_h['b_trd'] = pd.cut(df_h[TREND_COL], bins=trend_bins, labels=False) 
        df_h['b_vol'] = pd.cut(df_h['vol_ratio'], bins=vol_bins, labels=False) 
        
        stats = df_h.groupby(['hour', 'b_rng', 'b_rsi', 'b_trd', 'b_vol'])[['hit', 'loss']].agg(['mean', 'count'])
        valid_stats = stats[stats[('hit', 'count')] >= MIN_OBS]
        valid_stats.columns = ['_'.join(col) for col in valid_stats.columns] 
        prob_map = valid_stats[['hit_mean', 'loss_mean']].to_dict('index')
        
        # Trading
        dff = data[data['session_id'] == test_sess_id].copy().reset_index(drop=True)
        if len(dff) < 50: continue
        
        p_mid = dff['mid_price'].values
        p_ask = dff['close_ask'].values 
        p_low_ask = dff['low_ask'].values 
        p_bid = dff['close_bid'].values 
        p_high_bid = dff['high_bid'].values 
        
        hours = dff['hour'].values
        times = dff['time'].values
        rsis = dff[RSI_COL].values 
        trends = dff[TREND_COL].values
        vols = dff['vol_ratio'].values
        
        high_cum = np.maximum.accumulate(p_mid)
        low_cum = np.minimum.accumulate(p_mid)
        
        active_trades = []   
        pending_orders = []  
        last_signal_idx = -999 
        
        for t in range(50, len(dff)):
            curr_time = times[t]
            
            # 1. ACTIVE TRADES CHECK
            for k in range(len(active_trades) - 1, -1, -1):
                trade = active_trades[k]
                exit_signal = False
                reason = ""
                exit_price = 0.0
                
                if p_high_bid[t] >= trade['target_price']:
                    exit_signal = True; exit_price = trade['target_price']; reason = "TP"
                elif t == len(dff) - 1:
                    exit_signal = True; exit_price = p_bid[t]; reason = "EOD"
                
                if exit_signal:
                    raw_ret = (exit_price - trade['entry_price']) / trade['entry_price']
                    pnl = trade['stake'] * LEVERAGE * raw_ret
                    if pnl < -trade['stake']: pnl = -trade['stake']; reason = "LIQ"
                    
                    equity += pnl
                    action_log.append({'time': curr_time, 'action': f'EXIT_{reason}', 'pnl': pnl, 'equity': equity})
                    active_trades.pop(k) 

            # 2. PENDING ORDERS CHECK
            for k in range(len(pending_orders) - 1, -1, -1):
                order = pending_orders[k]
                if p_high_bid[t] >= order['target_price']:
                    pending_orders.pop(k); continue
                if t == len(dff) - 1:
                    pending_orders.pop(k); continue
                
                if t >= order['signal_idx'] + 2:
                    if p_low_ask[t] <= order['limit_price']:
                        stake = max(0, equity) / MAX_TRADES
                        if stake > 0:
                            new_trade = {
                                'entry_price': order['limit_price'],
                                'target_price': order['target_price'],
                                'stake': stake,
                                'entry_time': curr_time
                            }
                            active_trades.append(new_trade)
                            action_log.append({'time': curr_time, 'action': 'ENTRY', 'price': order['limit_price']})
                        pending_orders.pop(k)

            # 3. SIGNAL GENERATION
            if len(active_trades) + len(pending_orders) < MAX_TRADES:
                if t >= last_signal_idx + COOLDOWN:
                    if hours[t] < 22:
                        rng = high_cum[t] - low_cum[t]
                        if rng > 0:
                            rng_pos = (p_mid[t] - low_cum[t]) / rng
                            b_r = min(int(rng_pos * 5), 4)
                            b_rs = 0 if rsis[t] < 30 else (2 if rsis[t] > 70 else 1)
                            b_tr = 0 if trends[t] < -0.0005 else (2 if trends[t] > 0.0005 else 1)
                            b_vl = 0 if vols[t] < 0.9 else (2 if vols[t] > 1.2 else 1)
                            
                            key = (hours[t], b_r, b_rs, b_tr, b_vl)
                            if key in prob_map:
                                stats = prob_map[key]
                                if stats['hit_mean'] >= E_THRESH and stats['loss_mean'] <= MAX_DROP:
                                    limit_pr = p_ask[t]
                                    target_pr = limit_pr + (TP_R * rng)
                                    pending_orders.append({
                                        'limit_price': limit_pr, 'target_price': target_pr, 'signal_idx': t
                                    })
                                    last_signal_idx = t
        
        equity_curve.append(equity)
        dates_curve.append(dff['time'].iloc[-1])

    return equity_curve, dates_curve, action_log

if __name__ == "__main__":
    df_git = get_data_github()
    df_cap = fetch_live_data_capital()
    df_main = merge_and_process(df_git, df_cap)
    
    if df_main is not None and not df_main.empty:
        eq_curve, dates, logs = run_strategy(BEST_PARAMS, df_main)
        
        if logs:
            pd.DataFrame(logs).to_csv(os.path.join(OUTPUT_DIR, "trading_log.csv"), index=False)
            
        if len(dates) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(dates, eq_curve, color='#2980b9', linewidth=2)
            plt.title(f'Equity Curve (Eindsaldo: €{eq_curve[-1]:.2f})', fontsize=14)
            plt.grid(True, alpha=0.3)
            ax = plt.gca(); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
            plt.gcf().autofmt_xdate()
            plt.savefig(os.path.join(OUTPUT_DIR, "equity_curve.png"))
