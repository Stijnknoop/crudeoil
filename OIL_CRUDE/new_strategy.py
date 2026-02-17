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

# --- WTI SPECIFIEKE HANDELSTIJDEN (NEW YORK TIJD) ---
# WTI is het meest liquide tijdens de US Session.
# We negeren de Aziatische/Europese ochtend sessies voor entry signalen.
START_HOUR_NY = 9    # 09:00 NY (Open Vloer / 15:00 NL)
EXIT_HOUR_NY  = 16   # 16:00 NY (Uur voor settlement / 22:00 NL)

# PARAMETERS
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

START_CAPITAL = 10000.0

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
        
        # Capital data is UTC strings
        df["time"] = pd.to_datetime(df["time"])
        if df['time'].dt.tz is None:
             df['time'] = df['time'].dt.tz_localize('UTC')
        else:
             df['time'] = df['time'].dt.tz_convert('UTC')
             
        return df
    except: return None

def get_data_github():
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{FOLDER_PATH}?ref=master"
    try:
        r = requests.get(api_url, headers=headers).json()
        if isinstance(r, list):
            csv_file = next((f for f in r if f['name'].endswith('.csv')), None)
            if not csv_file: return None
            download_url = csv_file['download_url']
        else:
            download_url = r.get('download_url')
            
        if not download_url: return None

        df = pd.read_csv(download_url)
        
        # Github CSV is UTC
        df['time'] = pd.to_datetime(df['time'])
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        else:
            df['time'] = df['time'].dt.tz_convert('UTC')
            
        return df
    except Exception as e: 
        print(f"Github error: {e}")
        return None

def merge_and_process(df1, df2):
    """
    Verwerkt data met TWEE tijdzones:
    1. ny_time: Voor de strategie logica (WTI markttijden)
    2. nl_time: Voor logs en plots (leesbaarheid)
    """
    if df1 is None and df2 is None: return None
    if df1 is None: df = df2.copy()
    elif df2 is None: df = df1.copy()
    else:
        df = pd.concat([df1, df2]).drop_duplicates(subset="time", keep="last").sort_values("time").reset_index(drop=True)
    
    # 1. TIMEZONE SETUP
    # Zorg dat basis UTC is
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')
    else:
        df['time'] = df['time'].dt.tz_convert('UTC')

    # Maak specifieke kolommen
    ny_tz = pytz.timezone('America/New_York')
    nl_tz = pytz.timezone('Europe/Amsterdam')
    
    df['ny_time'] = df['time'].dt.tz_convert(ny_tz)
    df['nl_time'] = df['time'].dt.tz_convert(nl_tz)

    df = df.set_index('time').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.resample('1min').ffill().dropna().reset_index()

    # 2. SESSIE DEFINITIE (OP BASIS VAN NEW YORK DATUM)
    # Dit zorgt voor stabiliteit. 
    df['ny_date'] = df['ny_time'].dt.date
    unique_dates = sorted(df['ny_date'].unique())
    date_map = {d: i for i, d in enumerate(unique_dates)}
    df['session_id'] = df['ny_date'].map(date_map)
    
    # 3. GEBRUIK NY UREN VOOR LOGICA
    df['hour'] = df['ny_time'].dt.hour
    
    df['mid_price'] = (df['close_ask'] + df['close_bid']) / 2
    
    # Indicators
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
# 3. BACKTEST LOGICA
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
    equity_curve = [] 
    dates_curve = []
    action_log = []
    
    range_bins = np.linspace(0, 1.0, 6)
    rsi_bins = [0, 30, 70, 100]
    trend_bins = [-np.inf, -0.0005, 0.0005, np.inf]
    vol_bins = [-np.inf, 0.9, 1.2, np.inf]
    
    unique_sessions = sorted(data['session_id'].unique())
    print(f"Start simulatie over {len(unique_sessions) - W_SIZE} sessies (NY Market Logic)...")
    
    for i in range(W_SIZE, len(unique_sessions)):
        test_sess_id = unique_sessions[i]
        start_train = unique_sessions[i-W_SIZE]
        end_train = unique_sessions[i-1]
        
        # --- TRAINING (Op historische NY sessies) ---
        mask = (data['session_id'] >= start_train) & (data['session_id'] <= end_train)
        df_h = data[mask].copy()
        
        # Filter: Train alleen op de uren dat WTI actief is (09:00 - 16:00 NY)
        df_h = df_h[(df_h['hour'] >= START_HOUR_NY) & (df_h['hour'] <= EXIT_HOUR_NY)]
        
        if len(df_h) < 100: continue # Skip training bij te weinig data

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
        
        # --- TRADING (De 'Live' Test Dag) ---
        dff = data[data['session_id'] == test_sess_id].copy().reset_index(drop=True)
        if len(dff) < 50: continue
        
        # Arrays voor snelheid
        p_mid = dff['mid_price'].values
        p_ask = dff['close_ask'].values 
        p_low_ask = dff['low_ask'].values 
        p_bid = dff['close_bid'].values 
        p_high_bid = dff['high_bid'].values 
        
        hours = dff['hour'].values # NY Uren
        
        # Indicators
        rsis = dff[RSI_COL].values 
        trends = dff[TREND_COL].values
        vols = dff['vol_ratio'].values
        
        # Range Tracking
        high_cum = np.maximum.accumulate(p_mid)
        low_cum = np.minimum.accumulate(p_mid)
        
        active_trades = []    
        pending_orders = []   
        last_signal_idx = -999 
        
        for t in range(50, len(dff)):
            # Gebruik NL tijd voor logs, NY tijd voor logica
            log_time = dff['nl_time'].iloc[t]
            curr_ny_hour = hours[t]
            
            # --- HARDE EXIT CONDITIE (WTI SESSION CLOSE) ---
            # We sluiten om 16:00 NY (ca 22:00 NL).
            force_exit = (curr_ny_hour >= EXIT_HOUR_NY) or (t == len(dff) - 1)
            
            # 1. EXITS
            for k in range(len(active_trades) - 1, -1, -1):
                trade = active_trades[k]
                exit_signal = False; reason = ""; exit_price = 0.0
                
                if p_high_bid[t] >= trade['target_price']:
                    exit_signal = True; exit_price = trade['target_price']; reason = "TP"
                elif force_exit:
                    exit_signal = True; exit_price = p_bid[t]; reason = "TIME"
                
                if exit_signal:
                    raw_ret = (exit_price - trade['entry_price']) / trade['entry_price']
                    pnl = trade['stake'] * LEVERAGE * raw_ret
                    # Stop loss logic (Liquidatie)
                    if pnl < -trade['stake']: pnl = -trade['stake']; reason = "LIQ"
                    
                    equity += pnl
                    action_log.append({'time': log_time, 'action': f'EXIT_{reason}', 'pnl': pnl, 'equity': equity})
                    active_trades.pop(k) 

            # 2. PENDING ORDERS
            for k in range(len(pending_orders) - 1, -1, -1):
                order = pending_orders[k]
                
                # Order gemist?
                if p_high_bid[t] >= order['target_price']:
                    pending_orders.pop(k); continue
                
                # Market close?
                if force_exit:
                    pending_orders.pop(k); continue
                
                # Fill?
                if t >= order['signal_idx'] + 2:
                    if p_low_ask[t] <= order['limit_price']:
                        stake = max(0, equity) / MAX_TRADES
                        if stake > 10: 
                            new_trade = {
                                'entry_price': order['limit_price'],
                                'target_price': order['target_price'],
                                'stake': stake,
                                'entry_time': log_time
                            }
                            active_trades.append(new_trade)
                            action_log.append({'time': log_time, 'action': 'ENTRY', 'price': order['limit_price']})
                        pending_orders.pop(k)

            # 3. NIEUWE SIGNALEN
            # Alleen genereren tijdens liquide NY uren (09:00 - 16:00)
            if not force_exit and curr_ny_hour >= START_HOUR_NY:
                if len(active_trades) + len(pending_orders) < MAX_TRADES:
                    if t >= last_signal_idx + COOLDOWN:
                        rng = high_cum[t] - low_cum[t]
                        if rng > 0:
                            rng_pos = (p_mid[t] - low_cum[t]) / rng
                            b_r = min(int(rng_pos * 5), 4)
                            b_rs = 0 if rsis[t] < 30 else (2 if rsis[t] > 70 else 1)
                            b_tr = 0 if trends[t] < -0.0005 else (2 if trends[t] > 0.0005 else 1)
                            b_vl = 0 if vols[t] < 0.9 else (2 if vols[t] > 1.2 else 1)
                            
                            key = (curr_ny_hour, b_r, b_rs, b_tr, b_vl)
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
        # Gebruik NL tijd voor de datum in de grafiek
        dates_curve.append(dff['nl_time'].iloc[-1])

    return equity_curve, dates_curve, action_log

if __name__ == "__main__":
    print("--- START NIEUWE STRATEGIE ---")
    df_git = get_data_github()
    df_cap = fetch_live_data_capital()
    df_main = merge_and_process(df_git, df_cap)
    
    if df_main is not None and not df_main.empty:
        eq_curve, dates, logs = run_strategy(BEST_PARAMS, df_main)
        
        if logs:
            pd.DataFrame(logs).to_csv(os.path.join(OUTPUT_DIR, "trading_log.csv"), index=False)
            print("Trade log opgeslagen.")
            
        if len(dates) > 0:
            plt.figure(figsize=(12, 6))
            if dates:
                start_date = dates[0] - timedelta(days=1)
                dates = [start_date] + dates
                eq_curve = [START_CAPITAL] + eq_curve

            plt.plot(dates, eq_curve, color='#2980b9', linewidth=2)
            plt.title(f'Equity Curve (Eindsaldo: €{eq_curve[-1]:.2f})', fontsize=14)
            plt.grid(True, alpha=0.3)
            # Plot X-as in NL Tijd
            ax = plt.gca(); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m', tz=pytz.timezone('Europe/Amsterdam')))
            plt.gcf().autofmt_xdate()
            plt.savefig(os.path.join(OUTPUT_DIR, "equity_curve.png"))
            print("Equity curve opgeslagen.")
    else:
        print("Geen data gevonden!")
    print("--- KLAAR ---")
