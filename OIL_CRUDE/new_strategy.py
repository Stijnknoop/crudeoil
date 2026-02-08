import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ==============================================================================
# 1. CONFIGURATIE & INSTELLINGEN
# ==============================================================================
GITHUB_USER = "Stijnknoop"
GITHUB_REPO = "crudeoil"
FOLDER_PATH = "OIL_CRUDE"
OUTPUT_DIR = "OIL_CRUDE/Trading_details"
LOG_FILE = "trading_logs.csv"

# --- BACKTEST WINDOW ---
ROLLING_WINDOW_SESSIONS = 40  # Aantal sessies (ipv dagen) historie om op te trainen

# --- ACCOUNT & RISK ---
START_CAPITAL = 65.0        # Startkapitaal
MAX_SLOTS = 10
LEVERAGE = 10
COOLDOWN_MINUTES = 10

# --- STRATEGIE FILTERS (DYNAMISCH) ---
MIN_HISTORICAL_TRADES = 15       # Minimaal aantal trades in training window
MIN_EXPECTED_ROI = 0.0025        # 0.25% gemiddelde winst per trade vereist
MIN_RANGE = 0.0008               # Range moet min. 0.08% van de prijs zijn
TARGET_RANGE_RATIO = 0.5         # Target is 0.5x de Range

# ==============================================================================
# 2. DATA OPHALEN & VERWERKEN (NOTEBOOK STYLE)
# ==============================================================================
def get_data_and_process():
    """Haalt data op en past de Block-ID sessie logica toe."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{FOLDER_PATH}?ref=master"
    
    try:
        r = requests.get(api_url, headers=headers).json()
        csv_file = next((f for f in r if f['name'].endswith('.csv')), None)
        if not csv_file: return None
        
        print(f"Data gedownload: {csv_file['name']}")
        df = pd.read_csv(csv_file['download_url'])
        
        # Tijd en Gaps
        df['time'] = pd.to_datetime(df['time'], format='ISO8601')
        df = df.set_index('time').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample('1min').ffill().dropna().reset_index()
        
        # --- SESSIE LOGICA (BLOCK ID) ---
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
        
        return df
    except Exception as e:
        print(f"❌ Error bij data ophalen: {e}")
        return None

def add_features(df):
    """Berekent indicatoren op basis van Session ID."""
    df = df.copy()
    
    # RSI
    delta = df['close_bid'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + (avg_gain/avg_loss)))
    
    # Sessie Context (Vorige sessie trend)
    sess_stats = df[df['session_id'] != -1].groupby('session_id')['close_bid'].agg(['last'])
    sess_stats['prev_trend'] = np.where((sess_stats['last'] - sess_stats['last'].shift(1)) > 0, 'Groen', 'Rood')
    df = df.merge(sess_stats[['prev_trend']], on='session_id', how='left')
    
    # Intraday (So Far) - Essentieel om voorkennis te voorkomen
    # We groeperen op session_id, niet op datum!
    df['sess_min'] = df.groupby('session_id')['close_bid'].cummin()
    df['sess_max'] = df.groupby('session_id')['close_bid'].cummax()
    df['sess_range'] = df['sess_max'] - df['sess_min']
    
    # Features
    df['pos_pct'] = ((df['close_bid'] - df['sess_min']) / (df['sess_range'] + 1e-9)).fillna(0.5)
    df['range_pct'] = df['sess_range'] / df['close_bid']
    
    # Bins
    df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutraal', 'Overbought'])
    df['pos_bin'] = pd.cut(df['pos_pct'], bins=[-0.1, 0.3, 0.7, 1.1], labels=['Low', 'Mid', 'High'])
    df['hour'] = df['time'].dt.hour
    
    return df

# ==============================================================================
# 3. TRAINING LOGICA (NO TIME TRAVEL)
# ==============================================================================
def train_rules(train_df):
    """Zoekt winnende strategieën zonder look-ahead bias."""
    entry = train_df['close_ask']
    target_dist = train_df['sess_range'] * TARGET_RANGE_RATIO
    
    # --- TIME TRAVEL FIX ---
    # Max prijs in de toekomst (binnen dezelfde sessie) vanaf het huidige moment
    max_future = train_df.iloc[::-1].groupby('session_id')['close_bid'].cummax().iloc[::-1]
    
    is_win = (max_future >= (entry + target_dist)) & (train_df['range_pct'] > MIN_RANGE)
    
    # Als target niet gehit wordt, nemen we de laatste prijs van de sessie als exit
    sess_close = train_df.groupby('session_id')['close_bid'].transform('last')
    pnl = np.where(is_win, target_dist, sess_close - entry)
    
    train_df = train_df.assign(roi = pnl / entry)
    
    stats = train_df.groupby(['prev_trend', 'pos_bin', 'rsi_bin', 'hour'], observed=True)['roi'].agg(['mean', 'count'])
    winning = stats[(stats['count'] >= MIN_HISTORICAL_TRADES) & (stats['mean'] > MIN_EXPECTED_ROI)]
    
    return set(winning.index.tolist())

# ==============================================================================
# 4. HOOFDPROGRAMMA
# ==============================================================================
print(f"--- START GITHUB TRADING BOT ({ROLLING_WINDOW_SESSIONS} SESSIES) ---")

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(OUTPUT_DIR, LOG_FILE)

# A. Data laden & Verwerken
df_raw = get_data_and_process()
if df_raw is None: exit()

# Features toevoegen aan de hele dataset
df = add_features(df_raw)

# Alleen valide sessies (geen gaps)
df = df[df['session_id'] != -1].copy()
valid_sessions = sorted(df['session_id'].unique())

# B. Kapitaal & Historie Check
current_capital = START_CAPITAL
last_log_time = pd.Timestamp("1900-01-01")

if os.path.exists(log_path):
    existing_logs = pd.read_csv(log_path)
    if not existing_logs.empty:
        existing_logs['entry_time'] = pd.to_datetime(existing_logs['entry_time'])
        last_log_time = existing_logs['entry_time'].max()
        print(f"Laatst verwerkte tijdstip: {last_log_time}")
        
        # Bereken kapitaal uit logs
        if 'profit_abs' in existing_logs.columns:
            total_profit = existing_logs['profit_abs'].sum()
            current_capital = START_CAPITAL + total_profit
            print(f"Historie gedetecteerd. Huidig Kapitaal: €{current_capital:.2f}")

# C. Bepaal welke sessies we moeten verwerken
# We zoeken de sessie ID die bij de last_log_time hoort, en pakken alles daarna
last_processed_idx = df[df['time'] <= last_log_time].index.max()

if pd.isna(last_processed_idx):
    # Nog geen data verwerkt, begin bij window size
    start_session_idx = ROLLING_WINDOW_SESSIONS
else:
    # Kijk welke sessie ID de laatste log had
    last_sess_id = df.loc[last_processed_idx, 'session_id']
    try:
        # We beginnen bij de sessie index in de valid_sessions lijst
        # We verwerken de huidige sessie opnieuw voor nieuwe trades, of gaan naar de volgende
        start_session_idx = valid_sessions.index(last_sess_id)
    except ValueError:
        start_session_idx = ROLLING_WINDOW_SESSIONS

# Zorg dat we genoeg historie hebben
if start_session_idx < ROLLING_WINDOW_SESSIONS:
    start_session_idx = ROLLING_WINDOW_SESSIONS

sessions_to_process = valid_sessions[start_session_idx:]

if not sessions_to_process:
    print("Geen nieuwe sessies om te verwerken.")
    exit()

print(f"Te verwerken sessie IDs: {len(sessions_to_process)}")

# D. De Grote Loop (Over Sessies)
all_new_trades = []

for i, target_sess_id in enumerate(sessions_to_process):
    # Index in de valid_sessions lijst terugvinden
    global_idx = valid_sessions.index(target_sess_id)
    
    # 1. Training Window Bepalen (Vorige N sessies)
    train_start_sess = valid_sessions[global_idx - ROLLING_WINDOW_SESSIONS]
    train_end_sess = valid_sessions[global_idx - 1] # Tot de vorige sessie
    
    mask_train = (df['session_id'] >= train_start_sess) & (df['session_id'] <= train_end_sess)
    train_df = df.loc[mask_train].copy()
    
    mask_target = (df['session_id'] == target_sess_id)
    target_df = df.loc[mask_target].copy()
    
    if train_df.empty or target_df.empty: continue
    
    # 2. Trainen
    rules = train_rules(train_df)
    
    # 3. Traden (Intraday Loop)
    positions = []
    cooldown = 0
    
    for idx, row in target_df.iterrows():
        # Sla over wat al verwerkt is
        if row['time'] <= last_log_time: continue
        
        curr_time = row['time']
        
        # --- EXITS ---
        for pos in positions[:]:
            close_trade = False
            exit_price = 0.0
            exit_reason = ""
            
            # Target Hit
            if row['high_bid'] >= pos['target_price']:
                exit_price = pos['target_price']
                exit_reason = "Target Hit"
                close_trade = True
            # Einde Sessie (Laatste datapunt van deze sessie)
            elif idx == target_df.index[-1]:
                exit_price = row['close_bid']
                exit_reason = "Session End"
                close_trade = True
            # Harde tijd-exit (veiligheid)
            elif row['time'].hour >= 22:
                exit_price = row['close_bid']
                exit_reason = "Time Exit"
                close_trade = True
                
            if close_trade:
                roi = (exit_price - pos['entry_price']) / pos['entry_price']
                profit_abs = (exit_price - pos['entry_price']) * pos['units']
                
                # Kapitaal direct updaten!
                current_capital += profit_abs
                
                all_new_trades.append({
                    'entry_time': pos['entry_time'],
                    'entry_p': pos['entry_price'],
                    'side': 'Long',
                    'units': pos['units'],
                    'invested_cash': pos['invested_cash'],
                    'exit_time': curr_time,
                    'exit_p': exit_price,
                    'return': roi,
                    'profit_abs': profit_abs,
                    'exit_reason': exit_reason,
                    'session_id': target_sess_id
                })
                positions.remove(pos)
        
        # --- ENTRIES ---
        if cooldown > 0: cooldown -= 1
        
        # Check Rules & Filters
        valid_rng = row['range_pct'] > MIN_RANGE
        state = (row['prev_trend'], row['pos_bin'], row['rsi_bin'], row['hour'])
        
        # Alleen instappen als: 
        # 1. We rules hebben
        # 2. Slots vrij zijn
        # 3. Geen cooldown
        # 4. Tijd voor 20:00 (geen late entries)
        if (state in rules) and valid_rng and (len(positions) < MAX_SLOTS) and \
           (cooldown == 0) and (row['time'].hour < 20):
            
            ask = row['close_ask']
            target = ask + (row['sess_range'] * TARGET_RANGE_RATIO)
            
            # --- WHOLE UNIT CALCULATION ---
            slot_cash = current_capital / MAX_SLOTS
            buying_power = slot_cash * LEVERAGE
            units = int(buying_power / ask)
            
            if units >= 1:
                positions.append({
                    'entry_time': curr_time,
                    'entry_price': ask,
                    'target_price': target,
                    'units': units,
                    'invested_cash': slot_cash
                })
                cooldown = COOLDOWN_MINUTES

# E. Opslaan
if all_new_trades:
    new_trades_df = pd.DataFrame(all_new_trades)
    print(f"Nieuwe trades gemaakt: {len(new_trades_df)}")
    
    if not existing_logs.empty:
        # Zorg voor gelijke kolommen
        for col in new_trades_df.columns:
            if col not in existing_logs.columns: existing_logs[col] = np.nan
        final_df = pd.concat([existing_logs, new_trades_df], ignore_index=True)
    else:
        final_df = new_trades_df
        
    final_df = final_df.sort_values('entry_time')
    final_df.to_csv(log_path, index=False)
    print(f"Log opgeslagen. Nieuw saldo: €{current_capital:.2f}")
else:
    print("Geen nieuwe trades gevonden.")

print("--- RUN COMPLETE ---")
