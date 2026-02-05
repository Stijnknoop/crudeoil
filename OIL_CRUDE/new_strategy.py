import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATIE & INSTELLINGEN
# ==============================================================================
GITHUB_USER = "Stijnknoop"
GITHUB_REPO = "crudeoil"
FOLDER_PATH = "OIL_CRUDE"
OUTPUT_DIR = "OIL_CRUDE/Trading_details"
LOG_FILE = "trading_logs.csv"

# --- ACCOUNT & RISK ---
START_CAPITAL = 10000
MAX_SLOTS = 10
LEVERAGE = 5                     # <--- NIEUW: Hefboom (Multiplier)
SLOT_SIZE_CASH = START_CAPITAL / MAX_SLOTS 
COOLDOWN_MINUTES = 10

# --- STRATEGIE FILTERS (DYNAMISCH) ---
MIN_HISTORICAL_TRADES = 15       # Minimaal aantal keer voorgekomen in historie
MIN_EXPECTED_ROI = 0.0025        # <--- NIEUW: 0.0025 = 0.25% verwachte winst per trade (ipv harde 0.15)
MIN_RANGE_PCT = 0.0008           # <--- NIEUW: Range moet min. 0.08% van de prijs zijn (vervangt > 0.05)
TARGET_RANGE_RATIO = 0.5         # <--- NIEUW: Target is 0.5x de Range (was hardcoded)

# ==============================================================================
# 2. DATA OPHALEN & FEATURES
# ==============================================================================
def read_latest_csv_from_github():
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{FOLDER_PATH}?ref=master"
    
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code != 200: return None
        files = response.json()
        csv_file = next((f for f in files if f['name'].endswith('.csv')), None)
        if not csv_file: return None
        print(f"Data gedownload: {csv_file['name']}")
        return pd.read_csv(csv_file['download_url'])
    except Exception as e:
        print(f"Error: {e}")
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def prepare_data(df):
    df['time'] = pd.to_datetime(df['time'], format='ISO8601')
    df = df.sort_values('time').reset_index(drop=True)
    
    df['date'] = df['time'].dt.date
    df['time_str'] = df['time'].dt.strftime('%H:%M')
    df['hour'] = df['time'].dt.hour
    
    # 1. RSI
    df['rsi'] = calculate_rsi(df['close_bid'])
    
    # 2. Dag Context
    daily = df.groupby('date')['close_bid'].agg(['first', 'last']).rename(columns={'first': 'open', 'last': 'close'})
    daily['prev_close'] = daily['close'].shift(1)
    daily['day_change'] = (daily['close'] - daily['prev_close']) / daily['prev_close']
    df = df.merge(daily[['prev_close', 'day_change']], on='date', how='left')
    
    # 3. Intraday Positie
    df['min_so_far'] = df.groupby('date')['close_bid'].cummin()
    df['max_so_far'] = df.groupby('date')['close_bid'].cummax()
    df['range_so_far'] = df['max_so_far'] - df['min_so_far']
    
    # Positie %
    df['position_pct'] = (df['close_bid'] - df['min_so_far']) / df['range_so_far']
    df['position_pct'] = df['position_pct'].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # 4. Binning
    df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutraal', 'Overbought'])
    df['pos_bin'] = pd.cut(df['position_pct'], bins=[-0.1, 0.3, 0.7, 1.1], labels=['Low', 'Mid', 'High'])
    df['prev_day_trend'] = np.where(df['day_change'].fillna(0) > 0, 'Groen', 'Rood')
    
    # 5. Ruis Filter (Relatief aan prijs)
    # Range moet groter zijn dan X% van de huidige prijs
    df['range_pct_of_price'] = df['range_so_far'] / df['close_bid']
    
    return df

# ==============================================================================
# 3. DE "BRAIN" (DYNAMISCHE STRATEGIE TRAINING)
# ==============================================================================
def train_strategy_on_fly(df):
    print("Strategie aan het herberekenen (Percentueel)...")
    train_df = df.copy()
    
    entry = train_df['close_ask']
    # Target is variabel ingesteld (bv 0.5x Range)
    target_dist = train_df['range_so_far'] * TARGET_RANGE_RATIO
    eod_close = train_df.groupby('date')['close_bid'].transform('last')
    
    # Look-ahead
    max_future = train_df.iloc[::-1].groupby('date')['close_bid'].cummax().iloc[::-1]
    
    # Win Conditie: Target geraakt EN Range groot genoeg (Percentueel filter)
    valid_range = train_df['range_pct_of_price'] > MIN_RANGE_PCT
    is_win = (max_future >= (entry + target_dist)) & valid_range
    
    # P&L calculation (NU PERCENTUEEL!)
    pnl_abs = np.where(is_win, target_dist, eod_close - entry)
    train_df['roi_pct'] = pnl_abs / entry # Return on Investment per trade
    
    # Groepeer
    stats = train_df.groupby(
        ['prev_day_trend', 'pos_bin', 'rsi_bin', 'hour'], 
        observed=False
    )['roi_pct'].agg(['mean', 'count'])
    
    # FILTER: "Sniper" Regels op basis van ROI
    # We willen trades die gemiddeld > 0.25% (of user setting) opleveren
    approved_conditions = stats[
        (stats['count'] >= MIN_HISTORICAL_TRADES) & 
        (stats['mean'] > MIN_EXPECTED_ROI) 
    ].index.tolist()
    
    approved_set = set(approved_conditions)
    
    print(f"Training voltooid. {len(approved_set)} scenario's gevonden met >{MIN_EXPECTED_ROI*100}% verwachte winst.")
    return approved_set

# ==============================================================================
# 4. HOOFDPROGRAMMA
# ==============================================================================
print(f"--- START BOT (LEVERAGE: {LEVERAGE}x) ---")

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(OUTPUT_DIR, LOG_FILE)

# A. Data Ophalen
df_raw = read_latest_csv_from_github()
if df_raw is None: exit()

df = prepare_data(df_raw)

# B. Strategie Trainen
winning_rules_set = train_strategy_on_fly(df)

# C. Logs Checken
if os.path.exists(log_path):
    existing_logs = pd.read_csv(log_path)
    if not existing_logs.empty:
        existing_logs['entry_time'] = pd.to_datetime(existing_logs['entry_time'])
        last_trade_time = existing_logs['entry_time'].max()
        print(f"Laatst gelogde trade: {last_trade_time}")
    else:
        last_trade_time = pd.Timestamp.min
else:
    existing_logs = pd.DataFrame()
    last_trade_time = pd.Timestamp.min

# D. Nieuwe Data Selecteren
df_sim = df[df['time'] > last_trade_time].copy()

if df_sim.empty:
    print("Geen nieuwe data.")
    exit()

print(f"Analyseren van {len(df_sim)} nieuwe minuten...")

# E. Simulatie Loop
new_trades = []
positions = []
cooldown = 0

for i, row in df_sim.iterrows():
    current_time = row['time']
    current_bid = row['close_bid']
    current_ask = row['close_ask']
    
    # 1. Posities Managen
    for pos in positions[:]:
        # Target Hit?
        if row['high_bid'] >= pos['target_price']:
            roi = (pos['target_price'] - pos['entry_price']) / pos['entry_price']
            new_trades.append({
                'day': row['date'],
                'entry_time': pos['entry_time'],
                'entry_p': pos['entry_price'],
                'side': 'Long',
                'leverage': LEVERAGE, # Loggen voor administratie
                'exit_time': current_time,
                'exit_p': pos['target_price'],
                'return': roi, 
                'exit_reason': 'Target Hit',
                'outcome': 'WIN'
            })
            positions.remove(pos)
            
        # Einde Dag Timeout?
        elif row['time_str'] >= '22:00':
            roi = (current_bid - pos['entry_price']) / pos['entry_price']
            new_trades.append({
                'day': row['date'],
                'entry_time': pos['entry_time'],
                'entry_p': pos['entry_price'],
                'side': 'Long',
                'leverage': LEVERAGE,
                'exit_time': current_time,
                'exit_p': current_bid,
                'return': roi,
                'exit_reason': 'EOD Timeout',
                'outcome': 'WIN' if roi > 0 else 'LOSS'
            })
            positions.remove(pos)

    # 2. Entry Logica
    if cooldown > 0: cooldown -= 1
    
    # Check Ruis Filter (Percentueel)
    valid_range = (row['range_so_far'] / current_ask) > MIN_RANGE_PCT

    if (len(positions) < MAX_SLOTS) and (cooldown == 0) and \
       (row['time_str'] < '20:00') and valid_range:
        
        current_state = (
            row['prev_day_trend'], 
            row['pos_bin'], 
            row['rsi_bin'], 
            row['hour']
        )
        
        if current_state in winning_rules_set:
            # ENTRY BEREKENING MET LEVERAGE
            target_gain = row['range_so_far'] * TARGET_RANGE_RATIO
            
            # Positie grootte met hefboom
            # Effectief bedrag om te kopen = (Cash per Slot) * Leverage
            effective_investment = SLOT_SIZE_CASH * LEVERAGE
            units = effective_investment / current_ask
            
            positions.append({
                'entry_time': current_time,
                'entry_price': current_ask,
                'target_price': current_ask + target_gain,
                'units': units
            })
            cooldown = COOLDOWN_MINUTES

# F. Opslaan
if new_trades:
    new_trades_df = pd.DataFrame(new_trades)
    print(f"Nieuwe trades: {len(new_trades_df)}")
    
    if not existing_logs.empty:
        final_df = pd.concat([existing_logs, new_trades_df], ignore_index=True)
    else:
        final_df = new_trades_df
        
    final_df['entry_time'] = pd.to_datetime(final_df['entry_time'])
    final_df = final_df.sort_values('entry_time')
    
    final_df.to_csv(log_path, index=False)
    print(f"Log bijgewerkt: {log_path}")
else:
    print("Geen nieuwe trades.")

print("--- RUN VOLTOOID ---")
