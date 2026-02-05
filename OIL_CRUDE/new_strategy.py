import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATIE
# ==============================================================================
GITHUB_USER = "Stijnknoop"
GITHUB_REPO = "crudeoil"
FOLDER_PATH = "OIL_CRUDE"
OUTPUT_DIR = "OIL_CRUDE/Trading_details"
LOG_FILE = "trading_logs.csv"

# Trading Instellingen
START_CAPITAL = 10000
MAX_SLOTS = 10
SLOT_SIZE = START_CAPITAL / MAX_SLOTS
COOLDOWN_MINUTES = 10

# Strategie Filters (De "Sniper" settings)
MIN_HISTORICAL_TRADES = 15   # Minimaal aantal keer voorgekomen in historie
MIN_EXPECTED_PROFIT = 0.15   # Gemiddelde winst moet groter zijn dan dit (in prijs-eenheden)

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
    
    # 2. Dag Context (Gisteren vs Vandaag)
    daily = df.groupby('date')['close_bid'].agg(['first', 'last']).rename(columns={'first': 'open', 'last': 'close'})
    daily['prev_close'] = daily['close'].shift(1)
    daily['day_change'] = (daily['close'] - daily['prev_close']) / daily['prev_close']
    df = df.merge(daily[['prev_close', 'day_change']], on='date', how='left')
    
    # 3. Intraday Positie (Range tot nu toe)
    df['min_so_far'] = df.groupby('date')['close_bid'].cummin()
    df['max_so_far'] = df.groupby('date')['close_bid'].cummax()
    df['range_so_far'] = df['max_so_far'] - df['min_so_far']
    
    # Positie %
    df['position_pct'] = (df['close_bid'] - df['min_so_far']) / df['range_so_far']
    df['position_pct'] = df['position_pct'].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # 4. Binning (Het vertalen naar categorieën)
    df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutraal', 'Overbought'])
    df['pos_bin'] = pd.cut(df['position_pct'], bins=[-0.1, 0.3, 0.7, 1.1], labels=['Low', 'Mid', 'High'])
    df['prev_day_trend'] = np.where(df['day_change'].fillna(0) > 0, 'Groen', 'Rood')
    
    return df

# ==============================================================================
# 3. DE "BRAIN" (DYNAMISCHE STRATEGIE TRAINING)
# ==============================================================================
def train_strategy_on_fly(df):
    """
    Deze functie berekent on-the-fly wat de winstgevende condities zijn
    op basis van de huidige dataset.
    """
    print("Strategie aan het herberekenen op basis van historie...")
    
    # We maken een tijdelijke copy om targets te berekenen
    train_df = df.copy()
    
    # Definieer wat "Succes" was in het verleden
    entry = train_df['close_ask']
    target = train_df['range_so_far'] * 0.5
    eod_close = train_df.groupby('date')['close_bid'].transform('last')
    
    # Look-ahead (We kijken stiekem in de toekomst van het verleden om te leren)
    max_future = train_df.iloc[::-1].groupby('date')['close_bid'].cummax().iloc[::-1]
    
    is_win = (max_future >= (entry + target)) & (train_df['range_so_far'] > 0.05)
    
    # P&L per rij berekenen
    train_df['theoretical_pnl'] = np.where(is_win, target, eod_close - entry)
    
    # Groepeer op onze 4 variabelen
    stats = train_df.groupby(
        ['prev_day_trend', 'pos_bin', 'rsi_bin', 'hour'], 
        observed=False
    )['theoretical_pnl'].agg(['mean', 'count'])
    
    # FILTER: De "Sniper" Regels
    approved_conditions = stats[
        (stats['count'] >= MIN_HISTORICAL_TRADES) & 
        (stats['mean'] > MIN_EXPECTED_PROFIT)
    ].index.tolist()
    
    # Maak een set voor snelle lookup
    # Dit is nu je dynamische "Rulebook"
    approved_set = set(approved_conditions)
    
    print(f"Dynamische training voltooid. {len(approved_set)} winstgevende scenario's gevonden.")
    return approved_set

# ==============================================================================
# 4. HOOFDPROGRAMMA
# ==============================================================================
print("--- START AUTO-ADAPTIVE TRADING BOT ---")

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(OUTPUT_DIR, LOG_FILE)

# A. Data Ophalen
df_raw = read_latest_csv_from_github()
if df_raw is None: exit()

df = prepare_data(df_raw)

# B. Strategie Bepalen (Elke keer opnieuw trainen!)
winning_rules_set = train_strategy_on_fly(df)

# C. Logs Checken (Waar waren we gebleven?)
if os.path.exists(log_path):
    existing_logs = pd.read_csv(log_path)
    if not existing_logs.empty:
        # Zorg dat datum conversie goed gaat
        existing_logs['entry_time'] = pd.to_datetime(existing_logs['entry_time'])
        last_trade_time = existing_logs['entry_time'].max()
        print(f"Laatst gelogde trade: {last_trade_time}")
    else:
        last_trade_time = pd.Timestamp.min
else:
    existing_logs = pd.DataFrame()
    last_trade_time = pd.Timestamp.min

# D. Filter Nieuwe Data
# We willen alleen handelen op data die NA de laatste trade komt
df_sim = df[df['time'] > last_trade_time].copy()

if df_sim.empty:
    print("Geen nieuwe data om te verwerken.")
    exit()

print(f"Verwerken van {len(df_sim)} nieuwe minuten data...")

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
            new_trades.append({
                'day': row['date'],
                'entry_time': pos['entry_time'],
                'entry_p': pos['entry_price'],
                'side': 'Long',
                'exit_time': current_time,
                'exit_p': pos['target_price'],
                'return': (pos['target_price'] - pos['entry_price']) / pos['entry_price'],
                'exit_reason': 'Target Hit',
                'outcome': 'WIN'
            })
            positions.remove(pos)
            
        # Einde Dag Timeout?
        elif row['time_str'] >= '22:00':
            pnl = (current_bid - pos['entry_price']) / pos['entry_price']
            new_trades.append({
                'day': row['date'],
                'entry_time': pos['entry_time'],
                'entry_p': pos['entry_price'],
                'side': 'Long',
                'exit_time': current_time,
                'exit_p': current_bid,
                'return': pnl,
                'exit_reason': 'EOD Timeout',
                'outcome': 'WIN' if pnl > 0 else 'LOSS'
            })
            positions.remove(pos)

    # 2. Entry Logica
    if cooldown > 0: cooldown -= 1
    
    if (len(positions) < MAX_SLOTS) and (cooldown == 0) and \
       (row['time_str'] < '20:00') and (row['range_so_far'] > 0.05):
        
        # Wat is de markt situatie NU?
        current_state = (
            row['prev_day_trend'], 
            row['pos_bin'], 
            row['rsi_bin'], 
            row['hour']
        )
        
        # Is dit een situatie die historisch winstgevend is?
        if current_state in winning_rules_set:
            # JA! Kopen.
            target_gain = row['range_so_far'] * 0.5
            
            positions.append({
                'entry_time': current_time,
                'entry_price': current_ask,
                'target_price': current_ask + target_gain,
                'units': SLOT_SIZE / current_ask
            })
            cooldown = COOLDOWN_MINUTES

# F. Opslaan
if new_trades:
    new_trades_df = pd.DataFrame(new_trades)
    print(f"Nieuwe trades gevonden: {len(new_trades_df)}")
    
    if not existing_logs.empty:
        final_df = pd.concat([existing_logs, new_trades_df], ignore_index=True)
    else:
        final_df = new_trades_df
        
    final_df['entry_time'] = pd.to_datetime(final_df['entry_time'])
    final_df = final_df.sort_values('entry_time')
    
    final_df.to_csv(log_path, index=False)
    print(f"Log bijgewerkt: {log_path}")
else:
    print("Geen nieuwe trades gegenereerd.")

print("--- RUN VOLTOOID ---")
