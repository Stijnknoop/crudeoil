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
ROLLING_WINDOW_DAYS = 40     # Aantal dagen historie om op te trainen

# --- ACCOUNT & RISK ---
START_CAPITAL = 10000
MAX_SLOTS = 10
LEVERAGE = 5
# LET OP: SLOT_SIZE_CASH is nu dynamisch en wordt in de loop berekend!
COOLDOWN_MINUTES = 10

# --- STRATEGIE FILTERS (DYNAMISCH) ---
MIN_HISTORICAL_TRADES = 15       # Minimaal aantal trades in de afgelopen 40 dagen
MIN_EXPECTED_ROI = 0.0025        # 0.25% gemiddelde winst per trade vereist
MIN_RANGE_PCT = 0.0008           # Range moet min. 0.08% van de prijs zijn
TARGET_RANGE_RATIO = 0.5         # Target is 0.5x de Range

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
    
    # 4. Binning & Filters
    df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutraal', 'Overbought'])
    df['pos_bin'] = pd.cut(df['position_pct'], bins=[-0.1, 0.3, 0.7, 1.1], labels=['Low', 'Mid', 'High'])
    df['prev_day_trend'] = np.where(df['day_change'].fillna(0) > 0, 'Groen', 'Rood')
    
    # Percentuele range voor filter
    df['range_pct_of_price'] = df['range_so_far'] / df['close_bid']
    
    return df

# ==============================================================================
# 3. DE TRAINER (OP EEN SPECIFIEKE SLICE DATA)
# ==============================================================================
def train_on_window(train_df):
    """
    Berekent de winstgevende regels op basis van de meegegeven 'train_df' (de window).
    """
    entry = train_df['close_ask']
    target_dist = train_df['range_so_far'] * TARGET_RANGE_RATIO
    eod_close = train_df.groupby('date')['close_bid'].transform('last')
    
    # Look-ahead binnen de training set
    max_future = train_df.iloc[::-1].groupby('date')['close_bid'].cummax().iloc[::-1]
    
    # Win Condities
    valid_range = train_df['range_pct_of_price'] > MIN_RANGE_PCT
    is_win = (max_future >= (entry + target_dist)) & valid_range
    
    # ROI Berekening
    pnl_abs = np.where(is_win, target_dist, eod_close - entry)
    train_df = train_df.assign(roi_pct = pnl_abs / entry)
    
    # Statistieken per conditie
    stats = train_df.groupby(
        ['prev_day_trend', 'pos_bin', 'rsi_bin', 'hour'], 
        observed=False
    )['roi_pct'].agg(['mean', 'count'])
    
    # Filter de regels
    approved_conditions = stats[
        (stats['count'] >= MIN_HISTORICAL_TRADES) & 
        (stats['mean'] > MIN_EXPECTED_ROI) 
    ].index.tolist()
    
    return set(approved_conditions)

# ==============================================================================
# 4. HOOFDPROGRAMMA (ROLLING WINDOW BOT)
# ==============================================================================
print(f"--- START ROLLING WINDOW BOT ({ROLLING_WINDOW_DAYS} DAGEN) ---")

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(OUTPUT_DIR, LOG_FILE)

# A. Data laden
df_raw = read_latest_csv_from_github()
if df_raw is None: exit()
df = prepare_data(df_raw)

# B. Bepaal historie en startbalans
unique_dates = sorted(df['date'].unique())
current_capital = START_CAPITAL

if os.path.exists(log_path):
    existing_logs = pd.read_csv(log_path)
    if not existing_logs.empty:
        existing_logs['entry_time'] = pd.to_datetime(existing_logs['entry_time'])
        last_log_date = existing_logs['entry_time'].max().date()
        print(f"Laatst verwerkte datum in logs: {last_log_date}")
        
        # --- BEREKEN HUIDIG KAPITAAL OP BASIS VAN HISTORIE ---
        # Als we al kolommen 'profit_abs' hebben, gebruiken we die.
        # Zo niet (oude CSV), dan schatten we het op basis van de oude statische logica.
        if 'profit_abs' in existing_logs.columns:
            total_profit = existing_logs['profit_abs'].sum()
            current_capital = START_CAPITAL + total_profit
            print(f"Historie gedetecteerd. Kapitaal bijgewerkt naar: €{current_capital:.2f}")
        else:
            # Fallback voor oude logs zonder absolute profit data
            # We nemen aan dat oude trades statisch waren (Start / 10)
            static_invest = START_CAPITAL / MAX_SLOTS
            estimated_profit = (existing_logs['return'] * static_invest * LEVERAGE).sum()
            current_capital = START_CAPITAL + estimated_profit
            print(f"Oude logs gedetecteerd (geen cash data). Geschat kapitaal: €{current_capital:.2f}")

        # Start datum bepalen
        try:
            start_index = unique_dates.index(last_log_date) + 1
        except ValueError:
            future_dates = [d for d in unique_dates if d > last_log_date]
            if not future_dates:
                print("Alles is al bijgewerkt.")
                exit()
            start_index = unique_dates.index(future_dates[0])
    else:
        start_index = ROLLING_WINDOW_DAYS
else:
    existing_logs = pd.DataFrame()
    start_index = ROLLING_WINDOW_DAYS

# Check of we genoeg historie hebben om te starten
if start_index < ROLLING_WINDOW_DAYS:
    print(f"Nog niet genoeg data voor eerste training. Start bij index {ROLLING_WINDOW_DAYS}.")
    start_index = ROLLING_WINDOW_DAYS

# Dagen die we gaan verwerken
days_to_process = unique_dates[start_index:]

if not days_to_process:
    print("Geen nieuwe dagen om te verwerken.")
    exit()

print(f"Gevonden nieuwe dagen: {len(days_to_process)}")
print(f"Start Kapitaal voor deze run: €{current_capital:.2f}")

# C. De Grote Loop
all_new_trades = []

for target_day in days_to_process:
    target_idx = unique_dates.index(target_day)
    
    # 1. Bepaal de Training Window
    train_start_date = unique_dates[target_idx - ROLLING_WINDOW_DAYS]
    train_end_date = unique_dates[target_idx - 1]
    
    print(f"Processing {target_day} | Saldo: €{current_capital:.2f} | Training: {train_start_date} t/m {train_end_date}")
    
    # Slice Data
    mask_train = (df['date'] >= train_start_date) & (df['date'] <= train_end_date)
    train_df = df.loc[mask_train].copy()
    
    mask_test = (df['date'] == target_day)
    test_df = df.loc[mask_test].copy()
    
    if train_df.empty or test_df.empty:
        continue

    # 2. Trainen
    daily_rules_set = train_on_window(train_df)
    
    if not daily_rules_set:
        continue

    # 3. Traden op de Target Dag
    positions = []
    cooldown = 0
    
    for i, row in test_df.iterrows():
        current_time = row['time']
        current_bid = row['close_bid']
        current_ask = row['close_ask']
        
        # --- POSITIE MANAGEMENT (EXITS) ---
        for pos in positions[:]:
            close_trade = False
            exit_reason = ""
            exit_price = 0.0
            outcome = ""

            # Target Hit?
            if row['high_bid'] >= pos['target_price']:
                exit_price = pos['target_price']
                exit_reason = 'Target Hit'
                close_trade = True
                
            # EOD Timeout
            elif row['time_str'] >= '22:00':
                exit_price = current_bid
                exit_reason = 'EOD Timeout'
                close_trade = True

            if close_trade:
                roi = (exit_price - pos['entry_price']) / pos['entry_price']
                
                # BEREKEN CASH RESULTAAT (Winst/Verlies * Units)
                # Units = (Cash Inleg * Leverage) / Entry Price
                # Dus Cash Winst = (Units * Exit Price) - (Units * Entry Price)
                # Of simpeler: Invested_Cash * Leverage * ROI
                
                # Omdat we met CFD/Futures logica werken, berekenen we het via de units:
                exit_value = pos['units'] * exit_price
                entry_value = pos['units'] * pos['entry_price']
                gross_profit_abs = exit_value - entry_value
                
                # UPDATE HET KAPITAAL DIRECT (COMPOUNDING)
                current_capital += gross_profit_abs
                
                all_new_trades.append({
                    'day': row['date'],
                    'entry_time': pos['entry_time'],
                    'entry_p': pos['entry_price'],
                    'side': 'Long',
                    'leverage': LEVERAGE,
                    'invested_cash': pos['invested_cash'],  # Opslaan voor analyse
                    'exit_time': current_time,
                    'exit_p': exit_price,
                    'return': roi,
                    'profit_abs': gross_profit_abs,         # Opslaan voor balans berekening
                    'exit_reason': exit_reason,
                    'outcome': 'WIN' if roi > 0 else 'LOSS'
                })
                positions.remove(pos)
        
        # --- ENTRY LOGICA (ENTRIES) ---
        if cooldown > 0: cooldown -= 1
        
        valid_range = (row['range_so_far'] / current_ask) > MIN_RANGE_PCT
        
        if (len(positions) < MAX_SLOTS) and (cooldown == 0) and \
           (row['time_str'] < '20:00') and valid_range:
            
            current_state = (row['prev_day_trend'], row['pos_bin'], row['rsi_bin'], row['hour'])
            
            if current_state in daily_rules_set:
                target_gain = row['range_so_far'] * TARGET_RANGE_RATIO
                
                # --- DYNAMISCHE POSITIE GROOTTE ---
                # 1/10e van het HUIDIGE kapitaal
                dynamic_slot_size = current_capital / MAX_SLOTS
                
                # Totale koopkracht voor deze trade
                effective_investment = dynamic_slot_size * LEVERAGE
                
                # Aantal units (vaten) dat we kunnen kopen
                units = effective_investment / current_ask
                
                positions.append({
                    'entry_time': current_time,
                    'entry_price': current_ask,
                    'target_price': current_ask + target_gain,
                    'units': units,
                    'invested_cash': dynamic_slot_size 
                })
                cooldown = COOLDOWN_MINUTES


# D. Opslaan
if all_new_trades:
    new_trades_df = pd.DataFrame(all_new_trades)
    print(f"Totaal nieuwe trades gegenereerd: {len(new_trades_df)}")
    
    if not existing_logs.empty:
        # Zorg dat de kolommen matchen (vul lege kolommen in oude logs op indien nodig)
        if 'invested_cash' not in existing_logs.columns:
            existing_logs['invested_cash'] = np.nan
        if 'profit_abs' not in existing_logs.columns:
            existing_logs['profit_abs'] = np.nan
            
        final_df = pd.concat([existing_logs, new_trades_df], ignore_index=True)
    else:
        final_df = new_trades_df
        
    final_df['entry_time'] = pd.to_datetime(final_df['entry_time'])
    final_df = final_df.sort_values('entry_time')
    
    final_df.to_csv(log_path, index=False)
    print(f"Log bijgewerkt: {log_path}")
    print(f"Nieuw Eindsaldo: €{current_capital:.2f}")
else:
    print("Geen trades gemaakt in de verwerkte dagen.")

print("--- RUN VOLTOOID ---")
