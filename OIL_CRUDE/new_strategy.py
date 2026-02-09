import requests
import pandas as pd
import numpy as np
import os
import matplotlib
# Belangrijk voor GitHub Actions (geen scherm):
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Toegevoegd voor datum opmaak

# ==============================================================================
# 1. CONFIGURATIE
# ==============================================================================
GITHUB_USER = "Stijnknoop"
GITHUB_REPO = "crudeoil"
FOLDER_PATH = "OIL_CRUDE"
OUTPUT_DIR = "OIL_CRUDE/Trading_details"

# Zorg dat de output map bestaat
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Instellingen Backtest
ROLLING_WINDOW_SIZE = 40
START_CAPITAL = 650.0

CONFIG = {
    'target_range_ratio': 0.5,
    'min_trades': 20,
    'min_roi': 0.0045,
    'min_range': 0.0008,
    'cooldown_minutes': 15,
    'max_slots': 10,
    'leverage': 1
}

# ==============================================================================
# 2. DATA FUNCTIES
# ==============================================================================
def get_data_and_process():
    print("--- Data ophalen en verwerken ---")
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{FOLDER_PATH}?ref=master"

    try:
        r = requests.get(api_url, headers=headers).json()
        if isinstance(r, list):
            csv_file = next(f for f in r if f['name'].endswith('.csv'))
            download_url = csv_file['download_url']
        else:
            download_url = r['download_url']
            
        df = pd.read_csv(download_url)

        df['time'] = pd.to_datetime(df['time'], format='ISO8601')
        df = df.set_index('time').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample('1min').ffill().dropna().reset_index()

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
    df = df.copy()

    # RSI
    delta = df['close_bid'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + (avg_gain/avg_loss)))

    # Sessie info
    sess_stats = df[df['session_id'] != -1].groupby('session_id')['close_bid'].agg(['last'])
    sess_stats['prev_trend'] = np.where((sess_stats['last'] - sess_stats['last'].shift(1)) > 0, 'Groen', 'Rood')
    df = df.merge(sess_stats[['prev_trend']], on='session_id', how='left')

    # Intraday stats
    df['sess_min'] = df.groupby('session_id')['close_bid'].cummin()
    df['sess_max'] = df.groupby('session_id')['close_bid'].cummax()
    df['sess_range'] = df['sess_max'] - df['sess_min']

    df['pos_pct'] = ((df['close_bid'] - df['sess_min']) / (df['sess_range'] + 1e-9)).fillna(0.5)
    df['range_pct'] = df['sess_range'] / df['close_bid']

    # Bins
    df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutraal', 'Overbought'])
    df['pos_bin'] = pd.cut(df['pos_pct'], bins=[-0.1, 0.3, 0.7, 1.1], labels=['Low', 'Mid', 'High'])

    # Kwartier berekening
    df['quarter_hour'] = (df['time'].dt.hour * 4) + (df['time'].dt.minute // 15)

    return df

# ==============================================================================
# 3. TRAINING LOGICA
# ==============================================================================
def train_rules(train_df, cfg):
    entry = train_df['close_ask']
    target_dist = train_df['sess_range'] * cfg['target_range_ratio']
    max_future = train_df.iloc[::-1].groupby('session_id')['close_bid'].cummax().iloc[::-1]

    is_win = (max_future >= (entry + target_dist)) & (train_df['range_pct'] > cfg['min_range'])
    
    sess_close = train_df.groupby('session_id')['close_bid'].transform('last')
    pnl = np.where(is_win, target_dist, sess_close - entry)
    
    train_df = train_df.assign(roi = pnl / entry)

    stats = train_df.groupby(['prev_trend', 'pos_bin', 'rsi_bin', 'quarter_hour'], observed=True)['roi'].agg(['mean', 'count'])
    winning = stats[(stats['count'] >= cfg['min_trades']) & (stats['mean'] > cfg['min_roi'])]

    return set(winning.index.tolist())

# ==============================================================================
# 4. SIMULATIE
# ==============================================================================
def run_simulation():
    df_raw = get_data_and_process()
    if df_raw is None: 
        return

    valid_sessions = sorted(df_raw[df_raw['session_id'] != -1]['session_id'].unique())

    current_capital = START_CAPITAL
    capital_history = [START_CAPITAL]
    session_dates = [] # NIEUW: Lijst voor datums
    
    action_log = [] 

    print(f"\n--- START SIMULATIE ---")
    print(f"Start Kapitaal: €{current_capital:.2f}")

    for i in range(ROLLING_WINDOW_SIZE, len(valid_sessions)):
        target_session_id = valid_sessions[i]
        start_train_id = valid_sessions[i - ROLLING_WINDOW_SIZE]

        mask = (df_raw['session_id'] >= start_train_id) & (df_raw['session_id'] <= target_session_id)
        window_df = df_raw.loc[mask].copy()
        proc_df = add_features(window_df)

        train_df = proc_df[proc_df['session_id'] < target_session_id]
        target_df = proc_df[proc_df['session_id'] == target_session_id].copy().reset_index(drop=True)

        # NIEUW: Vang de datum van deze sessie
        if not target_df.empty:
            session_dates.append(target_df['time'].iloc[0])
        else:
            # Fallback voor het geval een sessie leeg is (zou niet moeten gebeuren)
            session_dates.append(pd.Timestamp.now())

        rules = train_rules(train_df, CONFIG)

        active_positions = []
        pending_orders = []
        last_entry_time = pd.Timestamp("2000-01-01")

        for idx, row in target_df.iterrows():
            curr_time = row['time']
            current_bid = row['close_bid']
            
            if 'low_ask' in row:
                current_low_ask = row['low_ask']
            else:
                spread = row['close_ask'] - row['close_bid']
                current_low_ask = row['low_bid'] + spread

            # Pending Management
            for order in pending_orders[:]:
                if row['high_bid'] >= order['target_price']:
                    pending_orders.remove(order); continue
                if row['time'].hour >= 22:
                    pending_orders.remove(order); continue
                if curr_time < order['active_from']:
                    continue
                
                if current_low_ask <= order['limit_price']:
                    active_positions.append({
                        'entry_price': order['limit_price'],
                        'target_price': order['target_price'],
                        'entry_time': curr_time,
                        'units': order['units']
                    })
                    
                    action_log.append({
                        'time': curr_time,
                        'session_id': target_session_id,
                        'action': 'ENTRY',
                        'price': order['limit_price'],
                        'units': order['units'],
                        'pnl_euro': 0.0,
                        'capital_after': current_capital
                    })
                    pending_orders.remove(order)

            # Exit Check
            for pos in active_positions[:]:
                exit_p = None
                exit_reason = ""
                
                if row['high_bid'] >= pos['target_price']:
                    exit_p = pos['target_price']
                    exit_reason = "TP"
                elif idx == len(target_df) - 1 or row['time'].hour >= 22:
                    exit_p = current_bid
                    exit_reason = "EOD"

                if exit_p:
                    profit_cash = (exit_p - pos['entry_price']) * pos['units']
                    current_capital += profit_cash
                    
                    action_log.append({
                        'time': curr_time,
                        'session_id': target_session_id,
                        'action': f'EXIT_{exit_reason}',
                        'price': exit_p,
                        'units': pos['units'],
                        'pnl_euro': round(profit_cash, 2),
                        'capital_after': round(current_capital, 2)
                    })
                    active_positions.remove(pos)

            # Signal Generation
            time_since = (curr_time - last_entry_time).total_seconds() / 60
            total_committed = len(active_positions) + len(pending_orders)
            entry_ok = (time_since >= CONFIG['cooldown_minutes']) and (total_committed < CONFIG['max_slots'])
            
            state = (row['prev_trend'], row['pos_bin'], row['rsi_bin'], row['quarter_hour'])
            valid_rng = row['range_pct'] > CONFIG['min_range']
            time_ok = row['time'].hour < 20

            if (state in rules) and valid_rng and entry_ok and time_ok:
                ask_at_signal = row['close_ask']
                target_dist = row['sess_range'] * CONFIG['target_range_ratio']
                target_price = ask_at_signal + target_dist
                units = int((current_capital / CONFIG['max_slots'] * CONFIG['leverage']) / ask_at_signal)

                if units >= 1:
                    pending_orders.append({
                        'limit_price': ask_at_signal,
                        'target_price': target_price,
                        'units': units,
                        'created_time': curr_time,
                        'active_from': curr_time + pd.Timedelta(minutes=2)
                    })
                    last_entry_time = curr_time

        capital_history.append(current_capital)

    print(f"Eind Kapitaal: €{current_capital:.2f}")
    
    # ==============================================================================
    # 5. OPSLAAN & VISUALISATIE (AANGEPAST)
    # ==============================================================================
    
    # Logboek opslaan
    if action_log:
        log_df = pd.DataFrame(action_log)
        log_file = os.path.join(OUTPUT_DIR, "trading_log.csv")
        log_df.to_csv(log_file, index=False)

    # Grafiek met DATUMS op de X-as
    # capital_history[1:] omdat index 0 de startwaarde is, en session_dates begint bij sessie 1
    res_df = pd.DataFrame({
        'Datum': session_dates,
        'Kapitaal': capital_history[1:]
    })

    plt.figure(figsize=(12, 6))
    
    # Plot de lijn
    plt.plot(res_df['Datum'], res_df['Kapitaal'], label='Account Saldo (€)', color='blue', linewidth=2)
    plt.axhline(START_CAPITAL, color='red', linestyle='--', label='Start Kapitaal')

    # Opmaak Titels
    plt.title(f'Simulatie Resultaat\nStart: €{START_CAPITAL} | Eind: €{current_capital:.2f}', fontsize=14)
    plt.ylabel('Account Waarde (€)')
    plt.xlabel('Datum') # Label aangepast
    
    # Datum Formattering voor de X-as
    ax = plt.gca()
    # Toon als Dag-Maand (bijv. 12-05)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    # Zet een interval zodat niet elke dag er staat als het te druk wordt
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Roteer de datums zodat ze leesbaar zijn
    plt.gcf().autofmt_xdate()

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_file = os.path.join(OUTPUT_DIR, "equity_curve.png")
    plt.savefig(plot_file)
    print(f"Grafiek opgeslagen in: {plot_file}")
    plt.close()

if __name__ == "__main__":
    run_simulation()
