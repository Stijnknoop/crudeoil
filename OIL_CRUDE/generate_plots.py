import requests
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Belangrijk voor GitHub Actions (Headless)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ==============================================================================
# 1. CONFIGURATIE
# ==============================================================================
GITHUB_USER = "Stijnknoop"
GITHUB_REPO = "crudeoil"
FOLDER_PATH = "OIL_CRUDE"
OUTPUT_DIR = "OIL_CRUDE/Trading_details"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Maak mappen aan als ze niet bestaan
os.makedirs(PLOT_DIR, exist_ok=True)

START_CAPITAL = 650.0

CONFIG = {
    'target_range_ratio': 0.5,
    'min_trades': 20,
    'min_roi': 0.0045,
    'min_range': 0.0008,
    'cooldown_minutes': 15,
    'max_slots': 10,
    'leverage': 10
}

# ==============================================================================
# 2. DATA & FEATURE FUNCTIES
# ==============================================================================
def get_data_and_process():
    print("--- Data ophalen... ---")
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

def add_features_quarter(df):
    df = df.copy()
    # RSI
    delta = df['close_bid'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + (avg_gain/avg_loss)))
    
    # Sessie
    sess_stats = df[df['session_id'] != -1].groupby('session_id')['close_bid'].agg(['last'])
    sess_stats['prev_trend'] = np.where((sess_stats['last'] - sess_stats['last'].shift(1)) > 0, 'Groen', 'Rood')
    df = df.merge(sess_stats[['prev_trend']], on='session_id', how='left')
    
    # Intraday
    df['sess_min'] = df.groupby('session_id')['close_bid'].cummin()
    df['sess_max'] = df.groupby('session_id')['close_bid'].cummax()
    df['sess_range'] = df['sess_max'] - df['sess_min']
    
    df['pos_pct'] = ((df['close_bid'] - df['sess_min']) / (df['sess_range'] + 1e-9)).fillna(0.5)
    df['range_pct'] = df['sess_range'] / df['close_bid']
    
    # Bins
    df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutraal', 'Overbought'])
    df['pos_bin'] = pd.cut(df['pos_pct'], bins=[-0.1, 0.3, 0.7, 1.1], labels=['Low', 'Mid', 'High'])
    
    # KWARTIEREN
    df['quarter_hour'] = (df['time'].dt.hour * 4) + (df['time'].dt.minute // 15)
    return df

def train_rules_quarter(train_df, cfg):
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
# 3. GENERATOR LOGICA
# ==============================================================================
def generate_daily_plot():
    # 1. Data laden
    df_raw = get_data_and_process()
    if df_raw is None: return

    valid_sessions = sorted(df_raw[df_raw['session_id'] != -1]['session_id'].unique())
    proc_df = add_features_quarter(df_raw)

    # 2. Pak de LAATSTE sessie (vandaag)
    target_id = valid_sessions[-1]
    
    # Train op alles DAARVOOR
    train_df = proc_df[proc_df['session_id'] < target_id]
    day_df = proc_df[proc_df['session_id'] == target_id].copy().reset_index(drop=True)
    
    if day_df.empty:
        print("Geen data voor vandaag.")
        return

    date_str = day_df['time'].iloc[0].strftime('%Y-%m-%d')
    print(f"Genereren plot voor sessie: {target_id} op datum {date_str}")

    rules = train_rules_quarter(train_df, CONFIG)

    # 3. Simulatie Loop (Precies zoals in notebook) om events te vangen
    active_positions = []
    pending_orders = []
    last_entry_time = pd.Timestamp("2000-01-01")
    cash_capital = START_CAPITAL
    equity_curve = []
    viz_events = []
    viz_lines = [] # Opslag voor de lijnen

    for idx, row in day_df.iterrows():
        curr_time = row['time']
        bc = row['close_bid']
        ac = row['close_ask']
        spread = ac - bc
        low_ask_calc = row['low_bid'] + spread

        # --- Check Pending ---
        for order in pending_orders[:]:
            # Missed
            if row['high_bid'] >= order['target_price']:
                pending_orders.remove(order)
                viz_events.append({'t': curr_time, 'type': 'MISSED', 'price': order['limit_price']})
                viz_lines.append({'t0': order['created_time'], 'p0': order['limit_price'], 't1': curr_time, 'p1': order['limit_price'], 'type': 'pending'})
                continue
            # EOD
            if row['time'].hour >= 22:
                pending_orders.remove(order)
                viz_lines.append({'t0': order['created_time'], 'p0': order['limit_price'], 't1': curr_time, 'p1': order['limit_price'], 'type': 'pending'})
                continue
            # Time check
            if curr_time < order['active_from']: continue
            # Fill
            if low_ask_calc <= order['limit_price']:
                active_positions.append({
                    'entry_price': order['limit_price'], 'target_price': order['target_price'],
                    'entry_time': curr_time, 'units': order['units']
                })
                viz_events.append({'t': curr_time, 'type': 'FILL', 'price': order['limit_price']})
                viz_lines.append({'t0': order['created_time'], 'p0': order['limit_price'], 't1': curr_time, 'p1': order['limit_price'], 'type': 'pending'})
                pending_orders.remove(order)

        # --- Check Exits ---
        for pos in active_positions[:]:
            exit_p = None
            if row['high_bid'] >= pos['target_price']:
                exit_p = pos['target_price']; viz_type = 'WIN'
            elif idx == len(day_df) - 1 or row['time'].hour >= 22:
                exit_p = bc; viz_type = 'TIME'

            if exit_p:
                pnl = (exit_p - pos['entry_price']) * pos['units']
                cash_capital += pnl
                active_positions.remove(pos)
                viz_events.append({'t': curr_time, 'type': viz_type, 'price': exit_p})
                viz_lines.append({'t0': pos['entry_time'], 'p0': pos['entry_price'], 't1': curr_time, 'p1': exit_p, 'type': 'trade', 'pnl': pnl})

        # --- Signals ---
        time_since = (curr_time - last_entry_time).total_seconds() / 60
        total_committed = len(active_positions) + len(pending_orders)
        entry_ok = (time_since >= CONFIG['cooldown_minutes']) and (total_committed < CONFIG['max_slots'])
        state = (row['prev_trend'], row['pos_bin'], row['rsi_bin'], row['quarter_hour'])
        
        if (state in rules) and (row['range_pct'] > CONFIG['min_range']) and entry_ok and (row['time'].hour < 20):
            limit_price = ac
            target_price = limit_price + (row['sess_range'] * CONFIG['target_range_ratio'])
            units = int((START_CAPITAL / CONFIG['max_slots'] * CONFIG['leverage']) / limit_price)
            if units >= 1:
                pending_orders.append({
                    'limit_price': limit_price, 'target_price': target_price,
                    'units': units, 'created_time': curr_time, 'active_from': curr_time + pd.Timedelta(minutes=2)
                })
                last_entry_time = curr_time
                viz_events.append({'t': curr_time, 'type': 'SIGNAL', 'price': limit_price})

        # Equity Tracking
        floating_pnl = sum([(bc - pos['entry_price']) * pos['units'] for pos in active_positions])
        equity_curve.append(cash_capital + floating_pnl)

    # ==============================================================================
    # 4. PLOTTING (STATIC)
    # ==============================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # Achtergrond Data
    ax1.plot(day_df['time'], day_df['close_bid'], color='black', lw=1, alpha=0.6, label='Prijs (Bid)')
    ax2.plot(day_df['time'], equity_curve, color='blue', lw=2, label='Equity')

    # Collecties maken voor performance
    seg_pending = []
    seg_wins = []
    seg_losses = []
    
    # Scatter lijsten
    sig_x, sig_y = [], []
    mis_x, mis_y = [], []
    fil_x, fil_y = [], []
    win_x, win_y = [], []
    los_x, los_y = [], []

    # Lijnen verwerken
    for line in viz_lines:
        t0, p0 = mdates.date2num(line['t0']), line['p0']
        t1, p1 = mdates.date2num(line['t1']), line['p1']
        
        if line['type'] == 'pending':
            seg_pending.append([(t0, p0), (t1, p1)])
        elif line['type'] == 'trade':
            segment = [(t0, p0), (t1, p1)]
            if line['pnl'] > 0: seg_wins.append(segment)
            else: seg_losses.append(segment)

    # Events verwerken
    for e in viz_events:
        if e['type'] == 'SIGNAL': sig_x.append(e['t']); sig_y.append(e['price'])
        elif e['type'] == 'MISSED': mis_x.append(e['t']); mis_y.append(e['price'])
        elif e['type'] == 'FILL': fil_x.append(e['t']); fil_y.append(e['price'])
        elif e['type'] == 'WIN': win_x.append(e['t']); win_y.append(e['price'])
        elif e['type'] == 'TIME': los_x.append(e['t']); los_y.append(e['price'])

    # Toevoegen aan plot
    # 1. Lijnen
    ax1.add_collection(LineCollection(seg_pending, colors='gold', linestyles=':', linewidths=1.5, alpha=0.8))
    ax1.add_collection(LineCollection(seg_wins, colors='lime', linewidths=2.5, alpha=1.0))
    ax1.add_collection(LineCollection(seg_losses, colors='red', linewidths=2.5, alpha=1.0))

    # 2. Markers
    ax1.scatter(sig_x, sig_y, color='gold', s=80, marker='o', zorder=5, label='Signaal')
    ax1.scatter(mis_x, mis_y, color='purple', s=100, marker='s', zorder=6, label='Target Hit (Missed)')
    ax1.scatter(fil_x, fil_y, color='green', s=120, marker='^', zorder=10, edgecolors='k', label='Buy Filled')
    ax1.scatter(win_x, win_y, color='lime', s=120, marker='v', zorder=10, edgecolors='k', label='Winst Trade')
    ax1.scatter(los_x, los_y, color='red', s=120, marker='v', zorder=10, edgecolors='k', label='Verlies Trade')

    # Opmaak
    final_eq = equity_curve[-1]
    color_eq = 'green' if final_eq >= START_CAPITAL else 'red'
    
    ax1.set_title(f"Trading Dag: {date_str} (Kwartier Strategie)", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Prijs')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    ax2.set_ylabel('Equity (€)')
    ax2.axhline(START_CAPITAL, color='r', ls='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.85, f"Equity: €{final_eq:.2f}", transform=ax2.transAxes, fontsize=12, fontweight='bold', color=color_eq)

    # Legenda (handmatig bouwen voor controle)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=8, label='Signaal'),
        Line2D([0], [0], color='gold', linestyle=':', lw=2, label='Pending Periode'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=8, label='Target Hit (Missed)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markeredgecolor='k', markersize=8, label='Buy Filled'),
        Line2D([0], [0], color='lime', lw=2, label='Winst Trade'),
        Line2D([0], [0], color='red', lw=2, label='Verlies Trade'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    # Opslaan
    filename = f"{date_str}_daily_plot.png"
    save_path = os.path.join(PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ Plot opgeslagen: {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_daily_plot()
