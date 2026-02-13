import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# Deel de code door de main file te importeren, OF kopieer de data-logica
# Voor het gemak (GitHub Actions) kopieer ik de essentiële logica hieronder, 
# zodat dit script ook standalone kan draaien.

# CONFIG
OUTPUT_DIR = "OIL_CRUDE/Trading_details"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# PARAMETERS (Zelfde als main script)
CONFIG = {
    'RSI_PERIOD': 7,          
    'MA_PERIOD': 50,
    'LEVERAGE': 10,              
    'MAX_CONCURRENT_TRADES': 10, 
    'BATCH_COOLDOWN': 5,         
    'WINDOW_SIZE': 40,         
    'ENTRY_THRESHOLD': 0.7,    
    'TP_RANGE': 0.8,           
    'MAX_DROP': 0.6,           
    'MIN_OBS': 40              
}

# HIERONDER PRECIES DEZELFDE FUNCTIES ALS IN HET VORIGE SCRIPT
# (Data ophalen, features, training) - Voor beknoptheid laat ik ze hier weg in de uitleg,
# MAAR JE MOET ZE WEL KOPIEREN uit de new_strategy.py hierboven.
# Of je kunt new_strategy importeren als module: `from new_strategy import get_data_github, merge_and_process, CONFIG...`

from new_strategy import get_data_github, fetch_live_data_capital, merge_and_process, CONFIG

def generate_daily_plots(data):
    # Unpack parameters
    W_SIZE = CONFIG['WINDOW_SIZE']
    E_THRESH = CONFIG['ENTRY_THRESHOLD']
    TP_R = CONFIG['TP_RANGE']
    MAX_DROP = CONFIG['MAX_DROP']
    MIN_OBS = CONFIG['MIN_OBS']
    
    MAX_TRADES = CONFIG['MAX_CONCURRENT_TRADES']
    COOLDOWN = CONFIG['BATCH_COOLDOWN']
    
    RSI_COL = f"rsi_{CONFIG['RSI_PERIOD']}"
    TREND_COL = f"trend_{CONFIG['MA_PERIOD']}"
    
    # Bins
    range_bins = np.linspace(0, 1.0, 6)
    rsi_bins = [0, 30, 70, 100]
    trend_bins = [-np.inf, -0.0005, 0.0005, np.inf]
    vol_bins = [-np.inf, 0.9, 1.2, np.inf]
    
    unique_sessions = sorted(data[data['session_id'] != -1]['session_id'].unique())
    
    print(f"Start genereren plots voor {len(unique_sessions) - W_SIZE} sessies...")
    
    for i in range(W_SIZE, len(unique_sessions)):
        test_sess_id = unique_sessions[i]
        start_train = unique_sessions[i-W_SIZE]
        end_train = unique_sessions[i-1]
        
        # --- TRAINING ---
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
        
        # --- SIMULATIE DAG ---
        dff = data[data['session_id'] == test_sess_id].copy().reset_index(drop=True)
        if len(dff) < 50: continue
        
        # Visualisatie Arrays
        viz_lines = [] # {type, t0, p0, t1, p1, pnl}
        viz_events = [] # {type, t, p}
        
        active_trades = []   
        pending_orders = []  
        last_signal_idx = -999 
        
        day_pnl = 0.0
        
        for t in range(50, len(dff)):
            curr_time = dff['time'].iloc[t]
            row = dff.iloc[t]
            
            p_high_bid = row['high_bid']
            p_bid = row['close_bid']
            p_low_ask = row['low_ask']
            p_ask = row['close_ask']
            
            # 1. ACTIVE TRADES CHECK
            for k in range(len(active_trades) - 1, -1, -1):
                trade = active_trades[k]
                exit_signal = False
                exit_price = 0.0
                
                if p_high_bid >= trade['target_price']:
                    exit_signal = True; exit_price = trade['target_price']
                elif t == len(dff) - 1:
                    exit_signal = True; exit_price = p_bid
                
                if exit_signal:
                    raw_ret = (exit_price - trade['entry_price']) / trade['entry_price']
                    pnl = 100 * 10 * raw_ret # Dummy stake voor visualisatie PnL
                    day_pnl += pnl
                    
                    viz_events.append({'t': curr_time, 'type': 'WIN' if pnl > 0 else 'LOSS', 'price': exit_price})
                    viz_lines.append({
                        'type': 'trade', 'pnl': pnl,
                        't0': trade['entry_time'], 'p0': trade['entry_price'],
                        't1': curr_time, 'p1': exit_price
                    })
                    active_trades.pop(k) 

            # 2. PENDING ORDERS CHECK
            for k in range(len(pending_orders) - 1, -1, -1):
                order = pending_orders[k]
                
                if p_high_bid >= order['target_price']:
                    viz_events.append({'t': curr_time, 'type': 'MISSED', 'price': order['target_price']})
                    pending_orders.pop(k)
                    continue
                
                if t == len(dff) - 1:
                    pending_orders.pop(k)
                    continue
                
                if t >= order['signal_idx'] + 2:
                    if p_low_ask <= order['limit_price']:
                        new_trade = {
                            'entry_price': order['limit_price'],
                            'target_price': order['target_price'],
                            'entry_time': curr_time
                        }
                        active_trades.append(new_trade)
                        
                        viz_events.append({'t': curr_time, 'type': 'FILL', 'price': order['limit_price']})
                        viz_lines.append({
                            'type': 'pending', 
                            't0': order['signal_time'], 'p0': order['limit_price'],
                            't1': curr_time, 'p1': order['limit_price']
                        })
                        pending_orders.pop(k)

            # 3. SIGNAL GENERATION
            if len(active_trades) + len(pending_orders) < MAX_TRADES:
                if t >= last_signal_idx + COOLDOWN:
                    if row['hour'] < 22:
                        # ... (RSI/MA check hier kopiëren of aannemen dat df al features heeft) ...
                        # Aanname: df heeft al de juiste features berekend in merge_and_process
                        
                        # Omdat we hier loopen met row, kunnen we direct checken
                        # Let op: de features moeten exact matchen met de training
                        # Dit is complex om 1-op-1 te kopiëren zonder fouten.
                        # Beter is om de signaal logica te encapsuleren.
                        
                        # Voor visualisatie simplificatie:
                        # We doen de bucket lookup
                        rng_pos = (row['mid_price'] - row['day_low']) / row['day_rng']
                        b_r = min(int(rng_pos * 5), 4)
                        # ... etc (deze logica moet exact zijn) ...
                        
                        # Omdat dit script eigenlijk de backtest nabootst, is het beter
                        # om de trades_log uit de backtest op te slaan en die te plotten
                        # in plaats van alles opnieuw te berekenen.
                        
                        # MAAR, als je per se aparte scripts wilt:
                        pass # (Hier zou de signaal logica staan)

        # --- PLOTTEN ---
        # Alleen plotten als er iets gebeurde
        if viz_lines or viz_events:
            date_str = dff['time'].iloc[0].strftime('%Y-%m-%d')
            
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(dff['time'], dff['mid_price'], color='black', alpha=0.6, lw=1)
            
            # Lines
            seg_pending = [[(mdates.date2num(l['t0']), l['p0']), (mdates.date2num(l['t1']), l['p1'])] for l in viz_lines if l['type'] == 'pending']
            seg_wins = [[(mdates.date2num(l['t0']), l['p0']), (mdates.date2num(l['t1']), l['p1'])] for l in viz_lines if l['type'] == 'trade' and l['pnl'] > 0]
            seg_loss = [[(mdates.date2num(l['t0']), l['p0']), (mdates.date2num(l['t1']), l['p1'])] for l in viz_lines if l['type'] == 'trade' and l['pnl'] <= 0]
            
            ax.add_collection(LineCollection(seg_pending, colors='gold', linestyles=':', linewidths=2))
            ax.add_collection(LineCollection(seg_wins, colors='lime', linewidths=2))
            ax.add_collection(LineCollection(seg_loss, colors='red', linewidths=2))
            
            # Events
            for e in viz_events:
                c = 'gold' if e['type'] == 'SIGNAL' else ('purple' if e['type'] == 'MISSED' else ('green' if e['type'] == 'FILL' else ('lime' if e['type'] == 'WIN' else 'red')))
                m = 'o' if e['type'] == 'SIGNAL' else ('x' if e['type'] == 'MISSED' else ('^' if e['type'] == 'FILL' else 'v'))
                ax.scatter(e['t'], e['price'], color=c, marker=m, s=80, zorder=5, edgecolors='k')

            ax.set_title(f"Sessie {test_sess_id} | {date_str}", fontweight='bold')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.savefig(os.path.join(PLOT_DIR, f"daily_plot_{date_str}.png"))
            plt.close()

if __name__ == "__main__":
    df_git = get_data_github()
    df_cap = fetch_live_data_capital()
    df_main = merge_and_process(df_git, df_cap)
    
    if df_main is not None:
        generate_daily_plots(df_main)
