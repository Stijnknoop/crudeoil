import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

try:
    from new_strategy import get_data_github, fetch_live_data_capital, merge_and_process, BEST_PARAMS
except ImportError:
    print("Fout: new_strategy.py niet gevonden.")
    exit(1)

OUTPUT_DIR = "OIL_CRUDE/Trading_details"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def generate_daily_plots(data):
    W_SIZE = BEST_PARAMS['WINDOW_SIZE']
    E_THRESH = BEST_PARAMS['ENTRY_THRESHOLD']
    TP_R = BEST_PARAMS['TP_RANGE']
    MAX_DROP = BEST_PARAMS['MAX_DROP']
    MIN_OBS = BEST_PARAMS['MIN_OBS']
    MAX_TRADES = BEST_PARAMS['MAX_CONCURRENT_TRADES']
    COOLDOWN = BEST_PARAMS['BATCH_COOLDOWN']
    
    RSI_COL = f"rsi_{BEST_PARAMS['RSI_PERIOD']}"
    TREND_COL = f"trend_{BEST_PARAMS['MA_PERIOD']}"
    
    range_bins = np.linspace(0, 1.0, 6)
    rsi_bins = [0, 30, 70, 100]
    trend_bins = [-np.inf, -0.0005, 0.0005, np.inf]
    vol_bins = [-np.inf, 0.9, 1.2, np.inf]
    
    unique_sessions = sorted(data[data['session_id'] != -1]['session_id'].unique())
    print(f"Start genereren plots...")
    
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
        
        # --- PLOT DAG ---
        dff = data[data['session_id'] == test_sess_id].copy().reset_index(drop=True)
        if len(dff) < 50: continue
        
        # BEREKEN DAG STATS VOOR VISUALISATIE
        dff['day_high'] = dff['mid_price'].cummax()
        dff['day_low'] = dff['mid_price'].cummin()
        dff['day_rng'] = dff['day_high'] - dff['day_low']
        dff['day_rng'] = dff['day_rng'].replace(0, 1e-9)
        
        viz_lines = [] 
        viz_events = [] 
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
            
            # Active Trades
            for k in range(len(active_trades) - 1, -1, -1):
                trade = active_trades[k]
                exit_signal = False; exit_price = 0.0
                
                if p_high_bid >= trade['target_price']:
                    exit_signal = True; exit_price = trade['target_price']
                elif t == len(dff) - 1:
                    exit_signal = True; exit_price = p_bid
                
                if exit_signal:
                    raw_ret = (exit_price - trade['entry_price']) / trade['entry_price']
                    # Dummy Stake (1000/10 = 100) * Leverage (10) = 1000 effectief
                    # PnL = 1000 * %
                    pnl = 1000 * raw_ret 
                    day_pnl += pnl
                    
                    viz_events.append({'t': curr_time, 'type': 'WIN' if pnl > 0 else 'LOSS', 'price': exit_price})
                    viz_lines.append({'type': 'trade', 'pnl': pnl, 't0': trade['entry_time'], 'p0': trade['entry_price'], 't1': curr_time, 'p1': exit_price})
                    active_trades.pop(k)

            # Pending
            for k in range(len(pending_orders) - 1, -1, -1):
                order = pending_orders[k]
                if p_high_bid >= order['target_price']:
                    viz_events.append({'t': curr_time, 'type': 'MISSED', 'price': order['target_price']})
                    pending_orders.pop(k); continue
                if t == len(dff) - 1:
                    pending_orders.pop(k); continue
                
                if t >= order['signal_idx'] + 2:
                    if p_low_ask <= order['limit_price']:
                        active_trades.append({'entry_price': order['limit_price'], 'target_price': order['target_price'], 'entry_time': curr_time})
                        viz_events.append({'t': curr_time, 'type': 'FILL', 'price': order['limit_price']})
                        viz_lines.append({'type': 'pending', 't0': order['signal_time'], 'p0': order['limit_price'], 't1': curr_time, 'p1': order['limit_price']})
                        pending_orders.pop(k)

            # Signal
            if len(active_trades) + len(pending_orders) < MAX_TRADES:
                if t >= last_signal_idx + COOLDOWN:
                    if row['hour'] < 22:
                        rng_pos = (row['mid_price'] - row['day_low']) / row['day_rng']
                        b_r = min(int(rng_pos * 5), 4)
                        
                        val_rsi = row[RSI_COL]
                        b_rs = 0 if val_rsi < 30 else (2 if val_rsi > 70 else 1)
                        val_trd = row[TREND_COL]
                        b_tr = 0 if val_trd < -0.0005 else (2 if val_trd > 0.0005 else 1)
                        val_vol = row['vol_ratio']
                        b_vl = 0 if val_vol < 0.9 else (2 if val_vol > 1.2 else 1)
                        
                        key = (row['hour'], b_r, b_rs, b_tr, b_vl)
                        
                        if key in prob_map:
                            stats = prob_map[key]
                            if stats['hit_mean'] >= E_THRESH and stats['loss_mean'] <= MAX_DROP:
                                limit_pr = p_ask
                                target_pr = limit_pr + (TP_R * row['day_rng'])
                                pending_orders.append({'limit_price': limit_pr, 'target_price': target_pr, 'signal_idx': t, 'signal_time': curr_time})
                                last_signal_idx = t
                                viz_events.append({'t': curr_time, 'type': 'SIGNAL', 'price': limit_pr})

        if viz_lines or viz_events:
            date_str = dff['time'].iloc[0].strftime('%Y-%m-%d')
            # PnL % berekenen o.b.v. dummy startkapitaal (1000)
            pnl_pct = (day_pnl / 1000.0) * 100
            
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(dff['time'], dff['mid_price'], color='black', alpha=0.6, lw=1, label='Koers')
            
            seg_pending = [[(mdates.date2num(l['t0']), l['p0']), (mdates.date2num(l['t1']), l['p1'])] for l in viz_lines if l['type'] == 'pending']
            seg_wins = [[(mdates.date2num(l['t0']), l['p0']), (mdates.date2num(l['t1']), l['p1'])] for l in viz_lines if l['type'] == 'trade' and l['pnl'] > 0]
            seg_loss = [[(mdates.date2num(l['t0']), l['p0']), (mdates.date2num(l['t1']), l['p1'])] for l in viz_lines if l['type'] == 'trade' and l['pnl'] <= 0]
            
            ax.add_collection(LineCollection(seg_pending, colors='gold', linestyles=':', linewidths=2))
            ax.add_collection(LineCollection(seg_wins, colors='lime', linewidths=2))
            ax.add_collection(LineCollection(seg_loss, colors='red', linewidths=2))
            
            for e in viz_events:
                c = 'gold' if e['type'] == 'SIGNAL' else ('purple' if e['type'] == 'MISSED' else ('green' if e['type'] == 'FILL' else ('lime' if e['type'] == 'WIN' else 'red')))
                m = 'o' if e['type'] == 'SIGNAL' else ('x' if e['type'] == 'MISSED' else ('^' if e['type'] == 'FILL' else 'v'))
                ax.scatter(e['t'], e['price'], color=c, marker=m, s=80, zorder=5, edgecolors='k')

            # TITEL MET PNL
            title_color = 'green' if day_pnl > 0 else ('red' if day_pnl < 0 else 'black')
            ax.set_title(f"Sessie {test_sess_id} | {date_str} | PnL: €{day_pnl:.2f} ({pnl_pct:.2f}%)", fontweight='bold', color=title_color)
            
            # LEGENDA
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', label='Signaal'),
                Line2D([0], [0], color='gold', linestyle=':', label='Pending Order'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='green', label='Gevuld'),
                Line2D([0], [0], color='lime', label='Winst Trade'),
                Line2D([0], [0], color='red', label='Verlies Trade'),
                Line2D([0], [0], marker='x', color='w', markerfacecolor='purple', label='Gemist'),
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.savefig(os.path.join(PLOT_DIR, f"daily_plot_{date_str}.png"))
            plt.close()

if __name__ == "__main__":
    df_git = get_data_github()
    df_cap = fetch_live_data_capital()
    df_main = merge_and_process(df_git, df_cap)
    
    if df_main is not None:
        generate_daily_plots(df_main)
