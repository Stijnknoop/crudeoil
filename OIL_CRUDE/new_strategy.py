import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import pytz # Zorg dat je 'pip install pytz' hebt gedaan

# --- CONFIGURATIE ---
OUTPUT_DIR = "OIL_CRUDE/Trading_details"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# HARDE GRENZEN (Nederlandse Tijd)
EXIT_HOUR_NL = 23  # Om 23:00 uur (of daarna) wordt alles gesloten
START_HOUR_NL = 0  # Sessie begint officieel om 00:00

try:
    from new_strategy import get_data_github, fetch_live_data_capital, merge_and_process, BEST_PARAMS
except ImportError:
    print("Fout: new_strategy.py niet gevonden.")
    exit(1)

def fix_data_structure(df):
    """
    Converteert alles naar Europe/Amsterdam tijd.
    Bepaalt sessies op basis van de Nederlandse datum.
    """
    print("Bezig met data conversie naar Europe/Amsterdam...")
    
    # 1. Zorg dat input datetime is
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # 2. Zorg dat input UTC is (Capital/Github data is standaard UTC)
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')
    else:
        df['time'] = df['time'].dt.tz_convert('UTC')

    # 3. Maak 'nl_time' kolom (Leidend voor jouw strategie!)
    # Dit regelt automatisch het verschil tussen zomer- (+2) en wintertijd (+1)
    amsterdam_tz = pytz.timezone('Europe/Amsterdam')
    df['nl_time'] = df['time'].dt.tz_convert(amsterdam_tz)
    
    # 4. Sessie ID bepalen op basis van de NEDERLANDSE datum
    # Alles wat op dezelfde dag in NL valt (00:00 - 23:59), hoort bij elkaar.
    df['nl_date'] = df['nl_time'].dt.date
    
    unique_dates = sorted(df['nl_date'].unique())
    date_map = {d: i for i, d in enumerate(unique_dates)}
    df['session_id'] = df['nl_date'].map(date_map)
    
    # 5. Update hulpmiddelen (voor statistieken gebruiken we nu NL uren)
    df['hour'] = df['nl_time'].dt.hour
    
    print(f"Data gefixt: {len(unique_dates)} sessies (NL Tijd).")
    return df

def generate_daily_plots(data):
    # Parameters unpacken
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
    
    unique_sessions = sorted(data['session_id'].unique())
    
    # --- LOOP DOOR SESSIES ---
    for i in range(W_SIZE, len(unique_sessions)):
        test_sess_id = unique_sessions[i]
        start_train = unique_sessions[i-W_SIZE]
        end_train = unique_sessions[i-1]
        
        # --- TRAINING (Op NL data) ---
        mask = (data['session_id'] >= start_train) & (data['session_id'] <= end_train)
        df_h = data[mask].copy()
        
        # Filter: We trainen alleen op data binnen de handelsuren (00:00 - 23:00)
        # Dit voorkomt dat rare data 's nachts je stats verpest
        df_h = df_h[(df_h['hour'] >= START_HOUR_NL) & (df_h['hour'] <= EXIT_HOUR_NL)]

        if len(df_h) == 0: continue

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
        
        # Stats berekenen op basis van NL uren
        stats = df_h.groupby(['hour', 'b_rng', 'b_rsi', 'b_trd', 'b_vol'])[['hit', 'loss']].agg(['mean', 'count'])
        valid_stats = stats[stats[('hit', 'count')] >= MIN_OBS]
        valid_stats.columns = ['_'.join(col) for col in valid_stats.columns] 
        prob_map = valid_stats[['hit_mean', 'loss_mean']].to_dict('index')
        
        # --- SIMULATIE DAG ---
        dff = data[data['session_id'] == test_sess_id].copy().reset_index(drop=True)
        if len(dff) < 10: continue
        
        dff['day_high'] = dff['mid_price'].cummax()
        dff['day_low'] = dff['mid_price'].cummin()
        dff['day_rng'] = dff['day_high'] - dff['day_low']
        dff['day_rng'] = dff['day_rng'].replace(0, 1e-9)
        
        viz_lines = []; viz_events = []; active_trades = []; pending_orders = []
        last_signal_idx = -999; day_pnl = 0.0
        
        for t in range(50, len(dff)):
            row = dff.iloc[t]
            curr_nl_time = row['nl_time'] # Gebruik NL tijd voor logica
            
            # --- HARDE EXIT REGEL ---
            # Als het 23:00 of later is in NL, sluiten we alles.
            # Of als dit de allerlaatste candle van de dataset is.
            is_closing_time = (curr_nl_time.hour >= EXIT_HOUR_NL)
            is_end_of_data = (t == len(dff) - 1)
            force_exit = is_closing_time or is_end_of_data
            
            p_high_bid = row['high_bid']; p_bid = row['close_bid']
            p_low_ask = row['low_ask']; p_ask = row['close_ask']
            
            # 1. Active Trades Manager
            for k in range(len(active_trades) - 1, -1, -1):
                trade = active_trades[k]
                exit_signal = False; exit_price = 0.0
                
                # Take Profit hit?
                if p_high_bid >= trade['target_price']:
                    exit_signal = True; exit_price = trade['target_price']
                    type_exit = 'TP'
                
                # Hard Time Exit (23:00 NL)
                elif force_exit:
                    exit_signal = True; exit_price = p_bid
                    type_exit = 'TIME'
                
                if exit_signal:
                    raw_ret = (exit_price - trade['entry_price']) / trade['entry_price']
                    pnl = 1000 * raw_ret 
                    day_pnl += pnl
                    
                    c_type = 'WIN' if pnl > 0 else 'LOSS'
                    viz_events.append({'t': curr_nl_time, 'type': c_type, 'price': exit_price})
                    viz_lines.append({'type': 'trade', 'pnl': pnl, 't0': trade['entry_time'], 'p0': trade['entry_price'], 't1': curr_nl_time, 'p1': exit_price})
                    active_trades.pop(k)

            # 2. Pending Orders Manager
            for k in range(len(pending_orders) - 1, -1, -1):
                order = pending_orders[k]
                
                # Als prijs al boven target is -> Gemist
                if p_high_bid >= order['target_price']:
                    viz_events.append({'t': curr_nl_time, 'type': 'MISSED', 'price': order['target_price']})
                    pending_orders.pop(k); continue
                
                # Als het 23:00 is -> Cancel pending
                if force_exit:
                    pending_orders.pop(k); continue
                
                # Check fill
                if t >= order['signal_idx'] + 2:
                    if p_low_ask <= order['limit_price']:
                        active_trades.append({
                            'entry_price': order['limit_price'], 
                            'target_price': order['target_price'], 
                            'entry_time': curr_nl_time
                        })
                        viz_events.append({'t': curr_nl_time, 'type': 'FILL', 'price': order['limit_price']})
                        viz_lines.append({'type': 'pending', 't0': order['signal_time'], 'p0': order['limit_price'], 't1': curr_nl_time, 'p1': order['limit_price']})
                        pending_orders.pop(k)

            # 3. Nieuwe Signalen (Alleen VÓÓR 23:00)
            if not force_exit and (len(active_trades) + len(pending_orders) < MAX_TRADES):
                if t >= last_signal_idx + COOLDOWN:
                    # Feature calculation
                    rng_pos = (row['mid_price'] - row['day_low']) / row['day_rng']
                    b_r = min(int(rng_pos * 5), 4)
                    
                    val_rsi = row[RSI_COL]
                    b_rs = 0 if val_rsi < 30 else (2 if val_rsi > 70 else 1)
                    val_trd = row[TREND_COL]
                    b_tr = 0 if val_trd < -0.0005 else (2 if val_trd > 0.0005 else 1)
                    val_vol = row['vol_ratio']
                    b_vl = 0 if val_vol < 0.9 else (2 if val_vol > 1.2 else 1)
                    
                    # Key op basis van NL uren
                    key = (curr_nl_time.hour, b_r, b_rs, b_tr, b_vl)
                    
                    if key in prob_map:
                        stats = prob_map[key]
                        if stats['hit_mean'] >= E_THRESH and stats['loss_mean'] <= MAX_DROP:
                            limit_pr = p_ask
                            target_pr = limit_pr + (TP_R * row['day_rng'])
                            pending_orders.append({
                                'limit_price': limit_pr, 
                                'target_price': target_pr, 
                                'signal_idx': t, 
                                'signal_time': curr_nl_time
                            })
                            last_signal_idx = t
                            viz_events.append({'t': curr_nl_time, 'type': 'SIGNAL', 'price': limit_pr})

        # --- PLOTTEN ---
        if viz_lines or viz_events:
            date_str = dff['nl_date'].iloc[0].strftime('%Y-%m-%d')
            pnl_pct = (day_pnl / 1000.0) * 100
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # X-as in NL tijd
            time_nums = mdates.date2num(dff['nl_time'])
            ax.plot(time_nums, dff['mid_price'], color='black', alpha=0.6, lw=1, label='Koers (NL Tijd)')
            
            # Helper voor lijntjes tekenen
            def make_segs(lines_list, filter_type, filter_pnl=None):
                segs = []
                for l in lines_list:
                    if l['type'] != filter_type: continue
                    if filter_pnl == 'pos' and l['pnl'] <= 0: continue
                    if filter_pnl == 'neg' and l['pnl'] > 0: continue
                    segs.append([(mdates.date2num(l['t0']), l['p0']), (mdates.date2num(l['t1']), l['p1'])])
                return segs

            ax.add_collection(LineCollection(make_segs(viz_lines, 'pending'), colors='gold', linestyles=':', linewidths=2))
            ax.add_collection(LineCollection(make_segs(viz_lines, 'trade', 'pos'), colors='lime', linewidths=2))
            ax.add_collection(LineCollection(make_segs(viz_lines, 'trade', 'neg'), colors='red', linewidths=2))
            
            for e in viz_events:
                c = 'gold' if e['type'] == 'SIGNAL' else ('purple' if e['type'] == 'MISSED' else ('green' if e['type'] == 'FILL' else ('lime' if e['type'] == 'WIN' else 'red')))
                m = 'o' if e['type'] == 'SIGNAL' else ('x' if e['type'] == 'MISSED' else ('^' if e['type'] == 'FILL' else 'v'))
                ax.scatter(mdates.date2num(e['t']), e['price'], color=c, marker=m, s=80, zorder=5, edgecolors='k')

            title_color = 'green' if day_pnl > 0 else ('red' if day_pnl < 0 else 'black')
            ax.set_title(f"Sessie {test_sess_id} | {date_str} (NL) | PnL: €{day_pnl:.2f} ({pnl_pct:.2f}%)", fontweight='bold', color=title_color)
            
            # X-as formatteren naar NL tijdzones
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Europe/Amsterdam')))
            ax.set_xlabel("Tijd (Europe/Amsterdam)")
            
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', label='Signaal'),
                Line2D([0], [0], color='gold', linestyle=':', label='Pending'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='green', label='Gevuld'),
                Line2D([0], [0], color='lime', label='Winst'),
                Line2D([0], [0], color='red', label='Verlies'),
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"daily_plot_{date_str}.png"))
            plt.close()

if __name__ == "__main__":
    df_git = get_data_github()
    df_cap = fetch_live_data_capital()
    df_main = merge_and_process(df_git, df_cap)
    
    if df_main is not None:
        df_fixed = fix_data_structure(df_main)
        generate_daily_plots(df_fixed)
