import requests
import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

# ==============================================================================
# 1. DATA OPHALEN
# ==============================================================================
def read_latest_csv_from_crudeoil():
    user = "Stijnknoop"
    repo = "crudeoil"
    folder_path = "OIL_CRUDE"
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder_path}?ref=master"
    r = requests.get(api_url, headers=headers)
    if r.status_code != 200:
        raise Exception(r.status_code)
    files = r.json()
    csv_file = next(f for f in files if f["name"].endswith(".csv"))
    return pd.read_csv(csv_file["download_url"])

df_raw = read_latest_csv_from_crudeoil()
df_raw["time"] = pd.to_datetime(df_raw["time"], format="ISO8601")
df_raw = df_raw.sort_values("time")

full_range = pd.date_range(df_raw["time"].min(), df_raw["time"].max(), freq="min")
df = pd.DataFrame({"time": full_range}).merge(df_raw, on="time", how="left")
df["has_data"] = ~df["open_bid"].isna()
df = df.set_index("time")
df[df.columns.difference(["has_data"])] = df[df.columns.difference(["has_data"])].ffill(limit=5)
df = df.reset_index()

df["date"] = df["time"].dt.date
valid_dates = df.groupby("date")["has_data"].any()
df = df[df["date"].isin(valid_dates[valid_dates].index)].copy()

df["gap_flag"] = (~df["has_data"]) & (df["time"].dt.hour >= 20)
df["gap_group"] = (df["gap_flag"] != df["gap_flag"].shift()).cumsum()
gap_groups = df[df["gap_flag"]].groupby("gap_group").agg(
    start_time=("time", "first"),
    length=("time", "count")
)
long_gaps = gap_groups[gap_groups["length"] >= 10]

df["trading_day"] = 1
for _, r in long_gaps.iterrows():
    nxt = df.index[(df["time"] > r["start_time"]) & (df["has_data"])]
    if len(nxt) > 0:
        df.loc[nxt[0]:, "trading_day"] += 1

dag_dict = {
    f"dag_{i}": d[d["has_data"]].reset_index(drop=True)
    for i, (_, d) in enumerate(df.groupby("trading_day"), start=1)
}

# ==============================================================================
# 2. FEATURES
# ==============================================================================
def add_features(df_in):
    df = df_in.copy().sort_values("time")
    df["hour"] = df["time"].dt.hour
    df["day_progression"] = np.clip(
        (df["hour"] * 60 + df["time"].dt.minute) / 1380.0, 0, 1
    )
    close = df["close_bid"]
    df["volatility_proxy"] = (df["high_bid"] - df["low_bid"]).rolling(15).mean() / (close + 1e-9)
    ma30, std30 = close.rolling(30).mean(), close.rolling(30).std()
    df["z_score_30m"] = (close - ma30) / (std30 + 1e-9)
    df["macd"] = close.ewm(span=12).mean() - close.ewm(span=26).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df["prev_close_bid"] = df["close_bid"].shift(1)
    df["prev_close_ask"] = df["close_ask"].shift(1)

    f_cols = [
        "z_score_30m", "rsi", "macd",
        "day_progression", "volatility_proxy", "hour"
    ]
    df[f_cols] = df[f_cols].shift(1)
    return df.dropna()

f_selected = [
    "z_score_30m", "rsi", "macd",
    "day_progression", "volatility_proxy", "hour"
]

HORIZON = 30

def get_xy(keys, d):
    X, yl, ys = [], [], []
    for k in keys:
        df = add_features(d[k])
        if len(df) > HORIZON + 10:
            p = df["close_bid"].values
            X.append(df[f_selected].values[:-HORIZON])
            yl.append([(np.max(p[i+1:i+1+HORIZON]) - p[i]) / p[i] for i in range(len(df)-HORIZON)])
            ys.append([(p[i] - np.min(p[i+1:i+1+HORIZON])) / p[i] for i in range(len(df)-HORIZON)])
    if not X:
        return None, None, None
    return np.vstack(X), np.concatenate(yl), np.concatenate(ys)

def calculate_dynamic_threshold(c):
    if np.isnan(c) or c < 0.01:
        return 99.9
    if c < 0.05:
        return 98.0
    if c < 0.10:
        return 96.0
    return 94.0

# ==============================================================================
# 3. TRADING
# ==============================================================================
output_dir = "OIL_CRUDE/Trading_details"
log_path = os.path.join(output_dir, "trading_logs.csv")
os.makedirs(output_dir, exist_ok=True)

existing = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
processed_days = set(existing["day"].astype(str)) if not existing.empty else set()

sorted_keys = sorted(dag_dict.keys(), key=lambda x: int(re.search(r"\d+", x).group()))
new_records = []

for key in [k for k in sorted_keys if k not in processed_days]:
    idx = sorted_keys.index(key)
    hist = sorted_keys[max(0, idx-40):idx]
    if len(hist) < 20:
        continue

    X_tr, yl_tr, ys_tr = get_xy(hist[:int(len(hist)*0.75)], dag_dict)
    X_val, yl_val, ys_val = get_xy(hist[int(len(hist)*0.75):], dag_dict)
    if X_tr is None:
        continue

    m_l = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, yl_tr)
    m_s = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42).fit(X_tr, ys_tr)

    corr_l = spearmanr(m_l.predict(X_val), yl_val)[0] if X_val is not None else np.nan
    corr_s = spearmanr(m_s.predict(X_val), ys_val)[0] if X_val is not None else np.nan

    t_l = np.percentile(m_l.predict(X_tr), calculate_dynamic_threshold(corr_l))
    t_s = np.percentile(m_s.predict(X_tr), calculate_dynamic_threshold(corr_s))

    df_day = add_features(dag_dict[key]).reset_index(drop=True)
    if df_day.empty:
        continue

    p_l = m_l.predict(df_day[f_selected].values)
    p_s = m_s.predict(df_day[f_selected].values)

    bids = df_day["close_bid"].values
    asks = df_day["close_ask"].values
    prev_bids = df_day["prev_close_bid"].values
    prev_asks = df_day["prev_close_ask"].values
    times = df_day["time"].values
    hours = df_day["hour"].values

    active = False

    for j in range(len(bids) - 1):
        if not active and hours[j] < 23:
            if p_l[j] > t_l:
                side = 1
                ent_p = prev_asks[j]
                entry_time = times[j]
                curr_sl = -0.004
                active = True
            elif p_s[j] > t_s:
                side = -1
                ent_p = prev_bids[j]
                entry_time = times[j]
                curr_sl = -0.004
                active = True

        elif active:
            r = (bids[j] - ent_p) / ent_p if side == 1 else (ent_p - asks[j]) / ent_p
            if r >= 0.0025:
                curr_sl = max(curr_sl, r - 0.002)

            end = hours[j] >= 23 or j == len(bids) - 2 or r >= 0.005 or r <= curr_sl
            if end:
                new_records.append({
                    "day": key,
                    "side": "Long" if side == 1 else "Short",
                    "entry_time": str(entry_time),
                    "entry_p": ent_p,
                    "exit_time": str(times[j]),
                    "exit_p": bids[j] if side == 1 else asks[j],
                    "return": r
                })
                active = False

final = pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
final["entry_time"] = pd.to_datetime(final["entry_time"], errors="coerce")
final = final.sort_values("entry_time")
final.to_csv(log_path, index=False)
