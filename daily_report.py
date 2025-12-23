# ===============================
# LEAK-FREE INTRADAY ML BACKTEST
# ===============================

import os
import re
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # required for GitHub Actions
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# ===============================
# CONFIG
# ===============================

OUTPUT_DIR = "Trading_details"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

INITIAL_BALANCE = 10_000.0
RISK_PER_TRADE = 0.02          # 2% per trade
STOP_LOSS_R = -1.0             # -1R
TAKE_PROFIT_R = 1.25           # +1.25R
TRAIL_START_R = 0.6
TRAIL_DISTANCE_R = 0.4

MAX_HOUR = 23                  # stop trading after 23h
LOOKAHEAD = 30                 # minutes

FEATURES = [
    "z_score_30m",
    "rsi",
    "macd",
    "volatility_proxy",
    "day_progression",
    "hour",
    "trend_1h"
]

# ===============================
# DATA INGEST
# ===============================

def load_latest_csv():
    user, repo, branch = "Stijnknoop", "crudeoil", "master"
    api = f"https://api.github.com/repos/{user}/{repo}/contents?ref={branch}"
    r = requests.get(api)
    r.raise_for_status()
    files = r.json()
    csv = next(f for f in files if f["name"].endswith(".csv"))
    return pd.read_csv(csv["download_url"])

def split_trading_days():
    df = load_latest_csv()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    full_idx = pd.date_range(df["time"].min(), df["time"].max(), freq="min")
    df = pd.DataFrame({"time": full_idx}).merge(df, on="time", how="left")

    df["has_data"] = df["open_bid"].notna()
    df["date"] = df["time"].dt.date

    df = df[df.groupby("date")["has_data"].transform("any")]
    df.loc[:, df.columns.difference(["time", "date", "has_data"])] = \
        df.groupby(df["has_data"].cumsum()).ffill()

    # identify session gaps
    gap = (~df["has_data"]) & (df["time"].dt.hour >= 20)
    df["gap_group"] = (gap != gap.shift()).cumsum()

    gap_info = df[gap].groupby("gap_group").size()
    long_gaps = gap_info[gap_info >= 10].index

    df["session"] = 1
    for g in long_gaps:
        end = df[df["gap_group"] == g]["time"].max()
        df.loc[df["time"] > end, "session"] += 1

    days = {}
    for i, (_, d) in enumerate(df.groupby("session"), 1):
        if d["has_data"].sum() > 200:
            days[f"day_{i}"] = d.reset_index(drop=True)

    return days

# ===============================
# FEATURE ENGINEERING (NO LEAKAGE)
# ===============================

def add_features(df):
    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["day_progression"] = (df["hour"] * 60 + df["minute"]) / 1440

    df["volatility_proxy"] = (
        (df["high_bid"] - df["low_bid"])
        .rolling(15)
        .mean()
        / (df["close_bid"] + 1e-9)
    )

    ma = df["close_bid"].rolling(30)
    df["z_score_30m"] = (df["close_bid"] - ma.mean()) / (ma.std() + 1e-9)

    delta = df["close_bid"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    ema12 = df["close_bid"].ewm(span=12).mean()
    ema26 = df["close_bid"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26

    # 1H trend (previous closed candle)
    h1 = df.resample("1h", on="time").agg({"close_bid": "last"}).shift(1)
    df["trend_1h"] = (
        df["close_bid"] - df["time"].dt.floor("1h").map(h1["close_bid"])
    ) / (df["close_bid"] + 1e-9)

    # IMPORTANT: shift everything by 1 bar
    df[FEATURES] = df[FEATURES].shift(1)

    return df.ffill()

# ===============================
# DATASET CREATION
# ===============================

def build_xy(keys, data):
    X, yL, yS = [], [], []

    for k in keys:
        df = add_features(data[k]).dropna()
        p = df["close_bid"].values

        for i in range(len(df) - LOOKAHEAD):
            future = p[i + 1:i + LOOKAHEAD + 1]
            yL.append((future.max() - p[i]) / p[i])
            yS.append((p[i] - future.min()) / p[i])
            X.append(df[FEATURES].iloc[i].values)

    return np.array(X), np.array(yL), np.array(yS)

# ===============================
# BACKTEST
# ===============================

def run_backtest():
    days = split_trading_days()
    keys = sorted(days, key=lambda k: int(re.search(r"\d+", k).group()))

    balance = INITIAL_BALANCE
    logs = []

    for i in range(40, len(keys)):
        train_keys = keys[:i - 10]
        val_keys = keys[i - 10:i]
        test_key = keys[i]

        X, yL, yS = build_xy(train_keys, days)

        model_L = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            random_state=SEED,
            n_jobs=-1
        ).fit(X, yL)

        model_S = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            random_state=SEED,
            n_jobs=-1
        ).fit(X, yS)

        Xv, _, _ = build_xy(val_keys, days)
        thr_L = np.percentile(model_L.predict(Xv), 97)
        thr_S = np.percentile(model_S.predict(Xv), 97)

        df = add_features(days[test_key]).dropna()
        prices = df["close_bid"].values
        hours = df["hour"].values

        preds_L = model_L.predict(df[FEATURES].values)
        preds_S = model_S.predict(df[FEATURES].values)

        active = False
        r_mult = 0.0
        trail = STOP_LOSS_R

        for j in range(len(df) - LOOKAHEAD):
            if not active and hours[j] < MAX_HOUR:
                if preds_L[j] > thr_L:
                    entry = prices[j]
                    side = 1
                    active = True
                elif preds_S[j] > thr_S:
                    entry = prices[j]
                    side = -1
                    active = True

            elif active:
                r_mult = ((prices[j] - entry) / entry) * side / abs(STOP_LOSS_R)

                if r_mult > TRAIL_START_R:
                    trail = max(trail, r_mult - TRAIL_DISTANCE_R)

                if (
                    r_mult >= TAKE_PROFIT_R or
                    r_mult <= trail or
                    hours[j] >= MAX_HOUR
                ):
                    break

        gain = r_mult * RISK_PER_TRADE
        old = balance
        balance *= (1 + gain)

        logs.append({
            "date": str(df["time"].iloc[0].date()),
            "R": r_mult,
            "pct": gain,
            "balance": balance,
            "profit_$": balance - old
        })

    return pd.DataFrame(logs), balance

# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    results, final_balance = run_backtest()
    results.to_csv(f"{OUTPUT_DIR}/trading_log.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(results["balance"])
    plt.title(f"Final balance: ${final_balance:,.2f}")
    plt.savefig(f"{OUTPUT_DIR}/equity_curve.png")
