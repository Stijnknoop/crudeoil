import os
import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://api-capital.backend-capital.com/api/v1"

def load_credentials():
    return {
        "identifier": os.environ["IDENTIFIER"],
        "password": os.environ["PASSWORD"],
        "api_key": os.environ["X_CAP_API_KEY"]
    }

def create_session():
    creds = load_credentials()

    session = requests.Session()
    session.headers.update({
        "X-CAP-API-KEY": creds["api_key"],
        "Content-Type": "application/json"
    })

    # Login
    payload = {
        "identifier": creds["identifier"],
        "password": creds["password"]
    }

    res = session.post(f"{BASE_URL}/session", json=payload)

    if res.status_code != 200:
        raise RuntimeError(f"Login failed: {res.status_code} - {res.text}")

    # Tokens automatisch toevoegen
    session.headers.update({
        "CST": res.headers["CST"],
        "X-SECURITY-TOKEN": res.headers["X-SECURITY-TOKEN"]
    })

    return session

def fetch_prices(session, epic="OIL_CRUDE", resolution="MINUTE", max_candles=1000):
    url = f"{BASE_URL}/prices/{epic}"
    params = {
        "resolution": resolution,
        "max": max_candles
    }

    res = session.get(url, params=params)

    if res.status_code != 200:
        raise RuntimeError(f"Data error: {res.status_code} - {res.text}")

    data = res.json()["prices"]

    df = pd.json_normalize(data)

    df = df[[
        'snapshotTime',
        'openPrice.bid', 'highPrice.bid', 'lowPrice.bid', 'closePrice.bid',
        'openPrice.ask', 'highPrice.ask', 'lowPrice.ask', 'closePrice.ask',
        'lastTradedVolume'
    ]]

    df.columns = [
        'time',
        'open_bid', 'high_bid', 'low_bid', 'close_bid',
        'open_ask', 'high_ask', 'low_ask', 'close_ask',
        'volume'
    ]

    df['time'] = pd.to_datetime(df['time'])
    return df

if __name__ == "__main__":
    session = create_session()
    df = fetch_prices(session)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"outputs_{timestamp}.csv"

    df.to_csv(filename, index=False)
    print(f"✅ CSV opgeslagen als {filename}")
