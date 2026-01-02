import http.client
import json
import pandas as pd
import os
from datetime import datetime

# Globale tokens
SECURITY_TOKEN = None
CST = None

def load_credentials():
    return {
        "identifier": os.environ["IDENTIFIER"],
        "password": os.environ["PASSWORD"],
        "api_key": os.environ["X_CAP_API_KEY"]
    }

def tokens():
    global SECURITY_TOKEN, CST
    creds = load_credentials()
    identifier = creds['identifier']
    password = creds['password']
    api_key = creds['api_key']

    conn = http.client.HTTPSConnection("api-capital.backend-capital.com")
    payload = json.dumps({"identifier": identifier, "password": password})
    headers = {"Content-Type": "application/json", "X-CAP-API-KEY": api_key}

    conn.request("POST", "/api/v1/session", payload, headers)
    res = conn.getresponse()
    
    CST = res.getheader("CST")
    SECURITY_TOKEN = res.getheader("X-SECURITY-TOKEN")
    
    # FIX: Verbinding sluiten na de respons om 'Request-sent' fouten te voorkomen
    conn.close() 
    return SECURITY_TOKEN, CST

def data_nu():
    global SECURITY_TOKEN, CST
    creds = load_credentials()
    api_key = creds['api_key']
    
    conn = http.client.HTTPSConnection("api-capital.backend-capital.com")
    payload = ''
    # FIX: Voeg X-CAP-API-KEY ook hier toe aan de headers
    headers = {
        'X-SECURITY-TOKEN': SECURITY_TOKEN, 
        'CST': CST,
        'X-CAP-API-KEY': api_key
    }
    
    epic = "OIL_CRUDE"
    resolution = "MINUTE"
    max_candles = 1000
    url = f"/api/v1/prices/{epic}?resolution={resolution}&max={max_candles}"

    conn.request("GET", url, payload, headers)
    res = conn.getresponse()
    
    raw_data = res.read().decode("utf-8")
    ohlc = json.loads(raw_data)
    
    # FIX: Altijd de verbinding sluiten
    conn.close()
    
    prices = ohlc.get('prices', [])

    df = pd.json_normalize(prices)
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
    tokens()
    df = data_nu()

    # Maak timestamp voor bestandsnaam
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"outputs_{timestamp}.csv"

    df.to_csv(filename, index=False)
    print(f"CSV opgeslagen als {filename}")
