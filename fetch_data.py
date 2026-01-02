import http.client
import json
import pandas as pd
import os
import argparse
import sys
from datetime import datetime

# --- Argument Parser toegevoegd ---
parser = argparse.ArgumentParser(description='Fetch market data.')
parser.add_argument('--epic', type=str, required=True, help='The EPIC symbol (e.g., OIL_CRUDE or NATURALGAS)')
args = parser.parse_args()

EPIC_SYMBOL = args.epic
OUTPUT_FOLDER = EPIC_SYMBOL  # Mapnaam is gelijk aan de EPIC

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
    
    conn = http.client.HTTPSConnection("api-capital.backend-capital.com")
    payload = json.dumps({"identifier": creds['identifier'], "password": creds['password']})
    headers = {"Content-Type": "application/json", "X-CAP-API-KEY": creds['api_key']}

    conn.request("POST", "/api/v1/session", payload, headers)
    res = conn.getresponse()
    CST = res.getheader("CST")
    SECURITY_TOKEN = res.getheader("X-SECURITY-TOKEN")
    return SECURITY_TOKEN, CST

def data_nu():
    global SECURITY_TOKEN, CST
    conn = http.client.HTTPSConnection("api-capital.backend-capital.com")
    payload = ''
    headers = {'X-SECURITY-TOKEN': SECURITY_TOKEN, 'CST': CST}
    
    resolution = "MINUTE"
    max_candles = 1000
    # Gebruik de variabele EPIC_SYMBOL hier
    url = f"/api/v1/prices/{EPIC_SYMBOL}?resolution={resolution}&max={max_candles}"

    conn.request("GET", url, payload, headers)
    res = conn.getresponse()
    
    if res.status != 200:
        print(f"❌ Fout bij ophalen data voor {EPIC_SYMBOL}: {res.status} {res.reason}")
        sys.exit(1)

    ohlc = json.loads(res.read().decode("utf-8"))
    prices = ohlc.get('prices', [])

    df = pd.json_normalize(prices)
    if df.empty:
        print(f"⚠️ Geen data ontvangen voor {EPIC_SYMBOL}")
        return pd.DataFrame()

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
    print(f"🚀 Starten voor: {EPIC_SYMBOL}")
    
    # Maak de map aan als deze niet bestaat
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📂 Map aangemaakt: {OUTPUT_FOLDER}")

    tokens()
    df = data_nu()

    if not df.empty:
        # Opslaan IN de map
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(OUTPUT_FOLDER, f"outputs_{timestamp}.csv")

        df.to_csv(filename, index=False)
        print(f"✅ CSV opgeslagen: {filename}")
