
# main.py (aangepast om credentials.json te gebruiken)
import http.client
import json
import pandas as pd

# Globale variabelen
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
    identifier = creds['app']['identifier']
    password = creds['app']['password']
    api_key = creds['api']['x_cap_api_key']

    conn = http.client.HTTPSConnection("api-capital.backend-capital.com")

    payload = json.dumps({
        "identifier": identifier,
        "password": password
    })

    headers = {
        "Content-Type": "application/json",
        "X-CAP-API-KEY": api_key
    } 

    conn.request("POST", "/api/v1/session", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print("Login response:", data.decode("utf-8"))

    # Tokens uit headers
    CST = res.getheader("CST")
    SECURITY_TOKEN = res.getheader("X-SECURITY-TOKEN")
    print("CST:", CST)
    print("X-SECURITY-TOKEN:", SECURITY_TOKEN)

    # Test ping
    conn = http.client.HTTPSConnection("api-capital.backend-capital.com")
    headers = {
        "X-SECURITY-TOKEN": SECURITY_TOKEN,
        "CST": CST
    }
    conn.request("GET", "/api/v1/ping", '', headers)
    res = conn.getresponse()
    data = res.read()
    print("Ping response:", data.decode("utf-8"))

    return SECURITY_TOKEN, CST


def data_nu():
    global SECURITY_TOKEN, CST

    conn = http.client.HTTPSConnection("api-capital.backend-capital.com")
    payload = ''
    headers = {
        'X-SECURITY-TOKEN': SECURITY_TOKEN,
        'CST': CST
    }

    epic = "OIL_CRUDE"
    resolution = "MINUTE"
    max_candles = 1000
    url = f"/api/v1/prices/{epic}?resolution={resolution}&max={max_candles}"

    conn.request("GET", url, payload, headers)
    res = conn.getresponse()
    data = res.read()

    ohlc = json.loads(data.decode("utf-8"))
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
    df.to_csv("outputs.csv", index=False)
    print("CSV opgeslagen als outputs.csv")
