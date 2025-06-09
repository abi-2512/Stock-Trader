import pandas as pd
import requests
import io
import torch
from model import DQN

def download_stooq(ticker, days):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        r = requests.get(url)
        print(f"URL: {url}, Status: {r.status_code}")
        if r.status_code != 200 or "Date" not in r.text:
            print("Failed to fetch data or missing Date column")
            return pd.DataFrame()
        
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty or "Date" not in df.columns:
            print(f"Invalid DataFrame for ticker: {ticker}")
            return pd.DataFrame()
        
        df["Datetime"] = pd.to_datetime(df["Date"])
        df.set_index("Datetime", inplace=True)
        return df
    except Exception as e:
        print(f"Exception in download_stooq: {e}")
        return pd.DataFrame()



def preprocess(df):
    df = df.dropna()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    x = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
    return x

def load_model():
    model = DQN(input_size=5, hidden_size=64, lstm_layers=1, num_actions=3)
    checkpoint = torch.load('dqn_stock_trader.pth', map_location=torch.device('cpu'))  # or 'cuda'
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_signal(model, ticker):
    df = download_stooq(ticker, days=7)
    if df.empty or len(df) < 20:
        raise ValueError("Invalid Ticker or insufficient data.")

    x = preprocess(df)
    with torch.no_grad():
        q_values = model(x)
        action = torch.argmax(q_values).item()

    hist = df.tail(50)[["Close"]].copy()
    hist["time"] = hist.index.strftime("%m-%d %H:%M")
    hist = hist.reset_index()[["time", "Close"]].rename(columns={"Close": "close"}).to_dict(orient="records")

    return {
        "action": ["Buy", "Sell", "Hold"][action],
        "q_values": q_values.numpy().tolist()[0],
        "history": hist
    }

def run_backtest(model, ticker):
    df = download_stooq(ticker, days=30)
    if df.empty or len(df) < 20:
        return {"error": "Not enough data."}
    
    equity = 10000
    shares = 0
    history = []

    for i in range(20, len(df)):
        window = preprocess(df.iloc[i-20:i])
        with torch.no_grad():
            q_values = model(window)
            action = torch.argmax(q_values).item()

        price = df.iloc[i]["Close"]

        # Buy
        if action == 0 and shares == 0:
            shares = equity // price  # Buy as many as possible
            equity -= shares * price
            entry_price = price

        # Sell
        elif action == 1 and shares > 0:
            equity += shares * price
            shares = 0

        # Equity = current cash + value of shares held
        total_equity = equity + shares * price
        history.append({
            "time": df.index[i].strftime("%m-%d %H:%M"),
            "equity": total_equity
        })

    return {"equity_curve": history, "final_equity": equity + shares * price}
