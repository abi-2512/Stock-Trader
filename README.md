
---

# 📈 LSTM-DQN Stock Trader

A lean, real-time stock trading reinforcement learning system that combines **LSTM-enhanced Deep Q-Networks** with **backtesting**, **signal visualization**, and a live **FastAPI backend**. Data is pulled from **Stooq** for fast experimentation.

---

## 🌐 Live Demo

- **Frontend**: [Frontend](https://stockbroker-z8np.onrender.com)
- **Backend API**: [Backend](https://stocktrader-ym0x.onrender.com)

> ⚠️ May take ~60s on first load due calling the model on CPU (especially backtesting). I highly suggest running it locally.

---

## 🚀 Features

* 🧠 **DQN with LSTM**: Captures temporal price patterns using RNN memory.
* ⏱ **Backtesting + Live Inference**: Simulate trades or predict in real-time.
* 🧾 **Stooq Integration**: Instant stock data download — no API key needed.
* 🖼 **Frontend Signals**: Visualize buy/hold/sell signals and net worth over time.
* ⚡️ **FastAPI Backend**: Clean REST interface for model predictions.

---

## 🧠 Model Architecture

```python
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers, num_actions):
        super(DQN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

* LSTM tracks sequence patterns over a sliding price window
* Final hidden state drives Q-values for buy/hold/sell

---

## 📦 Installation

```bash
clone repo
cd your-repo-name
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---


## 🔌 Run Backend API

```bash
cd backend
uvicorn main:app --reload
```

Endpoints:

* `/predict`: POST market state → model returns action
* `/backtest`: POST historical sequence → list of model actions + rewards
* `/status`: Model health

---

## 🎯 Frontend Usage

The `frontend/` folder contains a simple HTML/JS page:

* Shows recent prices
* Color-coded signals (Buy/Hold/Sell)
* Cumulative net worth over time

Launch it by opening `index.html` or serve via lightweight HTTP server.

---

## 📈 Example Output

```text
Step 0: Action Buy, Reward +66.15, Net Worth: $16,025.54
Step 1: Action Sell, Reward -1.00, Net Worth: $18,430.04
Step 2: Action Hold, Reward +0.89, Net Worth: $16,657.72
```

---

## ✅ Future Upgrades

* [ ] Reward shaping with Sharpe ratio or drawdown
* [ ] Real-time trading via Alpaca or Polygon API
* [ ] Frontend enhancements with Plotly/Chart.js
* [ ] Save/load models from frontend via upload

---

## 🧠 Credit

* Built using: PyTorch, FastAPI, Stooq, OpenAI Gym

---

