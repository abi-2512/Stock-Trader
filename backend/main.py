from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import get_signal, run_backtest, load_model

app = FastAPI()

class TickerRequest(BaseModel):
    ticker: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = load_model()

@app.post("/predict/")
def predict(data: TickerRequest):
    try:
        result = get_signal(agent, data.ticker.upper())
        return result
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Ticker or insufficient data.")

@app.get("/backtest/")
def backtest(ticker: str = "AAPL"):
    result = run_backtest(agent, ticker.upper())
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
      # Add this line

    return result
