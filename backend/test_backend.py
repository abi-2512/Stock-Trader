import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# A ticker expected to return valid data from Stooq (use a real one)
VALID_TICKER = "AAPL"

# A fake or extremely low-volume ticker to test error handling
INVALID_TICKER = "FAKETICKER123"

def test_predict_success():
    response = client.post("/predict/", json={"ticker": "msft"})  # Will become "aapl.us"
    print("Response:", response.json())
    assert "action" in response.json()
    assert response.status_code == 200
    print("Response:", response.json())  # ADD THIS
    data = response.json()
    assert "q_values" in data
    assert "history" in data
    assert isinstance(data["q_values"], list)
    assert len(data["q_values"]) == 3
    assert isinstance(data["history"], list)

def test_predict_insufficient_data():
    response = client.post("/predict/", json={"ticker": INVALID_TICKER})
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Invalid Ticker or insufficient data."

def test_predict_unknown_ticker():
    response = client.post("/predict/", json={"ticker": "ZZZZZ"})
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Invalid Ticker or insufficient data."
