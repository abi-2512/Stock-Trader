import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym
from gym import spaces
import multiprocessing as mp
import os
import time
from collections import Counter

torch.backends.mps.is_available()


# 1. Stock Trading Environment
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, window_size=50):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.balance = initial_balance
        self.shares_held = 0
        self.current_price = 0
        self.net_worth = initial_balance
        self.max_steps = len(df) - 1
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, df.shape[1]),
            dtype=np.float32
        )

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.window_size
        self.current_price = self.df.iloc[self.current_step]['Close']
        self.average_buy_price = 0
        self.buy_price = 0
        return self._get_observation()

    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, {}

        prev_price = self.df.iloc[self.current_step]['Close']
        current_price = self.df.iloc[self.current_step + 1]['Close']  # next price
        done = False
        reward = 0
        penalty = 0

        prev_net_worth = self.net_worth

        # Hold (0)
        if action == 0:
            if self.shares_held == 0:
                penalty = -1  # strong penalty for holding without a position
            else:
                # Reward holding through gain
                reward += max((current_price - prev_price) * self.shares_held * 0.05, -0.1)

        # Buy (1)
        elif action == 1:
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
                if self.shares_held == 1:
                    self.average_buy_price = current_price
                else:
                    self.average_buy_price = (
                        (self.average_buy_price * (self.shares_held - 1)) + current_price
                    ) / self.shares_held
            else:
                penalty = -1  # tried to buy without funds

        # Sell (2)
        elif action == 2:
            if self.shares_held > 0:
                profit = (current_price - self.average_buy_price) * self.shares_held
                reward += profit * 0.5  # stronger reward signal
                self.balance += self.shares_held * current_price
                self.shares_held = 0
                self.average_buy_price = 0
            else:
                penalty = -1  # tried to sell nothing

        self.net_worth = self.balance + self.shares_held * current_price
        reward += (self.net_worth - prev_net_worth) + penalty

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        return next_state, reward, done, {}


# 2. Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 3. LSTM-Based DQN
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers, num_actions):
        super(DQN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 4. DQN Agent
class DQNAgent:
    def __init__(self, state_shape, num_actions, hidden_size=64, lstm_layers=1,
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        input_size = state_shape[1]
        self.model = DQN(input_size, hidden_size, lstm_layers, num_actions).to(self.device)
        self.target_model = DQN(input_size, hidden_size, lstm_layers, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.max(1)[1].item()

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        q_values = self.model(state_batch).gather(1, action_batch)
        next_q_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filepath="models/dqn_stock_trader.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath="models/dqn_stock_trader.pth"):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()

# 5. Training
def train(agent, env, num_episodes=500, batch_size=16, target_update=10):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        actions_taken = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            actions_taken.append(action)
            agent.update(batch_size)

        if episode % target_update == 0:
            agent.update_target()

        print(f"Step {episode}: Action {action}, Reward {reward:.2f}, Net Worth {env.net_worth:.2f}")


# 6. Load Stock Data from Stooq
def load_stooq_data(ticker, period="5y"):
    symbol = ticker.lower()
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url)
    df.columns = [col.strip().capitalize() for col in df.columns]
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    df = df.sort_index()
    return df

# 7. Inference
def infer_action(model_path, ticker, window_size):
    try:
        df = load_stooq_data(ticker)
        if len(df) < window_size:
            return ticker, None
        obs = df.tail(window_size).values.astype(np.float32)  # shape (50, 5)
        state = torch.FloatTensor(obs).unsqueeze(0)  # shape (1, 50, 5)
        model = DQN(input_size=5, hidden_size=64, lstm_layers=1, num_actions=3)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # or 'cuda'
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            q_values = model(state)
        action = q_values.max(1)[1].item()
        return ticker, action
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return ticker, None


# 8. Multiprocessing Wrapper
def serial_inference(model_path, tickers, window_size):
    results = {}
    for ticker in tickers:
        try:
            result = infer_action(model_path, ticker, window_size)
            results[result[0]] = result[1]
        except Exception as e:
            results[ticker] = None
            print(f"Failed for {ticker}: {e}")
    return results

# 9. Main Execution
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    ticker = "AAPL.US"  # Stooq format
    df = load_stooq_data(ticker)
    env = StockTradingEnv(df)
    agent = DQNAgent(state_shape=env.observation_space.shape, num_actions=env.action_space.n)
    train(agent, env)
    model_path = "backend/dqn_stock_trader.pth"
    torch.save({
    'model_state_dict': agent.model.state_dict(),
    'input_size': agent.model.lstm.input_size,
    'hidden_size': agent.model.lstm.hidden_size,
    'lstm_layers': agent.model.lstm.num_layers,
    'num_actions': agent.num_actions
}, model_path)


    results = serial_inference(model_path, ["AAPL.US", "MSFT.US", "TSLA.US"], window_size=50)
    print(results)
