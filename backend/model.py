import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers, num_actions):
        super(DQN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
