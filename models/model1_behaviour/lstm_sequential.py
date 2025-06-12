import torch.nn as nn

class SequentialBehaviourLSTM(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq):
        _, (h_n, _) = self.lstm(x_seq)
        return self.fc(h_n[-1])
