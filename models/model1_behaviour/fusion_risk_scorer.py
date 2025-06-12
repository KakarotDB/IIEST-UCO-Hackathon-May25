import torch.nn as nn
import torch

class BehaviourRiskScoring(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(32 + 16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, static_vec, lstm_vec):
        fused = torch.cat((static_vec, lstm_vec), dim=1)
        return self.scorer(fused)