import torch.nn as nn

class StaticBehaviourEncoder(nn.Module):
    def __init__(self, input_dim=120, output_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)