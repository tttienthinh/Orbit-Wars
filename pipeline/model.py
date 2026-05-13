import torch.nn as nn


class OrbitMLP(nn.Module):
    """
    Multi-task MLP for imitation learning.
    Heads: from (45 classes, 44=stop), to (44 classes), ships (regression).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),        nn.ReLU(),
        )
        self.head_from  = nn.Linear(256, 45)
        self.head_to    = nn.Linear(256, 44)
        self.head_ships = nn.Linear(256, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.head_from(h), self.head_to(h), self.head_ships(h).squeeze(-1)
