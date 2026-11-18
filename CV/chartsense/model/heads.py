import torch.nn as nn


class PatternHead(nn.Module):
    def __init__(self, in_features, n_patterns):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_patterns)
        )
    
    def forward(self, x):
        return self.head(x)


class DirectionHead(nn.Module):
    def __init__(self, in_features, n_directions):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_directions)
        )
    
    def forward(self, x):
        return self.head(x)