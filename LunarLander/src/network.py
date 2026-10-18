import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        x = self.layer_1(state)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        q_values = self.output_layer(x)
        return q_values