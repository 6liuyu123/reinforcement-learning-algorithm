import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(config.state_size, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, config.action_size)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value