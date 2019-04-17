import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.hidden1 = nn.Linear(input_size, 400)
        self.hidden2 = nn.Linear(400, 300)
        self.output = nn.Linear(300, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output(x)


class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, low_bound, high_bound):
        super().__init__()

        self.hidden1 = nn.Linear(state_size, 400)
        self.hidden2 = nn.Linear(400, 300)
        self.output = nn.Linear(300, action_size)
        self.low_bound = low_bound
        self.high_bound = high_bound

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return ((torch.tanh(x)+1) / 2 * (self.high_bound - self.low_bound)) + self.low_bound
