import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):

    def __init__(self, input_size):
        super(CriticNetwork, self).__init__()

        self.hidden1 = nn.Linear(input_size, 8)
        self.hidden2 = nn.Linear(8, 8)
        self.hidden3 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return self.output(x)


class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, low_bound, high_bound):
        super(ActorNetwork, self).__init__()

        self.hidden1 = nn.Linear(state_size, 8)
        self.hidden2 = nn.Linear(8, 8)
        self.hidden3 = nn.Linear(8, 8)
        self.output = nn.Linear(8, action_size)
        self.low_bound = low_bound
        self.high_bound = high_bound

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        x = (torch.sigmoid(x) * (self.high_bound - self.low_bound)) + self.low_bound
        return x
