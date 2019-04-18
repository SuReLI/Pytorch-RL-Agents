import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):

    def __init__(self, input_size, hidden_layers_size):
        super(CriticNetwork, self).__init__()

        self.hiddens = nn.ModuleList([nn.Linear(input_size, hidden_layers_size[0])])
        for i in range(1,len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, x):

        for layer in self.hiddens:
            x = F.relu(layer(x))

        return self.output(x)


class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, low_bound, high_bound, hidden_layers_size):
        super(ActorNetwork, self).__init__()

        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1,len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], action_size)

        self.low_bound = low_bound
        self.high_bound = high_bound

    def forward(self, x):

        for layer in self.hiddens:
            x = F.relu(layer(x))

        x = self.output(x)
        x = (torch.sigmoid(x) * (self.high_bound - self.low_bound)) + self.low_bound
        return x
