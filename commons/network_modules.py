import torch
import torch.nn as nn
from torch.distributions import Normal


class QNetwork(nn.Module):

    def __init__(self, input_size, action_size, hidden_layers_size):
        super().__init__()

        self.hiddens = nn.ModuleList([nn.Linear(input_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], action_size)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return self.output(x)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size):
        super().__init__()

        self.hiddens = nn.ModuleList([nn.Linear(state_size+action_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return self.output(x)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size):
        super().__init__()
        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], action_size)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return torch.tanh(self.output(x))

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_layers_size):
        super().__init__()

        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return self.output(x)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class SoftActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size, device,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.device = device
        self.action_size = action_size

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))

        self.mean_output = nn.Linear(hidden_layers_size[-1], action_size)
        self.mean_output.weight.data.uniform_(-init_w, init_w)
        self.mean_output.bias.data.uniform_(-init_w, init_w)

        self.log_std_output = nn.Linear(hidden_layers_size[-1], action_size)
        self.log_std_output.weight.data.uniform_(-init_w, init_w)
        self.log_std_output.bias.data.uniform_(-init_w, init_w)

        self.normal = Normal(0, 1)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))

        mean = self.mean_output(x)
        log_std = torch.tanh(self.log_std_output(x))
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * (log_std+1) / 2

        return mean, log_std

    def evaluate(self, state):
        mean, log_std = self(state)
        std = log_std.exp()

        z = self.normal.sample((self.action_size, )).to(self.device)
        action = torch.tanh(mean + std*z)

        log_prob = Normal(mean, std).log_prob(mean + std*z).sum(dim=1).unsqueeze(1)
        # Cf Annexe C.
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1).unsqueeze(1)

        return action, log_prob

    def get_mu_sig(self, state):
        with torch.no_grad():
            mean, log_std = self(state)
        std = log_std.exp()
        return mean.cpu().numpy(), std.cpu().numpy()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_std = self(state)
            std = log_std.exp()

            z = self.normal.sample((self.action_size, )).to(self.device)
            action = torch.tanh(mean + std*z)

        action = action.cpu().numpy()
        return action[0]

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))
